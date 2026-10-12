# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
import os
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torchvision
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from tqdm import tqdm
from transformers.trainer import *
from trl.trainer.ppo_trainer import *

from gr00t.rl.agents.modules.data_utils import RolloutStorage
from gr00t.rl.trl.callbacks.hv_callback_handler import HVCallbackHandler
from gr00t.rl.trl.callbacks.model_save_callback import ModelSaveCallback
from gr00t.rl.trl.modules.temporal_action_noise import (
    marginal_temporal_gaussian_log_prob,
    marginal_temporal_gaussian_parameters,
    paired_temporal_gaussian_parameters,
)
from gr00t.rl.trl.modules.exploration_budget import normalized_action_variance
from gr00t.rl.trl.utils.common import wandb_run_exists
from gr00t.rl.trl.utils.metrics_log import (
    append_training_metrics_jsonl,
    take_remaining_episode_ids,
)
from gr00t.rl.trl.utils.rl import compute_episode_attnmask
from gr00t.rl.trl.utils.scheduler import update_scheduled_params
from gr00t.rl.utils.average_meters import TensorAverageMeterDict

console = Console()
import time

import numpy as np


class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model, ref_model=None) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.ref_model = ref_model

    def set_mode(self, mode):
        if hasattr(self.policy, "mode"):
            self.policy.mode = mode

    def transform_train(self):
        if hasattr(self.policy, "transform_train"):
            self.policy.transform_train()

    def transform_eval(self):
        if hasattr(self.policy, "transform_eval"):
            self.policy.transform_eval()

    def forward(self, modes, input_kwargs):
        results = {}
        for mode in modes:
            results[mode] = self.forward_component(mode, **input_kwargs[mode])
        return results

    def forward_component(self, mode, actions=None, **kwargs):
        if mode == "policy":
            self.policy.act(**kwargs, marginalize_temporal=True)
            log_probs = self.policy.get_actions_log_prob(actions=actions)
            results = {
                "logprobs": log_probs,
                "action_mean": self.policy.action_mean,
                "action_distribution_mean": self.policy.action_distribution_mean,
                "action_std": self.policy.action_std,
                "marginal_action_std": self.policy.marginal_action_std,
                "global_action_std": self.policy.global_action_std,
                "action_noise_attenuation": self.policy.action_noise_attenuation,
                "action_temporal_fraction": self.policy.temporal_action_noise_fraction,
                "entropy": self.policy.entropy,
            }
        elif mode == "policy_distill":
            results = self.policy.act(**kwargs)
        elif mode == "policy_distill_ppo":
            policy_state_dict = self.policy.act(**kwargs, marginalize_temporal=True)
            log_probs = self.policy.get_actions_log_prob(actions=actions)
            results = {
                "actions": policy_state_dict["actions"],
                "logprobs": log_probs,
                "action_mean": policy_state_dict["action_mean"],
                "action_distribution_mean": self.policy.action_distribution_mean,
                "action_std": policy_state_dict["action_sigma"],
                "marginal_action_std": self.policy.marginal_action_std,
                "global_action_std": self.policy.global_action_std,
                "action_noise_attenuation": self.policy.action_noise_attenuation,
                "action_temporal_fraction": self.policy.temporal_action_noise_fraction,
                "entropy": self.policy.entropy,
            }
            if "normalized_actions" in policy_state_dict:
                results["normalized_actions"] = policy_state_dict["normalized_actions"]
        elif mode == "policy_w_and_wo_imgaug":
            # The first forward is without image augmentation
            self.policy.transform_eval()
            policy_state_dict = self.policy.act(**kwargs, marginalize_temporal=True)
            # Use the distribution without image augmentation to get the log_probs
            log_probs = self.policy.get_actions_log_prob(actions=actions)
            results = {
                "actions": policy_state_dict["actions"],
                "logprobs": log_probs,
                "action_mean": policy_state_dict["action_mean"],
                "action_distribution_mean": self.policy.action_distribution_mean,
                "action_std": policy_state_dict["action_sigma"],
                "marginal_action_std": self.policy.marginal_action_std,
                "global_action_std": self.policy.global_action_std,
                "action_noise_attenuation": self.policy.action_noise_attenuation,
                "action_temporal_fraction": self.policy.temporal_action_noise_fraction,
                "entropy": self.policy.entropy,
            }

            # The second forward is with image augmentation
            self.policy.transform_train()
            # The second time doesn't need deepcopy
            policy_state_dict_w_imgaug = self.policy.act(
                **kwargs, marginalize_temporal=True
            )
            results["action_mean_w_imgaug"] = policy_state_dict_w_imgaug["action_mean"]
            results["actions_w_imgaug"] = policy_state_dict_w_imgaug["actions"]
            if "normalized_actions" in policy_state_dict_w_imgaug:
                results["normalized_actions_w_imgaug"] = policy_state_dict_w_imgaug[
                    "normalized_actions"
                ]
        elif mode == "policy_deterministic":
            results = {
                "action_mean": self.policy.deterministic_action(**kwargs),
            }
        elif mode == "vae_policy_deterministic":
            self.policy.act(**kwargs)
            prior_mu, prior_log_var = self.policy.eval_prior(**kwargs)
            results = {
                "action_mean": self.policy.action_mean,
                "vae_mu": self.policy.z_mu,
                "vae_log_var": self.policy.z_log_sigma,
                "prior_mu": prior_mu,
                "prior_log_var": prior_log_var,
            }
        elif mode == "value":
            results = self.value_model.evaluate(**kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return results


class PrinterHVCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            width = 80
            pad = 35
            print_str = f" \033[1m Learning iteration {state.global_step}  \033[0m "

            log_string = (
                f"""{print_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {logs['fps']:.0f} steps/s (Collection: {logs['collection_time']:.3f}s, Learning {logs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {logs['Policy/mean_noise_std']:.2f}\n"""
            )

            for k, v in logs.items():
                if k.startswith("objective/"):
                    # Keep the original logic
                    if k.startswith("objective/kin_"):
                        log_string += f"""{f'{k}:':>{pad}} {v:.5f}\n"""
                    else:
                        new_key = k.replace("objective/", "")
                        log_string += f"""{f'Mean {new_key}:':>{pad}} {v:.5f}\n"""
                elif k.startswith("loss/"):
                    # Add support for loss/ prefix metrics
                    new_key = k.replace("loss/", "")
                    log_string += f"""{f'{new_key}:':>{pad}} {v:.6f}\n"""
                elif k.startswith("train/stage_count/"):
                    log_string += f"""{f'{k}:':>{pad}} {v:.0f}\n"""
                elif k.startswith("train/stage_weight/"):
                    log_string += f"""{f'{k}:':>{pad}} {v:.6f}\n"""
                elif k.startswith("train/stage_reward_mean/"):
                    log_string += f"""{f'{k}:':>{pad}} {v:.6f}\n"""
                elif k.startswith("train/stage_action_advantage_signal/"):
                    log_string += f"""{f'{k}:':>{pad}} {v:.8f}\n"""
                elif k.startswith("train/push_open_causal/"):
                    log_string += f"""{f'{k}:':>{pad}} {v:.8f}\n"""

            env_log_string = ""
            ep_string = ""
            for k, v in logs.items():
                if k.startswith("Env/"):
                    entry = f"{f'{k}:':>{pad}} {v:.4f}"
                    env_log_string += f"{entry}\n"
                if k.startswith("Episode/"):
                    new_key = k.replace("Episode/", "")
                    ep_string += f"""{f'Mean episode {new_key}:':>{pad}} {v:.4f}\n"""

            log_string += env_log_string
            log_string += ep_string
            log_string += (
                f"""{'-' * width}\n"""
                f"""{'Total episodes:':>{pad}} {logs['episode']}\n"""
                f"""{'Total timesteps:':>{pad}} {logs['tot_timesteps']}\n"""
                f"""{'Iteration time:':>{pad}} {logs['collection_time'] + logs['learn_time']:.2f}s\n"""
                f"""{'Total time:':>{pad}} {logs['tot_time']:.2f}s\n"""
                f"""{'ETA:':>{pad}} {logs['tot_time'] / logs['batch_idx'] * (logs['num_total_batches'] - logs['batch_idx']):.1f}s\n"""
            )

            log_string += f"Logging Directory: {logs['experiment_save_dir']}"
            with Live(
                Panel(log_string, title="Training Log"), refresh_per_second=4, console=console
            ):
                # Your training loop or other operations
                pass


def process_ep_infos(ep_infos, device):
    infos = {}
    for key in ep_infos[0]:
        infotensor = torch.tensor([], device=device)
        for ep_info in ep_infos:
            # handle scalar and zero dimensional tensor infos
            if not isinstance(ep_info[key], torch.Tensor):
                ep_info[key] = torch.Tensor([ep_info[key]])
            if len(ep_info[key].shape) == 0:
                ep_info[key] = ep_info[key].unsqueeze(0)
            infotensor = torch.cat((infotensor, ep_info[key].to(device)))
        value = torch.mean(infotensor)
        infos[key] = value
    return infos


class TRLPPOTrainer(PPOTrainer):
    """
    Custom PPO Trainer that adapts TRL's PPOTrainer to work with Humanoid environments.
    """

    _tag_names = ["trl", "humanoid_ppo"]

    def __init__(
        self,
        args,
        config,
        env,
        model,
        ref_model=None,
        reward_model=None,
        processing_class=None,
        value_model=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        log_dir=None,
        # less commonly used
        optimizers=(None, None),
        callbacks=None,
        peft_config=None,
        use_ref_model=False,
        checkpoint=None,
        local_seed=None,
        schedule_dict=None,
        accelerator=None,
    ) -> None:
        self.accelerator = accelerator
        self._init_trl(
            args,
            config,
            env,
            processing_class,
            model,
            ref_model,
            reward_model,
            train_dataset,
            value_model,
            data_collator,
            eval_dataset,
            optimizers,
            callbacks,
            peft_config,
            use_ref_model,
            local_seed,
            log_dir,
            schedule_dict=schedule_dict,
        )
        self._init_config()
        self._setup_storage()

        # Initialize trajectory counter for recurrent policy training

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def _init_trl(
        self,
        args,
        config,
        env,
        processing_class,
        model,
        ref_model,
        reward_model,
        train_dataset,
        value_model,
        data_collator,
        eval_dataset,
        optimizers,
        callbacks,
        peft_config,
        use_ref_model,
        local_seed,
        log_dir,
        schedule_dict=None,
    ):
        self.args = args
        self.config = config
        self.env = env
        self.processing_class = processing_class
        self.policy_model = model
        self.learn_normalized_actions = model.has_normalized_actions
        self.episode_env_tensors = TensorAverageMeterDict()
        self.ep_infos = []
        self.eval_callbacks = []
        self.log_dir = log_dir
        self.schedule_dict = schedule_dict
        self.scheduled_params_dict = {}

        # peft support
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()

            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if use_ref_model:
            if ref_model:
                self.ref_model = ref_model
            elif self.is_peft_model:
                self.ref_model = None
            else:
                self.ref_model = create_reference_model(self.policy_model)
        else:
            self.ref_model = None

        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = (
            len(train_dataset) if train_dataset is not None else self.env.config.num_envs
        )
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset

        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########

        accelerator = self.accelerator

        self.device = accelerator.device
        args.global_rank = accelerator.process_index
        args.world_size = accelerator.num_processes
        args.is_main_process = accelerator.is_main_process
        args.local_batch_size = self.env.config.num_envs
        args.batch_size = int(args.local_batch_size * args.world_size)
        try:
            args.mini_batch_size = exact_div(
                args.batch_size,
                args.num_mini_batches,
                "`batch_size` must be a multiple of `num_mini_batches`",
            )
            args.local_mini_batch_size = exact_div(
                args.local_batch_size,
                args.num_mini_batches,
                "`local_batch_size` must be a multiple of `num_mini_batches`",
            )
        except Exception as e:
            print(f"Error: {e}")
            args.mini_batch_size = 1
            args.local_mini_batch_size = 1

        if args.per_device_train_batch_size is None:
            args.per_device_train_batch_size = (
                args.local_mini_batch_size
            )  # same as mini-batch size, which implies no micro-batching (num_micro_batches = 1)
        args.num_micro_batches = args.local_mini_batch_size // args.per_device_train_batch_size
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        if args.total_episodes is None:
            assert args.num_total_batches is not None
            args.total_episodes = args.num_total_batches * args.batch_size
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = local_seed
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(
                1, args.num_total_batches // args.num_sample_generations
            )
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        if self.config.get("disable_dropout", True):
            for module in [self.policy_model, self.ref_model, self.value_model, self.reward_model]:
                if module is not None:
                    disable_dropout_in_model(module)
        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        # self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to
        )
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = HVCallbackHandler(
            self.callbacks,
            self.model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
            self.env,
            self.accelerator,
        )
        self.add_callback(
            PrinterHVCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK
        )
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = (
            getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        )
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        if self.train_dataset is not None:
            self.dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.local_dataloader_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                drop_last=False,  # needed; otherwise the last batch will be of ragged shape
            )
        else:
            self.dataloader = None
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)

        self.model, self.optimizer, self.dataloader = accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
        self.unwrapped_model = unwrap_model(self.model)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                drop_last=False,
            )  # no need to shuffle eval dataset
            self.eval_dataloader = accelerator.prepare(self.eval_dataloader)
        else:
            self.eval_dataloader = None

        if self.is_deepspeed_enabled:
            if self.reward_model is not None:
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )

            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is None:
                # if not self.is_peft_model:
                #     raise ValueError("No reference model and model is not a Peft model.")
                pass
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)
        self.use_apex = False

        self.train_with_evaluating_env = self.config.get("train_with_evaluating_env", False)

        # Camera resolution
        if "vision_obs" in self.env.config.obs.obs_dict:
            if (
                self.env.config.obs.obs_dict.vision_obs[0] in ["depth_image", "height_map"]
                and not self.env.config.simulator.config.cameras.stack_depth_image
            ):
                num_channels = 1
            else:
                num_channels = 3

            if self.env.config.obs.obs_dict.vision_obs[0] == "height_map":
                heightmap_resolution = self.env.config.simulator.config.heightmap.resolution
                self.camera_resolution = [heightmap_resolution, heightmap_resolution] + [
                    num_channels
                ]
            else:
                self.camera_resolution = (
                    self.env.config.simulator.config.cameras.camera_resolutions + [num_channels]
                )
        else:
            self.camera_resolution = None

    def _init_config(self):
        # Env related Config
        self.num_envs: int = self.env.config.num_envs
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_act = self.policy_model.num_actions

        self.num_steps_per_env = self.config.num_steps_per_env
        self.use_padding_mask = self.config.get("use_padding_mask", False)
        self.ppo_shuffle_every_epoch = self.config.get("ppo_shuffle_every_epoch", True)
        self.empty_cache_every_n_ppo_epoch = self.config.get("empty_cache_every_n_ppo_epoch", -1)

        self.entropy_coef = self.config.entropy_coef
        self.state_dependent_action_noise = bool(
            getattr(self.policy_model, "state_dependent_action_noise", False)
        )
        if self.state_dependent_action_noise:
            self.exploration_variance_budget_fraction = float(
                self.config.exploration_variance_budget_fraction
            )
            self.exploration_variance_budget_dual_lr = float(
                self.config.exploration_variance_budget_dual_lr
            )
            if not 0.0 <= self.exploration_variance_budget_fraction <= 1.0:
                raise ValueError(
                    "exploration_variance_budget_fraction must be in [0, 1]"
                )
            if self.exploration_variance_budget_dual_lr <= 0.0:
                raise ValueError(
                    "exploration_variance_budget_dual_lr must be positive"
                )
            if not getattr(self.policy_model, "clamp_noise_std", False):
                raise ValueError(
                    "A normalized exploration budget requires clamp_noise_std"
                )
        self._last_exploration_mean_variance = None
        self._last_exploration_budget_shortfall = None
        self._last_temporal_fraction_stats = None
        self.desired_kl = self.config.desired_kl
        self.use_adaptive_kl_schedule = (
            self.config.get("schedule", None) == "adaptive"
            and self.desired_kl is not None
        )
        self.initial_learning_rate = float(self.args.learning_rate)
        self.adaptive_learning_rate = self.initial_learning_rate
        self.adaptive_kl_min_learning_rate = float(
            self.config.get(
                "adaptive_kl_min_learning_rate",
                self.initial_learning_rate / 10.0,
            )
        )
        self.adaptive_kl_max_learning_rate = float(
            self.config.get(
                "adaptive_kl_max_learning_rate",
                self.initial_learning_rate,
            )
        )
        if not (
            0.0 < self.adaptive_kl_min_learning_rate
            <= self.initial_learning_rate
            <= self.adaptive_kl_max_learning_rate
        ):
            raise ValueError(
                "Adaptive KL learning-rate bounds must satisfy "
                "0 < min <= initial <= max"
            )
        self.gamma = self.args.gamma
        self.lam = self.args.lam
        self.sync_advantage_normalization = self.config.get("sync_advantage_normalization", True)
        self.balance_staged_task_samples = bool(
            self.config.get("balance_staged_task_samples", False)
        )
        self.balance_staged_task_variants = bool(
            self.config.get("balance_staged_task_variants", False)
        )
        self.staged_task_max_sample_weight = float(
            self.config.get("staged_task_max_sample_weight", 10.0)
        )
        if self.balance_staged_task_samples:
            if not hasattr(self.env, "stage_buf") or not hasattr(self.env, "num_stages"):
                raise ValueError(
                    "balance_staged_task_samples requires a staged environment"
                )
            if self.staged_task_max_sample_weight < 1.0:
                raise ValueError("staged_task_max_sample_weight must be at least 1.0")
        if self.balance_staged_task_variants and not hasattr(
            self.env, "get_task_variant"
        ):
            raise ValueError(
                "balance_staged_task_variants requires get_task_variant() on the environment"
            )

        self.compute_imgaug_bc_loss = self.config.get("compute_imgaug_bc_loss", False)
        self.imgaug_bc_loss_coef = self.config.get("imgaug_bc_loss_coef", 1.0)
        self.imgaug_bc_loss_fn = torch.nn.MSELoss()
        self.policy_action_supervision_coef = float(
            self.config.get("policy_action_supervision_coef", 0.0)
        )
        self.policy_action_supervision_huber_beta = float(
            self.config.get("policy_action_supervision_huber_beta", 1.0)
        )
        if self.policy_action_supervision_huber_beta <= 0.0:
            raise ValueError("policy_action_supervision_huber_beta must be positive")
        self.compute_policy_action_supervision = self.policy_action_supervision_coef > 0.0
        if self.compute_policy_action_supervision and not hasattr(
            self.env, "get_policy_action_supervision"
        ):
            raise ValueError(
                "policy_action_supervision_coef requires the environment to implement "
                "get_policy_action_supervision()"
            )

    def _setup_storage(self):
        self.storage = RolloutStorage(
            self.env.num_envs, self.num_steps_per_env, device=self.accelerator.device
        )

        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            if obs_key == "vision_obs":
                # Handle vision observations - check if it's history-based
                if (
                    hasattr(self.env.config.obs, "obs_auxiliary")
                    and "rgb_image_history" in self.env.config.obs.obs_auxiliary
                ):
                    # The environment's HistoryHandler provides [num_envs, history_length, H, W, C]
                    # We need to store this as [num_steps, num_envs, history_length, H, W, C] in storage
                    history_config = self.env.config.obs.obs_auxiliary.rgb_image_history
                    history_length = history_config.rgb_image

                    # Register with full 5D shape: [history_length, H, W, C]
                    vision_shape = tuple([history_length] + self.camera_resolution)
                    self.storage.register_key(obs_key, shape=vision_shape, dtype=torch.float)
                    self.vision_history_length = history_length
                    self.vision_obs_key = obs_key
                else:
                    # Regular single-frame vision observation: [H, W, C]
                    assert obs_dim == np.prod(
                        self.camera_resolution
                    ), f"{obs_dim=}, {self.camera_resolution=}"
                    self.storage.register_key(
                        obs_key, shape=tuple(self.camera_resolution), dtype=torch.float
                    )
                    self.vision_history_length = None
                    self.vision_obs_key = None
            elif obs_key == "state_history_obs":
                # The environment's HistoryHandler provides [num_envs, history_length, obs_dim]
                # We need to store this as [num_steps, num_envs, history_length, obs_dim] in storage
                history_config = self.env.config.obs.obs_auxiliary.state_history
                # get the max history length from history_config values
                history_length = max(history_config.values())
                concat_obs_dim = 0
                for obs_name in history_config.keys():
                    concat_obs_dim += self.env.config.obs.obs_dims[obs_name]

                self.storage.register_key(
                    obs_key, shape=(history_length, concat_obs_dim), dtype=torch.float
                )
            else:
                self.storage.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)

        # Set vision history flags
        if not hasattr(self, "vision_history_length"):
            self.vision_history_length = None
            self.vision_obs_key = None

        ## Register others
        self.storage.register_key("actions", shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key("rewards", shape=(1,), dtype=torch.float)
        self.storage.register_key("dones", shape=(1,), dtype=torch.bool)
        self.storage.register_key("time_outs", shape=(1,), dtype=torch.bool)
        self.storage.register_key("values", shape=(1,), dtype=torch.float)
        self.storage.register_key("returns", shape=(1,), dtype=torch.float)
        self.storage.register_key("advantages", shape=(1,), dtype=torch.float)
        self.storage.register_key("actions_log_prob", shape=(1,), dtype=torch.float)
        self.storage.register_key("action_mean", shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key("action_sigma", shape=(self.num_act,), dtype=torch.float)
        if getattr(self.policy_model, "temporal_action_noise_enabled", False):
            self.storage.register_key(
                "action_temporal_fraction", shape=(self.num_act,), dtype=torch.float
            )
        if self.balance_staged_task_samples:
            self.storage.register_key("task_stage", shape=(1,), dtype=torch.long)
            if self.balance_staged_task_variants:
                self.storage.register_key("task_variant", shape=(1,), dtype=torch.long)
        if self.compute_policy_action_supervision:
            self.storage.register_key(
                "policy_action_target", shape=(self.num_act,), dtype=torch.float
            )
            self.storage.register_key(
                "policy_action_target_mask", shape=(self.num_act,), dtype=torch.float
            )
        if self.learn_normalized_actions:
            self.storage.register_key(
                "normalized_actions", shape=(self.num_act,), dtype=torch.float
            )

        self.state.rewbuffer = deque(maxlen=100)
        self.state.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.accelerator.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.accelerator.device
        )

        self.state.cur_reward_sum = self.cur_reward_sum
        self.state.cur_episode_length = self.cur_episode_length
        self.ep_infos = []
        self.state.tot_timesteps = 0
        self.state.tot_time = 0
        self.state.eval_step = 0
        self.state.eval_render_step = 0

    def policy_step(self, policy_model, obs_dict, cur_dones=None, store_hidden_states=True):
        actor_obs_dict = deepcopy(obs_dict)

        if cur_dones is None:
            dones = (
                self.storage.query_key("dones")
                .to(self.accelerator.device)[: self.storage.step + 1]
                .squeeze(-1)
                .transpose(0, 1)
            )
            episode_attnmask = compute_episode_attnmask(dones)
        else:
            episode_attnmask = None

        # Store hidden states BEFORE rollout for recurrent policies
        actor_hidden_states = None
        if (
            store_hidden_states
            and hasattr(policy_model, "is_recurrent")
            and policy_model.is_recurrent
        ):
            actor_hidden_states = policy_model.get_hidden_states()

        policy_state_dict = policy_model.rollout(
            obs_dict=actor_obs_dict, episode_attnmask=episode_attnmask, cur_dones=cur_dones
        )
        actions = policy_state_dict["actions"]
        actions_log_prob = policy_model.get_actions_log_prob(actions).unsqueeze(1)
        policy_state_dict["actions_log_prob"] = actions_log_prob

        self._last_policy_hidden_states = (
            (actor_hidden_states, None)
            if store_hidden_states and actor_hidden_states is not None
            else (None, None)
        )

        return policy_state_dict

    def _chunked_value_evaluate(self, value_model, obs_dict, episode_attnmask, chunk_size=1024):
        batch_size = list(obs_dict.values())[0].shape[0]
        if batch_size <= chunk_size:
            return value_model.evaluate(obs_dict=obs_dict, episode_attnmask=episode_attnmask)

        obs_chunks = {}
        for key, value in obs_dict.items():
            obs_chunks[key] = torch.split(value, chunk_size, dim=0)
        if episode_attnmask is not None:
            attnmask_chunks = torch.split(episode_attnmask, chunk_size, dim=0)
        else:
            attnmask_chunks = [None] * len(obs_chunks[list(obs_chunks.keys())[0]])

        value_chunks = []
        for i in range(len(attnmask_chunks)):
            chunk_obs_dict = {key: obs_chunks[key][i] for key in obs_chunks}
            chunk_values = value_model.evaluate(
                obs_dict=chunk_obs_dict, episode_attnmask=attnmask_chunks[i]
            )
            value_chunks.append(chunk_values)
        return torch.cat(value_chunks, dim=0)

    def _rollout_step(self, model, obs_dict):
        self._train_rollout_mode()
        device = self.accelerator.device
        policy_model = model.policy
        value_model = model.value_model
        policy_model.init_rollout()
        self.storage.clear()
        dones = torch.zeros(self.env.num_envs, device=device)
        with torch.no_grad():
            values = []
            for i in range(self.num_steps_per_env):
                # Compute the actions and values
                policy_state_dict = self.policy_step(policy_model, obs_dict, cur_dones=dones)
                policy_diagnostics = {}
                if hasattr(self.env, "get_policy_diagnostics"):
                    policy_diagnostics = self.env.get_policy_diagnostics(policy_state_dict)
                # For recurrent critics, compute values step-by-step and capture critic hidden states
                if self.value_model is not None:
                    value_is_recurrent = (
                        hasattr(self.value_model, "is_recurrent") and self.value_model.is_recurrent
                    )
                    if value_is_recurrent:
                        critic_hidden_states = (
                            self.value_model.get_hidden_states()
                            if hasattr(self.value_model, "get_hidden_states")
                            else None
                        )
                        critic_obs_dict = {k: v for k, v in obs_dict.items() if k != "actor_obs"}
                        step_values = self.value_model.evaluate(obs_dict=critic_obs_dict)
                        values.append(step_values)
                        # Store both actor and critic hidden states
                        combined_hidden_states = (
                            getattr(self, "_last_policy_hidden_states", (None, None))[0],
                            critic_hidden_states,
                        )

                # Append states to storage
                for key, value in obs_dict.items():
                    self.storage.update_key(key, value)

                for key, value in policy_state_dict.items():
                    self.storage.update_key(key, value)
                if self.balance_staged_task_samples:
                    self.storage.update_key("task_stage", self.env.stage_buf[:, None])
                    if self.balance_staged_task_variants:
                        self.storage.update_key(
                            "task_variant", self.env.get_task_variant()[:, None]
                        )
                if self.compute_policy_action_supervision:
                    action_target, action_target_mask = (
                        self.env.get_policy_action_supervision()
                    )
                    self.storage.update_key("policy_action_target", action_target)
                    self.storage.update_key(
                        "policy_action_target_mask", action_target_mask
                    )

                # Store hidden states separately for recurrent policies
                hidden_states = (
                    combined_hidden_states
                    if self.value_model is not None
                    and hasattr(self.value_model, "is_recurrent")
                    and self.value_model.is_recurrent
                    else getattr(self, "_last_policy_hidden_states", (None, None))
                )
                self.storage._save_hidden_states(hidden_states)  # type: ignore[attr-defined]

                # Step the environment
                actor_state = {"actions": policy_state_dict["actions"]}
                if hasattr(self.env, "set_unsafe_force_policy_step"):
                    self.env.set_unsafe_force_policy_step(
                        policy_state_dict["action_mean"],
                        policy_state_dict["actions"],
                    )
                # Pass gt_actions to environment if available (for distillation mode)
                if "gt_actions" in policy_state_dict:
                    actor_state["gt_actions"] = policy_state_dict["gt_actions"]
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                infos["to_log"].update(policy_diagnostics)
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(device)
                rewards, dones = rewards.to(device), dones.to(device)
                rewards_stored = rewards.clone().unsqueeze(1)
                assert len(rewards_stored.shape) == 2

                self.ep_infos.append(infos["episode"])
                self.storage.update_key("rewards", rewards_stored)
                self.storage.update_key("dones", dones.unsqueeze(1))
                self.storage.update_key("time_outs", infos["time_outs"].unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                self.cur_reward_sum += rewards
                self.cur_episode_length += 1

                new_ids = (dones > 0).nonzero(as_tuple=False)
                self.state.rewbuffer.extend(
                    self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                )
                self.state.lenbuffer.extend(
                    self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                )

                self.cur_reward_sum[new_ids] = 0
                self.cur_episode_length[new_ids] = 0

            if values:
                values = torch.stack(values, dim=0)

            if self.value_model is not None:
                # Check if critic is recurrent
                value_is_recurrent = (
                    hasattr(self.value_model, "is_recurrent") and self.value_model.is_recurrent
                )

                if value_is_recurrent:
                    # For recurrent critics: values were already computed step-by-step during rollout
                    # We only need to compute the final bootstrapping value
                    critic_obs_dict = {k: v for k, v in obs_dict.items() if k != "actor_obs"}
                    last_values = self.value_model.evaluate(
                        obs_dict=critic_obs_dict, update_running_stats=False
                    )  # Shape: [num_envs, 1]
                else:
                    # For non-recurrent critics: compute all values at once
                    dones = self.storage.query_key("dones").to(device).squeeze(-1).transpose(0, 1)
                    dones = torch.cat([dones, torch.zeros_like(dones[:, :1])], dim=1)
                    episode_attnmask = compute_episode_attnmask(dones)
                    all_obs_dict = {}
                    for key in obs_dict.keys():
                        if key not in ["actor_obs"]:  # actor_obs not required by value model
                            obs_value = self.storage.query_key(key).to(device)
                            obs_value = torch.cat([obs_value, obs_dict[key].unsqueeze(0)], dim=0)
                            all_obs_dict[key] = obs_value.transpose(0, 1)
                    all_values = self._chunked_value_evaluate(
                        value_model, all_obs_dict, episode_attnmask
                    ).transpose(0, 1)
                    values, last_values = all_values[:-1], all_values[-1]

                rewards = self.storage.query_key("rewards")
                new_rewards = (
                    rewards.to(device)
                    + self.gamma * self.storage.query_key("time_outs").to(device) * values
                )
                self.storage.batch_update_data("rewards", new_rewards)

                returns, advantages = self._compute_returns(
                    values=values,
                    last_values=last_values,
                    policy_state_dict={
                        "dones": self.storage.query_key("dones"),
                        "rewards": self.storage.query_key("rewards"),
                        "task_stages": (
                            self.storage.query_key("task_stage")
                            if self.balance_staged_task_samples
                            else None
                        ),
                        "task_variants": (
                            self.storage.query_key("task_variant")
                            if self.balance_staged_task_variants
                            else None
                        ),
                    },
                )
                self.storage.batch_update_data("values", values)
                self.storage.batch_update_data("returns", returns)
                self.storage.batch_update_data("advantages", advantages)

        policy_model.clear_rollout()

        return obs_dict

    def _process_env_step(self, rewards, dones, infos):
        self.policy_model.reset(dones)
        if self.value_model is not None:
            self.value_model.reset(dones)
        self.episode_env_tensors.add(infos["to_log"])

    def _register_stats_buffer(self):
        args = self.args
        device = self.accelerator.device

        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.num_micro_batches)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        weighted_ppo_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        if self.compute_imgaug_bc_loss:
            imgaug_bc_loss_stats = torch.zeros(stats_shape, device=device)
            weighted_imgaug_bc_loss_stats = torch.zeros(stats_shape, device=device)
        if self.compute_policy_action_supervision:
            policy_action_supervision_stats = torch.zeros(stats_shape, device=device)

        self.approxkl_stats = approxkl_stats
        self.pg_clipfrac_stats = pg_clipfrac_stats
        self.pg_loss_stats = pg_loss_stats
        self.vf_loss_stats = vf_loss_stats
        self.entropy_stats = entropy_stats
        self.weighted_ppo_loss_stats = weighted_ppo_loss_stats
        self.vf_clipfrac_stats = vf_clipfrac_stats
        self.ratio_stats = ratio_stats
        if self.compute_imgaug_bc_loss:
            self.imgaug_bc_loss_stats = imgaug_bc_loss_stats
            self.weighted_imgaug_bc_loss_stats = weighted_imgaug_bc_loss_stats
        if self.compute_policy_action_supervision:
            self.policy_action_supervision_stats = policy_action_supervision_stats

    def _get_rollout_data(self, obs_keys):
        device = self.accelerator.device

        # Handle all observations - vision history needs special handling for tensor dimensions
        all_obs_dict = {}
        for key in obs_keys:
            obs_data = self.storage.query_key(key).transpose(0, 1).to(device)
            all_obs_dict[key] = obs_data

        actions = self.storage.actions.transpose(0, 1).to(device)
        logprobs = self.storage.actions_log_prob.transpose(0, 1).squeeze(-1).to(device)
        values = self.storage.values.transpose(0, 1).squeeze(-1).to(device)
        rewards = self.storage.rewards.transpose(0, 1).squeeze(-1).to(device)
        dones = self.storage.dones.transpose(0, 1).squeeze(-1).to(device)
        temporal_fractions = None
        if getattr(self.policy_model, "temporal_action_noise_enabled", False):
            old_mu_batch = self.storage.action_mean.transpose(0, 1).to(device)
            old_sigma_batch = self.storage.action_sigma.transpose(0, 1).to(device)
            temporal_fractions = self.storage.action_temporal_fraction.transpose(
                0, 1
            ).to(device)
            logprobs = marginal_temporal_gaussian_log_prob(
                actions,
                old_mu_batch,
                old_sigma_batch,
                dones,
                self.policy_model.temporal_action_noise_rho,
                temporal_fractions,
            )
            flat_fractions = temporal_fractions.reshape(-1, self.num_act)
            local_count = torch.full(
                (self.num_act,),
                flat_fractions.shape[0],
                dtype=flat_fractions.dtype,
                device=device,
            )
            local_stats = torch.stack(
                [
                    flat_fractions.sum(dim=0),
                    flat_fractions.square().sum(dim=0),
                    flat_fractions.amin(dim=0),
                    flat_fractions.amax(dim=0),
                    local_count,
                ]
            )
            gathered_stats = self.accelerator.gather(local_stats).reshape(
                -1, 5, self.num_act
            )
            total_count = gathered_stats[:, 4].sum(dim=0).clamp_min(1.0)
            fraction_mean = gathered_stats[:, 0].sum(dim=0) / total_count
            fraction_second_moment = (
                gathered_stats[:, 1].sum(dim=0) / total_count
            )
            self._last_temporal_fraction_stats = {
                "mean": fraction_mean,
                "std": torch.sqrt(
                    (fraction_second_moment - fraction_mean.square()).clamp_min(0.0)
                ),
                "min": gathered_stats[:, 2].amin(dim=0),
                "max": gathered_stats[:, 3].amax(dim=0),
            }
        else:
            old_mu_batch = self.storage.action_mean.transpose(0, 1).to(device)
            old_sigma_batch = self.storage.action_sigma.transpose(0, 1).to(device)
        returns = self.storage.returns.transpose(0, 1).squeeze(-1).to(device)
        advantages = self.storage.advantages.transpose(0, 1).squeeze(-1).to(device)
        if self.state_dependent_action_noise:
            local_mean_variance = normalized_action_variance(
                old_sigma_batch, self.policy_model.max_noise_std
            )
            global_mean_variance = self.accelerator.reduce(
                local_mean_variance.detach(), reduction="mean"
            )
            target = torch.full_like(
                global_mean_variance,
                self.exploration_variance_budget_fraction,
            )
            signed_violation = target - global_mean_variance
            with torch.no_grad():
                self.policy_model.exploration_budget_dual_weight.add_(
                    self.exploration_variance_budget_dual_lr * signed_violation
                ).clamp_(min=0.0)
            self._last_exploration_mean_variance = global_mean_variance
            self._last_exploration_budget_shortfall = torch.relu(signed_violation)
        task_stage_weights = None
        task_stage_supervision_weights = None
        if self.balance_staged_task_samples:
            task_stages = (
                self.storage.task_stage.transpose(0, 1).squeeze(-1).to(device).long()
            )
            local_counts = torch.bincount(
                task_stages.reshape(-1), minlength=int(self.env.num_stages)
            ).to(torch.float32)
            global_counts = self.accelerator.reduce(local_counts, reduction="sum")
            present = global_counts > 0
            total_count = global_counts.sum().clamp(min=1.0)
            present_count = present.sum().clamp(min=1)
            stage_weights = torch.zeros_like(global_counts)
            inverse_frequency = total_count / (
                present_count * global_counts[present]
            )
            # Find a common scale for inverse-frequency weights while keeping
            # both the configured hard cap and a global sample-weight mean of 1.
            lower = torch.zeros((), device=device)
            upper = torch.full((), self.staged_task_max_sample_weight, device=device)
            for _ in range(32):
                scale = 0.5 * (lower + upper)
                candidate = (inverse_frequency * scale).clamp(
                    max=self.staged_task_max_sample_weight
                )
                candidate_mean = (
                    candidate * global_counts[present]
                ).sum() / total_count
                lower = torch.where(candidate_mean < 1.0, scale, lower)
                upper = torch.where(candidate_mean < 1.0, upper, scale)
            stage_weights[present] = (inverse_frequency * upper).clamp(
                max=self.staged_task_max_sample_weight
            )
            self._last_task_stage_counts = global_counts.detach()
            self._last_task_stage_weights = stage_weights.detach()
            task_stage_weights = stage_weights[task_stages]

            # This covariance is proportional to the mean score only for an
            # independent Gaussian. A sequence likelihood couples scores across
            # time, so reporting it there as a PPO signal would be misleading.
            temporal_noise_enabled = getattr(
                self.policy_model, "temporal_action_noise_enabled", False
            )
            local_score_sums = None
            if not temporal_noise_enabled:
                local_score_sums = torch.zeros(
                    int(self.env.num_stages), self.num_act, device=device
                )
            local_reward_sums = torch.zeros(int(self.env.num_stages), device=device)
            action_noise = actions - old_mu_batch
            for stage_idx in range(int(self.env.num_stages)):
                stage_mask = task_stages == stage_idx
                if local_score_sums is not None:
                    local_score_sums[stage_idx] = (
                        action_noise
                        * advantages.unsqueeze(-1)
                        * stage_mask.unsqueeze(-1)
                    ).sum(dim=(0, 1))
                local_reward_sums[stage_idx] = rewards[stage_mask].sum()
            global_score_sums = None
            if local_score_sums is not None:
                global_score_sums = self.accelerator.reduce(
                    local_score_sums, reduction="sum"
                )
            global_reward_sums = self.accelerator.reduce(
                local_reward_sums, reduction="sum"
            )
            safe_counts = global_counts.clamp(min=1.0)
            self._last_stage_action_advantage_signal = (
                None
                if global_score_sums is None
                else (global_score_sums / safe_counts[:, None]).detach()
            )
            self._last_stage_reward_mean = (
                global_reward_sums / safe_counts
            ).detach()

            # Diagnose PUSH_OPEN credit assignment without affecting the
            # rollout or loss.  The first panel-interaction coordinate is the
            # signed normal error from the 30 mm ready point, so its temporal
            # delta is exactly the panel-clearance delta.  A positive ee_x
            # action should therefore correlate with a negative delta here.
            actor_obs_names = sorted(
                self.env.config.obs.obs_dict.get("actor_obs", [])
            )
            panel_key = "b2z1_panel_interaction"
            push_stage = getattr(self.env, "STAGE_PUSH_OPEN", 4)
            if panel_key in actor_obs_names and actions.shape[1] > 1:
                panel_key_index = actor_obs_names.index(panel_key)
                panel_start = sum(
                    int(self.env.config.obs.obs_dims[name])
                    for name in actor_obs_names[:panel_key_index]
                )
                panel_normal_error = all_obs_dict["actor_obs"][..., panel_start]
                ee_x_noise = action_noise[..., 3]
                push_mask = task_stages == push_stage
                raw_advantages = returns - values

                # Recompute the exact reward-to-go available inside this
                # rollout with a zero value at the truncation boundary.  The
                # remainder of the trainer's raw advantage then isolates the
                # critic/baseline contribution, including the final bootstrap.
                rollout_reward_to_go = torch.zeros_like(rewards)
                running_reward_to_go = torch.zeros_like(rewards[:, 0])
                for step in reversed(range(rewards.shape[1])):
                    running_reward_to_go = rewards[:, step] + (
                        self.gamma
                        * (~dones[:, step].bool()).float()
                        * running_reward_to_go
                    )
                    rollout_reward_to_go[:, step] = running_reward_to_go

                def masked_covariance(x, y, mask):
                    mask_float = mask.float()
                    local_stats = torch.stack(
                        [
                            (x * mask_float).sum(),
                            (y * mask_float).sum(),
                            (x * y * mask_float).sum(),
                            mask_float.sum(),
                        ]
                    )
                    global_stats = self.accelerator.reduce(
                        local_stats, reduction="sum"
                    )
                    count = global_stats[3].clamp(min=1.0)
                    return (
                        global_stats[2] / count
                        - (global_stats[0] / count) * (global_stats[1] / count)
                    )

                self._last_push_causal_metrics = {
                    "immediate_reward_signal": masked_covariance(
                        ee_x_noise, rewards, push_mask
                    ).detach(),
                    "critic_bootstrapped_advantage_signal": masked_covariance(
                        ee_x_noise, raw_advantages, push_mask
                    ).detach(),
                    "rollout_reward_to_go_signal": masked_covariance(
                        ee_x_noise, rollout_reward_to_go, push_mask
                    ).detach(),
                    "critic_remainder_signal": masked_covariance(
                        ee_x_noise,
                        raw_advantages - rollout_reward_to_go,
                        push_mask,
                    ).detach(),
                    "current_value_baseline_signal": masked_covariance(
                        ee_x_noise, -values, push_mask
                    ).detach(),
                    "final_value_bootstrap_signal": masked_covariance(
                        ee_x_noise,
                        raw_advantages - rollout_reward_to_go + values,
                        push_mask,
                    ).detach(),
                }
                for lag in (1, 2, 4, 8, 16, 32):
                    if actions.shape[1] <= lag:
                        continue
                    valid = push_mask[:, :-lag] & push_mask[:, lag:]
                    # Exclude intervals crossing an episode boundary.
                    for done_offset in range(lag):
                        valid &= ~dones[:, done_offset : done_offset + valid.shape[1]].bool()
                    clearance_delta = (
                        panel_normal_error[:, lag:] - panel_normal_error[:, :-lag]
                    )
                    discounted_rewards = torch.zeros_like(ee_x_noise[:, :-lag])
                    for reward_offset in range(lag):
                        discounted_rewards += (
                            self.gamma**reward_offset
                            * rewards[
                                :,
                                reward_offset : reward_offset
                                + discounted_rewards.shape[1],
                            ]
                        )
                    bootstrapped_delta = (
                        self.gamma**lag * values[:, lag:] - values[:, :-lag]
                    )
                    self._last_push_causal_metrics[
                        f"ee_x_to_clearance_delta_lag_{lag}"
                    ] = masked_covariance(
                        ee_x_noise[:, :-lag], clearance_delta, valid
                    ).detach()
                    self._last_push_causal_metrics[
                        f"ee_x_to_reward_return_lag_{lag}"
                    ] = masked_covariance(
                        ee_x_noise[:, :-lag], discounted_rewards, valid
                    ).detach()
                    self._last_push_causal_metrics[
                        f"ee_x_to_value_bootstrap_lag_{lag}"
                    ] = masked_covariance(
                        ee_x_noise[:, :-lag], bootstrapped_delta, valid
                    ).detach()
                    self._last_push_causal_metrics[
                        f"ee_x_to_bootstrapped_return_lag_{lag}"
                    ] = masked_covariance(
                        ee_x_noise[:, :-lag],
                        discounted_rewards + bootstrapped_delta,
                        valid,
                    ).detach()
                    self._last_push_causal_metrics[f"sample_count_lag_{lag}"] = (
                        self.accelerator.reduce(valid.float().sum(), reduction="sum")
                        .detach()
                    )
            else:
                self._last_push_causal_metrics = {}

        policy_action_target = None
        policy_action_target_mask = None
        if self.compute_policy_action_supervision:
            policy_action_target = self.storage.policy_action_target.transpose(0, 1).to(device)
            policy_action_target_mask = (
                self.storage.policy_action_target_mask.transpose(0, 1).to(device)
            )
            if self.balance_staged_task_samples:
                local_active_counts = torch.zeros(
                    int(self.env.num_stages), device=device, dtype=torch.float32
                )
                for stage_idx in range(int(self.env.num_stages)):
                    local_active_counts[stage_idx] = policy_action_target_mask[
                        task_stages == stage_idx
                    ].sum()
                global_active_counts = self.accelerator.reduce(
                    local_active_counts, reduction="sum"
                )
                active_stages = global_active_counts > 0
                active_stage_count = active_stages.sum().clamp(min=1)
                total_active_count = global_active_counts.sum().clamp(min=1.0)
                supervision_stage_weights = torch.zeros_like(global_active_counts)
                supervision_stage_weights[active_stages] = total_active_count / (
                    active_stage_count * global_active_counts[active_stages]
                )
                self._last_task_stage_supervision_weights = (
                    supervision_stage_weights.detach()
                )
                task_stage_supervision_weights = supervision_stage_weights[task_stages]

        if self.use_padding_mask:
            padding_mask = dones.clone()
            padding_mask_p1 = padding_mask.clone()
            for i in range(padding_mask.shape[0]):
                true_indices = torch.where(padding_mask[i])[0]
                if len(true_indices) > 0:
                    padding_mask[i, true_indices[0]] = False
                    padding_mask_p1[
                        i, true_indices[0] : min(true_indices[0] + 2, padding_mask_p1.shape[1])
                    ] = False
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            values = torch.masked_fill(values, padding_mask_p1, 0)
        else:
            padding_mask = torch.zeros_like(dones)
            padding_mask_p1 = torch.zeros_like(dones)

        # For recurrent policies/critics, pre-split trajectories ONCE globally
        padded_obs_dict = None
        trajectory_masks = None
        trajectory_offsets = None
        trajectory_counts = None
        if (hasattr(self.policy_model, "is_recurrent") and self.policy_model.is_recurrent) or (
            hasattr(self.value_model, "is_recurrent") and self.value_model.is_recurrent
        ):
            from gr00t.rl.trl.utils.rl import split_and_pad_trajectories

            padded_obs_dict = {}
            for key in obs_keys:
                # all_obs_dict[key]: [num_envs, num_steps, ...]
                # Transpose to [num_steps, num_envs, ...] for split_and_pad_trajectories
                obs_transposed = all_obs_dict[key].transpose(0, 1)
                dones_transposed = dones.transpose(0, 1)
                padded_obs, traj_masks = split_and_pad_trajectories(
                    obs_transposed, dones_transposed
                )
                # padded_obs: [max_traj_len, num_trajectories, ...]
                # Transpose to [num_trajectories, max_traj_len, ...]
                padded_obs_dict[key] = padded_obs.transpose(0, 1)
                if trajectory_masks is None:
                    # traj_masks: [max_traj_len, num_trajectories]
                    # Transpose to [num_trajectories, max_traj_len]
                    trajectory_masks = traj_masks.transpose(0, 1)
            trajectory_counts = dones[:, :-1].bool().sum(dim=1).to(torch.long) + 1
            trajectory_offsets = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=device),
                    torch.cumsum(trajectory_counts, dim=0)[:-1],
                ]
            )

        rollout_data = dict(
            all_obs_dict=all_obs_dict,
            actions=actions,
            logprobs=logprobs,
            values=values,
            rewards=rewards,
            dones=dones,
            old_mu_batch=old_mu_batch,
            old_sigma_batch=old_sigma_batch,
            temporal_fractions=temporal_fractions,
            returns=returns,
            advantages=advantages,
            task_stage_weights=task_stage_weights,
            task_stage_supervision_weights=task_stage_supervision_weights,
            padding_mask=padding_mask,
            padding_mask_p1=padding_mask_p1,
            padded_obs_dict=padded_obs_dict,
            trajectory_masks=trajectory_masks,
            trajectory_offsets=trajectory_offsets,
            trajectory_counts=trajectory_counts,
            policy_action_target=policy_action_target,
            policy_action_target_mask=policy_action_target_mask,
        )
        return rollout_data

    def _get_mb_rollout_data(self, rollout_data, micro_batch_inds):
        mb_advantage = rollout_data["advantages"][micro_batch_inds]
        mb_logprobs = rollout_data["logprobs"][micro_batch_inds]
        mb_return = rollout_data["returns"][micro_batch_inds]
        mb_values = rollout_data["values"][micro_batch_inds]
        mb_dones = rollout_data["dones"][micro_batch_inds]
        mb_actions = rollout_data["actions"][micro_batch_inds]
        mb_old_mu = rollout_data["old_mu_batch"][micro_batch_inds]
        mb_old_sigma = rollout_data["old_sigma_batch"][micro_batch_inds]
        mb_temporal_fractions = None
        if rollout_data["temporal_fractions"] is not None:
            mb_temporal_fractions = rollout_data["temporal_fractions"][
                micro_batch_inds
            ]
        mb_padding_mask = rollout_data["padding_mask"][micro_batch_inds]
        mb_padding_mask_p1 = rollout_data["padding_mask_p1"][micro_batch_inds]
        mb_policy_action_target = None
        mb_policy_action_target_mask = None
        mb_task_stage_weights = None
        mb_task_stage_supervision_weights = None
        if rollout_data["task_stage_weights"] is not None:
            mb_task_stage_weights = rollout_data["task_stage_weights"][micro_batch_inds]
        if rollout_data["task_stage_supervision_weights"] is not None:
            mb_task_stage_supervision_weights = rollout_data[
                "task_stage_supervision_weights"
            ][micro_batch_inds]
        if self.compute_policy_action_supervision:
            mb_policy_action_target = rollout_data["policy_action_target"][micro_batch_inds]
            mb_policy_action_target_mask = rollout_data[
                "policy_action_target_mask"
            ][micro_batch_inds]

        episode_attnmask = compute_episode_attnmask(mb_dones)

        # For recurrent policies, SLICE pre-split trajectories instead of re-splitting
        mb_hidden_states = None
        if hasattr(self.policy_model, "is_recurrent") and self.policy_model.is_recurrent:
            if (
                rollout_data.get("padded_obs_dict") is None
                or rollout_data.get("trajectory_masks") is None
            ):
                raise RuntimeError(
                    "Recurrent policy requires padded_obs_dict and trajectory_masks in rollout_data! This should have been created in _get_rollout_data."
                )
            trajectory_offsets = rollout_data.get("trajectory_offsets")
            trajectory_counts = rollout_data.get("trajectory_counts")
            if trajectory_offsets is None or trajectory_counts is None:
                raise RuntimeError("Recurrent rollout data is missing trajectory index metadata.")
            trajectory_indices = torch.cat(
                [
                    torch.arange(
                        trajectory_offsets[env_idx],
                        trajectory_offsets[env_idx] + trajectory_counts[env_idx],
                        device=trajectory_offsets.device,
                    )
                    for env_idx in micro_batch_inds
                ]
            )
            mb_obs_dict = {
                key: rollout_data["padded_obs_dict"][key].index_select(0, trajectory_indices)
                for key in rollout_data["padded_obs_dict"].keys()
            }
            mb_masks = rollout_data["trajectory_masks"].index_select(0, trajectory_indices)

            # Extract hidden states at trajectory boundaries if available
            if self.storage.saved_hidden_states_a is not None or self.storage.saved_hidden_states_c is not None:  # type: ignore[attr-defined]
                # Determine which timesteps are right after dones (start of trajectories)
                dones_for_extraction = mb_dones.clone()
                last_was_done = torch.zeros_like(dones_for_extraction, dtype=torch.bool)
                last_was_done[:, 1:] = dones_for_extraction[:, :-1].bool()
                last_was_done[:, 0] = True
                # Permute to [num_steps, batch_size]
                last_was_done_transposed = last_was_done.transpose(0, 1)
                # Extract actor hidden states
                if self.storage.saved_hidden_states_a is not None:  # type: ignore[attr-defined]
                    hid_a_batch = []
                    for saved_hidden_states in self.storage.saved_hidden_states_a:  # type: ignore[attr-defined]
                        saved_hid_mb = saved_hidden_states[:, :, micro_batch_inds, :]
                        saved_hid_mb = saved_hid_mb.permute(2, 0, 1, 3)
                        saved_hid_mb_flat = saved_hid_mb.reshape(
                            -1, saved_hid_mb.shape[2], saved_hid_mb.shape[3]
                        )
                        hid_at_traj_starts = saved_hid_mb_flat[last_was_done.reshape(-1)]
                        hid_at_traj_starts = hid_at_traj_starts.transpose(1, 0).contiguous()
                        hid_a_batch.append(hid_at_traj_starts)
                    hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else tuple(hid_a_batch)
                else:
                    hid_a_batch = None
                # Extract critic hidden states
                if self.storage.saved_hidden_states_c is not None:  # type: ignore[attr-defined]
                    hid_c_batch = []
                    for saved_hidden_states in self.storage.saved_hidden_states_c:  # type: ignore[attr-defined]
                        saved_hid_mb = saved_hidden_states[:, :, micro_batch_inds, :]
                        saved_hid_mb = saved_hid_mb.permute(2, 0, 1, 3)
                        saved_hid_mb_flat = saved_hid_mb.reshape(
                            -1, saved_hid_mb.shape[2], saved_hid_mb.shape[3]
                        )
                        hid_at_traj_starts = saved_hid_mb_flat[last_was_done.reshape(-1)]
                        hid_at_traj_starts = hid_at_traj_starts.transpose(1, 0).contiguous()
                        hid_c_batch.append(hid_at_traj_starts)
                    hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else tuple(hid_c_batch)
                else:
                    hid_c_batch = None
                mb_hidden_states = (hid_a_batch, hid_c_batch)
        else:
            # Non-recurrent: standard [batch_size, num_steps, ...] format
            mb_obs_dict = {
                key: rollout_data["all_obs_dict"][key][micro_batch_inds]
                for key in rollout_data["all_obs_dict"].keys()
            }
            mb_masks = None

        mb_rollout_data = dict(
            micro_batch_inds=micro_batch_inds,
            mb_obs_dict=mb_obs_dict,
            mb_advantage=mb_advantage,
            mb_logprobs=mb_logprobs,
            mb_return=mb_return,
            mb_values=mb_values,
            mb_dones=mb_dones,
            mb_actions=mb_actions,
            mb_old_mu=mb_old_mu,
            mb_old_sigma=mb_old_sigma,
            mb_temporal_fractions=mb_temporal_fractions,
            mb_padding_mask=mb_padding_mask,
            mb_padding_mask_p1=mb_padding_mask_p1,
            episode_attnmask=episode_attnmask,
            mb_masks=mb_masks,
            mb_hidden_states=mb_hidden_states,
            mb_policy_action_target=mb_policy_action_target,
            mb_policy_action_target_mask=mb_policy_action_target_mask,
            mb_task_stage_weights=mb_task_stage_weights,
            mb_task_stage_supervision_weights=mb_task_stage_supervision_weights,
        )
        return mb_rollout_data

    def _forward_model(self, model, mb_rollout_data):
        mb_obs_dict = mb_rollout_data["mb_obs_dict"]
        mb_actions = mb_rollout_data["mb_actions"]
        episode_attnmask = mb_rollout_data["episode_attnmask"]
        mb_masks = mb_rollout_data.get("mb_masks", None)
        mb_dones = mb_rollout_data["mb_dones"]
        mb_hidden_states = mb_rollout_data.get("mb_hidden_states", None)

        # Extract separate hidden states for actor and critic (like rsl_rl)
        actor_hidden_states = mb_hidden_states[0] if mb_hidden_states is not None else None
        critic_hidden_states = mb_hidden_states[1] if mb_hidden_states is not None else None

        # We should only do one forward pass for especially DDP model
        if self.compute_imgaug_bc_loss:
            results = model.forward(
                modes=["policy_w_and_wo_imgaug", "value"],
                input_kwargs=dict(
                    policy_w_and_wo_imgaug=dict(
                        obs_dict=mb_obs_dict,
                        actions=mb_actions,
                        episode_attnmask=episode_attnmask,
                        masks=mb_masks,
                        hidden_states=actor_hidden_states,
                        original_dones=mb_dones,
                    ),
                    value=dict(
                        obs_dict=mb_obs_dict,
                        episode_attnmask=episode_attnmask,
                        masks=mb_masks,
                        hidden_states=critic_hidden_states,
                        original_dones=mb_dones,
                    ),
                ),
            )
            policy_results = results["policy_w_and_wo_imgaug"]
        else:
            results = model.forward(
                modes=["policy", "value"],
                input_kwargs=dict(
                    policy=dict(
                        obs_dict=mb_obs_dict,
                        actions=mb_actions,
                        episode_attnmask=episode_attnmask,
                        masks=mb_masks,
                        hidden_states=actor_hidden_states,
                        original_dones=mb_dones,
                    ),
                    value=dict(
                        obs_dict=mb_obs_dict,
                        episode_attnmask=episode_attnmask,
                        masks=mb_masks,
                        hidden_states=critic_hidden_states,
                        original_dones=mb_dones,
                    ),
                ),
            )
            policy_results = results["policy"]

        return dict(
            policy_results=policy_results,
            value_results=results["value"],
        )

    def _compute_loss(self, forward_results, mb_rollout_data):
        ppo_loss_dict = self._compute_ppo_loss(forward_results, mb_rollout_data)

        loss = ppo_loss_dict["ppo_loss"] * self.config.get("ppo_loss_coef", 1.0)

        ret_dict = dict(
            ppo_loss_dict=ppo_loss_dict,
        )

        if self.compute_imgaug_bc_loss:
            imgaug_bc_loss_dict = self._compute_imgaug_bc_loss(forward_results, mb_rollout_data)
            loss += imgaug_bc_loss_dict["imgaug_bc_loss"] * self.config.imgaug_bc_loss_coef
            ret_dict["imgaug_bc_loss_dict"] = imgaug_bc_loss_dict

        if self.compute_policy_action_supervision:
            action_mean = forward_results["policy_results"]["action_mean"]
            action_target = mb_rollout_data["mb_policy_action_target"]
            action_target_mask = mb_rollout_data["mb_policy_action_target_mask"]
            if action_mean.shape != action_target.shape:
                raise RuntimeError(
                    "Policy action supervision currently requires matching dense rollout "
                    f"shapes, got mean={action_mean.shape}, target={action_target.shape}"
                )
            supervision_error = torch.nn.functional.smooth_l1_loss(
                action_mean,
                action_target,
                reduction="none",
                beta=self.policy_action_supervision_huber_beta,
            )
            supervision_weight = action_target_mask
            stage_weights = mb_rollout_data["mb_task_stage_supervision_weights"]
            if stage_weights is not None:
                while stage_weights.ndim < supervision_weight.ndim:
                    stage_weights = stage_weights.unsqueeze(-1)
                supervision_weight = supervision_weight * stage_weights
            active_count = supervision_weight.sum().clamp(min=1.0)
            action_supervision_loss = (
                supervision_error * supervision_weight
            ).sum() / active_count
            loss += self.policy_action_supervision_coef * action_supervision_loss
            ret_dict["policy_action_supervision_loss"] = action_supervision_loss

        ret_dict["loss"] = loss

        return ret_dict

    def _compute_ppo_loss(self, forward_results, mb_rollout_data):
        args = self.args
        optimizer = self.optimizer

        policy_results = forward_results["policy_results"]
        value_results = forward_results["value_results"]

        mb_old_mu = mb_rollout_data["mb_old_mu"]
        mb_old_sigma = mb_rollout_data["mb_old_sigma"]
        mb_values = mb_rollout_data["mb_values"]
        mb_return = mb_rollout_data["mb_return"]
        mb_logprobs = mb_rollout_data["mb_logprobs"]
        mb_advantage = mb_rollout_data["mb_advantage"]
        mb_task_stage_weights = mb_rollout_data["mb_task_stage_weights"]
        padding_mask = mb_rollout_data["mb_padding_mask"]
        padding_mask_p1 = mb_rollout_data["mb_padding_mask_p1"]
        micro_batch_inds = mb_rollout_data["micro_batch_inds"]

        temporal_noise_enabled = getattr(
            self.policy_model, "temporal_action_noise_enabled", False
        )
        if temporal_noise_enabled:
            mu_batch = policy_results["action_mean"]
            sigma_batch = policy_results["marginal_action_std"]
            global_sigma_batch = policy_results["global_action_std"]
            new_temporal_fractions = policy_results["action_temporal_fraction"]
            (
                old_conditional_mean,
                old_conditional_std,
                new_conditional_mean,
                new_conditional_std,
            ) = paired_temporal_gaussian_parameters(
                mb_rollout_data["mb_actions"],
                mb_old_mu,
                mb_old_sigma,
                mb_rollout_data["mb_temporal_fractions"],
                mu_batch,
                sigma_batch,
                new_temporal_fractions,
                mb_rollout_data["mb_dones"],
                self.policy_model.temporal_action_noise_rho,
            )
            new_logprobs = torch.distributions.Normal(
                new_conditional_mean, new_conditional_std
            ).log_prob(mb_rollout_data["mb_actions"]).sum(dim=-1)
            _, global_conditional_std = marginal_temporal_gaussian_parameters(
                mb_rollout_data["mb_actions"],
                mu_batch,
                global_sigma_batch,
                mb_rollout_data["mb_dones"],
                self.policy_model.temporal_action_noise_rho,
                new_temporal_fractions.detach(),
            )
            entropy_batch = torch.distributions.Normal(
                torch.zeros_like(global_conditional_std), global_conditional_std
            ).entropy().sum(dim=-1)
            old_conditional_mean = old_conditional_mean.detach()
            old_conditional_std = old_conditional_std.detach()
        else:
            new_logprobs = policy_results["logprobs"]
            sigma_batch = policy_results["action_std"]
            mu_batch = policy_results["action_distribution_mean"]
            if self.state_dependent_action_noise:
                global_sigma_batch = policy_results["global_action_std"]
                entropy_batch = torch.distributions.Normal(
                    torch.zeros_like(global_sigma_batch), global_sigma_batch
                ).entropy().sum(dim=-1)
            else:
                entropy_batch = policy_results["entropy"]
            old_conditional_mean = mb_old_mu
            old_conditional_std = mb_old_sigma
            new_conditional_mean = mu_batch
            new_conditional_std = sigma_batch
        with torch.no_grad():
            kl = torch.sum(
                torch.log(new_conditional_std / old_conditional_std + 1.0e-5)
                + (
                    torch.square(old_conditional_std)
                    + torch.square(old_conditional_mean - new_conditional_mean)
                )
                / (2.0 * torch.square(new_conditional_std))
                - 0.5,
                axis=-1,
            )
            local_kl_mean = torch.mean(kl)
            kl_mean = self.accelerator.gather(local_kl_mean).mean()

        # Forward a DDP model twice will cause the error: "one of the variables needed for gradient computation has been modified by an inplace operation"
        vpred = value_results.squeeze(-1)
        vpredclipped = torch.clamp(
            vpred,
            mb_values - args.cliprange_value,
            mb_values + args.cliprange_value,
        )
        vf_losses1 = torch.square(vpred - mb_return)
        vf_losses2 = torch.square(vpredclipped - mb_return)
        vf_loss_max = torch.max(vf_losses1, vf_losses2)
        if mb_task_stage_weights is None:
            vf_loss = masked_mean(vf_loss_max, ~padding_mask_p1)
        else:
            vf_weights = mb_task_stage_weights * (~padding_mask_p1).float()
            vf_loss = (vf_loss_max * vf_weights).sum() / vf_weights.sum().clamp(
                min=1.0
            )
        vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), ~padding_mask_p1)
        logprobs_diff = new_logprobs - mb_logprobs
        ratio = torch.exp(logprobs_diff)
        pg_losses = -mb_advantage * ratio
        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        if mb_task_stage_weights is None:
            pg_loss = masked_mean(pg_loss_max, ~padding_mask)
        else:
            pg_weights = mb_task_stage_weights * (~padding_mask).float()
            pg_loss = (pg_loss_max * pg_weights).sum() / pg_weights.sum().clamp(
                min=1.0
            )

        entropy_loss = -masked_mean(entropy_batch, ~padding_mask)

        exploration_budget_loss = torch.zeros((), device=pg_loss.device)
        exploration_mean_variance = None
        if self.state_dependent_action_noise:
            exploration_mean_variance = normalized_action_variance(
                sigma_batch, self.policy_model.max_noise_std
            )
            target = torch.full_like(
                exploration_mean_variance,
                self.exploration_variance_budget_fraction,
            )
            active = (
                self._last_exploration_budget_shortfall > 0.0
            ).to(exploration_mean_variance.dtype)
            dual_weight = self.policy_model.exploration_budget_dual_weight.detach()
            exploration_budget_loss = (
                dual_weight * active * (target - exploration_mean_variance)
            ).sum()

        loss = (
            pg_loss
            + args.vf_coef * vf_loss
            + self.entropy_coef * entropy_loss
            + exploration_budget_loss
        )

        return dict(
            ppo_loss=loss,
            # logging metrics
            local_kl_mean=local_kl_mean,
            pg_losses=pg_losses,
            pg_losses2=pg_losses2,
            pg_loss=pg_loss,
            vf_loss=vf_loss,
            entropy_loss=entropy_loss,
            exploration_budget_loss=exploration_budget_loss,
            exploration_mean_variance=exploration_mean_variance,
            ratio=ratio,
            vf_clipfrac=vf_clipfrac,
        )

    def _compute_imgaug_bc_loss(self, forward_results, mb_rollout_data):
        policy_results = forward_results["policy_results"]
        mu_batch = policy_results["action_mean"]

        action_mean_w_imgaug = policy_results["action_mean_w_imgaug"]
        imgaug_bc_loss = self.imgaug_bc_loss_fn(action_mean_w_imgaug, mu_batch.detach())

        return dict(
            imgaug_bc_loss=imgaug_bc_loss,
        )

    def _update_stats_buffer(
        self,
        ppo_epoch_idx,
        minibatch_idx,
        microbatch_idx,
        loss_dict,
        forward_results,
        mb_rollout_data,
    ):
        local_kl_mean = loss_dict["ppo_loss_dict"]["local_kl_mean"]
        pg_losses = loss_dict["ppo_loss_dict"]["pg_losses"]
        pg_losses2 = loss_dict["ppo_loss_dict"]["pg_losses2"]
        pg_loss = loss_dict["ppo_loss_dict"]["pg_loss"]
        vf_loss = loss_dict["ppo_loss_dict"]["vf_loss"]
        entropy_loss = loss_dict["ppo_loss_dict"]["entropy_loss"]
        weighted_ppo_loss = loss_dict["ppo_loss_dict"]["ppo_loss"] * self.config.get(
            "ppo_loss_coef", 1.0
        )
        ratio = loss_dict["ppo_loss_dict"]["ratio"]
        vf_clipfrac = loss_dict["ppo_loss_dict"]["vf_clipfrac"]

        padding_mask = mb_rollout_data["mb_padding_mask"]
        micro_batch_inds = mb_rollout_data["micro_batch_inds"]

        self.approxkl_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = local_kl_mean
        pg_clipfrac = masked_mean((pg_losses2 > pg_losses).float(), ~padding_mask)
        self.pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = pg_clipfrac
        self.pg_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = pg_loss
        self.vf_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = vf_loss
        if self.compute_imgaug_bc_loss:
            imgaug_bc_loss = loss_dict["imgaug_bc_loss_dict"]["imgaug_bc_loss"]
            self.imgaug_bc_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = imgaug_bc_loss
            self.weighted_imgaug_bc_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = (
                self.config.imgaug_bc_loss_coef * imgaug_bc_loss
            )
        if self.compute_policy_action_supervision:
            self.policy_action_supervision_stats[
                ppo_epoch_idx, minibatch_idx, microbatch_idx
            ] = loss_dict["policy_action_supervision_loss"]
        self.entropy_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = -entropy_loss
        self.weighted_ppo_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = (
            weighted_ppo_loss
        )
        self.vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = vf_clipfrac
        self.ratio_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = ratio.mean()

    def _get_train_metrics(self):
        metrics = {}

        if self.balance_staged_task_samples:
            stage_names = getattr(self.env, "_STAGE_NAMES", ())
            for stage_idx in range(int(self.env.num_stages)):
                stage_name = (
                    stage_names[stage_idx]
                    if stage_idx < len(stage_names)
                    else str(stage_idx)
                )
                metrics[f"train/stage_count/{stage_name}"] = (
                    self._last_task_stage_counts[stage_idx].item()
                )
                metrics[f"train/stage_weight/{stage_name}"] = (
                    self._last_task_stage_weights[stage_idx].item()
                )
                if self.compute_policy_action_supervision:
                    metrics[f"train/stage_supervision_weight/{stage_name}"] = (
                        self._last_task_stage_supervision_weights[stage_idx].item()
                    )
                metrics[f"train/stage_reward_mean/{stage_name}"] = (
                    self._last_stage_reward_mean[stage_idx].item()
                )
                if self._last_stage_action_advantage_signal is not None:
                    for action_idx in range(self.num_act):
                        metrics[
                            f"train/stage_action_advantage_signal/{stage_name}/{action_idx}"
                        ] = self._last_stage_action_advantage_signal[
                            stage_idx, action_idx
                        ].item()
            for name, value in getattr(
                self, "_last_push_causal_metrics", {}
            ).items():
                metrics[f"train/push_open_causal/{name}"] = value.item()
        approxkl_avg = self.accelerator.gather_for_metrics(self.approxkl_stats).mean().item()

        metrics["policy/approxkl_avg"] = approxkl_avg
        metrics["policy/clipfrac_avg"] = (
            self.accelerator.gather_for_metrics(self.pg_clipfrac_stats).mean().item()
        )
        metrics["loss/policy_avg"] = (
            self.accelerator.gather_for_metrics(self.pg_loss_stats).mean().item()
        )
        if self.compute_imgaug_bc_loss:
            metrics["loss/imgaug_bc_avg"] = (
                self.accelerator.gather_for_metrics(self.imgaug_bc_loss_stats).mean().item()
            )
            metrics["loss/weighted_imgaug_bc_avg"] = (
                self.accelerator.gather_for_metrics(self.weighted_imgaug_bc_loss_stats)
                .mean()
                .item()
            )
        if self.compute_policy_action_supervision:
            metrics["loss/policy_action_supervision_avg"] = (
                self.accelerator.gather_for_metrics(
                    self.policy_action_supervision_stats
                )
                .mean()
                .item()
            )
        metrics["loss/value_avg"] = (
            self.accelerator.gather_for_metrics(self.vf_loss_stats).mean().item()
        )
        metrics["loss/entropy_avg"] = (
            self.accelerator.gather_for_metrics(self.entropy_stats).mean().item()
        )
        metrics["loss/weighted_ppo_loss_avg"] = (
            self.accelerator.gather_for_metrics(self.weighted_ppo_loss_stats).mean().item()
        )
        metrics["val/clipfrac_avg"] = (
            self.accelerator.gather_for_metrics(self.vf_clipfrac_stats).mean().item()
        )
        metrics["val/ratio"] = self.accelerator.gather_for_metrics(self.ratio_stats).mean().item()
        metrics["val/ratio_var"] = (
            self.accelerator.gather_for_metrics(self.ratio_stats).var(unbiased=False).item()
        )
        metrics["objective/entropy"] = metrics["loss/entropy_avg"]

        return metrics

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                if dataloader is not None:
                    yield from dataloader
                else:
                    yield None

        iter_dataloader = iter(repeat_generator())

        accelerator.print("===training policy===")
        start_time = time.time()
        self._register_stats_buffer()
        model.train()

        # trainer state initialization
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        # env
        obs_dict = self.env.reset_all()
        if self.config.get("init_at_random_ep_len", False):
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
            from gr00t.rl.envs.base_task.staged_task_base import StagedTaskBase

            if isinstance(self.env, StagedTaskBase):
                self.env.time_in_stage_buf[:] = torch.randint_like(
                    self.env.time_in_stage_buf, high=int(self.env.max_stage_time[0])
                )
                self.env.actual_time_in_stage_buf[:] = self.env.time_in_stage_buf
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(device)

        for batch_idx in range(1, args.num_total_batches + 1):
            batch_start_time = time.time()
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            # update scheduled params
            if self.schedule_dict is not None:
                self.scheduled_params_dict = update_scheduled_params(
                    self, self.schedule_dict, self.state.global_step
                )

            reinit_sim_freq = self.env.config.get("reinit_sim_freq", 0)
            if reinit_sim_freq > 0 and (self.state.global_step + 1) % reinit_sim_freq == 0:
                self.env.reinit_sim()
            # DEBUG: Print model weights and learning rate (every 5 batches to reduce clutter)
            if batch_idx == 0:
                # Check if model is wrapped in DDP
                model_type = type(self.model).__name__
                if self.accelerator.process_index == 0:
                    print(f"\n{'='*80}")
                    print(f"[CRITICAL] Model type: {model_type}")
                    print(f"[CRITICAL] Num processes: {self.accelerator.num_processes}")
                    print(f"[CRITICAL] Distributed type: {self.accelerator.distributed_type}")
                    print(f"[CRITICAL] Use distributed: {self.accelerator.use_distributed}")
                    print("  >> Model should be DistributedDataParallel if multi-GPU!")
                    print(f"{'='*80}\n")

            with torch.no_grad():
                with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as model:
                    obs_dict = self._rollout_step(model, obs_dict)

                end_collection_time = time.time()
                collection_time = end_collection_time - batch_start_time

                rollout_data = self._get_rollout_data(obs_keys=obs_dict.keys())
                torch.cuda.empty_cache()
                gc.collect()

            model = self.model
            self._train_mode()
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                minibatch_idx = 0
                if self.ppo_shuffle_every_epoch or ppo_epoch_idx == 0:
                    b_inds = torch.randperm(args.local_batch_size, device=device)
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    microbatch_idx = 0
                    for micro_batch_start in range(
                        0, args.local_mini_batch_size, args.per_device_train_batch_size
                    ):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            mb_rollout_data = self._get_mb_rollout_data(
                                rollout_data, micro_batch_inds
                            )

                            forward_results = self._forward_model(model, mb_rollout_data)

                            loss_dict = self._compute_loss(forward_results, mb_rollout_data)

                            accelerator.backward(loss_dict["loss"])
                            self._gradient_clipping()
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                self._update_stats_buffer(
                                    ppo_epoch_idx,
                                    minibatch_idx,
                                    microbatch_idx,
                                    loss_dict,
                                    forward_results,
                                    mb_rollout_data,
                                )
                            del loss_dict, forward_results, mb_rollout_data
                            microbatch_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                if (
                    self.empty_cache_every_n_ppo_epoch > 0
                    and ppo_epoch_idx % self.empty_cache_every_n_ppo_epoch == 0
                ):
                    torch.cuda.empty_cache()
                    gc.collect()

            with torch.no_grad():
                learn_time = time.time() - end_collection_time
                eps = int(self.state.episode / (time.time() - start_time))

                metrics = {}
                train_metrics = self._get_train_metrics()
                metrics.update(train_metrics)
                metrics["eps"] = eps

                metrics["objective/rewards"] = (
                    self.accelerator.gather_for_metrics(
                        torch.tensor(
                            np.mean(self.state.rewbuffer) if len(self.state.rewbuffer) > 0 else 0,
                            dtype=torch.float32,
                        ).to(device)
                    )
                    .mean()
                    .item()
                )
                metrics["objective/length"] = (
                    self.accelerator.gather_for_metrics(
                        torch.tensor(
                            np.mean(self.state.lenbuffer) if len(self.state.lenbuffer) > 0 else 0,
                            dtype=torch.float32,
                        ).to(device)
                    )
                    .mean()
                    .item()
                )

                # Add goal reached rate from environment if available
                # The environment already calculates the average, so we can use it directly
                # if "recent_goal_reached_rate" in infos:
                #     metrics["objective/goal_reached_rate"] = self.accelerator.gather_for_metrics(torch.tensor(infos["recent_goal_reached_rate"]).to(device)).mean().item()
                # else:
                #     metrics["objective/goal_reached_rate"] = 0.0

                metrics["lr"] = optimizer.param_groups[0]["lr"]
                metrics["episode"] = self.state.episode
                env_log_dict_local = self.episode_env_tensors.mean_and_clear()
                # Synchronize environment metrics across all GPUs to properly aggregate
                # metrics from all environments (e.g., 1024 envs/GPU * 8 GPUs = 8192 total envs)
                env_log_dict = {
                    k: (
                        self.accelerator.gather_for_metrics(
                            torch.tensor(v, dtype=torch.float32).to(device)
                        )
                        .mean()
                        .item()
                        if not isinstance(v, (int, float))
                        else self.accelerator.gather_for_metrics(
                            torch.tensor(v, dtype=torch.float32).to(device)
                        )
                        .mean()
                        .item()
                    )
                    for k, v in env_log_dict_local.items()
                }
                ep_infos = process_ep_infos(self.ep_infos, device)
                self.state.tot_timesteps += (
                    self.num_steps_per_env * self.env.num_envs * accelerator.num_processes
                )
                self.state.tot_time += collection_time + learn_time
                log_dict = {
                    "collection_time": collection_time,
                    "learn_time": learn_time,
                    "tot_timesteps": self.state.tot_timesteps,
                    "tot_time": self.state.tot_time,
                    "it": self.state.global_step,
                    "fps": int(
                        self.num_steps_per_env
                        * self.env.num_envs
                        * accelerator.num_processes
                        / (collection_time + learn_time)
                    ),
                    "experiment_save_dir": self.args.output_dir,
                    "batch_idx": batch_idx,
                    "num_total_batches": args.num_total_batches,
                }

                for key, value in ep_infos.items():
                    log_dict[f"Episode/{key}"] = value

                # Add scheduled parameters to metrics
                for param_name, param_value in self.scheduled_params_dict.items():
                    log_dict[f"scheduled_params/{param_name}"] = param_value

                if hasattr(self.storage, "action_sigma"):
                    metrics["Policy/mean_noise_std"] = (
                        self.storage.action_sigma.mean().item()
                    )
                else:
                    metrics["Policy/mean_noise_std"] = 0.0
                if (
                    self.state_dependent_action_noise
                    and self._last_exploration_mean_variance is not None
                ):
                    metrics["Policy/exploration_variance_mean"] = float(
                        self._last_exploration_mean_variance.mean()
                    )
                    metrics["Policy/exploration_budget_shortfall_max"] = float(
                        self._last_exploration_budget_shortfall.max()
                    )
                    metrics["Policy/exploration_budget_dual_mean"] = float(
                        self.policy_model.exploration_budget_dual_weight.mean()
                    )
                if self._last_temporal_fraction_stats is not None:
                    for action_idx in range(self.num_act):
                        for statistic, values in (
                            self._last_temporal_fraction_stats.items()
                        ):
                            metrics[
                                f"Policy/temporal_fraction/{action_idx}/{statistic}"
                            ] = float(values[action_idx])

                metrics.update({f"Env/{k}": v for k, v in env_log_dict.items()})
                metrics.update(log_dict)

                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1

                self.log(metrics)
                self.ep_infos.clear()

            if self.use_adaptive_kl_schedule:
                self._adjust_learning_rate_based_on_kl(
                    self.accelerator.gather_for_metrics(self.approxkl_stats)
                    .mean()
                    .item(),
                    optimizer,
                )
            else:
                self.lr_scheduler.step()

            self.control = self.callback_handler.on_step_end(args, self.state, self.control)

            del (
                metrics,
                rollout_data,
            )
            torch.cuda.empty_cache()
            gc.collect()

            if self.control.should_training_stop:
                break

        if self.control.should_training_stop:
            return

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        if wandb_run_exists():
            wandb.finish()

    def _eval_mode(self):
        self.model.eval()
        model = self.accelerator.unwrap_model(self.model)
        model.set_mode("eval")
        model.transform_eval()
        self.env.set_is_evaluating(is_evaluating=True, log_info=False)

    def _train_rollout_mode(self):
        self.model.eval()
        model = self.accelerator.unwrap_model(self.model)
        model.set_mode("train_rollout")
        model.transform_eval()
        if self.train_with_evaluating_env:
            self.env.set_is_evaluating(True, log_info=False)
        else:
            self.env.set_is_evaluating(False, log_info=False)

    def _train_mode(self):
        self.model.train()
        model = self.accelerator.unwrap_model(self.model)
        model.set_mode("train")
        model.transform_train()
        if self.train_with_evaluating_env:
            self.env.set_is_evaluating(True, log_info=False)
        else:
            self.env.set_is_evaluating(False, log_info=False)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        append_training_metrics_jsonl(
            self.args.output_dir,
            output,
            is_world_process_zero=self.state.is_world_process_zero,
        )

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _gradient_clipping(self):
        # Gradient clipping
        args = self.args
        model = self.model
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            # deepspeed does its own clipping

            if is_sagemaker_mp_enabled() and args.fp16:
                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
            elif self.use_apex:
                # Revert to normal clipping otherwise, handling Apex or full precision
                _grad_norm = nn.utils.clip_grad_norm_(
                    amp.master_params(self.optimizer),
                    args.max_grad_norm,
                )
            elif self.config.get("separate_actor_critic_grad_clip", False):
                actor_parameters = [
                    parameter
                    for parameter in self.policy_model.parameters()
                    if parameter.grad is not None
                ]
                critic_parameters = (
                    [
                        parameter
                        for parameter in self.value_model.parameters()
                        if parameter.grad is not None
                    ]
                    if self.value_model is not None
                    else []
                )
                clipped_norms = []
                if actor_parameters:
                    clipped_norms.append(
                        self.accelerator.clip_grad_norm_(
                            actor_parameters, args.max_grad_norm
                        )
                    )
                if critic_parameters:
                    clipped_norms.append(
                        self.accelerator.clip_grad_norm_(
                            critic_parameters, args.max_grad_norm
                        )
                    )
                _grad_norm = max(clipped_norms) if clipped_norms else 0.0
            else:
                _grad_norm = self.accelerator.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )

            if (
                is_accelerate_available()
                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
            ):
                grad_norm = model.get_global_grad_norm()
                # In some cases the grad norm may not return a float
                if hasattr(grad_norm, "item"):
                    grad_norm = grad_norm.item()
            else:
                grad_norm = _grad_norm

        return grad_norm

    def _compute_returns(self, values, last_values, policy_state_dict):
        """Compute the returns and advantages for the given policy state.
        This function calculates the returns and advantages for each step in the
        environment based on the provided observations and policy state. It uses
        Generalized Advantage Estimation (GAE) to compute the advantages, which
        helps in reducing the variance of the policy gradient estimates.
        Args:
            values (torch.Tensor): The values for each step.
            last_values (torch.Tensor): The last values for the last step.
            policy_state_dict (dict): A dictionary containing the policy state
                          information, including 'values', 'dones',
                          and 'rewards'.
        Returns:
            tuple: A tuple containing:
            - returns (torch.Tensor): The computed returns for each step.
            - advantages (torch.Tensor): The normalized advantages for each step.
        """
        device = self.accelerator.device
        advantage = 0

        dones = policy_state_dict["dones"]
        rewards = policy_state_dict["rewards"]

        dones = dones.to(device)
        rewards = rewards.to(device)

        returns = torch.zeros_like(values)

        num_steps = returns.shape[0]

        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]

        advantages = returns - values
        task_stages = policy_state_dict.get("task_stages")
        if self.balance_staged_task_samples:
            if task_stages is None:
                raise RuntimeError(
                    "Stage-balanced PPO requires task_stages in rollout state"
                )
            task_stages = task_stages.to(device=device, dtype=torch.long)
            task_variants = policy_state_dict.get("task_variants")
            if self.balance_staged_task_variants:
                if task_variants is None:
                    raise RuntimeError(
                        "Variant-balanced PPO requires task_variants in rollout state"
                    )
                task_variants = task_variants.to(device=device, dtype=torch.long)
            if self.sync_advantage_normalization:
                advantages = self.accelerator.gather(advantages)
                task_stages = self.accelerator.gather(task_stages)
                if self.balance_staged_task_variants:
                    task_variants = self.accelerator.gather(task_variants)

            normalized = torch.zeros_like(advantages)
            for stage_idx in range(int(self.env.num_stages)):
                stage_mask = task_stages == stage_idx
                variants = (
                    torch.unique(task_variants[stage_mask])
                    if self.balance_staged_task_variants
                    else torch.zeros(1, dtype=torch.long, device=device)
                )
                for variant in variants:
                    group_mask = stage_mask
                    if self.balance_staged_task_variants:
                        group_mask = group_mask & (task_variants == variant)
                    group_values = advantages[group_mask]
                    if group_values.numel() == 0:
                        continue
                    group_mean = group_values.mean()
                    group_std = group_values.std(unbiased=False)
                    normalized[group_mask] = (group_values - group_mean) / (
                        group_std + 1.0e-8
                    )
            advantages = normalized
            if self.sync_advantage_normalization:
                advantages = advantages.reshape(
                    self.accelerator.num_processes, -1, *advantages.shape[1:]
                )[self.accelerator.process_index].to(device)
        elif self.sync_advantage_normalization:
            # gather advantages from all processes before normalization
            advantages = self.accelerator.gather(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # ungather advantages
            advantages = advantages.reshape(
                self.accelerator.num_processes, -1, *advantages.shape[1:]
            )[self.accelerator.process_index].to(device)
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _adjust_learning_rate_based_on_kl(self, kl_mean, optimizer):
        """Adjust the learning rate based on the KL divergence.

        This function adjusts the learning rate once per PPO update from the aggregate
        KL divergence between the current policy and the rollout policy.
        If the KL divergence is too high, the learning rate is decreased.
        If the KL divergence is too low, the learning rate is increased.

        Args:
            kl_mean (float): The mean KL divergence across all processes.
            optimizer (torch.optim.Optimizer): The optimizer to update.
        """
        if not self.use_adaptive_kl_schedule:
            return

        if kl_mean > self.desired_kl * 2.0:
            new_lr = max(
                self.adaptive_kl_min_learning_rate,
                self.adaptive_learning_rate / 1.5,
            )
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            new_lr = min(
                self.adaptive_kl_max_learning_rate,
                self.adaptive_learning_rate * 1.5,
            )
        else:
            new_lr = self.adaptive_learning_rate
        self.adaptive_learning_rate = new_lr
        self.args.learning_rate = new_lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.args.learning_rate

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint to restore the state of model, optimizer, trainer etc.

        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.accelerator.device, weights_only=False
        )

        # Load model state
        model = self.accelerator.unwrap_model(self.model)
        if "actor_model_state_dict" in checkpoint:
            model.policy.load_state_dict(checkpoint["actor_model_state_dict"])
        elif "policy_state_dict" in checkpoint:
            model.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "value_state_dict" in checkpoint and model.value_model is not None:
            model.value_model.load_state_dict(checkpoint["value_state_dict"])

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Update learning rate if available
            if "args" in checkpoint and hasattr(checkpoint["args"], "learning_rate"):
                loaded_learning_rate = float(checkpoint["args"].learning_rate)
                if self.use_adaptive_kl_schedule:
                    loaded_learning_rate = min(
                        self.adaptive_kl_max_learning_rate,
                        max(
                            self.adaptive_kl_min_learning_rate,
                            loaded_learning_rate,
                        ),
                    )
                    self.adaptive_learning_rate = loaded_learning_rate
                self.args.learning_rate = loaded_learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.args.learning_rate

        resume_noise_std = self.config.get("resume_noise_std", None)
        if resume_noise_std is not None:
            policy = model.policy
            target_std = torch.as_tensor(
                resume_noise_std,
                dtype=policy.std.dtype,
                device=policy.std.device,
            )
            if target_std.numel() not in (1, policy.std.numel()):
                raise ValueError(
                    "resume_noise_std must be a scalar or contain one value per action; "
                    f"got {target_std.numel()} values for {policy.std.numel()} actions"
                )
            with torch.no_grad():
                policy.std.copy_(target_std.expand_as(policy.std))
            self.optimizer.state.pop(policy.std, None)
            print(f"Reset resumed action noise std to {policy.std.detach().cpu().tolist()}")

        # Load learning rate scheduler state
        if (
            "lr_scheduler_state_dict" in checkpoint
            and checkpoint["lr_scheduler_state_dict"] is not None
        ):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        if "env_state_dict" in checkpoint:
            env_state_dict = ModelSaveCallback.select_rank_env_state_dict(
                checkpoint["env_state_dict"],
                self.accelerator.process_index,
                self.accelerator.num_processes,
            )
            if env_state_dict is not None:
                self.env.load_env_state_dict(env_state_dict)

        if "state" in checkpoint:
            for key, value in checkpoint["state"].__dict__.items():
                # These buffers are sized for the rollout environment count.
                # Keep the freshly initialized tensors when resuming with a
                # different number of environments.
                if key in ["cur_reward_sum", "cur_episode_length"]:
                    continue
                if key not in [
                    "stateful_callbacks",
                    "is_local_process_zero",
                    "is_world_process_zero",
                    "log_history",
                ]:
                    setattr(self.state, key, value)

        print(f"Loaded checkpoint from step {self.state.global_step}")
        return checkpoint

    def eval(self, save_dirpath: Optional[str] = None):
        if self.config.eval.save_videos or self.config.eval.save_trajectories:
            assert (
                save_dirpath is not None
            ), "save_dirpath must be provided if save_videos or save_trajectories is True"

        if self.config.eval.save_videos:
            if (
                self.env.config.obs.obs_dict.vision_obs[0] == "depth_image"
                and not self.env.config.simulator.config.cameras.stack_depth_image
            ):
                num_channels = 1
            else:
                num_channels = 3
            self.camera_resolution = self.env.config.simulator.config.cameras.camera_resolutions + [
                num_channels
            ]

        self._eval_mode()
        self.policy_model.eval_mode()
        self.policy_model.init_rollout()
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.accelerator.device)

        # Initialize environment-based metrics tracking
        self.env.init_eval_metrics_tracking(self.accelerator.device)

        # Reuse the cur_reward_sum, cur_episode_length (will be reset at the end in case `train()` is called after `eval(_teacher)`)
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.accelerator.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.int32, device=self.accelerator.device
        )

        # Keep old buffer attributes for backward compatibility with terminate_rollout()
        self.eval_rewbuffer = []
        self.eval_lenbuffer = []
        self.eval_goal_reached_buffer = []

        # Get evaluation mode configuration
        eval_num_envs_episodes = self.config.eval.get(
            "eval_num_envs_episodes", False
        )  # Default to one episode per env
        # Determine maximum number of episodes to collect
        if eval_num_envs_episodes:
            max_episodes = self.env.num_envs  # One episode per environment
        elif self.config.eval.save_trajectories:
            max_episodes = self.config.eval.num_save_episodes
        else:
            max_episodes = self.config.eval.get(
                "num_eval_episodes", self.env.num_envs
            )  # Use config or fallback to num_envs

        # Track which environments have completed their first episode (only used when eval_num_envs_episodes=True)
        if eval_num_envs_episodes:
            self.env_episode_completed = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.accelerator.device
            )

        if self.config.eval.save_videos:
            self.saved_episode_cnt = 0
            self.batch_frames = [[] for _ in range(self.env.num_envs)]

        if self.config.eval.save_trajectories:
            self.actor_obs_to_save = [[] for _ in range(self.env.num_envs)]
            self.student_obs_to_save = [[] for _ in range(self.env.num_envs)]
            self.actions_to_save = [[] for _ in range(self.env.num_envs)]
            self.reward_to_save = [[] for _ in range(self.env.num_envs)]
            self.done_to_save = [[] for _ in range(self.env.num_envs)]

        def terminate_rollout():
            if eval_num_envs_episodes:
                # Stop when all environments have completed their first episode
                return torch.all(self.env_episode_completed).item()
            else:
                # Original behavior: stop when we have collected max_episodes episodes total
                if self.config.eval.save_trajectories:
                    return self.saved_episode_cnt >= max_episodes
                return len(self.eval_goal_reached_buffer) >= max_episodes

        start_time = time.time()
        expected_total_len = max_episodes
        pbar = tqdm(total=expected_total_len)
        dones = torch.zeros(self.env.num_envs, device=self.accelerator.device)
        with torch.no_grad():
            with unwrap_model_for_generation(
                self.model,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as model:
                while not terminate_rollout():
                    action_state = self.policy_step(model.policy, obs_dict, cur_dones=dones)

                    last_obs_dict = deepcopy(obs_dict)
                    action_state["actions"] = action_state["action_mean"]
                    obs_dict, rewards, dones, infos = self.env.step(action_state)

                    if self.config.eval.save_videos:
                        self.batch_write_frame(last_obs_dict["vision_obs"])

                    if self.config.eval.save_trajectories:
                        self.batch_write_low_dim_obs(
                            actor_obs=last_obs_dict["actor_obs"],
                            student_obs=last_obs_dict["student_obs"],
                            actions=action_state["actions"],
                            rewards=rewards,
                            dones=dones,
                        )

                    for obs_key in obs_dict.keys():
                        obs_dict[obs_key] = obs_dict[obs_key].to(self.accelerator.device)

                    rewards, dones = rewards.to(self.accelerator.device), dones.to(
                        self.accelerator.device
                    )
                    model.policy.reset(dones)

                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1

                    # Update environment metrics per step
                    self.env.update_eval_metrics_per_step(infos)

                    new_ids = (dones > 0).nonzero(as_tuple=False)

                    if len(new_ids) > 0:
                        if eval_num_envs_episodes:
                            # Only collect data from environments that haven't completed their first episode yet
                            valid_new_ids = new_ids[~self.env_episode_completed[new_ids][:, 0]]
                        else:
                            collected_count = (
                                self.saved_episode_cnt
                                if self.config.eval.save_trajectories
                                else len(self.eval_goal_reached_buffer)
                            )
                            valid_new_ids = take_remaining_episode_ids(
                                new_ids, collected_count, max_episodes
                            )

                        if len(valid_new_ids) > 0:
                            if self.config.eval.save_videos:
                                self.batch_reset_data_writer(valid_new_ids, save_dirpath)

                            # Process episode completions using environment callback
                            self.env.process_eval_episode_completions(
                                valid_new_ids, self.cur_reward_sum, self.cur_episode_length
                            )

                            # Also populate old buffer attributes for backward compatibility with terminate_rollout()
                            self.eval_rewbuffer.extend(
                                self.cur_reward_sum[valid_new_ids][:, 0].cpu().numpy().tolist()
                            )
                            self.eval_lenbuffer.extend(
                                self.cur_episode_length[valid_new_ids][:, 0].cpu().numpy().tolist()
                            )
                            self.eval_goal_reached_buffer.extend(
                                self.env.eval_metrics["episode_goal_reached"][valid_new_ids][:, 0]
                                .cpu()
                                .numpy()
                                .tolist()
                            )

                            if eval_num_envs_episodes:
                                # Mark these environments as having completed their first episode
                                self.env_episode_completed[valid_new_ids] = True
                                # Update progress bar with newly completed episodes
                                pbar.update(len(valid_new_ids))
                            else:
                                # Original behavior: update progress bar with collected episodes
                                if self.config.eval.save_trajectories:
                                    pbar.update(self.saved_episode_cnt - pbar.n)
                                else:
                                    pbar.update(
                                        len(self.env.eval_metrics["goal_reached_buffer"]) - pbar.n
                                    )

                        # Reset all environments that finished (regardless of whether we collected data)
                        self.cur_reward_sum[new_ids] = 0
                        self.cur_episode_length[new_ids] = 0

                        # Reset environment episode tracking
                        self.env.reset_eval_episode_tracking(new_ids)

                    # TODO: 0: make sure this is compatiable with all use cases
                    self.callback_handler.on_step_end(self.args, self.state, self.control)

                torch.cuda.empty_cache()
                gc.collect()

        pbar.close()
        end_time = time.time()

        # Get evaluation summary from environment (includes class-wise metrics)
        eval_dict = self.env.get_eval_metrics_summary()
        eval_dict["eval_time"] = end_time - start_time

        print(f"Eval time: {end_time - start_time} seconds")
        if self.config.eval.save_trajectories:
            print("-" * 30)
            print(f"Saved {self.saved_episode_cnt} episodes")
        if self.config.eval.save_videos:
            print(f"Please check {save_dirpath}")

        # Reset env (in the outer loop because we need the obs_dict), cur_reward_sum, cur_episode_length
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.accelerator.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.int32, device=self.accelerator.device
        )

        return eval_dict

    def batch_write_frame(self, vision_obs):
        """
        vision_obs: [B, H*W*C], [0, 1/255], float32 (The range is a bug)
        This function is for eval time, save all envs' frames,
            as opposed to write_frame, which is for training time, only save the frames for self.visualize_env_idx
        """
        flattened_rgb_images = vision_obs.clone()

        batch_size = flattened_rgb_images.shape[0]

        rgb_images = flattened_rgb_images.reshape(
            batch_size, *self.camera_resolution
        )  # [B, H, W, C]

        # To uint8
        rgb_images = (rgb_images * 255.0).to(torch.uint8)  # [B, H, W, C]

        # Append to frames list
        for i in range(self.env.num_envs):
            self.batch_frames[i].append(rgb_images[i])  # [H, W, C]

    def batch_write_low_dim_obs(
        self,
        actor_obs,
        student_obs,
        actions,
        rewards,
        dones,
    ):
        """
        actor_obs: [B, dim_actor_obs]
        student_obs: [B, dim_student_obs]
        actions: [B, dim_actions]
        rewards: [B]
        dones: [B]
        """
        for i in range(self.env.num_envs):
            self.actor_obs_to_save[i].append(actor_obs[i].cpu().numpy())
            self.student_obs_to_save[i].append(student_obs[i].cpu().numpy())
            self.actions_to_save[i].append(actions[i].cpu().numpy())
            self.reward_to_save[i].append(rewards[i].item())
            # Convert to bool
            self.done_to_save[i].append(bool(dones[i].item()))

    def batch_reset_data_writer(self, env_indices, save_dirpath):
        """
        :param env_indices: list of env indices that have finished the episode
        """
        for env_idx in env_indices:
            do_not_save = False

            cur_reward_sum = self.cur_reward_sum[env_idx].item()
            # Get goal_reached status from environment metrics
            if (
                hasattr(self.env, "eval_metrics")
                and "episode_goal_reached" in self.env.eval_metrics
            ):
                goal_reached = int(self.env.eval_metrics["episode_goal_reached"][env_idx].item())
            else:
                goal_reached = 1  # Default to saving if we can't determine status

            if self.config.eval.save_goal_reached_only:
                if goal_reached == 0:
                    do_not_save = True

            if np.random.rand() > self.config.eval.video_save_prob:
                do_not_save = True

            if do_not_save:
                # Simply empty the buffers
                self.batch_frames[env_idx] = []
                if self.config.eval.save_trajectories:
                    self.actor_obs_to_save[env_idx] = []
                    self.student_obs_to_save[env_idx] = []
                    self.actions_to_save[env_idx] = []
                    self.reward_to_save[env_idx] = []
                    self.done_to_save[env_idx] = []
                continue

            # Video saving
            video_save_dirpath = Path(save_dirpath) / "videos"

            save_filename = f"eps{self.saved_episode_cnt}_env{env_idx.item()}_len{len(self.batch_frames[env_idx])}_reward{cur_reward_sum:.2f}_goal{int(goal_reached)}_rank{self.accelerator.process_index}"
            video_path = video_save_dirpath / f"{save_filename}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)

            video_tensor = torch.stack(self.batch_frames[env_idx][1:])  # [T, H, W, C]

            fps = 20
            torchvision.io.write_video(str(video_path), video_tensor, fps=fps, video_codec="h264")

            # print(f"Video saved to {video_path}")

            self.batch_frames[env_idx] = []
            self.saved_episode_cnt += 1

            # Low dim data saving
            if self.config.eval.save_trajectories:
                low_dim_data_save_dirpath = Path(save_dirpath) / "data"
                low_dim_data_save_dirpath.mkdir(parents=True, exist_ok=True)

                save_data = {
                    "observation.state": self.student_obs_to_save[env_idx][1:],
                    "actor_obs": self.actor_obs_to_save[env_idx][1:],
                    "action": self.actions_to_save[env_idx][1:],
                    "reward": self.reward_to_save[env_idx][1:],
                    "done": self.done_to_save[env_idx][1:],
                }

                episode_length = len(video_tensor)
                for key, value in save_data.items():
                    assert len(value) == episode_length, f"{key}: {len(value)} != {episode_length}"

                save_data["timestamp"] = np.arange(episode_length) / fps

                data_save_path = low_dim_data_save_dirpath / f"{save_filename}.parquet"

                df = pd.DataFrame(save_data)
                df.to_parquet(data_save_path)

                # Empty the buffers
                self.actor_obs_to_save[env_idx] = []
                self.student_obs_to_save[env_idx] = []
                self.actions_to_save[env_idx] = []
                self.reward_to_save[env_idx] = []
                self.done_to_save[env_idx] = []

            if self.saved_episode_cnt >= self.config.eval.num_save_episodes:
                break

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(self.env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    @property
    def inference_model(self):
        return {"actor": self.model.policy, "critic": self.model.value_model}
