# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import gc
import os
import time
from copy import deepcopy
from typing import Optional

import onnxruntime as ort
import torch
import torch.nn as nn
from loguru import logger
from rich.console import Console

# Use TrainerState from transformers (compatible across versions)
from trl.trainer.ppo_trainer import (
    disable_dropout_in_model,
    unwrap_model_for_generation,
)

from gr00t.rl.trl.modules.homie_modules import (
    HIMActorCritic,
    HomieActorModule,
    init_actor_critic_dict,
)
from gr00t.rl.trl.trainer.distill_trainer_obj_pred import (
    TRLDistillTrainerObjPred,
)
from gr00t.rl.trl.utils.rl import compute_episode_attnmask

console = Console()


class PolicyAndValueWrapperObjPredHomie(nn.Module):
    """
    Combined PolicyAndValueWrapper that supports both object prediction and Homie API.
    Merges functionality from both PolicyAndValueWrapperObjPred and the Homie API version.
    """

    def __init__(
        self, policy, value_model, homie_walk_model, homie_stand_model, ref_model=None
    ) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.homie_walk_model = homie_walk_model
        self.homie_stand_model = homie_stand_model
        self.ref_model = ref_model
        self.opt_homie = False
        self.homie_switch_threshold = 0.1

    def set_mode(self, mode):
        if hasattr(self.policy, "mode"):
            self.policy.mode = mode
        if hasattr(self.homie_walk_model, "train") and hasattr(self.homie_walk_model, "eval"):
            if self.opt_homie and mode == "train":
                self.homie_walk_model.train()
            else:
                self.homie_walk_model.eval()
        if hasattr(self.homie_stand_model, "train") and hasattr(self.homie_stand_model, "eval"):
            if self.opt_homie and mode == "train":
                self.homie_stand_model.train()
            else:
                self.homie_stand_model.eval()

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
            # Standard PPO policy mode with Homie integration
            self.policy.act(**kwargs)
            homie_obs = kwargs["obs_dict"]["homie_obs"]

            # If recurrent policy, unsplit observations for homie models
            if (
                hasattr(self.policy, "is_recurrent")
                and self.policy.is_recurrent
                and "masks" in kwargs
                and "original_dones" in kwargs
            ):
                from gr00t.rl.trl.utils.rl import unsplit_trajectories

                masks = kwargs["masks"]
                original_dones = kwargs["original_dones"]
                homie_obs = unsplit_trajectories(homie_obs, masks, original_dones)
            walk_out = self.homie_walk_model(homie_obs)
            stand_out = self.homie_stand_model(homie_obs)
            homie_one_step_obs = init_actor_critic_dict[
                "num_one_step_obs"
            ]  # default value from homie modules
            commands = homie_obs[..., -homie_one_step_obs : -(homie_one_step_obs - 3)]
            walk_mask = torch.norm(commands, dim=-1, keepdim=True) > self.homie_switch_threshold

            def _sel(a_walk, a_stand):
                m = walk_mask
                while m.dim() < a_walk.dim():
                    m = m.unsqueeze(-1)
                while m.dim() > a_walk.dim():
                    m = m.squeeze(-1)
                return torch.where(m, a_walk, a_stand)

            homie_actions = _sel(walk_out["actions"], stand_out["actions"])
            homie_mean = _sel(walk_out["action_mean"], stand_out["action_mean"])
            homie_sigma = _sel(walk_out["action_sigma"], stand_out["action_sigma"])
            homie_entropy = _sel(walk_out["entropy"], stand_out["entropy"])

            if actions is not None:
                policy_log_probs = self.policy.get_actions_log_prob(
                    actions=actions[..., : self.policy.num_actions]
                )
            else:
                # If no actions provided, use zero log probs
                policy_log_probs = (
                    torch.zeros_like(homie_actions.squeeze(-1))
                    if homie_actions.dim() > 1
                    else torch.zeros_like(homie_actions)
                )
            walk_lp = self.homie_walk_model.get_actions_log_prob(actions=homie_actions)
            stand_lp = self.homie_stand_model.get_actions_log_prob(actions=homie_actions)
            homie_log_probs = torch.where(walk_mask.squeeze(-1), walk_lp, stand_lp)
            if getattr(self, "opt_homie", True):
                logprobs = policy_log_probs + homie_log_probs
                entropy = self.policy.entropy + homie_entropy
            else:
                logprobs = policy_log_probs
                entropy = self.policy.entropy
            results = {
                "logprobs": logprobs,
                "action_mean": torch.cat([self.policy.action_mean, homie_mean], dim=-1),
                "action_std": torch.cat([self.policy.action_std, homie_sigma], dim=-1),
                "entropy": entropy,
            }
        elif mode == "policy_distill_obj_pred":
            # Distillation mode with object prediction - use the new act_with_obj_pred method
            if hasattr(self.policy, "action_mean_with_obj_pred"):
                return self.policy.action_mean_with_obj_pred(**kwargs)
            else:
                # Fallback to regular act method if act_with_obj_pred is not available
                return self.policy.act(**kwargs)
        elif mode == "policy_distill":
            results = self.policy.act(**kwargs)
        elif mode == "policy_distill_ppo":
            policy_state_dict = self.policy.act(**kwargs)
            log_probs = self.policy.get_actions_log_prob(actions=actions)
            results = {
                "actions": policy_state_dict["actions"],
                "logprobs": log_probs,
                "action_mean": policy_state_dict["action_mean"],
                "action_std": policy_state_dict["action_sigma"],
                "entropy": self.policy.entropy,
            }
            if "normalized_actions" in policy_state_dict:
                results["normalized_actions"] = policy_state_dict["normalized_actions"]

        elif mode == "value":
            results = self.value_model.evaluate(**kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return results


def load_onnx_policy(path, device):
    model = ort.InferenceSession(path)

    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device=device)

    return run_inference


class TRLDistillTrainerObjPredHomieAPI(TRLDistillTrainerObjPred):
    """
    Combined Distillation trainer with object position prediction capability and Homie API integration.
    Inherits from TRLDistillTrainerObjPred and adds Homie locomotion integration.
    """

    _tag_names = ["trl", "humanoid_distill_obj_pred_homie_api"]

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
        **kwargs,
    ):
        # Call parent initialization first to get distillation + object prediction setup
        super()._init_trl(
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
            **kwargs,
        )

        # Now add Homie integration
        # Load homie walk and stand models
        device = (
            self.accelerator.device
            if self.accelerator
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        homie_walk_state_dict = torch.load(self.config.homie_walk_model_path, map_location=device)
        homie_walk_model = HIMActorCritic(**init_actor_critic_dict)
        homie_walk_model.load_state_dict(homie_walk_state_dict["model_state_dict"])
        self.homie_walk_model = HomieActorModule(homie_walk_model).to(device)

        homie_stand_state_dict = torch.load(self.config.homie_stand_model_path, map_location=device)
        homie_stand_model = HIMActorCritic(**init_actor_critic_dict)
        homie_stand_model.load_state_dict(homie_stand_state_dict["model_state_dict"])
        self.homie_stand_model = HomieActorModule(homie_stand_model).to(device)

        # Disable dropout in homie models
        if self.config.get("disable_dropout", True):
            for module in [self.homie_walk_model, self.homie_stand_model]:
                if module is not None:
                    disable_dropout_in_model(module)

        # Set up homie model training/freezing
        if not self.config.get("opt_homie", False):
            print("Freezing homie model parameters")
            for p in self.homie_walk_model.parameters():
                p.requires_grad = False
            for p in self.homie_stand_model.parameters():
                p.requires_grad = False
            self.homie_walk_model.eval()
            disable_dropout_in_model(self.homie_walk_model)
            self.homie_stand_model.eval()
            disable_dropout_in_model(self.homie_stand_model)

        # Override the model with our combined wrapper that supports both object prediction and Homie
        # Note: we must re-prepare the model to wrap it in DDP for multi-GPU training
        # The parent already prepared the old wrapper, so we need to prepare this new one
        self.model = PolicyAndValueWrapperObjPredHomie(
            self.policy_model,
            self.value_model,
            self.homie_walk_model,
            self.homie_stand_model,
            self.ref_model,
        )
        self.model.opt_homie = self.config.get("opt_homie", False)
        self.model.homie_switch_threshold = self.config.get("homie_switch_threshold", 0.5)

        # Re-prepare only the model (optimizer already references the correct parameters since
        # policy_model and value_model are the same, and homie models are frozen)
        # CRITICAL: Use find_unused_parameters=True because homie models are frozen
        # CRITICAL: Use static_graph=True to avoid "marked ready twice" errors
        if self.accelerator.num_processes > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model = DDP(
                self.model.to(self.accelerator.device),
                device_ids=[self.accelerator.local_process_index],
                output_device=self.accelerator.local_process_index,
                find_unused_parameters=True,  # Required because homie models are frozen and won't receive gradients
                static_graph=True,  # Graph structure doesn't change between iterations
            )
        else:
            self.model = self.model.to(self.accelerator.device)

        # Store unwrapped model reference for accessing attributes like homie_walk_model
        self.unwrapped_model = self.accelerator.unwrap_model(self.model)

        self.export_homie_model = False

    def _init_config(self):
        # Env related Config
        super()._init_config()
        self.num_act = self.policy_model.num_actions + self.homie_walk_model.num_actions

    def policy_step(self, policy_model, obs_dict, cur_dones=None):
        policy_state_dict = {}

        homie_obs = obs_dict["homie_obs"]
        # Use unwrapped_model to access homie models (model might be wrapped in DDP)
        walk_out = self.unwrapped_model.homie_walk_model(homie_obs)
        stand_out = self.unwrapped_model.homie_stand_model(homie_obs)
        homie_one_step_obs = init_actor_critic_dict[
            "num_one_step_obs"
        ]  # Default value from init_actor_critic_dict
        commands = obs_dict["homie_obs"][..., -homie_one_step_obs : -(homie_one_step_obs - 3)]
        walk_mask = (
            torch.norm(commands, dim=-1, keepdim=True) > self.unwrapped_model.homie_switch_threshold
        )  # [B,1]

        def _sel(a_walk, a_stand):
            m = walk_mask
            while m.dim() < a_walk.dim():
                m = m.unsqueeze(-1)
            return torch.where(m, a_walk, a_stand)

        homie_actions = _sel(walk_out["actions"], stand_out["actions"])
        homie_mean = _sel(walk_out["action_mean"], stand_out["action_mean"])
        homie_sigma = _sel(walk_out["action_sigma"], stand_out["action_sigma"])
        homie_entropy = _sel(walk_out["entropy"], stand_out["entropy"])

        if "teacher_obs" in obs_dict:
            if self.ref_model is not None:
                if hasattr(self.ref_model, "act_inference"):
                    teacher_actions = self.ref_model.act_inference(
                        obs_dict=deepcopy(obs_dict), input_key="teacher_obs"
                    )
                elif hasattr(self.ref_model, "act"):
                    teacher_actions = self.ref_model.act(obs_dict=deepcopy(obs_dict))["actions"]
                else:
                    teacher_actions = None

                if teacher_actions is not None:
                    # Concatenate teacher actions with homie actions for ground truth (reuse computed homie_actions)
                    gt_actions = torch.cat([teacher_actions, homie_actions], dim=-1)

                    if self.env.config.get("zero_vel", False) and hasattr(self.env, "stage_buf"):
                        zero_env_ids = (
                            ((self.env.stage_buf >= 1) & (self.env.stage_buf < 4))
                            .nonzero(as_tuple=False)
                            .squeeze(-1)
                        )
                        if len(zero_env_ids) > 0:
                            # gt_actions[zero_env_ids, :3] = 0.0
                            gt_actions[zero_env_ids, :3] = (
                                self.env._delta_actions[zero_env_ids, :3] * -10.0
                            )

                    if self.env.config.get("zero_finger", False) and hasattr(self.env, "stage_buf"):
                        zero_env_ids = (
                            ((self.env.stage_buf == 0) | (self.env.stage_buf == 1))
                            .nonzero(as_tuple=False)
                            .squeeze(-1)
                        )
                        if len(zero_env_ids) > 0:
                            gt_actions[zero_env_ids, 10] = (
                                self.env._delta_actions[zero_env_ids, 10] - 0.5
                            ) * -10.0

                    policy_state_dict["gt_actions"] = gt_actions
            else:
                pass

            gt_obj_pos_full = obs_dict["gt_obj_pos"]
            policy_state_dict["gt_obj_pos"] = gt_obj_pos_full

            obs_dict = deepcopy(obs_dict)

        actor_obs_dict = deepcopy(obs_dict)

        if cur_dones is None:
            device = (
                self.accelerator.device
                if self.accelerator
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            dones = (
                self.storage.query_key("dones")
                .to(device)[: self.storage.step + 1]
                .squeeze(-1)
                .transpose(0, 1)
            )
            episode_attnmask = compute_episode_attnmask(dones)
        else:
            episode_attnmask = None

        student_state_dict = policy_model.rollout_with_obj_pred(
            obs_dict=actor_obs_dict, episode_attnmask=episode_attnmask, cur_dones=cur_dones
        )

        policy_actions_log_prob = policy_model.get_actions_log_prob(
            actions=student_state_dict["action_mean"]
        )
        walk_lp = self.unwrapped_model.homie_walk_model.get_actions_log_prob(actions=homie_actions)
        stand_lp = self.unwrapped_model.homie_stand_model.get_actions_log_prob(
            actions=homie_actions
        )
        homie_actions_log_prob = torch.where(walk_mask.squeeze(-1), walk_lp, stand_lp)

        if self.unwrapped_model.opt_homie:
            actions_log_prob = (policy_actions_log_prob + homie_actions_log_prob).unsqueeze(1)
        else:
            actions_log_prob = policy_actions_log_prob.unsqueeze(1)

        # Zi: use action_mean for distillation rollout (we expect no std)
        # actions      = torch.cat([student_state_dict['actions'],     homie_actions], dim=-1)
        actions = torch.cat([student_state_dict["action_mean"], homie_actions], dim=-1)
        action_mean = torch.cat([student_state_dict["action_mean"], homie_mean], dim=-1)
        action_sigma = torch.cat([student_state_dict["action_sigma"], homie_sigma], dim=-1)
        # TODO (Zi): check homie_actions == homie_mean (yes!)
        # print("homie_actions", homie_actions)
        # print("homie_mean", homie_mean)

        student_state_dict.update(
            {
                "actions": actions,
                "action_mean": action_mean,
                "action_sigma": action_sigma,
                "actions_log_prob": actions_log_prob,
            }
        )
        policy_state_dict.update(student_state_dict)

        if self.export_homie_model:
            torch.onnx.export(
                self.unwrapped_model.homie_walk_model,
                (homie_obs, True),
                "homie_walk_model.onnx",
                input_names=["actor_obs"],
                output_names=["action"],
            )
            torch.onnx.export(
                self.unwrapped_model.homie_stand_model,
                (homie_obs, True),
                "homie_stand_model.onnx",
                input_names=["actor_obs"],
                output_names=["action"],
            )
            self.export_homie_model = False

        # if not in eval mode, rollout with teacher actions initially
        if self.unwrapped_model.training:
            global_step = self.state.global_step
            rollout_with_teacher_num_steps = self.config.get("rollout_with_teacher_num_steps", 0)
            if global_step < rollout_with_teacher_num_steps:
                logger.info(
                    "global_step / rollout_with_teacher_num_steps: %d / %d",
                    global_step,
                    rollout_with_teacher_num_steps,
                )
                logger.info("rollout with teacher")
                policy_state_dict["actions"][:] = policy_state_dict["gt_actions"]
                policy_state_dict["action_mean"][:] = policy_state_dict["gt_action"]

        if self.config.get("enforce_teacher_rollout", False):
            # logger.info("rollout with teacher")
            # import ipdb; ipdb.set_trace()
            # logger.info(f"gt_actions: {policy_state_dict['gt_actions']}")

            if self.config.get("ratio_teacher_rollout", 1.0) < 1.0:

                env_nums = policy_state_dict["gt_actions"].shape[0]
                env_ids_use_teacher_rollout = int(
                    env_nums * self.config.get("ratio_teacher_rollout", 1.0)
                )
                # logger.info(f"env_nums: {env_nums}")
                # logger.info(f"env_nums_use_teacher_rollout: {env_ids_use_teacher_rollout}")
                policy_state_dict["actions"][:env_ids_use_teacher_rollout, :] = policy_state_dict[
                    "gt_actions"
                ][:env_ids_use_teacher_rollout, :]
                policy_state_dict["action_mean"][:env_ids_use_teacher_rollout, :] = (
                    policy_state_dict["gt_actions"][:env_ids_use_teacher_rollout, :]
                )
            else:
                # import ipdb; ipdb.set_trace()
                policy_state_dict["actions"][:] = policy_state_dict["gt_actions"]
                policy_state_dict["action_mean"][:] = policy_state_dict["gt_actions"]

            # policy_state_dict["actions"][:] = policy_state_dict["gt_actions"]
            # policy_state_dict["action_mean"][:] = policy_state_dict["gt_actions"]

        return policy_state_dict

    def _compute_dagger_bc_loss(self, forward_results, mb_rollout_data):
        policy_results = forward_results["policy_results"]
        mb_gt_actions = mb_rollout_data["mb_gt_actions"]
        mb_normalized_gt_actions = mb_rollout_data["mb_normalized_gt_actions"]

        # clip the gt_actions
        # if self.env._clip_homie_command:
        #     mb_gt_actions[..., :3] = torch.clamp(mb_gt_actions[..., :3], self.env._homie_command_low_thres[:, :3], self.env._homie_command_high_thres[:, :3])

        dagger_bc_loss = self.bc_loss_fn(
            policy_results["action_mean"][..., : self.policy_model.num_actions],
            mb_gt_actions[..., : self.policy_model.num_actions],
        )
        return dict(
            dagger_bc_loss=dagger_bc_loss,
        )

    def _wrap_model(self, model, ref_model, value_model):
        """Override to use combined wrapper that supports both object prediction and Homie."""
        # This method is used by parent class during initialization
        # We override the model directly in _init_trl instead, so return parent's wrapper for now
        # The actual combined wrapper is set in _init_trl
        return super()._wrap_model(model, ref_model, value_model)

    def _get_train_metrics(self):
        # Get metrics from parent (includes object prediction metrics)
        metrics = {}

        metrics["loss/dagger_bc_loss"] = (
            self.accelerator.gather_for_metrics(self.dagger_bc_loss_stats).mean().item()
            if torch.any(self.dagger_bc_loss_stats != 0)
            else 0.0
        )
        metrics["loss/weighted_dagger_bc_loss"] = (
            self.accelerator.gather_for_metrics(self.weighted_dagger_bc_loss_stats).mean().item()
            if torch.any(self.weighted_dagger_bc_loss_stats != 0)
            else 0.0
        )

        metrics["loss/obj_pred_loss"] = (
            self.accelerator.gather_for_metrics(self.obj_pred_loss_stats).mean().item()
            if torch.any(self.obj_pred_loss_stats != 0)
            else 0.0
        )
        metrics["loss/weighted_obj_pred_loss"] = (
            self.accelerator.gather_for_metrics(self.weighted_obj_pred_loss_stats).mean().item()
            if torch.any(self.weighted_obj_pred_loss_stats != 0)
            else 0.0
        )

        # Add learning rate to metrics
        if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]

        # Add Homie-specific metrics if needed
        # (Currently no additional Homie metrics, but can be extended)

        torch.cuda.empty_cache()
        gc.collect()
        return metrics

    def eval(self, save_dirpath: Optional[str] = None):
        """
        Evaluation method with object position prediction visualization and Homie integration.
        """
        original_save_trajectories = self.config.eval.save_trajectories
        self.config.eval.save_trajectories = False

        if self.config.eval.save_videos:
            assert save_dirpath is not None, "save_dirpath must be provided if save_videos is True"

        if self.config.eval.save_videos:
            if self.env.config.obs.obs_dict.vision_obs[0] == "depth_image":
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
        device = (
            self.accelerator.device
            if hasattr(self, "accelerator") and self.accelerator
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(device)

        # Initialize environment-based metrics tracking
        device = (
            self.accelerator.device
            if hasattr(self, "accelerator") and self.accelerator
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.env.init_eval_metrics_tracking(device)

        # Reuse the cur_reward_sum, cur_episode_length (will be reset at the end in case `train()` is called after `eval(_teacher)`)
        device = (
            self.accelerator.device
            if self.accelerator
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.int32, device=device)

        # Get evaluation mode configuration
        eval_num_envs_episodes = self.config.eval.get(
            "eval_num_envs_episodes", False
        )  # Default to one episode per env
        # Determine maximum number of episodes to collect
        if eval_num_envs_episodes:
            max_episodes = self.env.num_envs  # One episode per environment
        else:
            max_episodes = self.config.eval.get(
                "num_eval_episodes", self.env.num_envs
            )  # Use config or fallback to num_envs

        # Track which environments have completed their first episode (only used when eval_num_envs_episodes=True)
        if eval_num_envs_episodes:
            device = (
                self.accelerator.device
                if self.accelerator
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.env_episode_completed = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=device
            )

        if self.config.eval.save_videos:
            self.saved_episode_cnt = 0
            self.batch_frames = [[] for _ in range(self.env.num_envs)]

        # Data saving is disabled - no trajectory data structures initialized

        def terminate_rollout():
            if eval_num_envs_episodes:
                # Stop when all environments have completed their first episode
                return torch.all(self.env_episode_completed).item()
            else:
                # Original behavior: stop when we have collected max_episodes episodes total
                return len(self.env.eval_metrics["goal_reached_buffer"]) >= max_episodes

        start_time = time.time()
        expected_total_len = max_episodes

        from tqdm import tqdm

        pbar = tqdm(total=expected_total_len)
        device = (
            self.accelerator.device
            if self.accelerator
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        dones = torch.zeros(self.env.num_envs, device=device)
        step_count = 0

        with torch.no_grad():
            if hasattr(self, "accelerator") and self.accelerator:
                context_manager = unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=getattr(self.args, "ds3_gather_for_generation", False),
                )
            else:
                # Simple context manager that just returns the model
                from contextlib import contextmanager

                @contextmanager
                def simple_context():
                    yield self.model

                context_manager = simple_context()

            with context_manager as model:
                while not terminate_rollout():
                    action_state = self.policy_step(model.policy, obs_dict, cur_dones=dones)
                    # Initialize variables for object position data
                    gt_obj_pos = None
                    pred_obj_pos = None

                    # If we have object prediction capability, get predictions
                    if hasattr(model.policy, "act_with_obj_pred"):
                        if "teacher_obs" in obs_dict:
                            pred_obj_pos = action_state.get("predicted_obj_pos", None)

                            gt_obj_pos_full = obs_dict.get("gt_obj_pos", None)
                            if gt_obj_pos_full is not None:
                                gt_obj_pos = gt_obj_pos_full[..., :3]  # First 3 dims are position

                            # Handle sequence dimension for both gt and pred
                            if pred_obj_pos is not None and len(pred_obj_pos.shape) == 3:
                                pred_obj_pos = pred_obj_pos[:, -1, :]  # Take last timestep
                            if gt_obj_pos is not None and len(gt_obj_pos.shape) == 3:
                                gt_obj_pos = gt_obj_pos[:, -1, :]  # Take last timestep

                            # Visualize at specified frequency and if enabled
                            if (
                                not self.env.headless
                                and self.enable_obj_pred_visualization
                                and step_count % self.obj_pred_vis_eval_frequency == 0
                                and pred_obj_pos is not None
                                and gt_obj_pos is not None
                            ):
                                max_vis_envs = min(
                                    self.obj_pred_vis_max_envs, pred_obj_pos.shape[0]
                                )
                                self.env.simulator.draw_obj_pos_predictions(
                                    pred_obj_pos[:max_vis_envs], gt_obj_pos[:max_vis_envs]
                                )

                    # Add object position data to action_state for episode data collection
                    if gt_obj_pos is not None:
                        action_state["gt_obj_pos"] = gt_obj_pos
                    if pred_obj_pos is not None:
                        action_state["pred_obj_pos"] = pred_obj_pos

                    last_obs_dict = obs_dict.copy()
                    action_state["actions"] = action_state["action_mean"]

                    self.env.render_results()
                    obs_dict, rewards, dones, infos = self.env.step(action_state)

                    if self.config.eval.save_videos:
                        self.batch_write_frame(last_obs_dict["vision_obs"])

                    # Data saving is disabled - no trajectory data collection

                    for obs_key in obs_dict.keys():
                        obs_dict[obs_key] = obs_dict[obs_key].to(device)

                    rewards, dones = rewards.to(device), dones.to(device)

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
                            # Original behavior: collect data from all completed episodes
                            valid_new_ids = new_ids

                        if len(valid_new_ids) > 0:
                            if self.config.eval.save_videos:
                                self.batch_reset_data_writer(valid_new_ids, save_dirpath)

                            # Process episode completions using environment callback
                            self.env.process_eval_episode_completions(
                                valid_new_ids, self.cur_reward_sum, self.cur_episode_length
                            )

                            if eval_num_envs_episodes:
                                # Mark these environments as having completed their first episode
                                self.env_episode_completed[valid_new_ids] = True
                                # Update progress bar with newly completed episodes
                                pbar.update(len(valid_new_ids))
                            else:
                                # Original behavior: update progress bar with collected episodes
                                pbar.update(
                                    len(self.env.eval_metrics["goal_reached_buffer"]) - pbar.n
                                )

                        # Reset all environments that finished (regardless of whether we collected data)
                        self.cur_reward_sum[new_ids] = 0
                        self.cur_episode_length[new_ids] = 0

                        # Reset environment episode tracking
                        self.env.reset_eval_episode_tracking(new_ids)

                    step_count += 1

                    # TODO: 0: make sure this is compatiable with all use cases
                    self.callback_handler.on_step_end(self.args, self.state, self.control)
                    torch.cuda.empty_cache()
        self.env.end_render_results()
        pbar.close()
        end_time = time.time()

        # # Get evaluation summary from environment
        eval_dict = self.env.get_eval_metrics_summary()
        eval_dict["eval_time"] = end_time - start_time

        print(f"Eval time: {end_time - start_time} seconds")
        print(f"Collected {eval_dict['collected_episodes']} episodes")
        print(f"Average goal reached rate: {eval_dict['goal_reached_rate']:.3f}")
        print(f"Average episode length: {eval_dict['episode_length']:.1f}")
        print(f"Average reward: {eval_dict['reward']:.1f}")
        print(f"Average DOF velocity: {eval_dict['eval_dof_vel']:.3f}")
        print(f"Average arm DOF velocity: {eval_dict['eval_arm_dof_vel']:.3f}")
        print(f"Average finger DOF velocity: {eval_dict['eval_finger_dof_vel']:.3f}")

        # save eval_dict to a file
        import json

        eval_output_dir = getattr(self.args, "eval_output_dir", self.args.output_dir)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir, exist_ok=True)
        metrics_eval_path = os.path.join(eval_output_dir, "metrics_eval.json")
        with open(metrics_eval_path, "w") as f:
            json.dump(eval_dict, f, indent=4)

        logger.info(f"Saved eval_dict to {metrics_eval_path}")  # self.args.eval_output_dir

        if self.config.eval.save_videos:
            print(f"Please check {save_dirpath}")

        # Restore original save_trajectories setting
        self.config.eval.save_trajectories = original_save_trajectories

        # Reset env (in the outer loop because we need the obs_dict), cur_reward_sum, cur_episode_length
        device = (
            self.accelerator.device
            if self.accelerator
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.int32, device=device)

        return None
