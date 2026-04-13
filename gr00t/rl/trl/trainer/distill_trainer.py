# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import gc
import logging as _logging
from copy import deepcopy

import torch
from rich.console import Console
from transformers.trainer import *
from trl.trainer.ppo_trainer import *

from gr00t.rl.trl.trainer.ppo_trainer import (
    TRLPPOTrainer,
)
from gr00t.rl.trl.utils.rl import compute_episode_attnmask
from gr00t.rl.trl.utils.scheduler import WarmupCosineScheduler

logger = _logging.getLogger(__name__)
console = Console()


class TRLDistillTrainer(TRLPPOTrainer):
    """
    Custom PPO Trainer that adapts TRL's PPOTrainer to work with Humanoid environments.
    """

    _tag_names = ["trl", "humanoid_distill"]

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
        self.start_time = time.time()  # Initialize start_time for proper logging
        self.custom_log = True  # Flag to indicate we're using custom logging
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
        dagger_bc_loss_type = self.config.get("dagger_bc_loss_type", "l2")
        if dagger_bc_loss_type == "l2":
            self.bc_loss_fn = torch.nn.MSELoss()
        elif dagger_bc_loss_type == "l1":
            self.bc_loss_fn = torch.nn.L1Loss()
        else:
            raise ValueError(f"Invalid dagger_bc_loss_type: {self.config.dagger_bc_loss_type}")

        self.train_with_evaluating_env = self.config.get("train_with_evaluating_env", True)

        self.compute_dagger_bc_loss_w_imgaug = self.config.get(
            "compute_dagger_bc_loss_w_imgaug", False
        )

        # Setup cosine learning rate scheduler if specified in config
        self._setup_cosine_scheduler(args)

        self.load_teacher_actor()

    def _setup_cosine_scheduler(self, args):
        """Setup cosine learning rate scheduler if specified in config."""
        use_cosine_scheduler = self.config.get("use_cosine_scheduler", False)

        if use_cosine_scheduler:
            # Ensure optimizer exists
            if not hasattr(self, "optimizer") or self.optimizer is None:
                raise RuntimeError("Optimizer must be created before setting up cosine scheduler")

            # Get scheduler parameters from config
            num_warmup_steps = self.config.get("cosine_scheduler_warmup_steps", 0)
            final_lr = self.config.get("cosine_scheduler_final_lr", 0.0)
            num_training_steps = args.num_total_batches

            # Validate parameters
            if num_warmup_steps < 0:
                raise ValueError(
                    f"cosine_scheduler_warmup_steps must be non-negative, got {num_warmup_steps}"
                )
            if num_warmup_steps >= num_training_steps:
                raise ValueError(
                    f"cosine_scheduler_warmup_steps ({num_warmup_steps}) must be less than num_training_steps ({num_training_steps})"
                )
            if final_lr < 0:
                raise ValueError(f"cosine_scheduler_final_lr must be non-negative, got {final_lr}")

            # Create cosine scheduler
            self.lr_scheduler = WarmupCosineScheduler(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                final_lr=final_lr,
            )

            logger.info(
                f"Using cosine learning rate scheduler with warmup_steps={num_warmup_steps}, "
                f"final_lr={final_lr}, num_training_steps={num_training_steps}"
            )
        # If not using cosine scheduler, keep the default scheduler from parent class

    def _setup_storage(self):
        super()._setup_storage()
        self.storage.register_key("gt_actions", shape=(self.num_act,), dtype=torch.float)
        if self.learn_normalized_actions:
            self.storage.register_key(
                "normalized_gt_actions", shape=(self.num_act,), dtype=torch.float
            )

    def load_teacher_actor(self):
        ckpt_path = self.config.network_load_dict.teacher_actor.path
        logger.info(f"Loading teacher actor from {ckpt_path}")
        loaded_dict = torch.load(ckpt_path, weights_only=False)
        if "actor_model_state_dict" in loaded_dict:
            self.ref_model.load_state_dict(loaded_dict["actor_model_state_dict"])
        elif "policy_state_dict" in loaded_dict:
            self.ref_model.load_state_dict(loaded_dict["policy_state_dict"])
        else:
            raise ValueError(f"Available keys: {loaded_dict.keys()}\nckpt_path: {ckpt_path}")

        self.ref_model.eval()

    # def policy_step(self, policy_model, obs_dict, cur_dones=None):
    #     policy_state_dict = {}
    #     if "teacher_obs" in obs_dict:
    #         # During training forward, we only need to forward the student model
    #         gt_actions = self.ref_model.act(obs_dict=deepcopy(obs_dict))["actions"]
    #         gt_action_mean = self.ref_model.action_mean
    #         assert torch.allclose(gt_actions, gt_action_mean), (
    #             f"Max diff: {torch.max(torch.abs(gt_actions - gt_action_mean))}"
    #             f"gt_actions range: {gt_actions.min()=}, {gt_actions.max()=}"
    #             f"gt_action_mean range: {gt_action_mean.min()=}, {gt_action_mean.max()=}"
    #         )
    #         policy_state_dict["gt_actions"] = gt_actions
    #         obs_dict = deepcopy(obs_dict)
    #         obs_dict["gt_actions"] = gt_actions
    #     student_state_dict = super().policy_step(policy_model, obs_dict)
    #     policy_state_dict.update(student_state_dict)

    #     if self.learn_normalized_actions:
    #         if "gt_actions" in obs_dict:
    #             assert "normalized_gt_actions" in policy_state_dict, f"{policy_state_dict.keys()=}"
    #         assert "normalized_actions" in policy_state_dict, f"{policy_state_dict.keys()=}"

    #     return policy_state_dict

    # workable version
    def policy_step(self, policy_model, obs_dict, cur_dones=None):
        policy_state_dict = {}
        if "teacher_obs" in obs_dict:
            # During training forward, we only need to forward the student model
            # replace actor_obs with teacher_obs
            gt_actions = self.ref_model.act_inference(
                obs_dict=deepcopy(obs_dict), input_key="teacher_obs"
            )  # Tairan: we use act_inference here, which is not the same as act(), to get action_mean

            policy_state_dict["gt_actions"] = gt_actions
            obs_dict = deepcopy(obs_dict)

        # use PPO's policy step since we need the PPO behavior to be consistent with the PPOTrainer
        student_state_dict = super().policy_step(policy_model, obs_dict, cur_dones=cur_dones)

        # student_actions = policy_model.act_inference(obs_dict=obs_dict)
        # student_state_dict = {}
        # student_state_dict["actions"] = student_actions
        # student_state_dict["action_mean"] = student_actions

        # if len(student_state_dict["actions"].shape) == 3:
        #     student_state_dict["actions"] = student_state_dict["actions"].squeeze(1)
        #     student_state_dict["actions_log_prob"] = student_state_dict["actions_log_prob"].squeeze(1)
        #     student_state_dict["action_mean"] = student_state_dict["action_mean"].squeeze(1)
        #     student_state_dict["action_sigma"] = student_state_dict["action_sigma"].squeeze(1)

        policy_state_dict.update(student_state_dict)

        # policy_state_dict['actions'] = policy_state_dict['gt_actions']; print("using GT actions") # ZL: debugging distillation
        # policy_state_dict['action_mean'] = policy_state_dict['gt_actions']; print("using GT actions") # ZL: debugging distillation

        if self.learn_normalized_actions:
            if "gt_actions" in policy_state_dict:
                assert "normalized_gt_actions" in policy_state_dict, f"{policy_state_dict.keys()=}"
            assert "normalized_actions" in policy_state_dict, f"{policy_state_dict.keys()=}"

        return policy_state_dict

    def _register_stats_buffer(self):
        args = self.args
        device = self.accelerator.device

        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        bc_loss_stats = torch.zeros(stats_shape, device=device)
        weighted_bc_loss_stats = torch.zeros(stats_shape, device=device)

        self.dagger_bc_loss_stats = bc_loss_stats
        self.weighted_dagger_bc_loss_stats = weighted_bc_loss_stats

    def _process_env_step(self, rewards, dones, infos):
        super()._process_env_step(rewards, dones, infos)
        # Ensure teacher's recurrent state resets on episode boundaries
        if (
            hasattr(self, "ref_model")
            and self.ref_model is not None
            and hasattr(self.ref_model, "reset")
        ):
            try:
                self.ref_model.reset(dones)
            except Exception:
                pass

    def _rollout_step(self, model, obs_dict):
        # Initialize teacher rollout context if supported (for recurrent teachers)
        if (
            hasattr(self, "ref_model")
            and self.ref_model is not None
            and hasattr(self.ref_model, "init_rollout")
        ):
            try:
                self.ref_model.init_rollout()
            except Exception:
                pass
        out_obs = super()._rollout_step(model, obs_dict)
        if (
            hasattr(self, "ref_model")
            and self.ref_model is not None
            and hasattr(self.ref_model, "clear_rollout")
        ):
            try:
                self.ref_model.clear_rollout()
            except Exception:
                pass
        return out_obs

    def _get_rollout_data(self, obs_keys):
        device = self.accelerator.device

        actor_obs = self.storage.actor_obs.transpose(0, 1).contiguous().to(device)
        vision_obs = (
            self.storage.vision_obs.transpose(0, 1).contiguous().to(device)
            if hasattr(self.storage, "vision_obs")
            else None
        )
        gt_actions = self.storage.gt_actions.transpose(0, 1).contiguous().to(device)
        if self.learn_normalized_actions:
            normalized_gt_actions = (
                self.storage.normalized_gt_actions.transpose(0, 1).contiguous().to(device)
            )
        else:
            normalized_gt_actions = None
        dones = self.storage.dones.transpose(0, 1).squeeze(-1).contiguous().to(device)

        # For recurrent policies, pre-split trajectories globally
        padded_obs_dict = None
        trajectory_masks = None
        padded_gt_actions = None
        padded_normalized_gt_actions = None
        if hasattr(self.policy_model, "is_recurrent") and self.policy_model.is_recurrent:
            from gr00t.rl.trl.utils.rl import split_and_pad_trajectories

            padded_obs_dict = {}
            # Build combined obs dict for splitting
            all_obs_dict = {"actor_obs": actor_obs}
            if vision_obs is not None:
                all_obs_dict["vision_obs"] = vision_obs
            for key, obs_tensor in all_obs_dict.items():
                # obs_tensor: [num_envs, num_steps, ...]
                obs_transposed = obs_tensor.transpose(0, 1)
                dones_transposed = dones.transpose(0, 1)
                padded_obs, traj_masks = split_and_pad_trajectories(
                    obs_transposed, dones_transposed
                )
                padded_obs_dict[key] = padded_obs.transpose(
                    0, 1
                )  # [num_trajectories, max_traj_len, ...]
                if trajectory_masks is None:
                    trajectory_masks = traj_masks.transpose(
                        0, 1
                    )  # [num_trajectories, max_traj_len]

            # Also split and pad ground-truth actions to match model output layout
            gt_actions_transposed = gt_actions.transpose(0, 1)  # [num_steps, num_envs, act]
            padded_gt_actions_t, _ = split_and_pad_trajectories(
                gt_actions_transposed, dones_transposed
            )
            padded_gt_actions = padded_gt_actions_t.transpose(
                0, 1
            )  # [num_trajectories, max_traj_len, act]
            if self.learn_normalized_actions and normalized_gt_actions is not None:
                norm_gt_transposed = normalized_gt_actions.transpose(0, 1)
                padded_norm_gt_t, _ = split_and_pad_trajectories(
                    norm_gt_transposed, dones_transposed
                )
                padded_normalized_gt_actions = padded_norm_gt_t.transpose(0, 1)

        rollout_data = dict(
            actor_obs=actor_obs,
            vision_obs=vision_obs,
            gt_actions=gt_actions,
            normalized_gt_actions=normalized_gt_actions,
            dones=dones,
            padded_obs_dict=padded_obs_dict,
            trajectory_masks=trajectory_masks,
            padded_gt_actions=padded_gt_actions,
            padded_normalized_gt_actions=padded_normalized_gt_actions,
        )

        torch.cuda.empty_cache()
        return rollout_data

    def _get_mb_rollout_data(self, rollout_data, micro_batch_inds):
        actor_obs = rollout_data["actor_obs"]
        vision_obs = rollout_data["vision_obs"]
        gt_actions = rollout_data["gt_actions"]
        normalized_gt_actions = rollout_data["normalized_gt_actions"]
        dones = rollout_data["dones"]

        # Prepare episode attention mask for recurrent
        mb_dones = dones[micro_batch_inds]
        episode_attnmask = compute_episode_attnmask(mb_dones)

        # Defaults for non-recurrent
        mb_gt_actions = gt_actions[micro_batch_inds]
        if self.learn_normalized_actions:
            mb_normalized_gt_actions = normalized_gt_actions[micro_batch_inds]
        else:
            mb_normalized_gt_actions = None

        # Slice padded trajectories for recurrent policies
        mb_hidden_states = None
        if (
            hasattr(self.policy_model, "is_recurrent")
            and self.policy_model.is_recurrent
            and rollout_data.get("padded_obs_dict") is not None
        ):
            padded_obs_dict = rollout_data["padded_obs_dict"]
            trajectory_masks = rollout_data["trajectory_masks"]
            padded_gt_actions = rollout_data.get("padded_gt_actions", None)
            padded_normalized_gt_actions = rollout_data.get("padded_normalized_gt_actions", None)

            # Count trajectories in this mini-batch
            last_was_done = torch.zeros_like(mb_dones, dtype=torch.bool)
            last_was_done[:, 1:] = mb_dones[:, :-1].bool()
            last_was_done[:, 0] = True
            num_trajectories = torch.sum(last_was_done).item()
            # Range in global pre-split tensors
            first_traj = getattr(self, "_current_first_traj", 0)
            last_traj = first_traj + num_trajectories
            # Build obs dict
            mb_obs_dict = {
                key: padded_obs_dict[key][first_traj:last_traj] for key in padded_obs_dict.keys()
            }
            mb_masks = trajectory_masks[first_traj:last_traj]
            # GT actions aligned to padded layout
            if padded_gt_actions is not None:
                mb_gt_actions = padded_gt_actions[first_traj:last_traj]
            if self.learn_normalized_actions and padded_normalized_gt_actions is not None:
                mb_normalized_gt_actions = padded_normalized_gt_actions[first_traj:last_traj]
            # Advance counter
            self._current_first_traj = last_traj

            # Extract hidden states saved at traj boundaries if present
            if (
                self.storage.saved_hidden_states_a is not None
                or self.storage.saved_hidden_states_c is not None
            ):
                # Determine which timesteps start trajectories
                dones_for_extraction = mb_dones.clone()
                last_was_done = torch.zeros_like(dones_for_extraction, dtype=torch.bool)
                last_was_done[:, 1:] = dones_for_extraction[:, :-1].bool()
                last_was_done[:, 0] = True
                # Actor hidden states
                if self.storage.saved_hidden_states_a is not None:
                    hid_a_batch = []
                    for saved_hidden_states in self.storage.saved_hidden_states_a:
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
                # Critic hidden states
                if self.storage.saved_hidden_states_c is not None:
                    hid_c_batch = []
                    for saved_hidden_states in self.storage.saved_hidden_states_c:
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
            # Non-recurrent
            mb_actor_obs = actor_obs[micro_batch_inds]
            mb_vision_obs = vision_obs[micro_batch_inds] if vision_obs is not None else None
            if mb_vision_obs is not None:
                mb_obs_dict = {
                    "actor_obs": mb_actor_obs,
                    "vision_obs": mb_vision_obs,
                }
            else:
                mb_obs_dict = {
                    "actor_obs": mb_actor_obs,
                }
            mb_masks = None

        return dict(
            mb_gt_actions=mb_gt_actions,
            mb_normalized_gt_actions=mb_normalized_gt_actions,
            mb_obs_dict=mb_obs_dict,
            episode_attnmask=episode_attnmask,
            mb_masks=mb_masks,
            mb_hidden_states=mb_hidden_states,
            mb_dones=mb_dones,
        )

    def _forward_model(self, model, mb_rollout_data):
        mb_obs_dict = mb_rollout_data["mb_obs_dict"]
        episode_attnmask = mb_rollout_data.get("episode_attnmask", None)
        mb_masks = mb_rollout_data.get("mb_masks", None)
        mb_dones = mb_rollout_data.get("mb_dones", None)
        mb_hidden_states = mb_rollout_data.get("mb_hidden_states", None)
        actor_hidden_states = mb_hidden_states[0] if mb_hidden_states is not None else None

        results = model.forward(
            modes=["policy_distill"],
            input_kwargs=dict(
                policy_distill=dict(
                    obs_dict=mb_obs_dict,
                    episode_attnmask=episode_attnmask,
                    masks=mb_masks,
                    hidden_states=actor_hidden_states,
                    original_dones=mb_dones,
                )
            ),
        )

        return dict(
            policy_results=results["policy_distill"],
        )

    def _compute_loss(self, forward_results, mb_rollout_data):
        dagger_bc_loss_dict = self._compute_dagger_bc_loss(forward_results, mb_rollout_data)

        return dict(
            loss=self.config.dagger_bc_loss_coef * dagger_bc_loss_dict["dagger_bc_loss"],
            **dagger_bc_loss_dict,
        )

    def _compute_dagger_bc_loss(self, forward_results, mb_rollout_data):
        policy_results = forward_results["policy_results"]
        mb_gt_actions = mb_rollout_data["mb_gt_actions"]
        mb_normalized_gt_actions = mb_rollout_data["mb_normalized_gt_actions"]

        if self.learn_normalized_actions:
            if self.compute_dagger_bc_loss_w_imgaug:
                normalized_actions = policy_results["normalized_actions_w_imgaug"]
            else:
                normalized_actions = policy_results["normalized_actions"]
            # Because the model can use bfloat16, for example, n1.5
            normalized_actions = normalized_actions.to(mb_normalized_gt_actions.dtype)
            dagger_bc_loss = self.bc_loss_fn(normalized_actions, mb_normalized_gt_actions)
        else:
            if self.compute_dagger_bc_loss_w_imgaug:
                actions = policy_results["action_mean_w_imgaug"]
            else:
                actions = policy_results["action_mean"]
            # Because the model can use bfloat16, for example, n1.5
            actions = actions.to(mb_gt_actions.dtype)
            dagger_bc_loss = self.bc_loss_fn(actions, mb_gt_actions)

        return dict(
            dagger_bc_loss=dagger_bc_loss,
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
        dagger_bc_loss = loss_dict["dagger_bc_loss"]

        self.dagger_bc_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = dagger_bc_loss
        self.weighted_dagger_bc_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = (
            self.config.dagger_bc_loss_coef * dagger_bc_loss
        )

    def _get_train_metrics(self):
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

        # Add learning rate to metrics
        if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]

        torch.cuda.empty_cache()
        gc.collect()
        return metrics
