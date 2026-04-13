# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import gc
import time
from copy import deepcopy
from typing import Optional

import torch
from rich.console import Console
from transformers.trainer import *
from trl.trainer.ppo_trainer import *

from gr00t.rl.trl.trainer.distill_trainer import TRLDistillTrainer
from gr00t.rl.trl.trainer.ppo_trainer import (
    PolicyAndValueWrapper,
)
from gr00t.rl.trl.utils.rl import compute_episode_attnmask

console = Console()


class PolicyAndValueWrapperObjPred(PolicyAndValueWrapper):
    """
    Extended PolicyAndValueWrapper that supports object position prediction.
    """

    def forward_component(self, mode, actions=None, **kwargs):
        if mode == "policy_distill_obj_pred":
            # Use the new act_with_obj_pred method for object position prediction
            if hasattr(self.policy, "act_with_obj_pred"):
                return self.policy.act_with_obj_pred(**kwargs)
            else:
                # Fallback to regular act method if act_with_obj_pred is not available
                return self.policy.act(**kwargs)
        else:
            # Use parent class implementation for other modes
            return super().forward_component(mode, actions=actions, **kwargs)


class TRLDistillTrainerObjPred(TRLDistillTrainer):
    """
    Distillation trainer with object position prediction capability.
    Inherits from TRLDistillTrainer and adds object position prediction loss.
    """

    _tag_names = ["trl", "humanoid_distill_obj_pred"]

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
        # Call parent initialization
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

        # Initialize object position prediction loss function
        obj_pred_loss_type = self.config.get("obj_pred_loss_type", "l2")
        if obj_pred_loss_type == "l2":
            self.obj_pred_loss_fn = torch.nn.MSELoss()
        elif obj_pred_loss_type == "l1":
            self.obj_pred_loss_fn = torch.nn.L1Loss()
        else:
            raise ValueError(f"Invalid obj_pred_loss_type: {obj_pred_loss_type}")

        # Initialize visualization settings
        self.enable_obj_pred_visualization = self.env.config.get(
            "enable_obj_pred_visualization", False
        )
        self.obj_pred_vis_frequency = self.env.config.get("obj_pred_vis_frequency", 1)
        self.obj_pred_vis_max_envs = self.env.config.get("obj_pred_vis_max_envs", 8)
        self.obj_pred_vis_eval_frequency = self.env.config.get("obj_pred_vis_eval_frequency", 1)

        # Override the model with our custom wrapper that supports object prediction
        self.model = PolicyAndValueWrapperObjPred(
            self.policy_model, self.value_model, self.ref_model
        )
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    def _wrap_model(self, model, ref_model, value_model):
        """Override to use custom wrapper that supports object prediction."""
        return PolicyAndValueWrapperObjPred(model, value_model, ref_model)

    def _setup_storage(self):
        # Call parent setup first
        super()._setup_storage()

        # Register the predicted object position key (3D position prediction)
        self.storage.register_key("predicted_obj_pos", shape=(3,), dtype=torch.float)

    def policy_step(self, policy_model, obs_dict, cur_dones=None):
        policy_state_dict = {}
        if "teacher_obs" in obs_dict:
            # Get ground truth actions from teacher
            gt_actions = self.ref_model.act_inference(
                obs_dict=deepcopy(obs_dict), input_key="teacher_obs"
            )
            policy_state_dict["gt_actions"] = gt_actions

            # Get ground truth object position from dedicated gt_obj_pos observation
            # This contains the full target_obj_pos (9 dims), store as-is for consistency
            gt_obj_pos_full = obs_dict["gt_obj_pos"]
            policy_state_dict["gt_obj_pos"] = gt_obj_pos_full

            obs_dict = deepcopy(obs_dict)

        # Use rollout infrastructure like parent class, but with object prediction
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

        # Use the new rollout_with_obj_pred method that handles observation buffering
        student_state_dict = policy_model.rollout_with_obj_pred(
            obs_dict=actor_obs_dict, episode_attnmask=episode_attnmask, cur_dones=cur_dones
        )

        policy_state_dict.update(student_state_dict)

        if self.learn_normalized_actions:
            if "gt_actions" in policy_state_dict:
                assert "normalized_gt_actions" in policy_state_dict, f"{policy_state_dict.keys()=}"
            assert "normalized_actions" in policy_state_dict, f"{policy_state_dict.keys()=}"

        return policy_state_dict

    def _register_stats_buffer(self):
        super()._register_stats_buffer()
        args = self.args
        device = self.accelerator.device

        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        obj_pred_loss_stats = torch.zeros(stats_shape, device=device)
        weighted_obj_pred_loss_stats = torch.zeros(stats_shape, device=device)

        self.obj_pred_loss_stats = obj_pred_loss_stats
        self.weighted_obj_pred_loss_stats = weighted_obj_pred_loss_stats

    def _get_rollout_data(self, obs_keys):
        rollout_data = super()._get_rollout_data(obs_keys)
        device = self.accelerator.device

        # Add ground truth object position to rollout data
        gt_obj_pos = self.storage.gt_obj_pos.transpose(0, 1).contiguous().to(device)
        rollout_data["gt_obj_pos"] = gt_obj_pos

        # Extract additional observation keys that might be needed for the forward pass
        additional_obs_keys = ["state_history_obs"]

        for key in additional_obs_keys:
            if hasattr(self.storage, key):
                obs_data = getattr(self.storage, key).transpose(0, 1).contiguous().to(device)
                rollout_data[key] = obs_data

        # If recurrent, also split and pad any additional obs and gt tensors to align
        if rollout_data.get("padded_obs_dict") is not None:
            # Merge additional obs into padded_obs_dict by splitting them with the same dones
            from gr00t.rl.trl.utils.rl import split_and_pad_trajectories

            dones = rollout_data["dones"]
            padded_obs_dict = rollout_data["padded_obs_dict"]
            trajectory_masks = rollout_data["trajectory_masks"]
            for key in additional_obs_keys:
                if key in rollout_data:
                    obs_tensor = rollout_data[
                        key
                    ]  # [num_envs, num_steps, ...] or [num_envs, num_steps, hist, dim]
                    obs_transposed = obs_tensor.transpose(0, 1)
                    dones_transposed = dones.transpose(0, 1)
                    padded_obs, traj_masks = split_and_pad_trajectories(
                        obs_transposed, dones_transposed
                    )
                    padded_obs_dict[key] = padded_obs.transpose(0, 1)
                    trajectory_masks = (
                        traj_masks.transpose(0, 1) if trajectory_masks is None else trajectory_masks
                    )
            # Also split gt_obj_pos to padded layout
            if "gt_obj_pos" in rollout_data:
                gt_obj_pos = rollout_data["gt_obj_pos"]  # [num_envs, num_steps, 9]
                gt_obj_pos_t = gt_obj_pos.transpose(0, 1)
                dones_transposed = dones.transpose(0, 1)
                padded_gt_obj_pos_t, _ = split_and_pad_trajectories(gt_obj_pos_t, dones_transposed)
                rollout_data["padded_gt_obj_pos"] = padded_gt_obj_pos_t.transpose(
                    0, 1
                )  # [num_trajectories, max_traj_len, 9]
            rollout_data["padded_obs_dict"] = padded_obs_dict
            rollout_data["trajectory_masks"] = trajectory_masks

        return rollout_data

    def _get_mb_rollout_data(self, rollout_data, micro_batch_inds):
        mb_rollout_data = super()._get_mb_rollout_data(rollout_data, micro_batch_inds)

        # Add ground truth object position to mini-batch rollout data
        if (
            hasattr(self.policy_model, "is_recurrent") and self.policy_model.is_recurrent
        ) and rollout_data.get("padded_gt_obj_pos") is not None:
            # Use padded layout and slice by trajectory indices consistent with mb_obs_dict length
            padded_gt_obj_pos = rollout_data[
                "padded_gt_obj_pos"
            ]  # [num_trajectories, max_traj_len, 9]
            # Recover first/last traj range from mb_obs_dict size: we can't get indices directly here
            # So recompute based on maintained _current_first_traj in base. super() already advanced it.
            # We need the same range used in super(); since we don't have it, recompute from shapes:
            num_traj = mb_rollout_data["mb_obs_dict"][
                list(mb_rollout_data["mb_obs_dict"].keys())[0]
            ].shape[0]
            last_traj = self._current_first_traj  # already advanced by super()
            first_traj = last_traj - num_traj
            mb_gt_obj_pos = padded_gt_obj_pos[first_traj:last_traj]
        else:
            gt_obj_pos = rollout_data["gt_obj_pos"]
            mb_gt_obj_pos = gt_obj_pos[micro_batch_inds]

        mb_rollout_data["mb_gt_obj_pos"] = mb_gt_obj_pos

        # Extend mb_obs_dict to include additional observation keys that might be present (non-recurrent path)
        mb_obs_dict = mb_rollout_data["mb_obs_dict"]
        additional_obs_keys = ["state_history_obs"]
        if not (
            (hasattr(self.policy_model, "is_recurrent") and self.policy_model.is_recurrent)
            and rollout_data.get("padded_obs_dict") is not None
        ):
            for key in additional_obs_keys:
                if key in rollout_data:
                    mb_obs_data = rollout_data[key][micro_batch_inds]
                    mb_obs_dict[key] = mb_obs_data

        return mb_rollout_data

    def _forward_model(self, model, mb_rollout_data):
        mb_obs_dict = mb_rollout_data["mb_obs_dict"]
        episode_attnmask = mb_rollout_data.get("episode_attnmask", None)
        mb_masks = mb_rollout_data.get("mb_masks", None)
        mb_dones = mb_rollout_data.get("mb_dones", None)
        mb_hidden_states = mb_rollout_data.get("mb_hidden_states", None)
        actor_hidden_states = mb_hidden_states[0] if mb_hidden_states is not None else None

        results = model.forward(
            modes=["policy_distill_obj_pred"],
            input_kwargs=dict(
                policy_distill_obj_pred=dict(
                    obs_dict=deepcopy(mb_obs_dict),
                    episode_attnmask=episode_attnmask,
                    masks=mb_masks,
                    hidden_states=actor_hidden_states,
                    original_dones=mb_dones,
                )
            ),
        )

        return dict(
            policy_results=results["policy_distill_obj_pred"],
        )

    def _visualize_obj_predictions(self, forward_results, mb_rollout_data):
        """
        Visualize predicted and ground truth object positions during training.
        """
        # Only visualize if enabled and we have a simulator and it's not headless
        if (
            self.enable_obj_pred_visualization
            and hasattr(self, "env")
            and hasattr(self.env, "simulator")
            and not self.env.headless
        ):

            policy_results = forward_results["policy_results"]
            mb_gt_obj_pos_full = mb_rollout_data["mb_gt_obj_pos"]

            # Extract only the first 3 dimensions (object position) from ground truth
            mb_gt_obj_pos = mb_gt_obj_pos_full[..., :3]

            # Get predicted object position from policy results
            pred_obj_pos = policy_results["predicted_obj_pos"]

            # Handle sequence dimension: take the last timestep for visualization
            # pred_obj_pos shape: [batch_size, sequence_length, 3]
            # mb_gt_obj_pos shape: [batch_size, sequence_length, 3]
            if len(pred_obj_pos.shape) == 3:
                pred_obj_pos = pred_obj_pos[:, -1, :]  # Take last timestep
            if len(mb_gt_obj_pos.shape) == 3:
                mb_gt_obj_pos = mb_gt_obj_pos[:, -1, :]  # Take last timestep

            # Take only the first few environments for visualization to avoid clutter
            max_vis_envs = min(self.obj_pred_vis_max_envs, pred_obj_pos.shape[0])
            pred_obj_pos_vis = pred_obj_pos[:max_vis_envs]
            gt_obj_pos_vis = mb_gt_obj_pos[:max_vis_envs]

            # Call simulator visualization methods
            self.env.simulator.draw_obj_pos_predictions(pred_obj_pos_vis, gt_obj_pos_vis)

    def _compute_loss(self, forward_results, mb_rollout_data):
        # Get BC loss from parent class
        loss_dict = super()._compute_loss(forward_results, mb_rollout_data)

        # Add object position prediction loss
        obj_pred_loss_dict = self._compute_obj_pred_loss(forward_results, mb_rollout_data)

        # Visualize predictions (only if enabled and at specified frequency)
        if (
            self.enable_obj_pred_visualization
            and self.state.global_step % self.obj_pred_vis_frequency == 0
        ):
            self._visualize_obj_predictions(forward_results, mb_rollout_data)

        # Combine losses
        obj_pred_coef = self.config.get("obj_pred_loss_coef", 1.0)
        loss_dict["loss"] += obj_pred_coef * obj_pred_loss_dict["obj_pred_loss"]

        # Add to loss dict
        for k, v in obj_pred_loss_dict.items():
            loss_dict[k] = v

        return loss_dict

    def _compute_obj_pred_loss(self, forward_results, mb_rollout_data):
        policy_results = forward_results["policy_results"]
        mb_gt_obj_pos_full = mb_rollout_data["mb_gt_obj_pos"]

        # Extract only the first 3 dimensions (object position) from ground truth
        # mb_gt_obj_pos_full contains [obj_relative_pos, obj_linear_vel, obj_ang_vel] (9 dims total)
        # We only want to predict obj_relative_pos (first 3 dims)
        mb_gt_obj_pos = mb_gt_obj_pos_full[..., :3]

        # Get predicted object position from policy results
        if self.config.get("use_obj_pred", True):
            pred_obj_pos = policy_results["predicted_obj_pos"]
            # Compute loss between predicted and ground truth object positions (both 3D)
            obj_pred_loss = self.obj_pred_loss_fn(pred_obj_pos, mb_gt_obj_pos)
        else:
            obj_pred_loss = 0.0

        return dict(
            obj_pred_loss=obj_pred_loss,
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
        super()._update_stats_buffer(
            ppo_epoch_idx,
            minibatch_idx,
            microbatch_idx,
            loss_dict,
            forward_results,
            mb_rollout_data,
        )

        # Update object prediction loss stats
        obj_pred_loss = loss_dict["obj_pred_loss"]
        obj_pred_coef = self.config.get("obj_pred_loss_coef", 1.0)

        self.obj_pred_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = obj_pred_loss
        self.weighted_obj_pred_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = (
            obj_pred_coef * obj_pred_loss
        )

    def _get_train_metrics(self):
        metrics = super()._get_train_metrics()

        # Add object prediction loss metrics
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

        torch.cuda.empty_cache()
        gc.collect()
        return metrics

    def eval(self, save_dirpath: Optional[str] = None):
        """
        Evaluation method with object position prediction visualization.
        Student episode data saving is controlled by configuration.
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

        # # Get evaluation mode configuration
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
            self.env_episode_completed = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.accelerator.device
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
        dones = torch.zeros(self.env.num_envs, device=self.accelerator.device)
        step_count = 0

        with torch.no_grad():
            with unwrap_model_for_generation(
                self.model,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as model:
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
                        obs_dict[obs_key] = obs_dict[obs_key].to(self.accelerator.device)

                    rewards, dones = rewards.to(self.accelerator.device), dones.to(
                        self.accelerator.device
                    )

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

        # Get evaluation summary from environment
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
        if self.config.eval.save_videos:
            print(f"Please check {save_dirpath}")

        # Restore original save_trajectories setting
        self.config.eval.save_trajectories = original_save_trajectories

        # Reset env (in the outer loop because we need the obs_dict), cur_reward_sum, cur_episode_length
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.accelerator.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.int32, device=self.accelerator.device
        )

        return eval_dict
