# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0



import numpy as np
import torch

from gr00t.rl.envs.base_task.staged_task_base import StagedTaskBase
from gr00t.rl.isaac_utils.rotations import quat_to_tan_norm
from gr00t.rl.utils.torch_utils import *
from gr00t.rl.utils.torch_utils import quat_conjugate, quat_mul, quat_rotate, quat_rotate_inverse


class WalkTableStand(StagedTaskBase):
    def __init__(self, config, device):
        super().__init__(config, device)
        # self.finger_frame_transformer_idx = [self.simulator.object_to_hand_frame_transformer.data.target_frame_names.index(link) for link in self.simulator.object_to_hand_frame_transformer.data.target_frame_names if "thumb" in link or "index" in link or "middle" in link]
        self.pelvis_idx = self.simulator.body_names.index("pelvis")
        self.torso_idx = self.simulator.body_names.index("torso_link")
        self.palm_index = self.simulator.find_rigid_body_indice(self.config.task.manipulate_hand)
        self.num_total_task = 0
        self.num_total_success = 0
        from collections import deque

        self.walk_stand_goal_reached_deque = deque(maxlen=500)
        self.average_walk_stand_goal_reached_rate = 0.0
        self.skip_initial_goal_tracking = True
        self.max_consecutive_success_frames = getattr(
            self.config, "max_consecutive_success_frames", 50
        )
        if self.config.termination_large_homie_linvel_command_curriculum.enable:
            self.termination_large_homie_linvel_command_threshold = to_torch(
                self.config.termination_large_homie_linvel_command_curriculum.init_limit,
                device=self.device,
            )
        else:
            self.termination_large_homie_linvel_command_threshold = (
                self.config.termination_scales.termination_large_homie_linvel_command_threshold
            )

    def _init_buffers(self):
        super()._init_buffers()

        if hasattr(self.simulator, "task_root_origin"):
            # Get table_grid base position
            if "table_grid" in self.simulator.task_root_origin:
                self.base_table_pos = self.simulator.task_root_origin["table_grid"][:3]
            else:
                raise ValueError("table_grid not found in task_root_origin")

            # Get random_object base position
            if "random_object" in self.simulator.task_root_origin:
                self.initial_object_pos = self.simulator.task_root_origin["random_object"][:3]
            elif "hold_object" in self.simulator.task_root_origin:
                self.initial_hold_object_pos = self.simulator.task_root_origin["hold_object"][:3]
                self.initial_front_object_pos = self.simulator.task_root_origin["front_object"][:3]
                self.initial_back_object_pos = self.simulator.task_root_origin["back_object"][:3]
            else:
                raise ValueError("required object not found in task_root_origin")
        else:
            raise ValueError("task_root_origin not found")

        # Initialize table_grid root state buffer
        self.table_grid_root_state_buf = torch.zeros(
            self.num_envs, 13, device=self.device, requires_grad=False
        )
        self.table_grid_root_state_buf[:, 3] = 1.0  # w
        self.table_grid_root_state_buf[:, :3] = self.base_table_pos[None, :] + self.env_origins

        if "random_object" in self.simulator.task_root_origin:
            self.bottle_root_state_buf = torch.zeros(
                self.num_envs, 13, device=self.device, requires_grad=False
            )
            self.bottle_root_state_buf[:, 3] = 1.0  # w
            self.bottle_root_state_buf[:, :3] = self.initial_object_pos[None, :] + self.env_origins
        elif "hold_object" in self.simulator.task_root_origin:
            self.hold_bottle_root_state_buf = torch.zeros(
                self.num_envs, 13, device=self.device, requires_grad=False
            )
            self.hold_bottle_root_state_buf[:, 3] = 1.0  # w
            self.hold_bottle_root_state_buf[:, :3] = (
                self.initial_hold_object_pos[None, :] + self.env_origins
            )
            self.front_bottle_root_state_buf = torch.zeros(
                self.num_envs, 13, device=self.device, requires_grad=False
            )
            self.front_bottle_root_state_buf[:, 3] = 1.0  # w
            self.front_bottle_root_state_buf[:, :3] = (
                self.initial_front_object_pos[None, :] + self.env_origins
            )
            self.back_bottle_root_state_buf = torch.zeros(
                self.num_envs, 13, device=self.device, requires_grad=False
            )
            self.back_bottle_root_state_buf[:, 3] = 1.0  # w
            self.back_bottle_root_state_buf[:, :3] = (
                self.initial_back_object_pos[None, :] + self.env_origins
            )

        self.walk_stand_goal_reached_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.goal_pos = torch.tensor(self.config.goal_pos, device=self.device)
        self.box_robot_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.box_target_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.box_palm_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.robot_target_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(
            self.num_envs, 3
        )
        self.walk_stand_thre = self.config.rewards.walk_stand_threshold_distance
        self.success_buffer = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.consecutive_success_count = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device, requires_grad=False
        )

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        if len(env_ids) > 0:
            self._update_average_walk_stand_goal_reached_rate(env_ids)
            # self._update_average_dof_vel(env_ids) # TODO(qben): should use this?
        super()._reset_buffers_callback(env_ids, target_buf)

    # TODO(qben) how to understand this part?
    def _update_average_walk_stand_goal_reached_rate(self, env_ids):
        """Update the running average of goal reached rate for completed episodes."""
        # Skip adding to deque during initial reset phase
        if self.skip_initial_goal_tracking:
            # Only skip if this is the initial reset with all environments
            if len(env_ids) == self.num_envs and torch.equal(
                env_ids, torch.arange(self.num_envs, device=self.device)
            ):
                # This is the initial reset, skip adding to deque but still update the average
                num = len(env_ids)
                current_average_walk_stand_goal_reached_rate = torch.mean(
                    self.walk_stand_goal_reached_buf[env_ids], dtype=torch.float
                )
                self.average_walk_stand_goal_reached_rate = (
                    self.average_walk_stand_goal_reached_rate
                    * (1 - num / self.num_compute_average_epl)
                    + current_average_walk_stand_goal_reached_rate
                    * (num / self.num_compute_average_epl)
                )

                # Set flag to False after initial reset
                self.skip_initial_goal_tracking = False

                # Create target_pos_buf but skip adding to deque
                self.target_pos_buf = torch.zeros(
                    self.num_envs, 3, device=self.device, requires_grad=False
                )
                return

        # Normal behavior for subsequent resets
        num = len(env_ids)
        current_average_walk_stand_goal_reached_rate = torch.mean(
            self.walk_stand_goal_reached_buf[env_ids], dtype=torch.float
        )

        self.average_walk_stand_goal_reached_rate = self.average_walk_stand_goal_reached_rate * (
            1 - num / self.num_compute_average_epl
        ) + current_average_walk_stand_goal_reached_rate * (num / self.num_compute_average_epl)
        # Only add to deque for non-initial resets
        self.walk_stand_goal_reached_deque.extend(
            self.walk_stand_goal_reached_buf[env_ids][:].cpu().numpy().tolist()
        )

        self.target_pos_buf = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

    def _post_compute_observations_callback(self):
        self.box_robot_dist[:] = torch.norm(
            self.simulator.all_root_states[self.config.task.target_obj][:, :2]
            - self.simulator.robot_root_states[:, :2],
            dim=1,
        )
        self.robot_target_dist[:] = torch.norm(
            self.simulator.all_root_states[self.config.task.target_obj][:, :2]
            - self.simulator.env_origins[:, :2]
            - self.goal_pos[None, :2],
            dim=1,
        )
        self.box_target_dist[:] = torch.norm(
            self.simulator.all_root_states[self.config.task.target_obj][:, :3]
            - self.simulator.env_origins[:, :]
            - self.goal_pos[None, :],
            dim=1,
        )
        self.box_palm_dist[:] = torch.norm(
            self.simulator.all_root_states[self.config.task.target_obj][:, :3]
            - self.simulator._rigid_body_pos[:, self.palm_index][:, :],
            dim=1,
        )
        self.success_buffer[:] = (
            self.box_robot_dist <= self.config.rewards.walk_stand_threshold_distance
        ) & (torch.norm(self._homie_commands[:, :3], dim=1) <= 0.1)
        self.consecutive_success_count[self.success_buffer] += 1
        self.consecutive_success_count[~self.success_buffer] = 0

        self.walk_stand_goal_reached_buf |= (
            self.consecutive_success_count >= self.max_consecutive_success_frames
        )

        self.extras["walk_stand_goal_reached"] = self.walk_stand_goal_reached_buf.clone()

        # Add recent goal reached rate to extras for logging
        # Now uses the comprehensive deque that tracks ALL episode completions
        if len(self.walk_stand_goal_reached_deque) > 0:
            recent_walk_stand_goal_reached_rate = sum(self.walk_stand_goal_reached_deque) / len(
                self.walk_stand_goal_reached_deque
            )
            self.extras["episode"][
                "recent_walk_stand_goal_reached_rate"
            ] = recent_walk_stand_goal_reached_rate
        else:
            self.extras["episode"]["recent_walk_stand_goal_reached_rate"] = 0.0

        super()._post_compute_observations_callback()

    def _reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        super().reset_envs_idx(env_ids, target_states, target_buf)

    def wrap_to_pi(self, angles: torch.Tensor) -> torch.Tensor:
        wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
        return torch.where((wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi)

    def _reset_root_states(self, env_ids, target_root_states=None):
        """Resets ROOT states position and velocities of selected environmments
            if target_root_states is not None, reset to target_root_states
        Args:
            env_ids (List[int]): Environemnt ids
            target_root_states (Tensor): Target root states
        """
        init_root_state = self.base_init_state.clone()[None, :].repeat(len(env_ids), 1)
        init_root_state[:, 3:7] = init_root_state[:, [6, 3, 4, 5]]  # xyzw to wxyz
        init_root_state[:, :3] += self.env_origins[env_ids]
        self.target_robot_root_states[env_ids] = init_root_state
        if self.config.randomize_init_pos:
            self.target_robot_root_states[env_ids, 0:1] += torch_rand_float(
                *self.config.randomize_init_pos_range.x, (len(env_ids), 1), device=str(self.device)
            )
            self.target_robot_root_states[env_ids, 1:2] += torch_rand_float(
                *self.config.randomize_init_pos_range.y, (len(env_ids), 1), device=str(self.device)
            )
            self.target_robot_root_states[env_ids, 7:13] = torch_rand_float(
                *self.config.randomize_init_pos_range.vel,
                (len(env_ids), 6),
                device=str(self.device),
            )  # [7:10]: lin vel, [10:13]: ang vel

    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)

        if (
            self.config.domain_rand.get("randomize_dome_light", False)
            or self.config.domain_rand.get("randomize_robot_material", False)
            or self.config.domain_rand.get("randomize_floor_material", False)
        ):
            self.simulator.trigger_reset_events(env_ids, self.common_step_counter)

        if self.config.domain_rand.get("randomize_object_material", False):
            self.simulator.set_task_visual_state_tensor(env_ids)

        self._reset_task_states(env_ids)
        self.walk_stand_goal_reached_buf[env_ids] = False
        self.consecutive_success_count[env_ids] = 0

        # Reset DOF velocity tracking buffers
        self.last_episode_length_buf[env_ids] = 0

    def _reset_task_states(self, env_ids):
        if "random_object" in self.simulator.task_root_origin:
            self.bottle_root_state_buf[env_ids, 0] = (
                torch_rand_float(
                    *self.config.bottle_pos_range_x, (len(env_ids), 1), device=str(self.device)
                ).squeeze(-1)
                + self.env_origins[env_ids, 0]
                + self.initial_object_pos[0]
            )
            self.bottle_root_state_buf[env_ids, 1] = (
                torch_rand_float(
                    *self.config.bottle_pos_range_y, (len(env_ids), 1), device=str(self.device)
                ).squeeze(-1)
                + self.env_origins[env_ids, 1]
                + self.initial_object_pos[1]
            )

            task_root_state_dict = {"random_object": self.bottle_root_state_buf}
        elif "hold_object" in self.simulator.task_root_origin:
            self.hold_bottle_root_state_buf[env_ids, 0] = (
                torch_rand_float(
                    *self.config.bottle_pos_range_x, (len(env_ids), 1), device=str(self.device)
                ).squeeze(-1)
                + self.env_origins[env_ids, 0]
                + self.initial_hold_object_pos[0]
            )
            self.hold_bottle_root_state_buf[env_ids, 1] = (
                torch_rand_float(
                    *self.config.bottle_pos_range_y, (len(env_ids), 1), device=str(self.device)
                ).squeeze(-1)
                + self.env_origins[env_ids, 1]
                + self.initial_hold_object_pos[1]
            )
            self.front_bottle_root_state_buf[env_ids, 0] = (
                torch_rand_float(
                    *self.config.bottle_pos_range_x, (len(env_ids), 1), device=str(self.device)
                ).squeeze(-1)
                + self.env_origins[env_ids, 0]
                + self.initial_front_object_pos[0]
            )
            self.front_bottle_root_state_buf[env_ids, 1] = (
                torch_rand_float(
                    *self.config.bottle_pos_range_y, (len(env_ids), 1), device=str(self.device)
                ).squeeze(-1)
                + self.env_origins[env_ids, 1]
                + self.initial_front_object_pos[1]
            )
            self.back_bottle_root_state_buf[env_ids, 0] = (
                torch_rand_float(
                    *self.config.bottle_pos_range_x, (len(env_ids), 1), device=str(self.device)
                ).squeeze(-1)
                + self.env_origins[env_ids, 0]
                + self.initial_back_object_pos[0]
            )
            self.back_bottle_root_state_buf[env_ids, 1] = (
                torch_rand_float(
                    *self.config.bottle_pos_range_y, (len(env_ids), 1), device=str(self.device)
                ).squeeze(-1)
                + self.env_origins[env_ids, 1]
                + self.initial_back_object_pos[1]
            )
        self.simulator.set_task_root_state_tensor(env_ids, task_root_state_dict)

    def step(self, actor_state):
        return super().step(actor_state)

    def _update_timeout_buf(self):
        super()._update_timeout_buf()
        consecutive_success_mask = (
            self.consecutive_success_count >= self.max_consecutive_success_frames
        )
        self.time_out_buf[consecutive_success_mask] = True

    # ===================================== Stage Control Functions =================================

    def _stage_0_reward_condition(self):
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _stage_0_to_complete_condition(self):
        return self.box_robot_dist < self.walk_stand_thre

    def _stage_0_to_1_advance_condition(self):
        return self._stage_0_to_complete_condition()

    def _stage_1_reward_condition(self):
        return self._stage_0_to_1_advance_condition()

    def _stage_1_to_complete_condition(self):
        return self.consecutive_success_count >= self.max_consecutive_success_frames

    # ===================================== Class-specific Observation =================================

    def _get_obs_target_obj_transform(self):
        obj_rigid_pos = self.simulator.all_root_states[self.config.task.target_obj][:, :3]
        obj_rigid_rot = self.simulator.all_root_states[self.config.task.target_obj][:, 3:7]
        obj_relative_pos = quat_rotate_inverse(
            self.base_quat, obj_rigid_pos - self.simulator.robot_root_states[:, :3]
        )
        obj_relative_rot = quat_mul(quat_conjugate(self.base_quat), obj_rigid_rot)
        return torch.cat(
            [obj_relative_pos, quat_to_tan_norm(obj_relative_rot, w_last=True)], dim=-1
        )

    def _get_obs_palm_state(self):
        palm_rigid_pos = self.simulator._rigid_body_pos[:, self.palm_index]
        palm_rigid_rot = self.simulator._rigid_body_rot[:, self.palm_index]
        palm_relative_pos = quat_rotate_inverse(
            self.base_quat, palm_rigid_pos - self.simulator.robot_root_states[:, :3]
        )
        palm_relative_rot = quat_mul(quat_conjugate(self.base_quat), palm_rigid_rot)
        return torch.cat(
            [palm_relative_pos, quat_to_tan_norm(palm_relative_rot, w_last=True)], dim=-1
        )

    def _get_obs_target_obj_pos(self):
        target_pos = (
            self.simulator.env_origins[:, :3]
            + self.goal_pos[None, :]
            - self.simulator.robot_root_states[:, :3]
        )
        relative_target_pos = quat_rotate_inverse(self.base_quat, target_pos)
        return relative_target_pos

    def _get_obs_dof_pos_wo_finger(self):
        return self.simulator.dof_pos[:, :-14] - self.default_dof_pos[:, :-14]

    def _get_obs_dof_vel_wo_finger(self):
        return self.simulator.dof_vel[:, :-14]

    # ===================================== Stage-based reward =================================

    def _reward_penalty_ref_height(self):
        return torch.abs(self.simulator.robot_root_states[:, 2] - self.config.ref_height)

    def _reward_penalty_contact_with_table(self):
        contact_force = (
            torch.sum(
                torch.norm(self.simulator.table_to_hand_contact_forces.squeeze(1), dim=-1), dim=-1
            )
            > 1.0
        )
        return contact_force

    @StagedTaskBase.effective_in_stage([0, 1])
    def _reward_task_robot_distance_to_box(self):
        pos_reward = torch.ones(self.num_envs, device=self.device)
        stage_0_mask = self.stage_buf == 0
        if stage_0_mask.any():
            pos_error = torch.square(self.box_robot_dist - self.walk_stand_thre)
            pos_reward[stage_0_mask] = torch.exp(-0.5 * pos_error[stage_0_mask])
        # pos_error = torch.square(self.box_robot_dist - self.walk_stand_thre)
        # pos_reward = torch.exp(self.config.rewards.robot_distance_to_box_exp_coeff * pos_error)
        return pos_reward

    def _reward_penalty_upfront_heading(self):
        box_to_robot = (
            self.simulator.all_root_states[self.config.task.target_obj][:, :2]
            - self.simulator.robot_root_states[:, :2]
        )
        box_yaw = torch.atan2(box_to_robot[:, 1], box_to_robot[:, 0])
        forward_vec = quat_rotate(self.base_quat, self.forward_vec)
        robot_yaw = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])
        face_reward = ((box_yaw - robot_yaw) / torch.pi).square()
        return face_reward

    @StagedTaskBase.effective_in_stage([0, 1])
    def _reward_task_toward_box(self):
        vel_reward = torch.ones(self.num_envs, device=self.device)
        stage_0_mask = self.stage_buf == 0
        if stage_0_mask.any():
            lin_vel = self.base_lin_vel[:, :2]
            tar_dir = (
                self.simulator.all_root_states[self.config.task.target_obj][:, :2]
                - self.simulator.robot_root_states[:, :2]
            )
            tar_dir = tar_dir / (tar_dir.norm(dim=-1, keepdim=True) + 1e-6)
            tar_dir_speed = torch.sum(lin_vel * tar_dir, dim=-1)
            stage_0_reward = torch.exp(-5.0 * torch.square(0.5 - tar_dir_speed))
            vel_reward[stage_0_mask] = stage_0_reward[stage_0_mask]
        return vel_reward

    @StagedTaskBase.effective_in_stage([1])
    def _reward_standing_still(self):
        still_reward = torch.norm(self.base_lin_vel[:, :3], dim=1) <= 0.1
        return still_reward

    @StagedTaskBase.effective_in_stage([1])
    def _reward_penalty_still_dof_vel(self):
        reward = torch.sum(torch.square(self.simulator.dof_vel[:, :12]), dim=-1)
        return reward

    @property
    def ground_height(self):
        return 0.0

    def init_eval_metrics_tracking(self, device):
        """Initialize evaluation metrics tracking. Called at the start of evaluation."""
        self.eval_metrics = {
            # Overall buffers
            "reward_buffer": [],
            "length_buffer": [],
            "walk_stand_goal_reached_buffer": [],
            "dof_vel_buffer": [],
            "arm_dof_vel_buffer": [],
            "finger_dof_vel_buffer": [],
            # Current episode tracking
            "episode_walk_stand_goal_reached": torch.zeros(
                self.num_envs, dtype=torch.bool, device=device
            ),
            "cur_episode_dof_vel_sum": torch.zeros(
                self.num_envs, dtype=torch.float32, device=device
            ),
            "cur_episode_arm_dof_vel_sum": torch.zeros(
                self.num_envs, dtype=torch.float32, device=device
            ),
            "cur_episode_finger_dof_vel_sum": torch.zeros(
                self.num_envs, dtype=torch.float32, device=device
            ),
        }

        # Add standardized buffer names for compatibility with different trainers
        self.eval_metrics["goal_reached_buffer"] = self.eval_metrics[
            "walk_stand_goal_reached_buffer"
        ]

    def update_eval_metrics_per_step(self, infos):
        """Update evaluation metrics for each simulation step. Called after each env.step()."""
        if not hasattr(self, "eval_metrics"):
            return

        # Accumulate DOF velocity for the current episode (similar to reward accumulation)
        if hasattr(self, "simulator") and hasattr(self.simulator, "dof_vel"):
            # Calculate mean absolute DOF velocity for this timestep, similar to training
            timestep_dof_vel = (
                torch.sum(torch.abs(self.simulator.dof_vel), dim=1)
                / self.simulator.dof_vel.shape[1]
            )
            self.eval_metrics["cur_episode_dof_vel_sum"] += timestep_dof_vel

            # Accumulate separate arm and finger DOF velocities if the environment supports it
            if hasattr(self, "right_arm_dof_indices") and len(self.right_arm_dof_indices) > 0:
                timestep_arm_dof_vel = torch.sum(
                    torch.abs(self.simulator.dof_vel[:, self.right_arm_dof_indices]), dim=1
                ) / len(self.right_arm_dof_indices)
                self.eval_metrics["cur_episode_arm_dof_vel_sum"] += timestep_arm_dof_vel

            # For finger DOF velocities, check if we have finger indices
            if hasattr(self, "right_finger_dof_idx") and len(self.right_finger_dof_idx) > 0:
                timestep_finger_dof_vel = torch.sum(
                    torch.abs(self.simulator.dof_vel[:, self.right_finger_dof_idx]), dim=1
                ) / len(self.right_finger_dof_idx)
                self.eval_metrics["cur_episode_finger_dof_vel_sum"] += timestep_finger_dof_vel

        # Update episode-level walk_stand_goal_reached using OR operation across all frames
        if "walk_stand_goal_reached" in infos:
            current_walk_stand_goal_reached = infos["walk_stand_goal_reached"].bool()
            self.eval_metrics["episode_walk_stand_goal_reached"] = (
                self.eval_metrics["episode_walk_stand_goal_reached"]
                | current_walk_stand_goal_reached
            )

    def process_eval_episode_completions(
        self, completed_env_ids, cur_reward_sum, cur_episode_length
    ):
        """Process completed episodes for evaluation metrics. Called when episodes complete."""
        if not hasattr(self, "eval_metrics") or len(completed_env_ids) == 0:
            return

        # Convert environment indices to CPU for processing
        env_ids_cpu = completed_env_ids[:, 0].cpu().numpy().tolist()
        rewards = cur_reward_sum[completed_env_ids][:, 0].cpu().numpy().tolist()
        lengths = cur_episode_length[completed_env_ids][:, 0].cpu().numpy().tolist()
        walk_stand_goal_reached = (
            self.eval_metrics["episode_walk_stand_goal_reached"][completed_env_ids][:, 0]
            .cpu()
            .numpy()
            .tolist()
        )

        # Add to overall buffers
        self.eval_metrics["reward_buffer"].extend(rewards)
        self.eval_metrics["length_buffer"].extend(lengths)
        self.eval_metrics["walk_stand_goal_reached_buffer"].extend(walk_stand_goal_reached)

        # Calculate and store average DOF velocity for completed episodes
        if len(completed_env_ids) > 0:
            # Calculate average DOF velocity over episode length for completed episodes
            completed_episode_lengths = cur_episode_length[completed_env_ids][:, 0].float()
            # Avoid division by zero
            valid_episodes = completed_episode_lengths > 0
            if valid_episodes.sum() > 0:
                valid_episode_ids = completed_env_ids[valid_episodes]

                # Overall DOF velocity
                avg_dof_vel_per_episode = (
                    (
                        self.eval_metrics["cur_episode_dof_vel_sum"][valid_episode_ids][:, 0]
                        / completed_episode_lengths[valid_episodes]
                    )
                    .cpu()
                    .numpy()
                )
                self.eval_metrics["dof_vel_buffer"].extend(avg_dof_vel_per_episode.tolist())

                # Arm DOF velocity
                avg_arm_dof_vel_per_episode = (
                    (
                        self.eval_metrics["cur_episode_arm_dof_vel_sum"][valid_episode_ids][:, 0]
                        / completed_episode_lengths[valid_episodes]
                    )
                    .cpu()
                    .numpy()
                )
                self.eval_metrics["arm_dof_vel_buffer"].extend(avg_arm_dof_vel_per_episode.tolist())

                # Finger DOF velocity
                avg_finger_dof_vel_per_episode = (
                    (
                        self.eval_metrics["cur_episode_finger_dof_vel_sum"][valid_episode_ids][:, 0]
                        / completed_episode_lengths[valid_episodes]
                    )
                    .cpu()
                    .numpy()
                )
                self.eval_metrics["finger_dof_vel_buffer"].extend(
                    avg_finger_dof_vel_per_episode.tolist()
                )

            # For episodes with zero length, add zeros
            invalid_episodes = ~valid_episodes
            if invalid_episodes.sum() > 0:
                num_invalid = invalid_episodes.sum().item()
                self.eval_metrics["dof_vel_buffer"].extend([0.0] * num_invalid)
                self.eval_metrics["arm_dof_vel_buffer"].extend([0.0] * num_invalid)
                self.eval_metrics["finger_dof_vel_buffer"].extend([0.0] * num_invalid)

    def reset_eval_episode_tracking(self, completed_env_ids):
        """Reset episode-level tracking for completed environments."""
        if not hasattr(self, "eval_metrics") or len(completed_env_ids) == 0:
            return

        # Reset episode-level walk_stand_goal_reached for completed episodes
        self.eval_metrics["episode_walk_stand_goal_reached"][completed_env_ids] = False

        # Reset DOF velocity accumulation for completed episodes
        self.eval_metrics["cur_episode_dof_vel_sum"][completed_env_ids] = 0
        self.eval_metrics["cur_episode_arm_dof_vel_sum"][completed_env_ids] = 0
        self.eval_metrics["cur_episode_finger_dof_vel_sum"][completed_env_ids] = 0

    def get_eval_metrics_summary(self):
        """Get final evaluation metrics summary. Called at the end of evaluation."""
        if not hasattr(self, "eval_metrics"):
            return {}


        metrics = self.eval_metrics

        # Basic metrics
        eval_dict = {
            "collected_episodes": len(metrics["walk_stand_goal_reached_buffer"]),
            "walk_stand_goal_reached_rate": (
                np.mean(metrics["walk_stand_goal_reached_buffer"])
                if len(metrics["walk_stand_goal_reached_buffer"]) > 0
                else 0.0
            ),
            "episode_length": (
                np.mean(metrics["length_buffer"]) if len(metrics["length_buffer"]) > 0 else 0.0
            ),
            "reward": (
                np.mean(metrics["reward_buffer"]) if len(metrics["reward_buffer"]) > 0 else 0.0
            ),
            "eval_dof_vel": (
                np.mean(metrics["dof_vel_buffer"]) if len(metrics["dof_vel_buffer"]) > 0 else 0.0
            ),
            "eval_arm_dof_vel": (
                np.mean(metrics["arm_dof_vel_buffer"])
                if len(metrics["arm_dof_vel_buffer"]) > 0
                else 0.0
            ),
            "eval_finger_dof_vel": (
                np.mean(metrics["finger_dof_vel_buffer"])
                if len(metrics["finger_dof_vel_buffer"]) > 0
                else 0.0
            ),
        }

        # Print overall summary metrics
        print(f"Collected {eval_dict['collected_episodes']} episodes")
        print(
            f"Average walk_stand_goal_reached rate: {eval_dict['walk_stand_goal_reached_rate']:.3f}"
        )
        print(f"Average episode length: {eval_dict['episode_length']:.1f}")
        print(f"Average reward: {eval_dict['reward']:.1f}")
        print(f"Average DOF velocity: {eval_dict['eval_dof_vel']:.3f}")
        print(f"Average arm DOF velocity: {eval_dict['eval_arm_dof_vel']:.3f}")
        print(f"Average finger DOF velocity: {eval_dict['eval_finger_dof_vel']:.3f}")

        return eval_dict
