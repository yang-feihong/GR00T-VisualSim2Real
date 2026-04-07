# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0



import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.utils.math import (
    axis_angle_from_quat,
)
from loguru import logger
from typing_extensions import override

from groot.rl.envs.base_task.finger_primitive_base import FingerPrimitiveBase
from groot.rl.envs.base_task.staged_task_base import StagedTaskBase
from groot.rl.envs.loco_manip.walk_table_stand import WalkTableStand
from groot.rl.isaac_utils.rotations import (
    quat_rotate_inverse,
    quat_to_tan_norm,
)
from groot.rl.utils.torch_utils import torch_rand_float


class WalkStandGrasp(WalkTableStand, FingerPrimitiveBase):
    def __init__(self, config, device):
        super().__init__(config, device)

        # Copied from StandGrasp.__init__
        self.right_thumb_tip_body_idx = self.simulator.body_names.index("right_hand_thumb_2_link")
        self.right_index_tip_body_idx = self.simulator.body_names.index("right_hand_index_1_link")
        self.right_middle_tip_body_idx = self.simulator.body_names.index("right_hand_middle_1_link")
        self.right_palm_body_idx = self.simulator.body_names.index("right_hand_palm_link")
        self.left_palm_body_idx = self.simulator.body_names.index("left_hand_palm_link")
        self.right_hand_indices = [
            self.simulator.body_names.index(link)
            for link in self.simulator.robot_config.right_hand_body_names
        ]
        self.left_hand_indices = []

        self.right_elbow_dof_idx = self.simulator.dof_names.index("right_elbow_joint")
        self.right_shoulder_roll_dof_idx = self.simulator.dof_names.index(
            "right_shoulder_roll_joint"
        )
        self.right_shoulder_pitch_dof_idx = self.simulator.dof_names.index(
            "right_shoulder_pitch_joint"
        )
        self.right_shoulder_yaw_dof_idx = self.simulator.dof_names.index("right_shoulder_yaw_joint")
        self.right_wrist_roll_dof_idx = self.simulator.dof_names.index("right_wrist_roll_joint")
        self.right_wrist_pitch_dof_idx = self.simulator.dof_names.index("right_wrist_pitch_joint")
        self.right_wrist_yaw_dof_idx = self.simulator.dof_names.index("right_wrist_yaw_joint")

        self.last_distance = torch.ones(self.num_envs, device=self.device)
        self.pelvis_idx = self.simulator.body_names.index("pelvis")

        # Define arm DOF indices (after individual indices are set)
        self.right_arm_dof_indices = [
            self.right_shoulder_pitch_dof_idx,
            self.right_shoulder_roll_dof_idx,
            self.right_shoulder_yaw_dof_idx,
            self.right_elbow_dof_idx,
            self.right_wrist_roll_dof_idx,
            self.right_wrist_pitch_dof_idx,
            self.right_wrist_yaw_dof_idx,
        ]

        # Goal reached rate tracking (similar to ManipBase) - tracks ALL episode completions
        from collections import deque

        self.grasp_goal_reached_deque = deque(
            maxlen=500
        )  # Track goal reached for last 500 episodes
        self.average_grasp_goal_reached_rate = 0.0
        self.skip_initial_goal_tracking = True  # Flag to prevent adding values during initial reset

        # finger primitive related
        self._left_p0 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.left.pos_0,
            device=self.device,
            requires_grad=False,
        )  # closed
        self._left_p1 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.left.pos_1,
            device=self.device,
            requires_grad=False,
        )  # open
        self._right_p0 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.right.pos_0,
            device=self.device,
            requires_grad=False,
        )  # closed
        self._right_p1 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.right.pos_1,
            device=self.device,
            requires_grad=False,
        )  # open
        self._left_mode = self.config.robot.finger_primitive.primitive_action_map.left.mode
        self._right_mode = self.config.robot.finger_primitive.primitive_action_map.right.mode
        self._left_hand_dof_idx = [
            self.dof_names.index(name)
            for name in self.config.robot.finger_primitive.primitive_action_map.left.dof_names
        ]
        self._right_hand_dof_idx = [
            self.dof_names.index(name)
            for name in self.config.robot.finger_primitive.primitive_action_map.right.dof_names
        ]
        self._upper_non_finger_dof_idx = [
            i
            for i in self.upper_dof_indices
            if i not in self._left_hand_dof_idx and i not in self._right_hand_dof_idx
        ]

        # Copied from StandGraspLift.__init__
        for i, body_expr in enumerate(
            self.simulator.object_to_hand_contact_sensor.cfg.filter_prim_paths_expr
        ):
            if "right_hand_thumb_2_link" in body_expr:
                self.right_thumb_tip_contact_sensor_idx = i
            elif "right_hand_index_1_link" in body_expr:
                self.right_index_tip_contact_sensor_idx = i
            elif "right_hand_middle_1_link" in body_expr:
                self.right_middle_tip_contact_sensor_idx = i
            elif "right_hand_palm_link" in body_expr:
                self.right_palm_contact_sensor_idx = i
        self.right_finger_tips_contact_sensor_idx = [
            self.right_thumb_tip_contact_sensor_idx,
            self.right_index_tip_contact_sensor_idx,
            self.right_middle_tip_contact_sensor_idx,
            self.right_palm_contact_sensor_idx,
        ]
        self.right_finger_dof_idx = [
            self.simulator.dof_names.index(dof_name)
            for dof_name in self.simulator.robot_config.right_hand_dof_names
        ]
        self.arm_dof_idx = [
            self.simulator.dof_names.index(dof_name)
            for dof_name in self.simulator.robot_config.arm_dof_names
        ]
        self.left_arm_dof_idx = [
            self.simulator.dof_names.index(dof_name)
            for dof_name in self.simulator.robot_config.left_arm_dof_names
        ]
        self.right_arm_dof_idx = [
            self.simulator.dof_names.index(dof_name)
            for dof_name in self.simulator.robot_config.right_arm_dof_names
        ]

        # Copied from StandGraspLiftGrab.__init__
        self.pregrasp_offset = torch.tensor([-0.04, -0.05, 0.0], device=self.device)[
            None, :
        ]  # no prior on the pregrasp offset

        # Initialize additional buffers that are needed
        self.num_total_task = 0
        self.num_total_success = 0

        self.last_policy_output = torch.zeros(
            self.num_envs,
            self._num_homie_commands + self._num_upper_dof,
            dtype=torch.float32,
            device=self.device,
        )
        self.policy_output = torch.zeros(
            self.num_envs,
            self._num_homie_commands + self._num_upper_dof,
            dtype=torch.float32,
            device=self.device,
        )

    def _init_buffers(self):
        super()._init_buffers()

        # Copied from StandGrasp._init_buffers
        if self.config.get("show_rgb_image", False):
            self.visualize_env_id = torch.randint(0, self.num_envs, (1,), device=self.device)

        if hasattr(self.simulator, "task_root_origin"):
            # Get table_grid base position
            if "table_grid" in self.simulator.task_root_origin:
                self.base_table_pos = self.simulator.task_root_origin["table_grid"][:3]
            else:
                raise ValueError("table_grid not found in task_root_origin")

        #     # Get random_object base position
        #     if 'random_object' in self.simulator.task_root_origin:
        #         self.initial_object_pos = self.simulator.task_root_origin['random_object'][:3]
        #     else:
        #         raise ValueError(f"random_object not found in task_root_origin")
        # else:
        #     raise ValueError(f"task_root_origin not found")
        # # Initialize table_grid root state buffer
        # self.table_grid_root_state_buf = torch.zeros(self.num_envs, 13, device=self.device, requires_grad=False)
        # self.table_grid_root_state_buf[:, 3] = 1.0  # w
        # self.table_grid_root_state_buf[:, :3] = self.base_table_pos[None, :] + self.env_origins

        # self.bottle_root_state_buf = torch.zeros(self.num_envs, 13, device=self.device, requires_grad=False)
        # self.bottle_root_state_buf[:, 3] = 1.0  # w
        # self.bottle_root_state_buf[:, :3] = self.initial_object_pos[None, :] + self.env_origins

        # DOF velocity tracking buffers for logging
        self.average_dof_vel = 0.0
        self.average_arm_dof_vel = 0.0
        self.average_finger_dof_vel = 0.0
        self.last_episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.last_episode_dof_vel_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.last_episode_arm_dof_vel_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.last_episode_finger_dof_vel_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.num_compute_average_epl = getattr(self.config, "num_compute_average_epl", 200)

        # Copied from StandGraspLift._init_buffers
        self.lift_goal_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_lift_distance = torch.ones(self.num_envs, device=self.device)

        # Initialize goal reached buffer for grasp tasks (separate from walk_stand goal_reached)
        self.grasp_goal_reached_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Initialize target position buffer
        self.target_pos_buf = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

    def _apply_force_in_physics_step(self):
        # Explicitly use FingerPrimitiveBase's implementation
        return FingerPrimitiveBase._apply_force_in_physics_step(self)

    def _reset_dofs(self, env_ids, target_state=None):
        # Copied and adapted from StandGrasp._reset_dofs
        self.target_robot_dof_state[env_ids, :, 0] = self.default_dof_pos + torch_rand_float(
            -0.05, 0.05, (len(env_ids), self.num_dof), device=str(self.device)
        )
        self.target_robot_dof_state[env_ids, self.right_shoulder_roll_dof_idx, 0] = (
            torch_rand_float(-1.5708, 0.05, (len(env_ids), 1), device=str(self.device)).squeeze(-1)
        )
        shoulder_pitch_plus_elbow = torch_rand_float(
            -1.5708, 0.0, (len(env_ids), 1), device=str(self.device)
        ).squeeze(-1)
        self.target_robot_dof_state[env_ids, self.right_elbow_dof_idx, 0] = torch_rand_float(
            -0.785398, 0.0, (len(env_ids), 1), device=str(self.device)
        ).squeeze(-1)
        self.target_robot_dof_state[env_ids, self.right_shoulder_pitch_dof_idx, 0] = (
            shoulder_pitch_plus_elbow
            - self.target_robot_dof_state[env_ids, self.right_elbow_dof_idx, 0]
        )
        self.target_robot_dof_state[env_ids, self.right_shoulder_yaw_dof_idx, 0] = torch_rand_float(
            -0.785398, 0.785398, (len(env_ids), 1), device=str(self.device)
        ).squeeze(-1)
        self.target_robot_dof_state[env_ids, self.right_shoulder_yaw_dof_idx, 0] = torch.where(
            (self.target_robot_dof_state[env_ids, self.right_shoulder_pitch_dof_idx, 0] > 0.1)
            & (self.target_robot_dof_state[env_ids, self.right_shoulder_yaw_dof_idx, 0] > 0),
            0.1 * self.target_robot_dof_state[env_ids, self.right_shoulder_yaw_dof_idx, 0],
            self.target_robot_dof_state[env_ids, self.right_shoulder_yaw_dof_idx, 0],
        )
        self.target_robot_dof_state[env_ids, self.right_wrist_roll_dof_idx, 0] = torch_rand_float(
            -0.785398, 0.785398, (len(env_ids), 1), device=str(self.device)
        ).squeeze(-1)
        self.target_robot_dof_state[env_ids, self.right_wrist_pitch_dof_idx, 0] = torch_rand_float(
            -0.785398, 0, (len(env_ids), 1), device=str(self.device)
        ).squeeze(-1)
        self.target_robot_dof_state[env_ids, self.right_wrist_yaw_dof_idx, 0] = torch_rand_float(
            -0.785398, 0.785398, (len(env_ids), 1), device=str(self.device)
        ).squeeze(-1)

        # Add finger DOF initialization from StandGraspLift._reset_dofs
        m, n = torch.meshgrid(env_ids, torch.tensor(self.right_finger_dof_idx, device=self.device))
        self.target_robot_dof_state[m, n, 0] = torch.where(
            torch.rand(len(env_ids), 1, device=str(self.device)) < 0.5,
            self._right_p1,
            self._right_p0,
        )
        self.target_robot_dof_state[env_ids, :, 1] = 0.0

    def _reset_root_states(self, env_ids, target_root_states=None):
        return WalkTableStand._reset_root_states(self, env_ids, target_root_states)

    def _update_timeout_buf(self):
        return StagedTaskBase._update_timeout_buf(self)

    def _check_termination(self):
        StagedTaskBase._check_termination(self)
        tgt_obj_pos = self.simulator.get_task_root_state(self.config.task.target_obj)[:, :3]
        moved_too_far = (
            self.simulator.get_task_root_state(self.config.task.target_obj)[:, :2]
            - self.bottle_root_state_buf[:, :2]
        ).norm(dim=-1) > 0.15
        moved_too_far &= self.stage_buf <= 5
        self.reset_buf |= (tgt_obj_pos[:, 2] < 0.3) | moved_too_far

        # DEBUG ONLY for teleop reset
        # self.reset_buf = torch.ones_like(self.reset_buf, dtype=torch.bool)

    # ============================ conditions ============================
    def _stage_0_reward_condition(self):
        return WalkTableStand._stage_0_reward_condition(self)

    def _stage_0_to_complete_condition(self):
        return WalkTableStand._stage_0_to_complete_condition(self)

    def _stage_0_to_1_advance_condition(self):
        return self._stage_0_to_complete_condition()

    def _stage_1_reward_condition(self):
        return WalkTableStand._stage_1_reward_condition(self)

    def _stage_1_to_complete_condition(self):
        # return WalkTableStand._stage_1_to_complete_condition(self)
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _stage_1_to_2_advance_condition(self):
        return self._stage_1_to_complete_condition()

    def _stage_2_reward_condition(self):
        # return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return self._stage_1_to_2_advance_condition()

    def _stage_2_to_3_advance_condition(self):
        palm_pos = self.simulator._rigid_body_pos[:, self.right_palm_body_idx, :]
        bottle_pos = self.simulator.get_task_root_state(self.config.task.target_obj)[:, :3]
        return palm_pos[:, 1] < bottle_pos[:, 1] - 0.1

    def _stage_3_reward_condition(self):
        return self._stage_2_to_3_advance_condition()

    def _stage_3_to_complete_condition(self):
        # Copied from StandGraspLiftGrab._stage_1_to_complete_condition (modified from StandGrasp)
        return self.last_distance < 0.05  # pregrasp distance, otherwise not finishing stage 3

    def _stage_3_to_4_advance_condition(self):
        return self._stage_3_to_complete_condition()

    def _stage_4_reward_condition(self):
        return self._stage_3_to_4_advance_condition()

    def _stage_4_to_5_advance_condition(self):
        return (
            self.simulator.object_to_hand_contact_forces[
                :, 0, self.right_finger_tips_contact_sensor_idx, :
            ]
            .norm(dim=-1)
            .mean(dim=-1)
            > 3.0
        )

    def _stage_5_reward_condition(self):
        return self._stage_4_to_5_advance_condition()

    def _stage_5_to_6_advance_condition(self):
        return (
            (
                self.simulator.object_to_hand_contact_forces[
                    :, 0, self.right_finger_tips_contact_sensor_idx, :
                ].norm(dim=-1)
                > 3.0
            ).sum(dim=-1)
            >= 3
        ) & (self.actual_time_in_stage_buf == self.config.max_stage_time[5] - 1)

    def _stage_6_reward_condition(self):
        return (
            self.simulator.object_to_hand_contact_forces[
                :, 0, self.right_finger_tips_contact_sensor_idx, :
            ].norm(dim=-1)
            > 3.0
        ).sum(dim=-1) >= 3

    def _stage_6_to_complete_condition(self):
        return self.last_lift_distance < 0.10

    # ============================ rewards ============================

    def _reward_task_robot_distance_to_box(self):
        pos_reward = torch.ones(self.num_envs, device=self.device)
        stage_01_mask = self.stage_buf <= 1
        if stage_01_mask.any():
            pos_error = torch.square(self.box_robot_dist - self.walk_stand_thre)
            pos_reward[stage_01_mask] = torch.exp(
                self.config.rewards.robot_distance_to_box_exp_coeff * pos_error[stage_01_mask]
            )
        # pos_error = torch.square(self.box_robot_dist - self.walk_stand_thre)
        # pos_reward = torch.exp(self.config.rewards.robot_distance_to_box_exp_coeff * pos_error)
        return pos_reward

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3])
    def _reward_bottle_in_middle(self):
        right_palm_pos = self.simulator._rigid_body_pos[:, self.right_palm_body_idx, :]
        left_palm_pos = self.simulator._rigid_body_pos[:, self.left_palm_body_idx, :]
        right_tip_pos = self.simulator._rigid_body_pos[:, self.right_index_tip_body_idx, :]
        right_thumb_tip_pos = self.simulator._rigid_body_pos[:, self.right_thumb_tip_body_idx, :]
        bottle_pos = self.simulator.get_task_root_state(self.config.task.target_obj)[:, :3]
        rew = (right_palm_pos[:, 1] > bottle_pos[:, 1] - 0.1) + (
            left_palm_pos[:, 1] < bottle_pos[:, 1] + 0.1
        )

        finger_pos_rew = (
            (right_palm_pos[:, 0] > bottle_pos[:, 0])
            + (right_tip_pos[:, 0] < bottle_pos[:, 0] + 0.05)
            + (right_thumb_tip_pos[:, 0] > bottle_pos[:, 0] - 0.02)
        )
        env_id = (self.stage_buf <= 2).nonzero(as_tuple=False).flatten()
        finger_pos_rew[env_id] = 3.0
        return rew + finger_pos_rew

    @StagedTaskBase.effective_in_stage([4, 5, 6])
    def _reward_grasp(self):
        return torch.clamp(
            self.simulator.object_to_hand_contact_forces[
                :, 0, self.right_finger_tips_contact_sensor_idx, :
            ]
            .norm(dim=-1)
            .mean(dim=-1),
            min=0.0,
            max=2.0,
        )

    @StagedTaskBase.effective_in_stage(6)
    def _reward_lift_velocity(self):
        target_direction = F.normalize(
            self.lift_goal_buf
            - self.simulator.get_task_root_state(self.config.task.target_obj)[:, :3],
            dim=-1,
        )
        obj_vel = self.simulator.get_task_root_state(self.config.task.target_obj)[:, 7:10]
        obj_vel_along_target_dir = torch.sum(obj_vel * target_direction, dim=-1)
        return obj_vel_along_target_dir

    @StagedTaskBase.effective_in_stage(6)
    def _reward_lift_distance(self):
        obj_pos = self.simulator.get_task_root_state(self.config.task.target_obj)[:, :3]
        lift_distance = torch.norm(self.lift_goal_buf - obj_pos, dim=-1)
        pos_reward = torch.exp(self.config.rewards.lift_distance_to_goal_exp_coeff * lift_distance)
        return pos_reward

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3])
    def _reward_penalty_keep_hand_open(self):
        return (
            self.processed_commands[:, -1] - self.config.rewards.finger_primitive_value
        ) ** 2  # -1 for right hand

    @StagedTaskBase.effective_in_stage([4, 5, 6])
    def _reward_penalty_keep_hand_closed(self):
        return (
            self.processed_commands[:, -1] - (-self.config.rewards.finger_primitive_value)
        ) ** 2  # -1 for right hand

    @StagedTaskBase.effective_in_stage([4, 5, 6])
    def _reward_positive_hand_closed(self):
        return torch.exp(
            self.config.rewards.positive_hand_closed_exp_coeff
            * torch.abs(
                self.processed_commands[:, -1] - (-self.config.rewards.finger_primitive_value)
            )
        )  # -1 for right hand

    def _reward_pregrasp_finger_dof_pos_l1(self):
        right_diff = self.simulator.dof_pos[:, self._right_hand_dof_idx] - self._right_p1
        pos_track = self._tracking_reward_util(
            right_diff,
            std=self.config.rewards.finger_dof_pos_l1_std,
            target=0.0,
            scale=1.0,
            offset=0.0,
        ).mean(dim=-1)
        stage_id = (self.stage_buf >= 3).nonzero(as_tuple=False).flatten()
        pos_track[stage_id] = 1.0
        return (pos_track).clamp(max=1.0)

    @StagedTaskBase.effective_in_stage([4, 5, 6])
    def _reward_grasp_finger_dof_pos_l1(self):
        right_diff = self.simulator.dof_pos[:, self._right_hand_dof_idx] - self._right_p0

        pos_track = self._tracking_reward_util(
            right_diff,
            std=self.config.rewards.finger_dof_pos_l1_std,
            target=0.0,
            scale=1.0,
            offset=0.0,
        ).mean(dim=-1)

        return (pos_track).clamp(max=1.0)

    def _reward_penalty_finger_primitive_limit(self):
        # penalize if finger primitive is out of [-1, 1] for linear mode
        # [-primitive_value, primitive_value] for discrete mode
        if self._right_mode == "linear":
            lower_limit = -1.0 / self.config.robot.control.action_scale
            upper_limit = 1.0 / self.config.robot.control.action_scale
        elif self._right_mode == "discrete":
            lower_limit = -self.config.rewards.finger_primitive_value
            upper_limit = self.config.rewards.finger_primitive_value

        out_of_limits = -(self.processed_commands[:, -1] - lower_limit).clip(max=0.0) + (
            self.processed_commands[:, -1] - upper_limit
        ).clip(min=0.0)
        return out_of_limits

    @StagedTaskBase.effective_in_stage([2, 3])
    def _reward_penalty_hand_object_contact(self):
        return self.simulator.object_to_hand_contact_forces[:, 0, :, :].norm(dim=-1).mean(dim=-1)

    @StagedTaskBase.effective_in_stage([2, 3, 4, 5, 6])
    def _reward_hand_object_velocity(self):
        palm_vel = self.simulator._rigid_body_vel[:, self.right_palm_body_idx, :]
        target_dir = F.normalize(self.target_pos_buf, dim=-1)
        palm_vel_along_target_dir = torch.sum(palm_vel * target_dir, dim=-1).clamp(max=1.0)
        full_env_id = (self.stage_buf >= 5).nonzero(as_tuple=False).flatten()
        palm_vel_along_target_dir[full_env_id] = 1.0
        return palm_vel_along_target_dir

    @StagedTaskBase.effective_in_stage([2, 3, 4, 5, 6])
    def _reward_hand_object_distance(self):
        pos_reward = torch.exp(
            self.config.rewards.hand_distance_to_object_exp_coeff * self.last_distance
        )
        full_env_id = (self.stage_buf >= 5).nonzero(as_tuple=False).flatten()
        pos_reward[full_env_id] = 1.0
        return pos_reward

    @StagedTaskBase.effective_in_stage([1, 2, 3, 4, 5, 6])
    def _reward_penalty_still_dof_vel(self):
        reward = torch.sum(torch.square(self.simulator.dof_vel[:, :12]), dim=-1)
        return reward

    # Add rewards to penalize upper body actions during walking stage
    @StagedTaskBase.effective_in_stage([0, 1])
    def _reward_penalty_upper_body_actions(self):
        return torch.sum(torch.square(self.simulator.dof_pos[:, self.right_arm_dof_idx]), dim=-1)

    def _reward_alive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_penalty_ref_height(self):
        return torch.abs(self.simulator.robot_root_states[:, 2] - self.config.ref_height)

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3, 4])
    def _reward_hand_orientation(self):
        return axis_angle_from_quat(self.simulator.right_hand_transform_rot[:, 0, :]).norm(dim=-1)

    # @StagedTaskBase.effective_in_stage([0, 1, 2, 3, 4, 5, 6])
    def _reward_penalty_humanly_dof_limit(self):
        zeros = torch.zeros_like(self.simulator.dof_pos[:, self.right_shoulder_yaw_dof_idx])
        # right should yaw should be between -70 deg and 100 deg
        shoulder_yaw_penalty = torch.maximum(
            -1.22173 - self.simulator.dof_pos[:, self.right_shoulder_yaw_dof_idx], zeros
        )
        shoulder_yaw_penalty += torch.maximum(
            self.simulator.dof_pos[:, self.right_shoulder_yaw_dof_idx] - 1.7453, zeros
        )
        # elbow should not go over 90 deg
        elbow_penalty = torch.maximum(
            self.simulator.dof_pos[:, self.right_elbow_dof_idx] - 1.5708, zeros
        )
        return shoulder_yaw_penalty + elbow_penalty

    def _reward_penalty_root_pos_deviation(self):
        return (self.simulator.robot_root_states[:, :2] - self.env_origins[:, :2]).norm(dim=-1)

    def _reward_penalty_hand_table_contact(self):
        return (
            (
                self.simulator.contact_forces[
                    :, self.right_hand_indices + self.left_hand_indices, :
                ].norm(dim=-1)
                > 1
            )
            .any(dim=-1)
            .float()
        )

    def _reward_penalty_root_vel_deviation(self):
        return self.simulator.robot_root_states[:, 7:].norm(dim=-1)

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3, 4])
    def _reward_penalty_bottle_movement(self):
        bottle_height = self.simulator.get_task_root_state(self.config.task.target_obj)[:, 2]
        rew = (
            (
                self.simulator.get_task_root_state(self.config.task.target_obj)[:, :2]
                - self.bottle_root_state_buf[:, :2]
            )
            ** 2
        ).sum(dim=-1)
        stage_id = (self.stage_buf <= 3).nonzero(as_tuple=False).flatten()
        rew[stage_id] = 1.0

        return rew

    # @StagedTaskBase.effective_in_stage([2, 3, 4, 5, 6])
    def _reward_penalty_dof_overspeed(self):
        # TODO need to fix this according to right / left, now use default right
        return (
            torch.maximum(
                torch.abs(self.simulator.dof_vel[:, self.right_arm_dof_idx]) - 1.0,
                torch.zeros_like(self.simulator.dof_vel[:, self.right_arm_dof_idx]),
            )
            ** 2
        ).mean(dim=-1)

    def _reward_penalty_dof_vel_finger(self):
        return torch.sum(self.simulator.dof_vel[:, self.right_finger_dof_idx] ** 2, dim=-1)

    def _reward_penalty_action_rate_finger(self):
        # -1 for right hand
        return torch.square(self.last_policy_output[:, -1] - self.policy_output[:, -1])

    @override
    def _reward_limits_dof_pos(self):
        # Penalize dof positions too close to the limit - from StandGraspLift with arm_dof_idx
        if self.use_reward_limits_dof_pos_curriculum:
            m = (
                self.simulator.hard_dof_pos_limits[:, 0] + self.simulator.hard_dof_pos_limits[:, 1]
            ) / 2
            r = self.simulator.hard_dof_pos_limits[:, 1] - self.simulator.hard_dof_pos_limits[:, 0]
            lower_soft_limit = m - 0.5 * r * self.soft_dof_pos_curriculum_value
            upper_soft_limit = m + 0.5 * r * self.soft_dof_pos_curriculum_value
        else:
            lower_soft_limit = self.simulator.dof_pos_limits[:, 0]
            upper_soft_limit = self.simulator.dof_pos_limits[:, 1]
        out_of_limits = -(self.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
        out_of_limits += (self.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
        return torch.sum(out_of_limits[:, self.arm_dof_idx], dim=1)

    @StagedTaskBase.effective_in_stage([2, 3, 4, 5, 6])
    def _reward_penalty_hand_table_contact(self):
        # Override from StandGraspLift
        return (
            (self.simulator.table_to_hand_contact_forces[:, 0, :, :].norm(dim=-1) > 1)
            .any(dim=-1)
            .float()
        )

    def _reward_walk_to_table(self):
        target_dir = F.normalize(
            self.simulator.all_root_states[self.config.task.target_obj][:, :3]
            - self.simulator.robot_root_states[:, :3],
            dim=-1,
        )
        current_root_vel = self.simulator.robot_root_states[:, 7:10]

        stage_0_mask = (self.stage_buf == 0).unsqueeze(-1)
        target_vel_stage_0 = self.config.get("target_root_vel", 0.3) * target_dir
        target_vel_other = torch.zeros_like(current_root_vel)
        target_vel = torch.where(stage_0_mask, target_vel_stage_0, target_vel_other)
        return self._tracking_reward_util(
            torch.linalg.norm(current_root_vel - target_vel, dim=-1),
            std=self.config.rewards.walk_to_table_reward_std,
            target=0.0,
            scale=1.0,
            offset=0.0,
        )

    @StagedTaskBase.effective_in_stage([2, 3, 4, 5, 6])
    def _reward_tracking_grasp_target_distance(self):
        hand_object_dist = self.last_distance
        return self._tracking_reward_util(
            hand_object_dist,
            std=self.config.rewards.tracking_grasp_target_std,
            target=0.0,
            scale=1.0,
            offset=0.0,
        )

    # ============================ observation methods ============================
    def _get_obs_hand_object_transform(self):
        translation = self.simulator.right_hand_transform_pos[:, 0, :3]
        rotation_xyzw = self.simulator.right_hand_transform_rot[:, 0, [1, 2, 3, 0]]
        rotation_6d = quat_to_tan_norm(rotation_xyzw, w_last=True)
        return torch.cat([translation, rotation_6d], dim=-1)

    def _get_obs_target_obj_pos(self):
        obj_pos_in_world_frame = (
            self.simulator.get_task_root_state(self.config.task.target_obj)[:, :3]
            - self.simulator._rigid_body_pos[:, self.pelvis_idx, :]
        )
        obj_pos_in_base_frame = quat_rotate_inverse(
            self.base_quat, obj_pos_in_world_frame, w_last=True
        )
        return obj_pos_in_base_frame

    def _get_obs_placement_pos(self):
        return self.bottle_root_state_buf[:, :2] - self.env_origins[:, :2]

    def _get_obs_target_lift_pos(self):
        return quat_rotate_inverse(
            self.base_quat,
            self.lift_goal_buf - self.simulator._rigid_body_pos[:, self.pelvis_idx, :],
            w_last=True,
        )

    def _get_obs_finger_tips_force(self):
        return self.simulator.object_to_hand_contact_forces[
            :, 0, self.right_finger_tips_contact_sensor_idx, :
        ].reshape(self.num_envs, -1)

    def _get_obs_dof_pos_partial(self):
        # right finger dof pos
        return self.simulator.dof_pos[:, self.right_finger_dof_idx]

    def _get_obs_dof_vel_partial(self):
        # right finger dof vel
        return self.simulator.dof_vel[:, self.right_finger_dof_idx]

    def _get_obs_dof_pos(self):
        return self.simulator.dof_pos

    def _get_obs_dof_vel(self):
        return self.simulator.dof_vel

    def _get_obs_random_object_idx_one_hot(self):
        return self.simulator.random_object_idx_one_hot

    # ============================ callback methods ============================
    def _pre_compute_observations_callback(self, env_ids=None):
        self.target_pos_buf[:] = self.simulator.get_task_root_state(self.config.task.target_obj)[
            :, :3
        ]
        palm_pos = self.simulator._rigid_body_pos[:, self.right_palm_body_idx, :]

        self.target_pos_buf[:, 0] = torch.where(
            self.stage_buf == 2, self.target_pos_buf[:, 0] - 0.05, self.target_pos_buf[:, 0]
        )  # stage 2 = grasp stage 0
        self.target_pos_buf[:, 1] = torch.where(
            self.stage_buf == 2, self.target_pos_buf[:, 1] - 0.15, self.target_pos_buf[:, 1]
        )
        # self.simulator.draw_spheres_batch(self.target_pos_buf)
        self.target_pos_buf[:] = self.target_pos_buf - palm_pos + self.pregrasp_offset

        return super()._pre_compute_observations_callback(env_ids)

    def _post_compute_observations_callback(self):
        this_distance = self.target_pos_buf.norm(dim=-1)
        # Accumulate average absolute DOF velocities per timestep
        self.last_episode_dof_vel_buf += (
            torch.sum(torch.abs(self.simulator.dof_vel), dim=1) / self.num_dof
        )

        # Accumulate separate arm and finger DOF velocities
        if hasattr(self, "right_arm_dof_indices") and len(self.right_arm_dof_indices) > 0:
            self.last_episode_arm_dof_vel_buf += torch.sum(
                torch.abs(self.simulator.dof_vel[:, self.right_arm_dof_indices]), dim=1
            ) / len(self.right_arm_dof_indices)

        if hasattr(self, "right_finger_dof_idx") and len(self.right_finger_dof_idx) > 0:
            self.last_episode_finger_dof_vel_buf += torch.sum(
                torch.abs(self.simulator.dof_vel[:, self.right_finger_dof_idx]), dim=1
            ) / len(self.right_finger_dof_idx)

        self.last_episode_length_buf = self.episode_length_buf.clone()

        self.last_distance = this_distance

        # Update grasp goal reached (separate from walk_stand goal_reached)
        self.grasp_goal_reached_buf |= self.current_last_stage_completed_task_buf
        self.extras["grasp_goal_reached"] = self.grasp_goal_reached_buf.clone()

        # Handle periodic task reset
        reset_task_mask = (self.episode_length_buf % self.config.reset_task_every_n_steps == 0) & (
            self.episode_length_buf > 0
        )
        if reset_task_mask.any():
            env_ids = torch.where(reset_task_mask)[0]
            self.num_total_success += self.current_last_stage_completed_task_buf[env_ids].sum()
            self.num_total_task += len(env_ids)
            self._reset_task_states(env_ids)
            self.set_to_stage(
                env_ids, torch.zeros(len(env_ids), device=self.device, dtype=torch.long)
            )

        # Add recent goal reached rate to extras for logging
        if len(self.grasp_goal_reached_deque) > 0:
            recent_grasp_goal_reached_rate = sum(self.grasp_goal_reached_deque) / len(
                self.grasp_goal_reached_deque
            )
            self.extras["episode"][
                "recent_grasp_goal_reached_rate"
            ] = recent_grasp_goal_reached_rate
        else:
            self.extras["episode"]["recent_grasp_goal_reached_rate"] = 0.0

        # Add average DOF velocities to extras for wandb logging
        self.extras["episode"]["average_dof_vel"] = torch.tensor(
            self.average_dof_vel, dtype=torch.float, device=self.device
        )
        self.extras["episode"]["average_arm_dof_vel"] = torch.tensor(
            self.average_arm_dof_vel, dtype=torch.float, device=self.device
        )
        self.extras["episode"]["average_finger_dof_vel"] = torch.tensor(
            self.average_finger_dof_vel, dtype=torch.float, device=self.device
        )

        # Update lift distance tracking from StandGraspLift._post_compute_observations_callback
        this_lift_distance = (
            self.lift_goal_buf
            - self.simulator.get_task_root_state(self.config.task.target_obj)[:, :3]
        ).norm(dim=-1)
        self.last_lift_distance = this_lift_distance

        if self.config.get("enforce_vis", False):
            self._draw_goal_position()

        return super()._post_compute_observations_callback()

    # ============================ reset and task management methods ============================
    def _reset_buffers_callback(self, env_ids, target_buf=None):
        if len(env_ids) > 0:
            self._update_average_grasp_goal_reached_rate(env_ids)
            self._update_average_dof_vel(env_ids)

        # Call parent method
        super()._reset_buffers_callback(env_ids, target_buf)

    def _update_average_grasp_goal_reached_rate(self, env_ids):
        if self.skip_initial_goal_tracking:
            # Only skip if this is the initial reset with all environments
            if len(env_ids) == self.num_envs and torch.equal(
                env_ids, torch.arange(self.num_envs, device=self.device)
            ):
                # This is the initial reset, skip adding to deque but still update the average
                num = len(env_ids)
                current_average_grasp_goal_reached_rate = torch.mean(
                    self.grasp_goal_reached_buf[env_ids], dtype=torch.float
                )
                self.average_grasp_goal_reached_rate = self.average_grasp_goal_reached_rate * (
                    1 - num / self.num_compute_average_epl
                ) + current_average_grasp_goal_reached_rate * (num / self.num_compute_average_epl)

                # Set flag to False after initial reset
                self.skip_initial_goal_tracking = False
                return

        # Normal behavior for subsequent resets
        num = len(env_ids)
        current_average_grasp_goal_reached_rate = torch.mean(
            self.grasp_goal_reached_buf[env_ids], dtype=torch.float
        )

        self.average_grasp_goal_reached_rate = self.average_grasp_goal_reached_rate * (
            1 - num / self.num_compute_average_epl
        ) + current_average_grasp_goal_reached_rate * (num / self.num_compute_average_epl)
        # Only add to deque for non-initial resets
        self.grasp_goal_reached_deque.extend(
            self.grasp_goal_reached_buf[env_ids][:].cpu().numpy().tolist()
        )

    def _update_average_dof_vel(self, env_ids):
        num = len(env_ids)
        # Calculate average DOF velocity over the episode for the environments being reset
        episode_lengths = self.last_episode_length_buf[env_ids].float()
        valid_episodes = episode_lengths > 0

        if valid_episodes.sum() > 0:
            # Only calculate for episodes that actually ran
            valid_env_ids = env_ids[valid_episodes]
            valid_num = len(valid_env_ids)

            # Overall DOF velocity
            current_average_dof_vel = torch.mean(
                self.last_episode_dof_vel_buf[valid_env_ids]
                / self.last_episode_length_buf[valid_env_ids].float(),
                dtype=torch.float,
            )
            self.average_dof_vel = self.average_dof_vel * (
                1 - valid_num / self.num_compute_average_epl
            ) + current_average_dof_vel * (valid_num / self.num_compute_average_epl)

            # Arm DOF velocity
            if hasattr(self, "right_arm_dof_indices") and len(self.right_arm_dof_indices) > 0:
                current_average_arm_dof_vel = torch.mean(
                    self.last_episode_arm_dof_vel_buf[valid_env_ids]
                    / self.last_episode_length_buf[valid_env_ids].float(),
                    dtype=torch.float,
                )
                self.average_arm_dof_vel = self.average_arm_dof_vel * (
                    1 - valid_num / self.num_compute_average_epl
                ) + current_average_arm_dof_vel * (valid_num / self.num_compute_average_epl)

            # Finger DOF velocity
            if hasattr(self, "right_finger_dof_idx") and len(self.right_finger_dof_idx) > 0:
                current_average_finger_dof_vel = torch.mean(
                    self.last_episode_finger_dof_vel_buf[valid_env_ids]
                    / self.last_episode_length_buf[valid_env_ids].float(),
                    dtype=torch.float,
                )
                self.average_finger_dof_vel = self.average_finger_dof_vel * (
                    1 - valid_num / self.num_compute_average_epl
                ) + current_average_finger_dof_vel * (valid_num / self.num_compute_average_epl)

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
        self.last_distance[env_ids] = torch.ones(len(env_ids), device=self.device)
        self.grasp_goal_reached_buf[env_ids] = False

        # Reset lift goal from StandGraspLift._reset_tasks_callback
        self.lift_goal_buf[env_ids, 0] = (
            torch_rand_float(
                *self.config.bottle_pos_range_x, (len(env_ids), 1), device=str(self.device)
            ).squeeze(-1)
            + self.env_origins[env_ids, 0]
            + self.initial_object_pos[0]
        )
        self.lift_goal_buf[env_ids, 1] = (
            torch_rand_float(
                *self.config.bottle_pos_range_y, (len(env_ids), 1), device=str(self.device)
            ).squeeze(-1)
            + self.env_origins[env_ids, 1]
            + self.initial_object_pos[1]
        )
        self.lift_goal_buf[env_ids, 2] = (
            torch_rand_float(
                *self.config.bottle_pos_range_z, (len(env_ids), 1), device=str(self.device)
            ).squeeze(-1)
            + self.env_origins[env_ids, 2]
        )
        self.last_lift_distance[env_ids] = (
            self.lift_goal_buf[env_ids, :] - self.bottle_root_state_buf[env_ids, :3]
        ).norm(dim=-1)

        # Reset DOF velocity tracking buffers
        self.last_episode_length_buf[env_ids] = 0
        self.last_episode_dof_vel_buf[env_ids] = 0.0
        self.last_episode_arm_dof_vel_buf[env_ids] = 0.0
        self.last_episode_finger_dof_vel_buf[env_ids] = 0.0

    def _reset_task_states(self, env_ids):
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
        self.simulator.set_task_root_state_tensor(env_ids, task_root_state_dict)

    def _draw_goal_position(self):
        self.simulator.draw_spheres_batch(self.lift_goal_buf)

    @property
    def ground_height(self):
        return 0.0

    def get_object_class_names(self, env_ids=None):
        return self.simulator.get_object_class_names(env_ids)

    # ============================ evaluation methods ============================
    def init_eval_metrics_tracking(self, device):
        self.eval_metrics = {
            # Overall buffers
            "reward_buffer": [],
            "length_buffer": [],
            "goal_reached_buffer": [],
            "dof_vel_buffer": [],
            "arm_dof_vel_buffer": [],
            "finger_dof_vel_buffer": [],
            # Class-wise tracking
            "class_wise_stats": {},  # {class_name: {'goal_reached': [], 'rewards': [], 'lengths': []}}
            "object_class_buffer": [],  # Track which object class each completed episode used
            # Current episode tracking
            "episode_goal_reached": torch.zeros(self.num_envs, dtype=torch.bool, device=device),
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

        # Get object class names for each environment (this stays constant during evaluation)
        self.eval_env_object_classes = self.get_object_class_names()
        if self.eval_env_object_classes is not None:
            logger.info(
                f"Evaluation will track class-wise success rates for {len(set(self.eval_env_object_classes))} unique object classes"
            )

    def update_eval_metrics_per_step(self, infos):
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

            # For finger DOF velocities, check if we're in StandGraspLift class which has finger indices
            if hasattr(self, "right_finger_dof_idx") and len(self.right_finger_dof_idx) > 0:
                timestep_finger_dof_vel = torch.sum(
                    torch.abs(self.simulator.dof_vel[:, self.right_finger_dof_idx]), dim=1
                ) / len(self.right_finger_dof_idx)
                self.eval_metrics["cur_episode_finger_dof_vel_sum"] += timestep_finger_dof_vel

        # Update episode-level goal_reached using OR operation across all frames
        # Use grasp_goal_reached instead of goal_reached to differentiate from walk_stand task
        if "grasp_goal_reached" in infos:
            current_goal_reached = infos["grasp_goal_reached"].bool()
            self.eval_metrics["episode_goal_reached"] = (
                self.eval_metrics["episode_goal_reached"] | current_goal_reached
            )

    def process_eval_episode_completions(
        self, completed_env_ids, cur_reward_sum, cur_episode_length
    ):
        if not hasattr(self, "eval_metrics") or len(completed_env_ids) == 0:
            return

        # Convert environment indices to CPU for processing
        env_ids_cpu = completed_env_ids[:, 0].cpu().numpy().tolist()
        rewards = cur_reward_sum[completed_env_ids][:, 0].cpu().numpy().tolist()
        lengths = cur_episode_length[completed_env_ids][:, 0].cpu().numpy().tolist()
        goal_reached = (
            self.eval_metrics["episode_goal_reached"][completed_env_ids][:, 0]
            .cpu()
            .numpy()
            .tolist()
        )

        # Add to overall buffers
        self.eval_metrics["reward_buffer"].extend(rewards)
        self.eval_metrics["length_buffer"].extend(lengths)
        self.eval_metrics["goal_reached_buffer"].extend(goal_reached)

        # Track class-wise statistics
        if self.eval_env_object_classes is not None:
            for i, env_id in enumerate(env_ids_cpu):
                class_name = self.eval_env_object_classes[env_id]
                self.eval_metrics["object_class_buffer"].append(class_name)

                # Initialize class stats if not exists
                if class_name not in self.eval_metrics["class_wise_stats"]:
                    self.eval_metrics["class_wise_stats"][class_name] = {
                        "goal_reached": [],
                        "rewards": [],
                        "lengths": [],
                    }

                # Add episode data to class-specific buffers
                self.eval_metrics["class_wise_stats"][class_name]["goal_reached"].append(
                    goal_reached[i]
                )
                self.eval_metrics["class_wise_stats"][class_name]["rewards"].append(rewards[i])
                self.eval_metrics["class_wise_stats"][class_name]["lengths"].append(lengths[i])

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
        if not hasattr(self, "eval_metrics") or len(completed_env_ids) == 0:
            return

        # Reset episode-level goal_reached for completed episodes
        self.eval_metrics["episode_goal_reached"][completed_env_ids] = False

        # Reset DOF velocity accumulation for completed episodes
        self.eval_metrics["cur_episode_dof_vel_sum"][completed_env_ids] = 0
        self.eval_metrics["cur_episode_arm_dof_vel_sum"][completed_env_ids] = 0
        self.eval_metrics["cur_episode_finger_dof_vel_sum"][completed_env_ids] = 0

    def get_eval_metrics_summary(self):
        if not hasattr(self, "eval_metrics"):
            return {}


        metrics = self.eval_metrics

        # Basic metrics
        eval_dict = {
            "collected_episodes": len(metrics["goal_reached_buffer"]),
            "goal_reached_rate": np.mean(metrics["goal_reached_buffer"]),
            "episode_length": np.mean(metrics["length_buffer"]),
            "reward": np.mean(metrics["reward_buffer"]),
            "eval_dof_vel": np.mean(metrics["dof_vel_buffer"]),
            "eval_arm_dof_vel": np.mean(metrics["arm_dof_vel_buffer"]),
            "eval_finger_dof_vel": np.mean(metrics["finger_dof_vel_buffer"]),
        }

        # Print class-wise statistics
        if metrics["class_wise_stats"]:
            print("\n" + "=" * 50)
            print("CLASS-WISE SUCCESS RATES:")
            print("=" * 50)

            # Calculate and display class-wise metrics
            class_wise_results = {}
            overall_class_episodes = 0

            for class_name, stats in metrics["class_wise_stats"].items():
                if len(stats["goal_reached"]) > 0:
                    success_rate = np.mean(stats["goal_reached"])
                    avg_reward = np.mean(stats["rewards"])
                    avg_length = np.mean(stats["lengths"])
                    num_episodes = len(stats["goal_reached"])
                    overall_class_episodes += num_episodes

                    print(
                        f"{class_name:12s}: {success_rate:.3f} success rate ({num_episodes:3d} episodes), "
                        f"reward: {avg_reward:.1f}, length: {avg_length:.1f}"
                    )

                    class_wise_results[f"class_{class_name}_success_rate"] = success_rate
                    class_wise_results[f"class_{class_name}_reward"] = avg_reward
                    class_wise_results[f"class_{class_name}_length"] = avg_length
                    class_wise_results[f"class_{class_name}_episodes"] = num_episodes

            for class_name, stats in metrics["class_wise_stats"].items():
                print(
                    f"{class_name}_success_rate: {class_wise_results[f'class_{class_name}_success_rate']:.3f};"
                )

            print("=" * 50)
            print(f"Total episodes with class info: {overall_class_episodes}")

            # Calculate weighted average success rate (should match overall rate)
            if overall_class_episodes > 0:
                weighted_success_rate = (
                    sum(
                        np.mean(stats["goal_reached"]) * len(stats["goal_reached"])
                        for stats in metrics["class_wise_stats"].values()
                    )
                    / overall_class_episodes
                )
                print(f"Weighted average success rate: {weighted_success_rate:.3f}")

            # Add class-wise metrics to eval_dict
            eval_dict.update(class_wise_results)

        return eval_dict
