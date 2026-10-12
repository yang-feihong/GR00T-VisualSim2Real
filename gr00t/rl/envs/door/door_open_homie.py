# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
import omni.usd
import torch
import torch.nn.functional as F
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.utils.math import (
    axis_angle_from_quat,
    euler_xyz_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
    wrap_to_pi,
)
from pxr import Usd
from typing_extensions import override

from gr00t.rl.envs.base_task.b2z1_command_base import B2Z1CommandBase
from gr00t.rl.envs.base_task.delta_action_base import DeltaActionBase
from gr00t.rl.envs.base_task.staged_task_base import StagedTaskBase
from gr00t.rl.envs.base_task.staged_task_events import stage_completion_event
from gr00t.rl.envs.base_task.warped_action_base import WarpedActionBase
from gr00t.rl.envs.door.stage_conditions import (
    bilateral_contact_force_progress,
    bottleneck_balanced_progress,
    excessive_relative_speed_penalty,
    grasp_collision_aware_position_progress,
    grasp_completion_condition,
    grasp_enclosure_progress,
    open_latch_completion_condition,
    opposed_handle_contact,
    walk_to_door_completion_condition,
)
from gr00t.rl.envs.door.b2z1_reset_geometry import (
    b2z1_gripper_pose_and_jacobian,
    initial_gripper_door_collision,
    solve_b2z1_arm_ik,
)
from gr00t.rl.envs.door.eval_diagnostics import (
    closest_event_mask,
    max_acceptance_event_mask,
    update_event_snapshot,
)
from gr00t.rl.envs.door.task_frame import (
    build_door_task_basis,
    canonicalize_jaw_axial,
    canonicalize_jaw_order,
    task_axial_to_world,
    task_polar_to_world,
    world_axial_to_task,
    world_polar_to_task,
)
from gr00t.rl.envs.door.unsafe_force_diagnostics import UnsafeForceEventRecorder
from gr00t.rl.isaac_utils.rotations import (
    axis_angle_to_quaternion,
    quat_to_tan_norm,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from gr00t.rl.utils.torch_utils import torch_rand_float


class DoorPregrasp(
    StagedTaskBase,
    DeltaActionBase,
    WarpedActionBase,
    B2Z1CommandBase,
):
    STAGE_WALK_TO_DOOR = 0
    STAGE_PREGRASP = 1
    STAGE_GRASP = 2
    STAGE_OPEN = 3
    STAGE_SWING = 4
    STAGE_THROUGH = 5

    def __init__(self, config, device):
        super().__init__(config, device)

        self.gripper_dof_idx = torch.tensor(
            [self.dof_names.index(name) for name in self.config.robot.gripper_dof_names],
            dtype=torch.long,
            device=self.device,
        )
        self.arm_dof_idx = torch.tensor(
            [self.dof_names.index(name) for name in self.config.robot.arm_dof_names],
            dtype=torch.long,
            device=self.device,
        )
        self._upper_non_finger_dof_idx = self.arm_dof_idx
        self._gripper_open_pos = torch.tensor(
            self.config.gripper_open_pos, device=self.device, requires_grad=False
        )
        self._gripper_closed_pos = torch.tensor(
            self.config.gripper_closed_pos, device=self.device, requires_grad=False
        )
        self._gripper_pos_tracking_std = self.config.robot.get(
            "task_gripper_pos_tracking_std", 0.3
        )
        self._gripper_vel_tracking_std = self.config.robot.get(
            "task_gripper_vel_tracking_std", 0.2
        )
        self._gripper_target_vel = self.config.robot.get("task_gripper_target_vel", 0.6)
        self._gripper_open_tolerance = self.config.robot.get(
            "task_gripper_open_tolerance", 0.174533
        )
        self._gripper_closed_tolerance = self.config.robot.get(
            "task_gripper_closed_tolerance", 0.25
        )
        self._approach_root_distance = self.config.robot.get(
            "task_approach_root_distance", 0.3
        )

        # read the door metadata
        stage: Usd.Stage = omni.usd.get_context().get_stage()
        self.door_width = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_height = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_handle_height = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.door_handle_width = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_total_wall_height = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.door_weight = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_open_lr = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_open_io = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_axle_length = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_handle_length = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_hook_length = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_handle_radius = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_spawn_hook = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_hinge_joint_friction = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.door_hinge_drive_max_force = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.door_hinge_drive_damping = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.door_hinge_drive_stiffness = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.door_handle_drive_max_force = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        for env_id in range(self.num_envs):
            door_prim_path = f"/World/envs/env_{env_id}/door"
            door_prim = stage.GetPrimAtPath(door_prim_path)
            door_metadata = door_prim.GetPrim().GetMetadata("customData")
            self.door_width[env_id] = door_metadata["doorWidth"]
            self.door_height[env_id] = door_metadata["doorHeight"]
            self.door_handle_height[env_id] = door_metadata["doorHandleHeight"]
            self.door_handle_width[env_id] = door_metadata["doorHandleWidth"]
            self.door_total_wall_height[env_id] = door_metadata["totalWallHeight"]
            self.door_weight[env_id] = door_metadata["doorWeight"]
            self.door_open_lr[env_id] = door_metadata["doorOpenLR"]
            self.door_open_io[env_id] = door_metadata["doorOpenIO"]
            self.door_axle_length[env_id] = door_metadata["axleLength"]
            self.door_handle_length[env_id] = door_metadata["handleLength"]
            self.door_hook_length[env_id] = door_metadata["hookLength"]
            self.door_handle_radius[env_id] = door_metadata["handleRadius"]
            self.door_spawn_hook[env_id] = float(door_metadata["spawnHook"])
            self.door_hinge_joint_friction[env_id] = door_metadata["hingeJointFriction"]
            self.door_hinge_drive_max_force[env_id] = door_metadata["hingeDriveMaxForce"]
            self.door_hinge_drive_damping[env_id] = door_metadata["hingeDriveDamping"]
            self.door_hinge_drive_stiffness[env_id] = door_metadata["hingeDriveStiffness"]
            self.door_handle_drive_max_force[env_id] = door_metadata["handleDriveMaxForce"]

        # body indices
        self.gripper_idx = self.simulator.body_names.index(self.config.robot.gripper_body_name)
        self.root_idx = self.simulator.body_names.index(self.config.robot.root_body_name)
        self.hand_indices = [
            self.simulator.body_names.index(link)
            for link in self.simulator.robot_config.task_contact_body_names
        ]
        self.hand_indices_tgt_ct_sensor = list(range(len(self.hand_indices)))
        self.hand_indices_convert = list(range(len(self.hand_indices)))
        if len(self.simulator.task_contact_prim_body_ids) != len(self.hand_indices):
            raise RuntimeError(
                "Each task contact force channel must have one contact-body orientation."
            )
        palm_side_directions = self.config.robot.get(
            "task_palm_side_directions", ["+x"] * len(self.hand_indices)
        )
        if len(palm_side_directions) != len(self.hand_indices):
            raise ValueError(
                "robot.task_palm_side_directions must correspond one-to-one with "
                "robot.task_contact_body_names"
            )
        self.hand_palm_side_direction = self._parse_palm_side_direction(
            list(palm_side_directions)
        )
        self._task_grasp_reward_mode = self.config.robot.get(
            "task_grasp_reward_mode", "directional_fingers"
        )
        if self._task_grasp_reward_mode not in (
            "directional_fingers",
            "single_gripper_contact",
        ):
            raise ValueError(
                "Unsupported robot.task_grasp_reward_mode: "
                f"{self._task_grasp_reward_mode}"
            )

        self._task_grasp_orientation_mode = self.config.robot.get(
            "task_grasp_orientation_mode", "door_target_relative"
        )
        if self._task_grasp_orientation_mode not in (
            "door_target_relative",
            "target_frame_with_source_offset",
        ):
            raise ValueError(
                "Unsupported robot.task_grasp_orientation_mode: "
                f"{self._task_grasp_orientation_mode}"
            )
        target_quat = self.config.robot.get("task_grasp_target_quat_wxyz", None)
        self._task_grasp_target_quat = (
            F.normalize(torch.tensor(target_quat, dtype=torch.float32, device=self.device), dim=0)
            if target_quat is not None
            else None
        )
        source_offset_quat = self.config.robot.get(
            "task_grasp_source_quat_target_wxyz", None
        )
        self._task_grasp_source_quat_target = (
            F.normalize(
                torch.tensor(source_offset_quat, dtype=torch.float32, device=self.device), dim=0
            )
            if source_offset_quat is not None
            else None
        )
        if (
            self._task_grasp_orientation_mode == "target_frame_with_source_offset"
            and self._task_grasp_source_quat_target is None
        ):
            raise ValueError(
                "robot.task_grasp_source_quat_target_wxyz is required for "
                "target_frame_with_source_offset"
            )

        # dof indices
        self.finger_dof_idx = self.gripper_dof_idx
        self.non_finger_dof_idx = [
            self.simulator.dof_names.index(dof)
            for dof in self.simulator.dof_names
            if dof not in self.config.robot.gripper_dof_names
        ]
        self.wrist_dof_idx = self.arm_dof_idx
        task_reward_dof_names = self.config.robot.get("task_reward_dof_names", self.dof_names)
        self._task_reward_dof_idx = torch.tensor(
            [self.dof_names.index(name) for name in task_reward_dof_names],
            dtype=torch.long,
            device=self.device,
        )
        overspeed_limit_ratio = self.config.robot.get("task_arm_overspeed_limit_ratio", None)
        self._arm_reward_vel_limits = (
            self.dof_vel_limits[self._upper_non_finger_dof_idx] * overspeed_limit_ratio
            if overspeed_limit_ratio is not None
            else torch.full_like(self.dof_vel_limits[self._upper_non_finger_dof_idx], 2.0)
        )
        termination_limit_ratio = self.config.robot.get(
            "task_arm_termination_limit_ratio", None
        )
        self._arm_termination_vel_limits = (
            self.dof_vel_limits[self._upper_non_finger_dof_idx] * termination_limit_ratio
            if termination_limit_ratio is not None
            else torch.full_like(self.dof_vel_limits[self._upper_non_finger_dof_idx], 20.0)
        )
        self.dof_pos_humanly_lower_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_humanly_lower_limit_list, device=self.device
        )[None, :]
        self.dof_pos_humanly_upper_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_humanly_upper_limit_list, device=self.device
        )[None, :]

        self._left_arm_dof_idx = self.arm_dof_idx
        self._right_arm_dof_idx = self.arm_dof_idx

        self._register_task_state_to_track(self.simulator.scene.articulations["door"], "door")
        self._register_buffer_to_track(
            "delta_actions",
            self._get_delta_actions_buffer_shape(),
            self._store_delta_actions_buffer,
            self._load_delta_actions_buffer,
            dtype=torch.float32,
        )
        self._register_buffer_to_track(
            "last_delta_actions",
            self._get_delta_actions_buffer_shape(),
            self._store_last_delta_actions_buffer,
            self._load_last_delta_actions_buffer,
            dtype=torch.float32,
        )

        self.resting_dof_pos = torch.tensor([self.config.resting_dof_pos], device=self.device)

        self.target_root_pos = torch.tensor(self.config.target_root_pos, device=self.device)[
            None, :
        ]

    def _init_buffers(self):
        super()._init_buffers()
        self.relative_door_pos_buf = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        self.relative_door_rot_buf = torch.zeros(
            self.num_envs, 4, device=self.device, requires_grad=False
        )

        # door state buffer
        self.door_root_state_buf = torch.zeros(
            self.num_envs, 13, device=self.device, requires_grad=False
        )
        self.door_root_state_buf[:, 3] = 1.0  # w
        self.door_dof_state_buf = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        self.door_root_state_buf[:, :3] += self.env_origins

    def _pre_compute_observations_callback(self, env_ids=None):
        super()._pre_compute_observations_callback(env_ids)
        env_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids

        current_root_pos = self.simulator.robot_root_states[env_ids, :3].clone()
        current_root_rot = self.simulator.robot_root_states[env_ids, 3:7].clone()
        current_root_rot_wxyz = xyzw_to_wxyz(current_root_rot)

        door_root_pos = self.simulator.get_task_root_state("door")[env_ids, :3].clone()
        door_root_pos[:, 2] = current_root_pos[:, 2]
        door_root_rot_wxyz = self.simulator.get_task_root_state("door")[env_ids, 3:7].clone()

        relative_door_pos, relative_door_rot = subtract_frame_transforms(
            current_root_pos, current_root_rot_wxyz, door_root_pos, door_root_rot_wxyz
        )
        self.relative_door_pos_buf[env_ids] = relative_door_pos
        self.relative_door_rot_buf[env_ids] = wxyz_to_xyzw(relative_door_rot)

    @StagedTaskBase.effective_in_stage(STAGE_WALK_TO_DOOR)
    def _reward_walk_to_door(self):
        current_root_pos = self.simulator.robot_root_states[:, :3].clone()
        door_root_pos = self.simulator.get_task_root_state("door")[:, :3].clone()
        door_root_pos[:, 2] = current_root_pos[:, 2]
        door_direction = door_root_pos - current_root_pos
        target_dir = F.normalize(door_direction, dim=-1)
        current_root_vel = self.simulator.robot_root_states[:, 7:10].clone()

        target_vel = self.config.get("target_root_vel", 0.3) * target_dir

        return self._tracking_reward_util(
            torch.linalg.norm(current_root_vel - target_vel, dim=-1),
            std=0.15,
            target=0.0,
            scale=1.0,
            offset=0.0,
        )

    @StagedTaskBase.effective_in_stage([STAGE_WALK_TO_DOOR, STAGE_THROUGH])
    def _reward_penalty_upper_body_non_finger_deviation_l1(self):
        """Maintain upper body pose during walking to the door"""
        return torch.abs(
            self.simulator.dof_pos[:, self._upper_non_finger_dof_idx]
            - self.resting_dof_pos[:, self._upper_non_finger_dof_idx]
        ).sum(dim=-1)

    @StagedTaskBase.effective_in_stage([STAGE_WALK_TO_DOOR, STAGE_PREGRASP, STAGE_THROUGH])
    def _reward_pregrasp_finger_dof_pos_l1(self):
        pos_diff = self.simulator.dof_pos[:, self.gripper_dof_idx] - self._gripper_open_pos
        pos_track = self._tracking_reward_util(
            pos_diff,
            std=self._gripper_pos_tracking_std,
            target=0.0,
            scale=1.0,
            offset=0.0,
        ).mean(dim=-1)

        vel_diff = -self.simulator.dof_vel[:, self.gripper_dof_idx] * torch.sign(pos_diff)
        vel_track = self._tracking_reward_util(
            vel_diff,
            std=self._gripper_vel_tracking_std,
            target=self._gripper_target_vel,
            scale=1.0,
            offset=0.0,
        ).mean(dim=-1)

        return (pos_track + vel_track).clamp(max=1.0)

    @StagedTaskBase.effective_in_stage([STAGE_PREGRASP, STAGE_GRASP, STAGE_OPEN, STAGE_SWING])
    def _reward_penalty_unused_dof_deviation_l1(self):
        """Penalize the deviation of the unused arm dof during door opening"""
        diff = self.simulator.dof_pos[:, self.arm_dof_idx] - self.resting_dof_pos[:, self.arm_dof_idx]
        return diff.abs().sum(dim=-1)

    @StagedTaskBase.effective_in_stage([STAGE_PREGRASP, STAGE_GRASP, STAGE_OPEN, STAGE_SWING])
    def _reward_hand_handle_orientation(self):
        current_rot = xyzw_to_wxyz(
            self.simulator._rigid_body_rot[:, self.gripper_idx, :]
        )
        if self._task_grasp_orientation_mode == "target_frame_with_source_offset":
            target_frame_rot = self.simulator.scene.sensors[
                "hand_frame_transformer"
            ].data.target_quat_w[:, 0, :]
            source_offset = self._task_grasp_source_quat_target.expand(self.num_envs, -1)
            target_rot = quat_mul(target_frame_rot, source_offset)
            relative_rot = quat_mul(current_rot, quat_inv(target_rot))
            return self._tracking_reward_util(
                wrap_to_pi(axis_angle_from_quat(relative_rot).norm(dim=-1)),
                std=0.6,
                target=0.0,
                scale=1.0,
                offset=0.0,
            )

        current_hand_rot = self.simulator.hand_transform_rot[:, 0, :]
        if self._task_grasp_target_quat is not None:
            target_rot = self._task_grasp_target_quat.expand(self.num_envs, -1)
            relative_rot = quat_mul(current_hand_rot, quat_inv(target_rot))
            return self._tracking_reward_util(
                wrap_to_pi(axis_angle_from_quat(relative_rot).norm(dim=-1)),
                std=0.6,
                target=0.0,
                scale=1.0,
                offset=0.0,
            )

        # Preserve the original humanoid palm-frame convention for robots that
        # do not provide a morphology-specific grasp target orientation.
        mask = (self.door_open_lr < 0)[:, None]
        rot_90 = quat_from_euler_xyz(
            torch.full((self.num_envs,), torch.pi / 2.0, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
        )
        rot_neg_90 = quat_from_euler_xyz(
            torch.full((self.num_envs,), -torch.pi / 2.0, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
        )
        relative_rot = quat_mul(current_hand_rot, torch.where(mask, rot_90, rot_neg_90))
        return self._tracking_reward_util(
            wrap_to_pi(axis_angle_from_quat(relative_rot).norm(dim=-1)),
            std=0.6,
            target=0.0,
            scale=1.0,
            offset=0.0,
        )

    @StagedTaskBase.effective_in_stage([STAGE_PREGRASP, STAGE_GRASP, STAGE_OPEN])
    def _reward_standing_still(self):
        norm = self._get_base_motion_norm()
        return self._tracking_reward_util(norm, std=0.05, target=0.0, scale=1.0, offset=0.0)

    @StagedTaskBase.effective_in_stage([STAGE_PREGRASP, STAGE_GRASP, STAGE_OPEN])
    def _reward_penalty_not_standing_still(self):
        return self._get_base_motion_norm()

    @StagedTaskBase.effective_in_stage(STAGE_SWING)
    def _reward_penalty_standing_still(self):
        norm = self._get_base_motion_norm()
        return self._tracking_reward_util(norm, std=0.05, target=0.0, scale=1.0, offset=0.0)

    def _get_base_motion_norm(self):
        return torch.linalg.vector_norm(
            torch.cat([self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:3]], dim=-1), dim=-1
        )

    @StagedTaskBase.effective_in_stage(STAGE_PREGRASP)
    def _reward_pregrasp_target_distance(self):
        pre_grasp_target = self._compute_pre_grasp_target()

        hand_pos = self.simulator._rigid_body_pos[:, self.gripper_idx, :]
        hand_pos_to_pre_grasp_target = pre_grasp_target - hand_pos
        hand_pos_to_pre_grasp_target_norm = torch.norm(hand_pos_to_pre_grasp_target, dim=-1)

        pos_reward = self._tracking_reward_util(
            hand_pos_to_pre_grasp_target_norm,
            std=0.2,
            target=0.0,
            scale=1.0,
            offset=0.0,
        )

        current_direction = F.normalize(pre_grasp_target - hand_pos, dim=-1)
        palm_vel = self.simulator._rigid_body_vel[:, self.gripper_idx, :]
        pregrasp_target_vel = self.config.get("pregrasp_target_vel", 0.5)
        target_vel = pregrasp_target_vel * current_direction

        vel_reward = self._tracking_reward_util(
            torch.linalg.norm(palm_vel - target_vel, dim=-1),
            std=0.15,
            target=0.0,
            scale=1.0,
            offset=0.0,
        )
        return (pos_reward + vel_reward).clamp(max=1.0)

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_OPEN, STAGE_SWING])
    def _reward_grasp_finger_dof_pos_l1(self):
        pos_diff = self.simulator.dof_pos[:, self.gripper_dof_idx] - self._gripper_closed_pos
        pos_track = self._tracking_reward_util(
            pos_diff,
            std=self._gripper_pos_tracking_std,
            target=0.0,
            scale=1.0,
            offset=0.0,
        ).mean(dim=-1)

        vel_diff = -self.simulator.dof_vel[:, self.gripper_dof_idx] * torch.sign(pos_diff)
        vel_track = self._tracking_reward_util(
            vel_diff,
            std=self._gripper_vel_tracking_std,
            target=self._gripper_target_vel,
            scale=1.0,
            offset=0.0,
        ).mean(dim=-1)

        return (pos_track + vel_track).clamp(max=1.0)

    @StagedTaskBase.effective_in_stage([STAGE_GRASP, STAGE_OPEN, STAGE_SWING])
    def _reward_grasp_target_distance(self):
        grasp_target = self._compute_grasp_target()

        hand_pos = self.simulator._rigid_body_pos[:, self.gripper_idx, :]
        hand_pos_to_grasp_target_norm = torch.norm(grasp_target - hand_pos, dim=-1)

        return self._tracking_reward_util(
            hand_pos_to_grasp_target_norm,
            std=0.1,
            target=0.0,
            scale=1.0,
            offset=0.0,
        )

    @StagedTaskBase.effective_in_stage([STAGE_PREGRASP, STAGE_GRASP, STAGE_OPEN, STAGE_SWING])
    def _reward_grasp(self):
        contact_forces = self._get_object_to_hand_contact_forces()
        if self._task_grasp_reward_mode == "single_gripper_contact":
            reward = contact_forces.norm(dim=-1).clamp(max=10.0).mean(dim=-1)
            pregrasp = self.stage_buf == DoorPregrasp.STAGE_PREGRASP
            reward[pregrasp] = -reward[pregrasp]
            return reward

        contact_forces_flattened = contact_forces.reshape(-1, 3)
        if contact_forces.shape[1] != len(self.simulator.task_contact_prim_body_ids):
            raise RuntimeError(
                "Contact force channels and task contact body frames are not one-to-one."
            )
        hand_rot = self.simulator.task_contact_prim_rot_wxyz
        hand_rot_flattened = hand_rot.reshape(-1, 4)
        palm_side_repeat = torch.tile(
            self.hand_palm_side_direction, (contact_forces.shape[0], 1)
        )
        contact_forces_hand_frame = quat_apply(
            quat_inv(hand_rot_flattened), contact_forces_flattened
        )
        contact_forces_palm_frame = quat_apply(
            quat_inv(palm_side_repeat), contact_forces_hand_frame
        )
        reward = (
            (
                -1.0 * torch.abs(contact_forces_palm_frame[:, 1:]).sum(dim=-1)
                + contact_forces_palm_frame[:, 0]
            )
            .clamp(min=-10, max=10)
            .reshape(self.num_envs, -1)
            .mean(dim=-1)
        )

        reward[self.stage_buf == DoorPregrasp.STAGE_PREGRASP] = -1.0 * torch.abs(
            reward[self.stage_buf == DoorPregrasp.STAGE_PREGRASP]
        )

        return reward

    @StagedTaskBase.effective_in_stage(STAGE_OPEN)
    def _reward_push_door_force(self):
        net_force = self._get_object_to_hand_contact_forces().sum(dim=-2)
        # reward -x direction force (pushing the door)
        return (net_force[:, 0] * self.door_open_lr).clamp(min=0.0, max=20.0)

    @StagedTaskBase.effective_in_stage(STAGE_OPEN)
    def _reward_push_door_handle(self):
        handle_vel_reward = self.simulator.scene.articulations["door"].data.joint_vel[:, 1]
        handle_pos_reward = (
            self.simulator.scene.articulations["door"]
            .data.joint_pos[:, 1]
            .clamp(min=0.0, max=0.785398)
            / 0.785398
        )
        return (handle_vel_reward + handle_pos_reward).clamp(max=1.0, min=-1.0)

    @StagedTaskBase.effective_in_stage([STAGE_SWING, STAGE_THROUGH])
    def _reward_dont_push_door_handle(self):
        handle_vel_reward = -1.0 * self.simulator.scene.articulations["door"].data.joint_vel[:, 1]
        handle_pos_reward = (
            0.785398 - self.simulator.scene.articulations["door"].data.joint_pos[:, 1]
        ).clamp(min=0.0, max=0.785398) / 0.785398
        return (handle_vel_reward + handle_pos_reward).clamp(max=1.0, min=-1.0)

    @StagedTaskBase.effective_in_stage([STAGE_OPEN, STAGE_SWING])
    def _reward_push_door_hinge(self):
        hinge_vel_reward = self.simulator.scene.articulations["door"].data.joint_vel[:, 0] * 10
        hinge_pos_reward = (
            self.simulator.scene.articulations["door"]
            .data.joint_pos[:, 0]
            .clamp(min=0.0, max=1.5708)
            / 1.5708
        )
        return (hinge_vel_reward + hinge_pos_reward).clamp(max=1.0, min=-1.0)

    @StagedTaskBase.effective_in_stage([STAGE_SWING, STAGE_THROUGH])
    def _reward_target_root_distance(self):
        target_direction = F.normalize(
            self.target_root_pos - (self.simulator.robot_root_states[:, :3] - self.env_origins),
            dim=-1,
        )
        root_vel = self.simulator._rigid_body_vel[:, self.root_idx, :]
        root_vel_along_target_direction = torch.sum(root_vel * target_direction, dim=-1)
        root_vel_target = self.config.get("target_root_vel", 0.3)
        root_vel_reward = self._tracking_reward_util(
            root_vel_along_target_direction, std=0.2, target=root_vel_target, scale=1.0, offset=0.0
        )

        root_pos_diff = torch.norm(
            self.simulator.robot_root_states[:, :3] - self.env_origins - self.target_root_pos,
            dim=-1,
        )
        root_pos_reward = self._tracking_reward_util(
            root_pos_diff, std=0.2, target=0.0, scale=1.0, offset=0.0
        )
        reward = (root_vel_reward + root_pos_reward).clamp(max=1.0)
        reward[self.stage_buf == DoorPregrasp.STAGE_SWING] *= 0.5
        return reward

    @override
    def _reward_limits_dof_pos(self):
        # Penalize dof positions too close to the limit
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
        return torch.sum(out_of_limits[:, self._upper_non_finger_dof_idx], dim=1)

    def _reward_penalty_humanly_dof_limit(self):
        lower_limit_violations = -1.0 * (
            self.simulator.dof_pos - self.dof_pos_humanly_lower_limit
        ).clip(max=0.0)
        upper_limit_violations = (
            self.simulator.dof_pos - self.dof_pos_humanly_upper_limit
        ).clip(min=0.0)
        return (
            lower_limit_violations[:, self._task_reward_dof_idx]
            + upper_limit_violations[:, self._task_reward_dof_idx]
        ).sum(dim=-1)

    def _reward_penalty_door_frame_contact(self):
        door_frame_unwanted_contact_forces = self.simulator.scene.sensors[
            "door_frame_unwanted_contact_sensor"
        ].data.net_forces_w
        return door_frame_unwanted_contact_forces.norm(dim=-1).sum(dim=-1)

    def _reward_penalty_door_panel_contact(self):
        door_panel_unwanted_contact_forces = self.simulator.scene.sensors[
            "door_panel_unwanted_contact_sensor"
        ].data.net_forces_w
        return door_panel_unwanted_contact_forces.norm(dim=-1).sum(dim=-1)

    def _reward_penalty_upper_body_dof_vel(self):
        return torch.sum(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx] ** 2, dim=-1)

    @StagedTaskBase.effective_in_stage(
        [STAGE_WALK_TO_DOOR, STAGE_PREGRASP, STAGE_GRASP, STAGE_THROUGH]
    )
    def _reward_penalty_face_door(self):
        relative_door_rot = xyzw_to_wxyz(self.relative_door_rot_buf)
        if self.config.robot.get("task_face_target_yaw_only", False):
            _, _, yaw = euler_xyz_from_quat(relative_door_rot)
            return torch.abs(wrap_to_pi(yaw))
        return wrap_to_pi(axis_angle_from_quat(relative_door_rot).norm(dim=-1))

    def _reward_penalty_upright(self):
        upright_vec = torch.repeat_interleave(
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device), self.num_envs, dim=0
        )
        torso_quat_wxyz = xyzw_to_wxyz(self.simulator._rigid_body_rot[:, self.root_idx])
        rotated_vec = quat_apply(torso_quat_wxyz, upright_vec)
        return torch.sum(torch.square(rotated_vec - upright_vec), dim=-1)

    @override
    def _reward_penalty_dof_acc(self):
        return torch.sum(
            torch.square(self.simulator.dof_acc[:, self._upper_non_finger_dof_idx]), dim=-1
        )

    @override
    def _reward_penalty_dof_vel(self):
        return torch.sum(
            torch.square(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx]), dim=-1
        )

    @override
    def _reward_penalty_undesired_contact(self):
        undesired_contact = torch.sum(
            torch.norm(self.simulator.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
            > 1,
            dim=1,
            dtype=torch.float,
        )
        return undesired_contact

    def _reward_penalty_dof_overspeed(self):
        arm_vel = self.simulator.dof_vel[:, self._upper_non_finger_dof_idx]
        return torch.square(
            (torch.abs(arm_vel) - self._arm_reward_vel_limits).clip(min=0.0)
        ).sum(dim=-1)

    def _get_obs_relative_to_door(self):
        relative_door_rot_6d = quat_to_tan_norm(self.relative_door_rot_buf, w_last=True)
        return torch.cat([self.relative_door_pos_buf, relative_door_rot_6d], dim=-1)

    def _get_obs_hand_handle_transform(self):
        hand_pos = self.simulator.hand_transform_pos[:, 0, :]
        hand_rot_wxyz = self.simulator.hand_transform_rot[:, 0, :]
        hand_rot_6d = quat_to_tan_norm(wxyz_to_xyzw(hand_rot_wxyz), w_last=True)
        return torch.cat([hand_pos, hand_rot_6d], dim=-1)

    def _get_obs_hand_force(self):
        hand_force = getattr(self.simulator, "task_hand_contact_forces", None)
        if hand_force is None:
            hand_force = self.simulator.contact_forces[:, self.hand_indices, :]
        hand_force = hand_force.clone()
        # ContactSensor keeps its previous force cache when an articulation is teleported.
        hand_force[self.episode_length_buf == 0] = 0.0
        return hand_force.reshape(hand_force.shape[0], -1)

    def _get_object_to_hand_contact_forces(self):
        contact_forces = self.simulator.object_to_hand_contact_forces[
            :, 0, self.hand_indices_tgt_ct_sensor, :
        ][:, self.hand_indices_convert, :].clone()
        contact_forces[self.episode_length_buf == 0] = 0.0
        return contact_forces

    def _get_obs_privileged_door_info(self):
        opens_left = (self.door_open_lr > 0).float()
        opens_right = (self.door_open_lr < 0).float()
        return torch.stack(
            [
                self.door_width,
                self.door_height,
                self.door_handle_height,
                self.door_handle_width,
                self.door_weight / 100.0,
                opens_left,
                opens_right,
                self.door_open_io,
            ],
            dim=1,
        )

    def _get_obs_privileged_door_geometry(self):
        return torch.stack(
            [
                self.door_axle_length,
                self.door_handle_length,
                self.door_hook_length,
                self.door_handle_radius,
                self.door_spawn_hook,
            ],
            dim=1,
        )

    def _get_obs_privileged_door_dynamics(self):
        return torch.stack(
            [
                self.door_hinge_joint_friction,
                self.door_hinge_drive_max_force,
                self.door_hinge_drive_damping,
                self.door_hinge_drive_stiffness,
                self.door_handle_drive_max_force,
            ],
            dim=1,
        )

    def _get_obs_door_dof_pos(self):
        return self.simulator.get_task_dof_pos("door")[:, :2]

    def _get_obs_door_latch_pos(self):
        return self.simulator.get_task_dof_pos("door")[:, 2:3]

    def _get_obs_door_dof_vel(self):
        return self.simulator.get_task_dof_vel("door")[:, :3]

    def _get_obs_handle_contact_force(self):
        return self._get_object_to_hand_contact_forces().sum(dim=-2)

    def _get_obs_dof_pos_non_finger(self):
        return self.simulator.dof_pos[:, self.non_finger_dof_idx]

    def _get_obs_dof_vel_non_finger(self):
        return self.simulator.dof_vel[:, self.non_finger_dof_idx]

    def _get_obs_target_obj_pos(self):
        return (
            self.simulator.scene.sensors["root_target_frame_transformer"]
            .data.target_pos_source[:, 0, :]
            .clone()
        )

    def _compute_grasp_target(self):
        grasp_target_pos_w = (
            self.simulator.scene.sensors["hand_frame_transformer"]
            .data.target_pos_w[:, 0, :]
            .clone()
        )
        return grasp_target_pos_w

    def _compute_pre_grasp_target(self):
        grasp_target_pos_w = self._compute_grasp_target()
        grasp_target_pos_w[:, 2] += 0.1
        return grasp_target_pos_w

    @override
    def _reset_object_states_callback(self, env_ids):
        self._reset_door_states(env_ids)
        return super()._reset_object_states_callback(env_ids)

    @override
    def _reset_root_states(self, env_ids, target_root_states=None):
        if self.config.robot.asset.get("robot_type", "") == "b2z1":
            self.target_robot_root_states[env_ids] = self.base_init_state
            self.target_robot_root_states[env_ids, :3] += self.env_origins[env_ids]
            self.target_robot_root_states[env_ids, 7:13] = 0.0
        else:
            self.target_robot_root_states[env_ids, 7:13] = torch_rand_float(
                -0.5, 0.5, (len(env_ids), 6), device=str(self.device)
            )  # [7:10]: lin vel, [10:13]: ang vel

        root_rot_wxyz = xyzw_to_wxyz(self.target_robot_root_states[env_ids, 3:7])
        r, p, _ = euler_xyz_from_quat(root_rot_wxyz)
        self.target_robot_root_states[env_ids, 0:1] = torch_rand_float(
            -1.5, -0.6, (len(env_ids), 1), device=str(self.device)
        )
        self.target_robot_root_states[env_ids, 1:2] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 1), device=str(self.device)
        )
        self.target_robot_root_states[env_ids, 0:2] += self.env_origins[env_ids, 0:2]
        random_yaw = torch_rand_float(
            -torch.pi / 4, torch.pi / 4, (len(env_ids), 1), device=str(self.device)
        )[:, 0]
        self.target_robot_root_states[env_ids, 3:7] = wxyz_to_xyzw(
            quat_from_euler_xyz(r, p, random_yaw)
        )

    @override
    def _reset_dofs(self, env_ids, target_state=None):
        if self.config.robot.asset.get("robot_type", "") == "b2z1":
            self.target_robot_dof_state[env_ids, :, 0] = self.default_dof_pos
            self.target_robot_dof_state[env_ids, :, 1] = 0.0
            return

        # randomize wrist in +- 80 deg
        xx, yy = torch.meshgrid(env_ids, self.wrist_dof_idx)
        self.target_robot_dof_state[xx, yy, 0] = torch_rand_float(
            -1.39626, 1.39626, (len(env_ids), len(self.wrist_dof_idx)), device=str(self.device)
        )

        # completely randomize finger dofs
        xx, yy = torch.meshgrid(env_ids, self.finger_dof_idx)
        upper_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_upper_limit_list, device=str(self.device)
        )[None, self.finger_dof_idx]
        lower_limit = torch.tensor(
            self.simulator.robot_config.dof_pos_lower_limit_list, device=str(self.device)
        )[None, self.finger_dof_idx]
        self.target_robot_dof_state[xx, yy, 0] = lower_limit + (
            upper_limit - lower_limit
        ) * torch_rand_float(
            0.0, 1.0, (len(env_ids), len(self.finger_dof_idx)), device=str(self.device)
        )

        # set velocities to 0
        self.target_robot_dof_state[env_ids, :, 1] = 0.0

    def _reset_door_states(self, env_ids):
        randomize_door_init_state = self.config.get("randomize_door_init_state", False)
        self.door_dof_state_buf[:] = 0.0
        if randomize_door_init_state:
            # 33% of the environments to have a different initial state
            rand_env_ids = env_ids[torch.randperm(len(env_ids))[: len(env_ids) // 3]]
            self.door_dof_state_buf[rand_env_ids, 0] = torch_rand_float(
                0.261799, 1.74533, (len(rand_env_ids), 1), device=self.device
            ).squeeze(-1)
        door_dof_state_dict = {
            "door": (
                self.door_dof_state_buf,
                torch.zeros_like(self.door_dof_state_buf),
                torch.tensor([0, 1, 2], device=self.device, dtype=torch.long),
            )
        }
        self.simulator.set_task_dof_state_tensor(env_ids, door_dof_state_dict)

        door_dof_target = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        door_dof_target[:, 0] = 0.0
        door_dof_target[:, 1] = 15 * torch.pi / 180.0  # tension the door handle
        self.simulator.apply_torques_at_task_dof(env_ids, {"door": door_dof_target})

    @override
    def _check_termination(self):
        super()._check_termination()
        far_from_door = self.relative_door_pos_buf.norm(dim=-1) > 4.0
        self.reset_buf |= far_from_door

        arm_dof_velocity = self.simulator.dof_vel[:, self._upper_non_finger_dof_idx]
        arm_dof_speed_ratio = (
            torch.abs(arm_dof_velocity) / self._arm_termination_vel_limits
        )
        dof_overspeed = torch.any(
            arm_dof_speed_ratio > self.termination_level,
            dim=-1,
        )
        not_just_resetted = self.episode_length_buf > 20
        arm_dof_overspeed = dof_overspeed & not_just_resetted

        self.reset_buf |= arm_dof_overspeed

        # reset if the base velocity command is too large when grasping or opening the door

    @property
    def ground_height(self):
        return 0.0

    def _stage_0_reward_condition(self):
        # walk to the door
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _stage_0_to_complete_condition(self):
        return self._stage_0_to_1_advance_condition()

    def _stage_0_to_1_advance_condition(self):
        # get close enough to the door
        grasp_target = self._compute_grasp_target()
        root_pos = self.simulator.robot_root_states[:, :3].clone()
        root_pos[:, 2] = grasp_target[:, 2]
        cond = (root_pos - grasp_target).norm(dim=-1) < self._approach_root_distance

        # keep hands down
        max_deviation = (
            torch.abs(
                self.simulator.dof_pos[:, self._upper_non_finger_dof_idx]
                - self.resting_dof_pos[:, self._upper_non_finger_dof_idx]
            )
            .max(dim=-1)
            .values
        )
        cond &= max_deviation < 0.25
        return cond

    def _stage_1_reward_condition(self):
        # The robot must actually be settled; a zero command does not imply zero motion.
        cond = self._get_base_motion_norm() <= 0.1
        # stay close to the door
        cond &= self._stage_0_to_1_advance_condition()
        return cond

    def _stage_1_to_complete_condition(self):
        return self._stage_1_to_2_advance_condition()

    def _stage_1_to_2_advance_condition(self):
        # raise hand to pre-grasp position
        pre_grasp_target = self._compute_pre_grasp_target()

        hand_pos = self.simulator._rigid_body_pos[:, self.gripper_idx, :]
        hand_above_handle = hand_pos[:, 2] > self.door_handle_height + 0.05
        hand_close_to_pre_grasp_target = (hand_pos - pre_grasp_target).norm(dim=-1) < 0.1
        hand_close_to_pre_grasp_dof_target = (
            torch.abs(self.simulator.dof_pos[:, self.gripper_dof_idx] - self._gripper_open_pos)
            .mean(dim=-1)
            < self._gripper_open_tolerance
        )
        cond = hand_above_handle & hand_close_to_pre_grasp_target & hand_close_to_pre_grasp_dof_target
        cond &= self._reward_hand_handle_orientation() > 0.2

        cond &= self._get_base_motion_norm() <= 0.1

        door_opened = self.simulator.scene.articulations["door"].data.joint_pos[:, 0] > 0.174533

        return cond | door_opened

    def _stage_2_reward_condition(self):
        return self._get_base_motion_norm() <= 0.1

    def _stage_2_to_complete_condition(self):
        # grasp the door handle
        hand_handle_contact_count = (
            self._get_object_to_hand_contact_forces().norm(dim=-1)
            > 1
        ).sum(dim=-1)
        hand_grasped = hand_handle_contact_count >= 1
        gripper_closed = (
            torch.abs(self.simulator.dof_pos[:, self.gripper_dof_idx] - self._gripper_closed_pos)
            .mean(dim=-1)
            < self._gripper_closed_tolerance
        )
        return hand_grasped | gripper_closed

    def _stage_2_to_3_advance_condition(self):
        # grasp the door handle
        door_opened = self.simulator.scene.articulations["door"].data.joint_pos[:, 0] > 0.174533
        return self._stage_2_to_complete_condition() | door_opened

    def _stage_3_reward_condition(self):
        # keep grasping the door handle
        return self._stage_2_to_3_advance_condition() & self._stage_2_reward_condition()

    def _stage_3_to_4_advance_condition(self):
        # rotate the door handle and open the door
        door_opened = self.simulator.scene.articulations["door"].data.joint_pos[:, 0] > 0.174533
        return door_opened

    def _stage_4_reward_condition(self):
        # keep grasping the door handle
        return self._stage_3_to_4_advance_condition()

    def _stage_4_to_5_advance_condition(self):
        # walk through the door and leave handle up
        walked_through_door = (
            self.simulator.robot_root_states[:, 0] - self.env_origins[:, 0]
        ) > 0.0
        door_opened = self.simulator.scene.articulations["door"].data.joint_pos[:, 0] > 1.0472
        handle_up = self.simulator.scene.articulations["door"].data.joint_pos[:, 1] < 0.2
        return walked_through_door & handle_up & door_opened

    def _stage_5_reward_condition(self):
        # keep walking through the door
        return self._stage_4_to_5_advance_condition()

    def _stage_5_to_complete_condition(self):
        return (self.simulator.robot_root_states[:, 0] - self.env_origins[:, 0]) > 1.5

    def scene_creation_callback(self, simulator):
        door_frame_unwanted_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/{simulator.task_config.target_obj}/root",
        )

        door_panel_unwanted_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/{simulator.task_config.target_obj}/door_panel",
        )
        simulator.scene.sensors["door_frame_unwanted_contact_sensor"] = ContactSensor(
            door_frame_unwanted_contact_sensor_config
        )
        simulator.scene.sensors["door_panel_unwanted_contact_sensor"] = ContactSensor(
            door_panel_unwanted_contact_sensor_config
        )

        root_target_frame_transformer_config: FrameTransformerCfg = FrameTransformerCfg(
            prim_path=f"/World/envs/env_.*/Robot/{simulator.robot_config.root_body_name}",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=simulator.scene.sensors["hand_frame_transformer"]
                    .cfg.target_frames[0]
                    .prim_path
                ),
            ],
        )
        simulator.scene.sensors["root_target_frame_transformer"] = FrameTransformer(
            root_target_frame_transformer_config
        )

    def _parse_palm_side_direction(self, palm_side_direction: list[str]) -> torch.Tensor:
        """
        Convert the palm side direction to a quaternion that rotates anything
        expressed in the finger frame to point into the palm.
        """
        output = torch.zeros(len(palm_side_direction), 4, device=self.device)  # wxyz
        for i, direction in enumerate(palm_side_direction):
            if direction == "+x":
                output[i] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            elif direction == "-x":
                output[i] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            elif direction == "+y":
                output[i] = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], device=self.device)
            elif direction == "-y":
                output[i] = torch.tensor([0.7071068, 0.0, 0.0, -0.7071068], device=self.device)
            elif direction == "+z":
                output[i] = torch.tensor([0.7071068, 0.0, -0.7071068, 0.0], device=self.device)
            elif direction == "-z":
                output[i] = torch.tensor([0.7071068, 0.0, 0.7071068, 0.0], device=self.device)
            else:
                raise ValueError(f"Invalid palm side direction: {direction}")
        return output


class B2Z1DoorSixStage(DoorPregrasp):
    """B2Z1 door task expressed as six physical goal states."""

    STAGE_WALK_TO_DOOR = 0
    STAGE_GRASP = 1
    STAGE_OPEN_LATCH = 2
    STAGE_REPOSITION = 3
    STAGE_PUSH_OPEN = 4
    STAGE_THROUGH = 5

    _STAGE_NAMES = (
        "walk_to_door",
        "grasp",
        "open_latch",
        "reposition",
        "push_open",
        "through",
    )

    _MIDDLE_ARM = (-1.57, 1.6, -0.63, -1.1, 0.0, 0.0)
    # Centers of the distal inner collision surfaces, expressed in their
    # simulation rigid-body frames.  These come directly from the refined
    # B2Z1 gripper hulls used by AutoOpen (fixed_023/024 and moving_000).
    _FIXED_INNER_CENTER_LINK06 = (0.16364627, -0.01396101, 0.00088488)
    _MOVING_INNER_CENTER = (0.07321262, -0.00014943, -0.00634019)
    _FIXED_POCKET_PROXIMAL_DEPTH = 0.04491097
    # AutoOpen measures these values from the same refined B2Z1 collision
    # meshes.  With the gripper facing the panel, its foremost collider is
    # 100.4 mm beyond gripper_link; the randomized door panel is 40 mm thick.
    _GRIPPER_PANEL_SUPPORT_OFFSET = 0.1004
    _DOOR_PANEL_HALF_THICKNESS = 0.0200
    _PANEL_READY_CLEARANCE = 0.0300
    _REPOSITION_BASE_POSITION_TOLERANCE = 0.08
    _REPOSITION_BASE_YAW_TOLERANCE = 0.08
    _REPOSITION_EE_POSITION_TOLERANCE = 0.05
    _REPOSITION_EE_ORIENTATION_TOLERANCE = 0.08
    _REPOSITION_GRIPPER_TOLERANCE = 0.20
    _REPOSITION_BASE_MOTION_TOLERANCE = 0.08
    _REPOSITION_EE_SPEED_TOLERANCE = 0.08
    _PUSH_TARGET_ANGULAR_SPEED = math.radians(20.0)
    _PUSH_SAFE_PANEL_RELATIVE_SPEED = 0.2
    _EVAL_MAX_DIAGNOSTICS = (
        "hinge_angle",
        "hinge_speed",
        "hinge_opening_speed",
        "panel_force",
        "handle_force",
        "task_force",
    )
    _EVAL_BOOL_DIAGNOSTICS = (
        "contact_seen",
        "unsafe_force",
        "door_overspeed",
        "shoulder_collision",
        "overtime",
    )
    _EVAL_GRASP_MAX_DIAGNOSTICS = (
        "aperture_quality",
        "between_jaws_quality",
        "capture_progress",
        "acceptance_count",
    )
    _EVAL_GRASP_MIN_DIAGNOSTICS = (
        "pocket_distance",
        "pocket_normal",
        "pocket_tangent",
        "pocket_vertical",
        "axis_error",
    )
    _EVAL_GRASP_CLOSEST_DIAGNOSTICS = (
        "pocket_normal",
        "pocket_tangent",
        "pocket_vertical",
    )
    _EVAL_GRASP_BOOL_DIAGNOSTICS = (
        "any_valid_contact",
        "both_valid_contact",
        "opposed_contact",
        "precise_bilateral_grasp",
    )
    _EVAL_GRASP_ACCEPTANCE_DIAGNOSTICS = (
        "opposed",
        "any_valid_contact",
        "pocket_distance",
        "pocket_normal",
        "axis",
        "relative_speed",
        "gripper_speed",
        "contact_force",
        "no_unexpected_contact",
        "panel_free",
    )
    _EVAL_GRASP_EVENT_NAMES = ("closest_pocket", "max_acceptance")
    _EVAL_GRASP_EVENT_FIELDS = {
        "metadata": 4,
        "error_previous": 3,
        "error_current": 3,
        "error_delta": 3,
        "policy_mean": 10,
        "policy_delta": 10,
        "accumulated_command": 10,
        "physical_command": 10,
        "unclipped_command": 10,
        "stage_components": 5,
        "acceptance_checks": 10,
        "reward_terms": 7,
        "contact_terms": 8,
    }
    _EVAL_THROUGH_MAX_DIAGNOSTICS = (
        "through_distance",
        "through_stable_steps",
    )
    _EVAL_THROUGH_MIN_DIAGNOSTICS = (
        "middle_arm_error",
        "middle_arm_speed",
        "base_motion",
    )
    _EVAL_THROUGH_BOOL_DIAGNOSTICS = (
        "through_distance_ok",
        "through_arm_stowed",
        "through_door_open",
        "through_base_stable",
        "through_all_checks_now",
    )

    def __init__(self, config, device):
        super().__init__(config, device)
        self._canonical_action_scale = torch.as_tensor(
            config.canonical_action_scale,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        if self._canonical_action_scale.shape != (1, self.COMMAND_DIM):
            raise ValueError(
                "canonical_action_scale must have one value per policy action"
            )
        if torch.any(self._canonical_action_scale <= 0.0):
            raise ValueError("canonical_action_scale values must be positive")
        door = self.simulator.scene.articulations["door"]
        panel_ids, _ = door.find_bodies("door_panel", preserve_order=True)
        handle_ids, _ = door.find_bodies("door_handle", preserve_order=True)
        if len(panel_ids) != 1:
            raise RuntimeError(f"Expected one door_panel body, got {panel_ids}")
        if len(handle_ids) != 1:
            raise RuntimeError(f"Expected one door_handle body, got {handle_ids}")
        self._door_panel_idx = panel_ids[0]
        self._door_handle_idx = handle_ids[0]
        self._link01_idx = self.simulator.body_names.index("link01")
        self._link06_idx = self.simulator.body_names.index("link06")
        mover_ids, _ = self.simulator._robot.find_bodies(
            "gripperMover", preserve_order=True
        )
        if len(mover_ids) != 1:
            raise RuntimeError(f"Expected one gripperMover body, got {mover_ids}")
        # gripperMover is intentionally absent from simulator.body_names: that
        # list fixes the low-level controller's body ordering.  Read this extra
        # diagnostic body from the complete IsaacLab articulation instead.
        self._gripper_mover_articulation_idx = mover_ids[0]
        self._middle_arm = torch.tensor(self._MIDDLE_ARM, device=self.device)
        self._reset_collision_margin = float(
            config.get("reset_collision_margin", 0.01)
        )
        self._reset_sampling_max_attempts = int(
            config.get("reset_sampling_max_attempts", 64)
        )
        if self._reset_collision_margin < 0.0:
            raise ValueError("reset_collision_margin must be non-negative")
        if self._reset_sampling_max_attempts < 1:
            raise ValueError("reset_sampling_max_attempts must be positive")
        self._fixed_inner_center_link06 = torch.tensor(
            self._FIXED_INNER_CENTER_LINK06, device=self.device
        )
        self._moving_inner_center = torch.tensor(
            self._MOVING_INNER_CENTER, device=self.device
        )
        self._operation_station_distance = float(
            config.get("operation_station_distance", 0.77)
        )
        self._reposition_comfort_radius = float(
            config.get("reposition_comfort_radius", 0.70)
        )
        workspace_radius_lower = float(
            self._b2z1_ee_goal_sphere_lower.reshape(-1)[0]
        )
        workspace_radius_upper = float(
            self._b2z1_ee_goal_sphere_upper.reshape(-1)[0]
        )
        if not (
            workspace_radius_lower
            < self._reposition_comfort_radius
            < workspace_radius_upper
        ):
            raise ValueError(
                "reposition_comfort_radius must lie strictly inside the B2Z1 "
                f"workspace radius [{workspace_radius_lower}, {workspace_radius_upper}], "
                f"got {self._reposition_comfort_radius}"
            )

    def _init_buffers(self):
        super()._init_buffers()
        self._task_stage_best = torch.zeros(
            self.num_envs, 6, dtype=torch.float32, device=self.device
        )
        self._task_stage_progress_initialized = torch.zeros(
            self.num_envs, 6, dtype=torch.bool, device=self.device
        )
        self._grasp_component_episode_best = torch.zeros(
            self.num_envs, 5, dtype=torch.float32, device=self.device
        )
        self._grasp_stable_episode_max = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._grasp_check_episode_seen = torch.zeros(
            self.num_envs, 3, dtype=torch.bool, device=self.device
        )
        self._task_stage_stable = torch.zeros(
            self.num_envs, 6, dtype=torch.long, device=self.device
        )
        self._door_overspeed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._push_gripper_contact_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._through_start_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self._task_geometry_obs = torch.zeros(
            self.num_envs, 26, dtype=torch.float32, device=self.device
        )
        self._panel_interaction_obs = torch.zeros(
            self.num_envs, 12, dtype=torch.float32, device=self.device
        )
        self._stage_progress = torch.zeros(
            self.num_envs, 6, dtype=torch.float32, device=self.device
        )
        self._grasp_contact_last_seen = torch.full(
            (self.num_envs, 2), -1000, dtype=torch.long, device=self.device
        )
        self._grasp_contact_last_points = torch.full(
            (self.num_envs, 2, 3), float("nan"), dtype=torch.float32, device=self.device
        )
        self._grasp_contact_last_forces = torch.zeros(
            self.num_envs, 2, dtype=torch.float32, device=self.device
        )

    @override
    def step(self, actor_state):
        if hasattr(self, "eval_metrics"):
            policy_mean = actor_state.get("action_mean", actor_state["actions"])
            self._eval_policy_mean.copy_(policy_mean[:, :10])
        return super().step(actor_state)

    def enable_unsafe_force_diagnostics(self):
        """Enable reset-before event capture without changing task behavior."""
        self._unsafe_force_event_recorder = UnsafeForceEventRecorder(
            self.num_envs, self.device
        )
        self._unsafe_force_policy_mean = torch.zeros(
            self.num_envs, 10, dtype=torch.float32, device=self.device
        )
        self._unsafe_force_sampled_action = torch.zeros_like(
            self._unsafe_force_policy_mean
        )

    def set_unsafe_force_policy_step(self, policy_mean, sampled_action):
        if not hasattr(self, "_unsafe_force_event_recorder"):
            return
        self._unsafe_force_policy_mean.copy_(policy_mean[:, :10])
        self._unsafe_force_sampled_action.copy_(sampled_action[:, :10])

    def export_unsafe_force_diagnostics(self, path, metadata):
        if not hasattr(self, "_unsafe_force_event_recorder"):
            raise RuntimeError("unsafe-force diagnostics were not enabled")
        self._unsafe_force_event_recorder.save(path, metadata)

    def _door_task_basis(self):
        return build_door_task_basis(
            self._frame_normal, self._frame_tangent, self.door_open_lr
        )

    def _canonical_base_action_to_physical_command(self, canonical_base_action):
        """Convert canonical normalized base action to physical body commands."""
        task_basis, task_handedness = self._door_task_basis()
        base_yaw_quat = xyzw_to_wxyz(self._get_b2z1_base_yaw_quat())
        canonical_linear = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        canonical_linear[:, :2] = (
            canonical_base_action[:, :2] * self._canonical_action_scale[:, :2]
        )
        linear_world = task_polar_to_world(canonical_linear, task_basis)
        linear_body = quat_apply(quat_inv(base_yaw_quat), linear_world)
        canonical_angular = torch.zeros_like(canonical_linear)
        canonical_angular[:, 2] = (
            canonical_base_action[:, 2] * self._canonical_action_scale[:, 2]
        )
        angular_world = task_axial_to_world(
            canonical_angular, task_basis, task_handedness
        )
        return torch.cat(
            [linear_body[:, :2], angular_world[:, 2:3]], dim=-1
        )

    def _canonical_action_to_physical_commands(
        self, canonical_action, current_physical_commands
    ):
        """Apply one canonical policy action without mutating environment state."""
        task_basis, task_handedness = self._door_task_basis()
        physical_base_command = self._canonical_base_action_to_physical_command(
            canonical_action[:, :3]
        )
        current_goal_position, current_goal_quat_xyzw = (
            self._compute_b2z1_ee_goal_world_from_commands(
                current_physical_commands
            )
        )
        canonical_position_step = (
            canonical_action[:, 3:6]
            * self._canonical_action_scale[:, 3:6]
            * self._delta_action_scale
        )
        canonical_rotation_step = (
            canonical_action[:, 6:9]
            * self._canonical_action_scale[:, 6:9]
            * self._delta_action_scale
        )
        goal_position = current_goal_position + task_polar_to_world(
            canonical_position_step, task_basis
        )
        rotation_step_world = task_axial_to_world(
            canonical_rotation_step, task_basis, task_handedness
        )
        goal_quat = quat_mul(
            axis_angle_to_quaternion(rotation_step_world),
            xyzw_to_wxyz(current_goal_quat_xyzw),
        )
        position_command, orientation_command, _ = (
            self._compute_b2z1_commands_for_world_ee_goal(
                goal_position, wxyz_to_xyzw(goal_quat)
            )
        )
        physical_commands = current_physical_commands.clone()
        physical_commands[:, :3] = physical_base_command
        physical_commands[:, 3:6] = position_command
        physical_commands[:, 6:9] = orientation_command
        physical_commands[:, 9] += (
            canonical_action[:, 9]
            * self._canonical_action_scale[:, 9]
            * self._delta_action_scale
        )
        return torch.maximum(
            torch.minimum(physical_commands, self._command_upper),
            self._command_lower,
        )

    @override
    def _integrate_policy_delta_actions(
        self, bounded_delta_actions, delta_actions_clip
    ):
        """Apply canonical door-task deltas to physical B2Z1 commands."""
        current_physical_commands = self._delta_actions * self._command_scale
        physical_commands = self._canonical_action_to_physical_commands(
            bounded_delta_actions, current_physical_commands
        )
        normalized_commands = physical_commands / self._command_scale
        self._delta_actions.copy_(
            normalized_commands.clamp(-delta_actions_clip, delta_actions_clip)
        )

    @override
    def _pre_compute_observations_callback(self, env_ids=None):
        super()._pre_compute_observations_callback(env_ids)
        self._update_task_stage_metrics()

    def _compute_grasp_orientation(self, normal, tangent):
        """Build the mirrored grasp frame required by the horizontal handle."""
        target_x = -F.normalize(normal, dim=-1)
        target_y = tangent * self.door_open_lr[:, None]
        target_y = F.normalize(
            target_y - target_x * torch.sum(target_y * target_x, dim=-1, keepdim=True),
            dim=-1,
        )
        target_z = F.normalize(torch.cross(target_x, target_y, dim=-1), dim=-1)
        return quat_from_matrix(torch.stack((target_x, target_y, target_z), dim=-1))

    def _align_gripper_insertion_axis_to_surface(self, gripper_quat, normal):
        """Minimally align the gripper insertion axis with the panel normal."""
        unit_x = torch.zeros(self.num_envs, 3, device=self.device)
        unit_x[:, 0] = 1.0
        source = F.normalize(quat_apply(gripper_quat, unit_x), dim=-1)
        target = -F.normalize(normal, dim=-1)
        dot = torch.sum(source * target, dim=-1).clamp(-1.0, 1.0)
        cross = torch.cross(source, target, dim=-1)

        delta = torch.cat([(1.0 + dot)[:, None], cross], dim=-1)
        opposite = dot < -0.9999
        if torch.any(opposite):
            candidate_x = torch.cross(source, unit_x, dim=-1)
            unit_y = torch.zeros_like(unit_x)
            unit_y[:, 1] = 1.0
            candidate_y = torch.cross(source, unit_y, dim=-1)
            use_y = torch.linalg.vector_norm(candidate_x, dim=-1) < 1.0e-5
            opposite_axis = torch.where(use_y[:, None], candidate_y, candidate_x)
            opposite_axis = F.normalize(opposite_axis, dim=-1)
            opposite_delta = torch.cat(
                [torch.zeros(self.num_envs, 1, device=self.device), opposite_axis],
                dim=-1,
            )
            delta = torch.where(opposite[:, None], opposite_delta, delta)

        delta = F.normalize(delta, dim=-1)
        return quat_mul(delta, gripper_quat)

    def _update_task_stage_metrics(self):
        door = self.simulator.scene.articulations["door"]
        root_state = self.simulator.get_task_root_state("door")
        frame_pos = root_state[:, :3]
        frame_quat = root_state[:, 3:7]
        panel_pos = door.data.body_pos_w[:, self._door_panel_idx]
        panel_quat = door.data.body_quat_w[:, self._door_panel_idx]

        unit_x = torch.zeros(self.num_envs, 3, device=self.device)
        unit_x[:, 0] = 1.0
        unit_y = torch.zeros_like(unit_x)
        unit_y[:, 1] = 1.0
        unit_z = torch.zeros_like(unit_x)
        unit_z[:, 2] = 1.0
        self._frame_normal = F.normalize(quat_apply(frame_quat, unit_x), dim=-1)
        self._frame_tangent = F.normalize(quat_apply(frame_quat, unit_y), dim=-1)
        self._through_direction = -self._frame_normal
        self._panel_normal = F.normalize(quat_apply(panel_quat, unit_x), dim=-1)
        self._panel_tangent = F.normalize(quat_apply(panel_quat, unit_y), dim=-1)

        handle_pos = self._compute_grasp_target()
        gripper_pos = self.simulator._rigid_body_pos[:, self.gripper_idx]
        gripper_quat = xyzw_to_wxyz(self.simulator._rigid_body_rot[:, self.gripper_idx])
        gripper_axis = F.normalize(quat_apply(gripper_quat, unit_x), dim=-1)

        link06_pos = self.simulator._rigid_body_pos[:, self._link06_idx]
        link06_quat = xyzw_to_wxyz(
            self.simulator._rigid_body_rot[:, self._link06_idx]
        )
        mover_pos = self.simulator._robot.data.body_pos_w[
            :, self._gripper_mover_articulation_idx
        ]
        mover_quat = self.simulator._robot.data.body_quat_w[
            :, self._gripper_mover_articulation_idx
        ]
        fixed_inner_center = link06_pos + quat_apply(
            link06_quat,
            self._fixed_inner_center_link06.expand(self.num_envs, -1),
        )
        moving_inner_center = mover_pos + quat_apply(
            mover_quat, self._moving_inner_center.expand(self.num_envs, -1)
        )
        self._fixed_inner_center = fixed_inner_center
        self._moving_inner_center_world = moving_inner_center
        self._grasp_channel_center = 0.5 * (
            fixed_inner_center + moving_inner_center
        )
        grasp_depth = (
            self._FIXED_POCKET_PROXIMAL_DEPTH
            + self.door_handle_radius
            + 0.0015
        )
        grasp_depth_point = gripper_pos + grasp_depth[:, None] * gripper_axis
        depth_delta = torch.sum(
            (grasp_depth_point - self._grasp_channel_center) * self._frame_normal,
            dim=-1,
        )
        self._grasp_pocket_pos = (
            self._grasp_channel_center + depth_delta[:, None] * self._frame_normal
        )
        # APPROACH and GRASP share this physical channel center. Switching from
        # a fixed local approximation at the stage boundary displaced mirrored
        # jaw configurations vertically into the handle.
        self._pocket_pos = self._grasp_pocket_pos
        self._gripper_axis_error = torch.acos(
            torch.sum(gripper_axis * self._through_direction, dim=-1).clamp(-1.0, 1.0)
        )

        root_pos = self.simulator.robot_root_states[:, :3]
        root_quat = xyzw_to_wxyz(self.simulator.robot_root_states[:, 3:7])
        root_forward = quat_apply(root_quat, unit_x)
        root_forward[:, 2] = 0.0
        desired_forward = self._through_direction.clone()
        desired_forward[:, 2] = 0.0
        root_forward = F.normalize(root_forward, dim=-1)
        desired_forward = F.normalize(desired_forward, dim=-1)
        heading_dot = torch.sum(root_forward * desired_forward, dim=-1).clamp(-1.0, 1.0)
        heading_cross = (
            root_forward[:, 0] * desired_forward[:, 1]
            - root_forward[:, 1] * desired_forward[:, 0]
        )
        self._base_yaw_error_signed = torch.atan2(heading_cross, heading_dot)
        self._base_yaw_error = torch.abs(self._base_yaw_error_signed)

        # The first-stage goal is a physical station measured from the actual
        # handle center. It is an outcome criterion, never an action target.
        operation_target = (
            handle_pos + self._operation_station_distance * self._frame_normal
        )
        operation_target[:, 2] = root_pos[:, 2]
        operation_delta = root_pos - operation_target
        self._operation_delta_world = operation_delta
        self._operation_error = self._vector_in_door_frame(operation_delta)[:, :2]

        pocket_delta = self._pocket_pos - handle_pos
        self._pocket_error_frame = self._vector_in_door_frame(pocket_delta)

        _, _, root_yaw = euler_xyz_from_quat(root_quat)
        base_yaw_quat = quat_from_euler_xyz(
            torch.zeros_like(root_yaw), torch.zeros_like(root_yaw), root_yaw
        )
        target_ee_quat = self._compute_grasp_orientation(
            self._frame_normal, self._frame_tangent
        )
        grasp_target_origin = gripper_pos - (self._grasp_pocket_pos - handle_pos)
        self._grasp_target_origin = grasp_target_origin
        grasp_orientation_delta = quat_mul(gripper_quat, quat_inv(target_ee_quat))
        self._grasp_ee_orientation_error = axis_angle_from_quat(
            grasp_orientation_delta
        ).norm(dim=-1)

        finger_forces = self._get_object_to_hand_contact_forces().norm(dim=-1)
        canonical_finger_forces = canonicalize_jaw_order(
            finger_forces, self.door_open_lr
        )
        self._finger_handle_forces = finger_forces
        self._handle_opening_force = self._get_obs_handle_contact_force()[
            :, 0
        ].clamp(min=0.0)
        self._handle_contact_free = torch.all(finger_forces < 1.0, dim=-1)
        self._panel_gripper_force = self._get_panel_gripper_force()
        self._panel_contact_free = self._panel_gripper_force < 1.0
        self._update_precise_grasp_metrics(handle_pos, gripper_pos)
        pushing = self.stage_buf == self.STAGE_PUSH_OPEN
        self._push_gripper_contact_seen |= pushing & (self._panel_gripper_force > 1.0)

        panel_center_offset = unit_z * (0.5 * self.door_height[:, None])
        self._panel_center = panel_pos + quat_apply(panel_quat, panel_center_offset)
        # The 30 mm panel-ready clearance belongs to the foremost gripper
        # collider, not the gripper_link origin.  Omitting the morphology
        # support offset drove the closed fingers roughly 100 mm into the
        # panel while REPOSITION simultaneously required contact-free motion.
        panel_ready_origin_offset = (
            self._DOOR_PANEL_HALF_THICKNESS
            + self._GRIPPER_PANEL_SUPPORT_OFFSET
            + self._PANEL_READY_CLEARANCE
        )
        self._push_ready_point = (
            self._panel_center + panel_ready_origin_offset * self._panel_normal
        )
        # Measure the reposition station from the outside-handle plane. This is
        # a physical completion criterion and does not prescribe how to reach it.
        reposition_station_distance = (
            self._operation_station_distance
            + 0.5 * self.door_axle_length
        )
        nominal_reposition_target = (
            frame_pos
            + reposition_station_distance[:, None] * self._frame_normal
        )
        nominal_reposition_target[:, 2] = root_pos[:, 2]

        # Keep the base centred on and perpendicular to the doorway, while
        # advancing it along the frame normal when panel rotation would put
        # the push-ready point outside the arm's comfortable workspace.
        desired_base_lateral = -self._frame_tangent
        nominal_push_delta = self._push_ready_point - nominal_reposition_target
        goal_center_offset = self._b2z1_goal_center_offset.reshape(-1)
        nominal_goal_cart = torch.stack(
            [
                torch.sum(nominal_push_delta * self._through_direction, dim=-1)
                - goal_center_offset[0],
                torch.sum(nominal_push_delta * desired_base_lateral, dim=-1)
                - goal_center_offset[1],
                self._push_ready_point[:, 2] - goal_center_offset[2],
            ],
            dim=-1,
        )
        transverse_radius_sq = torch.sum(
            torch.square(nominal_goal_cart[:, 1:]), dim=-1
        )
        comfortable_forward = torch.sqrt(
            (
                self._reposition_comfort_radius**2 - transverse_radius_sq
            ).clamp(min=0.0)
        )
        reposition_advance = (
            nominal_goal_cart[:, 0] - comfortable_forward
        ).clamp(min=0.0)
        reposition_target = (
            nominal_reposition_target
            + reposition_advance[:, None] * self._through_direction
        )
        reposition_delta = root_pos - reposition_target
        self._reposition_error = self._vector_in_door_frame(reposition_delta)[:, :2]
        self._push_ready_distance = torch.linalg.vector_norm(
            gripper_pos - self._push_ready_point, dim=-1
        )

        # Reposition succeeds on the measured EE pose relative to the panel.
        # The policy remains free to choose any collision-free path to it.
        self._push_ready_quat = self._align_gripper_insertion_axis_to_surface(
            gripper_quat, self._panel_normal
        )
        push_ready_goal_origin = torch.zeros_like(root_pos)
        push_ready_goal_origin[:, :2] = root_pos[:, :2]
        self._reposition_raw_goal_cart = (
            quat_apply(
                quat_inv(base_yaw_quat),
                self._push_ready_point - push_ready_goal_origin,
            )
            - self._b2z1_goal_center_offset
        )
        self._reposition_raw_goal_sphere = self._cart_to_sphere(
            self._reposition_raw_goal_cart
        )
        reposition_clipped_goal_sphere = torch.maximum(
            torch.minimum(
                self._reposition_raw_goal_sphere,
                self._b2z1_ee_goal_sphere_upper,
            ),
            self._b2z1_ee_goal_sphere_lower,
        )
        self._reposition_goal_sphere_clip = (
            reposition_clipped_goal_sphere - self._reposition_raw_goal_sphere
        )
        self._reposition_nominal_goal_radius = torch.linalg.vector_norm(
            nominal_goal_cart, dim=-1
        )
        self._reposition_station_advance = reposition_advance
        target_goal_cart = nominal_goal_cart.clone()
        target_goal_cart[:, 0] -= reposition_advance
        self._reposition_target_goal_radius = torch.linalg.vector_norm(
            target_goal_cart, dim=-1
        )
        self._reposition_transverse_goal_radius = torch.sqrt(
            transverse_radius_sq
        )
        push_ready_orientation_delta = quat_mul(
            gripper_quat, quat_inv(self._push_ready_quat)
        )
        self._push_ready_orientation_error = axis_angle_from_quat(
            push_ready_orientation_delta
        ).norm(dim=-1)
        self._push_ready_gripper_error = torch.abs(
            self.simulator.dof_pos[:, self.gripper_dof_idx] - self._gripper_closed_pos
        ).mean(dim=-1)
        panel_task_basis, panel_task_handedness = build_door_task_basis(
            self._panel_normal, self._panel_tangent, self.door_open_lr
        )
        push_ready_delta = gripper_pos - self._push_ready_point
        self._push_ready_error_frame = world_polar_to_task(
            push_ready_delta, panel_task_basis
        )
        gripper_velocity = self.simulator._rigid_body_vel[:, self.gripper_idx, :3]
        panel_position = door.data.body_pos_w[:, self._door_panel_idx]
        panel_linear_velocity = door.data.body_lin_vel_w[:, self._door_panel_idx]
        panel_angular_velocity = door.data.body_ang_vel_w[:, self._door_panel_idx]
        panel_point_velocity = panel_linear_velocity + torch.linalg.cross(
            panel_angular_velocity,
            gripper_pos - panel_position,
            dim=-1,
        )
        gripper_panel_relative_velocity = gripper_velocity - panel_point_velocity
        self._gripper_panel_relative_velocity_frame = world_polar_to_task(
            gripper_panel_relative_velocity, panel_task_basis
        )
        self._gripper_panel_relative_speed = torch.linalg.vector_norm(
            gripper_panel_relative_velocity, dim=-1
        )
        push_ready_orientation_error_world = axis_angle_from_quat(
            quat_mul(gripper_quat, quat_inv(self._push_ready_quat))
        )
        push_ready_orientation_error_task = world_axial_to_task(
            push_ready_orientation_error_world,
            panel_task_basis,
            panel_task_handedness,
        )
        self._panel_gripper_clearance = (
            torch.sum((gripper_pos - self._panel_center) * self._panel_normal, dim=-1)
            - self._DOOR_PANEL_HALF_THICKNESS
            - self._GRIPPER_PANEL_SUPPORT_OFFSET
        )
        self._panel_gripper_approach_speed = torch.sum(
            (panel_point_velocity - gripper_velocity) * self._panel_normal, dim=-1
        )

        door_pos = self.simulator.get_task_dof_pos("door")
        door_vel = self.simulator.get_task_dof_vel("door")
        self._hinge_angle = torch.abs(door_pos[:, 0])
        self._hinge_speed = torch.abs(door_vel[:, 0])
        self._hinge_opening_speed = torch.sign(door_pos[:, 0]) * door_vel[:, 0]
        self._latch_pos = door_pos[:, 2]
        self._gap_angle = self._compute_gap_angle()
        self._through_distance = torch.sum(
            (root_pos - self._through_start_pos) * self._through_direction, dim=-1
        )

        arm_error = torch.abs(
            self.simulator.dof_pos[:, self.arm_dof_idx] - self._middle_arm
        )
        self._middle_arm_joint_error = arm_error
        self._middle_arm_error = arm_error.max(dim=-1).values
        self._middle_arm_joint_speed = torch.abs(
            self.simulator.dof_vel[:, self.arm_dof_idx]
        )
        self._middle_arm_speed = self._middle_arm_joint_speed.max(dim=-1).values
        self._arm_safely_stowed = (
            (self._middle_arm_error < 0.05) & (self._middle_arm_speed < 0.08)
        )

        shoulder_pos = self.simulator._rigid_body_pos[:, self._link01_idx]
        shoulder_local = quat_apply(quat_inv(panel_quat), shoulder_pos - panel_pos)
        overlaps_panel = (
            (torch.abs(shoulder_local[:, 1]) < 0.5 * self.door_width + 0.175)
            & (shoulder_local[:, 2] > -0.175)
            & (shoulder_local[:, 2] < self.door_height + 0.175)
        )
        plane_clearance = torch.abs(shoulder_local[:, 0]) - 0.175
        self._shoulder_panel_clearance = torch.where(
            overlaps_panel, plane_clearance, torch.ones_like(plane_clearance)
        )
        self._panel_interaction_obs = torch.cat(
            [
                self._push_ready_error_frame,
                push_ready_orientation_error_task,
                self._gripper_panel_relative_velocity_frame,
                (self._panel_gripper_force / 100.0).clamp(max=10.0)[:, None],
                self._push_gripper_contact_seen.float()[:, None],
                self._shoulder_panel_clearance[:, None],
            ],
            dim=-1,
        )
        self._base_motion = self._get_base_motion_norm()
        ee_velocity = self.simulator._rigid_body_vel[:, self.gripper_idx, :3]
        self._ee_speed = torch.linalg.vector_norm(ee_velocity, dim=-1)
        ee_velocity_frame = self._vector_in_door_frame(ee_velocity)

        self._update_stage_progress()
        self._update_stage_stability()
        self._validate_staged_reset_initial_states()
        self._update_training_diagnostics()
        self._task_geometry_obs = torch.cat(
            [
                self._operation_error,
                self._base_yaw_error[:, None],
                ee_velocity_frame,
                self._gripper_axis_error[:, None],
                self._pocket_error_frame,
                canonical_finger_forces,
                self._latch_pos[:, None],
                self._gap_angle[:, None],
                self._reposition_error,
                self._hinge_angle[:, None],
                self._hinge_speed[:, None],
                self._grasp_contact_obs,
            ],
            dim=-1,
        )

    def _update_training_diagnostics(self):
        """Expose stage-local learning signals without changing task behavior."""
        for stage_idx, stage_name in enumerate(self._STAGE_NAMES):
            active = self.stage_buf == stage_idx
            active_count = active.float().sum().clamp(min=1.0)
            self.log_dict[f"stage/{stage_name}/occupancy"] = active.float().mean()
            self.log_dict[f"stage/{stage_name}/reached"] = (
                self.current_max_stage_buf >= stage_idx
            ).float().mean()
            self.log_dict[f"stage/{stage_name}/progress"] = (
                self._stage_progress[:, stage_idx] * active.float()
            ).sum() / active_count
            for side_name, side_mask in (
                ("hinge_left", self.door_open_lr > 0.0),
                ("hinge_right", self.door_open_lr < 0.0),
            ):
                side_count = side_mask.float().sum().clamp(min=1.0)
                prefix = f"stage/{stage_name}/{side_name}"
                self.log_dict[f"{prefix}/occupancy"] = (
                    active & side_mask
                ).float().sum() / side_count
                self.log_dict[f"{prefix}/reached"] = (
                    (self.current_max_stage_buf >= stage_idx) & side_mask
                ).float().sum() / side_count

        open_latch = self.stage_buf == self.STAGE_OPEN_LATCH
        open_latch_count = open_latch.float().sum().clamp(min=1.0)

        def open_latch_mean(value):
            return (value * open_latch.float()).sum() / open_latch_count

        latch_progress = (self._latch_pos / 0.02).clamp(0.0, 1.0)
        gap_progress = (
            self._hinge_angle / self._gap_angle.clamp(min=1.0e-4)
        ).clamp(0.0, 1.0)
        self.log_dict["stage/open_latch/latch_position"] = open_latch_mean(
            self._latch_pos
        )
        self.log_dict["stage/open_latch/hinge_angle"] = open_latch_mean(
            self._hinge_angle
        )
        self.log_dict["stage/open_latch/gap_angle"] = open_latch_mean(
            self._gap_angle
        )
        self.log_dict["stage/open_latch/latch_progress"] = open_latch_mean(
            latch_progress
        )
        self.log_dict["stage/open_latch/gap_progress"] = open_latch_mean(
            gap_progress
        )
        self.log_dict["stage/open_latch/latch_released"] = open_latch_mean(
            (self._latch_pos >= 0.02).float()
        )
        self.log_dict["stage/open_latch/gap_open"] = open_latch_mean(
            (self._hinge_angle >= self._gap_angle).float()
        )
        self.log_dict["stage/open_latch/episode_best_progress"] = open_latch_mean(
            self._task_stage_best[:, self.STAGE_OPEN_LATCH]
        )
        self.log_dict["stage/open_latch/precise_bilateral_grasp"] = open_latch_mean(
            self._precise_bilateral_grasp.float()
        )
        self.log_dict["stage/open_latch/any_valid_handle_contact"] = open_latch_mean(
            self._grasp_on_lever_capsule.any(dim=-1).float()
        )
        self.log_dict["stage/open_latch/opposed_handle_contact"] = open_latch_mean(
            self._grasp_opposed_contact.float()
        )
        self.log_dict["stage/open_latch/contact_quality"] = open_latch_mean(
            self._grasp_contact_quality
        )
        self.log_dict["stage/open_latch/retention_quality"] = open_latch_mean(
            self._open_latch_retention_quality
        )
        for component_idx, component_name in enumerate(
            ("pocket_retention", "axis_retention", "contact_retention")
        ):
            self.log_dict[f"stage/open_latch/{component_name}"] = open_latch_mean(
                self._open_latch_retention_components[:, component_idx]
            )
        self.log_dict["stage/open_latch/mean_handle_force"] = open_latch_mean(
            self._finger_handle_forces.mean(dim=-1)
        )
        self.log_dict["stage/open_latch/max_handle_force"] = open_latch_mean(
            self._finger_handle_forces.max(dim=-1).values
        )
        self.log_dict["stage/open_latch/base_motion"] = open_latch_mean(
            self._base_motion
        )
        self.log_dict["stage/open_latch/ee_speed"] = open_latch_mean(self._ee_speed)
        self.log_dict["stage/open_latch/time_steps"] = open_latch_mean(
            self.actual_time_in_stage_buf.float()
        )
        for action_idx, action_name in enumerate(
            (
                "base_x",
                "base_y",
                "base_yaw",
                "ee_x",
                "ee_y",
                "ee_z",
                "ee_roll",
                "ee_pitch",
                "ee_yaw",
                "gripper",
            )
        ):
            raw_action = self._last_delta_actions[:, action_idx]
            self.log_dict[f"stage/open_latch/action/{action_name}"] = open_latch_mean(
                raw_action
            )
            self.log_dict[
                f"stage/open_latch/action_abs/{action_name}"
            ] = open_latch_mean(torch.abs(raw_action))
            self.log_dict[
                f"stage/open_latch/command/{action_name}"
            ] = open_latch_mean(self._b2z1_commands[:, action_idx])

        reposition = self.stage_buf == self.STAGE_REPOSITION
        reposition_count = reposition.float().sum().clamp(min=1.0)

        def reposition_mean(value):
            return (value * reposition.float()).sum() / reposition_count

        reposition_distance = torch.linalg.vector_norm(
            self._reposition_error, dim=-1
        )
        reposition_checks = {
            "handle_contact_free": self._handle_contact_free,
            "panel_contact_free": self._panel_contact_free,
            "base_position_ok": reposition_distance
            < self._REPOSITION_BASE_POSITION_TOLERANCE,
            "base_yaw_ok": self._base_yaw_error
            < self._REPOSITION_BASE_YAW_TOLERANCE,
            "push_ready_distance_ok": self._push_ready_distance
            < self._REPOSITION_EE_POSITION_TOLERANCE,
            "push_ready_orientation_ok": self._push_ready_orientation_error
            < self._REPOSITION_EE_ORIENTATION_TOLERANCE,
            "gripper_closed_ok": self._push_ready_gripper_error
            < self._REPOSITION_GRIPPER_TOLERANCE,
            "base_still": self._base_motion
            < self._REPOSITION_BASE_MOTION_TOLERANCE,
            "ee_still": self._ee_speed < self._REPOSITION_EE_SPEED_TOLERANCE,
        }
        self.log_dict["stage/reposition/base_position_error"] = reposition_mean(
            reposition_distance
        )
        self.log_dict["stage/reposition/base_normal_error_signed"] = reposition_mean(
            self._reposition_error[:, 0]
        )
        self.log_dict["stage/reposition/base_tangent_error_signed"] = reposition_mean(
            self._reposition_error[:, 1]
        )
        self.log_dict["stage/reposition/base_yaw_error"] = reposition_mean(
            self._base_yaw_error
        )
        self.log_dict["stage/reposition/push_ready_distance"] = reposition_mean(
            self._push_ready_distance
        )
        self.log_dict["stage/reposition/push_ready_orientation_error"] = (
            reposition_mean(self._push_ready_orientation_error)
        )
        self.log_dict["stage/reposition/gripper_closed_error"] = reposition_mean(
            self._push_ready_gripper_error
        )
        self.log_dict["stage/reposition/nominal_goal_radius"] = reposition_mean(
            self._reposition_nominal_goal_radius
        )
        self.log_dict["stage/reposition/station_advance"] = reposition_mean(
            self._reposition_station_advance
        )
        self.log_dict["stage/reposition/target_goal_radius"] = reposition_mean(
            self._reposition_target_goal_radius
        )
        self.log_dict[
            "stage/reposition/transverse_goal_radius"
        ] = reposition_mean(self._reposition_transverse_goal_radius)
        for axis, axis_name in enumerate(("x", "y", "z")):
            self.log_dict[
                f"stage/reposition/raw_goal_cart_{axis_name}"
            ] = reposition_mean(self._reposition_raw_goal_cart[:, axis])
        for axis, axis_name in enumerate(("radius", "pitch", "yaw")):
            self.log_dict[
                f"stage/reposition/raw_goal_sphere_{axis_name}"
            ] = reposition_mean(self._reposition_raw_goal_sphere[:, axis])
            self.log_dict[
                f"stage/reposition/goal_sphere_clip_{axis_name}"
            ] = reposition_mean(self._reposition_goal_sphere_clip[:, axis])
        self.log_dict["stage/reposition/hinge_angle"] = reposition_mean(
            self._hinge_angle
        )
        self.log_dict["stage/reposition/hinge_speed"] = reposition_mean(
            self._hinge_speed
        )
        for axis, axis_name in enumerate(("normal", "tangent", "vertical")):
            self.log_dict[
                f"stage/reposition/push_ready_error_{axis_name}_signed"
            ] = reposition_mean(self._push_ready_error_frame[:, axis])
        self.log_dict["stage/reposition/base_motion"] = reposition_mean(
            self._base_motion
        )
        self.log_dict["stage/reposition/ee_speed"] = reposition_mean(self._ee_speed)
        self.log_dict["stage/reposition/handle_force"] = reposition_mean(
            self._finger_handle_forces.max(dim=-1).values
        )
        self.log_dict["stage/reposition/panel_force"] = reposition_mean(
            self._panel_gripper_force
        )
        self.log_dict["stage/reposition/stable_steps"] = reposition_mean(
            self._task_stage_stable[:, self.STAGE_REPOSITION].float()
        )
        self.log_dict["stage/reposition/time_steps"] = reposition_mean(
            self.actual_time_in_stage_buf.float()
        )
        for check_name, condition in reposition_checks.items():
            self.log_dict[f"stage/reposition/{check_name}"] = reposition_mean(
                condition.float()
            )
        all_reposition_checks = torch.stack(
            tuple(reposition_checks.values()), dim=-1
        ).all(dim=-1)
        self.log_dict["stage/reposition/all_checks_now"] = reposition_mean(
            all_reposition_checks.float()
        )
        for action_idx, action_name in enumerate(
            (
                "base_x",
                "base_y",
                "base_yaw",
                "ee_x",
                "ee_y",
                "ee_z",
                "ee_roll",
                "ee_pitch",
                "ee_yaw",
                "gripper",
            )
        ):
            raw_action = self._last_delta_actions[:, action_idx]
            self.log_dict[f"stage/reposition/action/{action_name}"] = reposition_mean(
                raw_action
            )
            self.log_dict[
                f"stage/reposition/action_abs/{action_name}"
            ] = reposition_mean(torch.abs(raw_action))
            self.log_dict[
                f"stage/reposition/command/{action_name}"
            ] = reposition_mean(self._b2z1_commands[:, action_idx])

        push_open = self.stage_buf == self.STAGE_PUSH_OPEN
        push_open_count = push_open.float().sum().clamp(min=1.0)

        def push_open_mean(value):
            return (value * push_open.float()).sum() / push_open_count

        self.log_dict["stage/push_open/hinge_angle"] = push_open_mean(
            self._hinge_angle
        )
        self.log_dict["stage/push_open/hinge_speed"] = push_open_mean(
            self._hinge_speed
        )
        self.log_dict["stage/push_open/hinge_opening_speed"] = push_open_mean(
            self._hinge_opening_speed
        )
        self.log_dict["stage/push_open/panel_force"] = push_open_mean(
            self._panel_gripper_force
        )
        self.log_dict["stage/push_open/contact_seen"] = push_open_mean(
            self._push_gripper_contact_seen.float()
        )
        self.log_dict["stage/push_open/contact_progress"] = push_open_mean(
            self._push_contact_progress
        )
        self.log_dict["stage/push_open/shoulder_clearance"] = push_open_mean(
            self._shoulder_panel_clearance
        )
        self.log_dict["stage/push_open/base_motion"] = push_open_mean(
            self._base_motion
        )
        self.log_dict["stage/push_open/ee_speed"] = push_open_mean(self._ee_speed)
        self.log_dict["stage/push_open/panel_clearance"] = push_open_mean(
            self._panel_gripper_clearance
        )
        self.log_dict["stage/push_open/panel_approach_speed"] = push_open_mean(
            self._panel_gripper_approach_speed
        )
        self.log_dict["stage/push_open/panel_relative_speed"] = push_open_mean(
            self._gripper_panel_relative_speed
        )
        self.log_dict[
            "stage/push_open/panel_relative_speed_penalty"
        ] = push_open_mean(
            excessive_relative_speed_penalty(
                self._gripper_panel_relative_velocity_frame,
                safe_speed=self._PUSH_SAFE_PANEL_RELATIVE_SPEED,
            )
        )
        self.log_dict[
            "stage/push_open/push_ready_orientation_error"
        ] = push_open_mean(self._push_ready_orientation_error)
        for axis, axis_name in enumerate(("normal", "tangent", "vertical")):
            self.log_dict[
                f"stage/push_open/push_ready_error_{axis_name}_signed"
            ] = push_open_mean(self._push_ready_error_frame[:, axis])
        self.log_dict["stage/push_open/time_steps"] = push_open_mean(
            self.actual_time_in_stage_buf.float()
        )
        push_entry = push_open & (self.actual_time_in_stage_buf < 5)
        push_entry_count = push_entry.float().sum().clamp(min=1.0)

        def push_entry_mean(value):
            return (value * push_entry.float()).sum() / push_entry_count

        self.log_dict["stage/push_open/entry/fraction"] = push_entry.float().mean()
        self.log_dict["stage/push_open/entry/panel_clearance"] = push_entry_mean(
            self._panel_gripper_clearance
        )
        self.log_dict["stage/push_open/entry/panel_approach_speed"] = push_entry_mean(
            self._panel_gripper_approach_speed
        )
        self.log_dict["stage/push_open/entry/push_ready_distance"] = push_entry_mean(
            self._push_ready_distance
        )
        self.log_dict[
            "stage/push_open/entry/push_ready_orientation_error"
        ] = push_entry_mean(self._push_ready_orientation_error)
        self.log_dict["stage/push_open/entry/command_ee_x"] = push_entry_mean(
            self._b2z1_commands[:, 3]
        )
        self.log_dict["stage/push_open/entry/action_ee_x"] = push_entry_mean(
            self._last_delta_actions[:, 3]
        )
        for action_idx, action_name in enumerate(
            (
                "base_x",
                "base_y",
                "base_yaw",
                "ee_x",
                "ee_y",
                "ee_z",
                "ee_roll",
                "ee_pitch",
                "ee_yaw",
                "gripper",
            )
        ):
            raw_action = self._last_delta_actions[:, action_idx]
            self.log_dict[f"stage/push_open/action/{action_name}"] = push_open_mean(
                raw_action
            )
            self.log_dict[
                f"stage/push_open/action_abs/{action_name}"
            ] = push_open_mean(torch.abs(raw_action))
            self.log_dict[
                f"stage/push_open/command/{action_name}"
            ] = push_open_mean(self._b2z1_commands[:, action_idx])

        operation_dist = torch.linalg.vector_norm(self._operation_error, dim=-1)

        walk = self.stage_buf == self.STAGE_WALK_TO_DOOR
        walk_count = walk.float().sum().clamp(min=1.0)
        walk_checks = {
            "base_position_ok": operation_dist < 0.16,
            "base_yaw_ok": self._base_yaw_error < 0.08,
            "contact_free": self._handle_contact_free & self._panel_contact_free,
            "base_still": self._base_motion < 0.05,
            "middle_arm_ok": self._middle_arm_error < 0.05,
            "arm_still": self._middle_arm_speed < 0.08,
        }
        for name, condition in walk_checks.items():
            self.log_dict[f"stage/walk_to_door/{name}"] = (
                condition.float() * walk.float()
            ).sum() / walk_count
        self.log_dict["stage/walk_to_door/all_checks_now"] = (
            torch.stack(tuple(walk_checks.values()), dim=-1).all(dim=-1).float()
            * walk.float()
        ).sum() / walk_count
        for idx, name in enumerate(
            ("base_position", "base_yaw", "middle_arm")
        ):
            self.log_dict[f"stage/walk_to_door/{name}_quality"] = (
                self._walk_component_progress[:, idx] * walk.float()
            ).sum() / walk_count

        grasp = self.stage_buf == self.STAGE_GRASP
        grasp_count = grasp.float().sum().clamp(min=1.0)
        grasp_checks = {
            "base_position_ok": operation_dist < 0.16,
            "base_yaw_ok": self._base_yaw_error < 0.08,
            "precise_bilateral_grasp": self._precise_bilateral_grasp,
        }
        grasp_check_values = torch.stack(tuple(grasp_checks.values()), dim=-1)
        self._grasp_check_episode_seen |= grasp[:, None] & grasp_check_values
        for idx, (name, condition) in enumerate(grasp_checks.items()):
            self.log_dict[f"stage/grasp/{name}"] = (
                condition.float() * grasp.float()
            ).sum() / grasp_count
            self.log_dict[f"stage/grasp/{name}_episode_seen"] = (
                self._grasp_check_episode_seen[:, idx].float().mean()
            )
        for idx, name in enumerate(
            ("base_position", "base_yaw", "pocket_position", "gripper_orientation", "contact")
        ):
            self.log_dict[f"stage/grasp/{name}_quality"] = (
                self._grasp_component_progress[:, idx] * grasp.float()
            ).sum() / grasp_count
            self.log_dict[f"stage/grasp/{name}_episode_best"] = (
                self._grasp_component_episode_best[:, idx].mean()
            )
        self.log_dict["stage/grasp/pocket_distance_quality"] = (
            self._grasp_pocket_distance_progress * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/pocket_normal_quality"] = (
            self._grasp_pocket_normal_progress * grasp.float()
        ).sum() / grasp_count
        for name, value in (
            ("path_clearance_quality", self._grasp_path_clearance_quality),
            ("path_transverse_quality", self._grasp_path_transverse_quality),
            ("direct_position_quality", self._grasp_direct_position_quality),
            ("path_between_quality", self._grasp_path_between_quality),
        ):
            self.log_dict[f"stage/grasp/{name}"] = (
                value * grasp.float()
            ).sum() / grasp_count
        self.log_dict["stage/grasp/all_checks_now"] = (
            grasp_check_values.all(dim=-1).float() * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/stable_episode_max_mean"] = (
            self._grasp_stable_episode_max.float().mean()
        )
        for steps in (1, 5, 10):
            self.log_dict[f"stage/grasp/stable_reached_{steps}"] = (
                self._grasp_stable_episode_max >= steps
            ).float().mean()

        planar_command_speed = torch.linalg.vector_norm(
            self._b2z1_commands[:, :2], dim=-1
        )
        command_body = torch.zeros(self.num_envs, 3, device=self.device)
        command_body[:, :2] = self._b2z1_commands[:, :2]
        root_quat_wxyz = xyzw_to_wxyz(self.simulator.robot_root_states[:, 3:7])
        command_velocity_w = quat_apply(root_quat_wxyz, command_body)
        root_velocity_w = self.simulator.robot_root_states[:, 7:10]
        desired_motion_w = F.normalize(-self._operation_delta_world, dim=-1)
        command_goalward_speed = torch.sum(command_velocity_w * desired_motion_w, dim=-1)
        goalward_speed = torch.sum(root_velocity_w * desired_motion_w, dim=-1)
        moving_walk = walk & (operation_dist >= 0.16)
        moving_count = moving_walk.float().sum().clamp(min=1.0)
        self.log_dict["stage/walk_to_door/normal_error_abs"] = (
            self._operation_error[:, 0].abs() * walk.float()
        ).sum() / walk_count
        self.log_dict["stage/walk_to_door/tangent_error_abs"] = (
            self._operation_error[:, 1].abs() * walk.float()
        ).sum() / walk_count
        self.log_dict["stage/walk_to_door/planar_command_speed"] = (
            planar_command_speed * moving_walk.float()
        ).sum() / moving_count
        self.log_dict["stage/walk_to_door/command_below_0p2"] = (
            (planar_command_speed < 0.2).float() * moving_walk.float()
        ).sum() / moving_count
        self.log_dict["stage/walk_to_door/actual_planar_speed"] = (
            torch.linalg.vector_norm(root_velocity_w[:, :2], dim=-1)
            * moving_walk.float()
        ).sum() / moving_count
        self.log_dict["stage/walk_to_door/command_goalward_speed"] = (
            command_goalward_speed * moving_walk.float()
        ).sum() / moving_count
        self.log_dict["stage/walk_to_door/command_goalward_alignment"] = (
            command_goalward_speed / planar_command_speed.clamp(min=1.0e-3)
            * moving_walk.float()
        ).sum() / moving_count
        self.log_dict["stage/walk_to_door/goalward_speed"] = (
            goalward_speed * moving_walk.float()
        ).sum() / moving_count
        self.log_dict["stage/grasp/bilateral_contact"] = (
            self._precise_bilateral_grasp.float() * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/contact_quality"] = (
            self._grasp_contact_quality * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/bilateral_force_progress"] = (
            self._grasp_bilateral_force_progress * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/fixed_jaw_force_quality"] = (
            self._grasp_per_jaw_force_quality[:, 0] * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/moving_jaw_force_quality"] = (
            self._grasp_per_jaw_force_quality[:, 1] * grasp.float()
        ).sum() / grasp_count
        for name, value in (
            ("aperture_quality", self._grasp_aperture_quality),
            ("between_jaws_quality", self._grasp_between_jaws_quality),
            ("closure_quality", self._grasp_closure_quality),
            ("capture_progress", self._grasp_capture_progress),
        ):
            self.log_dict[f"stage/grasp/{name}"] = (
                value * grasp.float()
            ).sum() / grasp_count
        grasp_pocket_error = self._vector_in_door_frame(
            self._pocket_pos - self._compute_grasp_target()
        )
        grasp_target_delta = self._vector_in_door_frame(
            self._grasp_target_origin - self.simulator._rigid_body_pos[:, self.gripper_idx]
        )
        grasp_pocket_offset = self._vector_in_door_frame(
            self._grasp_pocket_pos - self.simulator._rigid_body_pos[:, self.gripper_idx]
        )
        grasp_open_error = torch.abs(
            self.simulator.dof_pos[:, self.gripper_dof_idx] - self._gripper_open_pos
        ).mean(dim=-1)
        self.log_dict["stage/grasp/pocket_distance"] = (
            torch.linalg.vector_norm(grasp_pocket_error, dim=-1) * grasp.float()
        ).sum() / grasp_count
        for axis, axis_name in enumerate(("normal", "tangent", "vertical")):
            self.log_dict[f"stage/grasp/pocket_error_{axis_name}"] = (
                torch.abs(grasp_pocket_error[:, axis]) * grasp.float()
            ).sum() / grasp_count
            self.log_dict[f"stage/grasp/pocket_error_{axis_name}_signed"] = (
                grasp_pocket_error[:, axis] * grasp.float()
            ).sum() / grasp_count
        gripper_velocity = self.simulator._rigid_body_vel[:, self.gripper_idx, :3]
        gripper_velocity_frame = self._vector_in_door_frame(gripper_velocity)
        for axis, axis_name in enumerate(("normal", "tangent", "vertical")):
            self.log_dict[f"stage/grasp/gripper_velocity_{axis_name}_signed"] = (
                gripper_velocity_frame[:, axis] * grasp.float()
            ).sum() / grasp_count
            self.log_dict[f"stage/grasp/target_delta_{axis_name}_signed"] = (
                grasp_target_delta[:, axis] * grasp.float()
            ).sum() / grasp_count
            self.log_dict[f"stage/grasp/pocket_offset_{axis_name}_signed"] = (
                grasp_pocket_offset[:, axis] * grasp.float()
            ).sum() / grasp_count
        self.log_dict["stage/grasp/base_operation_distance"] = (
            torch.linalg.vector_norm(self._operation_error, dim=-1) * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/base_motion"] = (
            self._base_motion * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/ee_speed"] = (
            self._ee_speed * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/hinge_angle"] = (
            self._hinge_angle * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/latch_position"] = (
            self._latch_pos * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/gripper_open_error"] = (
            grasp_open_error * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/gripper_joint_position"] = (
            self.simulator.dof_pos[:, self.gripper_dof_idx].mean(dim=-1)
            * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/gripper_command"] = (
            self._b2z1_commands[:, 9] * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/full_orientation_error"] = (
            self._grasp_ee_orientation_error * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/axis_error"] = (
            self._gripper_axis_error * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/any_valid_contact"] = (
            self._grasp_on_lever_capsule.any(dim=-1).float() * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/both_valid_contact"] = (
            self._grasp_on_lever_capsule.all(dim=-1).float() * grasp.float()
        ).sum() / grasp_count
        self.log_dict["stage/grasp/opposed_contact"] = (
            self._grasp_opposed_contact.float() * grasp.float()
        ).sum() / grasp_count
        for check_idx, check_name in enumerate(
            (
                "accept_opposed",
                "accept_any_valid_contact",
                "accept_pocket_distance",
                "accept_pocket_normal",
                "accept_axis",
                "accept_relative_speed",
                "accept_gripper_speed",
                "accept_contact_force",
                "accept_no_unexpected_contact",
                "accept_panel_free",
            )
        ):
            self.log_dict[f"stage/grasp/{check_name}"] = (
                self._grasp_acceptance_checks[:, check_idx].float()
                * grasp.float()
            ).sum() / grasp_count
            other_checks = torch.cat(
                [
                    self._grasp_acceptance_checks[:, :check_idx],
                    self._grasp_acceptance_checks[:, check_idx + 1 :],
                ],
                dim=-1,
            ).all(dim=-1)
            self.log_dict[f"stage/grasp/all_except_{check_name}"] = (
                other_checks.float() * grasp.float()
            ).sum() / grasp_count
        self.log_dict["stage/grasp/unexpected_contact"] = (
            self._grasp_unexpected_handle_contact.any(dim=-1).float() * grasp.float()
        ).sum() / grasp_count

    def get_policy_diagnostics(self, policy_state_dict):
        """Report actor outputs and resulting commands without scripted targets."""
        action_mean = policy_state_dict["action_mean"]
        sampled_action = policy_state_dict["actions"]
        action_names = (
            "base_x",
            "base_y",
            "base_yaw",
            "ee_x",
            "ee_y",
            "ee_z",
            "ee_roll",
            "ee_pitch",
            "ee_yaw",
            "gripper",
        )
        diagnostics = {}
        for stage_idx, stage_name in enumerate(self._STAGE_NAMES):
            stage_mask = self.stage_buf == stage_idx
            for side_name, side_mask in (
                ("hinge_left", self.door_open_lr > 0.0),
                ("hinge_right", self.door_open_lr < 0.0),
            ):
                active = stage_mask & side_mask
                prefix = f"stage/{stage_name}/{side_name}"
                for action_idx, action_name in enumerate(action_names):
                    mean = action_mean[active, action_idx]
                    sampled = sampled_action[active, action_idx]
                    diagnostics[f"{prefix}/policy_mean/{action_name}"] = mean
                    diagnostics[f"{prefix}/sampled/{action_name}"] = sampled
                    diagnostics[f"{prefix}/sample_noise_abs/{action_name}"] = torch.abs(
                        sampled - mean
                    )
                    diagnostics[f"{prefix}/command/{action_name}"] = (
                        self._b2z1_commands[active, action_idx]
                    )
        return diagnostics

    def get_task_variant(self):
        """Return the door-handedness stratum used for balanced PPO statistics."""
        return (self.door_open_lr < 0.0).long()

    def _update_precise_grasp_metrics(self, handle_center, gripper_pos):
        door = self.simulator.scene.articulations["door"]
        handle_quat = door.data.body_quat_w[:, self._door_handle_idx]
        forces = torch.zeros(self.num_envs, 2, 3, device=self.device)
        points = torch.full(
            (self.num_envs, 2, 3), float("nan"), device=self.device
        )
        for finger_idx, sensor_name in enumerate(
            ("b2z1_fixed_handle_contact", "b2z1_moving_handle_contact")
        ):
            sensor = self.simulator.scene.sensors[sensor_name]
            force_matrix = sensor.data.force_matrix_w
            contact_points = sensor.data.contact_pos_w
            if force_matrix is not None and force_matrix.shape[1:3] == (1, 1):
                forces[:, finger_idx] = force_matrix[:, 0, 0]
            if contact_points is not None and contact_points.shape[1:3] == (1, 1):
                points[:, finger_idx] = contact_points[:, 0, 0]

        force_norms = torch.linalg.vector_norm(forces, dim=-1)
        handle_axis_local = torch.zeros(self.num_envs, 3, device=self.device)
        handle_axis_local[:, 1] = -1.0
        handle_axis = F.normalize(quat_apply(handle_quat, handle_axis_local), dim=-1)
        closing_axis = F.normalize(
            torch.cross(-self._frame_normal, handle_axis, dim=-1), dim=-1
        )
        delta = points - handle_center[:, None, :]
        axial = torch.sum(delta * handle_axis[:, None, :], dim=-1)
        radial = delta - axial[:, :, None] * handle_axis[:, None, :]
        radial_distance = torch.linalg.vector_norm(radial, dim=-1)
        closing_coordinate = torch.sum(radial * closing_axis[:, None, :], dim=-1)
        finite = torch.isfinite(points).all(dim=-1)
        usable_half_length = (0.5 * self.door_handle_length - 0.005).clamp(min=0.005)
        valid_contact_geometry = (
            finite
            & (torch.abs(axial) <= usable_half_length[:, None])
            & (radial_distance <= self.door_handle_radius[:, None] + 0.006)
            & (radial_distance >= (self.door_handle_radius[:, None] - 0.008).clamp(min=0.0))
        )
        on_lever_capsule = valid_contact_geometry & (force_norms >= 1.0)
        unexpected_handle_contact = (force_norms >= 1.0) & ~on_lever_capsule
        self._grasp_on_lever_capsule = on_lever_capsule
        self._grasp_unexpected_handle_contact = unexpected_handle_contact

        clock = self.total_time_buf[:, None].expand(-1, 2)
        self._grasp_contact_last_seen = torch.where(
            on_lever_capsule, clock, self._grasp_contact_last_seen
        )
        self._grasp_contact_last_points = torch.where(
            on_lever_capsule[:, :, None], points, self._grasp_contact_last_points
        )
        self._grasp_contact_last_forces = torch.where(
            on_lever_capsule, force_norms, self._grasp_contact_last_forces
        )

        recent = (clock - self._grasp_contact_last_seen) <= 5
        self._grasp_recent_contact_fraction = recent.float().mean(dim=-1)
        stored_delta = self._grasp_contact_last_points - handle_center[:, None, :]
        stored_axial = torch.sum(stored_delta * handle_axis[:, None, :], dim=-1)
        stored_radial = stored_delta - stored_axial[:, :, None] * handle_axis[:, None, :]
        stored_closing = torch.sum(stored_radial * closing_axis[:, None, :], dim=-1)
        opposed = opposed_handle_contact(
            recent,
            torch.isfinite(self._grasp_contact_last_points).all(dim=(-1, -2)),
            stored_axial,
            usable_half_length,
            stored_closing,
            self.door_handle_radius,
        )
        self._grasp_opposed_contact = opposed
        self._grasp_contact_quality = (
            self._grasp_recent_contact_fraction
            + opposed.float()
            + (~unexpected_handle_contact.any(dim=-1)).float()
        ) / 3.0
        (
            self._grasp_bilateral_force_progress,
            self._grasp_per_jaw_force_quality,
        ) = bilateral_contact_force_progress(
            force_norms,
            valid_contact_geometry,
        )

        # Continuous bilateral-enclosure quality before contact.  It evaluates
        # the measured inner jaw surfaces against the randomized handle radius;
        # it does not prescribe a gripper joint angle or an intermediate pose.
        jaw_centers = torch.stack(
            [self._fixed_inner_center, self._moving_inner_center_world], dim=1
        )
        jaw_delta = jaw_centers - handle_center[:, None, :]
        jaw_axial = torch.sum(jaw_delta * handle_axis[:, None, :], dim=-1)
        jaw_closing = torch.sum(jaw_delta * closing_axis[:, None, :], dim=-1)
        jaw_panel = torch.sum(
            jaw_delta * self._frame_normal[:, None, :], dim=-1
        )
        sorted_closing = torch.sort(jaw_closing, dim=-1).values
        self._grasp_jaw_closing_coordinates = sorted_closing
        desired_closing = torch.stack(
            [-self.door_handle_radius, self.door_handle_radius], dim=-1
        )
        closing_quality = torch.exp(
            -torch.abs(sorted_closing - desired_closing).mean(dim=-1) / 0.02
        )
        panel_quality = torch.exp(-torch.abs(jaw_panel).mean(dim=-1) / 0.02)
        axial_excess = (
            torch.abs(jaw_axial) - usable_half_length[:, None]
        ).clamp(min=0.0)
        axial_quality = torch.exp(-axial_excess.mean(dim=-1) / 0.02)
        self._grasp_enclosure_quality = (
            closing_quality + panel_quality + axial_quality
        ) / 3.0

        gripper_velocity = self.simulator._rigid_body_vel[:, self.gripper_idx, :3]
        handle_velocity = door.data.body_lin_vel_w[:, self._door_handle_idx]
        self._gripper_handle_relative_speed = torch.linalg.vector_norm(
            gripper_velocity - handle_velocity, dim=-1
        )
        gripper_speed = torch.abs(
            self.simulator.dof_vel[:, self.gripper_dof_idx]
        ).amax(dim=-1)
        pocket_error = torch.linalg.vector_norm(self._pocket_pos - handle_center, dim=-1)
        pocket_normal_error = torch.abs(
            torch.sum(
                (self._pocket_pos - handle_center) * self._frame_normal, dim=-1
            )
        )
        self._grasp_acceptance_checks = torch.stack(
            [
                opposed,
                on_lever_capsule.any(dim=-1),
                pocket_error < 0.020,
                pocket_normal_error < 0.012,
                self._gripper_axis_error < 0.05,
                self._gripper_handle_relative_speed < 0.08,
                gripper_speed < 0.08,
                torch.all(self._grasp_contact_last_forces <= 50.0, dim=-1),
                ~unexpected_handle_contact.any(dim=-1),
                self._panel_contact_free,
            ],
            dim=-1,
        )
        self._precise_bilateral_grasp = self._grasp_acceptance_checks.all(dim=-1)
        safe_half_length = (0.5 * self.door_handle_length).clamp(min=0.01)
        safe_radius = self.door_handle_radius.clamp(min=0.005)
        canonical_axial = canonicalize_jaw_axial(
            torch.nan_to_num(axial / safe_half_length[:, None]), self.door_open_lr
        )
        canonical_closing = canonicalize_jaw_order(
            torch.nan_to_num(closing_coordinate / safe_radius[:, None]),
            self.door_open_lr,
        )
        canonical_force_norms = canonicalize_jaw_order(
            (force_norms / 50.0).clamp(max=2.0), self.door_open_lr
        )
        canonical_on_capsule = canonicalize_jaw_order(
            on_lever_capsule.float(), self.door_open_lr
        )
        self._grasp_contact_obs = torch.cat(
            [
                canonical_axial,
                canonical_closing,
                canonical_force_norms,
                canonical_on_capsule,
            ],
            dim=-1,
        )

    def _vector_in_door_frame(self, vector):
        task_basis, _ = self._door_task_basis()
        return world_polar_to_task(vector, task_basis)

    @override
    def _get_obs_relative_to_door(self):
        task_basis, _ = self._door_task_basis()
        root_state = self.simulator.robot_root_states
        door_position = self.simulator.get_task_root_state("door")[:, :3].clone()
        root_position = root_state[:, :3]
        door_position[:, 2] = root_position[:, 2]
        door_from_robot = world_polar_to_task(
            door_position - root_position, task_basis
        )
        root_quat = xyzw_to_wxyz(root_state[:, 3:7])
        unit_x = torch.zeros(self.num_envs, 3, device=self.device)
        unit_x[:, 0] = 1.0
        unit_z = torch.zeros_like(unit_x)
        unit_z[:, 2] = 1.0
        root_forward = world_polar_to_task(
            quat_apply(root_quat, unit_x), task_basis
        )
        root_up = world_polar_to_task(
            quat_apply(root_quat, unit_z), task_basis
        )
        return torch.cat([door_from_robot, root_forward, root_up], dim=-1)

    @override
    def _get_obs_hand_force(self):
        force = super()._get_obs_hand_force().reshape(self.num_envs, -1, 3)
        task_basis, _ = self._door_task_basis()
        canonical_force = world_polar_to_task(
            force,
            task_basis[:, None, :, :],
        )
        canonical_force = canonicalize_jaw_order(
            canonical_force, self.door_open_lr
        )
        return canonical_force.reshape(self.num_envs, -1)

    @override
    def _get_obs_handle_contact_force(self):
        force_world = self._get_object_to_hand_contact_forces().sum(dim=-2)
        task_basis, _ = self._door_task_basis()
        return world_polar_to_task(force_world, task_basis)

    @override
    def _get_obs_base_lin_vel(self):
        task_basis, _ = self._door_task_basis()
        root_quat = xyzw_to_wxyz(self.simulator.robot_root_states[:, 3:7])
        velocity_world = quat_apply(root_quat, self.base_lin_vel)
        return world_polar_to_task(velocity_world, task_basis)

    @override
    def _get_obs_base_ang_vel(self):
        task_basis, task_handedness = self._door_task_basis()
        root_quat = xyzw_to_wxyz(self.simulator.robot_root_states[:, 3:7])
        velocity_world = quat_apply(root_quat, self.base_ang_vel)
        return world_axial_to_task(
            velocity_world, task_basis, task_handedness
        )

    @override
    def _get_obs_b2z1_commands(self):
        task_basis, task_handedness = self._door_task_basis()
        base_yaw_quat = xyzw_to_wxyz(self._get_b2z1_base_yaw_quat())
        base_linear_body = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        base_linear_body[:, :2] = self._b2z1_commands[:, :2]
        base_linear_task = world_polar_to_task(
            quat_apply(base_yaw_quat, base_linear_body), task_basis
        )
        base_angular_world = torch.zeros_like(base_linear_body)
        base_angular_world[:, 2] = self._b2z1_commands[:, 2]
        base_angular_task = world_axial_to_task(
            base_angular_world, task_basis, task_handedness
        )

        goal_position, goal_quat_xyzw = self._compute_b2z1_ee_goal_world()
        actual_position = self.simulator._rigid_body_pos[:, self.gripper_idx]
        actual_quat_xyzw = self.simulator._rigid_body_rot[:, self.gripper_idx]
        position_error_task = world_polar_to_task(
            goal_position - actual_position, task_basis
        )
        orientation_error_world = axis_angle_from_quat(
            quat_mul(
                xyzw_to_wxyz(goal_quat_xyzw),
                quat_inv(xyzw_to_wxyz(actual_quat_xyzw)),
            )
        )
        orientation_error_task = world_axial_to_task(
            orientation_error_world, task_basis, task_handedness
        )
        return torch.cat(
            [
                base_linear_task[:, :2],
                base_angular_task[:, 2:3],
                position_error_task,
                orientation_error_task,
                self._b2z1_commands[:, 9:10],
            ],
            dim=-1,
        )

    @override
    def _compute_grasp_target(self):
        """Return the physical center of the robot-side lever capsule."""
        if not hasattr(self, "_door_handle_idx"):
            return super()._compute_grasp_target()
        door = self.simulator.scene.articulations["door"]
        handle_pos = door.data.body_pos_w[:, self._door_handle_idx]
        handle_quat = door.data.body_quat_w[:, self._door_handle_idx]
        center_local = torch.zeros(self.num_envs, 3, device=self.device)
        center_local[:, 0] = 0.5 * self.door_axle_length
        center_local[:, 1] = -0.5 * self.door_handle_length * self.door_open_lr
        return handle_pos + quat_apply(handle_quat, center_local)

    @override
    def _get_obs_hand_handle_transform(self):
        gripper_pos = self.simulator._rigid_body_pos[:, self.gripper_idx]
        gripper_quat = xyzw_to_wxyz(self.simulator._rigid_body_rot[:, self.gripper_idx])
        door = self.simulator.scene.articulations["door"]
        handle_quat = door.data.body_quat_w[:, self._door_handle_idx]
        relative_pos, relative_quat = subtract_frame_transforms(
            gripper_pos, gripper_quat, self._compute_grasp_target(), handle_quat
        )
        relative_rot_6d = quat_to_tan_norm(wxyz_to_xyzw(relative_quat), w_last=True)
        return torch.cat([relative_pos, relative_rot_6d], dim=-1)

    def _compute_gap_angle(self):
        length = self.door_width.clamp(min=0.1)
        thickness = torch.full_like(length, 0.04)
        clearance = torch.full_like(length, 0.002)
        lower = 2.0 * torch.atan2(thickness, length)
        upper = torch.full_like(lower, torch.pi / 2.0)
        for _ in range(24):
            angle = 0.5 * (lower + upper)
            gap = length * (1.0 - torch.cos(angle)) - thickness * torch.sin(angle)
            lower = torch.where(gap < clearance, angle, lower)
            upper = torch.where(gap < clearance, upper, angle)
        return 0.5 * (lower + upper)

    @staticmethod
    def _unit_progress(error, scale):
        return (1.0 - error / scale).clamp(0.0, 1.0)

    def _push_free_motion_credit(self):
        """Credit separation only while the door is opening under momentum."""
        return (
            self._hinge_opening_speed.clamp(min=0.0)
            / self._PUSH_TARGET_ANGULAR_SPEED
        ).clamp(max=1.0)

    def _update_stage_progress(self):
        operation_dist = torch.linalg.vector_norm(self._operation_error, dim=-1)
        pocket_dist = torch.linalg.vector_norm(self._pocket_error_frame, dim=-1)
        reposition_dist = torch.linalg.vector_norm(self._reposition_error, dim=-1)
        base_position_progress = (
            torch.exp(-operation_dist / 1.0) / math.exp(-0.16 / 1.0)
        ).clamp(max=1.0)
        base_yaw_progress = (
            torch.exp(-self._base_yaw_error / 0.7) / math.exp(-0.08 / 0.7)
        ).clamp(max=1.0)
        middle_arm_progress = (
            torch.exp(-self._middle_arm_joint_error / 0.5)
            / math.exp(-0.05 / 0.5)
        ).clamp(max=1.0).mean(dim=-1)
        self._walk_component_progress = torch.stack(
            [base_position_progress, base_yaw_progress, middle_arm_progress],
            dim=-1,
        )
        self._stage_progress[:, self.STAGE_WALK_TO_DOOR] = bottleneck_balanced_progress(
            self._walk_component_progress,
            bottleneck_weight=0.5,
        )

        (
            self._grasp_pocket_distance_progress,
            self._grasp_path_clearance_quality,
            self._grasp_path_transverse_quality,
            self._grasp_direct_position_quality,
            self._grasp_path_between_quality,
        ) = grasp_collision_aware_position_progress(
            self._pocket_error_frame,
            self._grasp_jaw_closing_coordinates,
            self.door_handle_radius,
        )
        self._grasp_pocket_normal_progress = torch.exp(
            -(torch.abs(self._pocket_error_frame[:, 0]) - 0.012).clamp(min=0.0)
            / 0.090
        )
        pocket_progress = self._grasp_pocket_distance_progress
        orientation_progress = (
            torch.exp(-self._grasp_ee_orientation_error / 0.5)
            / math.exp(-0.05 / 0.5)
        ).clamp(max=1.0)
        relative_speed_quality = torch.exp(
            -self._gripper_handle_relative_speed / 0.08
        )
        gripper_speed = torch.abs(
            self.simulator.dof_vel[:, self.gripper_dof_idx]
        ).amax(dim=-1)
        gripper_speed_quality = torch.exp(-gripper_speed / 0.08)
        contact_stability = torch.sqrt(
            relative_speed_quality * gripper_speed_quality
        )
        physical_contact_progress = (
            self._grasp_bilateral_force_progress * contact_stability
        )
        safe_grasp_geometry = (
            ~self._grasp_unexpected_handle_contact.any(dim=-1)
        ).float() * self._panel_contact_free.float()
        physical_contact_progress *= safe_grasp_geometry
        alignment_quality = torch.sqrt(pocket_progress * orientation_progress)
        (
            contact_progress,
            self._grasp_aperture_quality,
            self._grasp_between_jaws_quality,
            self._grasp_closure_quality,
            self._grasp_capture_progress,
        ) = grasp_enclosure_progress(
            self._grasp_jaw_closing_coordinates,
            self.door_handle_radius,
            alignment_quality,
            physical_contact_progress,
        )
        contact_progress *= safe_grasp_geometry
        self._grasp_component_progress = torch.stack(
            [
                base_position_progress,
                base_yaw_progress,
                pocket_progress,
                orientation_progress,
                contact_progress,
            ],
            dim=-1,
        )
        self._stage_progress[:, self.STAGE_GRASP] = bottleneck_balanced_progress(
            self._grasp_component_progress,
            bottleneck_weight=0.5,
        )
        grasp = self.stage_buf == self.STAGE_GRASP
        self._grasp_component_episode_best = torch.where(
            grasp[:, None],
            torch.maximum(
                self._grasp_component_episode_best,
                self._grasp_component_progress,
            ),
            self._grasp_component_episode_best,
        )
        latch_progress = (self._latch_pos / 0.02).clamp(0.0, 1.0)
        gap_progress = (
            self._hinge_angle / self._gap_angle.clamp(min=1.0e-4)
        ).clamp(0.0, 1.0)
        pocket_retention = self._unit_progress(pocket_dist, 0.08)
        axis_retention = self._unit_progress(self._gripper_axis_error, 0.30)
        contact_retention = 0.5 * (
            self._grasp_recent_contact_fraction
            + self._grasp_opposed_contact.float()
        )
        contact_retention *= (
            ~self._grasp_unexpected_handle_contact.any(dim=-1)
        ).float()
        contact_retention = torch.maximum(
            contact_retention, self._grasp_enclosure_quality
        )
        self._open_latch_retention_components = torch.stack(
            [
                base_position_progress,
                base_yaw_progress,
                pocket_retention,
                axis_retention,
                contact_retention,
            ],
            dim=-1,
        )
        self._open_latch_retention_quality = torch.amin(
            self._open_latch_retention_components, dim=-1
        )
        # Retention is a constraint, not a gate on progress already made.
        # Multiplication made normal contact compliance erase an increasing
        # fraction of latch/gap progress precisely while the door started to
        # move.  The additive potential keeps all three terms signed and
        # bounded without prescribing a handle trajectory.
        self._stage_progress[:, self.STAGE_OPEN_LATCH] = (
            0.10 * self._open_latch_retention_quality
            + 0.45 * latch_progress
            + 0.45 * gap_progress
        )
        handle_release_progress = torch.exp(
            -self._finger_handle_forces.max(dim=-1).values / 10.0
        )
        panel_release_progress = torch.exp(-self._panel_gripper_force / 10.0)
        gap_retention_progress = (
            self._hinge_angle / self._gap_angle.clamp(min=1.0e-4)
        ).clamp(0.0, 1.0)
        reposition_component_progress = torch.stack(
            [
                torch.exp(-reposition_dist / 1.0)
                / math.exp(-self._REPOSITION_BASE_POSITION_TOLERANCE),
                torch.exp(-self._base_yaw_error / 0.7)
                / math.exp(-self._REPOSITION_BASE_YAW_TOLERANCE / 0.7),
                torch.exp(-self._push_ready_distance / 0.7)
                / math.exp(-self._REPOSITION_EE_POSITION_TOLERANCE / 0.7),
                torch.exp(-self._push_ready_orientation_error / 0.5)
                / math.exp(-self._REPOSITION_EE_ORIENTATION_TOLERANCE / 0.5),
                torch.exp(-self._push_ready_gripper_error / 0.3)
                / math.exp(-self._REPOSITION_GRIPPER_TOLERANCE / 0.3),
                torch.exp(-self._base_motion / 0.5)
                / math.exp(-self._REPOSITION_BASE_MOTION_TOLERANCE / 0.5),
                torch.exp(-self._ee_speed / 0.5)
                / math.exp(-self._REPOSITION_EE_SPEED_TOLERANCE / 0.5),
                handle_release_progress,
                panel_release_progress,
                gap_retention_progress,
            ],
            dim=-1,
        ).clamp(max=1.0)
        self._stage_progress[:, self.STAGE_REPOSITION] = bottleneck_balanced_progress(
            reposition_component_progress, bottleneck_weight=0.5
        )
        push_hinge_progress = (
            self._hinge_angle / torch.deg2rad(torch.tensor(89.0, device=self.device))
        ).clamp(0.0, 1.0)
        push_clearance_progress = (
            1.0 - self._panel_gripper_clearance.clamp(min=0.0) / 0.05
        ).clamp(max=1.0)
        push_orientation_progress = (
            torch.exp(-self._push_ready_orientation_error / 0.5)
            / math.exp(-self._REPOSITION_EE_ORIENTATION_TOLERANCE / 0.5)
        ).clamp(max=1.0)
        free_motion_credit = self._push_free_motion_credit()
        push_clearance_or_motion_progress = (
            free_motion_credit
            + (1.0 - free_motion_credit) * push_clearance_progress
        )
        self._push_contact_progress = torch.minimum(
            push_clearance_or_motion_progress, push_orientation_progress
        )
        push_base_safety_progress = torch.exp(
            -(0.10 - self._shoulder_panel_clearance).clamp(min=0.0) / 0.10
        )
        push_speed_progress = torch.exp(
            -torch.abs(
                self._hinge_opening_speed - self._PUSH_TARGET_ANGULAR_SPEED
            )
            / self._PUSH_TARGET_ANGULAR_SPEED
        )
        # Clearance remains a signed potential while the door is slow.  As the
        # door approaches its target opening speed, momentum continuously
        # replaces the need for contact.  Orientation remains required in both
        # cases so the arm is ready to resume pushing when the door slows.
        self._stage_progress[:, self.STAGE_PUSH_OPEN] = (
            0.10 * push_base_safety_progress
            + 0.20 * self._push_contact_progress
            + 0.55 * push_hinge_progress
            + 0.15 * push_speed_progress
        )
        # THROUGH must provide a gradient before either terminal condition is
        # met.  Keep forward displacement signed so reversing loses progress,
        # and shape arm stowage continuously around the same middle-preset
        # error used by the acceptance condition.
        through_distance_progress = (self._through_distance / 1.0).clamp(max=1.0)
        through_arm_joint_progress = (
            torch.exp(-self._middle_arm_joint_error / 0.5)
            / math.exp(-0.05 / 0.5)
        ).clamp(max=1.0)
        through_arm_progress = bottleneck_balanced_progress(
            through_arm_joint_progress,
            bottleneck_weight=0.5,
        )
        through_door_progress = (
            self._hinge_angle / math.radians(65.0)
        ).clamp(0.0, 1.0)
        through_component_progress = torch.stack(
            [through_distance_progress, through_arm_progress, through_door_progress],
            dim=-1,
        )
        self._stage_progress[:, self.STAGE_THROUGH] = bottleneck_balanced_progress(
            through_component_progress,
            bottleneck_weight=0.5,
        )

    def _update_stage_stability(self):
        operation_distance = torch.linalg.vector_norm(
            self._operation_error, dim=-1
        )
        conditions = torch.stack(
            [
                walk_to_door_completion_condition(
                    operation_distance,
                    self._base_yaw_error,
                    self._handle_contact_free,
                    self._panel_contact_free,
                    self._base_motion,
                    self._middle_arm_error,
                    self._middle_arm_speed,
                ),
                grasp_completion_condition(
                    operation_distance,
                    self._base_yaw_error,
                    self._precise_bilateral_grasp,
                ),
                open_latch_completion_condition(
                    operation_distance,
                    self._base_yaw_error,
                    self._latch_pos,
                    self._hinge_angle,
                    self._gap_angle,
                    self._precise_bilateral_grasp,
                ),
                self._handle_contact_free
                & self._panel_contact_free
                & (self._hinge_angle >= self._gap_angle)
                & (
                    torch.linalg.vector_norm(self._reposition_error, dim=-1)
                    < self._REPOSITION_BASE_POSITION_TOLERANCE
                )
                & (
                    self._base_yaw_error
                    < self._REPOSITION_BASE_YAW_TOLERANCE
                )
                & (
                    self._push_ready_distance
                    < self._REPOSITION_EE_POSITION_TOLERANCE
                )
                & (
                    self._push_ready_orientation_error
                    < self._REPOSITION_EE_ORIENTATION_TOLERANCE
                )
                & (
                    self._push_ready_gripper_error
                    < self._REPOSITION_GRIPPER_TOLERANCE
                )
                & (
                    self._base_motion
                    < self._REPOSITION_BASE_MOTION_TOLERANCE
                )
                & (self._ee_speed < self._REPOSITION_EE_SPEED_TOLERANCE),
                (self._hinge_angle >= torch.deg2rad(torch.tensor(89.0, device=self.device)))
                & self._push_gripper_contact_seen,
                (self._through_distance > 1.0)
                & self._arm_safely_stowed
                & (self._hinge_angle >= math.radians(65.0))
                & (self._base_motion < 0.12),
            ],
            dim=-1,
        )
        self._task_stage_acceptance_now = conditions
        active = F.one_hot(self.stage_buf.clamp(max=5), 6).bool()
        keep = conditions & active
        self._task_stage_stable = torch.where(
            keep, self._task_stage_stable + 1, torch.zeros_like(self._task_stage_stable)
        )
        self._grasp_stable_episode_max = torch.maximum(
            self._grasp_stable_episode_max,
            self._task_stage_stable[:, self.STAGE_GRASP],
        )
        overspeed = self._hinge_speed > torch.deg2rad(
            torch.tensor(30.0, device=self.device)
        )
        self._door_overspeed_steps = torch.where(
            overspeed, self._door_overspeed_steps + 1, torch.zeros_like(self._door_overspeed_steps)
        )

    def _validate_staged_reset_initial_states(self):
        restored = self.just_resetted_buf & (self.stage_buf > 0)
        if not torch.any(restored):
            return
        restored_count = restored.float().sum()
        preceding_stage = (self.stage_buf - 1).clamp(min=0)
        valid = self._task_stage_acceptance_now.gather(
            1, preceding_stage[:, None]
        ).squeeze(1)
        through = restored & (self.stage_buf == self.STAGE_THROUGH)
        through_contact_free = self._handle_contact_free & self._panel_contact_free
        valid &= ~through | through_contact_free
        self.log_dict["reset/stage_initial_state_valid"] = (
            valid.float() * restored.float()
        ).sum() / restored_count
        invalid = restored & ~valid
        if torch.any(invalid):
            invalid_stages = torch.unique(self.stage_buf[invalid]).tolist()
            raise RuntimeError(
                "Staged-reset initial states no longer satisfy the preceding "
                f"stage acceptance conditions for stages {invalid_stages}"
            )

    def _stage_0_reward_condition(self):
        return self._task_stage_stable[:, 0] >= 10

    def _stage_0_to_1_advance_condition(self):
        return self._stage_0_reward_condition()

    def _stage_1_reward_condition(self):
        # Contact manifolds can alternate between jaws on adjacent PhysX steps.
        # Five policy steps still require 100 ms of stable geometric capture.
        return self._task_stage_stable[:, 1] >= 5

    def _stage_1_to_2_advance_condition(self):
        return self._stage_1_reward_condition()

    def _stage_2_reward_condition(self):
        return self._task_stage_stable[:, 2] >= 1

    def _stage_2_to_3_advance_condition(self):
        return self._stage_2_reward_condition()

    def _stage_3_reward_condition(self):
        return self._task_stage_stable[:, 3] >= 15

    def _stage_3_to_4_advance_condition(self):
        return self._stage_3_reward_condition()

    def _stage_4_reward_condition(self):
        return self._task_stage_stable[:, 4] >= 1

    def _stage_4_to_5_advance_condition(self):
        return self._stage_4_reward_condition()

    def _stage_5_reward_condition(self):
        return self._task_stage_stable[:, 5] >= 10

    def _stage_4_to_5_advance_callback(self, env_ids):
        self._through_start_pos[env_ids] = self.simulator.robot_root_states[env_ids, :3]

    @override
    def _post_staged_reset_state_selected(self, env_ids, selected_stages):
        super()._post_staged_reset_state_selected(env_ids, selected_stages)
        through_env_ids = env_ids[selected_stages == self.STAGE_THROUGH]
        self._through_start_pos[through_env_ids] = self.target_robot_root_states[
            through_env_ids, :3
        ]

    def _procedural_reset_stages(self):
        return (self.STAGE_GRASP, self.STAGE_PUSH_OPEN, self.STAGE_THROUGH)

    def _unsupported_fixed_reset_stages(self):
        return (self.STAGE_OPEN_LATCH, self.STAGE_REPOSITION)

    def _procedural_panel_pose(self, env_ids, hinge_angle):
        door_root = self.simulator.get_task_root_state("door")[env_ids]
        frame_pos = door_root[:, :3]
        frame_quat = door_root[:, 3:7]
        hinge_local = torch.zeros(len(env_ids), 3, device=self.device)
        hinge_local[:, 0] = 0.02
        hinge_local[:, 1] = -0.5 * self.door_width[env_ids] * self.door_open_lr[env_ids]
        signed_angle = hinge_angle * self.door_open_lr[env_ids]
        panel_local_quat = quat_from_euler_xyz(
            torch.zeros_like(signed_angle),
            torch.zeros_like(signed_angle),
            signed_angle,
        )
        panel_quat = quat_mul(frame_quat, panel_local_quat)
        hinge_world = frame_pos + quat_apply(frame_quat, hinge_local)
        panel_pos = hinge_world + quat_apply(panel_quat, -hinge_local)
        return frame_pos, frame_quat, panel_pos, panel_quat

    def _set_procedural_door_state(self, env_ids, hinge_angle):
        door = self.simulator.scene.articulations["door"]
        dof_pos = door.data.joint_pos.clone()
        dof_vel = torch.zeros_like(dof_pos)
        dof_pos[env_ids] = 0.0
        dof_pos[env_ids, 0] = hinge_angle
        self.simulator.set_task_dof_state_tensor(
            env_ids,
            {
                "door": (
                    dof_pos,
                    dof_vel,
                    torch.arange(door.num_joints, device=self.device, dtype=torch.long),
                )
            },
        )

    def _set_procedural_base_pose(self, env_ids, position, forward, position_jitter):
        forward = F.normalize(forward, dim=-1)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        tangent = torch.stack((-forward[:, 1], forward[:, 0], torch.zeros_like(yaw)), dim=-1)
        normal_noise = torch_rand_float(
            -position_jitter, position_jitter, (len(env_ids), 1), device=str(self.device)
        )
        tangent_noise = torch_rand_float(
            -position_jitter, position_jitter, (len(env_ids), 1), device=str(self.device)
        )
        position = position + normal_noise * forward + tangent_noise * tangent
        position[:, 2] = self.base_init_state[2] + self.env_origins[env_ids, 2]
        yaw += torch_rand_float(-0.02, 0.02, (len(env_ids), 1), device=str(self.device))[:, 0]
        self.target_robot_root_states[env_ids] = self.base_init_state
        self.target_robot_root_states[env_ids, :3] = position
        self.target_robot_root_states[env_ids, 3:7] = wxyz_to_xyzw(
            quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        )
        self.target_robot_root_states[env_ids, 7:13] = 0.0

    def _sync_procedural_command_to_target_state(self, env_ids):
        root_position = self.target_robot_root_states[env_ids, :3]
        root_quat_xyzw = self.target_robot_root_states[env_ids, 3:7]
        arm_position = self.target_robot_dof_state[
            env_ids[:, None], self.arm_dof_idx[None, :], 0
        ]
        ee_position, ee_rotation, _ = b2z1_gripper_pose_and_jacobian(
            root_position, root_quat_xyzw, arm_position
        )
        root_quat_wxyz = xyzw_to_wxyz(root_quat_xyzw)
        origin = torch.zeros_like(root_position)
        origin[:, :2] = root_position[:, :2]
        raw_goal_cart = (
            quat_apply(quat_inv(root_quat_wxyz), ee_position - origin)
            - self._b2z1_goal_center_offset
        )
        goal_sphere = self._cart_to_sphere(raw_goal_cart)
        goal_sphere = torch.maximum(
            torch.minimum(goal_sphere, self._b2z1_ee_goal_sphere_upper),
            self._b2z1_ee_goal_sphere_lower,
        )
        goal_cart = self._sphere_to_cart(goal_sphere)
        position_command = torch.maximum(
            torch.minimum(
                goal_cart - self._b2z1_default_ee_goal_cart,
                self._command_upper[:, 3:6],
            ),
            self._command_lower[:, 3:6],
        )
        goal_quat_xyzw = wxyz_to_xyzw(quat_from_matrix(ee_rotation))
        orientation_command = self._compute_b2z1_orn_delta_for_world_quat(
            goal_cart, goal_quat_xyzw, root_quat_wxyz
        )
        command = torch.zeros(len(env_ids), self.COMMAND_DIM, device=self.device)
        command[:, 3:6] = position_command
        command[:, 6:9] = orientation_command
        command[:, 9] = self._b2z1_gripper_closed_command
        self._delta_actions[env_ids] = command / self._command_scale
        self._b2z1_commands[env_ids] = command
        self._b2z1_commands_unclipped[env_ids] = command
        self._last_b2z1_commands[env_ids] = command
        self._b2z1_arm_pos_targets[env_ids] = arm_position

    def _reset_procedural_stage_initial_states(self, env_ids, selected_stages):
        unsupported = (selected_stages == self.STAGE_OPEN_LATCH) | (
            selected_stages == self.STAGE_REPOSITION
        )
        if torch.any(unsupported):
            stages = torch.unique(selected_stages[unsupported]).tolist()
            raise RuntimeError(
                f"Procedural initial-state generation is not implemented for stages {stages}"
            )

        self.target_robot_dof_state[env_ids, :, 0] = self.default_dof_pos
        self.target_robot_dof_state[env_ids, :, 1] = 0.0
        self.target_robot_dof_state[
            env_ids[:, None], self.gripper_dof_idx[None, :], 0
        ] = self._gripper_closed_pos

        grasp_env_ids = env_ids[selected_stages == self.STAGE_GRASP]
        if len(grasp_env_ids) > 0:
            door_root = self.simulator.get_task_root_state("door")[grasp_env_ids]
            frame_pos = door_root[:, :3]
            frame_quat = door_root[:, 3:7]
            frame_normal = quat_apply(
                frame_quat,
                torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(len(grasp_env_ids), -1),
            )
            handle_local = torch.zeros(len(grasp_env_ids), 3, device=self.device)
            handle_local[:, 0] = 0.5 * self.door_axle_length[grasp_env_ids]
            handle_local[:, 1] = (
                0.5 * self.door_width[grasp_env_ids]
                - self.door_handle_width[grasp_env_ids]
                - 0.5 * self.door_handle_length[grasp_env_ids]
            ) * self.door_open_lr[grasp_env_ids]
            handle_local[:, 2] = self.door_handle_height[grasp_env_ids]
            handle_position = frame_pos + quat_apply(frame_quat, handle_local)
            base_position = handle_position + self._operation_station_distance * frame_normal
            self._set_procedural_base_pose(
                grasp_env_ids, base_position, -frame_normal, position_jitter=0.035
            )
            arm_noise = torch_rand_float(
                -0.02,
                0.02,
                (len(grasp_env_ids), len(self.arm_dof_idx)),
                device=str(self.device),
            )
            self.target_robot_dof_state[
                grasp_env_ids[:, None], self.arm_dof_idx[None, :], 0
            ] = self._middle_arm + arm_noise
            self._set_procedural_door_state(
                grasp_env_ids, torch.zeros(len(grasp_env_ids), device=self.device)
            )

        push_env_ids = env_ids[selected_stages == self.STAGE_PUSH_OPEN]
        if len(push_env_ids) > 0:
            gap_angle = self._compute_gap_angle()[push_env_ids]
            hinge_angle = gap_angle + torch_rand_float(
                math.radians(2.0),
                math.radians(6.0),
                (len(push_env_ids), 1),
                device=str(self.device),
            )[:, 0]
            frame_pos, frame_quat, panel_pos, panel_quat = self._procedural_panel_pose(
                push_env_ids, hinge_angle
            )
            unit_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(len(push_env_ids), -1)
            unit_y = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(len(push_env_ids), -1)
            unit_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(len(push_env_ids), -1)
            frame_normal = F.normalize(quat_apply(frame_quat, unit_x), dim=-1)
            frame_tangent = F.normalize(quat_apply(frame_quat, unit_y), dim=-1)
            through = -frame_normal
            panel_normal = F.normalize(quat_apply(panel_quat, unit_x), dim=-1)
            panel_tangent = F.normalize(quat_apply(panel_quat, unit_y), dim=-1)
            panel_center = panel_pos + quat_apply(
                panel_quat, 0.5 * self.door_height[push_env_ids, None] * unit_z
            )
            push_ready = panel_center + (
                self._DOOR_PANEL_HALF_THICKNESS
                + self._GRIPPER_PANEL_SUPPORT_OFFSET
                + self._PANEL_READY_CLEARANCE
            ) * panel_normal

            station_distance = (
                self._operation_station_distance
                + 0.5 * self.door_axle_length[push_env_ids]
            )
            base_position = frame_pos + station_distance[:, None] * frame_normal
            base_position[:, 2] = self.base_init_state[2] + self.env_origins[push_env_ids, 2]
            desired_lateral = -frame_tangent
            nominal_delta = push_ready - base_position
            goal_center = self._b2z1_goal_center_offset.reshape(-1)
            nominal_goal = torch.stack(
                (
                    torch.sum(nominal_delta * through, dim=-1) - goal_center[0],
                    torch.sum(nominal_delta * desired_lateral, dim=-1) - goal_center[1],
                    push_ready[:, 2] - goal_center[2],
                ),
                dim=-1,
            )
            transverse_sq = torch.square(nominal_goal[:, 1:]).sum(dim=-1)
            comfortable_forward = torch.sqrt(
                (self._reposition_comfort_radius**2 - transverse_sq).clamp(min=0.0)
            )
            advance = (nominal_goal[:, 0] - comfortable_forward).clamp(min=0.0)
            base_position += advance[:, None] * through
            self._set_procedural_base_pose(
                push_env_ids, base_position, through, position_jitter=0.01
            )

            target_x = -panel_normal
            target_y = F.normalize(
                panel_tangent * self.door_open_lr[push_env_ids, None], dim=-1
            )
            target_z = F.normalize(torch.cross(target_x, target_y, dim=-1), dim=-1)
            target_rotation = torch.stack((target_x, target_y, target_z), dim=-1)
            initial = self._middle_arm.expand(len(push_env_ids), -1).clone()
            lower = self.simulator.hard_dof_pos_limits[self.arm_dof_idx, 0]
            upper = self.simulator.hard_dof_pos_limits[self.arm_dof_idx, 1]
            arm_position, position_error, orientation_error = solve_b2z1_arm_ik(
                self.target_robot_root_states[push_env_ids, :3],
                self.target_robot_root_states[push_env_ids, 3:7],
                push_ready,
                target_rotation,
                initial,
                lower,
                upper,
            )
            failed = (position_error >= 0.02) | (orientation_error >= 0.04)
            if torch.any(failed):
                raise RuntimeError(
                    "B2Z1 procedural PUSH_OPEN IK failed for "
                    f"{int(failed.sum().item())}/{len(push_env_ids)} states; "
                    f"max position error={float(position_error.max()):.4f} m, "
                    f"max orientation error={float(orientation_error.max()):.4f} rad"
                )
            self.target_robot_dof_state[
                push_env_ids[:, None], self.arm_dof_idx[None, :], 0
            ] = arm_position
            self._set_procedural_door_state(push_env_ids, hinge_angle)

        through_env_ids = env_ids[selected_stages == self.STAGE_THROUGH]
        if len(through_env_ids) > 0:
            hinge_angle = torch_rand_float(
                math.radians(91.0),
                math.radians(105.0),
                (len(through_env_ids), 1),
                device=str(self.device),
            )[:, 0]
            frame_pos, frame_quat, _, _ = self._procedural_panel_pose(
                through_env_ids, hinge_angle
            )
            frame_normal = quat_apply(
                frame_quat,
                torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(len(through_env_ids), -1),
            )
            base_position = frame_pos + self._operation_station_distance * frame_normal
            self._set_procedural_base_pose(
                through_env_ids, base_position, -frame_normal, position_jitter=0.03
            )
            arm_noise = torch_rand_float(
                -0.02,
                0.02,
                (len(through_env_ids), len(self.arm_dof_idx)),
                device=str(self.device),
            )
            self.target_robot_dof_state[
                through_env_ids[:, None], self.arm_dof_idx[None, :], 0
            ] = self._middle_arm + arm_noise
            self._set_procedural_door_state(through_env_ids, hinge_angle)
            self._push_gripper_contact_seen[through_env_ids] = True
            self._through_start_pos[through_env_ids] = self.target_robot_root_states[
                through_env_ids, :3
            ]

        self._sync_procedural_command_to_target_state(env_ids)

    def _stage_5_to_complete_condition(self):
        return self._stage_5_reward_condition()

    def _reward_task_stage_progress(self):
        stage = self.stage_buf.clamp(max=5)
        current = self._stage_progress.gather(1, stage[:, None]).squeeze(1)
        previous = self._task_stage_best.gather(1, stage[:, None]).squeeze(1)
        initialized = self._task_stage_progress_initialized.gather(
            1, stage[:, None]
        ).squeeze(1)
        walk_to_door = stage == self.STAGE_WALK_TO_DOOR
        grasp = stage == self.STAGE_GRASP
        open_latch = stage == self.STAGE_OPEN_LATCH
        reposition = stage == self.STAGE_REPOSITION
        push_open = stage == self.STAGE_PUSH_OPEN
        signed_potential = (
            walk_to_door
            | grasp
            | open_latch
            | reposition
            | push_open
            | (stage == self.STAGE_THROUGH)
        )
        delta = torch.where(
            initialized,
            torch.where(
                signed_potential,
                current - previous,
                (current - previous).clamp(min=0.0),
            ),
            0.0,
        )
        self._task_stage_best.scatter_(
            1,
            stage[:, None],
            torch.where(
                initialized,
                torch.where(
                    signed_potential,
                    current,
                    torch.maximum(current, previous),
                ),
                current,
            )[:, None],
        )
        self._task_stage_progress_initialized.scatter_(
            1, stage[:, None], torch.ones_like(stage[:, None], dtype=torch.bool)
        )
        delta = torch.where(
            walk_to_door | grasp | (stage == self.STAGE_THROUGH),
            20.0 * delta,
            delta,
        )
        # OPEN_LATCH has only a short physical travel before acceptance.  Keep
        # it as a signed potential, but make that motion visible above the
        # per-step time and command costs; regressions are penalized equally.
        delta = torch.where(open_latch, 10.0 * delta, delta)
        return delta / self.dt

    def _reward_penalty_stage_residual(self):
        active = (
            (self.stage_buf == self.STAGE_WALK_TO_DOOR)
            | (self.stage_buf == self.STAGE_GRASP)
            | (self.stage_buf == self.STAGE_THROUGH)
        )
        current = self._stage_progress.gather(
            1, self.stage_buf.clamp(max=5)[:, None]
        ).squeeze(1)
        residual = torch.where(active, 1.0 - current, 0.0)
        return residual

    def _reward_task_stage_complete(self):
        transition = self._get_stage_advance_event()
        final_stage = self.time_since_completion_buf == 1
        return stage_completion_event(transition, final_stage).float() / self.dt

    def _reward_task_success(self):
        return (self.time_since_completion_buf == 1).float() / self.dt

    def _reward_success_time_bonus(self):
        success = self.time_since_completion_buf == 1
        remaining = (1.0 - self.total_time_buf.float() * self.dt / 70.0).clamp(0.0, 1.0)
        return success.float() * remaining / self.dt

    def _reward_task_time(self):
        return torch.ones(self.num_envs, device=self.device)

    def _reward_penalty_door_frame_collision(self):
        force = self.simulator.scene.sensors[
            "door_frame_unwanted_contact_sensor"
        ].data.net_forces_w.norm(dim=-1).sum(dim=-1)
        return (force / 100.0).clamp(max=10.0)

    def _reward_penalty_unsafe_task_force(self):
        handle_force = self._finger_handle_forces.max(dim=-1).values
        return (
            (torch.maximum(handle_force, self._panel_gripper_force) - 600.0) / 100.0
        ).clamp(min=0.0)

    def _reward_penalty_door_overspeed(self):
        excess = (
            self._hinge_speed - torch.deg2rad(torch.tensor(30.0, device=self.device))
        ).clamp(min=0.0)
        return excess

    def _reward_penalty_push_speed_error(self):
        active = self.stage_buf == self.STAGE_PUSH_OPEN
        opening = self._hinge_opening_speed > 0.0
        return torch.where(
            active & opening,
            torch.abs(
                self._hinge_opening_speed - self._PUSH_TARGET_ANGULAR_SPEED
            ),
            0.0,
        )

    def _reward_penalty_push_relative_speed(self):
        active = self.stage_buf == self.STAGE_PUSH_OPEN
        penalty = excessive_relative_speed_penalty(
            self._gripper_panel_relative_velocity_frame,
            safe_speed=self._PUSH_SAFE_PANEL_RELATIVE_SPEED,
        )
        return torch.where(active, penalty, 0.0)

    def _reward_penalty_push_pose_constraint(self):
        active = self.stage_buf == self.STAGE_PUSH_OPEN
        # Separation is acceptable only to the extent that the door is already
        # opening at the requested speed.  A stalled door therefore restores a
        # smooth incentive to re-approach instead of retaining a lifetime
        # exemption from one historical contact.
        clearance_error = (
            self._panel_gripper_clearance.clamp(min=0.0)
            / 0.05
            * (1.0 - self._push_free_motion_credit())
        )
        orientation_error = (
            (
                self._push_ready_orientation_error
                - self._REPOSITION_EE_ORIENTATION_TOLERANCE
            ).clamp(min=0.0)
            / 0.25
        )
        error = 0.5 * (clearance_error + orientation_error)
        return torch.where(active, error, 0.0)

    def _reward_penalty_shoulder_clearance(self):
        active = self.stage_buf == self.STAGE_PUSH_OPEN
        violation = (-self._shoulder_panel_clearance).clamp(min=0.0)
        return torch.where(active, violation, 0.0)

    def _get_panel_gripper_force(self):
        force_matrix = self.simulator.scene.sensors[
            "b2z1_panel_gripper_contact_sensor"
        ].data.force_matrix_w
        force = force_matrix.norm(dim=-1).amax(dim=(-1, -2)).clone()
        force[self.episode_length_buf == 0] = 0.0
        return force

    def _get_obs_b2z1_task_geometry(self):
        return self._task_geometry_obs

    def _get_obs_b2z1_panel_interaction(self):
        return self._panel_interaction_obs

    @override
    def _reset_tasks_callback(self, env_ids):
        if hasattr(self, "_unsafe_force_event_recorder"):
            self._unsafe_force_event_recorder.invalidate(env_ids)
        self._task_stage_best[env_ids] = 0.0
        self._task_stage_progress_initialized[env_ids] = False
        self._grasp_component_episode_best[env_ids] = 0.0
        self._grasp_stable_episode_max[env_ids] = 0
        self._grasp_check_episode_seen[env_ids] = False
        self._task_stage_stable[env_ids] = 0
        self._door_overspeed_steps[env_ids] = 0
        self._push_gripper_contact_seen[env_ids] = False
        self._through_start_pos[env_ids] = self.simulator.robot_root_states[env_ids, :3]
        self._grasp_contact_last_seen[env_ids] = -1000
        self._grasp_contact_last_points[env_ids] = float("nan")
        self._grasp_contact_last_forces[env_ids] = 0.0
        return super()._reset_tasks_callback(env_ids)

    @override
    def _reset_root_states(self, env_ids, target_root_states=None):
        self.target_robot_root_states[env_ids] = self.base_init_state
        self.target_robot_root_states[env_ids, 7:13] = 0.0
        door_state = self.simulator.get_task_root_state("door")
        remaining = env_ids
        candidate_count = 0
        rejected_count = 0
        max_attempts_used = 0
        for attempt in range(1, self._reset_sampling_max_attempts + 1):
            if len(remaining) == 0:
                break
            max_attempts_used = attempt
            candidate_count += len(remaining)
            self._sample_initial_root_target(remaining, door_state[remaining])
            collision = initial_gripper_door_collision(
                self.target_robot_root_states[remaining, :3],
                self.target_robot_root_states[remaining, 3:7],
                self.target_robot_dof_state[
                    remaining[:, None], self.arm_dof_idx[None, :], 0
                ],
                door_state[remaining, :3],
                door_state[remaining, 3:7],
                self.door_width[remaining],
                self.door_height[remaining],
                self.door_handle_height[remaining],
                self.door_handle_width[remaining],
                self.door_open_lr[remaining],
                self.door_axle_length[remaining],
                self.door_handle_length[remaining],
                self.door_handle_radius[remaining],
                self.door_total_wall_height[remaining],
                separation_margin=self._reset_collision_margin,
            )
            rejected_count += int(collision.sum().item())
            remaining = remaining[collision]
            if len(remaining) > 0:
                self._sample_initial_arm_target(remaining)
        if len(remaining) > 0:
            raise RuntimeError(
                "Unable to sample collision-free B2Z1 reset states for "
                f"{len(remaining)} environments after "
                f"{self._reset_sampling_max_attempts} attempts"
            )
        self.log_dict["reset/geometry_rejection_rate"] = torch.tensor(
            rejected_count / max(candidate_count, 1), device=self.device
        )
        self.log_dict["reset/geometry_max_attempts"] = torch.tensor(
            float(max_attempts_used), device=self.device
        )

    def _sample_initial_root_target(self, env_ids, door_state):
        frame_pos = door_state[:, :3]
        frame_quat = door_state[:, 3:7]
        local_pos = torch.zeros(len(env_ids), 3, device=self.device)
        local_pos[:, 0] = torch_rand_float(
            0.55, 2.0, (len(env_ids), 1), device=str(self.device)
        )[:, 0]
        local_pos[:, 1] = torch_rand_float(
            -2.0, 2.0, (len(env_ids), 1), device=str(self.device)
        )[:, 0]
        world_pos = frame_pos + quat_apply(frame_quat, local_pos)
        world_pos[:, 2] = self.base_init_state[2] + self.env_origins[env_ids, 2]
        self.target_robot_root_states[env_ids, :3] = world_pos

        through = -quat_apply(
            frame_quat,
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(len(env_ids), -1),
        )
        target_yaw = torch.atan2(through[:, 1], through[:, 0])
        yaw_noise = torch_rand_float(
            -torch.pi / 3.0, torch.pi / 3.0, (len(env_ids), 1), device=str(self.device)
        )[:, 0]
        self.target_robot_root_states[env_ids, 3:7] = wxyz_to_xyzw(
            quat_from_euler_xyz(
                torch.zeros_like(target_yaw),
                torch.zeros_like(target_yaw),
                target_yaw + yaw_noise,
            )
        )

    def _sample_initial_arm_target(self, env_ids):
        arm_noise = torch_rand_float(
            -0.12, 0.12, (len(env_ids), len(self.arm_dof_idx)), device=str(self.device)
        )
        arm_target = self._middle_arm[None, :] + arm_noise
        lower = self.simulator.hard_dof_pos_limits[self.arm_dof_idx, 0]
        upper = self.simulator.hard_dof_pos_limits[self.arm_dof_idx, 1]
        arm_target = torch.maximum(torch.minimum(arm_target, upper), lower)
        self.target_robot_dof_state[
            env_ids[:, None], self.arm_dof_idx[None, :], 0
        ] = arm_target

    @override
    def _reset_dofs(self, env_ids, target_state=None):
        self.target_robot_dof_state[env_ids, :, 0] = self.default_dof_pos
        self.target_robot_dof_state[env_ids, :, 1] = 0.0
        self._sample_initial_arm_target(env_ids)
        self.target_robot_dof_state[env_ids[:, None], self.gripper_dof_idx[None, :], 0] = 0.0

    @override
    def _staged_reset_snapshot_exclusion(self):
        return self._task_safety_termination

    def _grasp_eval_event_features(self, pocket_distance, acceptance_count):
        previous_error = torch.where(
            self._eval_grasp_previous_error_valid[:, None],
            self._eval_grasp_previous_error,
            self._pocket_error_frame,
        )
        stage_progress = self._stage_progress[:, self.STAGE_GRASP]
        previous_progress = self._task_stage_best[:, self.STAGE_GRASP]
        progress_initialized = self._task_stage_progress_initialized[
            :, self.STAGE_GRASP
        ]
        progress_delta = torch.where(
            progress_initialized,
            stage_progress - previous_progress,
            0.0,
        )
        progress_reward = (
            20.0
            * progress_delta
            / self.dt
            * self.reward_scales["task_stage_progress"]
        )
        residual_reward = (
            (1.0 - stage_progress)
            * self.reward_scales["penalty_stage_residual"]
        )
        gripper_speed = torch.abs(
            self.simulator.dof_vel[:, self.gripper_dof_idx]
        ).amax(dim=-1)
        return {
            "metadata": torch.stack(
                [
                    self.episode_length_buf.float(),
                    self.door_open_lr.float(),
                    pocket_distance,
                    acceptance_count,
                ],
                dim=-1,
            ),
            "error_previous": previous_error,
            "error_current": self._pocket_error_frame,
            "error_delta": self._pocket_error_frame - previous_error,
            "policy_mean": self._eval_policy_mean,
            "policy_delta": self._last_delta_actions,
            "accumulated_command": self._delta_actions,
            "physical_command": self._b2z1_commands,
            "unclipped_command": self._b2z1_commands_unclipped,
            "stage_components": self._grasp_component_progress,
            "acceptance_checks": self._grasp_acceptance_checks.float(),
            "reward_terms": torch.stack(
                [
                    stage_progress,
                    progress_delta,
                    progress_reward,
                    residual_reward,
                    self._grasp_component_progress[:, 2],
                    self._grasp_component_progress[:, 3],
                    self._grasp_component_progress[:, 4],
                ],
                dim=-1,
            ),
            "contact_terms": torch.stack(
                [
                    self._grasp_aperture_quality,
                    self._grasp_between_jaws_quality,
                    self._grasp_closure_quality,
                    self._grasp_capture_progress,
                    self._grasp_bilateral_force_progress,
                    self._gripper_handle_relative_speed,
                    gripper_speed,
                    self._gripper_axis_error,
                ],
                dim=-1,
            ),
        }

    def _update_grasp_eval_event_snapshots(self, pocket_distance):
        acceptance_count = self._grasp_acceptance_checks.float().sum(dim=-1)
        active = self.stage_buf == self.STAGE_GRASP
        closest = active & closest_event_mask(
            pocket_distance, self._eval_grasp_closest_distance
        )
        max_acceptance = active & max_acceptance_event_mask(
            acceptance_count,
            pocket_distance,
            self._eval_grasp_max_acceptance_count,
            self._eval_grasp_max_acceptance_distance,
        )
        features = self._grasp_eval_event_features(
            pocket_distance, acceptance_count
        )
        for event_name, replace in (
            ("closest_pocket", closest),
            ("max_acceptance", max_acceptance),
        ):
            update_event_snapshot(
                self._eval_grasp_event_snapshots[event_name],
                replace,
                features,
                self._pocket_error_frame,
            )
        self._eval_grasp_closest_distance = torch.where(
            closest, pocket_distance, self._eval_grasp_closest_distance
        )
        self._eval_grasp_max_acceptance_count = torch.where(
            max_acceptance,
            acceptance_count,
            self._eval_grasp_max_acceptance_count,
        )
        self._eval_grasp_max_acceptance_distance = torch.where(
            max_acceptance,
            pocket_distance,
            self._eval_grasp_max_acceptance_distance,
        )
        self._eval_grasp_previous_error.copy_(self._pocket_error_frame)
        self._eval_grasp_previous_error_valid |= active

    def _unsafe_force_diagnostic_features(
        self,
        handle_force,
        frame_force,
        other_force,
    ):
        door = self.simulator.scene.articulations["door"]
        gripper_linear_velocity = self.simulator._rigid_body_vel[
            :, self.gripper_idx, :
        ]
        gripper_angular_velocity = self.simulator._rigid_body_ang_vel[
            :, self.gripper_idx, :
        ]
        handle_linear_velocity = door.data.body_lin_vel_w[:, self._door_handle_idx]
        task_basis, _ = self._door_task_basis()
        handle_relative_velocity_frame = world_polar_to_task(
            gripper_linear_velocity - handle_linear_velocity,
            task_basis,
        )
        force_components = torch.stack(
            [
                self._finger_handle_forces[:, 0],
                self._finger_handle_forces[:, 1],
                handle_force,
                self._panel_gripper_force,
                frame_force,
                other_force,
                torch.maximum(handle_force, self._panel_gripper_force),
            ],
            dim=-1,
        )
        return {
            "episode_step": self.episode_length_buf.float()[:, None],
            "total_time": self.total_time_buf.float()[:, None],
            "stage": self.stage_buf.float()[:, None],
            "hinge_side": self.door_open_lr.float()[:, None],
            "pocket_signed_error": self._pocket_error_frame,
            "axis_error": self._gripper_axis_error[:, None],
            "handle_relative_velocity_frame": handle_relative_velocity_frame,
            "handle_relative_speed": self._gripper_handle_relative_speed[:, None],
            "panel_relative_velocity_frame": self._gripper_panel_relative_velocity_frame,
            "panel_relative_speed": self._gripper_panel_relative_speed[:, None],
            "panel_clearance": self._panel_gripper_clearance[:, None],
            "force_components": force_components,
            "base_actual_velocity": self.simulator.robot_root_states[:, 7:13],
            "ee_linear_velocity": gripper_linear_velocity,
            "ee_angular_velocity": gripper_angular_velocity,
            "policy_mean": self._unsafe_force_policy_mean,
            "sampled_action": self._unsafe_force_sampled_action,
            "physical_command": self._b2z1_commands,
            "unclipped_physical_command": self._b2z1_commands_unclipped,
            "gripper_dof_position": self.simulator.dof_pos[:, self.gripper_dof_idx],
            "gripper_dof_velocity": self.simulator.dof_vel[:, self.gripper_dof_idx],
        }

    @override
    def _check_termination(self):
        handle_force = self._finger_handle_forces.max(dim=-1).values
        max_task_force = torch.maximum(handle_force, self._panel_gripper_force)
        force_termination = max_task_force > 600.0
        if hasattr(self, "_unsafe_force_event_recorder"):
            frame_force = self._reward_penalty_door_frame_contact()
            if len(self.penalised_contact_indices) > 0:
                other_force = torch.linalg.vector_norm(
                    self.simulator.contact_forces[
                        :, self.penalised_contact_indices, :
                    ],
                    dim=-1,
                ).amax(dim=-1)
            else:
                other_force = torch.zeros_like(handle_force)
            # The 600 N termination currently depends only on handle and panel.
            # Frame and other forces are retained to rule out attribution errors.
            trigger_object = torch.where(
                handle_force >= self._panel_gripper_force,
                torch.zeros_like(self.stage_buf),
                torch.ones_like(self.stage_buf),
            )
            self._unsafe_force_event_recorder.record(
                force_termination,
                trigger_object,
                self._unsafe_force_diagnostic_features(
                    handle_force, frame_force, other_force
                ),
            )
        overspeed_termination = self._door_overspeed_steps >= 3
        pushing = self.stage_buf == self.STAGE_PUSH_OPEN
        shoulder_termination = pushing & (self._shoulder_panel_clearance < -0.02)
        self._task_safety_termination = (
            force_termination | overspeed_termination | shoulder_termination
        )
        super()._check_termination()
        stage_overtime = self.time_in_stage_buf >= self.max_stage_time[self.stage_buf]
        existing_termination = self.reset_buf & ~stage_overtime

        self.reset_buf |= force_termination
        self.reset_buf |= overspeed_termination
        self.reset_buf |= shoulder_termination

        if hasattr(self, "eval_metrics"):
            current = self.eval_metrics
            current["_current_max_hinge_angle"] = torch.maximum(
                current["_current_max_hinge_angle"], self._hinge_angle
            )
            current["_current_final_hinge_angle"].copy_(self._hinge_angle)
            current["_current_max_hinge_speed"] = torch.maximum(
                current["_current_max_hinge_speed"], self._hinge_speed
            )
            current["_current_max_hinge_opening_speed"] = torch.maximum(
                current["_current_max_hinge_opening_speed"],
                self._hinge_opening_speed,
            )
            current["_current_max_panel_force"] = torch.maximum(
                current["_current_max_panel_force"], self._panel_gripper_force
            )
            current["_current_max_handle_force"] = torch.maximum(
                current["_current_max_handle_force"], handle_force
            )
            current["_current_max_task_force"] = torch.maximum(
                current["_current_max_task_force"], max_task_force
            )
            current["_current_min_shoulder_clearance"] = torch.minimum(
                current["_current_min_shoulder_clearance"],
                self._shoulder_panel_clearance,
            )
            current["_current_contact_seen"] |= self._push_gripper_contact_seen
            current["_current_unsafe_force"] |= force_termination
            current["_current_door_overspeed"] |= overspeed_termination
            current["_current_shoulder_collision"] |= shoulder_termination
            current["_current_overtime"] |= stage_overtime
            for name, value in (
                ("aperture_quality", self._grasp_aperture_quality),
                ("between_jaws_quality", self._grasp_between_jaws_quality),
                ("capture_progress", self._grasp_capture_progress),
                (
                    "acceptance_count",
                    self._grasp_acceptance_checks.float().sum(dim=-1),
                ),
            ):
                key = f"_current_max_grasp_{name}"
                current[key] = torch.maximum(current[key], value)
            pocket_abs = torch.abs(self._pocket_error_frame)
            pocket_distance = torch.linalg.vector_norm(
                self._pocket_error_frame, dim=-1
            )
            self._update_grasp_eval_event_snapshots(pocket_distance)
            for name, value in (
                ("pocket_distance", pocket_distance),
                ("pocket_normal", pocket_abs[:, 0]),
                ("pocket_tangent", pocket_abs[:, 1]),
                ("pocket_vertical", pocket_abs[:, 2]),
                ("axis_error", self._gripper_axis_error),
            ):
                key = f"_current_min_grasp_{name}"
                current[key] = torch.minimum(current[key], value)
            closer = pocket_distance < current["_current_closest_grasp_distance"]
            current["_current_closest_grasp_distance"] = torch.minimum(
                current["_current_closest_grasp_distance"], pocket_distance
            )
            for axis, name in enumerate(self._EVAL_GRASP_CLOSEST_DIAGNOSTICS):
                key = f"_current_closest_grasp_{name}"
                current[key] = torch.where(closer, pocket_abs[:, axis], current[key])
            current["_current_grasp_any_valid_contact"] |= (
                self._grasp_on_lever_capsule.any(dim=-1)
            )
            current["_current_grasp_both_valid_contact"] |= (
                self._grasp_on_lever_capsule.all(dim=-1)
            )
            current["_current_grasp_opposed_contact"] |= self._grasp_opposed_contact
            current["_current_grasp_precise_bilateral_grasp"] |= (
                self._precise_bilateral_grasp
            )
            for index, name in enumerate(self._EVAL_GRASP_ACCEPTANCE_DIAGNOSTICS):
                current[f"_current_grasp_accept_{name}"] |= (
                    self._grasp_acceptance_checks[:, index]
                )
                other_checks = torch.cat(
                    [
                        self._grasp_acceptance_checks[:, :index],
                        self._grasp_acceptance_checks[:, index + 1 :],
                    ],
                    dim=-1,
                ).all(dim=-1)
                current[f"_current_grasp_all_except_{name}"] |= other_checks
            current["_current_grasp_consecutive_precise"] = torch.where(
                self._precise_bilateral_grasp,
                current["_current_grasp_consecutive_precise"] + 1.0,
                torch.zeros_like(current["_current_grasp_consecutive_precise"]),
            )
            current["_current_max_grasp_consecutive_precise"] = torch.maximum(
                current["_current_max_grasp_consecutive_precise"],
                current["_current_grasp_consecutive_precise"],
            )
            through_checks = {
                "through_distance_ok": self._through_distance > 1.0,
                "through_arm_stowed": self._arm_safely_stowed,
                "through_door_open": self._hinge_angle >= math.radians(65.0),
                "through_base_stable": self._base_motion < 0.12,
            }
            through_checks["through_all_checks_now"] = (
                through_checks["through_distance_ok"]
                & through_checks["through_arm_stowed"]
                & through_checks["through_door_open"]
                & through_checks["through_base_stable"]
            )
            for name, value in (
                ("through_distance", self._through_distance),
                ("through_stable_steps", self._task_stage_stable[:, self.STAGE_THROUGH]),
            ):
                key = f"_current_max_{name}"
                current[key] = torch.maximum(current[key], value.float())
            for name, value in (
                ("middle_arm_error", self._middle_arm_error),
                ("middle_arm_speed", self._middle_arm_speed),
                ("base_motion", self._base_motion),
            ):
                key = f"_current_min_{name}"
                current[key] = torch.minimum(current[key], value)
            for name, value in through_checks.items():
                current[f"_current_{name}"] |= value
            current["_current_final_through_distance"].copy_(self._through_distance)
            current["_current_final_middle_arm_error"].copy_(self._middle_arm_error)
            current["_current_final_middle_arm_speed"].copy_(self._middle_arm_speed)
            current["_current_final_base_motion"].copy_(self._base_motion)
            current["_current_final_hinge_side"].copy_(self.door_open_lr.float())

        self.log_dict["termination/existing"] = existing_termination.float().mean()
        self.log_dict["termination/unsafe_force"] = force_termination.float().mean()
        self.log_dict["termination/unsafe_force/reset_step"] = (
            force_termination & (self.total_time_buf == 1)
        ).float().mean()
        self.log_dict["termination/door_overspeed"] = overspeed_termination.float().mean()
        self.log_dict["termination/shoulder_collision"] = shoulder_termination.float().mean()
        for stage_idx, stage_name in enumerate(self._STAGE_NAMES):
            self.log_dict[f"termination/unsafe_force/{stage_name}"] = (
                force_termination & (self.stage_buf == stage_idx)
            ).float().mean()
            self.log_dict[f"termination/overtime/{stage_name}"] = (
                stage_overtime & (self.stage_buf == stage_idx)
            ).float().mean()

    @override
    def init_eval_metrics_tracking(self, device):
        super().init_eval_metrics_tracking(device)
        self._eval_policy_mean = torch.zeros(
            self.num_envs, 10, dtype=torch.float32, device=device
        )
        self._eval_grasp_previous_error = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=device
        )
        self._eval_grasp_previous_error_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self._eval_grasp_closest_distance = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float32, device=device
        )
        self._eval_grasp_max_acceptance_count = torch.full(
            (self.num_envs,), -1.0, dtype=torch.float32, device=device
        )
        self._eval_grasp_max_acceptance_distance = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float32, device=device
        )
        self._eval_grasp_event_snapshots = {}
        for event_name in self._EVAL_GRASP_EVENT_NAMES:
            snapshot = {
                "pending_next": torch.zeros(
                    self.num_envs, dtype=torch.bool, device=device
                ),
                "error_next": torch.full(
                    (self.num_envs, 3),
                    float("nan"),
                    dtype=torch.float32,
                    device=device,
                ),
            }
            for field_name, width in self._EVAL_GRASP_EVENT_FIELDS.items():
                snapshot[field_name] = torch.full(
                    (self.num_envs, width),
                    float("nan"),
                    dtype=torch.float32,
                    device=device,
                )
                self.eval_metrics[
                    f"episode_grasp_{event_name}_{field_name}_buffer"
                ] = []
            self.eval_metrics[
                f"episode_grasp_{event_name}_error_next_buffer"
            ] = []
            self._eval_grasp_event_snapshots[event_name] = snapshot
        for name in self._EVAL_MAX_DIAGNOSTICS:
            self.eval_metrics[f"_current_max_{name}"] = torch.zeros(
                self.num_envs, dtype=torch.float32, device=device
            )
            self.eval_metrics[f"episode_max_{name}_buffer"] = []
        self.eval_metrics["_current_final_hinge_angle"] = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        self.eval_metrics["episode_final_hinge_angle_buffer"] = []
        self.eval_metrics["_current_min_shoulder_clearance"] = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float32, device=device
        )
        self.eval_metrics["episode_min_shoulder_clearance_buffer"] = []
        for name in self._EVAL_BOOL_DIAGNOSTICS:
            self.eval_metrics[f"_current_{name}"] = torch.zeros(
                self.num_envs, dtype=torch.bool, device=device
            )
            self.eval_metrics[f"episode_{name}_buffer"] = []
        for name in self._EVAL_GRASP_MAX_DIAGNOSTICS:
            self.eval_metrics[f"_current_max_grasp_{name}"] = torch.zeros(
                self.num_envs, dtype=torch.float32, device=device
            )
            self.eval_metrics[f"episode_max_grasp_{name}_buffer"] = []
        for name in self._EVAL_GRASP_MIN_DIAGNOSTICS:
            self.eval_metrics[f"_current_min_grasp_{name}"] = torch.full(
                (self.num_envs,), float("inf"), dtype=torch.float32, device=device
            )
            self.eval_metrics[f"episode_min_grasp_{name}_buffer"] = []
        self.eval_metrics["_current_closest_grasp_distance"] = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float32, device=device
        )
        for name in self._EVAL_GRASP_CLOSEST_DIAGNOSTICS:
            self.eval_metrics[f"_current_closest_grasp_{name}"] = torch.full(
                (self.num_envs,), float("inf"), dtype=torch.float32, device=device
            )
            self.eval_metrics[f"episode_closest_grasp_{name}_buffer"] = []
        for name in self._EVAL_GRASP_BOOL_DIAGNOSTICS:
            self.eval_metrics[f"_current_grasp_{name}"] = torch.zeros(
                self.num_envs, dtype=torch.bool, device=device
            )
            self.eval_metrics[f"episode_grasp_{name}_buffer"] = []
        for name in self._EVAL_GRASP_ACCEPTANCE_DIAGNOSTICS:
            self.eval_metrics[f"_current_grasp_accept_{name}"] = torch.zeros(
                self.num_envs, dtype=torch.bool, device=device
            )
            self.eval_metrics[f"episode_grasp_accept_{name}_buffer"] = []
            self.eval_metrics[f"_current_grasp_all_except_{name}"] = torch.zeros(
                self.num_envs, dtype=torch.bool, device=device
            )
            self.eval_metrics[f"episode_grasp_all_except_{name}_buffer"] = []
        self.eval_metrics["_current_grasp_consecutive_precise"] = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        self.eval_metrics["_current_max_grasp_consecutive_precise"] = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        self.eval_metrics["episode_max_grasp_consecutive_precise_buffer"] = []
        for name in self._EVAL_THROUGH_MAX_DIAGNOSTICS:
            self.eval_metrics[f"_current_max_{name}"] = torch.zeros(
                self.num_envs, dtype=torch.float32, device=device
            )
            self.eval_metrics[f"episode_max_{name}_buffer"] = []
        for name in self._EVAL_THROUGH_MIN_DIAGNOSTICS:
            self.eval_metrics[f"_current_min_{name}"] = torch.full(
                (self.num_envs,), float("inf"), dtype=torch.float32, device=device
            )
            self.eval_metrics[f"episode_min_{name}_buffer"] = []
        for name in self._EVAL_THROUGH_BOOL_DIAGNOSTICS:
            self.eval_metrics[f"_current_{name}"] = torch.zeros(
                self.num_envs, dtype=torch.bool, device=device
            )
            self.eval_metrics[f"episode_{name}_buffer"] = []
        for name in (
            "through_distance",
            "middle_arm_error",
            "middle_arm_speed",
            "base_motion",
            "hinge_side",
        ):
            self.eval_metrics[f"_current_final_{name}"] = torch.zeros(
                self.num_envs, dtype=torch.float32, device=device
            )
            self.eval_metrics[f"episode_final_{name}_buffer"] = []

    @override
    def process_eval_episode_completions(
        self, completed_env_ids, cur_reward_sum, cur_episode_length
    ):
        super().process_eval_episode_completions(
            completed_env_ids, cur_reward_sum, cur_episode_length
        )
        env_ids = completed_env_ids.reshape(-1)
        for event_name in self._EVAL_GRASP_EVENT_NAMES:
            snapshot = self._eval_grasp_event_snapshots[event_name]
            for field_name in (*self._EVAL_GRASP_EVENT_FIELDS, "error_next"):
                values = snapshot[field_name][env_ids]
                self.eval_metrics[
                    f"episode_grasp_{event_name}_{field_name}_buffer"
                ].extend(values.detach().cpu().tolist())
        for name in self._EVAL_MAX_DIAGNOSTICS:
            values = self.eval_metrics[f"_current_max_{name}"][env_ids]
            self.eval_metrics[f"episode_max_{name}_buffer"].extend(
                values.detach().cpu().tolist()
            )
        final_angle = self.eval_metrics["_current_final_hinge_angle"][env_ids]
        self.eval_metrics["episode_final_hinge_angle_buffer"].extend(
            final_angle.detach().cpu().tolist()
        )
        min_clearance = self.eval_metrics["_current_min_shoulder_clearance"][env_ids]
        self.eval_metrics["episode_min_shoulder_clearance_buffer"].extend(
            min_clearance.detach().cpu().tolist()
        )
        for name in self._EVAL_BOOL_DIAGNOSTICS:
            values = self.eval_metrics[f"_current_{name}"][env_ids]
            self.eval_metrics[f"episode_{name}_buffer"].extend(
                values.detach().cpu().tolist()
            )
        for name in self._EVAL_GRASP_MAX_DIAGNOSTICS:
            values = self.eval_metrics[f"_current_max_grasp_{name}"][env_ids]
            self.eval_metrics[f"episode_max_grasp_{name}_buffer"].extend(
                values.detach().cpu().tolist()
            )
        for name in self._EVAL_GRASP_MIN_DIAGNOSTICS:
            values = self.eval_metrics[f"_current_min_grasp_{name}"][env_ids]
            self.eval_metrics[f"episode_min_grasp_{name}_buffer"].extend(
                values.detach().cpu().tolist()
            )
        for name in self._EVAL_GRASP_CLOSEST_DIAGNOSTICS:
            values = self.eval_metrics[f"_current_closest_grasp_{name}"][env_ids]
            self.eval_metrics[f"episode_closest_grasp_{name}_buffer"].extend(
                values.detach().cpu().tolist()
            )
        for name in self._EVAL_GRASP_BOOL_DIAGNOSTICS:
            values = self.eval_metrics[f"_current_grasp_{name}"][env_ids]
            self.eval_metrics[f"episode_grasp_{name}_buffer"].extend(
                values.detach().cpu().tolist()
            )
        for name in self._EVAL_GRASP_ACCEPTANCE_DIAGNOSTICS:
            values = self.eval_metrics[f"_current_grasp_accept_{name}"][env_ids]
            self.eval_metrics[f"episode_grasp_accept_{name}_buffer"].extend(
                values.detach().cpu().tolist()
            )
            values = self.eval_metrics[f"_current_grasp_all_except_{name}"][env_ids]
            self.eval_metrics[f"episode_grasp_all_except_{name}_buffer"].extend(
                values.detach().cpu().tolist()
            )
        values = self.eval_metrics["_current_max_grasp_consecutive_precise"][env_ids]
        self.eval_metrics["episode_max_grasp_consecutive_precise_buffer"].extend(
            values.detach().cpu().tolist()
        )
        for prefix, names in (
            ("max", self._EVAL_THROUGH_MAX_DIAGNOSTICS),
            ("min", self._EVAL_THROUGH_MIN_DIAGNOSTICS),
        ):
            for name in names:
                values = self.eval_metrics[f"_current_{prefix}_{name}"][env_ids]
                self.eval_metrics[f"episode_{prefix}_{name}_buffer"].extend(
                    values.detach().cpu().tolist()
                )
        for name in self._EVAL_THROUGH_BOOL_DIAGNOSTICS:
            values = self.eval_metrics[f"_current_{name}"][env_ids]
            self.eval_metrics[f"episode_{name}_buffer"].extend(
                values.detach().cpu().tolist()
            )
        for name in (
            "through_distance",
            "middle_arm_error",
            "middle_arm_speed",
            "base_motion",
            "hinge_side",
        ):
            values = self.eval_metrics[f"_current_final_{name}"][env_ids]
            self.eval_metrics[f"episode_final_{name}_buffer"].extend(
                values.detach().cpu().tolist()
            )

    @override
    def reset_eval_episode_tracking(self, completed_env_ids):
        super().reset_eval_episode_tracking(completed_env_ids)
        env_ids = completed_env_ids.reshape(-1)
        self._eval_policy_mean[env_ids] = 0.0
        self._eval_grasp_previous_error[env_ids] = 0.0
        self._eval_grasp_previous_error_valid[env_ids] = False
        self._eval_grasp_closest_distance[env_ids] = float("inf")
        self._eval_grasp_max_acceptance_count[env_ids] = -1.0
        self._eval_grasp_max_acceptance_distance[env_ids] = float("inf")
        for event_name in self._EVAL_GRASP_EVENT_NAMES:
            snapshot = self._eval_grasp_event_snapshots[event_name]
            snapshot["pending_next"][env_ids] = False
            snapshot["error_next"][env_ids] = float("nan")
            for field_name in self._EVAL_GRASP_EVENT_FIELDS:
                snapshot[field_name][env_ids] = float("nan")
        for name in self._EVAL_MAX_DIAGNOSTICS:
            self.eval_metrics[f"_current_max_{name}"][env_ids] = 0.0
        self.eval_metrics["_current_final_hinge_angle"][env_ids] = 0.0
        self.eval_metrics["_current_min_shoulder_clearance"][env_ids] = float("inf")
        for name in self._EVAL_BOOL_DIAGNOSTICS:
            self.eval_metrics[f"_current_{name}"][env_ids] = False
        for name in self._EVAL_GRASP_MAX_DIAGNOSTICS:
            self.eval_metrics[f"_current_max_grasp_{name}"][env_ids] = 0.0
        for name in self._EVAL_GRASP_MIN_DIAGNOSTICS:
            self.eval_metrics[f"_current_min_grasp_{name}"][env_ids] = float("inf")
        self.eval_metrics["_current_closest_grasp_distance"][env_ids] = float("inf")
        for name in self._EVAL_GRASP_CLOSEST_DIAGNOSTICS:
            self.eval_metrics[f"_current_closest_grasp_{name}"][env_ids] = float("inf")
        for name in self._EVAL_GRASP_BOOL_DIAGNOSTICS:
            self.eval_metrics[f"_current_grasp_{name}"][env_ids] = False
        for name in self._EVAL_GRASP_ACCEPTANCE_DIAGNOSTICS:
            self.eval_metrics[f"_current_grasp_accept_{name}"][env_ids] = False
            self.eval_metrics[f"_current_grasp_all_except_{name}"][env_ids] = False
        self.eval_metrics["_current_grasp_consecutive_precise"][env_ids] = 0.0
        self.eval_metrics["_current_max_grasp_consecutive_precise"][env_ids] = 0.0
        for name in self._EVAL_THROUGH_MAX_DIAGNOSTICS:
            self.eval_metrics[f"_current_max_{name}"][env_ids] = 0.0
        for name in self._EVAL_THROUGH_MIN_DIAGNOSTICS:
            self.eval_metrics[f"_current_min_{name}"][env_ids] = float("inf")
        for name in self._EVAL_THROUGH_BOOL_DIAGNOSTICS:
            self.eval_metrics[f"_current_{name}"][env_ids] = False
        for name in (
            "through_distance",
            "middle_arm_error",
            "middle_arm_speed",
            "base_motion",
            "hinge_side",
        ):
            self.eval_metrics[f"_current_final_{name}"][env_ids] = 0.0

    @override
    def get_eval_metrics_summary(self):
        summary = super().get_eval_metrics_summary()
        result = {
            key: value
            for key, value in summary.items()
            if not key.startswith("_current_") and not key.endswith("_buffer")
        }
        result["goal_reached_buffer"] = list(summary["goal_reached_buffer"])
        result["grasp_event_metadata_fields"] = (
            "episode_step",
            "hinge_side",
            "pocket_distance",
            "acceptance_count",
        )
        result["grasp_event_action_fields"] = (
            "base_x",
            "base_y",
            "base_yaw",
            "ee_x",
            "ee_y",
            "ee_z",
            "ee_roll",
            "ee_pitch",
            "ee_yaw",
            "gripper",
        )
        result["grasp_event_stage_component_fields"] = (
            "base_position",
            "base_yaw",
            "approach",
            "orientation",
            "contact",
        )
        result["grasp_event_acceptance_fields"] = self._EVAL_GRASP_ACCEPTANCE_DIAGNOSTICS
        result["grasp_event_reward_fields"] = (
            "stage_progress",
            "stage_progress_delta",
            "stage_progress_reward_contribution",
            "stage_residual_reward_contribution",
            "approach_progress",
            "orientation_progress",
            "contact_progress",
        )
        result["grasp_event_contact_fields"] = (
            "aperture_quality",
            "between_jaws_quality",
            "closure_quality",
            "capture_progress",
            "bilateral_force_progress",
            "relative_speed",
            "gripper_speed",
            "axis_error",
        )
        for event_name in self._EVAL_GRASP_EVENT_NAMES:
            for field_name in (*self._EVAL_GRASP_EVENT_FIELDS, "error_next"):
                result[f"episode_grasp_{event_name}_{field_name}"] = list(
                    summary[
                        f"episode_grasp_{event_name}_{field_name}_buffer"
                    ]
                )
        for name in self._EVAL_MAX_DIAGNOSTICS:
            result[f"episode_max_{name}"] = list(
                summary[f"episode_max_{name}_buffer"]
            )
        result["episode_final_hinge_angle"] = list(
            summary["episode_final_hinge_angle_buffer"]
        )
        result["episode_min_shoulder_clearance"] = list(
            summary["episode_min_shoulder_clearance_buffer"]
        )
        for name in self._EVAL_BOOL_DIAGNOSTICS:
            result[f"episode_{name}"] = list(
                summary[f"episode_{name}_buffer"]
            )
        for name in self._EVAL_GRASP_MAX_DIAGNOSTICS:
            result[f"episode_max_grasp_{name}"] = list(
                summary[f"episode_max_grasp_{name}_buffer"]
            )
        for name in self._EVAL_GRASP_MIN_DIAGNOSTICS:
            result[f"episode_min_grasp_{name}"] = list(
                summary[f"episode_min_grasp_{name}_buffer"]
            )
        for name in self._EVAL_GRASP_CLOSEST_DIAGNOSTICS:
            result[f"episode_closest_grasp_{name}"] = list(
                summary[f"episode_closest_grasp_{name}_buffer"]
            )
        for name in self._EVAL_GRASP_BOOL_DIAGNOSTICS:
            result[f"episode_grasp_{name}"] = list(
                summary[f"episode_grasp_{name}_buffer"]
            )
        for name in self._EVAL_GRASP_ACCEPTANCE_DIAGNOSTICS:
            result[f"episode_grasp_accept_{name}"] = list(
                summary[f"episode_grasp_accept_{name}_buffer"]
            )
            result[f"episode_grasp_all_except_{name}"] = list(
                summary[f"episode_grasp_all_except_{name}_buffer"]
            )
        result["episode_max_grasp_consecutive_precise"] = list(
            summary["episode_max_grasp_consecutive_precise_buffer"]
        )
        for prefix, names in (
            ("max", self._EVAL_THROUGH_MAX_DIAGNOSTICS),
            ("min", self._EVAL_THROUGH_MIN_DIAGNOSTICS),
        ):
            for name in names:
                result[f"episode_{prefix}_{name}"] = list(
                    summary[f"episode_{prefix}_{name}_buffer"]
                )
        for name in self._EVAL_THROUGH_BOOL_DIAGNOSTICS:
            result[f"episode_{name}"] = list(summary[f"episode_{name}_buffer"])
        for name in (
            "through_distance",
            "middle_arm_error",
            "middle_arm_speed",
            "base_motion",
            "hinge_side",
        ):
            result[f"episode_final_{name}"] = list(
                summary[f"episode_final_{name}_buffer"]
            )
        return result

    @override
    def scene_creation_callback(self, simulator):
        super().scene_creation_callback(simulator)
        for sensor_name, body_name in (
            ("b2z1_fixed_handle_contact", "link06"),
            ("b2z1_moving_handle_contact", "gripperMover"),
        ):
            cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{body_name}",
                history_length=8,
                track_air_time=False,
                track_contact_points=True,
                max_contact_data_count_per_prim=128,
                filter_prim_paths_expr=[
                    f"/World/envs/env_.*/{simulator.task_config.target_obj}/door_handle"
                ],
            )
            simulator.scene.sensors[sensor_name] = ContactSensor(cfg)
        prim_names = simulator.robot_config.task_contact_prim_names
        jaw_paths = [f"/World/envs/env_.*/Robot/{name}" for name in prim_names]
        cfg = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/{simulator.task_config.target_obj}/door_panel",
            filter_prim_paths_expr=jaw_paths,
        )
        simulator.scene.sensors["b2z1_panel_gripper_contact_sensor"] = ContactSensor(cfg)
