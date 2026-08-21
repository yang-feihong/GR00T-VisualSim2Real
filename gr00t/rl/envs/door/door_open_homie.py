# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import omni.usd
import torch
import torch.nn.functional as F
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.utils.math import (
    axis_angle_from_quat,
    euler_xyz_from_quat,
    quat_apply,
    quat_from_euler_xyz,
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
from gr00t.rl.envs.base_task.warped_action_base import WarpedActionBase
from gr00t.rl.isaac_utils.rotations import quat_to_tan_norm, wxyz_to_xyzw, xyzw_to_wxyz
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
        self.door_weight = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_open_lr = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.door_open_io = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for env_id in range(self.num_envs):
            door_prim_path = f"/World/envs/env_{env_id}/door"
            door_prim = stage.GetPrimAtPath(door_prim_path)
            door_metadata = door_prim.GetPrim().GetMetadata("customData")
            self.door_width[env_id] = door_metadata["doorWidth"]
            self.door_height[env_id] = door_metadata["doorHeight"]
            self.door_handle_height[env_id] = door_metadata["doorHandleHeight"]
            self.door_handle_width[env_id] = door_metadata["doorHandleWidth"]
            self.door_weight[env_id] = door_metadata["doorWeight"]
            self.door_open_lr[env_id] = door_metadata["doorOpenLR"]
            self.door_open_io[env_id] = door_metadata["doorOpenIO"]

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

    def _get_obs_door_dof_pos(self):
        return self.simulator.get_task_dof_pos("door")[:, :2]

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
        self.reset_buf |= self.relative_door_pos_buf.norm(dim=-1) > 4.0

        dof_overspeed = torch.any(
            torch.abs(self.simulator.dof_vel[:, self._upper_non_finger_dof_idx])
            > self.termination_level * self._arm_termination_vel_limits,
            dim=-1,
        )
        not_just_resetted = self.episode_length_buf > 20

        self.reset_buf |= dof_overspeed & not_just_resetted

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
