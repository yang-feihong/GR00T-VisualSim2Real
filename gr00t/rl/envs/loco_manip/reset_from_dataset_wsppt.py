# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
from typing_extensions import override

from gr00t.rl.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from gr00t.rl.utils.teleop_reset_lib.dataset_wsppt import MotionDatasetWSPPT
from gr00t.rl.utils.torch_utils import *


class ResetFromDatasetWSPPT(LeggedRobotBase):
    def __init__(self, config, device):
        super().__init__(config, device)
        config.reset_from_dataset.device = device
        self.reset_dataset = MotionDatasetWSPPT(config.reset_from_dataset)
        self.reset_count = 0

        self.groot2motion_dof_idx = [
            self.simulator.dof_names.index(dof_name) for dof_name in self.reset_dataset.dof_names
        ]
        self.groot2motion_body_idx = [
            (
                self.simulator.body_names.index(body_name)
                if body_name in self.simulator.body_names
                else -1
            )
            for body_name in self.reset_dataset.body_names
        ]

    def _post_compute_observations_callback(self):
        self.reset_count += 1
        if self.reset_count % self.config.reset_from_dataset.resample_every == 0:
            self.reset_dataset.resample()
            self.reset_count = 0
        return super()._post_compute_observations_callback()

    def _get_sample_idx(self, env_ids, processed_ratio=None, reset_from_teleop_start=False):
        env_ids_to_motion_idx = env_ids % len(
            self.reset_dataset.sample_idx_based_on_demo_clip_number
        )
        if reset_from_teleop_start:
            self.sample_idx = torch.zeros((len(env_ids),), device=self.device)
            for i in range(len(env_ids)):
                self.sample_idx[i] = self.reset_dataset.sample_idx_based_on_demo_clip_number[
                    env_ids_to_motion_idx[i]
                ][0]
            self.sample_idx = self.sample_idx.int()
            return self.sample_idx
        if processed_ratio is None:
            self.sample_idx = torch.zeros((len(env_ids),), device=self.device)
            for i in range(len(env_ids)):
                self.sample_idx[i] = torch.randint(
                    self.reset_dataset.sample_idx_based_on_demo_clip_number[
                        env_ids_to_motion_idx[i]
                    ][0],
                    self.reset_dataset.sample_idx_based_on_demo_clip_number[
                        env_ids_to_motion_idx[i]
                    ][1],
                    (1,),
                    device=self.device,
                ).squeeze(-1)
            self.sample_idx = self.sample_idx.int()
        else:
            self.sample_idx = torch.zeros((len(env_ids),), device=self.device)
            for i in range(len(env_ids)):
                upper_bound = self.reset_dataset.sample_idx_based_on_demo_clip_number[
                    env_ids_to_motion_idx[i]
                ][1]
                lower_bound = self.reset_dataset.sample_idx_based_on_demo_clip_number[
                    env_ids_to_motion_idx[i]
                ][0]
                index = lower_bound + int(processed_ratio[i] * (upper_bound - lower_bound))
                index = max(min(index, upper_bound), lower_bound)
                self.sample_idx[i] = (
                    index  # torch.randint(lower_bound, upper, (1,), device=self.device).squeeze(-1)
                )
            self.sample_idx = self.sample_idx.int()
        return self.sample_idx

    @override
    def _reset_robot_states_callback(self, env_ids, target_states=None):
        if self.config.reset_from_dataset.enable:
            if not hasattr(self, "sample_idx") or not hasattr(self, "sample_idx_in_seq"):
                self.sample_idx = self._get_sample_idx(env_ids)
                # self.sample_idx = torch.randint(0, len(self.reset_dataset), (len(env_ids),), device=self.device)
                self.sample_idx_in_seq = torch.randint(
                    0, self.reset_dataset.num_per_sample, (len(env_ids),), device=self.device
                )
                # logger.warning("robot sample idx: {self.sample_idx}, {self.sample_idx_in_seq}")
                if self.config.reset_from_dataset.get("reset_from_teleop_curriculum", False):
                    _ratio = self.reset_from_teleop_curri_ratio
                    _ratio = float(max(0.0, min(1.0, float(_ratio))))
                    # processed_ratio = torch.randn(len(env_ids), device=self.device) * 0.1 + _ratio
                    # _upper_bound = max(1, int(len(self.reset_dataset) * (1.0 - _ratio)))
                    ratio = torch.as_tensor(_ratio, dtype=torch.float32, device=self.device).clamp_(
                        0.0, 1.0
                    )
                    N = len(env_ids)
                    if ratio.ndim == 0:
                        ratio = ratio.expand(N)
                    else:
                        assert ratio.shape[0] == N, "ratio shape must match len(env_ids)"
                    alpha_min = 0.20
                    beta_max = 6.00
                    alpha = 1.0 - ratio * (1.0 - alpha_min)  # alpha: 1 -> alpha_min
                    beta = 1.0 + ratio * (beta_max - 1.0)  # beta : 1 -> beta_max
                    eps = 1e-4
                    alpha = alpha.clamp_min(eps)
                    beta = beta.clamp_min(eps)

                    dist = torch.distributions.Beta(alpha, beta)
                    processed_ratio = dist.rsample()  # shape [N], support (0,1)

                    processed_ratio = processed_ratio.clamp_(0.0, 1.0)
                    self.sample_idx = self._get_sample_idx(env_ids, processed_ratio)

                    self.sample_idx_in_seq = torch.randint(
                        0, self.reset_dataset.num_per_sample, (len(env_ids),), device=self.device
                    )
                    logger.warning(
                        f"reset_from_teleop_curriculum: {self.sample_idx}, {self.sample_idx_in_seq}"
                    )
                if self.config.get("reset_from_teleop_start", False):
                    self.sample_idx = self._get_sample_idx(env_ids, reset_from_teleop_start=True)
                    self.sample_idx_in_seq = torch.randint(
                        0, self.reset_dataset.num_per_sample, (len(env_ids),), device=self.device
                    )
                    logger.warning(
                        f"reset_from_teleop_start: {self.sample_idx}, {self.sample_idx_in_seq}"
                    )

            # logger.warning(f"reset_robot_states_callback: {self.sample_idx}, {self.sample_idx_in_seq}")
            # logger.warning(f"robot sample idx: {self.sample_idx}, {self.sample_idx_in_seq}") # for debug
            dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel = (
                self.reset_dataset.get_item_robot_sample(self.sample_idx)
            )
            dof_pos = dof_pos[torch.arange(len(env_ids)), self.sample_idx_in_seq]
            dof_vel = dof_vel[torch.arange(len(env_ids)), self.sample_idx_in_seq]
            body_pos = body_pos[torch.arange(len(env_ids)), self.sample_idx_in_seq]
            body_rot = body_rot[torch.arange(len(env_ids)), self.sample_idx_in_seq]
            body_lin_vel = body_lin_vel[torch.arange(len(env_ids)), self.sample_idx_in_seq]
            body_ang_vel = body_ang_vel[torch.arange(len(env_ids)), self.sample_idx_in_seq]

            root_states = torch.zeros((len(env_ids), 13), device=self.device)
            root_states[:, :3] = (
                body_pos[:, 0] + self.env_origins[env_ids, :3]
            )  # TODO(haoru): more rigorous check on if the first body is the root
            root_states[:, 3:7] = body_rot[:, 0]
            root_states[:, 7:10] = body_lin_vel[:, 0]
            root_states[:, 10:13] = body_ang_vel[:, 0]

            groot_dof_pos = torch.tile(self.default_dof_pos, (len(env_ids), 1))
            groot_dof_pos[:, self.groot2motion_dof_idx] = dof_pos

            groot_dof_vel = torch.zeros((len(env_ids), self.num_dof), device=self.device)
            groot_dof_vel[:, self.groot2motion_dof_idx] = dof_vel

            self.target_robot_dof_state[env_ids, :, 0] = groot_dof_pos
            self.target_robot_dof_state[env_ids, :, 1] = groot_dof_vel
            self.target_robot_root_states[env_ids, :] = root_states

            # Hardcode assuming there is 11-dim homie comamnds
            # import ipdb; ipdb.set_trace()

            if self.config.get("reset_delta_actions_with_backmap", False):
                homie_commands, target_dof_positions, finger_primitive_actions = (
                    self.reset_dataset.get_item_homie_command_sample(self.sample_idx)
                )
                homie_commands = homie_commands[torch.arange(len(env_ids)), self.sample_idx_in_seq]
                target_dof_positions = target_dof_positions[
                    torch.arange(len(env_ids)), self.sample_idx_in_seq
                ]
                finger_primitive_actions = finger_primitive_actions[
                    torch.arange(len(env_ids)), self.sample_idx_in_seq
                ]

                groot_dof_pos_target = torch.tile(self.default_dof_pos, (len(env_ids), 1))
                groot_dof_pos_target[:, self.groot2motion_dof_idx] = target_dof_positions

                # Linvel & Angular Vel
                # import ipdb; ipdb.set_trace()
                # self._last_delta_actions[env_ids, 0:3] = homie_commands[:, 0:3]
                # self._delta_actions[env_ids, 0:3] = homie_commands[:, 0:3]

                # reset from root state
                linvel_in_world_frame = root_states[:, 7:10]
                linvel_in_local_frame = quat_rotate_inverse(
                    root_states[:, 3:7], linvel_in_world_frame
                )
                self._last_delta_actions[env_ids, 0:3] = linvel_in_local_frame
                self._delta_actions[env_ids, 0:3] = linvel_in_local_frame

                # ARM
                self._last_delta_actions[env_ids, 3:10] = (
                    groot_dof_pos_target[:, 22:29] * 4.0
                )  # right arm because self.delta_action got *0.25 + default joint angle before doing actuation
                self._delta_actions[env_ids, 3:10] = (
                    groot_dof_pos_target[:, 22:29] * 4.0
                )  # right arm

                # FINGER
                # infer finger primitive action from finger dof pos
                # random init between -0.5 and 0.5
                self._last_delta_actions[env_ids, 10:11] = finger_primitive_actions[
                    :, 1:2
                ]  # right hand
                self._delta_actions[env_ids, 10:11] = finger_primitive_actions[:, 1:2]  # right hand

                # import ipdb; ipdb.set_trace()

                # logger.warning(f"[LINVEL]reset_delta_actions_from_dataset: {self._delta_actions[env_ids, 0:2]}")
                # logger.warning(f"[ARM]reset_delta_actions_from_dataset: {self._delta_actions[env_ids, 3:10]}")
                # logger.warning(f"[FINGER]reset_delta_actions_from_dataset: {self._delta_actions[env_ids, 10:11]}")

            # logger.warning(f"reset_dof_state_from_dataset: {groot_dof_pos}")
            # logger.warning(f"reset_root_state_from_dataset: {root_states}")
        else:
            super()._reset_robot_states_callback(env_ids, target_states)

    @override
    def _reset_task_states(self, env_ids):
        # self.sample_idx = torch.randint(0, len(self.reset_dataset), (len(env_ids),), device=self.device)
        self.sample_idx = self._get_sample_idx(env_ids)
        self.sample_idx_in_seq = torch.randint(
            0, self.reset_dataset.num_per_sample, (len(env_ids),), device=self.device
        )

        if self.config.get("reset_from_teleop_start", False):
            self.sample_idx = self._get_sample_idx(env_ids, reset_from_teleop_start=True)
            self.sample_idx_in_seq = torch.randint(
                0, self.reset_dataset.num_per_sample, (len(env_ids),), device=self.device
            )
            logger.warning(f"reset_from_teleop_start: {self.sample_idx}, {self.sample_idx_in_seq}")
        if self.config.reset_from_dataset.get("reset_from_teleop_curriculum", False):
            _ratio = self.reset_from_teleop_curri_ratio
            _ratio = float(max(0.0, min(1.0, float(_ratio))))
            # processed_ratio = torch.randn(len(env_ids), device=self.device) * 0.1 + _ratio
            # _upper_bound = max(1, int(len(self.reset_dataset) * (1.0 - _ratio)))
            ratio = torch.as_tensor(_ratio, dtype=torch.float32, device=self.device).clamp_(
                0.0, 1.0
            )
            N = len(env_ids)
            if ratio.ndim == 0:
                ratio = ratio.expand(N)
            else:
                assert ratio.shape[0] == N, "ratio shape must match len(env_ids)"

            alpha_min = 0.20
            beta_max = 6.00
            alpha = 1.0 - ratio * (1.0 - alpha_min)  # alpha: 1 -> alpha_min
            beta = 1.0 + ratio * (beta_max - 1.0)  # beta : 1 -> beta_max

            eps = 1e-4
            alpha = alpha.clamp_min(eps)
            beta = beta.clamp_min(eps)

            dist = torch.distributions.Beta(alpha, beta)
            processed_ratio = dist.rsample()  # shape [N], support (0,1)

            processed_ratio = processed_ratio.clamp_(0.0, 1.0)
            self.sample_idx = self._get_sample_idx(env_ids, processed_ratio)

            self.sample_idx_in_seq = torch.randint(
                0, self.reset_dataset.num_per_sample, (len(env_ids),), device=self.device
            )
            logger.warning(
                f"reset_from_teleop_curriculum: {self.sample_idx}, {self.sample_idx_in_seq}"
            )

        # logger.warning(f"task sample idx: {self.sample_idx}, {self.sample_idx_in_seq}")
        # 1 - front object
        (
            bottle_body_pos_table,
            bottle_body_rot_table,
            bottle_body_lin_vel_table,
            bottle_body_ang_vel_table,
        ) = self.reset_dataset.get_item_bottle_table_sample(self.sample_idx)
        bottle_body_pos_table = bottle_body_pos_table[
            torch.arange(len(env_ids)), self.sample_idx_in_seq
        ]
        bottle_body_rot_table = bottle_body_rot_table[
            torch.arange(len(env_ids)), self.sample_idx_in_seq
        ]
        bottle_body_lin_vel_table = bottle_body_lin_vel_table[
            torch.arange(len(env_ids)), self.sample_idx_in_seq
        ]
        bottle_body_ang_vel_table = bottle_body_ang_vel_table[
            torch.arange(len(env_ids)), self.sample_idx_in_seq
        ]

        bottle_root_states_table = torch.zeros((len(env_ids), 13), device=self.device)
        bottle_root_states_table[:, :3] = (
            bottle_body_pos_table[:, 0] + self.env_origins[env_ids, :3]
        )
        bottle_root_states_table[:, 3:7] = bottle_body_rot_table[:, 0]
        bottle_root_states_table[:, 7:10] = bottle_body_lin_vel_table[:, 0]
        bottle_root_states_table[:, 10:13] = bottle_body_ang_vel_table[:, 0]

        self.front_bottle_root_state_buf[env_ids, :] = bottle_root_states_table

        # 2 - front object
        (
            bottle_body_pos_hand,
            bottle_body_rot_hand,
            bottle_body_lin_vel_hand,
            bottle_body_ang_vel_hand,
        ) = self.reset_dataset.get_item_bottle_hand_sample(self.sample_idx)
        bottle_body_pos_hand = bottle_body_pos_hand[
            torch.arange(len(env_ids)), self.sample_idx_in_seq
        ]
        bottle_body_rot_hand = bottle_body_rot_hand[
            torch.arange(len(env_ids)), self.sample_idx_in_seq
        ]
        bottle_body_lin_vel_hand = bottle_body_lin_vel_hand[
            torch.arange(len(env_ids)), self.sample_idx_in_seq
        ]
        bottle_body_ang_vel_hand = bottle_body_ang_vel_hand[
            torch.arange(len(env_ids)), self.sample_idx_in_seq
        ]

        bottle_root_states_hand = torch.zeros((len(env_ids), 13), device=self.device)
        bottle_root_states_hand[:, :3] = bottle_body_pos_hand[:, 0] + self.env_origins[env_ids, :3]
        bottle_root_states_hand[:, 3:7] = bottle_body_rot_hand[:, 0]
        bottle_root_states_hand[:, 7:10] = bottle_body_lin_vel_hand[:, 0]
        bottle_root_states_hand[:, 10:13] = bottle_body_ang_vel_hand[:, 0]

        self.hold_bottle_root_state_buf[env_ids, :] = bottle_root_states_hand

        # 3 - tray object
        (
            tray_body_pos_table,
            tray_body_rot_table,
            tray_body_lin_vel_table,
            tray_body_ang_vel_table,
        ) = self.reset_dataset.get_item_tray_table_sample(self.sample_idx)
        tray_body_pos_table = tray_body_pos_table[
            torch.arange(len(env_ids)), self.sample_idx_in_seq
        ]
        tray_body_rot_table = tray_body_rot_table[
            torch.arange(len(env_ids)), self.sample_idx_in_seq
        ]
        tray_body_lin_vel_table = tray_body_lin_vel_table[
            torch.arange(len(env_ids)), self.sample_idx_in_seq
        ]
        tray_body_ang_vel_table = tray_body_ang_vel_table[
            torch.arange(len(env_ids)), self.sample_idx_in_seq
        ]

        tray_root_states_table = torch.zeros((len(env_ids), 13), device=self.device)
        tray_root_states_table[:, :3] = tray_body_pos_table[:, 0] + self.env_origins[env_ids, :3]
        tray_root_states_table[:, 3:7] = tray_body_rot_table[:, 0]
        tray_root_states_table[:, 7:10] = tray_body_lin_vel_table[:, 0]
        tray_root_states_table[:, 10:13] = tray_body_ang_vel_table[:, 0]
        self.tray_root_state_buf[env_ids, :] = tray_root_states_table

        # 3 - back object
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

        task_root_state_dict = {
            "front_object": self.front_bottle_root_state_buf,
            "hold_object": self.hold_bottle_root_state_buf,
            "back_object": self.back_bottle_root_state_buf,
            "tray_object": self.tray_root_state_buf,
        }
        # logger.warning(f"task root state dict: {task_root_state_dict}")
        self.simulator.set_task_root_state_tensor(env_ids, task_root_state_dict)

        # logger.warning(f"reset_bottle_root_state_from_dataset: {bottle_root_states}")

        # reuse self.sample_idx and self.sample_idx_in_seq
