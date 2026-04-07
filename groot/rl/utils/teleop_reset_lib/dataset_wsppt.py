# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Optional

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import Dataset

from groot.rl.utils.teleop_reset_lib.motion_loader_wsppt import MotionLoaderWSPPT


class MotionDatasetWSPPT(Dataset):
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.motion_files = config.motion_files
        self.device = config.device
        # make sure motion_
        if self.config.use_motion_file_dir:
            import os

            motion_dir = self.config.motion_file_dir
            all_motion_files_under_dir = [
                os.path.join(motion_dir, f) for f in os.listdir(motion_dir) if f.endswith(".npz")
            ]
            logger.info(
                f"number of motion files under dir {motion_dir}: {len(all_motion_files_under_dir)}"
            )
            logger.info(f"motion files: {all_motion_files_under_dir}")
            logger.info(f"overriding motion files with motion files under dir {motion_dir}")
            self.motion_files = all_motion_files_under_dir
        self.motion_loaders = [
            MotionLoaderWSPPT.from_file(motion_file, self.device)
            for motion_file in self.motion_files
        ]
        logger.info(f"number of motion loaders: {len(self.motion_loaders)}")
        assert len(self.motion_loaders) > 0, "No motion files provided"
        self.body_names = self.motion_loaders[0].body_names
        self.dof_names = self.motion_loaders[0].dof_names
        self.num_per_sample = config.num_per_sample
        self.sample_interval_s = config.sample_interval_s
        self.samples_robot: Optional[
            tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ]
        ] = None
        self.samples_table_grid: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self.samples_bottle: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self.samples_tray_table: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self.samples_homie_command: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        self.sample_idx_based_on_demo_clip_number = []
        self.resample()

    def resample(self) -> None:
        samples_robot = []
        samples_table_grid = []
        samples_bottle_table = []
        samples_bottle_hand = []
        samples_homie_command = []
        samples_tray_table = []

        self.sample_idx_based_on_demo_clip_number = []

        for i, motion_loader in enumerate(self.motion_loaders):
            duration_per_sample = self.num_per_sample * self.sample_interval_s
            if duration_per_sample * 2 > motion_loader.duration:
                logging.warning(
                    f"Motion loader {i} has duration {motion_loader.duration}s, which is less than the sample duration {duration_per_sample}s * 2"
                )
                continue

            num_samples = int((motion_loader.duration - duration_per_sample) / duration_per_sample)
            if i == 0:
                self.sample_idx_based_on_demo_clip_number.append([0, num_samples - 1])
            else:
                self.sample_idx_based_on_demo_clip_number.append(
                    [
                        self.sample_idx_based_on_demo_clip_number[-1][1] + 1,
                        self.sample_idx_based_on_demo_clip_number[-1][1] + num_samples,
                    ]
                )
            sample_times = (
                np.arange(0, num_samples * self.num_per_sample) * self.sample_interval_s
                + np.random.rand() * duration_per_sample
            )
            dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel = motion_loader.sample(
                0, sample_times
            )
            dof_pos = dof_pos.reshape(num_samples, self.num_per_sample, -1)
            dof_vel = dof_vel.reshape(num_samples, self.num_per_sample, -1)
            body_pos = body_pos.reshape(num_samples, self.num_per_sample, -1, 3)
            body_rot = body_rot.reshape(num_samples, self.num_per_sample, -1, 4)
            body_lin_vel = body_lin_vel.reshape(num_samples, self.num_per_sample, -1, 3)
            body_ang_vel = body_ang_vel.reshape(num_samples, self.num_per_sample, -1, 3)
            samples_robot.append((dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel))

            (
                table_grid_body_pos,
                table_grid_body_rot,
                table_grid_body_lin_vel,
                table_grid_body_ang_vel,
            ) = motion_loader.sample_table_grid()
            table_grid_body_pos = table_grid_body_pos.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            table_grid_body_rot = table_grid_body_rot.reshape(
                num_samples, self.num_per_sample, -1, 4
            )
            table_grid_body_lin_vel = table_grid_body_lin_vel.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            table_grid_body_ang_vel = table_grid_body_ang_vel.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            samples_table_grid.append(
                (
                    table_grid_body_pos,
                    table_grid_body_rot,
                    table_grid_body_lin_vel,
                    table_grid_body_ang_vel,
                )
            )

            (
                bottle_body_pos_table,
                bottle_body_rot_table,
                bottle_body_lin_vel_table,
                bottle_body_ang_vel_table,
            ) = motion_loader.sample_bottle_table()
            bottle_body_pos_table = bottle_body_pos_table.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            bottle_body_rot_table = bottle_body_rot_table.reshape(
                num_samples, self.num_per_sample, -1, 4
            )
            bottle_body_lin_vel_table = bottle_body_lin_vel_table.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            bottle_body_ang_vel_table = bottle_body_ang_vel_table.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            samples_bottle_table.append(
                (
                    bottle_body_pos_table,
                    bottle_body_rot_table,
                    bottle_body_lin_vel_table,
                    bottle_body_ang_vel_table,
                )
            )

            (
                bottle_body_pos_hand,
                bottle_body_rot_hand,
                bottle_body_lin_vel_hand,
                bottle_body_ang_vel_hand,
            ) = motion_loader.sample_bottle_hand()
            bottle_body_pos_hand = bottle_body_pos_hand.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            bottle_body_rot_hand = bottle_body_rot_hand.reshape(
                num_samples, self.num_per_sample, -1, 4
            )
            bottle_body_lin_vel_hand = bottle_body_lin_vel_hand.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            bottle_body_ang_vel_hand = bottle_body_ang_vel_hand.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            samples_bottle_hand.append(
                (
                    bottle_body_pos_hand,
                    bottle_body_rot_hand,
                    bottle_body_lin_vel_hand,
                    bottle_body_ang_vel_hand,
                )
            )

            (
                tray_body_pos_table,
                tray_body_rot_table,
                tray_body_lin_vel_table,
                tray_body_ang_vel_table,
            ) = motion_loader.sample_tray_table()
            tray_body_pos_table = tray_body_pos_table.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            tray_body_rot_table = tray_body_rot_table.reshape(
                num_samples, self.num_per_sample, -1, 4
            )
            tray_body_lin_vel_table = tray_body_lin_vel_table.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            tray_body_ang_vel_table = tray_body_ang_vel_table.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            samples_tray_table.append(
                (
                    tray_body_pos_table,
                    tray_body_rot_table,
                    tray_body_lin_vel_table,
                    tray_body_ang_vel_table,
                )
            )

            homie_commands, target_dof_positions, finger_primitive_actions = (
                motion_loader.sample_homie_command()
            )
            homie_commands = homie_commands.reshape(num_samples, self.num_per_sample, 7)
            target_dof_positions = target_dof_positions.reshape(
                num_samples, self.num_per_sample, -1
            )  # 43dof
            finger_primitive_actions = finger_primitive_actions.reshape(
                num_samples, self.num_per_sample, -1
            )  # 2dof
            samples_homie_command.append(
                (homie_commands, target_dof_positions, finger_primitive_actions)
            )

        self.samples_robot = tuple(
            torch.cat([sample[i] for sample in samples_robot], dim=0) for i in range(6)
        )
        self.samples_table_grid = tuple(
            torch.cat([sample[i] for sample in samples_table_grid], dim=0) for i in range(4)
        )
        self.samples_bottle_table = tuple(
            torch.cat([sample[i] for sample in samples_bottle_table], dim=0) for i in range(4)
        )
        self.samples_bottle_hand = tuple(
            torch.cat([sample[i] for sample in samples_bottle_hand], dim=0) for i in range(4)
        )
        self.samples_tray_table = tuple(
            torch.cat([sample[i] for sample in samples_tray_table], dim=0) for i in range(4)
        )
        self.samples_homie_command = tuple(
            torch.cat([sample[i] for sample in samples_homie_command], dim=0) for i in range(3)
        )

    def __len__(self) -> int:
        return self.samples_robot[0].shape[0]

    # def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     return tuple(sample[index] for sample in self.samples)

    def get_item_robot_sample(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(sample[index] for sample in self.samples_robot)

    def get_item_table_grid_sample(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(sample[index] for sample in self.samples_table_grid)

    def get_item_bottle_table_sample(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(sample[index] for sample in self.samples_bottle_table)

    def get_item_bottle_hand_sample(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(sample[index] for sample in self.samples_bottle_hand)

    def get_item_tray_table_sample(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(sample[index] for sample in self.samples_tray_table)

    def get_item_homie_command_sample(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(sample[index] for sample in self.samples_homie_command)


@hydra.main(config_path="../configs", config_name="train_dataset.yaml", version_base=None)
def test_motion_dataset(config: DictConfig) -> None:
    dataset = MotionDataset(config.dataset)
    print(dataset.samples[0].shape)


if __name__ == "__main__":
    test_motion_dataset()
