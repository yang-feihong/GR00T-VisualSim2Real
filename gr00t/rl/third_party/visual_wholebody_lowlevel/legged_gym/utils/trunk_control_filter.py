"""Measured trunk conditioning at the IK transform, not on EE commands/state.

Two 0.27 s poles match lab30 / IK Bridge. Clocks use simulation seconds.
"""

from __future__ import annotations

import math

import torch

from legged_gym.utils.isaaclab_math import (
    euler_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)

TRUNK_CONTROL_FILTER_ENABLED = False


class TrunkControlFilter:
    def __init__(self):
        self.stages = None
        self.valid = None
        self.timestamp = None

    def reset(self, env_ids=None):
        if env_ids is None:
            self.stages = self.valid = self.timestamp = None
        elif self.valid is not None:
            self.valid[env_ids] = False

    def update(self, timestamp, measured):
        timestamp = float(timestamp)
        dt = None if self.timestamp is None else timestamp - self.timestamp
        if self.stages is None or dt < 0 or dt > 0.5:
            self.stages = measured.unsqueeze(0).repeat(2, 1, 1).clone()
            self.valid = torch.ones(measured.shape[0], dtype=torch.bool, device=measured.device)
        else:
            self.stages[:, ~self.valid] = measured[~self.valid]
            self.valid.fill_(True)
            if dt > 0:
                alpha = -math.expm1(-dt / 0.27)
                value = measured
                for stage in self.stages:
                    stage.add_(alpha * (value - stage))
                    value = stage
        self.timestamp = timestamp
        return self.stages[-1]

    def control_target(
        self, timestamp, root_position, root_quat, ground_z, goal_position, goal_quat
    ):
        if not TRUNK_CONTROL_FILTER_ENABLED:
            return goal_position, goal_quat
        roll, pitch, yaw = euler_from_quat(root_quat)
        measured = torch.stack((root_position[:, 2] - ground_z, roll, pitch), dim=-1)
        selected = self.update(timestamp, measured)
        filtered_position = root_position.clone()
        filtered_position[:, 2] = ground_z + selected[:, 0]
        filtered_quat = quat_from_euler_xyz(selected[:, 1], selected[:, 2], yaw)
        # Equivalent arm-local goal: inverse(T_filtered) * raw_goal.
        correction = quat_mul(root_quat, quat_conjugate(filtered_quat))
        return (
            root_position + quat_apply(correction, goal_position - filtered_position),
            quat_mul(correction, goal_quat),
        )
