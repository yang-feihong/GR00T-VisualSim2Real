"""Arrival configuration, episode initialization and passive audit helpers."""

import math
from dataclasses import dataclass

import numpy as np
import torch

from gr00t.rl.auto_open.geometry import _quat_rotate_np
from gr00t.rl.auto_open.utils import as_numpy as _as_np

ROBOT_TRUNK_REAR_X = -0.284
THROUGH_TAIL_TARGET_DISTANCE = 1.0
THROUGH_TARGET_LEAD_DISTANCE = 0.05
THROUGH_TAIL_RELEASE_DISTANCE = 0.95


@dataclass(frozen=True)
class BaseArrivalProfile:
    """Stage-owned thresholds for the shared hysteretic arrival controller."""

    inner_distance_m: float
    outer_distance_m: float
    inner_yaw_rad: float
    outer_yaw_rad: float
    settle_frames: int
    max_linear_speed_m_s: float
    max_angular_speed_rad_s: float

    def __post_init__(self):
        if self.inner_distance_m > self.outer_distance_m:
            raise ValueError("inner distance must not exceed outer distance")
        if self.inner_yaw_rad > self.outer_yaw_rad:
            raise ValueError("inner yaw must not exceed outer yaw")
        if self.settle_frames <= 0:
            raise ValueError("settle_frames must be positive")
        if self.max_linear_speed_m_s <= 0.0:
            raise ValueError("maximum linear speed must be positive")
        if self.max_angular_speed_rad_s <= 0.0:
            raise ValueError("maximum angular speed must be positive")


# These values are deliberately centralized and stage-specific.  The controller
# architecture does not depend on them, so they can be tuned without changing
# stage logic.
APPROACH_ARRIVAL = BaseArrivalProfile(0.02, 0.10, 0.05, 0.15, 25, 0.02, 0.05)
ALIGN_PANEL_ARRIVAL = BaseArrivalProfile(0.02, 0.10, 0.05, 0.15, 25, 0.02, 0.05)

# Distance used by traversal progress/stall diagnostics, not arrival tolerance.
BASE_STATION_TOLERANCE_M = 0.12


def update_base_target_arrival(
    confirmed,
    settle_steps,
    inside_inner,
    inside_outer,
    *,
    required_settle_frames,
):
    """Advance a zero-command arrival window with inner/outer hysteresis.

    Entering the inner region starts dispatching exact zero commands.  During
    that braking window the state may drift into the outer region without
    restarting motion.  Leaving the outer region cancels the window.
    """
    device = confirmed.device
    was_confirmed = torch.as_tensor(confirmed, device=device, dtype=torch.bool)
    steps = torch.as_tensor(settle_steps, device=device)
    required = torch.as_tensor(required_settle_frames, device=device, dtype=steps.dtype)
    inner = torch.as_tensor(inside_inner, device=device, dtype=torch.bool)
    outer = torch.as_tensor(inside_outer, device=device, dtype=torch.bool)
    settling = steps > 0
    zero_command = torch.where(was_confirmed | settling, outer, inner)
    next_steps = torch.where(
        zero_command,
        torch.minimum(
            steps + 1,
            torch.ones_like(steps) * required,
        ),
        torch.zeros_like(steps),
    )
    next_confirmed = (was_confirmed & outer) | (~was_confirmed & (next_steps >= required))
    return next_confirmed, next_steps, zero_command


def _perturb_base_target(self, label, target_xy, target_yaw):
    config = self.target_noise.config
    if label == "approach":
        bounds = (
            config.approach_normal_m,
            config.approach_tangent_m,
            config.approach_yaw_rad,
        )
    elif label == "align_panel":
        bounds = (
            config.align_panel_normal_m,
            config.align_panel_tangent_m,
            config.align_panel_yaw_rad,
        )
    elif label == "through":
        bounds = (
            config.through_forward_m,
            config.through_tangent_m,
            config.through_yaw_rad,
        )
    else:
        raise ValueError(f"unknown base target-noise semantic {label!r}")
    normal_xy = self.door_forward[:2] if label == "through" else self.door_normal[:2]
    return self.target_noise.perturb_base(
        label,
        target_xy,
        target_yaw,
        normal_xy,
        normal_m=bounds[0],
        tangent_m=bounds[1],
        yaw_rad=bounds[2],
    )


def _approach_base_target(self):
    """Return the episode initialization target for APPROACH_HANDLE."""
    target_yaw = math.atan2(float(-self.door_normal[1]), float(-self.door_normal[0]))
    return self.station_xy, target_yaw


def _quat_distance(q0, q1):
    q0 = np.asarray(q0, dtype=np.float32)
    q1 = np.asarray(q1, dtype=np.float32)
    q0 /= max(float(np.linalg.norm(q0)), 1e-8)
    q1 /= max(float(np.linalg.norm(q1)), 1e-8)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0)))


def _through_tail_clearance(self, state):
    """Signed distance from the trunk rear edge to the door-frame plane."""
    base = _as_np(state["root_states"])[0]
    rear_offset_world = _quat_rotate_np(
        base[3:7],
        np.array([ROBOT_TRUNK_REAR_X, 0.0, 0.0], dtype=np.float32),
    )
    tail_xy = base[:2] + rear_offset_world[:2]
    return float(
        np.dot(
            tail_xy - self.initial_door_root[:2],
            self.door_forward[:2],
        )
    )


def _effective_grasp_target(self, grasp_center, ee_quat=None):
    target = np.asarray(grasp_center, dtype=np.float32).copy()
    if ee_quat is None:
        offset_world = getattr(self, "grasp_ee_offset_world", np.zeros(3, dtype=np.float32))
    else:
        offset_world = _quat_rotate_np(ee_quat, self.grasp_ee_offset_local)
    target -= offset_world
    return target
