"""GPU-resident batched state machine for AutoOpen.

This module owns all mutable runtime state and never calls
``DoorController.command`` or ``DoorController.advance``.  Per-environment
controller objects provide immutable task configuration and sparse planning
context during initialization and stage-boundary requests only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


MAX_BASE_PLANAR_SPEED = 0.4
MAX_BASE_YAW_RATE = 0.6
BASE_REVERSAL_JUMP = 0.15
BASE_REVERSAL_CONFIRM_FRAMES = 3
BASE_REVERSAL_LOCK_FRAMES = 5
BASE_REVERSAL_EPSILON = 1.0e-4
BASE_STATION_TOLERANCE_M = 0.12
BASE_YAW_TOLERANCE_RAD = 0.08
THROUGH_STOP_DISTANCE_M = 0.08
THROUGH_STOP_YAW_RAD = 0.05
THROUGH_ENTRY_STABILITY_FRAMES = 10
THROUGH_ENTRY_UPRIGHT_COS = math.cos(math.radians(10.0))
TERMINAL_BASE_LINEAR_SPEED_M_S = 0.05
TERMINAL_BASE_ANGULAR_SPEED_RAD_S = 0.05
TERMINAL_ARM_JOINT_SPEED_RAD_S = 0.08
TERMINAL_GRIPPER_SPEED_RAD_S = 0.05
TERMINAL_DOOR_SPEED_RAD_S = math.radians(2.0)
TERMINAL_STABILITY_FRAMES = 50
ROBOT_TRUNK_REAR_X = -0.284


def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )


def _wrapped_angle(target: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    return torch.remainder(target - actual + math.pi, 2.0 * math.pi) - math.pi


def _quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vector = quaternion[:, 1:]
    first_cross = torch.cross(-q_vector, vector, dim=-1)
    return vector + 2.0 * (
        quaternion[:, :1] * first_cross
        + torch.cross(-q_vector, first_cross, dim=-1)
    )


def _quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vector = quaternion[:, 1:]
    first_cross = torch.cross(q_vector, vector, dim=-1)
    return vector + 2.0 * (
        quaternion[:, :1] * first_cross
        + torch.cross(q_vector, first_cross, dim=-1)
    )


def _quat_distance(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    actual = actual / torch.linalg.vector_norm(actual, dim=-1, keepdim=True).clamp_min(1e-8)
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1e-8)
    dot = torch.sum(actual * target, dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _axis_angle_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
    half = 0.5 * angle
    return torch.cat((torch.cos(half)[..., None], axis * torch.sin(half)[..., None]), dim=-1)


def _yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(angle)
    return torch.stack((torch.cos(0.5 * angle), zeros, zeros, torch.sin(0.5 * angle)), dim=-1)


def _euler_xyz_quat(rpy: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = rpy.unbind(dim=-1)
    cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def _rotate_about_z(
    point: torch.Tensor, center: torch.Tensor, angle: torch.Tensor
) -> torch.Tensor:
    delta = point - center
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    result = point.clone()
    result[..., 0] = center[..., 0] + cosine * delta[..., 0] - sine * delta[..., 1]
    result[..., 1] = center[..., 1] + sine * delta[..., 0] + cosine * delta[..., 1]
    return result


def _slew_position(
    current: torch.Tensor, target: torch.Tensor, max_step: float
) -> torch.Tensor:
    delta = target - current
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_step) / distance.clamp_min(1e-8), max=1.0)
    return current + delta * scale


def _slew_quaternion(
    current: torch.Tensor, target: torch.Tensor, max_angle: float = 0.025
) -> torch.Tensor:
    current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp_min(1e-8)
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1e-8)
    dot = torch.sum(current * target, dim=-1, keepdim=True)
    target = torch.where(dot < 0.0, -target, target)
    dot = dot.abs().clamp(0.0, 1.0)
    angle = 2.0 * torch.acos(dot)
    fraction = torch.clamp(float(max_angle) / angle.clamp_min(1e-8), max=1.0)
    theta = torch.acos(dot.clamp(-1.0, 1.0))
    denominator = torch.sin(theta)
    slerp = (
        torch.sin((1.0 - fraction) * theta) * current
        + torch.sin(fraction * theta) * target
    ) / denominator.clamp_min(1e-8)
    output = torch.where((theta < 1e-6) | (angle <= max_angle), target, slerp)
    return output / torch.linalg.vector_norm(output, dim=-1, keepdim=True).clamp_min(1e-8)


def _orientation_delta(
    cart_local: torch.Tensor,
    desired_world: torch.Tensor,
    reference_world: torch.Tensor,
    arm_induced_pitch: torch.Tensor,
) -> torch.Tensor:
    local = _quat_multiply(_quat_conjugate(reference_world), desired_world)
    w, x, y, z = local.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    radius = torch.linalg.vector_norm(cart_local, dim=-1).clamp_min(1.0e-8)
    sphere_pitch = torch.asin((cart_local[:, 2] / radius).clamp(-1.0, 1.0))
    sphere_yaw = torch.atan2(cart_local[:, 1], cart_local[:, 0])
    return torch.stack(
        (
            _wrapped_angle(roll + math.pi / 2.0, torch.zeros_like(roll)),
            _wrapped_angle(
                pitch + sphere_pitch - arm_induced_pitch,
                torch.zeros_like(pitch),
            ),
            _wrapped_angle(yaw - sphere_yaw, torch.zeros_like(yaw)),
        ),
        dim=-1,
    )


def _limit_base_command(command: torch.Tensor) -> torch.Tensor:
    planar = command[:, :2]
    norm = torch.linalg.vector_norm(planar, dim=-1, keepdim=True)
    scale = torch.clamp(MAX_BASE_PLANAR_SPEED / norm.clamp_min(1.0e-8), max=1.0)
    result = command.clone()
    result[:, :2] = planar * scale
    result[:, 2] = result[:, 2].clamp(-MAX_BASE_YAW_RATE, MAX_BASE_YAW_RATE)
    return result


def track_base_pose(
    root_states: torch.Tensor,
    command_xy: torch.Tensor,
    command_yaw: torch.Tensor,
    acceptance_xy: torch.Tensor,
    acceptance_yaw: torch.Tensor,
    *,
    minimum_speed_toward_target_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched equivalent of the legacy base-pose controller."""
    yaw = _yaw_from_wxyz(root_states[:, 3:7])
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    error = command_xy - root_states[:, :2]
    local_x = cosine * error[:, 0] + sine * error[:, 1]
    local_y = -sine * error[:, 0] + cosine * error[:, 1]
    local_vx = cosine * root_states[:, 7] + sine * root_states[:, 8]
    local_vy = -sine * root_states[:, 7] + cosine * root_states[:, 8]
    yaw_error = _wrapped_angle(command_yaw, yaw)

    command = torch.stack(
        (
            1.2 * local_x - local_vx,
            1.2 * local_y - local_vy,
            1.5 * yaw_error - 0.35 * root_states[:, 12],
        ),
        dim=-1,
    )
    command = _limit_base_command(command)
    command_distance = torch.linalg.vector_norm(error, dim=-1)
    planar_norm = torch.linalg.vector_norm(command[:, :2], dim=-1)
    deadzone = (
        (command_distance > 0.08)
        & (planar_norm > 1.0e-6)
        & (planar_norm < 0.15)
    )
    if minimum_speed_toward_target_only:
        deadzone &= (command[:, 0] * local_x + command[:, 1] * local_y) > 0.0
    scale = torch.where(deadzone, 0.15 / planar_norm.clamp_min(1.0e-8), 1.0)
    command[:, :2] *= scale[:, None]
    yaw_deadzone = (
        (yaw_error.abs() > BASE_YAW_TOLERANCE_RAD)
        & (command[:, 2].abs() > 1.0e-6)
        & (command[:, 2].abs() < 0.2)
    )
    if minimum_speed_toward_target_only:
        yaw_deadzone &= command[:, 2] * yaw_error > 0.0
    command[:, 2] = torch.where(
        yaw_deadzone,
        torch.copysign(torch.full_like(command[:, 2], 0.2), command[:, 2]),
        command[:, 2],
    )

    acceptance_distance = torch.linalg.vector_norm(
        acceptance_xy - root_states[:, :2], dim=-1
    )
    acceptance_yaw_error = _wrapped_angle(acceptance_yaw, yaw).abs()
    return command, acceptance_distance, acceptance_yaw_error


@dataclass
class TensorCommandBuffers:
    base: torch.Tensor
    ee_cart: torch.Tensor
    ee_rpy: torch.Tensor
    gripper: torch.Tensor
    ee_quat: torch.Tensor
    ee_quat_enabled: torch.Tensor
    arm_joint: torch.Tensor
    arm_joint_mode: torch.Tensor

    @classmethod
    def from_env(cls, env) -> "TensorCommandBuffers":
        return cls(
            base=env.teleop_raw_commands.clone(),
            ee_cart=env.teleop_raw_ee_goal_cart.clone(),
            ee_rpy=env.teleop_raw_ee_goal_orn_delta_rpy.clone(),
            gripper=env.gripper_pos_targets.clone(),
            ee_quat=env.teleop_world_ee_goal_orn_quat.clone(),
            ee_quat_enabled=env.teleop_world_ee_goal_orn_quat_enabled.clone(),
            arm_joint=env.teleop_arm_joint_pos_targets.clone(),
            arm_joint_mode=env.teleop_arm_joint_control_mask.clone(),
        )

    def flush(self, env) -> None:
        env.teleop_raw_commands.copy_(self.base)
        env.teleop_raw_ee_goal_cart.copy_(self.ee_cart)
        env.teleop_raw_ee_goal_orn_delta_rpy.copy_(self.ee_rpy)
        env.gripper_pos_targets.copy_(self.gripper)
        env.teleop_world_ee_goal_orn_quat.copy_(self.ee_quat)
        env.teleop_world_ee_goal_orn_quat_enabled.copy_(self.ee_quat_enabled)
        env.teleop_arm_joint_pos_targets.copy_(self.arm_joint)
        env.teleop_arm_joint_control_mask.copy_(self.arm_joint_mode)
        env._teleop_inputs_dirty = True


class PaddedPathBank:
    """Fixed-capacity device paths with independent per-env length/index."""

    def __init__(self, num_envs: int, capacity: int, device):
        self.capacity = int(capacity)
        self.positions = torch.zeros(
            (num_envs, capacity, 3), dtype=torch.float32, device=device
        )
        self.orientations = torch.zeros(
            (num_envs, capacity, 4), dtype=torch.float32, device=device
        )
        self.length = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.index = torch.zeros(num_envs, dtype=torch.int32, device=device)

    def install(self, env_ids, positions, orientations) -> None:
        """Install sparse planner results; never used from the per-frame path."""
        if len(env_ids) != len(positions) or len(env_ids) != len(orientations):
            raise ValueError("path result batch lengths disagree")
        for env_id, position_path, orientation_path in zip(
            env_ids, positions, orientations, strict=True
        ):
            position = torch.as_tensor(
                position_path, dtype=torch.float32, device=self.positions.device
            )
            orientation = torch.as_tensor(
                orientation_path,
                dtype=torch.float32,
                device=self.orientations.device,
            )
            if position.ndim != 2 or position.shape[1] != 3:
                raise ValueError(f"invalid position path shape {tuple(position.shape)}")
            if orientation.shape != (position.shape[0], 4):
                raise ValueError(
                    "orientation path shape does not match position path: "
                    f"{tuple(orientation.shape)} vs {tuple(position.shape)}"
                )
            count = int(position.shape[0])
            if count < 1 or count > self.capacity:
                raise ValueError(
                    f"path length {count} is outside [1, {self.capacity}]"
                )
            row = int(env_id)
            self.positions[row, :count] = position
            self.orientations[row, :count] = orientation
            self.length[row] = count
            self.index[row] = 0

    def clear(self, rows: torch.Tensor) -> None:
        self.length[rows] = 0
        self.index[rows] = 0

    def current(self, rows: torch.Tensor, *, advance: bool) -> tuple[torch.Tensor, torch.Tensor]:
        length = self.length[rows]
        if torch.any(length <= 0):
            raise RuntimeError("stage entered without an installed path")
        index = torch.minimum(self.index[rows], length - 1).to(torch.int64)
        position = self.positions[rows, index]
        orientation = self.orientations[rows, index]
        if advance:
            self.index[rows] = torch.minimum(index + 1, length - 1).to(torch.int32)
        return position, orientation

    def complete(self, rows: torch.Tensor) -> torch.Tensor:
        return (self.length[rows] > 0) & (self.index[rows] >= self.length[rows] - 1)


class TensorStabilityWindow:
    """Device-only equivalent of the legacy 15-frame EE stability deque."""

    def __init__(self, num_envs: int, device, window_size: int = 15):
        self.window_size = int(window_size)
        self.positions = torch.zeros(
            (num_envs, window_size, 3), dtype=torch.float32, device=device
        )
        self.orientations = torch.zeros(
            (num_envs, window_size, 4), dtype=torch.float32, device=device
        )
        self.count = torch.zeros(num_envs, dtype=torch.int32, device=device)

    def clear(self, rows: torch.Tensor) -> None:
        self.count[rows] = 0

    def update(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.positions[rows] = torch.roll(self.positions[rows], -1, dims=1)
        self.orientations[rows] = torch.roll(
            self.orientations[rows], -1, dims=1
        )
        self.positions[rows, -1] = state["ee_pos"][rows]
        self.orientations[rows, -1] = state["ee_orn"][rows]
        self.count[rows] = torch.clamp(self.count[rows] + 1, max=self.window_size)

        positions = self.positions[rows]
        position_span = torch.linalg.vector_norm(
            positions - positions.mean(dim=1, keepdim=True), dim=-1
        ).amax(dim=1)
        orientations = self.orientations[rows]
        orientations = orientations / torch.linalg.vector_norm(
            orientations, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        current = orientations[:, -1]
        dots = torch.abs(torch.sum(orientations * current[:, None, :], dim=-1))
        orientation_span = (2.0 * torch.acos(dots.clamp(0.0, 1.0))).amax(dim=1)
        stable = (
            (self.count[rows] >= self.window_size)
            & (position_span <= 0.001)
            & (orientation_span <= 0.003)
        )
        return stable, position_span, orientation_span


class TensorDoorFSM:
    """Batched AutoOpen controller whose mutable per-env state is on device."""

    def __init__(
        self,
        *,
        stage_enum,
        controllers,
        env,
        approach_command_xy=None,
        approach_command_yaw=None,
        approach_nominal_xy=None,
        approach_nominal_yaw=None,
    ):
        self.Stage = stage_enum
        self.env = env
        self.device = env.device
        self.num_envs = len(controllers)
        device = self.device

        def stack(name, *, dtype=torch.float32):
            return torch.as_tensor(
                [getattr(controller, name) for controller in controllers],
                dtype=dtype,
                device=device,
            )

        self.stage = torch.full(
            (self.num_envs,), int(stage_enum.APPROACH), dtype=torch.int64, device=device
        )
        self.stage_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self.approach_settle_steps = torch.zeros_like(self.stage_steps)
        self.approach_stop_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self.base_command_previous = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=device
        )
        self.base_reversal_pending_sign = torch.zeros(
            (self.num_envs, 3), dtype=torch.int8, device=device
        )
        self.base_reversal_pending_steps = torch.zeros(
            (self.num_envs, 3), dtype=torch.int32, device=device
        )
        self.base_reversal_lock_steps = torch.zeros_like(
            self.base_reversal_pending_steps
        )
        door_normal = stack("door_normal")
        default_xy = stack("station_xy")
        default_yaw = torch.atan2(-door_normal[:, 1], -door_normal[:, 0])

        def supplied(value, default):
            if value is None:
                return default.clone()
            return torch.as_tensor(value, dtype=torch.float32, device=device).clone()

        self.station_xy = supplied(approach_command_xy, default_xy)
        self.station_yaw = supplied(approach_command_yaw, default_yaw)
        self.nominal_station_xy = supplied(approach_nominal_xy, default_xy)
        self.nominal_station_yaw = supplied(approach_nominal_yaw, default_yaw)
        self.initial_arm_joint_target = stack("initial_arm_joint_target")
        self.default_ee_goal_local = stack("default_ee_goal_local")
        self.initial_ee_orientation = stack("initial_ee_orientation")
        self.q_closed = stack("q_closed")
        self.q_open = stack("q_open")
        self.arm_induced_pitch = torch.as_tensor(
            [controller.env.arm_induced_pitch for controller in controllers],
            dtype=torch.float32,
            device=device,
        )
        self.station_tolerance = stack("station_tolerance")
        self.pregrasp_position_tolerance = stack("pregrasp_position_tolerance")
        self.grasp_orientation_tolerance = stack("grasp_orientation_tolerance")
        self.arm_align_desired_target_world = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=device
        )
        self.arm_align_target_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self.pregrasp_settle_steps = torch.zeros_like(self.stage_steps)
        self.gripper_open_settle = torch.zeros_like(self.stage_steps)
        self.handle_align_settle_steps = torch.zeros_like(self.stage_steps)
        self.grasp_settle_steps = torch.zeros_like(self.stage_steps)
        self.grasp_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self.grasp_bilateral_contact_seen = torch.zeros_like(self.grasp_latched)
        self.grasp_vertical_balance_started = torch.zeros_like(self.grasp_latched)
        self.grasp_vertical_lift = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        self.grasp_vertical_lift_target = torch.zeros_like(self.grasp_vertical_lift)
        self.handle_contact_last_seen = torch.full(
            (self.num_envs, 2), -1000, dtype=torch.int32, device=device
        )
        self.handle_contact_last_points = torch.full(
            (self.num_envs, 2, 3), torch.nan, dtype=torch.float32, device=device
        )
        self.arm_forward_desired_target_world = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=device
        )
        self.arm_forward_command_world = torch.zeros_like(
            self.arm_forward_desired_target_world
        )
        self.arm_forward_command_quat = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=device
        )
        self.arm_forward_target_valid = torch.zeros_like(self.grasp_latched)
        self.handle_body_idx = int(controllers[0].handle_body_idx)
        self.handle_center_local = stack("handle_collider_center_local")
        self.handle_axis_local = stack("handle_axis_local")
        self.handle_half_length = stack("handle_half_length")
        self.handle_radius = 0.5 * (stack("handle_upper_z") - stack("handle_lower_z"))
        self.door_normal = stack("door_normal")
        self.depth_axis_local = torch.as_tensor(
            [controller.open_template["depth_axis"] for controller in controllers],
            dtype=torch.float32,
            device=device,
        )
        self.base_hold_target_xy = self.station_xy.clone()
        self.base_hold_target_yaw = self.station_yaw.clone()
        self.base_hold_nominal_xy = self.nominal_station_xy.clone()
        self.base_hold_nominal_yaw = self.nominal_station_yaw.clone()
        self.pending_station_distance = torch.full(
            (self.num_envs,), torch.inf, dtype=torch.float32, device=device
        )
        self.pending_yaw_error = self.pending_station_distance.clone()
        self.arm_align_path = PaddedPathBank(self.num_envs, 512, device)
        self.arm_forward_path = PaddedPathBank(self.num_envs, 512, device)
        self.release_open_path = PaddedPathBank(self.num_envs, 512, device)
        self.release_retreat_path = PaddedPathBank(self.num_envs, 512, device)
        self.release_restore_path = PaddedPathBank(self.num_envs, 512, device)
        self.turn_path = PaddedPathBank(self.num_envs, 512, device)
        self.arm_stability = TensorStabilityWindow(self.num_envs, device)
        self.hinge_idx = int(controllers[0].hinge_idx)
        self.handle_joint_idx = int(controllers[0].handle_joint_idx)
        self.latch_joint_idx = int(controllers[0].latch_joint_idx)
        self.panel_body_idx = int(controllers[0].panel_body_idx)
        self.handle_joint_axis_panel_local = stack("handle_joint_axis_panel_local")
        self.hinge_world = stack("hinge_world")
        self.hinge_axis_sign = stack("hinge_axis_sign")
        self.turn_gap_angle = stack("turn_gap_angle")
        self.turn_target_margin = torch.as_tensor(
            [
                controller.target_noise.positive_margin(
                    "turn_handle_overtravel_rad",
                    controller.target_noise.config.turn_handle_overtravel_rad,
                )
                for controller in controllers
            ],
            dtype=torch.float32,
            device=device,
        )
        self.turn_gap_margin = torch.as_tensor(
            [
                controller.target_noise.positive_margin(
                    "turn_gap_overtravel_rad",
                    controller.target_noise.config.turn_gap_overtravel_rad,
                )
                for controller in controllers
            ],
            dtype=torch.float32,
            device=device,
        )
        scalar_shape = (self.num_envs,)
        vector_shape = (self.num_envs, 3)
        quat_shape = (self.num_envs, 4)
        self.turn_start_handle_angle = torch.zeros(scalar_shape, device=device)
        self.turn_start_latch_position = torch.zeros(scalar_shape, device=device)
        self.turn_start_hinge_angle = torch.zeros(scalar_shape, device=device)
        self.turn_target_handle_angle = torch.zeros(scalar_shape, device=device)
        self.turn_target_hinge_angle = torch.zeros(scalar_shape, device=device)
        self.turn_handle_axis_world = torch.zeros(vector_shape, device=device)
        self.turn_axle_world = torch.zeros(vector_shape, device=device)
        self.turn_grasp_offset_world = torch.zeros(vector_shape, device=device)
        self.turn_grasp_orientation = torch.zeros(quat_shape, device=device)
        self.turn_endpoint_world = torch.zeros(vector_shape, device=device)
        self.turn_endpoint_quat = torch.zeros(quat_shape, device=device)
        self.turn_command_bias_world = torch.zeros(vector_shape, device=device)
        self.turn_gap_started = torch.zeros(scalar_shape, dtype=torch.bool, device=device)
        self.turn_release_slow_steps = torch.zeros_like(self.stage_steps)
        self.turn_release_slow_steps_required = torch.as_tensor(
            [controller.turn_release_slow_steps_required for controller in controllers],
            dtype=torch.int32,
            device=device,
        )
        self.turn_release_max_hinge_speed = stack("turn_release_max_hinge_speed")
        self.release_return_steps = torch.as_tensor(
            [controller.release_return_steps for controller in controllers],
            dtype=torch.int32,
            device=device,
        )
        self.release_start_world = torch.zeros(vector_shape, device=device)
        self.release_start_quat = torch.zeros(quat_shape, device=device)
        self.release_return_axle_world = torch.zeros(vector_shape, device=device)
        self.release_return_axis_world = torch.zeros(vector_shape, device=device)
        self.release_return_offset_world = torch.zeros(vector_shape, device=device)
        self.release_return_start_quat = torch.zeros(quat_shape, device=device)
        self.release_start_handle_angle = torch.zeros(scalar_shape, device=device)
        self.release_start_hinge_angle = torch.zeros(scalar_shape, device=device)
        self.release_vertical_delta = torch.zeros(scalar_shape, device=device)
        self.release_stable_steps = torch.zeros_like(self.stage_steps)
        self.release_retreat_target_world = torch.zeros(vector_shape, device=device)
        self.release_endpoint_world = torch.zeros(vector_shape, device=device)
        self.release_endpoint_quat = torch.zeros(quat_shape, device=device)
        self.align_panel_arm_joint_target = torch.zeros_like(
            self.initial_arm_joint_target
        )
        self.panel_align_ee_goal_local = torch.zeros(vector_shape, device=device)
        self.panel_align_ee_orientation_local = torch.zeros(quat_shape, device=device)
        self.panel_station_settle_steps = torch.zeros_like(self.stage_steps)
        self.panel_station_xy = stack("panel_station_xy")
        self.door_forward = stack("door_forward")
        align_targets = [
            controller.target_noise.perturb_base(
                "align_panel",
                controller.panel_station_xy,
                math.atan2(controller.door_forward[1], controller.door_forward[0]),
                controller.door_normal[:2],
                normal_m=controller.target_noise.config.align_panel_normal_m,
                tangent_m=controller.target_noise.config.align_panel_tangent_m,
                yaw_rad=controller.target_noise.config.align_panel_yaw_rad,
            )
            for controller in controllers
        ]
        self.panel_station_command_xy = torch.as_tensor(
            [target[0] for target in align_targets], dtype=torch.float32, device=device
        )
        self.panel_station_command_yaw = torch.as_tensor(
            [target[1] for target in align_targets], dtype=torch.float32, device=device
        )
        self.panel_station_nominal_yaw = torch.atan2(
            self.door_forward[:, 1], self.door_forward[:, 0]
        )
        self.panel_contact_target_world = torch.zeros(vector_shape, device=device)
        self.panel_contact_desired_ee_world = torch.zeros(vector_shape, device=device)
        self.panel_contact_command_target_world = torch.zeros(vector_shape, device=device)
        self.panel_contact_command_orientation_world = torch.zeros(quat_shape, device=device)
        self.panel_contact_orientation_seed = torch.zeros(quat_shape, device=device)
        self.panel_contact_orientation_seed_valid = torch.zeros_like(
            self.grasp_latched
        )
        self.panel_contact_plan_valid = torch.zeros_like(self.grasp_latched)
        self.panel_contact_settle_steps = torch.zeros_like(self.stage_steps)
        self.panel_last_finger_clearance = torch.full(
            scalar_shape, torch.nan, dtype=torch.float32, device=device
        )
        self.panel_last_lateral_error = torch.full_like(
            self.panel_last_finger_clearance, torch.nan
        )
        self.panel_last_axis_error = torch.full_like(
            self.panel_last_finger_clearance, torch.nan
        )
        self.panel_last_ee_speed = torch.full_like(
            self.panel_last_finger_clearance, torch.nan
        )
        self.panel_standoff_acceptance_clearance = stack(
            "panel_standoff_acceptance_clearance"
        )
        self.ee_command_world = torch.zeros(vector_shape, device=device)
        self.ee_orientation_command = torch.zeros(quat_shape, device=device)
        self.ee_command_valid = torch.zeros_like(self.grasp_latched)
        self.push_start_hinge_angle = torch.zeros(scalar_shape, device=device)
        self.push_last_progress_hinge = torch.zeros(scalar_shape, device=device)
        self.push_no_progress_steps = torch.zeros_like(self.stage_steps)
        self.push_contact_world = torch.zeros(vector_shape, device=device)
        self.push_orientation = torch.zeros(quat_shape, device=device)
        self.push_command_hinge_angle = torch.zeros(scalar_shape, device=device)
        self.push_speed_error_integral = torch.zeros(scalar_shape, device=device)
        self.push_overspeed_steps = torch.zeros_like(self.stage_steps)
        self.push_max_observed_angular_speed = torch.zeros(scalar_shape, device=device)
        self.push_comfort_reach_radius = torch.full(
            scalar_shape, 0.7, dtype=torch.float32, device=device
        )
        self.push_arm_detached = torch.zeros_like(self.grasp_latched)
        self.push_detached_ee_goal_local = torch.zeros(vector_shape, device=device)
        self.active_arm_retract_pose_local = stack("active_arm_retract_pose_local")
        self.push_recovery_settle_steps = torch.zeros_like(self.stage_steps)
        self.push_reengage_reachable = torch.zeros_like(self.grasp_latched)
        self.push_previous_finger_clearance = torch.full(
            scalar_shape, torch.nan, dtype=torch.float32, device=device
        )
        self.push_target_angular_speed = stack("push_target_angular_speed")
        self.push_max_angular_speed = stack("push_max_angular_speed")
        self.push_speed_kp = stack("push_speed_kp")
        self.push_speed_ki = stack("push_speed_ki")
        self.push_speed_integral_limit = stack("push_speed_integral_limit")
        self.push_max_angular_lead = stack("push_max_angular_lead")
        self.push_goal_hinge_angle = stack("push_goal_hinge_angle")
        self.push_reengage_reach_radius = stack("push_reengage_reach_radius")
        self.push_reengage_release_radius = stack("push_reengage_release_radius")
        self.push_reengage_target_lead = stack("push_reengage_target_lead")
        self.push_base_start_xy = torch.zeros((self.num_envs, 2), device=device)
        self.push_entry_ee_world = torch.zeros(vector_shape, device=device)
        self.push_entry_command_world = torch.zeros(vector_shape, device=device)
        self.base_advance_arm_joint_target = torch.zeros_like(
            self.initial_arm_joint_target
        )
        self.base_advance_target_longitudinal = torch.full(
            scalar_shape, -torch.inf, dtype=torch.float32, device=device
        )
        self.panel_center_world = stack("panel_center_world")
        self.panel_half_thickness = stack("panel_half_thickness")
        self.front_shoulder_body_indices = torch.as_tensor(
            controllers[0].front_shoulder_body_indices,
            dtype=torch.int64,
            device=device,
        )
        self.push_shoulder_collision_radius = stack(
            "push_shoulder_collision_radius"
        )
        self.push_shoulder_safety_margin = stack("push_shoulder_safety_margin")
        self.dt = float(env.dt)
        self.middle_arm_joint_target = stack("middle_arm_joint_target")
        self.through_station_xy = stack("through_station_xy")
        self.through_station_yaw = stack("through_station_yaw")
        through_targets = [
            controller.target_noise.perturb_base(
                "through",
                controller.through_station_xy,
                controller.through_station_yaw,
                controller.door_forward[:2],
                normal_m=controller.target_noise.config.through_forward_m,
                tangent_m=controller.target_noise.config.through_tangent_m,
                yaw_rad=controller.target_noise.config.through_yaw_rad,
            )
            for controller in controllers
        ]
        self.through_command_xy = torch.as_tensor(
            [target[0] for target in through_targets], dtype=torch.float32, device=device
        )
        self.through_command_yaw = torch.as_tensor(
            [target[1] for target in through_targets], dtype=torch.float32, device=device
        )
        self.through_best_station_error = torch.full(
            scalar_shape, torch.inf, dtype=torch.float32, device=device
        )
        self.through_no_progress_steps = torch.zeros_like(self.stage_steps)
        self.through_settle_steps = torch.zeros_like(self.stage_steps)
        self.through_stop_latched = torch.zeros_like(self.grasp_latched)
        self.through_motion_ready = torch.zeros_like(self.grasp_latched)
        self.through_entry_stable_steps = torch.zeros_like(self.stage_steps)
        self.through_min_base_height = stack("through_min_base_height")
        self.initial_door_root = stack("initial_door_root")
        self.failure_code = torch.zeros(
            self.num_envs, dtype=torch.int16, device=device
        )
        self.failure_from_stage = torch.zeros(
            self.num_envs, dtype=torch.int16, device=device
        )
        self.stage_limits = torch.zeros(19, dtype=torch.int32, device=device)
        for stage, limit in controllers[0].stage_limits.items():
            self.stage_limits[int(stage)] = int(limit)

    def mask(self, stage) -> torch.Tensor:
        return self.stage == int(stage)

    def transition(self, rows: torch.Tensor, stage) -> None:
        self.stage[rows] = int(stage)
        self.stage_steps[rows] = 0

    def fail(self, rows: torch.Tensor, code: int) -> None:
        if not len(rows):
            return
        self.failure_from_stage[rows] = self.stage[rows].to(torch.int16)
        self.failure_code[rows] = int(code)
        self.stage[rows] = int(self.Stage.FAILURE)

    def install_pregrasp_plan(
        self,
        env_ids,
        positions,
        orientations,
        desired_targets,
    ) -> None:
        self.arm_align_path.install(env_ids, positions, orientations)
        rows = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        self.arm_align_desired_target_world[rows] = torch.as_tensor(
            desired_targets, dtype=torch.float32, device=self.device
        )
        self.arm_align_target_valid[rows] = True
        self.arm_stability.clear(rows)
        self.transition(rows, self.Stage.PREGRASP)
        self.stage_steps[rows] = 1

    def install_center_plan(
        self,
        env_ids,
        positions,
        orientations,
        desired_targets,
    ) -> None:
        self.arm_forward_path.install(env_ids, positions, orientations)
        rows = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        self.arm_forward_desired_target_world[rows] = torch.as_tensor(
            desired_targets, dtype=torch.float32, device=self.device
        )
        lengths = self.arm_forward_path.length[rows].to(torch.int64)
        self.arm_forward_command_world[rows] = self.arm_forward_path.positions[
            rows, lengths - 1
        ]
        self.arm_forward_command_quat[rows] = self.arm_forward_path.orientations[
            rows, lengths - 1
        ]
        self.arm_forward_target_valid[rows] = True
        self.arm_stability.clear(rows)
        self.handle_align_settle_steps[rows] = 0
        self.transition(rows, self.Stage.CENTER_HANDLE)
        self.stage_steps[rows] = 1

    def _set_cartesian_targets(
        self,
        rows: torch.Tensor,
        target_world: torch.Tensor,
        orientation_world: torch.Tensor,
        buffers: TensorCommandBuffers,
    ) -> None:
        origin = self.env.get_goal_reference_origin()[rows]
        reference = self.env.get_goal_reference_quat()[rows]
        offset = self.env.get_goal_center_offset_local()[rows]
        local = _quat_rotate_inverse(reference, target_world - origin) - offset
        buffers.ee_cart[rows] = local
        buffers.ee_quat[rows] = orientation_world
        buffers.ee_quat_enabled[rows] = True
        buffers.ee_rpy[rows] = _orientation_delta(
            local,
            orientation_world,
            reference,
            self.arm_induced_pitch[rows],
        )
        buffers.arm_joint_mode[rows] = False

    def _base_hold_command(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        command, distance, yaw_error = track_base_pose(
            state["root_states"][rows],
            self.base_hold_target_xy[rows],
            self.base_hold_target_yaw[rows],
            self.base_hold_nominal_xy[rows],
            self.base_hold_nominal_yaw[rows],
            minimum_speed_toward_target_only=False,
        )
        accepted = (distance < BASE_STATION_TOLERANCE_M) & (
            yaw_error < BASE_YAW_TOLERANCE_RAD
        )
        command[accepted] = 0.0
        command = self._stabilize_base(rows, command)
        self.pending_station_distance[rows] = distance
        self.pending_yaw_error[rows] = yaw_error
        return command, distance, yaw_error

    def _stabilize_base(
        self, rows: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        previous = self.base_command_previous[rows]
        pending_sign = self.base_reversal_pending_sign[rows]
        pending_steps = self.base_reversal_pending_steps[rows]
        lock_steps = self.base_reversal_lock_steps[rows]
        locked = lock_steps > 0
        reversing = (
            (previous.abs() > BASE_REVERSAL_EPSILON)
            & (target.abs() > BASE_REVERSAL_EPSILON)
            & (previous * target < 0.0)
            & ((target - previous).abs() >= BASE_REVERSAL_JUMP)
            & ~locked
        )
        new_sign = torch.sign(target).to(torch.int8)
        same_pending = pending_sign == new_sign
        next_steps = torch.where(same_pending, pending_steps + 1, 1)
        pending_steps = torch.where(reversing, next_steps, 0)
        pending_sign = torch.where(reversing, new_sign, 0)
        confirmed = reversing & (pending_steps >= BASE_REVERSAL_CONFIRM_FRAMES)
        hold = locked | (reversing & ~confirmed)
        output = torch.where(hold, previous, target)
        lock_steps = torch.where(
            locked, lock_steps - 1, torch.where(confirmed, BASE_REVERSAL_LOCK_FRAMES, lock_steps)
        )
        pending_sign = torch.where(confirmed, 0, pending_sign)
        pending_steps = torch.where(confirmed, 0, pending_steps)
        self.base_command_previous[rows] = output
        self.base_reversal_pending_sign[rows] = pending_sign
        self.base_reversal_pending_steps[rows] = pending_steps
        self.base_reversal_lock_steps[rows] = lock_steps
        return output

    def command_approach(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        mask = self.mask(self.Stage.APPROACH)
        if not torch.any(mask):
            return
        rows = torch.where(mask)[0]
        roots = state["root_states"][rows]
        command, distance, yaw_error = track_base_pose(
            roots,
            self.station_xy[rows],
            self.station_yaw[rows],
            self.nominal_station_xy[rows],
            self.nominal_station_yaw[rows],
            minimum_speed_toward_target_only=True,
        )
        planar_speed = torch.linalg.vector_norm(roots[:, 7:9], dim=-1)
        angular_speed = torch.linalg.vector_norm(roots[:, 10:13], dim=-1)
        latched = self.approach_stop_latched[rows]
        inside_acceptance = (
            (distance < BASE_STATION_TOLERANCE_M)
            & (yaw_error < BASE_YAW_TOLERANCE_RAD)
        )
        latched = torch.where(
            latched,
            inside_acceptance,
            inside_acceptance & (planar_speed < 0.02) & (angular_speed < 0.05),
        )
        self.approach_stop_latched[rows] = latched
        command[latched] = 0.0

        previous = self.base_command_previous[rows]
        crossing = (previous * command < 0.0) & (
            (command - previous).abs() >= BASE_REVERSAL_JUMP
        )
        command = torch.where(crossing, torch.zeros_like(command), command)
        self.base_command_previous[rows] = command
        self.pending_station_distance[rows] = distance
        self.pending_yaw_error[rows] = yaw_error

        buffers.base[rows] = command
        buffers.arm_joint_mode[rows] = True
        buffers.arm_joint[rows] = self.initial_arm_joint_target[rows]
        buffers.ee_cart[rows] = self.default_ee_goal_local[rows]
        buffers.ee_rpy[rows] = 0.0
        buffers.ee_quat[rows] = self.initial_ee_orientation[rows]
        buffers.ee_quat_enabled[rows] = True
        buffers.gripper[rows, 0] = self.q_closed[rows]

    def advance_approach(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Advance APPROACH and return env ids requesting pregrasp planning."""
        mask = self.mask(self.Stage.APPROACH)
        if not torch.any(mask):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        rows = torch.where(mask)[0]
        roots = state["root_states"][rows]
        linear_speed = torch.linalg.vector_norm(roots[:, 7:10], dim=-1)
        angular_speed = torch.linalg.vector_norm(roots[:, 10:13], dim=-1)
        stable = (
            (self.pending_station_distance[rows] < BASE_STATION_TOLERANCE_M)
            & (self.pending_yaw_error[rows] < BASE_YAW_TOLERANCE_RAD)
            & (linear_speed < 0.02)
            & (angular_speed < 0.05)
        )
        settled = torch.where(
            stable, self.approach_settle_steps[rows] + 1, 0
        )
        self.approach_settle_steps[rows] = settled
        self.stage_steps[rows] += 1
        return rows[settled >= 25]

    def command_pregrasp_open(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        mask = self.mask(self.Stage.PREGRASP) | self.mask(self.Stage.OPEN_GRIPPER)
        if not torch.any(mask):
            return
        rows = torch.where(mask)[0]
        command, _, _ = self._base_hold_command(state, rows)
        buffers.base[rows] = command
        positions, orientations = self.arm_align_path.current(rows, advance=True)
        self._set_cartesian_targets(rows, positions, orientations, buffers)
        pregrasp = self.stage[rows] == int(self.Stage.PREGRASP)
        buffers.gripper[rows, 0] = torch.where(
            pregrasp, self.q_closed[rows], self.q_open[rows]
        )

    def advance_pregrasp_open(
        self, state: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Advance both stages and return rows requesting CENTER_HANDLE paths."""
        mask = self.mask(self.Stage.PREGRASP) | self.mask(self.Stage.OPEN_GRIPPER)
        if not torch.any(mask):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        rows = torch.where(mask)[0]
        stable, _, _ = self.arm_stability.update(state, rows)
        position_error = torch.linalg.vector_norm(
            state["ee_pos"][rows] - self.arm_align_desired_target_world[rows], dim=-1
        )
        orientation_error = _quat_distance(
            state["ee_orn"][rows],
            self.arm_align_path.orientations[
                rows, (self.arm_align_path.length[rows] - 1).to(torch.int64)
            ],
        )
        root_speed = torch.linalg.vector_norm(state["root_states"][rows, 7:10], dim=-1)
        contacts = state.get("handle_finger_contact_forces_w")
        no_contact = torch.ones(len(rows), dtype=torch.bool, device=self.device)
        if contacts is not None:
            no_contact = torch.all(
                torch.linalg.vector_norm(contacts[rows], dim=-1) < 1.0, dim=-1
            )
        aligned = (
            self.arm_align_path.complete(rows)
            & (position_error <= self.pregrasp_position_tolerance[rows])
            & (orientation_error <= self.grasp_orientation_tolerance[rows])
            & (self.pending_station_distance[rows] < self.station_tolerance[rows])
            & (self.pending_yaw_error[rows] < BASE_YAW_TOLERANCE_RAD)
            & (root_speed < 0.02)
            & no_contact
            & stable
        )

        pregrasp = self.stage[rows] == int(self.Stage.PREGRASP)
        pre_rows = rows[pregrasp]
        pre_settled = torch.where(
            aligned[pregrasp], self.pregrasp_settle_steps[pre_rows] + 1, 0
        )
        self.pregrasp_settle_steps[pre_rows] = pre_settled
        to_open = pre_rows[pre_settled >= 5]
        if len(to_open):
            self.transition(to_open, self.Stage.OPEN_GRIPPER)
            self.gripper_open_settle[to_open] = 0

        open_mask = self.stage[rows] == int(self.Stage.OPEN_GRIPPER)
        open_rows = rows[open_mask]
        if len(open_rows):
            gripper_joint_id = int(self.env.gripper_joint_ids[0].item())
            open_stable = (
                aligned[open_mask]
                & ((state["gripper_position"][open_rows, 0] - self.q_open[open_rows]).abs() < 0.02)
                & (state["dof_vel"][open_rows, gripper_joint_id].abs() < 0.05)
            )
            open_settled = torch.where(
                open_stable, self.gripper_open_settle[open_rows] + 1, 0
            )
            self.gripper_open_settle[open_rows] = open_settled
            requests = open_rows[open_settled >= 10]
        else:
            requests = torch.empty(0, dtype=torch.int64, device=self.device)

        active = self.mask(self.Stage.PREGRASP) | self.mask(self.Stage.OPEN_GRIPPER)
        self.stage_steps[active] += 1
        return requests

    def command_center_grasp(
        self,
        state: dict[str, torch.Tensor],
        geometry: dict[str, torch.Tensor],
        buffers: TensorCommandBuffers,
    ) -> None:
        mask = self.mask(self.Stage.CENTER_HANDLE) | self.mask(self.Stage.GRASP_HANDLE)
        if not torch.any(mask):
            return
        rows = torch.where(mask)[0]
        command, _, _ = self._base_hold_command(state, rows)
        buffers.base[rows] = command
        center = self.stage[rows] == int(self.Stage.CENTER_HANDLE)
        target = self.arm_forward_command_world[rows].clone()
        orientation = self.arm_forward_command_quat[rows]
        if torch.any(center):
            center_rows = rows[center]
            path_position, path_orientation = self.arm_forward_path.current(
                center_rows, advance=True
            )
            target[center] = path_position
            orientation = orientation.clone()
            orientation[center] = path_orientation
        grasp = ~center
        if torch.any(grasp):
            grasp_rows = rows[grasp]
            moving_clearance = geometry["moving_handle_clearance"][grasp]
            contact_forces = state.get("handle_finger_contact_forces_w")
            if contact_forces is None:
                contact_norms = torch.zeros(
                    (len(grasp_rows), 2), dtype=torch.float32, device=self.device
                )
            else:
                contact_norms = torch.linalg.vector_norm(
                    contact_forces[grasp_rows], dim=-1
                )
            fixed_contact = contact_norms[:, 0] >= 1.0
            moving_contact = contact_norms[:, 1] >= 1.0
            fixed_center = geometry["fixed_center"][grasp]
            moving_center = geometry["moving_center"][grasp]
            fixed_is_upper = fixed_center[:, 2] > moving_center[:, 2]
            balance_direction = torch.where(
                fixed_is_upper, -torch.ones_like(moving_clearance), torch.ones_like(moving_clearance)
            )
            start = (~self.grasp_vertical_balance_started[grasp_rows]) & (
                (moving_clearance <= 0.003) | moving_contact
            )
            upper_gap = geometry["upper_gap"][grasp]
            lower_gap = geometry["lower_gap"][grasp]
            fixed_gap = torch.where(fixed_is_upper, upper_gap, lower_gap).clamp_min(0.0)
            proposed = balance_direction * fixed_gap.clamp(0.0, 0.030)
            self.grasp_vertical_lift_target[grasp_rows] = torch.where(
                start, proposed, self.grasp_vertical_lift_target[grasp_rows]
            )
            self.grasp_vertical_balance_started[grasp_rows] |= start
            imbalance = moving_contact != fixed_contact
            adjusted = (
                self.grasp_vertical_lift_target[grasp_rows]
                + balance_direction * 0.00005
            ).clamp(-0.030, 0.030)
            self.grasp_vertical_lift_target[grasp_rows] = torch.where(
                imbalance, adjusted, self.grasp_vertical_lift_target[grasp_rows]
            )
            lift_delta = (
                self.grasp_vertical_lift_target[grasp_rows]
                - self.grasp_vertical_lift[grasp_rows]
            ).clamp(-0.0002, 0.0002)
            self.grasp_vertical_lift[grasp_rows] += lift_delta
            target[grasp, 2] += self.grasp_vertical_lift[grasp_rows]
        self._set_cartesian_targets(rows, target, orientation, buffers)
        buffers.gripper[rows, 0] = torch.where(
            center, self.q_open[rows], self.q_closed[rows]
        )

    def _handle_contacts(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forces = state.get("handle_finger_contact_forces_w")
        points = state.get("handle_finger_contact_points_w")
        if forces is None or points is None:
            shape = (len(rows), 2)
            return (
                torch.zeros(shape, dtype=torch.bool, device=self.device),
                torch.zeros(shape, dtype=torch.float32, device=self.device),
                torch.full((*shape, 3), torch.nan, dtype=torch.float32, device=self.device),
            )
        points = points[rows]
        norms = torch.linalg.vector_norm(forces[rows], dim=-1)
        handle_pose = state["door_body_state_w"][rows, self.handle_body_idx, :7]
        center = handle_pose[:, :3] + _quat_rotate(
            handle_pose[:, 3:7], self.handle_center_local[rows]
        )
        axis = _quat_rotate(handle_pose[:, 3:7], self.handle_axis_local[rows])
        delta = points - center[:, None, :]
        axial_raw = torch.sum(delta * axis[:, None, :], dim=-1)
        axial = torch.maximum(
            torch.minimum(axial_raw, self.handle_half_length[rows, None]),
            -self.handle_half_length[rows, None],
        )
        radial = torch.linalg.vector_norm(
            delta - axial[..., None] * axis[:, None, :], dim=-1
        )
        present = (
            torch.isfinite(points).all(dim=-1)
            & (radial <= 0.018)
            & (axial_raw.abs() <= self.handle_half_length[rows, None] + 0.008)
            & (norms >= 1.0)
        )
        return present, torch.where(present, norms, 0.0), points

    def _opposed_grasp(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        handle_pose = state["door_body_state_w"][rows, self.handle_body_idx, :7]
        center = handle_pose[:, :3] + _quat_rotate(
            handle_pose[:, 3:7], self.handle_center_local[rows]
        )
        axis = _quat_rotate(handle_pose[:, 3:7], self.handle_axis_local[rows])
        axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
        delta = points - center[:, None, :]
        axial = torch.sum(delta * axis[:, None, :], dim=-1)
        radial = delta - axial[..., None] * axis[:, None, :]
        closing_axis = torch.cross(-self.door_normal[rows], axis, dim=-1)
        closing_axis /= torch.linalg.vector_norm(
            closing_axis, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        closing = torch.sum(radial * closing_axis[:, None, :], dim=-1)
        return (
            torch.isfinite(points).all(dim=(1, 2))
            & torch.all(
                axial.abs()
                <= (self.handle_half_length[rows] - 0.005).clamp_min(0.0)[:, None],
                dim=-1,
            )
            & (closing[:, 0] * closing[:, 1] < 0.0)
            & torch.all(
                closing.abs() >= 0.25 * self.handle_radius[rows, None], dim=-1
            )
        )

    def advance_center_grasp(
        self,
        state: dict[str, torch.Tensor],
        geometry: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Advance CENTER/GRASP and return rows entering TURN_HANDLE."""
        mask = self.mask(self.Stage.CENTER_HANDLE) | self.mask(self.Stage.GRASP_HANDLE)
        if not torch.any(mask):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        rows = torch.where(mask)[0]
        stable, _, _ = self.arm_stability.update(state, rows)
        center = self.stage[rows] == int(self.Stage.CENTER_HANDLE)
        if torch.any(center):
            center_rows = rows[center]
            panel_collision = geometry["panel_clearance"][center] < 0.005
            self.fail(center_rows[panel_collision], 10)
            capture_ready = (
                (geometry["pocket_normal_error"][center].abs() <= 0.005)
                & (geometry["handle_axle_margin"][center] >= 0.020)
                & (geometry["handle_free_margin"][center] >= 0.010)
                & (geometry["channel_vertical_error"][center].abs() <= 0.010)
                & (geometry["grasp_depth"][center] >= 0.015)
                & (geometry["panel_clearance"][center] >= 0.005)
            )
            ready = (
                (torch.linalg.vector_norm(
                    state["ee_pos"][center_rows]
                    - self.arm_forward_desired_target_world[center_rows], dim=-1
                ) <= 0.012)
                & (_quat_distance(
                    state["ee_orn"][center_rows], self.arm_forward_command_quat[center_rows]
                ) <= self.grasp_orientation_tolerance[center_rows])
                & ((state["gripper_position"][center_rows, 0] - self.q_open[center_rows]).abs() < 0.02)
                & capture_ready
                & (torch.linalg.vector_norm(state["root_states"][center_rows, 7:10], dim=-1) < 0.02)
                & stable[center]
                & ~panel_collision
            )
            settled = torch.where(
                ready, self.handle_align_settle_steps[center_rows] + 1, 0
            )
            self.handle_align_settle_steps[center_rows] = settled
            to_grasp = center_rows[settled >= 5]
            if len(to_grasp):
                self.transition(to_grasp, self.Stage.GRASP_HANDLE)
                self.grasp_settle_steps[to_grasp] = 0

        grasp_rows = torch.where(self.mask(self.Stage.GRASP_HANDLE))[0]
        turn_rows = torch.empty(0, dtype=torch.int64, device=self.device)
        if len(grasp_rows):
            # Geometry rows correspond to the original combined stage rows.
            lookup = torch.searchsorted(rows, grasp_rows)
            panel_collision = geometry["panel_clearance"][lookup] < 0.005
            self.fail(grasp_rows[panel_collision], 11)
            present, norms, points = self._handle_contacts(state, grasp_rows)
            step = self.stage_steps[grasp_rows, None]
            self.handle_contact_last_seen[grasp_rows] = torch.where(
                present, step, self.handle_contact_last_seen[grasp_rows]
            )
            self.handle_contact_last_points[grasp_rows] = torch.where(
                present[..., None], points, self.handle_contact_last_points[grasp_rows]
            )
            recent = torch.all(
                step - self.handle_contact_last_seen[grasp_rows] <= 5, dim=-1
            )
            opposed = self._opposed_grasp(
                state, grasp_rows, self.handle_contact_last_points[grasp_rows]
            )
            newly_latched = (~self.grasp_latched[grasp_rows]) & recent & opposed
            self.grasp_latched[grasp_rows] |= newly_latched
            peak = norms.amax(dim=-1)
            joint_id = int(self.env.gripper_joint_ids[0].item())
            grasp_stable = (
                self.grasp_latched[grasp_rows]
                & recent
                & opposed
                & (state["dof_vel"][grasp_rows, joint_id].abs() < 0.08)
                & (peak <= 50.0)
                & ~panel_collision
            )
            settled = torch.where(
                grasp_stable, self.grasp_settle_steps[grasp_rows] + 1, 0
            )
            self.grasp_settle_steps[grasp_rows] = settled
            turn_rows = grasp_rows[settled >= 10]
            self.grasp_bilateral_contact_seen[turn_rows] = True

        active = self.mask(self.Stage.CENTER_HANDLE) | self.mask(self.Stage.GRASP_HANDLE)
        self.stage_steps[active] += 1
        return turn_rows

    def begin_turn(self, state: dict[str, torch.Tensor], rows: torch.Tensor) -> None:
        joints = state["door_joint_position"]
        self.turn_start_handle_angle[rows] = joints[rows, self.handle_joint_idx]
        self.turn_start_latch_position[rows] = joints[rows, self.latch_joint_idx]
        self.turn_start_hinge_angle[rows] = joints[rows, self.hinge_idx]
        self.turn_target_handle_angle[rows] = torch.maximum(
            0.74 + self.turn_target_margin[rows],
            self.turn_start_handle_angle[rows],
        )
        self.turn_target_hinge_angle[rows] = self.hinge_axis_sign[rows] * (
            self.turn_gap_angle[rows] + math.radians(2.0) + self.turn_gap_margin[rows]
        )
        handle_pose = state["door_body_state_w"][rows, self.handle_body_idx, :7]
        panel_pose = state["door_body_state_w"][rows, self.panel_body_idx, :7]
        axis = _quat_rotate(
            panel_pose[:, 3:7], self.handle_joint_axis_panel_local[rows]
        )
        axis /= torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
        self.turn_handle_axis_world[rows] = axis
        self.turn_axle_world[rows] = handle_pose[:, :3]
        self.turn_grasp_offset_world[rows] = state["ee_pos"][rows] - handle_pose[:, :3]
        self.turn_grasp_orientation[rows] = state["ee_orn"][rows]
        current_goal = state.get("ee_goal_cart_world")
        if current_goal is None:
            self.turn_command_bias_world[rows] = 0.0
        else:
            self.turn_command_bias_world[rows] = current_goal[rows] - state["ee_pos"][rows]
        handle_delta = (
            self.turn_target_handle_angle[rows] - self.turn_start_handle_angle[rows]
        )
        handle_rotation = _axis_angle_quat(axis, handle_delta)
        handle_endpoint = self.turn_axle_world[rows] + _quat_rotate(
            handle_rotation, self.turn_grasp_offset_world[rows]
        )
        self.turn_endpoint_world[rows] = _rotate_about_z(
            handle_endpoint, self.hinge_world[rows], self.turn_target_hinge_angle[rows]
        )
        self.turn_endpoint_quat[rows] = _quat_multiply(
            _yaw_quat(self.turn_target_hinge_angle[rows]),
            _quat_multiply(handle_rotation, self.turn_grasp_orientation[rows]),
        )
        self.turn_gap_started[rows] = False
        self.turn_release_slow_steps[rows] = 0
        self.transition(rows, self.Stage.TURN_HANDLE)
        self.stage_steps[rows] = 1
        self._rebuild_turn_path(state, rows)

    def _rebuild_turn_path(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor
    ) -> None:
        start = state["ee_pos"][rows]
        start_q = state["ee_orn"][rows]
        current_angle = state["door_joint_position"][rows, self.handle_joint_idx]
        remaining_angle = (
            self.turn_target_handle_angle[rows] - current_angle
        ).clamp_min(0.0)
        current_hinge = state["door_joint_position"][rows, self.hinge_idx]
        current_hinge_delta = self.hinge_axis_sign[rows] * (
            current_hinge - self.turn_start_hinge_angle[rows]
        )
        remaining_hinge = torch.where(
            self.turn_gap_started[rows],
            self.turn_target_hinge_angle[rows] - current_hinge_delta,
            torch.zeros_like(current_hinge_delta),
        )
        distance = torch.linalg.vector_norm(
            self.turn_endpoint_world[rows] - start, dim=-1
        )
        angle = _quat_distance(start_q, self.turn_endpoint_quat[rows])
        counts = torch.maximum(
            torch.full_like(self.stage_steps[rows], 120),
            torch.maximum(
                torch.ceil(distance / 0.004).to(torch.int32) + 1,
                torch.ceil(angle / 0.03).to(torch.int32) + 1,
            ),
        )
        max_count = int(counts.max().item())
        if max_count > self.turn_path.capacity:
            raise RuntimeError(
                f"turn path length {max_count} exceeds {self.turn_path.capacity}"
            )
        sample = torch.arange(max_count, device=self.device, dtype=torch.float32)
        fraction = sample[None, :] / (counts[:, None] - 1).clamp_min(1)
        fraction = fraction.clamp(max=1.0)
        nominal_current_rotation = _axis_angle_quat(
            self.turn_handle_axis_world[rows],
            current_angle - self.turn_start_handle_angle[rows],
        )
        nominal_handle_current = self.turn_axle_world[rows] + _quat_rotate(
            nominal_current_rotation, self.turn_grasp_offset_world[rows]
        )
        nominal_current = _rotate_about_z(
            nominal_handle_current,
            self.hinge_world[rows],
            current_hinge_delta,
        )
        tracking_offset = start - nominal_current
        planned_angle = (
            current_angle - self.turn_start_handle_angle[rows]
        )[:, None] + fraction * remaining_angle[:, None]
        planned_hinge = current_hinge_delta[:, None] + fraction * remaining_hinge[:, None]
        axis = self.turn_handle_axis_world[rows, None, :].expand(-1, max_count, -1)
        handle_rotation = _axis_angle_quat(axis, planned_angle)
        axle = self.turn_axle_world[rows, None, :]
        offset = self.turn_grasp_offset_world[rows, None, :].expand(-1, max_count, -1)
        nominal_handle = axle + _quat_rotate(
            handle_rotation.reshape(-1, 4), offset.reshape(-1, 3)
        ).reshape(len(rows), max_count, 3)
        nominal = _rotate_about_z(
            nominal_handle,
            self.hinge_world[rows, None, :],
            planned_hinge,
        )
        path = (
            nominal
            + self.turn_command_bias_world[rows, None, :]
            + (1.0 - fraction[..., None]) * tracking_offset[:, None, :]
        )
        orientation = _quat_multiply(
            _yaw_quat(planned_hinge),
            _quat_multiply(
                handle_rotation,
                self.turn_grasp_orientation[rows, None, :].expand(-1, max_count, -1),
            ),
        )
        self.turn_path.positions[rows, :max_count] = path
        self.turn_path.orientations[rows, :max_count] = orientation
        self.turn_path.length[rows] = counts
        self.turn_path.index[rows] = 1

    def command_turn(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.TURN_HANDLE))[0]
        if not len(rows):
            return
        latch = state["door_joint_position"][rows, self.latch_joint_idx]
        start_gap = (~self.turn_gap_started[rows]) & (latch >= 0.02)
        if torch.any(start_gap):
            gap_rows = rows[start_gap]
            self.turn_gap_started[gap_rows] = True
            self._rebuild_turn_path(state, gap_rows)
        command, _, _ = self._base_hold_command(state, rows)
        buffers.base[rows] = command
        position, orientation = self.turn_path.current(rows, advance=True)
        self._set_cartesian_targets(rows, position, orientation, buffers)
        buffers.gripper[rows, 0] = self.q_closed[rows]

    def advance_turn(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.TURN_HANDLE))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        joint = state["door_joint_position"]
        latch_released = joint[rows, self.latch_joint_idx] >= 0.02
        hinge = joint[rows, self.hinge_idx]
        gap_open = hinge.abs() >= self.turn_gap_angle[rows]
        hinge_speed = state["door_joint_velocity"][rows, self.hinge_idx].abs()
        slow = hinge_speed < self.turn_release_max_hinge_speed[rows]
        qualifying = latch_released & gap_open & slow
        settled = torch.where(
            qualifying, self.turn_release_slow_steps[rows] + 1, 0
        )
        self.turn_release_slow_steps[rows] = settled
        present, _, _ = self._handle_contacts(state, rows)
        retained = self.grasp_bilateral_contact_seen[rows] & torch.any(present, dim=-1)
        complete = (
            qualifying
            & (settled >= self.turn_release_slow_steps_required[rows])
            & retained
        )
        self.stage_steps[rows] += 1
        return rows[complete]

    def begin_return(self, state: dict[str, torch.Tensor], rows: torch.Tensor) -> None:
        handle_pose = state["door_body_state_w"][rows, self.handle_body_idx, :7]
        panel_pose = state["door_body_state_w"][rows, self.panel_body_idx, :7]
        axis = _quat_rotate(
            panel_pose[:, 3:7], self.handle_joint_axis_panel_local[rows]
        )
        axis /= torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
        self.release_start_world[rows] = state["ee_pos"][rows]
        self.release_start_quat[rows] = state["ee_orn"][rows]
        self.release_start_handle_angle[rows] = state["door_joint_position"][
            rows, self.handle_joint_idx
        ]
        self.release_start_hinge_angle[rows] = state["door_joint_position"][
            rows, self.hinge_idx
        ]
        self.release_return_axle_world[rows] = handle_pose[:, :3]
        self.release_return_axis_world[rows] = axis
        self.release_return_offset_world[rows] = (
            state["ee_pos"][rows] - handle_pose[:, :3]
        )
        self.release_return_start_quat[rows] = state["ee_orn"][rows]
        self.release_vertical_delta[rows] = -self.grasp_vertical_lift[rows]
        self.release_stable_steps[rows] = 0
        self.turn_path.clear(rows)
        self.transition(rows, self.Stage.RETURN_HANDLE)
        self.stage_steps[rows] = 1

    def command_return(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.RETURN_HANDLE))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows)
        buffers.base[rows] = command
        fraction = (
            self.stage_steps[rows].to(torch.float32)
            / self.release_return_steps[rows].to(torch.float32)
        ).clamp(0.0, 1.0)
        desired_angle = (1.0 - fraction) * self.release_start_handle_angle[rows]
        delta = desired_angle - self.release_start_handle_angle[rows]
        rotation = _axis_angle_quat(self.release_return_axis_world[rows], delta)
        target = self.release_return_axle_world[rows] + _quat_rotate(
            rotation, self.release_return_offset_world[rows]
        )
        orientation = _quat_multiply(
            rotation, self.release_return_start_quat[rows]
        )
        self._set_cartesian_targets(rows, target, orientation, buffers)
        buffers.gripper[rows, 0] = self.q_closed[rows]

    def advance_return(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.RETURN_HANDLE))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        horizontal = (
            state["door_joint_position"][rows, self.handle_joint_idx].abs() < 0.05
        )
        complete = (self.stage_steps[rows] >= self.release_return_steps[rows]) & horizontal
        settled = torch.where(
            complete, self.release_stable_steps[rows] + 1, 0
        )
        self.release_stable_steps[rows] = settled
        self.stage_steps[rows] += 1
        return rows[settled >= 10]

    def install_release_open_plan(self, env_ids, positions, orientations) -> None:
        self.release_open_path.install(env_ids, positions, orientations)
        rows = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        self.release_stable_steps[rows] = 0
        self.transition(rows, self.Stage.OPEN_AND_CLEAR)
        self.stage_steps[rows] = 1

    def command_open_and_clear(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.OPEN_AND_CLEAR))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows)
        buffers.base[rows] = command
        position, orientation = self.release_open_path.current(rows, advance=True)
        self._set_cartesian_targets(rows, position, orientation, buffers)
        fraction = (self.stage_steps[rows].to(torch.float32) / 35.0).clamp(0.0, 1.0)
        buffers.gripper[rows, 0] = (
            (1.0 - fraction) * self.q_closed[rows] + fraction * self.q_open[rows]
        )

    def advance_open_and_clear(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.OPEN_AND_CLEAR))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        present, _, _ = self._handle_contacts(state, rows)
        contact_clear = ~torch.any(present, dim=-1)
        gripper_open = (
            state["gripper_position"][rows, 0] - self.q_open[rows]
        ).abs() < 0.08
        settled = torch.where(
            contact_clear & gripper_open, self.release_stable_steps[rows] + 1, 0
        )
        self.release_stable_steps[rows] = settled
        self.stage_steps[rows] += 1
        return rows[settled >= 10]

    def install_retreat_plan(
        self, env_ids, positions, orientations, acceptance_targets
    ) -> None:
        self.release_retreat_path.install(env_ids, positions, orientations)
        rows = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        self.release_retreat_target_world[rows] = torch.as_tensor(
            acceptance_targets, dtype=torch.float32, device=self.device
        )
        self.release_stable_steps[rows] = 0
        self.transition(rows, self.Stage.RETREAT_ARM)
        self.stage_steps[rows] = 1

    def command_retreat(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.RETREAT_ARM))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows)
        buffers.base[rows] = command
        position, orientation = self.release_retreat_path.current(rows, advance=True)
        self._set_cartesian_targets(rows, position, orientation, buffers)
        buffers.gripper[rows, 0] = self.q_open[rows]

    def advance_retreat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.RETREAT_ARM))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        present, _, _ = self._handle_contacts(state, rows)
        complete = (
            self.release_retreat_path.complete(rows)
            & (torch.linalg.vector_norm(
                state["ee_pos"][rows] - self.release_retreat_target_world[rows], dim=-1
            ) <= 0.010)
            & ~torch.any(present, dim=-1)
        )
        settled = torch.where(
            complete, self.release_stable_steps[rows] + 1, 0
        )
        self.release_stable_steps[rows] = settled
        self.stage_steps[rows] += 1
        return rows[settled >= 10]

    def install_restore_plan(
        self,
        env_ids,
        positions,
        orientations,
        acceptance_positions,
        acceptance_orientations,
    ) -> None:
        self.release_restore_path.install(env_ids, positions, orientations)
        rows = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        self.release_endpoint_world[rows] = torch.as_tensor(
            acceptance_positions, dtype=torch.float32, device=self.device
        )
        self.release_endpoint_quat[rows] = torch.as_tensor(
            acceptance_orientations, dtype=torch.float32, device=self.device
        )
        self.release_stable_steps[rows] = 0
        self.arm_stability.clear(rows)
        self.transition(rows, self.Stage.RESTORE_GRIPPER)
        self.stage_steps[rows] = 1

    def command_restore(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.RESTORE_GRIPPER))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows)
        buffers.base[rows] = command
        position, orientation = self.release_restore_path.current(rows, advance=True)
        self._set_cartesian_targets(rows, position, orientation, buffers)
        fraction = (self.stage_steps[rows].to(torch.float32) / 50.0).clamp(0.0, 1.0)
        buffers.gripper[rows, 0] = (
            (1.0 - fraction) * self.q_open[rows] + fraction * self.q_closed[rows]
        )

    def advance_restore(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.RESTORE_GRIPPER))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        stable, _, _ = self.arm_stability.update(state, rows)
        restored = (
            (self.stage_steps[rows] >= 50)
            & ((state["gripper_position"][rows, 0] - self.q_closed[rows]).abs() < 0.03)
            & (_quat_distance(
                state["ee_orn"][rows], self.release_endpoint_quat[rows]
            ) < 0.05)
            & (torch.linalg.vector_norm(
                state["ee_pos"][rows] - self.release_endpoint_world[rows], dim=-1
            ) <= 0.010)
            & stable
        )
        settled = torch.where(
            restored, self.release_stable_steps[rows] + 1, 0
        )
        self.release_stable_steps[rows] = settled
        completed = rows[settled >= 10]
        if len(completed):
            arm_ids = self.env.arm_joint_ids.to(device=self.device, dtype=torch.int64)
            self.align_panel_arm_joint_target[completed] = state["dof_pos"][
                completed[:, None], arm_ids[None, :]
            ]
            self.panel_align_ee_goal_local[completed] = state["ee_goal_cart_local"][completed]
            reference = self.env.get_goal_reference_quat()[completed]
            self.panel_align_ee_orientation_local[completed] = _quat_multiply(
                _quat_conjugate(reference), state["ee_orn"][completed]
            )
            self.panel_station_settle_steps[completed] = 0
            self.transition(completed, self.Stage.ALIGN_PANEL)
            self.stage_steps[completed] = 1
        active = self.mask(self.Stage.RESTORE_GRIPPER)
        self.stage_steps[active] += 1
        return completed

    def command_align_panel(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.ALIGN_PANEL))[0]
        if not len(rows):
            return
        command, distance, yaw_error = track_base_pose(
            state["root_states"][rows],
            self.panel_station_command_xy[rows],
            self.panel_station_command_yaw[rows],
            self.panel_station_xy[rows],
            self.panel_station_nominal_yaw[rows],
            minimum_speed_toward_target_only=False,
        )
        command = self._stabilize_base(rows, command)
        self.pending_station_distance[rows] = distance
        self.pending_yaw_error[rows] = yaw_error
        self.base_hold_target_xy[rows] = self.panel_station_command_xy[rows]
        self.base_hold_target_yaw[rows] = self.panel_station_command_yaw[rows]
        self.base_hold_nominal_xy[rows] = self.panel_station_xy[rows]
        self.base_hold_nominal_yaw[rows] = self.panel_station_nominal_yaw[rows]
        buffers.base[rows] = command
        buffers.arm_joint_mode[rows] = True
        buffers.arm_joint[rows] = self.align_panel_arm_joint_target[rows]
        buffers.gripper[rows, 0] = self.q_closed[rows]
        self.ee_command_world[rows] = state["ee_pos"][rows]
        self.ee_orientation_command[rows] = state["ee_orn"][rows]
        self.ee_command_valid[rows] = True

    def advance_align_panel(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.ALIGN_PANEL))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        stable = (
            (self.pending_station_distance[rows] < 0.08)
            & (self.pending_yaw_error[rows] < BASE_YAW_TOLERANCE_RAD)
            & (torch.linalg.vector_norm(state["root_states"][rows, 7:10], dim=-1) < 0.08)
        )
        settled = torch.where(
            stable, self.panel_station_settle_steps[rows] + 1, 0
        )
        self.panel_station_settle_steps[rows] = settled
        completed = rows[settled >= 15]
        if len(completed):
            self.panel_contact_plan_valid[completed] = False
            self.panel_contact_settle_steps[completed] = 0
            self.transition(completed, self.Stage.CONTACT_PANEL)
            self.stage_steps[completed] = 1
        active = self.mask(self.Stage.ALIGN_PANEL)
        self.stage_steps[active] += 1
        return completed

    def contact_plan_requests(self) -> torch.Tensor:
        return torch.where(
            self.mask(self.Stage.CONTACT_PANEL) & ~self.panel_contact_plan_valid
        )[0]

    def install_contact_plan(
        self,
        env_ids,
        panel_targets,
        desired_ee_targets,
        command_targets,
        command_orientations,
    ) -> None:
        rows = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        self.panel_contact_target_world[rows] = torch.as_tensor(
            panel_targets, dtype=torch.float32, device=self.device
        )
        self.panel_contact_desired_ee_world[rows] = torch.as_tensor(
            desired_ee_targets, dtype=torch.float32, device=self.device
        )
        self.panel_contact_command_target_world[rows] = torch.as_tensor(
            command_targets, dtype=torch.float32, device=self.device
        )
        self.panel_contact_command_orientation_world[rows] = torch.as_tensor(
            command_orientations, dtype=torch.float32, device=self.device
        )
        initialize = ~self.ee_command_valid[rows]
        if torch.any(initialize):
            init_rows = rows[initialize]
            self.ee_command_world[init_rows] = self.panel_contact_command_target_world[
                init_rows
            ]
            self.ee_orientation_command[init_rows] = (
                self.panel_contact_command_orientation_world[init_rows]
            )
            self.ee_command_valid[init_rows] = True
        self.panel_contact_plan_valid[rows] = True

    def command_contact_panel(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(
            self.mask(self.Stage.CONTACT_PANEL) & self.panel_contact_plan_valid
        )[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows)
        buffers.base[rows] = command
        self.ee_command_world[rows] = _slew_position(
            self.ee_command_world[rows],
            self.panel_contact_command_target_world[rows],
            0.003,
        )
        self.ee_orientation_command[rows] = _slew_quaternion(
            self.ee_orientation_command[rows],
            self.panel_contact_command_orientation_world[rows],
            0.025,
        )
        self._set_cartesian_targets(
            rows,
            self.ee_command_world[rows],
            self.ee_orientation_command[rows],
            buffers,
        )
        buffers.gripper[rows, 0] = self.q_closed[rows]

    def advance_contact_panel(
        self,
        state: dict[str, torch.Tensor],
        finger_clearance: torch.Tensor,
    ) -> torch.Tensor:
        rows = torch.where(
            self.mask(self.Stage.CONTACT_PANEL) & self.panel_contact_plan_valid
        )[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        current_normal = _quat_rotate(
            _yaw_quat(self.hinge_axis_sign[rows] * hinge), self.door_normal[rows]
        )
        ee_error = state["ee_pos"][rows] - self.panel_contact_target_world[rows]
        signed = torch.sum(ee_error * current_normal, dim=-1)
        tangent_error = torch.linalg.vector_norm(
            ee_error - signed[:, None] * current_normal, dim=-1
        )
        insertion_axis = _quat_rotate(
            state["ee_orn"][rows], self.depth_axis_local[rows]
        )
        insertion_axis /= torch.linalg.vector_norm(
            insertion_axis, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        orientation_error = torch.acos(
            torch.sum(insertion_axis * -current_normal, dim=-1).clamp(-1.0, 1.0)
        )
        ee_speed = torch.linalg.vector_norm(state["ee_vel"][rows, :3], dim=-1)
        clearance = finger_clearance[rows]
        self.panel_last_finger_clearance[rows] = clearance
        self.panel_last_lateral_error[rows] = tangent_error
        self.panel_last_axis_error[rows] = orientation_error
        self.panel_last_ee_speed[rows] = ee_speed
        sample = (
            (clearance >= 0.0)
            & (clearance <= self.panel_standoff_acceptance_clearance[rows])
            & (tangent_error < 0.035)
            & (orientation_error <= self.grasp_orientation_tolerance[rows])
            & (ee_speed < 0.06)
        )
        settled = torch.where(
            sample, self.panel_contact_settle_steps[rows] + 1, 0
        )
        self.panel_contact_settle_steps[rows] = settled
        ready = rows[
            (settled >= 8)
            & (state["gripper_position"][rows, 0].abs() < 0.20)
        ]
        self.stage_steps[rows] += 1
        return ready

    def begin_push(
        self,
        state: dict[str, torch.Tensor],
        rows: torch.Tensor,
        finger_clearance: torch.Tensor,
    ) -> None:
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        self.push_start_hinge_angle[rows] = hinge
        self.push_last_progress_hinge[rows] = hinge
        self.push_no_progress_steps[rows] = 0
        # The push arc follows the physical panel contact point.  The inverse
        # command is only the actuator-side target used to reach the standoff.
        self.push_contact_world[rows] = self.panel_contact_target_world[rows]
        self.push_orientation[rows] = state["ee_orn"][rows]
        self.push_command_hinge_angle[rows] = hinge
        self.push_speed_error_integral[rows] = 0.0
        self.push_overspeed_steps[rows] = 0
        self.push_max_observed_angular_speed[rows] = state[
            "door_joint_velocity"
        ][rows, self.hinge_idx].abs()
        self.push_arm_detached[rows] = False
        self.push_recovery_settle_steps[rows] = 0
        self.push_reengage_reachable[rows] = False
        self.push_previous_finger_clearance[rows] = finger_clearance[rows]
        self.push_base_start_xy[rows] = state["root_states"][rows, :2]
        self.push_entry_ee_world[rows] = state["ee_pos"][rows]
        self.push_entry_command_world[rows] = self.ee_command_world[rows]
        self.transition(rows, self.Stage.PUSH_OPEN)
        self.stage_steps[rows] = 1

    def push_retract_requests(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.PUSH_OPEN))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        reach = torch.linalg.vector_norm(
            state["ee_goal_cart_local"][rows, :2], dim=-1
        )
        return rows[reach > self.push_comfort_reach_radius[rows]]

    def install_push_retract_pose(
        self, state: dict[str, torch.Tensor], env_ids, poses_local
    ) -> None:
        rows = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        self.active_arm_retract_pose_local[rows] = torch.as_tensor(
            poses_local, dtype=torch.float32, device=self.device
        )
        self.push_arm_detached[rows] = True
        self.push_recovery_settle_steps[rows] = 0
        self.push_detached_ee_goal_local[rows] = state["ee_goal_cart_local"][rows]
        self.transition(rows, self.Stage.PUSH_ARM_RETRACT)

    def _world_orientation_from_local_rpy(
        self, rows: torch.Tensor, local_rpy: torch.Tensor
    ) -> torch.Tensor:
        return _quat_multiply(
            self.env.get_goal_reference_quat()[rows], _euler_xyz_quat(local_rpy)
        )

    def command_push_open(
        self,
        state: dict[str, torch.Tensor],
        finger_clearance: torch.Tensor,
        buffers: TensorCommandBuffers,
    ) -> None:
        rows = torch.where(self.mask(self.Stage.PUSH_OPEN))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows)
        buffers.base[rows] = command
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        hinge_velocity = state["door_joint_velocity"][rows, self.hinge_idx]
        speed_error = self.push_target_angular_speed[rows] - hinge_velocity
        integral = (
            self.push_speed_error_integral[rows] + speed_error * self.dt
        ).clamp(
            -self.push_speed_integral_limit[rows],
            self.push_speed_integral_limit[rows],
        )
        self.push_speed_error_integral[rows] = integral
        angular_offset = (
            self.push_speed_kp[rows] * speed_error
            + self.push_speed_ki[rows] * integral
        ).clamp(-self.push_max_angular_lead[rows], self.push_max_angular_lead[rows])
        target_hinge = torch.maximum(
            torch.minimum(hinge + angular_offset, self.push_goal_hinge_angle[rows]),
            self.push_start_hinge_angle[rows] - math.radians(2.0),
        )
        self.push_command_hinge_angle[rows] = target_hinge
        delta = self.hinge_axis_sign[rows] * (
            target_hinge - self.push_start_hinge_angle[rows]
        )
        target = _rotate_about_z(
            self.push_contact_world[rows], self.hinge_world[rows], delta
        )
        self.ee_command_world[rows] = _slew_position(
            self.ee_command_world[rows], target, 0.003
        )
        # Preserve the current baseline: PUSH_OPEN ultimately holds the
        # accepted contact orientation instead of rotating it with the door.
        self.ee_orientation_command[rows] = self.push_orientation[rows]
        self._set_cartesian_targets(
            rows,
            self.ee_command_world[rows],
            self.ee_orientation_command[rows],
            buffers,
        )
        buffers.gripper[rows, 0] = self.q_closed[rows]
        self.push_previous_finger_clearance[rows] = finger_clearance[rows]

    def command_push_retract(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.PUSH_ARM_RETRACT))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows)
        buffers.base[rows] = command
        delta = (
            self.active_arm_retract_pose_local[rows, :3]
            - self.push_detached_ee_goal_local[rows]
        )
        distance = torch.linalg.vector_norm(delta, dim=-1)
        moving = self.stage_steps[rows] >= 5
        step = delta * torch.clamp(0.004 / distance.clamp_min(1e-8), max=1.0)[:, None]
        self.push_detached_ee_goal_local[rows] = torch.where(
            moving[:, None], self.push_detached_ee_goal_local[rows] + step,
            self.push_detached_ee_goal_local[rows],
        )
        orientation = self._world_orientation_from_local_rpy(
            rows, self.active_arm_retract_pose_local[rows, 3:]
        )
        self.ee_orientation_command[rows] = _slew_quaternion(
            self.ee_orientation_command[rows], orientation, 0.025
        )
        buffers.ee_cart[rows] = self.push_detached_ee_goal_local[rows]
        buffers.ee_quat[rows] = self.ee_orientation_command[rows]
        buffers.ee_quat_enabled[rows] = True
        buffers.ee_rpy[rows] = _orientation_delta(
            buffers.ee_cart[rows], self.ee_orientation_command[rows],
            self.env.get_goal_reference_quat()[rows], self.arm_induced_pitch[rows]
        )
        buffers.arm_joint_mode[rows] = False
        buffers.gripper[rows, 0] = self.q_closed[rows]

    def advance_push_retract(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.PUSH_ARM_RETRACT))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        origin = self.env.get_goal_reference_origin()[rows]
        reference = self.env.get_goal_reference_quat()[rows]
        offset = self.env.get_goal_center_offset_local()[rows]
        actual_local = _quat_rotate_inverse(
            reference, state["ee_pos"][rows] - origin
        ) - offset
        orientation_target = self._world_orientation_from_local_rpy(
            rows, self.active_arm_retract_pose_local[rows, 3:]
        )
        retracted = (
            (torch.linalg.vector_norm(
                actual_local - self.active_arm_retract_pose_local[rows, :3], dim=-1
            ) <= 0.010)
            & (_quat_distance(state["ee_orn"][rows], orientation_target) < 0.05)
            & (torch.linalg.vector_norm(state["ee_vel"][rows, :3], dim=-1) < 0.04)
        )
        settled = torch.where(
            retracted, self.push_recovery_settle_steps[rows] + 1, 0
        )
        self.push_recovery_settle_steps[rows] = settled
        completed = rows[settled >= 10]
        if len(completed):
            arm_ids = self.env.arm_joint_ids.to(device=self.device, dtype=torch.int64)
            self.base_advance_arm_joint_target[completed] = state["dof_pos"][
                completed[:, None], arm_ids[None, :]
            ]
            self.push_reengage_reachable[completed] = False
            self.base_advance_target_longitudinal[completed] = -torch.inf
            self.transition(completed, self.Stage.BASE_ADVANCE)
        active = self.mask(self.Stage.PUSH_ARM_RETRACT)
        self.stage_steps[active] += 1
        return completed

    def advance_push_common(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = (
            self.mask(self.Stage.PUSH_OPEN)
            | self.mask(self.Stage.PUSH_ARM_RETRACT)
            | self.mask(self.Stage.BASE_ADVANCE)
        )
        rows = torch.where(mask)[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        speed = state["door_joint_velocity"][rows, self.hinge_idx]
        self.push_max_observed_angular_speed[rows] = torch.maximum(
            self.push_max_observed_angular_speed[rows], speed.clamp_min(0.0)
        )
        overspeed = speed > self.push_max_angular_speed[rows]
        self.push_overspeed_steps[rows] = torch.where(
            overspeed, self.push_overspeed_steps[rows] + 1, 0
        )
        failure = (self.push_overspeed_steps[rows] >= 3) | (
            hinge > math.radians(94.0)
        )
        self.fail(rows[failure], 20)
        through = rows[(~failure) & (hinge >= math.radians(89.0))]
        push = (self.stage[rows] == int(self.Stage.PUSH_OPEN)) & ~failure
        push_rows = rows[push]
        if len(push_rows):
            push_hinge = state["door_joint_position"][push_rows, self.hinge_idx]
            progressed = push_hinge >= (
                self.push_last_progress_hinge[push_rows] + math.radians(0.25)
            )
            self.push_last_progress_hinge[push_rows] = torch.where(
                progressed, push_hinge, self.push_last_progress_hinge[push_rows]
            )
            commanded = self.push_command_hinge_angle[push_rows] > (
                push_hinge + math.radians(0.5)
            )
            self.push_no_progress_steps[push_rows] = torch.where(
                progressed | ~commanded,
                0,
                self.push_no_progress_steps[push_rows] + 1,
            )
            stalled = self.push_no_progress_steps[push_rows] > 300
            self.fail(push_rows[stalled], 21)
        return through

    def _tracked_push_contact(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor
    ) -> torch.Tensor:
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        delta = self.hinge_axis_sign[rows] * (
            hinge - self.push_start_hinge_angle[rows]
        )
        return _rotate_about_z(
            self.push_contact_world[rows], self.hinge_world[rows], delta
        )

    def _base_advance_targets(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = state["root_states"][rows]
        forward = self.door_forward[rows, :2]
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        line_origin = self.panel_station_xy[rows]
        current_longitudinal = torch.sum(
            (base[:, :2] - line_origin) * forward, dim=-1
        )
        line_position = line_origin + current_longitudinal[:, None] * forward
        contact = self._tracked_push_contact(state, rows)
        origin = self.env.get_goal_reference_origin()[rows]
        reference = self.env.get_goal_reference_quat()[rows]
        offset = self.env.get_goal_center_offset_local()[rows]
        contact_local = _quat_rotate_inverse(reference, contact - origin) - offset
        forward_world = torch.cat(
            (forward, torch.zeros((len(rows), 1), device=self.device)), dim=-1
        )
        forward_local = _quat_rotate_inverse(reference, forward_world)[:, :2]
        lateral_world_xy = line_position - base[:, :2]
        lateral_world = torch.cat(
            (lateral_world_xy, torch.zeros((len(rows), 1), device=self.device)), dim=-1
        )
        contact_xy = contact_local[:, :2] - _quat_rotate_inverse(
            reference, lateral_world
        )[:, :2]
        radius = self.push_reengage_reach_radius[rows]
        a = torch.sum(forward_local.square(), dim=-1)
        b = torch.sum(contact_xy * forward_local, dim=-1)
        c = torch.sum(contact_xy.square(), dim=-1) - radius.square()
        discriminant = b.square() - a * c
        root = torch.sqrt(discriminant.clamp_min(0.0))
        interval_start = (b - root) / a.clamp_min(1e-10)
        interval_end = (b + root) / a.clamp_min(1e-10)
        solvable = (c > 0.0) & (a > 1e-10) & (discriminant >= 0.0) & (interval_end >= 0.0)
        required = torch.where(
            c <= 0.0,
            torch.zeros_like(c),
            torch.where(solvable, interval_start.clamp_min(0.0), torch.zeros_like(c)),
        )
        required += torch.where(
            (c <= 0.0) | solvable,
            self.push_reengage_target_lead[rows],
            torch.zeros_like(required),
        )

        closed_free_end = 2.0 * self.panel_center_world[rows] - self.hinge_world[rows]
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        panel_free_end = _rotate_about_z(
            closed_free_end,
            self.hinge_world[rows],
            self.hinge_axis_sign[rows] * hinge,
        )
        panel_start = self.hinge_world[rows, :2]
        panel_end = panel_free_end[:, :2]
        shoulders = state["robot_body_state_w"][
            rows[:, None], self.front_shoulder_body_indices[None, :], :2
        ]
        count = torch.maximum(
            torch.full_like(self.stage_steps[rows], 2),
            torch.ceil(required / 0.005).to(torch.int32) + 1,
        )
        capacity = 256
        if torch.any(count > capacity):
            raise RuntimeError("base-advance shoulder search exceeds 256 candidates")
        candidate_id = torch.arange(capacity, device=self.device)
        valid = candidate_id[None, :] < count[:, None]
        fraction = candidate_id[None, :].to(torch.float32) / (
            count[:, None] - 1
        ).clamp_min(1)
        advances = fraction * required[:, None]
        targets = line_position[:, None, :] + advances[..., None] * forward[:, None, :]
        displacement = targets - base[:, None, :2]
        points = shoulders[:, None, :, :] + displacement[:, :, None, :]
        segment = panel_end - panel_start
        segment_norm_sq = torch.sum(segment.square(), dim=-1)
        projection = (
            torch.sum(
                (points - panel_start[:, None, None, :])
                * segment[:, None, None, :],
                dim=-1,
            )
            / segment_norm_sq[:, None, None].clamp_min(1e-10)
        ).clamp(0.0, 1.0)
        closest = panel_start[:, None, None, :] + projection[..., None] * segment[:, None, None, :]
        distances = torch.linalg.vector_norm(points - closest, dim=-1).amin(dim=-1)
        required_clearance = (
            self.push_shoulder_collision_radius[rows]
            + self.panel_half_thickness[rows]
            + self.push_shoulder_safety_margin[rows]
        )
        safe = (distances >= required_clearance[:, None]) & valid
        unsafe = (~safe) & valid
        first_unsafe = torch.argmax(unsafe.to(torch.int32), dim=-1)
        has_unsafe = torch.any(unsafe, dim=-1)
        target_index = torch.where(
            has_unsafe, (first_unsafe - 1).clamp_min(0), count.to(torch.int64) - 1
        )
        target_advance = advances[
            torch.arange(len(rows), device=self.device), target_index
        ]
        proposed = current_longitudinal + target_advance
        previous = self.base_advance_target_longitudinal[rows]
        longitudinal = torch.maximum(
            current_longitudinal, torch.maximum(proposed, previous)
        )
        longitudinal = torch.where(
            self.push_reengage_reachable[rows], current_longitudinal, longitudinal
        )
        self.base_advance_target_longitudinal[rows] = longitudinal
        return line_origin + longitudinal[:, None] * forward, yaw

    def command_base_advance(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.BASE_ADVANCE))[0]
        if not len(rows):
            return
        target_xy, target_yaw = self._base_advance_targets(state, rows)
        command, distance, yaw_error = track_base_pose(
            state["root_states"][rows],
            target_xy,
            target_yaw,
            target_xy,
            target_yaw,
            minimum_speed_toward_target_only=False,
        )
        contact = self._tracked_push_contact(state, rows)
        origin = self.env.get_goal_reference_origin()[rows]
        reference = self.env.get_goal_reference_quat()[rows]
        offset = self.env.get_goal_center_offset_local()[rows]
        contact_local = _quat_rotate_inverse(reference, contact - origin) - offset
        reach = torch.linalg.vector_norm(contact_local[:, :2], dim=-1)
        reachable = self.push_reengage_reachable[rows]
        reachable = torch.where(
            reachable,
            reach <= self.push_reengage_release_radius[rows],
            reach <= self.push_reengage_reach_radius[rows],
        )
        self.push_reengage_reachable[rows] = reachable
        command[reachable] = 0.0
        command = self._stabilize_base(rows, command)
        self.pending_station_distance[rows] = distance
        self.pending_yaw_error[rows] = yaw_error
        self.base_hold_target_xy[rows] = target_xy
        self.base_hold_target_yaw[rows] = target_yaw
        self.base_hold_nominal_xy[rows] = target_xy
        self.base_hold_nominal_yaw[rows] = target_yaw
        buffers.base[rows] = command
        buffers.arm_joint_mode[rows] = True
        buffers.arm_joint[rows] = self.base_advance_arm_joint_target[rows]
        buffers.gripper[rows, 0] = self.q_closed[rows]
        ready = (
            reachable
            & (state["door_joint_velocity"][rows, self.hinge_idx].abs()
               < self.turn_release_max_hinge_speed[rows])
            & (torch.linalg.vector_norm(state["root_states"][rows, 7:9], dim=-1) < 0.04)
        )
        settled = torch.where(
            ready, self.push_recovery_settle_steps[rows] + 1, 0
        )
        self.push_recovery_settle_steps[rows] = settled

    def advance_base_advance(
        self, state: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.BASE_ADVANCE))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        completed = rows[self.push_recovery_settle_steps[rows] >= 10]
        if len(completed):
            hinge = state["door_joint_position"][completed, self.hinge_idx]
            delta = self.hinge_axis_sign[completed] * (
                hinge - self.push_start_hinge_angle[completed]
            )
            self.panel_contact_orientation_seed[completed] = _quat_multiply(
                _yaw_quat(delta), self.push_orientation[completed]
            )
            self.panel_contact_orientation_seed_valid[completed] = True
            self.ee_command_world[completed] = state["ee_pos"][completed]
            self.panel_contact_plan_valid[completed] = False
            self.panel_contact_settle_steps[completed] = 0
            self.transition(completed, self.Stage.CONTACT_PANEL)
            self.stage_steps[completed] = 1
        active = self.mask(self.Stage.BASE_ADVANCE)
        self.stage_steps[active] += 1
        return completed

    def begin_through(self, rows: torch.Tensor) -> None:
        self.through_best_station_error[rows] = torch.inf
        self.through_no_progress_steps[rows] = 0
        self.through_settle_steps[rows] = 0
        self.through_stop_latched[rows] = False
        self.through_motion_ready[rows] = False
        self.through_entry_stable_steps[rows] = 0
        self.transition(rows, self.Stage.THROUGH)
        self.stage_steps[rows] = 1

    def _base_velocity_damping(
        self, roots: torch.Tensor
    ) -> torch.Tensor:
        yaw = _yaw_from_wxyz(roots[:, 3:7])
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        command = torch.stack(
            (
                -(cosine * roots[:, 7] + sine * roots[:, 8]),
                -(-sine * roots[:, 7] + cosine * roots[:, 8]),
                -0.35 * roots[:, 12],
            ),
            dim=-1,
        )
        return _limit_base_command(command)

    def command_through(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(
            self.mask(self.Stage.THROUGH) | self.mask(self.Stage.SUCCESS)
        )[0]
        if not len(rows):
            return
        command, distance, yaw_error = track_base_pose(
            state["root_states"][rows],
            self.through_command_xy[rows],
            self.through_command_yaw[rows],
            self.through_station_xy[rows],
            self.through_station_yaw[rows],
            minimum_speed_toward_target_only=False,
        )
        latched = self.through_stop_latched[rows]
        accepted = (
            (distance < BASE_STATION_TOLERANCE_M)
            & (yaw_error < BASE_YAW_TOLERANCE_RAD)
        )
        core = (distance < THROUGH_STOP_DISTANCE_M) & (
            yaw_error < THROUGH_STOP_YAW_RAD
        )
        latched = torch.where(latched, accepted, core)
        self.through_stop_latched[rows] = latched
        through = self.stage[rows] == int(self.Stage.THROUGH)
        command = torch.where(
            (through & latched)[:, None],
            self._base_velocity_damping(state["root_states"][rows]),
            command,
        )
        command = torch.where(
            (through & ~self.through_motion_ready[rows])[:, None],
            torch.zeros_like(command),
            command,
        )
        command = self._stabilize_base(rows, command)
        self.pending_station_distance[rows] = distance
        self.pending_yaw_error[rows] = yaw_error
        buffers.base[rows] = command
        buffers.arm_joint_mode[rows] = True
        buffers.arm_joint[rows] = self.middle_arm_joint_target[rows]
        buffers.gripper[rows, 0] = self.q_closed[rows]

    def _tail_clearance(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor
    ) -> torch.Tensor:
        base = state["root_states"][rows]
        rear_local = torch.zeros((len(rows), 3), device=self.device)
        rear_local[:, 0] = ROBOT_TRUNK_REAR_X
        rear_world = _quat_rotate(base[:, 3:7], rear_local)
        tail_xy = base[:, :2] + rear_world[:, :2]
        return torch.sum(
            (tail_xy - self.initial_door_root[rows, :2])
            * self.door_forward[rows, :2],
            dim=-1,
        )

    def advance_through(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.THROUGH))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        base = state["root_states"][rows]
        speed = torch.linalg.vector_norm(base[:, 7:10], dim=-1)
        angular_speed = torch.linalg.vector_norm(base[:, 10:13], dim=-1)
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        hinge_speed = state["door_joint_velocity"][rows, self.hinge_idx]
        up = _quat_rotate(
            base[:, 3:7],
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(len(rows), -1),
        )[:, 2]
        entry_stable = (
            (base[:, 2] >= self.through_min_base_height[rows])
            & (up >= THROUGH_ENTRY_UPRIGHT_COS)
            & (speed < TERMINAL_BASE_LINEAR_SPEED_M_S)
            & (angular_speed < TERMINAL_BASE_ANGULAR_SPEED_RAD_S)
            & (hinge_speed.abs() < TERMINAL_DOOR_SPEED_RAD_S)
        )
        entry_steps = torch.where(
            entry_stable, self.through_entry_stable_steps[rows] + 1, 0
        )
        self.through_entry_stable_steps[rows] = entry_steps
        self.through_motion_ready[rows] |= entry_steps >= THROUGH_ENTRY_STABILITY_FRAMES
        distance = self.pending_station_distance[rows]
        improved = distance < self.through_best_station_error[rows] - 0.01
        self.through_best_station_error[rows] = torch.where(
            improved, distance, self.through_best_station_error[rows]
        )
        no_progress = torch.where(
            ~self.through_motion_ready[rows] | improved | (distance < BASE_STATION_TOLERANCE_M),
            0,
            self.through_no_progress_steps[rows] + 1,
        )
        self.through_no_progress_steps[rows] = no_progress
        arm_ids = self.env.arm_joint_ids.to(device=self.device, dtype=torch.int64)
        arm_position = state["dof_pos"][rows[:, None], arm_ids[None, :]]
        arm_velocity = state["dof_vel"][rows[:, None], arm_ids[None, :]]
        middle_error = (
            arm_position - self.middle_arm_joint_target[rows]
        ).abs().amax(dim=-1)
        arm_speed = arm_velocity.abs().amax(dim=-1)
        gripper_id = int(self.env.gripper_joint_ids[0].item())
        gripper_speed = state["dof_vel"][rows, gripper_id].abs()
        door_speed = state["door_joint_velocity"][rows].abs().amax(dim=-1)
        terminal = (
            (distance < BASE_STATION_TOLERANCE_M)
            & (self.pending_yaw_error[rows] < BASE_YAW_TOLERANCE_RAD)
            & (speed < TERMINAL_BASE_LINEAR_SPEED_M_S)
            & (angular_speed < TERMINAL_BASE_ANGULAR_SPEED_RAD_S)
            & (middle_error < 0.05)
            & (arm_speed < TERMINAL_ARM_JOINT_SPEED_RAD_S)
            & (gripper_speed < TERMINAL_GRIPPER_SPEED_RAD_S)
            & (door_speed < TERMINAL_DOOR_SPEED_RAD_S)
        )
        settled = torch.where(
            terminal, self.through_settle_steps[rows] + 1, 0
        )
        self.through_settle_steps[rows] = settled
        tail = self._tail_clearance(state, rows)
        fallen = (up < 0.75) | (base[:, 2] < 0.30)
        swingback = (hinge < math.radians(65.0)) & (tail < 0.8)
        stalled = (distance >= BASE_STATION_TOLERANCE_M) & (no_progress > 150)
        failure = fallen | swingback | stalled
        self.fail(rows[failure], 30)
        success = rows[(~failure) & (settled > TERMINAL_STABILITY_FRAMES)]
        self.stage[success] = int(self.Stage.SUCCESS)
        self.stage_steps[rows[(~failure) & ~(settled > TERMINAL_STABILITY_FRAMES)]] += 1
        return success

    def apply_timeouts(self) -> torch.Tensor:
        active = (self.stage != int(self.Stage.SUCCESS)) & (
            self.stage != int(self.Stage.FAILURE)
        )
        limits = self.stage_limits[self.stage]
        timed_out = active & (limits > 0) & (self.stage_steps > limits)
        self.fail(torch.where(timed_out)[0], 1)
        return torch.where(timed_out)[0]

    def command(
        self,
        state: dict[str, torch.Tensor],
        buffers: TensorCommandBuffers,
        *,
        grasp_geometry: dict[str, torch.Tensor] | None,
        finger_clearance: torch.Tensor,
    ) -> None:
        """Issue one command for every active env using stage masks only."""
        terminal = self.mask(self.Stage.SUCCESS) | self.mask(self.Stage.FAILURE)
        buffers.base[terminal] = 0.0
        self.command_approach(state, buffers)
        self.command_pregrasp_open(state, buffers)
        if grasp_geometry is not None:
            self.command_center_grasp(state, grasp_geometry, buffers)
        self.command_turn(state, buffers)
        self.command_return(state, buffers)
        self.command_open_and_clear(state, buffers)
        self.command_retreat(state, buffers)
        self.command_restore(state, buffers)
        self.command_align_panel(state, buffers)
        self.command_contact_panel(state, buffers)
        self.command_push_open(state, finger_clearance, buffers)
        self.command_push_retract(state, buffers)
        self.command_base_advance(state, buffers)
        self.command_through(state, buffers)

    def advance(
        self,
        state: dict[str, torch.Tensor],
        *,
        grasp_geometry: dict[str, torch.Tensor] | None,
        finger_clearance: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Advance all envs once and return sparse planning event masks."""
        events: dict[str, torch.Tensor] = {}
        # Reverse stage order prevents a forward transition from being
        # evaluated a second time in the same physics frame.
        events["success"] = self.advance_through(state)
        push_to_through = self.advance_push_common(state)
        if len(push_to_through):
            self.begin_through(push_to_through)
        events["contact_replan"] = self.advance_base_advance(state)
        self.advance_push_retract(state)
        push_rows = torch.where(self.mask(self.Stage.PUSH_OPEN))[0]
        self.stage_steps[push_rows] += 1
        contact_to_push = self.advance_contact_panel(state, finger_clearance)
        if len(contact_to_push):
            self.begin_push(state, contact_to_push, finger_clearance)
        events["contact_plan"] = self.advance_align_panel(state)
        self.advance_restore(state)
        events["restore_plan"] = self.advance_retreat(state)
        events["retreat_plan"] = self.advance_open_and_clear(state)
        events["release_open_plan"] = self.advance_return(state)
        turn_to_return = self.advance_turn(state)
        if len(turn_to_return):
            self.begin_return(state, turn_to_return)
        if grasp_geometry is not None:
            grasp_to_turn = self.advance_center_grasp(state, grasp_geometry)
            if len(grasp_to_turn):
                self.begin_turn(state, grasp_to_turn)
        events["center_plan"] = self.advance_pregrasp_open(state)
        events["pregrasp_plan"] = self.advance_approach(state)
        events["timeout"] = self.apply_timeouts()
        return events
