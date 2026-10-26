"""Shared device-only pose math and locomotion tracking."""

import math

import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_dot,
    fixed_norm,
    fixed_normalize,
)

MAX_BASE_PLANAR_SPEED = 0.4
MAX_BASE_YAW_RATE = 0.6


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
        quaternion[:, :1] * first_cross + torch.cross(-q_vector, first_cross, dim=-1)
    )


def _quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vector = quaternion[:, 1:]
    first_cross = torch.cross(q_vector, vector, dim=-1)
    return vector + 2.0 * (
        quaternion[:, :1] * first_cross + torch.cross(q_vector, first_cross, dim=-1)
    )


def _quat_distance(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    actual = fixed_normalize(actual, eps=1e-8)
    target = fixed_normalize(target, eps=1e-8)
    dot = fixed_dot(actual, target).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _signed_quat_rotation_about_axis(
    start: torch.Tensor, target: torch.Tensor, axis: torch.Tensor
) -> torch.Tensor:
    """Return the signed world-axis rotation taking start toward target."""
    start = fixed_normalize(start, eps=1e-8)
    target = fixed_normalize(target, eps=1e-8)
    axis = fixed_normalize(axis, eps=1e-8)
    relative = fixed_normalize(_quat_multiply(target, _quat_conjugate(start)), eps=1e-8)
    relative = torch.where(relative[..., :1] < 0.0, -relative, relative)
    sin_half = fixed_dot(relative[..., 1:], axis)
    return _wrapped_angle(
        2.0 * torch.atan2(sin_half, relative[..., 0]),
        torch.zeros_like(sin_half),
    )


def _axis_angle_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = fixed_normalize(axis, eps=1e-8)
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


def _rotate_about_z(point: torch.Tensor, center: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    delta = point - center
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    result = point.clone()
    result[..., 0] = center[..., 0] + cosine * delta[..., 0] - sine * delta[..., 1]
    result[..., 1] = center[..., 1] + sine * delta[..., 0] + cosine * delta[..., 1]
    return result


def _slew_position(current: torch.Tensor, target: torch.Tensor, max_step: float) -> torch.Tensor:
    delta = target - current
    distance = fixed_norm(delta, keepdim=True)
    scale = torch.clamp(float(max_step) / distance.clamp_min(1e-8), max=1.0)
    return current + delta * scale


def _slew_quaternion(
    current: torch.Tensor, target: torch.Tensor, max_angle: float = 0.025
) -> torch.Tensor:
    current = fixed_normalize(current, eps=1e-8)
    target = fixed_normalize(target, eps=1e-8)
    dot = fixed_dot(current, target, keepdim=True)
    target = torch.where(dot < 0.0, -target, target)
    dot = dot.abs().clamp(0.0, 1.0)
    angle = 2.0 * torch.acos(dot)
    fraction = torch.clamp(float(max_angle) / angle.clamp_min(1e-8), max=1.0)
    theta = torch.acos(dot.clamp(-1.0, 1.0))
    denominator = torch.sin(theta)
    slerp = (
        torch.sin((1.0 - fraction) * theta) * current + torch.sin(fraction * theta) * target
    ) / denominator.clamp_min(1e-8)
    output = torch.where((theta < 1e-6) | (angle <= max_angle), target, slerp)
    return fixed_normalize(output, eps=1e-8)


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
    radius = fixed_norm(cart_local).clamp_min(1.0e-8)
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
    norm = fixed_norm(planar, keepdim=True)
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
    reference_velocity_world: torch.Tensor | None = None,
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
    if reference_velocity_world is not None:
        vx, vy, wz = reference_velocity_world.unbind(-1)
        command[:, 0] += 2.0 * (cosine * vx + sine * vy)
        command[:, 1] += 2.0 * (-sine * vx + cosine * vy)
        command[:, 2] += 1.35 * wz
    command = _limit_base_command(command)
    planar_norm = fixed_norm(command[:, :2])
    deadzone = (planar_norm > 1.0e-6) & (planar_norm < 0.15)
    if minimum_speed_toward_target_only:
        deadzone &= (command[:, 0] * local_x + command[:, 1] * local_y) > 0.0
    scale = torch.where(deadzone, 0.15 / planar_norm.clamp_min(1.0e-8), 1.0)
    command[:, :2] *= scale[:, None]
    yaw_deadzone = (command[:, 2].abs() > 1.0e-6) & (command[:, 2].abs() < 0.2)
    if minimum_speed_toward_target_only:
        yaw_deadzone &= command[:, 2] * yaw_error > 0.0
    command[:, 2] = torch.where(
        yaw_deadzone,
        torch.copysign(torch.full_like(command[:, 2], 0.2), command[:, 2]),
        command[:, 2],
    )

    acceptance_distance = fixed_norm(acceptance_xy - root_states[:, :2])
    acceptance_yaw_error = _wrapped_angle(acceptance_yaw, yaw).abs()
    return command, acceptance_distance, acceptance_yaw_error
