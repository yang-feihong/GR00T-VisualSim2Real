"""GPU-batched steady EE inverse without gravity compensation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gr00t.rl.envs.door.b2z1_reset_geometry import (
    b2z1_gripper_pose_and_jacobian,
)
from .constrained_ik import batched_constrained_dls_delta


@dataclass(frozen=True)
class BatchedSteadyEeInverseResult:
    command_position_world: torch.Tensor
    command_orientation_world_wxyz: torch.Tensor
    arm_joint_positions: torch.Tensor
    arm_position_targets: torch.Tensor
    position_error: torch.Tensor
    orientation_error: torch.Tensor
    exact: torch.Tensor


def _matrix_to_quaternion_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to normalized wxyz quaternions on device."""
    m00, m11, m22 = rotation[..., 0, 0], rotation[..., 1, 1], rotation[..., 2, 2]
    candidates = torch.stack(
        (
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ),
        dim=-1,
    ).clamp_min(0.0)
    index = candidates.argmax(dim=-1)
    scale = 0.5 / torch.sqrt(
        candidates.gather(-1, index.unsqueeze(-1)).squeeze(-1).clamp_min(1.0e-12)
    )
    rows = (
        torch.stack((1.0 + m00 + m11 + m22, rotation[..., 2, 1] - rotation[..., 1, 2], rotation[..., 0, 2] - rotation[..., 2, 0], rotation[..., 1, 0] - rotation[..., 0, 1]), dim=-1),
        torch.stack((rotation[..., 2, 1] - rotation[..., 1, 2], 1.0 + m00 - m11 - m22, rotation[..., 0, 1] + rotation[..., 1, 0], rotation[..., 0, 2] + rotation[..., 2, 0]), dim=-1),
        torch.stack((rotation[..., 0, 2] - rotation[..., 2, 0], rotation[..., 0, 1] + rotation[..., 1, 0], 1.0 - m00 + m11 - m22, rotation[..., 1, 2] + rotation[..., 2, 1]), dim=-1),
        torch.stack((rotation[..., 1, 0] - rotation[..., 0, 1], rotation[..., 0, 2] + rotation[..., 2, 0], rotation[..., 1, 2] + rotation[..., 2, 1], 1.0 - m00 - m11 + m22), dim=-1),
    )
    quaternion = torch.stack(rows, dim=-2).gather(
        -2, index[..., None, None].expand(*index.shape, 1, 4)
    ).squeeze(-2)
    quaternion = quaternion * scale.unsqueeze(-1)
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    return torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)


def _quaternion_nlerp_wxyz(
    start: torch.Tensor, end: torch.Tensor, fraction: torch.Tensor
) -> torch.Tensor:
    end = torch.where((start * end).sum(dim=-1, keepdim=True) < 0.0, -end, end)
    result = start + fraction.unsqueeze(-1) * (end - start)
    return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )


def _quaternion_multiply_wxyz(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    w1, x1, y1, z1 = left.unbind(dim=-1)
    w2, x2, y2, z2 = right.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _cpu_orientation_error_and_jacobian(
    desired_orientation: torch.Tensor,
    current_rotation: torch.Tensor,
    angular_jacobian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the CPU inverse quaternion residual and its command Jacobian."""
    current_orientation = _matrix_to_quaternion_wxyz(current_rotation)
    current_conjugate = current_orientation.clone()
    current_conjugate[..., 1:] *= -1.0
    relative = _quaternion_multiply_wxyz(
        desired_orientation, current_conjugate
    )
    relative = torch.where(relative[..., :1] < 0.0, -relative, relative)
    scalar = relative[..., 0]
    vector = relative[..., 1:]

    vx, vy, vz = vector.unbind(dim=-1)
    zero = torch.zeros_like(vx)
    skew = torch.stack(
        (
            zero, -vz, vy,
            vz, zero, -vx,
            -vy, vx, zero,
        ),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)
    identity = torch.eye(
        3, dtype=vector.dtype, device=vector.device
    ).expand(*vector.shape[:-1], 3, 3)
    command_jacobian = 0.5 * torch.matmul(
        scalar[..., None, None] * identity + skew,
        angular_jacobian,
    )
    return vector, command_jacobian


def _cpu_orientation_error(
    desired_orientation: torch.Tensor, current_rotation: torch.Tensor
) -> torch.Tensor:
    current_orientation = _matrix_to_quaternion_wxyz(current_rotation)
    current_conjugate = current_orientation.clone()
    current_conjugate[..., 1:] *= -1.0
    relative = _quaternion_multiply_wxyz(
        desired_orientation, current_conjugate
    )
    relative = torch.where(relative[..., :1] < 0.0, -relative, relative)
    return relative[..., 1:]


def batched_steady_ee_inverse_no_gravity(
    desired_position_world: torch.Tensor,
    desired_orientation_world_wxyz: torch.Tensor,
    *,
    base_position_world: torch.Tensor,
    base_orientation_world_xyzw: torch.Tensor,
    initial_arm_joint_positions: torch.Tensor,
    joint_limits: torch.Tensor,
    position_tolerance_m: float = 5.0e-3,
    orientation_tolerance: float = 5.0e-2,
    ik_damping: float = 1.0e-4,
    continuation_position_step_m: float = 0.03,
    continuation_orientation_step_rad: float = 0.15,
    iterations_per_waypoint: int = 40,
) -> BatchedSteadyEeInverseResult:
    """Match steady inverse semantics on GPU, omitting only gravity compensation.

    With gravity removed, the equilibrium arm target equals the solved arm state,
    and the Cartesian command equals the reachable IK pose.
    """
    desired_orientation = desired_orientation_world_wxyz / torch.linalg.vector_norm(
        desired_orientation_world_wxyz, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    if not desired_position_world.is_cuda:
        raise ValueError("steady GPU inverse requires CUDA inputs")
    from .ee_steady_inverse_kernel import steady_inverse_kernel
    import warp as wp

    count = len(initial_arm_joint_positions)
    limits = joint_limits
    if limits.ndim == 2:
        limits = limits.expand(count, -1, -1)
    # Uniform double-precision internal arithmetic avoids world-origin rounding
    # and is independent of the caller's dtype and neighboring batch rows.
    values = [
        value.to(dtype=torch.float64).contiguous()
        for value in (
            desired_position_world, desired_orientation, base_position_world,
            base_orientation_world_xyzw, initial_arm_joint_positions, limits,
        )
    ]
    solved = torch.empty_like(values[4])
    iterations = torch.empty(count, device=solved.device, dtype=torch.int32)
    if count:
        wp.launch(
            steady_inverse_kernel,
            dim=count,
            inputs=[
                *[wp.from_torch(value) for value in values],
                continuation_position_step_m, continuation_orientation_step_rad,
                ik_damping, iterations_per_waypoint,
                wp.from_torch(solved), wp.from_torch(iterations),
            ],
            stream=wp.stream_from_torch(torch.cuda.current_stream(solved.device)),
        )
    arm = solved.to(dtype=initial_arm_joint_positions.dtype)

    position, rotation, _ = b2z1_gripper_pose_and_jacobian(
        base_position_world, base_orientation_world_xyzw, arm
    )
    orientation = _matrix_to_quaternion_wxyz(rotation)
    position_error = position - desired_position_world
    orientation_error = _cpu_orientation_error(desired_orientation, rotation)
    exact = (
        torch.linalg.vector_norm(position_error, dim=-1) <= position_tolerance_m
    ) & (
        torch.linalg.vector_norm(orientation_error, dim=-1)
        <= orientation_tolerance
    )
    return BatchedSteadyEeInverseResult(
        command_position_world=position,
        command_orientation_world_wxyz=orientation,
        arm_joint_positions=arm,
        arm_position_targets=arm.clone(),
        position_error=position_error,
        orientation_error=orientation_error,
        exact=exact,
    )
