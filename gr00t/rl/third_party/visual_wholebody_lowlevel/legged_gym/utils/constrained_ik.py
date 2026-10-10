"""GPU-batched constrained damped least-squares helpers."""

from __future__ import annotations

import torch


def batched_constrained_dls_delta(
    jacobian: torch.Tensor,
    pose_error: torch.Tensor,
    joint_position: torch.Tensor,
    joint_limits: torch.Tensor,
    damping: float,
    joint_limit_margin: float = 1.0e-3,
) -> torch.Tensor:
    """Match the scalar active-set DLS solve for every batch row."""
    batch_size, _, joint_count = jacobian.shape
    if pose_error.shape != (batch_size, jacobian.shape[1]):
        raise ValueError("pose_error shape does not match jacobian")
    if joint_position.shape != (batch_size, joint_count):
        raise ValueError("joint_position shape does not match jacobian")
    if joint_limits.shape not in ((joint_count, 2), (batch_size, joint_count, 2)):
        raise ValueError("joint_limits must have shape (J, 2) or (N, J, 2)")

    lower_delta = joint_limits[..., 0] + joint_limit_margin - joint_position
    upper_delta = joint_limits[..., 1] - joint_limit_margin - joint_position
    delta = torch.zeros_like(joint_position)
    free = torch.ones_like(joint_position, dtype=torch.bool)
    solved = torch.zeros(batch_size, device=jacobian.device, dtype=torch.bool)
    eye = torch.eye(joint_count, device=jacobian.device, dtype=jacobian.dtype)

    for _ in range(joint_count + 1):
        active_batch = ~solved
        if not torch.any(active_batch):
            break

        free_float = free.to(jacobian.dtype)
        fixed_delta = torch.where(free, torch.zeros_like(delta), delta)
        residual = pose_error - torch.bmm(
            jacobian, fixed_delta.unsqueeze(-1)
        ).squeeze(-1)
        free_jacobian = jacobian * free_float.unsqueeze(1)
        normal = torch.bmm(free_jacobian.transpose(1, 2), free_jacobian)
        diagonal = (
            free_float * (float(damping) ** 2) + (~free).to(jacobian.dtype)
        )
        normal = normal + eye.unsqueeze(0) * diagonal.unsqueeze(-1)
        rhs = torch.bmm(
            free_jacobian.transpose(1, 2), residual.unsqueeze(-1)
        ).squeeze(-1)
        candidate = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)

        violating_low = free & (candidate < lower_delta)
        violating_high = free & (candidate > upper_delta)
        violating = (violating_low | violating_high) & active_batch.unsqueeze(-1)
        has_violation = torch.any(violating, dim=-1)
        newly_solved = active_batch & ~has_violation
        delta = torch.where(
            newly_solved.unsqueeze(-1) & free, candidate, delta
        )
        delta = torch.where(violating_low, lower_delta, delta)
        delta = torch.where(violating_high, upper_delta, delta)
        free = free & ~violating
        solved = solved | newly_solved | ~torch.any(free, dim=-1)

    return delta
