"""GPU-batched constrained damped least-squares helpers."""

from __future__ import annotations

import torch


def _fixed_order_spd_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve a small SPD batch without batch-shape-dependent library kernels."""
    batch_size, size, width = matrix.shape
    if size != width or rhs.shape != (batch_size, size):
        raise ValueError("SPD solve shapes do not agree")

    lower = torch.zeros_like(matrix)
    for row in range(size):
        for column in range(row + 1):
            value = matrix[:, row, column]
            for inner in range(column):
                value = value - lower[:, row, inner] * lower[:, column, inner]
            if row == column:
                lower[:, row, column] = torch.sqrt(value.clamp_min(1.0e-12))
            else:
                lower[:, row, column] = value / lower[:, column, column]

    forward = torch.zeros_like(rhs)
    for row in range(size):
        value = rhs[:, row]
        for column in range(row):
            value = value - lower[:, row, column] * forward[:, column]
        forward[:, row] = value / lower[:, row, row]

    solution = torch.zeros_like(rhs)
    for row in range(size - 1, -1, -1):
        value = forward[:, row]
        for column in range(row + 1, size):
            value = value - lower[:, column, row] * solution[:, column]
        solution[:, row] = value / lower[:, row, row]
    return solution


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
        fixed_effect = torch.zeros_like(pose_error)
        for joint in range(joint_count):
            fixed_effect = fixed_effect + (
                jacobian[:, :, joint] * fixed_delta[:, joint].unsqueeze(-1)
            )
        residual = pose_error - fixed_effect
        free_jacobian = jacobian * free_float.unsqueeze(1)
        normal = torch.zeros(
            batch_size,
            joint_count,
            joint_count,
            device=jacobian.device,
            dtype=jacobian.dtype,
        )
        rhs = torch.zeros_like(joint_position)
        for task_axis in range(jacobian.shape[1]):
            row = free_jacobian[:, task_axis, :]
            normal = normal + row.unsqueeze(-1) * row.unsqueeze(-2)
            rhs = rhs + row * residual[:, task_axis].unsqueeze(-1)
        diagonal = (
            free_float * (float(damping) ** 2) + (~free).to(jacobian.dtype)
        )
        normal = normal + eye.unsqueeze(0) * diagonal.unsqueeze(-1)
        candidate = _fixed_order_spd_solve(normal, rhs)

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
