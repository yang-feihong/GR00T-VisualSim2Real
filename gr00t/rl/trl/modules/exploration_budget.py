# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def normalized_action_variance(
    marginal_std: torch.Tensor,
    max_std: float | torch.Tensor,
) -> torch.Tensor:
    """Return the batch mean variance relative to each action's ceiling."""
    if marginal_std.ndim < 2:
        raise ValueError("marginal_std must include sample and action dimensions")
    if torch.any(marginal_std <= 0.0):
        raise ValueError("marginal_std must be strictly positive")
    ceiling = torch.as_tensor(
        max_std, dtype=marginal_std.dtype, device=marginal_std.device
    )
    if ceiling.numel() not in (1, marginal_std.shape[-1]):
        raise ValueError("max_std must be scalar or contain one value per action")
    if torch.any(ceiling <= 0.0):
        raise ValueError("max_std must be strictly positive")
    try:
        normalized = marginal_std / ceiling
    except RuntimeError as error:
        raise ValueError("max_std does not broadcast over the action dimension") from error
    if torch.any(normalized > 1.0 + 1.0e-6):
        raise ValueError("marginal_std exceeds max_std")
    sample_dims = tuple(range(marginal_std.ndim - 1))
    return normalized.square().mean(dim=sample_dims)


def exploration_variance_budget_loss(
    marginal_std: torch.Tensor,
    max_std: float | torch.Tensor,
    minimum_variance_fraction: float | torch.Tensor,
    dual_weight: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enforce a per-action global exploration budget without state labels.

    The constraint is on the batch mean normalized variance, not each sample.
    Therefore a sufficiently exploratory subset can support low-noise states.
    ``dual_weight`` is a non-negative Lagrange multiplier, not a reward scale.
    """
    mean_variance = normalized_action_variance(marginal_std, max_std)
    target = torch.as_tensor(
        minimum_variance_fraction,
        dtype=marginal_std.dtype,
        device=marginal_std.device,
    )
    weight = torch.as_tensor(
        dual_weight, dtype=marginal_std.dtype, device=marginal_std.device
    )
    for name, value in (("minimum_variance_fraction", target), ("dual_weight", weight)):
        if value.numel() not in (1, marginal_std.shape[-1]):
            raise ValueError(f"{name} must be scalar or contain one value per action")
    if torch.any((target < 0.0) | (target > 1.0)):
        raise ValueError("minimum_variance_fraction must be in [0, 1]")
    if torch.any(weight < 0.0):
        raise ValueError("dual_weight must be non-negative")

    shortfall = torch.relu(target - mean_variance)
    return (weight * shortfall).sum(), mean_variance, shortfall


def exploration_budget_dual_loss(
    log_dual_weight: torch.Tensor,
    mean_variance: torch.Tensor,
    minimum_variance_fraction: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update non-negative multipliers from detached budget violations.

    Minimizing this loss increases a multiplier while its variance budget is
    violated and decreases it while the budget has slack. The policy and dual
    optimizers remain separate by construction.
    """
    if log_dual_weight.shape != mean_variance.shape:
        raise ValueError("log_dual_weight and mean_variance must have equal shapes")
    target = torch.as_tensor(
        minimum_variance_fraction,
        dtype=mean_variance.dtype,
        device=mean_variance.device,
    )
    if target.numel() not in (1, mean_variance.numel()):
        raise ValueError(
            "minimum_variance_fraction must be scalar or contain one value per action"
        )
    if torch.any((target < 0.0) | (target > 1.0)):
        raise ValueError("minimum_variance_fraction must be in [0, 1]")
    dual_weight = torch.nn.functional.softplus(log_dual_weight)
    signed_violation = target - mean_variance.detach()
    return -(dual_weight * signed_violation).sum(), dual_weight
