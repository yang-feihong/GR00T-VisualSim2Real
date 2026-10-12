# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
from torch.distributions import Normal


def _sqrt_nonnegative_with_zero_boundary_gradient(value: torch.Tensor) -> torch.Tensor:
    """Return an exact square root with a defined zero derivative at zero."""
    positive = value > 0.0
    safe_value = torch.where(positive, value, torch.ones_like(value))
    return torch.sqrt(safe_value) * positive.to(value.dtype)


def validate_temporal_noise_parameters(
    rho: float, correlated_fraction: float | torch.Tensor
) -> None:
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    fraction = torch.as_tensor(correlated_fraction)
    if torch.any((fraction < 0.0) | (fraction >= 1.0)):
        raise ValueError(
            "correlated_fraction must be in [0, 1), got "
            f"{correlated_fraction}"
        )


def advance_standardized_ar1_context(
    context: torch.Tensor | None,
    dones: torch.Tensor,
    innovation: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """Advance a stationary unit-variance AR(1) context.

    ``dones`` refers to the previous environment step. A completed environment
    receives a fresh stationary context instead of inheriting its old episode.
    """
    validate_temporal_noise_parameters(rho, 0.0)
    if innovation.ndim < 1:
        raise ValueError("innovation must include an environment dimension")
    if dones.shape != innovation.shape[:-1]:
        raise ValueError(
            "dones must match innovation without its action dimension, got "
            f"dones={tuple(dones.shape)}, innovation={tuple(innovation.shape)}"
        )
    if context is None:
        return innovation
    if context.shape != innovation.shape:
        raise ValueError(
            f"context and innovation shapes differ: {context.shape} vs "
            f"{innovation.shape}"
        )

    next_context = rho * context + math.sqrt(1.0 - rho * rho) * innovation
    return torch.where(dones[..., None].bool(), innovation, next_context)


def conditional_gaussian_parameters(
    base_mean: torch.Tensor,
    marginal_std: torch.Tensor,
    standardized_context: torch.Tensor,
    correlated_fraction: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Gaussian parameters conditional on parameter-independent context.

    Marginalizing a stationary unit-normal context preserves ``marginal_std``:
    the context carries ``fraction`` of the variance and independent action
    noise carries the remainder.
    """
    validate_temporal_noise_parameters(0.0, correlated_fraction)
    if base_mean.shape != marginal_std.shape:
        raise ValueError(
            f"base_mean and marginal_std shapes differ: {base_mean.shape} vs "
            f"{marginal_std.shape}"
        )
    if base_mean.shape != standardized_context.shape:
        raise ValueError(
            "base_mean and standardized_context shapes differ: "
            f"{base_mean.shape} vs {standardized_context.shape}"
        )
    if torch.any(marginal_std <= 0.0):
        raise ValueError("marginal_std must be strictly positive")

    fraction = torch.as_tensor(
        correlated_fraction,
        dtype=marginal_std.dtype,
        device=marginal_std.device,
    )
    correlated_scale = _sqrt_nonnegative_with_zero_boundary_gradient(fraction)
    independent_scale = torch.sqrt(1.0 - fraction)
    conditional_mean = (
        base_mean + marginal_std * correlated_scale * standardized_context
    )
    conditional_std = marginal_std * independent_scale
    return conditional_mean, conditional_std


def conditional_gaussian_log_prob(
    actions: torch.Tensor,
    base_mean: torch.Tensor,
    marginal_std: torch.Tensor,
    standardized_context: torch.Tensor,
    correlated_fraction: float | torch.Tensor,
) -> torch.Tensor:
    conditional_mean, conditional_std = conditional_gaussian_parameters(
        base_mean,
        marginal_std,
        standardized_context,
        correlated_fraction,
    )
    return Normal(conditional_mean, conditional_std).log_prob(actions).sum(dim=-1)


def marginal_temporal_gaussian_parameters(
    actions: torch.Tensor,
    base_mean: torch.Tensor,
    marginal_std: torch.Tensor,
    dones: torch.Tensor,
    rho: float,
    correlated_fraction: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact per-step conditionals after marginalizing the AR(1) state.

    The leading dimensions are ``[batch, time, action]``. ``dones[:, t]``
    terminates the episode after action ``t``, so the filter returns to its
    stationary prior before ``t + 1``.
    """
    validate_temporal_noise_parameters(rho, correlated_fraction)
    if actions.ndim != 3:
        raise ValueError("actions must have shape [batch, time, action]")
    if base_mean.shape != actions.shape or marginal_std.shape != actions.shape:
        raise ValueError("actions, base_mean, and marginal_std must have equal shapes")
    if dones.shape != actions.shape[:2]:
        raise ValueError("dones must have shape [batch, time]")
    if torch.any(marginal_std <= 0.0):
        raise ValueError("marginal_std must be strictly positive")

    fraction = torch.as_tensor(
        correlated_fraction,
        dtype=actions.dtype,
        device=actions.device,
    )
    try:
        fraction = torch.broadcast_to(fraction, actions.shape)
    except RuntimeError as error:
        raise ValueError(
            "correlated_fraction must broadcast to [batch, time, action]"
        ) from error
    batch_size, num_steps, num_actions = actions.shape
    prior_mean = torch.zeros(
        batch_size, num_actions, dtype=actions.dtype, device=actions.device
    )
    prior_variance = torch.ones_like(prior_mean)
    conditional_means = []
    conditional_stds = []

    for step in range(num_steps):
        step_fraction = fraction[:, step]
        sqrt_fraction = _sqrt_nonnegative_with_zero_boundary_gradient(step_fraction)
        observation_variance = 1.0 - step_fraction
        if step > 0:
            reset = dones[:, step - 1, None].bool()
            prior_mean = torch.where(reset, torch.zeros_like(prior_mean), prior_mean)
            prior_variance = torch.where(
                reset, torch.ones_like(prior_variance), prior_variance
            )

        residual_mean = sqrt_fraction * prior_mean
        residual_variance = step_fraction * prior_variance + observation_variance
        conditional_means.append(
            base_mean[:, step] + marginal_std[:, step] * residual_mean
        )
        conditional_stds.append(
            marginal_std[:, step] * torch.sqrt(residual_variance)
        )

        standardized_action = (
            actions[:, step] - base_mean[:, step]
        ) / marginal_std[:, step]
        kalman_gain = prior_variance * sqrt_fraction / residual_variance
        posterior_mean = prior_mean + kalman_gain * (
            standardized_action - residual_mean
        )
        posterior_variance = prior_variance * (
            1.0 - kalman_gain * sqrt_fraction
        )
        prior_mean = rho * posterior_mean
        prior_variance = (
            rho * rho * posterior_variance + (1.0 - rho * rho)
        )

    return torch.stack(conditional_means, dim=1), torch.stack(
        conditional_stds, dim=1
    )


def marginal_temporal_gaussian_log_prob(
    actions: torch.Tensor,
    base_mean: torch.Tensor,
    marginal_std: torch.Tensor,
    dones: torch.Tensor,
    rho: float,
    correlated_fraction: float | torch.Tensor,
) -> torch.Tensor:
    conditional_mean, conditional_std = marginal_temporal_gaussian_parameters(
        actions,
        base_mean,
        marginal_std,
        dones,
        rho,
        correlated_fraction,
    )
    return Normal(conditional_mean, conditional_std).log_prob(actions).sum(dim=-1)


def paired_temporal_gaussian_parameters(
    actions: torch.Tensor,
    old_base_mean: torch.Tensor,
    old_marginal_std: torch.Tensor,
    old_correlated_fraction: float | torch.Tensor,
    new_base_mean: torch.Tensor,
    new_marginal_std: torch.Tensor,
    new_correlated_fraction: float | torch.Tensor,
    dones: torch.Tensor,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter old and new temporal policies with their own parameters."""
    old_conditional_mean, old_conditional_std = (
        marginal_temporal_gaussian_parameters(
            actions,
            old_base_mean,
            old_marginal_std,
            dones,
            rho,
            old_correlated_fraction,
        )
    )
    new_conditional_mean, new_conditional_std = (
        marginal_temporal_gaussian_parameters(
            actions,
            new_base_mean,
            new_marginal_std,
            dones,
            rho,
            new_correlated_fraction,
        )
    )
    return (
        old_conditional_mean,
        old_conditional_std,
        new_conditional_mean,
        new_conditional_std,
    )
