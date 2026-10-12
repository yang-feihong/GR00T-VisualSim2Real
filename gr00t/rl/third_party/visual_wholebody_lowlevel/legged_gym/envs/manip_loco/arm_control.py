from __future__ import annotations

import torch


def advance_cartesian_joint_transition(
    raw_targets: torch.Tensor,
    limited_targets: torch.Tensor,
    active: torch.Tensor,
    max_step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rate-limit rows entering Cartesian control and retire caught-up rows."""
    if raw_targets.shape != limited_targets.shape:
        raise ValueError("raw_targets and limited_targets must have equal shape")
    if active.shape != raw_targets.shape[:1]:
        raise ValueError("active must contain one value per environment")
    if max_step.shape != raw_targets.shape[1:]:
        raise ValueError("max_step must contain one value per arm joint")

    output = raw_targets.clone()
    next_limited = limited_targets.clone()
    next_active = active.clone()
    if not torch.any(active):
        return output, next_limited, next_active

    delta = raw_targets[active] - limited_targets[active]
    bounded_delta = torch.clamp(delta, min=-max_step, max=max_step)
    advanced = limited_targets[active] + bounded_delta
    output[active] = advanced
    next_limited[active] = advanced
    next_active[active] = ~torch.all(torch.abs(delta) <= max_step, dim=1)
    return output, next_limited, next_active
