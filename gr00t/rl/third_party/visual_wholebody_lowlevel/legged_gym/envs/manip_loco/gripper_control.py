from __future__ import annotations

import torch


def advance_gripper_actuator_targets(
    semantic_targets: torch.Tensor,
    actual_positions: torch.Tensor,
    actuator_targets: torch.Tensor,
    hold_active: torch.Tensor,
    finger_forces: torch.Tensor,
    *,
    contact_force_threshold: float = 1.0,
    overload_force_threshold: float = 50.0,
    free_close_step: float = 0.016,
    free_command_lead: float = 0.080,
    contact_close_step: float = 0.008,
    contact_command_lead: float = 0.024,
    overload_release_step: float = 0.001,
    fully_open_position: float = -torch.pi / 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance physical jaw targets while preserving semantic commands.

    Angles increase from the fully-open position toward zero when closing.
    The returned tensors are new actuator state; ``semantic_targets`` is never
    modified or replaced with feedback-derived values.
    """
    if semantic_targets.shape != actual_positions.shape:
        raise ValueError("semantic_targets and actual_positions must have equal shape")
    if actuator_targets.shape != semantic_targets.shape:
        raise ValueError("actuator_targets must match semantic_targets")
    if hold_active.shape != semantic_targets.shape[:1]:
        raise ValueError("hold_active must contain one value per environment")
    if finger_forces.shape != (semantic_targets.shape[0], 2):
        raise ValueError("finger_forces must have shape (num_envs, 2)")

    next_targets = actuator_targets.clone()
    next_hold = hold_active.clone()
    closing = torch.all(semantic_targets >= -1.0e-4, dim=1)
    opening = ~closing
    next_targets[opening] = semantic_targets[opening]
    next_hold[opening] = False

    contacts = finger_forces >= contact_force_threshold
    bilateral = torch.all(contacts, dim=1)
    unilateral = torch.any(contacts, dim=1) & ~bilateral

    newly_latched = closing & bilateral & ~next_hold
    next_targets[newly_latched] = actual_positions[newly_latched]
    next_hold[newly_latched] = True

    lost_contact = closing & next_hold & ~bilateral
    next_hold[lost_contact] = False

    free_closing = closing & ~next_hold
    if torch.any(free_closing):
        step = torch.where(
            unilateral,
            torch.full_like(unilateral, contact_close_step, dtype=actual_positions.dtype),
            torch.full_like(unilateral, free_close_step, dtype=actual_positions.dtype),
        ).unsqueeze(1)
        lead = torch.where(
            unilateral,
            torch.full_like(unilateral, contact_command_lead, dtype=actual_positions.dtype),
            torch.full_like(unilateral, free_command_lead, dtype=actual_positions.dtype),
        ).unsqueeze(1)
        advanced = torch.minimum(next_targets + step, actual_positions + lead)
        advanced = torch.minimum(advanced, semantic_targets)
        next_targets[free_closing] = advanced[free_closing]

    peak_force = torch.max(finger_forces, dim=1).values
    overloaded = closing & next_hold & (peak_force > overload_force_threshold)
    if torch.any(overloaded):
        released = next_targets[overloaded] - overload_release_step
        next_targets[overloaded] = torch.maximum(
            released,
            torch.full_like(released, fully_open_position),
        )

    return next_targets, next_hold
