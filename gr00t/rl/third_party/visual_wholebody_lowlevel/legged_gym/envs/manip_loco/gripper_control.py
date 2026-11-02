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
    free_close_step: float = 0.016,
    free_command_lead: float = 0.080,
    contact_close_step: float = 0.008,
    contact_command_lead: float = 0.024,
    retain_contact_preload: bool = False,
    hold_contact_force_threshold: float | None = None,
    prevent_reopening_while_closing: bool = False,
    maintain_contact_preload: bool = False,
    hold_command_lead: float | None = None,
    bound_contact_preload: bool = False,
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
    preload_lead = contact_command_lead if hold_command_lead is None else hold_command_lead
    next_hold = hold_active.clone()
    closing = torch.all(semantic_targets >= -1.0e-4, dim=1)
    opening = ~closing
    next_targets[opening] = semantic_targets[opening]
    next_hold[opening] = False

    contacts = finger_forces >= contact_force_threshold
    bilateral = torch.all(contacts, dim=1)
    unilateral = torch.any(contacts, dim=1) & ~bilateral

    newly_latched = closing & bilateral & ~next_hold
    if retain_contact_preload:
        retained = torch.minimum(
            next_targets[newly_latched], actual_positions[newly_latched] + preload_lead
        )
        next_targets[newly_latched] = torch.maximum(actual_positions[newly_latched], retained)
    else:
        next_targets[newly_latched] = actual_positions[newly_latched]
    next_hold[newly_latched] = True

    holding_contact = (
        bilateral
        if hold_contact_force_threshold is None
        else torch.all(finger_forces >= hold_contact_force_threshold, dim=1)
    )
    lost_contact = closing & next_hold & ~holding_contact
    next_hold[lost_contact] = False
    if maintain_contact_preload:
        held = closing & next_hold
        preloaded = torch.minimum(actual_positions + preload_lead, semantic_targets)
        next_targets[held] = (
            preloaded[held]
            if bound_contact_preload
            else torch.maximum(next_targets[held], preloaded[held])
        )

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

    if prevent_reopening_while_closing:
        next_targets = torch.where(
            closing[:, None], torch.maximum(next_targets, actuator_targets), next_targets
        )
    return next_targets, next_hold
