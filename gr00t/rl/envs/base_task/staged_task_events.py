import torch


def fixed_stage_for_mode(
    training_stage: int | None,
    evaluation_stage: int | None,
    is_evaluating: bool,
) -> int | None:
    """Select the explicit fixed stage for the current rollout mode."""
    return evaluation_stage if is_evaluating else training_stage


def compute_stage_advance_event(
    stage_buf: torch.Tensor,
    stage_advance_conditions: list[torch.Tensor],
) -> torch.Tensor:
    event = torch.zeros_like(stage_buf, dtype=torch.bool)
    for stage_idx, condition in enumerate(stage_advance_conditions):
        if condition.shape != stage_buf.shape or condition.dtype != torch.bool:
            raise ValueError(
                "stage advance conditions must be boolean tensors shaped like stage_buf"
            )
        event |= condition & (stage_buf == stage_idx)
    return event


def mask_stage_advance_event(
    event: torch.Tensor,
    reset: torch.Tensor,
    just_reset: torch.Tensor,
) -> torch.Tensor:
    return event & ~reset.bool() & ~just_reset


def advance_stage_for_next_observation(
    stage: torch.Tensor,
    event: torch.Tensor,
    reset: torch.Tensor,
    just_reset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve stage transitions before constructing the next policy observation."""
    if event.shape != stage.shape:
        raise ValueError("stage advance event must have the same shape as stage")
    advance = mask_stage_advance_event(event, reset, just_reset)
    next_stage = stage.clone()
    next_stage[advance] += 1
    return advance, next_stage


def stage_completion_event(
    stage_advance: torch.Tensor,
    final_stage_complete: torch.Tensor,
) -> torch.Tensor:
    return stage_advance | final_stage_complete
