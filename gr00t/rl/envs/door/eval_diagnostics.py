import torch

def closest_event_mask(candidate_distance, best_distance):
    """Select strictly closer frames without modifying either input."""
    return candidate_distance < best_distance


def max_acceptance_event_mask(
    candidate_count,
    candidate_distance,
    best_count,
    best_distance,
):
    """Select more simultaneous checks, breaking ties by pocket distance."""
    return (candidate_count > best_count) | (
        (candidate_count == best_count) & (candidate_distance < best_distance)
    )


def update_event_snapshot(snapshot, replace, features, current_error):
    """Update a diagnostic snapshot and attach the next frame to the old event.

    All values are copied into tracker-owned tensors.  Source tensors are never
    mutated, so this helper cannot affect control, rewards, or task state.
    """
    pending = snapshot["pending_next"]
    snapshot["error_next"] = torch.where(
        pending[:, None], current_error, snapshot["error_next"]
    )

    for name, value in features.items():
        target = snapshot[name]
        mask = replace
        while mask.ndim < target.ndim:
            mask = mask.unsqueeze(-1)
        snapshot[name] = torch.where(mask, value, target)

    next_reset = torch.full_like(snapshot["error_next"], float("nan"))
    snapshot["error_next"] = torch.where(
        replace[:, None], next_reset, snapshot["error_next"]
    )
    snapshot["pending_next"] = replace
