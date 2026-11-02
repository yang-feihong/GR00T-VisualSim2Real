"""Screen small GPU candidate cohorts; stop work independently for solved rows."""

import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_dot,
    fixed_norm,
)


def _merge_batch_report(report, diagnostic, rows, batch):
    for key, value in diagnostic.items():
        if isinstance(value, dict):
            _merge_batch_report(report.setdefault(key, {}), value, rows, batch)
        elif not torch.is_tensor(value) or not value.ndim:
            report[key] = value
        else:
            if key not in report:
                fill = torch.nan if value.is_floating_point() else 0
                report[key] = torch.full(
                    (batch, *value.shape[1:]), fill, device=value.device, dtype=value.dtype
                )
            report[key][rows] = value


def compact_grasp_bank(bank):
    """Drop globally ineligible slots, preserving every row's feasible candidates."""
    orientations, goals, quality, coarse, fallback = bank
    if fallback:
        raise ValueError("Compaction requires a single candidate cohort")
    columns = torch.where(quality["pocket_fit"].any(0))[0]
    if not len(columns):
        columns = torch.zeros(1, device=goals.device, dtype=torch.long)
    report = {
        "uncompressed_candidate_count": torch.full(
            (len(goals),), orientations.shape[1], device=goals.device, dtype=torch.long
        ),
        "candidate_source_indices": columns[None].expand(len(goals), -1),
    }
    for key, value in quality.items():
        if value.dtype == torch.bool:
            report[f"uncompressed_geometry_pass_counts.{key}"] = value.sum(1)
    return (
        orientations[:, columns],
        goals[:, columns],
        {key: value[:, columns] for key, value in quality.items()},
        len(columns),
        0,
    ), report


def search_grasp_batches(
    state, bank, evaluate, *, capacity, cohort_size=4, candidate_mask=None, candidate_priority=None
):
    orientations, goals, quality, coarse, fallback = bank
    batch, width = orientations.shape[:2]
    rotation = (orientations * state["ee_orn"][:, None]).sum(-1).abs().clamp_max(1)
    priority = 1 - rotation if candidate_priority is None else candidate_priority
    if priority.shape != quality["pocket_fit"].shape:
        raise ValueError("Candidate priorities must match the grasp bank")
    priority = priority.masked_fill(~quality["pocket_fit"], torch.inf)
    order = priority.argsort(dim=-1, stable=True)
    positions = state["ee_pos"][:, None].expand(-1, capacity, -1).clone()
    quaternions = state["ee_orn"][:, None].expand(-1, capacity, -1).clone()
    lengths = torch.ones(batch, device=goals.device, dtype=torch.long)
    valid = torch.zeros(batch, device=goals.device, dtype=torch.bool)
    checked = torch.zeros(batch, width, device=goals.device, dtype=torch.bool)
    report = {"selected": torch.full((batch,), -1, device=goals.device, dtype=torch.long)}
    available = quality["pocket_fit"]
    if candidate_mask is not None:
        available = available & candidate_mask
    ordered = available.gather(1, order)
    padding = (-width) % cohort_size
    groups = torch.nn.functional.pad(ordered, (0, padding)).reshape(batch, -1, cohort_size)
    cohorts = torch.where(groups.any(dim=(0, 2)))[0]
    offsets = torch.arange(cohort_size, device=goals.device)
    for cohort in cohorts.unbind():
        slots = cohort * cohort_size + offsets
        selected = order[:, slots.clamp_max(width - 1)]
        eligible = available.gather(1, selected) & (slots < width)[None]
        pending = torch.where(~valid & eligible.any(-1))[0]
        if not len(pending):
            continue
        selected, eligible = selected[pending], eligible[pending]
        active = torch.zeros(len(pending), width + 1, device=goals.device, dtype=torch.bool)
        active.scatter_(1, torch.where((slots < width)[None], selected, width), eligible)
        active = active[:, :width]
        local_quality = {key: value[pending] for key, value in quality.items()}
        local_quality["pocket_fit"] = local_quality["pocket_fit"] & active
        checked[pending] |= local_quality["pocket_fit"]
        local_bank = (orientations[pending], goals[pending], local_quality, coarse, fallback)
        path, accepted, diagnostic = evaluate(pending, local_bank)
        _merge_batch_report(report, diagnostic, pending, batch)
        solved = pending[accepted]
        if len(solved):
            n = path[0].shape[1]
            positions[solved, :n] = path[0][accepted]
            quaternions[solved, :n] = path[1][accepted]
            lengths[solved] = path[2][accepted]
            ends = (lengths[solved] - 1).long()
            positions[solved, n:] = positions[solved, ends][:, None]
            quaternions[solved, n:] = quaternions[solved, ends][:, None]
            valid[solved] = True
    report["candidate_cohort_evaluated"] = checked
    report["candidate_cohort_size"] = cohort_size
    report["candidate_orientations_world_wxyz"] = orientations
    report["candidate_grasp_origins_world_m"] = goals
    return (positions, quaternions, lengths), valid, report


def balanced_pocket_priority(bank, reference_orientation):
    """Prefer a central force line and deeper seating before pose similarity."""
    orientations, _, quality, _, _ = bank
    fixed, moving = quality["fixed_contact_local"], quality["moving_contact_local"]
    axis = moving - fixed
    axis = axis / fixed_norm(axis, keepdim=True).clamp_min(1e-8)
    offset = quality["center_local"] - fixed
    lever = fixed_norm(offset - fixed_dot(offset, axis)[..., None] * axis)
    depth = quality["center_depth_margin_m"]
    rotation = 1 - fixed_dot(orientations, reference_orientation[:, None]).abs().clamp_max(1)
    return lever - depth + 0.0001 * rotation, lever, depth
