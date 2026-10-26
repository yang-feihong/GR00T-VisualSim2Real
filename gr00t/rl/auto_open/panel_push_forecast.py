"""Small shared panel-push search and joint-safe arc endpoint selection."""

import math

import torch

from gr00t.rl.auto_open.panel_motion import PUSH_TIP_MAX_ANGULAR_LEAD_RAD

PANEL_PUSH_JOINT_MARGIN_RAD = 0.02
PANEL_PUSH_MIN_HINGE_DISTANCE_M = 0.20
PANEL_RADIAL_STRATA = 9
PANEL_HEIGHT_SAMPLES = 9
PANEL_YAW_SAMPLES = 7


def panel_candidate_bank(
    desired,
    orientation,
    normal,
    tangent,
    jaw_local,
    depth_local,
    hinge,
    root,
    center_offset,
    clearance,
    edge,
    bottom,
    top,
    edge_margin,
    comfort,
    handle_bounds,
    hinge_sign,
    *,
    height_upper_offset_m=0.20,
):
    """Sample the panel/workspace intersection, without seed-dependent point lists."""
    from gr00t.rl.auto_open.motion_math import _quat_multiply, _yaw_quat
    from gr00t.rl.auto_open.panel_motion import panel_fingertip_reference
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate, align_vectors

    desired, orientation = desired.double(), orientation.double()
    normal, tangent, jaw_local = normal.double(), tangent.double(), jaw_local.double()
    depth_local, hinge, root = depth_local.double(), hinge.double(), root.double()
    aligned = align_vectors(_quat_rotate(orientation, depth_local), -normal, orientation)
    yaw = torch.linspace(-30.0, 30.0, PANEL_YAW_SAMPLES, device=desired.device, dtype=desired.dtype)
    quats = _quat_multiply(
        _yaw_quat(hinge_sign.double()[:, None] * torch.deg2rad(yaw)), aligned[:, None]
    )
    # Reduce collider support once per yaw, not once per position/height sample.
    offsets = _quat_rotate(quats[:, :, None], jaw_local[:, None])
    old_offsets = _quat_rotate(orientation[:, None], jaw_local)
    old_min = (old_offsets * normal[:, None]).sum(-1).amin(-1)
    support = (offsets * normal[:, None, None]).sum(-1).amin(-1)
    base = desired[:, None] + (old_min[:, None] - support)[..., None] * normal[:, None]
    tip_offset, tip_surface = panel_fingertip_reference(
        base,
        offsets,
        _quat_rotate(quats, depth_local[:, None]),
        normal[:, None],
        clearance.double()[:, None],
    )
    ee_radius = ((base - hinge[:, None]) * tangent[:, None]).sum(-1)
    tip_radius = ((tip_surface - hinge[:, None]) * tangent[:, None]).sum(-1)
    w, x, y, z = root[:, 3:7].unbind(-1)
    base_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    center = root[:, :3] + _quat_rotate(_yaw_quat(base_yaw), center_offset.double())
    zero = base - ee_radius[..., None] * tangent[:, None]
    relative = center[:, None, :2] - zero[..., :2]
    middle = (relative * tangent[:, None, :2]).sum(-1)
    perpendicular_sq = (relative * relative).sum(-1) - middle.square()
    half_sq = comfort.double()[:, None].square() - perpendicular_sq
    half = half_sq.clamp_min(0.0).sqrt()
    tip_shift = tip_radius - ee_radius
    lower = torch.maximum(
        middle - half,
        edge_margin.double().clamp_min(PANEL_PUSH_MIN_HINGE_DISTANCE_M)[:, None] - tip_shift,
    )
    upper = torch.minimum(middle + half, (edge - edge_margin).double()[:, None] - tip_shift)
    interval_valid = (half_sq >= 0.0) & (upper >= lower)
    radial_bounds = torch.stack((lower, upper), -1)
    upper = torch.maximum(upper, lower)
    fractions = (
        torch.arange(PANEL_RADIAL_STRATA, device=desired.device, dtype=desired.dtype) + 0.5
    ) / PANEL_RADIAL_STRATA
    radial = lower[..., None] + (upper - lower)[..., None] * fractions
    anchor = torch.minimum(torch.maximum(ee_radius, lower), upper)
    radial = torch.cat((anchor[..., None], radial), -1)
    height_samples = PANEL_HEIGHT_SAMPLES + round((height_upper_offset_m - 0.20) / 0.05)
    heights = torch.linspace(
        -0.20, height_upper_offset_m, height_samples, device=desired.device, dtype=desired.dtype
    )
    shift = (radial - ee_radius[..., None])[..., None] * tangent[:, None, None]
    positions = (
        (base[:, :, None, None] + shift[:, :, :, None]).expand(-1, -1, -1, len(heights), -1).clone()
    )
    positions[..., 2] += heights
    surface = (tip_surface[:, :, None, None] + shift[:, :, :, None]).expand_as(positions).clone()
    surface[..., 2] += heights
    rotations = quats[:, :, None, None].expand(*positions.shape[:-1], 4)
    tip_offsets = tip_offset[:, :, None, None].expand_as(positions)
    low = positions[..., 2] + offsets[..., 2].amin(-1)[:, :, None, None]
    high = positions[..., 2] + offsets[..., 2].amax(-1)[:, :, None, None]
    valid = (
        interval_valid[:, :, None, None]
        & (positions[..., 2] >= (bottom + edge_margin).double()[:, None, None, None])
        & (positions[..., 2] <= (top - edge_margin).double()[:, None, None, None])
        & (
            (low >= handle_bounds[:, 1, None, None, None] + 0.01)
            | (high <= handle_bounds[:, 0, None, None, None] - 0.01)
        )
    )
    parameters = torch.zeros_like(positions)
    parameters[..., 0] = radial[..., None] - ee_radius[:, :, None, None]
    parameters[..., 1] = heights
    parameters[..., 2] = yaw[None, :, None, None]

    def flatten(value):
        return value.flatten(1, 3)

    result = dict(
        targets=flatten(positions),
        quats=flatten(rotations),
        tip_offset=flatten(tip_offsets),
        tip_surface=flatten(surface),
        geometry_valid=flatten(valid),
        parameters=flatten(parameters),
    )
    distance = torch.linalg.vector_norm(result["targets"] - desired[:, None], dim=-1)
    # Integer micrometer ranks keep symmetric ties stable across scene origins.
    distance = torch.round(distance * 1.0e6)
    order = result["parameters"][..., 2].abs().argsort(dim=-1, stable=True)
    order = order.gather(1, distance.gather(1, order).argsort(dim=-1, stable=True))
    for name, value in result.items():
        index = order if value.ndim == 2 else order[..., None].expand_as(value)
        result[name] = value.gather(1, index)
    result["radial_bounds"] = radial_bounds
    return result


def select_panel_arc(
    valid,
    lengths,
    angles,
    candidate_valid,
    *,
    minimum_progress_rad=math.radians(5.0),
    prefer_longest=False,
):
    """Select a joint-safe arc with mode-specific progress requirements."""
    samples = torch.arange(valid.shape[-1], device=valid.device)
    active = samples[None, None] < lengths[..., None]
    first_bad = torch.where(active & ~valid, samples, valid.shape[-1]).amin(-1)
    complete = (first_bad >= lengths) & candidate_valid
    # Sample zero is standoff; sample one starts the physical push arc.
    last = (first_bad - 2).clamp(0, angles.shape[-1] - 1)
    safe_angle = angles.gather(-1, last[..., None]).squeeze(-1)
    safe_angle = torch.minimum(
        safe_angle, angles.gather(-1, (lengths - 2).clamp_min(0)[..., None]).squeeze(-1)
    )
    start = angles[..., 0]
    # Reserve the fingertip feedback lead before the first unsafe sample.
    stop = torch.where(
        complete,
        torch.full_like(safe_angle, math.radians(89.0)),
        safe_angle - PUSH_TIP_MAX_ANGULAR_LEAD_RAD,
    )
    minimum_progress = (math.radians(89.0) - start).clamp(0.0, minimum_progress_rad)
    useful = candidate_valid & torch.where(
        complete,
        safe_angle - start >= minimum_progress - 1.0e-6,
        stop - start >= minimum_progress_rad,
    )
    complete &= useful
    score = torch.where(useful, stop, torch.full_like(stop, -torch.inf))
    first_complete = complete.long().argmax(-1)
    selected = torch.where(complete.any(-1), first_complete, score.argmax(-1))
    if prefer_longest:
        # Short push arcs may hand off to the existing retract/base-advance loop.
        progress = torch.where(complete, safe_angle - start, stop - start)
        progress = torch.round(progress * 1.0e6)
        score = torch.where(useful, progress, torch.full_like(progress, -torch.inf))
        selected = score.argmax(-1)
    owner = torch.arange(len(selected), device=selected.device)
    return selected, useful.any(-1), stop[owner, selected], complete[owner, selected]
