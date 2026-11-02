"""Closed-jaw basket repositioning in bounded, scene-local GPU request pools."""

import math

import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.basket_access import shifted_grasp_counts_batch
from gr00t.rl.pick.basket_inverse import safe_inverse_paths_batch
from gr00t.rl.pick.basket_nudge import (
    pose_segments,
    pose_segments_batch,
)
from gr00t.rl.pick.basket_nudge_geometry import closed_nudge_geometry_precheck
from gr00t.rl.pick.pocket import GRIP_STATIC_FRICTION

MAX_NUDGE_PATH_POSES = 65536


def plan_closed_nudge_batch(
    state,
    sizes,
    intrinsic_sizes,
    payload_mask,
    basket,
    limits,
    collision,
    geometry,
    target_geometry,
    target_surface,
    dt,
    *,
    support_points,
    support_mask,
    current_path_unavailable,
    contact_height_fractions=(0.35, 0.70, 0.90),
    shift_samples=24,
    capacity=4096,
    require_sliding_stability=False,
    object_centers_local=None,
):
    root, start = state["root_states"], state["ee_pos"]
    batch = len(root)
    precheck = closed_nudge_geometry_precheck(
        state,
        sizes,
        payload_mask,
        basket,
        target_surface,
        support_points=support_points,
        support_mask=support_mask,
        object_centers_local=object_centers_local,
        contact_height_fractions=contact_height_fractions,
        shift_samples=shift_samples,
        require_sliding_stability=require_sliding_stability,
    )
    shifts, before, after, sweep_space, good, bbox_good, improving, direction = (
        precheck[key]
        for key in (
            "shifts",
            "before",
            "after",
            "sweep_space",
            "good",
            "bbox_good",
            "improving",
            "direction",
        )
    )
    sliding_precheck = precheck["sliding_precheck"]
    probe_count = precheck["sliding_probe_count"]
    target = state["object_state"][:, :3]
    proposed = torch.cat((target[:, None], target[:, None] + shifts[None, :, None] * direction), 1)
    access, access_report = shifted_grasp_counts_batch(
        state,
        intrinsic_sizes,
        proposed,
        collision,
        geometry,
        support_points,
        limits,
        support_mask=support_mask,
        target_mask=torch.cat((good.any(-1, keepdim=True), good), 1),
    )
    worthwhile = (
        good
        & (access[:, 1:] > 0)
        & ((access[:, :1] == 0) | improving | current_path_unavailable[:, None])
    )
    not_required = (before >= 0.006) & ~current_path_unavailable
    report = {
        "space_before_m": before,
        "space_after_by_shift_m": after,
        "swept_minimum_space_by_shift_m": sweep_space,
        "good_direction_mask": good,
        "bbox_good_direction_mask": bbox_good,
        "sliding_precheck_evaluated_contacts": probe_count,
        "sliding_stability_unique_contacts": probe_count,
        "sliding_stability_expanded_candidates": 0,
        "clearance_query_caps_m": {"scene": 0.0031, "target_entry": 0.0021, "arm": 0.0021},
        "worthwhile_direction_mask": worthwhile,
        "current_access_evaluated": good.any(-1),
        "current_geometric_grasp_count": access[:, 0].masked_fill(~good.any(-1), -1),
        "predicted_grasp_counts_by_shift": access[:, 1:],
        "current_grasp_path_known_unavailable": current_path_unavailable,
        "candidate_shift_distances_m": shifts[None].expand(batch, -1),
        "not_required": not_required,
        "sliding_stability_required": require_sliding_stability,
        **{f"access.{key}": value for key, value in access_report.items()},
    }
    shift_owner, shift_id = torch.where(worthwhile)
    failure = torch.where(bbox_good.any(-1), 2, 1)
    if not len(shift_owner):
        report["planning_failure_code"] = failure
        return (
            (
                start[:, None],
                state["ee_orn"][:, None],
                torch.ones(batch, device=root.device, dtype=torch.long),
            ),
            torch.zeros_like(not_required),
            report,
        )
    combinations = torch.cartesian_prod(
        torch.arange(4, device=root.device),
        torch.arange(8, device=root.device),
        torch.arange(len(contact_height_fractions), device=root.device),
    )
    width = len(combinations)
    owner, chosen_shift = shift_owner.repeat_interleave(width), shift_id.repeat_interleave(width)
    combo = combinations.repeat(len(shift_owner), 1)
    d = direction[owner, chosen_shift]
    tilt = root.new_tensor([0, 20, 35, 50])[combo[:, 0]] * (math.pi / 180)
    yaw = torch.atan2(-d[:, 1], -d[:, 0])
    rotation = _euler_xyz_quat(
        torch.stack((combo[:, 1].to(root.dtype) * (math.pi / 4), math.pi / 2 - tilt, yaw), -1)
    )
    origins = target[owner] - 0.20 * d
    origins[:, 2] = (
        target[owner, 2]
        + sizes[owner, 0, 2] / 2 * root.new_tensor(contact_height_fractions)[combo[:, 2]]
    )
    hits, normal, _ = target_surface.select(owner).rays(
        origins[:, None], (origins + 0.40 * d)[:, None]
    )
    hits, normal = hits[:, 0], normal[:, 0]
    hit_ok = normal.norm(dim=-1) > 0.9
    fixed, moving = collision._gripper_local_surfaces(start)
    closed = torch.cat(
        (
            fixed,
            _quat_rotate(
                root.new_tensor([math.cos(math.pi / 6), 0, math.sin(math.pi / 6), 0]), moving
            ),
        )
    )
    tip = closed[closed[:, 0] >= closed[:, 0].max() - 0.01]
    rotated = _quat_rotate(rotation[:, None], tip[None])
    leading = (rotated * d[:, None]).sum(-1).argmax(-1)
    pool = torch.arange(len(owner), device=root.device)
    contact = hits - rotated[pool, leading]
    pre = contact - d * 0.008
    end = contact + d * shifts[chosen_shift, None]
    world, _ = basket.world_mesh(root)
    hover = pre.clone()
    hover[:, 2] = torch.maximum(hover[:, 2], world[..., 2].amax(-1)[owner] + 0.16)
    knots = torch.stack((start[owner], hover, pre, end, hover), 1)
    rotations = torch.cat((state["ee_orn"][owner, None], rotation[:, None].expand(-1, 4, -1)), 1)
    scene_gap, entry_gap = (
        root.new_full((len(owner),), torch.nan),
        root.new_full((len(owner),), torch.nan),
    )
    block_size = max(1, MAX_NUDGE_PATH_POSES // 36)
    for begin in range(0, len(owner), block_size):
        stop = min(begin + block_size, len(owner))
        p, q, bounds = pose_segments(knots[begin:stop], rotations[begin:stop], dt, samples=9)
        source = owner[begin:stop]
        scene_gap[begin:stop] = collision.gripper_pose_clearance(
            geometry.select(source),
            p,
            q,
            p.new_zeros(p.shape[:2]),
            maximum_distance_m=0.0031,
        ).amin(-1)
        entry_gap[begin:stop] = collision.gripper_pose_clearance(
            target_geometry.select(source),
            p[:, : bounds[2]],
            q[:, : bounds[2]],
            p.new_zeros(len(p), bounds[2]),
            maximum_distance_m=0.0021,
        ).amin(-1)
    geometric = hit_ok & (scene_gap >= 0.003) & (entry_gap >= 0.002)
    stability_fields = ()
    if require_sliding_stability:
        friction = (GRIP_STATIC_FRICTION + basket.floor_static_friction) / 2
        surface_key = (owner * len(shifts) + chosen_shift) * len(contact_height_fractions) + combo[
            :, 2
        ]
        stable, reserve, radius, height = (field[surface_key] for field in sliding_precheck)
        report["sliding_stability_unique_contacts"] = probe_count
        report["sliding_stability_expanded_candidates"] = len(owner)
        stable_count = torch.zeros(batch, device=root.device, dtype=torch.long)
        stable_count.scatter_add_(0, owner, (geometric & stable).long())
        report["sliding_stable_candidate_count"] = stable_count
        report["effective_floor_static_friction"] = friction
        stability_fields = (
            ("sliding_stable_mask", stable, False),
            ("tipping_lever_reserve_m", reserve, torch.nan),
            ("support_footprint_radius_m", radius, torch.nan),
            ("effective_contact_height_m", height, torch.nan),
        )
        geometric &= stable
    eligible = torch.where(geometric)[0]
    robust = torch.zeros_like(geometric)
    travel = root.new_full((len(owner),), torch.inf)
    residual = travel.clone()
    arm_gap = travel.clone()
    for begin in range(0, len(eligible), block_size):
        ids = eligible[begin : begin + block_size]
        p, q, _ = pose_segments(knots[ids], rotations[ids], dt, samples=9)
        source = owner[ids]
        safe, distance, diagnostic = safe_inverse_paths_batch(
            {key: state[key][source] for key in ("root_states", "dof_pos")},
            p[:, None],
            q[:, None],
            limits,
            collision,
            geometry.select(source),
            maximum_arm_query_distance_m=0.0021,
        )
        robust[ids], travel[ids] = safe[:, 0], distance[:, 0]
        residual[ids] = diagnostic["best_max_position_error_by_height_m"][:, 0, 0]
        arm_gap[ids] = diagnostic["best_minimum_arm_clearance_by_height_m"][:, 0, 0]
    cost = -after[owner, chosen_shift] + 0.001 * travel + shifts[chosen_shift]
    best = root.new_full((batch,), torch.inf)
    best.scatter_reduce_(0, owner, cost.masked_fill(~robust, torch.inf), reduce="amin")
    selected = torch.full((batch,), len(owner), device=root.device, dtype=torch.long)
    selected.scatter_reduce_(
        0, owner, torch.where(robust & (cost == best[owner]), pool, len(owner)), reduce="amin"
    )
    found = selected < len(owner)
    chosen = selected.clamp_max(len(owner) - 1)
    path, timed, boundaries = pose_segments_batch(
        knots[chosen], rotations[chosen], dt, contact_segment=2, capacity=capacity
    )
    dense_gap = root.new_full((batch,), torch.inf)
    frame_owner, frame = torch.where(
        (torch.arange(capacity, device=root.device)[None] < path[2][:, None])
        & (found & timed)[:, None]
    )
    if len(frame_owner):
        gap = collision.gripper_pose_clearance(
            geometry.select(frame_owner),
            path[0][frame_owner, frame, None],
            path[1][frame_owner, frame, None],
            root.new_zeros(len(frame_owner), 1),
            maximum_distance_m=0.0031,
        )[:, 0]
        dense_gap.scatter_reduce_(0, frame_owner, gap, reduce="amin")
    valid = found & timed & (dense_gap >= 0.003)
    geometric_count = torch.zeros(batch, device=root.device, dtype=torch.long)
    geometric_count.scatter_add_(0, owner, geometric.long())
    failure = torch.where(worthwhile.any(-1), torch.where(geometric_count > 0, 4, 3), failure)
    failure = torch.where(found & ~timed, 5, failure)
    failure = torch.where(found & timed & ~valid, 6, failure)
    failure = torch.where(valid, 0, failure)
    bank_index = chosen_shift * width + pool.remainder(width)
    full_width = len(shifts) * width
    for name, values, fill in (
        ("surface_hit_mask", hit_ok, False),
        ("geometric_candidate_mask", geometric, False),
        ("robust_inverse_mask", robust, False),
        ("scene_clearance_m", scene_gap, torch.nan),
        ("target_entry_clearance_m", entry_gap, torch.nan),
        ("candidate_cost", cost, torch.inf),
        ("best_max_position_error_m", residual, torch.nan),
        ("best_minimum_arm_clearance_m", arm_gap, torch.nan),
    ) + stability_fields:
        field = torch.full((batch, full_width), fill, device=root.device, dtype=values.dtype)
        field[owner, bank_index] = values
        report[name] = field
    report.update(
        planning_failure_code=failure,
        direction_world=torch.where(valid[:, None], d[chosen], 0),
        shift_m=torch.where(valid, shifts[chosen_shift[chosen]], 0),
        selected_bank_index=torch.where(valid, bank_index[chosen], -1),
        candidate_count=worthwhile.sum(-1) * width,
        selected_predicted_grasp_count=torch.where(
            valid, access[owner[chosen], chosen_shift[chosen] + 1], 0
        ),
        space_after_m=torch.where(valid, after[owner[chosen], chosen_shift[chosen]], before),
        closed_gripper_clearance_m=dense_gap,
        segment_boundaries=torch.where(valid[:, None], boundaries, -1),
        dense_geometry_checked=found & timed,
        contact_target_world=torch.where(valid[:, None], hits[chosen], torch.nan),
        selected_tool_tilt_deg=torch.where(valid, tilt[chosen] * (180 / math.pi), torch.nan),
        selected_tool_roll_deg=torch.where(valid, combo[chosen, 1] * 45, -1),
        selected_contact_height_fraction=torch.where(
            valid, root.new_tensor(contact_height_fractions)[combo[chosen, 2]], torch.nan
        ),
        shift_selection_policy="future_open_and_closing_jaw_clearance_before_bbox_gap",
    )
    path = (
        torch.where(valid[:, None, None], path[0], start[:, None]),
        torch.where(valid[:, None, None], path[1], state["ee_orn"][:, None]),
        torch.where(valid, path[2], 1),
    )
    return path, valid, report
