"""Batched measured floor-grasp station proposals, without host-side pose selection."""

import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.grasp_clearance import GRASP_CLEARANCE_M
from gr00t.rl.pick.grasp_quality import convex_grasp_quality
from gr00t.rl.pick.planning import MAX_POCKET_SAMPLE_EVALUATIONS
from gr00t.rl.pick.station import station_corridors_clear


def select_measured_station_batch(
    state,
    destination,
    support_points,
    support_mask,
    floor_height,
    commands,
    collision,
    geometry,
    *,
    max_evaluations=MAX_POCKET_SAMPLE_EVALUATIONS,
):
    """Preserve all command/yaw/depth candidates with a fused GPU working-set bound."""
    if max_evaluations < 1:
        raise ValueError("Measured station work budget must be positive")
    root = state["root_states"]
    local_goal, local_quat, sample_id, checkpoint = commands
    batch = len(root)
    offsets = torch.arange(24, device=root.device, dtype=root.dtype) * (torch.pi / 12)
    yaw = offsets[None].expand(len(local_goal), -1).flatten()
    body_quat = _euler_xyz_quat(torch.stack((yaw * 0, yaw * 0, yaw), -1))
    orientations = _quat_multiply(body_quat, local_quat.repeat_interleave(24, 0))
    source = torch.arange(len(local_goal), device=root.device).repeat_interleave(24)
    width = len(orientations)
    # Fused CUDA centers store local points once, not once per pocket depth.
    point_factor = (
        1
        if (
            support_points.is_cuda
            and support_points.dtype == torch.float32
            and root.dtype == torch.float32
        )
        else 17
    )
    scene_capacity = max(1, max_evaluations // (point_factor * support_points.shape[-2]))
    positions = root.new_full((batch, 3), torch.nan)
    selected_pose, selected_q = destination.clone(), state["ee_orn"].clone()
    selected_depth = root.new_zeros(batch)
    selected_id = torch.full((batch,), -1, device=root.device, dtype=torch.long)
    selected_open = root.new_full((batch,), torch.nan)
    selected_close = selected_open.clone()
    best = root.new_full((batch,), torch.inf)
    matched_counts = torch.zeros(batch, device=root.device, dtype=torch.long)
    safe_counts = torch.zeros_like(matched_counts)
    open_counts = torch.zeros_like(matched_counts)
    close_counts = torch.zeros_like(matched_counts)
    corridor_counts = torch.zeros_like(matched_counts)
    # Equal-cost ties retain the original command/yaw/depth order across blocks.
    selected_order = torch.full_like(matched_counts, width * 17)
    for start in range(0, batch, scene_capacity):
        stop = min(start + scene_capacity, batch)
        scenes = torch.arange(start, stop, device=root.device)
        candidate_capacity = max(
            1, max_evaluations // ((stop - start) * point_factor * support_points.shape[-2])
        )
        for first in range(0, width, candidate_capacity):
            last = min(first + candidate_capacity, width)
            q = orientations[None, first:last].expand(stop - start, -1, -1)
            quality = convex_grasp_quality(
                q,
                state["object_state"][start:stop, 3:7],
                support_points[start:stop] if support_points.ndim == 3 else support_points,
                support_mask=(
                    support_mask[start:stop]
                    if support_mask is not None and support_mask.ndim > 1
                    else support_mask
                ),
                return_depth_candidates=True,
                centers_only=True,
            )
            center = quality["center_local"].clone()
            center[..., 2] -= 0.0005
            ee = state["object_state"][start:stop, None, None, :3] - _quat_rotate(
                q[:, :, None], center
            )
            height_error = (
                ee[..., 2]
                - floor_height[start:stop, None, None]
                - local_goal[source[first:last], 2][None, :, None]
            )
            matched = quality["pocket_fit"] & (height_error.abs() <= 0.002)
            owner, index, depth = torch.where(matched)
            owner = scenes[owner]
            if not len(owner):
                continue
            candidate = index + first
            # Height is independent of moving-jaw closure. Run the expensive
            # contact solve only for exact height matches, retaining every depth.
            quality = convex_grasp_quality(
                orientations[candidate, None],
                state["object_state"][owner, 3:7],
                support_points[owner] if support_points.ndim == 3 else support_points,
                support_mask=(
                    support_mask[owner]
                    if support_mask is not None and support_mask.ndim > 1
                    else support_mask
                ),
                return_depth_candidates=True,
                depth_fractions=(depth.to(root.dtype) / 16)[:, None, None],
            )
            contact_valid = quality["pocket_fit"][:, 0, 0] & (
                quality["antipodal_margin"][:, 0, 0] >= 0
            )
            goal = ee[owner - start, index, depth][contact_valid]
            aperture = quality["aperture_angle_rad"][:, 0, 0][contact_valid]
            owner, candidate, depth = (value[contact_valid] for value in (owner, candidate, depth))
            matched_counts.index_add_(0, owner, torch.ones_like(owner))
            if not len(owner):
                continue
            xy = (
                goal[:, :2]
                - _quat_rotate(body_quat[candidate], local_goal[source[candidate]])[:, :2]
            )
            station = torch.cat((xy, yaw[candidate, None]), -1)
            opened = root.new_empty(len(owner))
            closing = torch.empty_like(opened)
            corridor = torch.empty(len(owner), device=root.device, dtype=torch.bool)
            query_capacity = max(1, max_evaluations // (40 * 3 * 32))
            for query_start in range(0, len(owner), query_capacity):
                sl = slice(query_start, query_start + query_capacity)
                selected_geometry = geometry.select(owner[sl])
                p, orn = goal[sl, None], orientations[candidate[sl], None]
                opened[sl] = collision.gripper_clearance(
                    selected_geometry, p, orn, maximum_distance_m=GRASP_CLEARANCE_M + 0.0001
                )[:, 0]
                closing[sl] = collision.gripper_closing_clearance(
                    selected_geometry,
                    p,
                    orn,
                    aperture[sl, None],
                    maximum_distance_m=GRASP_CLEARANCE_M + 0.0001,
                )[:, 0]
                corridor[sl] = station_corridors_clear(
                    root[owner[sl]], station[sl, None], selected_geometry
                )[:, 0]
            feasible = (opened >= GRASP_CLEARANCE_M) & (closing >= GRASP_CLEARANCE_M) & corridor
            open_counts.index_add_(0, owner, (opened >= GRASP_CLEARANCE_M).long())
            close_counts.index_add_(0, owner, (closing >= GRASP_CLEARANCE_M).long())
            corridor_counts.index_add_(0, owner, corridor.long())
            safe_counts.index_add_(0, owner, feasible.long())
            distance = (
                (station[:, :2] - root[owner, :2]).norm(dim=-1).masked_fill(~feasible, torch.inf)
            )
            block_best = root.new_full((batch,), torch.inf)
            block_best.scatter_reduce_(0, owner, distance, reduce="amin")
            order = candidate * 17 + depth
            eligible = torch.isfinite(distance) & (distance == block_best[owner])
            block_order = torch.full_like(selected_order, width * 17)
            block_order.scatter_reduce_(
                0, owner, order.masked_fill(~eligible, width * 17), reduce="amin"
            )
            selected = (
                eligible
                & (order == block_order[owner])
                & (
                    (distance < best[owner])
                    | ((distance == best[owner]) & (order < selected_order[owner]))
                )
            )
            row, item = owner[selected], torch.where(selected)[0]
            best[row], selected_order[row] = distance[item], order[item]
            selected_pose[row], selected_q[row] = station[item], orientations[candidate[item]]
            selected_depth[row] = depth[item].to(root.dtype) / 16
            selected_id[row] = sample_id[source[candidate[item]]]
            positions[row] = goal[item]
            selected_open[row], selected_close[row] = opened[item], closing[item]
    valid = torch.isfinite(best)
    return (
        selected_pose,
        selected_q,
        selected_depth,
        valid,
        {
            "selection_policy": "shortest_scene_clear_measured_low_command",
            "selected_station": selected_pose,
            "selected_orientation": selected_q,
            "selected_sample_id": selected_id,
            "selected_pocket_depth_preference": selected_depth,
            "selected_grasp_goal_world_m": positions,
            "selected_open_clearance_m": selected_open,
            "selected_closing_clearance_m": selected_close,
            "scene_safe_candidate_count": safe_counts,
            "matched_geometry_count": matched_counts,
            "open_clear_candidate_count": open_counts,
            "closing_clear_candidate_count": close_counts,
            "corridor_clear_candidate_count": corridor_counts,
            "measured_command_count": len(local_goal),
            "checkpoint_sha256": checkpoint,
            "not_a_physical_grasp_guarantee": True,
        },
    )
