"""Batch open-jaw release paths using the shared inverse, geometry and path cache."""

import torch

from gr00t.rl.auto_open.motion_math import _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.auto_open.pose_planning import best_feasible
from gr00t.rl.pick.basket_inverse import safe_inverse_paths_batch
from gr00t.rl.pick.basket_nudge import (
    monotone_release_mask,
    pose_segments,
    pose_segments_batch,
    transverse_tilts,
)
from gr00t.rl.utils.b2z1_asset import read_arm_mount


def plan_clear_for_nudge_batch(
    state,
    basket,
    limits,
    collision,
    geometry,
    target_geometry,
    target_surface,
    dt,
    *,
    capacity=4096,
):
    start, root = state["ee_pos"], state["root_states"]
    batch = len(start)
    cone, rotation = transverse_tilts(start)
    offsets = start.new_tensor([0, 0.04, 0.08, 0.12])
    bank = torch.cat(
        (cone.repeat_interleave(len(offsets), 0), offsets.repeat(len(cone))[:, None]), -1
    )
    candidates = len(bank)
    end_q = _quat_multiply(
        state["ee_orn"][:, None], rotation.repeat_interleave(len(offsets), 0)[None]
    )
    mount = root[:, :3] + _quat_rotate(root[:, 3:7], start.new_tensor(read_arm_mount()[0]))
    direction = mount - start
    direction[:, 2] = 0
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    world, _ = basket.world_mesh(root)
    end = start[:, None] + bank[None, :, 2, None] * direction[:, None]
    end[..., 2] = torch.maximum(start[:, 2], world[..., 2].amax(-1) + 0.16)[:, None]
    knots = torch.stack((start[:, None].expand_as(end), end), 2)
    rotations = torch.stack((state["ee_orn"][:, None].expand_as(end_q), end_q), 2)
    p, q, _ = pose_segments(knots.flatten(0, 1), rotations.flatten(0, 1), dt, samples=17)
    p, q = p.reshape(batch, candidates, -1, 3), q.reshape(batch, candidates, -1, 4)
    local = torch.cat(collision._gripper_local_surfaces(start))

    def release(g, surface, points, quats):
        scene = collision.gripper_clearance(g, points, quats, maximum_distance_m=0.0021)
        target = surface.rigid_surface_clearance(
            points, quats, local, check_rays=False, signed_distance=True
        )
        return scene, target

    scene, target = release(geometry, target_surface, p, q)
    geometric = (scene.amin(-1) >= 0.002) & monotone_release_mask(target)
    inverse, travel, inverse_report = safe_inverse_paths_batch(
        state,
        p,
        q,
        limits,
        collision,
        target_geometry,
        candidate_mask=geometric,
        maximum_arm_query_distance_m=0.0021,
    )
    cost = bank[None, :, 0] + 3 * bank[None, :, 2] + 0.001 * travel
    selected, found = best_feasible(geometric & inverse, cost)
    rows = torch.arange(batch, device=start.device)
    path, timed, boundaries = pose_segments_batch(
        knots[rows, selected], rotations[rows, selected], dt, capacity=capacity
    )
    owner, frame = torch.where(
        (torch.arange(capacity, device=start.device)[None] < path[2][:, None])
        & (found & timed)[:, None]
    )
    dense_scene = start.new_full((batch, capacity), torch.inf)
    dense_target = dense_scene.clone()
    if len(owner):
        s, t = release(
            geometry.select(owner),
            target_surface.select(owner),
            path[0][owner, frame, None],
            path[1][owner, frame, None],
        )
        dense_scene[owner, frame] = s[:, 0]
        dense_target[owner, frame] = t[:, 0]
    dense = (dense_scene.amin(-1) >= 0.002) & monotone_release_mask(dense_target)
    valid = found & timed & dense
    failure = torch.where(geometric.any(-1), 2, 1)
    failure = torch.where(found & ~timed, 3, failure)
    failure = torch.where(found & timed & ~dense, 4, failure)
    failure = torch.where(valid, 0, failure)
    return (
        path,
        valid,
        {
            "planning_failure_code": failure,
            "candidate_count": candidates,
            "geometric_candidate_mask": geometric,
            "robust_inverse_mask": inverse,
            "candidate_cost": cost,
            "selected": torch.where(valid, selected, -1),
            "selected_tilt_rad_azimuth_rad_retract_m": bank[selected],
            "target_world": end[rows, selected],
            "scene_start_clearance_m": scene[..., 0],
            "scene_path_clearance_m": scene.amin(-1),
            "scene_end_clearance_m": scene[..., -1],
            "target_surface_start_clearance_m": target[..., 0],
            "target_surface_path_clearance_m": target.amin(-1),
            "target_surface_end_clearance_m": target[..., -1],
            "minimum_clearance_m": dense_scene.amin(-1),
            "initial_target_surface_clearance_m": dense_target[:, 0],
            "final_target_surface_clearance_m": dense_target.gather(1, (path[2] - 1)[:, None])[
                :, 0
            ],
            "dense_geometry_checked": found & timed,
            "segment_boundaries": boundaries,
            "target_contact_release_policy": "monotone_signed_surface_separation",
            **inverse_report,
        },
    )
