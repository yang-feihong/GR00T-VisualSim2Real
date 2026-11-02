"""Shared bounded GPU inverse checks for basket candidate/path banks."""

import torch

from gr00t.rl.auto_open.pose_planning import solve_pose_grid

MAX_ARM_QUERY_POSES = 65536


def safe_inverse_paths_batch(
    state,
    positions,
    quats,
    limits,
    collision,
    geometry,
    *,
    candidate_mask=None,
    body_height_offsets=(0.0,),
    final_joint_margin_rad=0.0,
    minimum_arm_clearance_m=0.002,
    maximum_arm_query_distance_m=None,
):
    """Preserve whole-path seed continuity and require every body-height hypothesis."""
    if positions.ndim != 4 or positions.shape[-1] != 3 or positions.shape[2] < 1:
        raise ValueError("Basket paths require [environments, candidates, samples, xyz]")
    batch, candidates, length = positions.shape[:3]
    if quats.shape != (*positions.shape[:-1], 4):
        raise ValueError("Basket path orientations must match the path dimensions")
    if candidate_mask is not None and candidate_mask.shape != (batch, candidates):
        raise ValueError("Basket candidate mask must match environments and candidates")
    if batch > 1 and not hasattr(geometry, "select"):
        raise ValueError("Multiple basket scenes require device-side geometry selection")
    mask = (
        torch.ones(batch, candidates, device=positions.device, dtype=torch.bool)
        if candidate_mask is None
        else candidate_mask
    )
    owner, candidate = torch.where(mask)
    safe = torch.zeros_like(mask)
    travel = positions.new_full(mask.shape, torch.inf)
    diagnostics = {
        "geometry_eligible": mask,
        "arm_clearance_query_cap_m": 2.0
        if maximum_arm_query_distance_m is None
        else maximum_arm_query_distance_m,
    }
    if not len(owner):
        return safe, travel, diagnostics
    roots = state["root_states"][owner]
    actual = state["dof_pos"][owner][:, collision.arm_joint_ids]
    seeds = torch.stack((actual, limits.mean(-1).expand_as(actual), limits.sum(-1) - actual))
    offsets = roots.new_tensor(body_height_offsets)
    if not len(offsets):
        raise ValueError("Basket inverse checks require at least one body-height hypothesis")
    shape = (len(offsets), len(seeds), len(owner))
    bodies = roots[None, None].expand(*shape, 13).clone()
    bodies[..., 2] += offsets[:, None, None]
    result = solve_pose_grid(
        positions[owner, candidate],
        quats[owner, candidate],
        bodies,
        seeds[None].expand(*shape, 6),
        limits,
    )
    arm, exact = result.arm_joint_positions, result.exact
    margin = torch.minimum(arm - limits[:, 0], limits[:, 1] - arm).amin(-1)
    position_error = result.position_error.norm(dim=-1)
    possible = exact.all(-1)
    if final_joint_margin_rad > 0:
        possible &= (margin[..., length // 2 :] >= final_joint_margin_rad).all(-1)
        possible &= (position_error <= 0.002).all(-1)
    chunks = []
    capacity = max(1, MAX_ARM_QUERY_POSES // (shape[0] * shape[1] * length))
    for start in range(0, len(owner), capacity):
        stop = min(start + capacity, len(owner))
        selected_geometry = (
            geometry.select(owner[start:stop]) if hasattr(geometry, "select") else geometry
        )
        body = bodies[:, :, start:stop, None].expand(-1, -1, -1, length, -1)
        gap = collision.clearance(
            selected_geometry,
            body.permute(2, 0, 1, 3, 4),
            arm[:, :, start:stop].permute(2, 0, 1, 3, 4),
            include_gripper=False,
            query_mask=possible[:, :, start:stop]
            .permute(2, 0, 1)[..., None]
            .expand(-1, -1, -1, length),
            **(
                {}
                if maximum_arm_query_distance_m is None
                else {
                    "maximum_distance_m": maximum_arm_query_distance_m,
                }
            ),
        )
        chunks.append(gap.permute(1, 2, 0, 3))
    arm_clearance = torch.cat(chunks, 2)
    robust = possible & (arm_clearance >= minimum_arm_clearance_m).all(-1)
    safe[owner, candidate] = robust.any(1).all(0)
    motion = (arm - actual[None, None, :, None]).norm(dim=-1).amax(-1)
    travel[owner, candidate] = motion.amin(1).amax(0).to(travel.dtype)
    best_clearance = arm_clearance.amin(-1).masked_fill(~possible, -torch.inf).amax(1)
    best_clearance = best_clearance.masked_fill(~possible.any(1), torch.nan)
    for name, values in (
        ("inverse_exact_path_by_height", exact.all(-1).any(1)),
        ("inverse_safe_path_by_height", robust.any(1)),
        ("best_max_position_error_by_height_m", position_error.amax(-1).amin(1)),
        ("best_minimum_arm_clearance_by_height_m", best_clearance),
        ("arm_clearance_query_evaluated_by_height", possible.any(1)),
        ("best_final_half_joint_margin_by_height_rad", margin[..., length // 2 :].amin(-1).amax(1)),
    ):
        field = torch.full(
            (batch, candidates, len(offsets)),
            False if values.dtype == torch.bool else torch.nan,
            device=positions.device,
            dtype=values.dtype,
        )
        field[owner, candidate] = values.T
        diagnostics[name] = field
    return safe, travel, diagnostics
