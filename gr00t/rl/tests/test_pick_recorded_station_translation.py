"""Endpoint counterfactuals isolate station placement from whole-body response."""

import json
import os
import time
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


@pytest.mark.skipif(
    not os.environ.get("PICK_RECORDED_TRANSLATION_ROOT"), reason="Recorded probe required"
)
def test_recorded_terminal_goals_with_local_station_translation(tmp_path):
    from gr00t.rl.auto_open.pose_planning import solve_pose_grid
    from gr00t.rl.pick.station_batch import station_corridors_clear

    root = Path(os.environ["PICK_RECORDED_TRANSLATION_ROOT"])
    state, _, _, _, _, geometry, collision, limits = _recorded_batch(root, count=4, frame=-1)
    paths = sorted(
        root.glob("logs/rank*-control/seed_*/control_record.pt"),
        key=lambda p: int(p.parent.name.removeprefix("seed_")),
    )
    records = [torch.load(path, weights_only=False) for path in paths]
    positions = torch.stack([r["post"]["ee_goal_cart_world"][-1] for r in records]).cuda()
    quats = torch.stack([r["post"]["ee_goal_orn_quat_world"][-1] for r in records]).cuda()
    offsets = torch.linspace(-0.25, 0.25, 11, device="cuda")
    forward, lateral = torch.meshgrid(offsets, offsets, indexing="ij")
    local = torch.stack((forward.flatten(), lateral.flatten()), -1)
    actual_root = state["root_states"]
    w, x, y, z = actual_root[:, 3:7].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    dx = local[:, 0, None] * yaw.cos() - local[:, 1, None] * yaw.sin()
    dy = local[:, 0, None] * yaw.sin() + local[:, 1, None] * yaw.cos()
    roots = actual_root[None].expand(len(local), -1, -1).clone()
    roots[..., 0] += dx
    roots[..., 1] += dy
    actual = state["dof_pos"][:, collision.arm_joint_ids]
    seeds = torch.stack((actual, limits.mean(-1).expand_as(actual), limits.sum(-1) - actual))
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        result = solve_pose_grid(
            positions[:, None],
            quats[:, None],
            roots[:, None].expand(-1, len(seeds), -1, -1),
            seeds[None].expand(len(local), -1, -1, -1),
            limits,
        )
        joints = result.arm_joint_positions[..., 0, :]
        clearance = (
            collision.clearance(
                geometry,
                roots.permute(1, 0, 2)[:, :, None].expand(-1, -1, 3, -1).flatten(1, 2),
                joints.permute(2, 0, 1, 3).flatten(1, 2),
                include_gripper=False,
                maximum_distance_m=0.0021,
            )
            .reshape(len(actual), len(local), 3)
            .permute(1, 2, 0)
        )
        good = result.exact[..., 0] & (clearance >= 0.002)
        poses = torch.cat(
            (roots.permute(1, 0, 2)[..., :2], yaw[:, None, None].expand(-1, len(local), 1)), -1
        )
        corridor = station_corridors_clear(actual_root, poses, geometry).T
        valid = good.any(1) & corridor
        costs = local.norm(dim=-1)[:, None].expand(-1, len(actual))
        selected = costs.masked_fill(~valid, torch.inf).argmin(0)
    torch.cuda.synchronize()
    cases = []
    for row, seed in enumerate(state["episode_seed"].tolist()):
        cases.append(
            {
                "seed": seed,
                "world_body_height_m": float(actual_root[row, 2]),
                "exact_endpoint_offsets": int(result.exact[..., 0].any(1)[:, row].sum()),
                "arm_and_corridor_safe_offsets": int(valid[:, row].sum()),
                "nearest_safe_offset_forward_left_m": local[selected[row]].tolist()
                if valid[:, row].any()
                else None,
            }
        )
    report = {
        "physics_simulated": False,
        "definition": "unchanged_raw_terminal_goal_actual_body_height_and_orientation",
        "limitations": "Endpoint and base corridor only; not a grasp path or policy guarantee. "
        "Entry failures may hold a gaze goal rather than a grasp goal.",
        "offset_count_per_seed": len(local),
        "joint_seeds": len(seeds),
        "gpu_elapsed_s": time.perf_counter() - started,
        "cases": cases,
    }
    (tmp_path / "recorded_station_translation.json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.skipif(
    not os.environ.get("PICK_RECORDED_TRANSLATION_ROOT"), reason="Recorded probe required"
)
def test_recorded_grasp_global_orientation_at_realized_body(tmp_path):
    from gr00t.rl.auto_open.pose_planning import solve_pose_grid
    from gr00t.rl.auto_open.tensor_random import episode_uniform
    from gr00t.rl.pick.planning import geometric_grasp_mask, grasp_candidate_bank

    state, _, sizes, support, mask, geometry, collision, limits = _recorded_batch(
        Path(os.environ["PICK_RECORDED_TRANSLATION_ROOT"]), count=4, frame=-1
    )
    draws = episode_uniform(state["episode_seed"], 0x4F524945, 1000, lanes=3)
    a, b = 2 * torch.pi * draws[..., 1], 2 * torch.pi * draws[..., 2]
    quats = torch.stack(
        (
            draws[..., 0].sqrt() * b.cos(),
            (1 - draws[..., 0]).sqrt() * a.sin(),
            (1 - draws[..., 0]).sqrt() * a.cos(),
            draws[..., 0].sqrt() * b.sin(),
        ),
        -1,
    )
    with torch.inference_mode():
        orientations, goals, quality, _, _ = grasp_candidate_bank(
            state,
            state["object_state"][:, :3],
            sizes,
            collision,
            geometry,
            support,
            support_mask=mask,
            candidate_orientations=quats,
            object_frame_fallback=False,
        )
        physical = geometric_grasp_mask(orientations, quality, 0.002, None, None)
        owner, candidate = torch.where(physical)
        exact, safe = torch.zeros_like(physical), torch.zeros_like(physical)
        if len(owner):
            roots = state["root_states"][owner]
            actual = state["dof_pos"][owner][:, collision.arm_joint_ids]
            seeds = torch.stack(
                (actual, limits.mean(-1).expand_as(actual), limits.sum(-1) - actual)
            )
            result = solve_pose_grid(
                goals[owner, candidate, None],
                orientations[owner, candidate, None],
                roots[None].expand(3, -1, -1),
                seeds,
                limits,
            )
            clearance = collision.clearance(
                geometry.select(owner),
                roots[:, None].expand(-1, 3, -1),
                result.arm_joint_positions[..., 0, :].transpose(0, 1),
                maximum_distance_m=0.0021,
                include_gripper=False,
            ).T
            exact[owner, candidate] = result.exact[..., 0].any(0)
            safe[owner, candidate] = (result.exact[..., 0] & (clearance >= 0.002)).any(0)
    report = {
        "physics_simulated": False,
        "definition": "1000_uniform_SO3_grasp_proposals_actual_recorded_body_pose",
        "limitations": "Three joint initializations, endpoint only; no guarantee of global "
        "IK infeasibility, full-path safety, or whole-body response.",
        "seeds": state["episode_seed"].tolist(),
        "physical_pockets": physical.sum(-1).tolist(),
        "exact_endpoint_candidates": exact.sum(-1).tolist(),
        "arm_safe_endpoint_candidates": safe.sum(-1).tolist(),
    }
    (tmp_path / "recorded_global_orientation.json").write_text(json.dumps(report, indent=2) + "\n")
