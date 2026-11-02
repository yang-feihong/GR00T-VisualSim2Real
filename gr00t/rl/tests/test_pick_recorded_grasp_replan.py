"""Replay first grasp entries through the deployed batched stage tools, without physics."""

import json
import os
import time
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


@pytest.mark.skipif(not os.environ.get("PICK_GRASP_REPLAN_ROOT"), reason="Records required")
@pytest.mark.parametrize("refresh_station", [False, True])
def test_recorded_grasp_replan(tmp_path, refresh_station):
    from gr00t.rl.auto_open.pose_planning import solve_pose_grid
    from gr00t.rl.pick.grasp_inverse import solve_default_grasp_inverse
    from gr00t.rl.pick.grasp_paths import score_default_grasp_paths
    from gr00t.rl.pick.planning import grasp_body_height_hypotheses, grasp_candidate_bank
    from gr00t.rl.pick.scene_object_geometry import SceneObjectGeometryBatch
    from gr00t.rl.pick.tensor_rows import pad_device_rows

    root = Path(os.environ["PICK_GRASP_REPLAN_ROOT"])
    paths = sorted(root.glob("logs/rank*-control/seed_*/control_record.pt"))
    state, destination, sizes, support, mask, geometry, collision, limits = _recorded_batch(
        root, count=4
    )
    if refresh_station:
        from gr00t.rl.pick.station_batch import select_initial_pick_station_batch

        station = select_initial_pick_station_batch(
            state, destination, sizes, support, limits, collision, geometry, support_mask=mask
        )
        assert station[3].all(), station[4]["planning_failure_code"]
    snapshots, episodes, preferred = [], [], []
    for path in paths:
        record = torch.load(path, weights_only=False)
        frame = record["stages"].index("MOVE_TO_GRASP")
        if os.environ.get("PICK_GRASP_TERMINAL"):
            frame = -1
        snapshots.append(
            {
                key: value[frame].cuda()
                for key, value in record["pre"].items()
                if torch.is_tensor(value)
            }
        )
        episodes.append(json.loads(path.with_name("episode.json").read_text()))
        summary = json.loads(path.with_name("summary.json").read_text())
        first = summary["planning"].get("grasp_attempts", [summary["planning"]["grasp"]])[0]
        preferred.append(first["candidate_orientations_world_wxyz"][0][0])
    local = {
        key: torch.stack([snapshot[key] for snapshot in snapshots])
        for key in ("root_states", "object_state", "dof_pos", "ee_pos", "ee_orn")
    }
    if refresh_station:
        from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply
        from gr00t.rl.auto_open.planning_geometry import _quat_rotate

        body = local["root_states"]
        w, x, y, z = body[:, 3:7].unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        delta = station[0][:, 2] - yaw
        turn = _euler_xyz_quat(torch.stack((delta * 0, delta * 0, delta), -1))
        offset = _quat_rotate(turn, local["ee_pos"] - body[:, :3])
        body[:, :2] = station[0][:, :2]
        body[:, 3:7] = _quat_multiply(turn, body[:, 3:7])
        local["ee_pos"] = body[:, :3] + offset
        local["ee_orn"] = _quat_multiply(turn, local["ee_orn"])
        preferred = station[1].tolist()
    local["distractor_states"], _ = pad_device_rows([s["distractor_states"] for s in snapshots])
    geometry = SceneObjectGeometryBatch(episodes, geometry.components[0]).update(local)
    support_counts = mask.sum(-1).tolist()
    if os.environ.get("PICK_UNIFORM_SUPPORT"):
        from gr00t.rl.pick.support_features import uniform_surface_support

        support, mask = uniform_surface_support(support, mask, count=9984)
    torch.cuda.synchronize()
    started = time.perf_counter()
    bank = grasp_candidate_bank(
        local,
        local["object_state"][:, :3],
        sizes,
        collision,
        geometry,
        support,
        support_mask=mask,
        preferred_orientation=support.new_tensor(preferred),
        **(
            {"candidate_orientations": support.new_tensor(preferred)[:, None]}
            if os.environ.get("PICK_PREFERRED_TRANSIT")
            else {}
        ),
    )
    torch.cuda.synchronize()
    bank_done = time.perf_counter()
    inverse = solve_default_grasp_inverse(
        local, bank, limits, collision.arm_joint_ids, local["object_state"][:, 2], 0.02
    )
    torch.cuda.synchronize()
    inverse_done = time.perf_counter()
    edge_checks = []
    if os.environ.get("PICK_CAPTURE_EDGES"):
        original_edges = collision._moving_envelope_intersects

        def capture_edges(g, p, q, query_mask=None):
            original_visible = g.visible

            def visible(starts, ends, tolerance_m=0.01):
                hit, normal, distance = g.rays(starts, ends)
                blocked = distance < (ends - starts).norm(dim=-1) - tolerance_m
                ids = torch.where(blocked.flatten())[0][:10]
                if len(ids):
                    edge_checks.append(
                        {
                            "pose_shape": list(p.shape),
                            "hits": hit.reshape(-1, 3)[ids].tolist(),
                            "normals": normal.reshape(-1, 3)[ids].tolist(),
                            "origin": starts.reshape(-1, 3)[ids].tolist(),
                            "labels": g.last_hit_types.flatten()[ids].tolist(),
                        }
                    )
                return ~blocked

            g.visible = visible
            try:
                return original_edges(g, p, q, query_mask)
            finally:
                g.visible = original_visible

        collision._moving_envelope_intersects = capture_edges
    path, valid, detail = score_default_grasp_paths(
        local,
        bank,
        inverse,
        limits,
        collision,
        geometry,
        0.02,
        prefer_realized_body=torch.full(
            (4,), bool(os.environ.get("PICK_GRASP_TERMINAL")), device="cuda", dtype=torch.bool
        ),
        elevated_transit=bool(os.environ.get("PICK_PREFERRED_TRANSIT")),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    rows = torch.arange(4, device="cuda")
    selected = detail["selected"].clamp_min(0)
    goals, quats = bank[1][rows, selected], bank[0][rows, selected]
    body = local["root_states"][None].expand(2, -1, -1).clone()
    body[0, :, 2] = 0.36
    body[1, :, 2] = 0.30
    seeds = local["dof_pos"][:, collision.arm_joint_ids][None].expand(2, -1, -1)
    result = solve_pose_grid(goals[:, None], quats[:, None], body, seeds, limits)
    report = {
        "physics_simulated": False,
        "edge_checks": edge_checks,
        "planning_s": elapsed,
        "original_support_counts": support_counts,
        "support_width": support.shape[1],
        "bank_s": bank_done - started,
        "inverse_s": inverse_done - bank_done,
        "score_s": started + elapsed - inverse_done,
        "valid": valid.tolist(),
        "selected": detail["selected"].tolist(),
        "grasp_goals_world": goals.tolist(),
        "body_036_030_position_error_m": result.position_error.norm(dim=-1).tolist(),
        "candidates": bank[0].shape[1],
        "path_lengths": path[2].tolist(),
    }
    report["candidate0"] = {
        key: value[:, 0].tolist()
        for key, value in detail.items()
        if torch.is_tensor(value)
        and value.shape[:2] == bank[0].shape[:2]
        and key not in ("candidate_orientations_world_wxyz",)
    }
    report["predicate_counts"] = {
        key: value.sum(-1).tolist()
        for key, value in detail.items()
        if torch.is_tensor(value) and value.dtype == torch.bool and value.shape == bank[0].shape[:2]
    }
    own, option = torch.where(inverse["lookup"] >= 0)
    grasp_endpoint = inverse["counts"][own, :2].sum(-1, keepdim=True) - 1
    slot = torch.searchsorted(inverse["references"][own].contiguous(), grasp_endpoint.contiguous())
    raw_exact = inverse["result"]["exact"]
    end_exact = raw_exact.gather(-1, slot[None, None].expand(*raw_exact.shape[:2], -1, -1))
    heights = grasp_body_height_hypotheses(local["root_states"])
    highest = heights[own].T.masked_fill(~end_exact.any(1)[..., 0], -torch.inf).amax(0)
    height_bank = heights.new_full(bank[0].shape[:2], -torch.inf)
    height_bank[own, option] = highest
    report["highest_feasible_grasp_body_height"] = (
        height_bank.masked_fill(~detail["feasible"], -torch.inf).amax(-1).tolist()
    )
    report["highest_fitted_grasp_body_height"] = height_bank.amax(-1).tolist()
    report["highest_fitted_candidate"] = height_bank.argmax(-1).tolist()
    report["selected_grasp_body_height"] = height_bank[rows, selected].tolist()
    (tmp_path / "recorded_grasp_replan.json").write_text(json.dumps(report, indent=2))
    print(report)
    assert valid[2] if os.environ.get("PICK_GRASP_TERMINAL") else valid.all(), report
