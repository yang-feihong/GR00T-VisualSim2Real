"""Offline counterfactual: is side-grasp rejection caused by support-edge setback?"""

import json
import os
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


@pytest.mark.skipif(not os.environ.get("PICK_SIDE_EDGE_ROOT"), reason="Recorded probe required")
def test_recorded_side_grasp_edge_setback(tmp_path):
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.auto_open.pose_planning import solve_pose_grid
    from gr00t.rl.pick.grasp_quality import load_grasp_center, load_grasp_support
    from gr00t.rl.pick.planning import pocket_pose_targets

    source = Path(os.environ["PICK_SIDE_EDGE_ROOT"])
    _, _, _, _, _, geometry, collision, limits = _recorded_batch(source, count=4)
    paths = sorted(source.glob("logs/rank*-control/seed_*/control_record.pt"))
    report = {
        "physics_simulated": False,
        "support_test": "vertical_ray_under_object_origin_only",
        "episodes": [],
    }
    for row, path in enumerate(paths):
        summary = json.loads(path.with_name("summary.json").read_text())
        if len(summary["planning"].get("grasp_attempts", [])) > 1:
            continue
        episode = json.loads(path.with_name("episode.json").read_text())
        record = torch.load(path, map_location="cpu", weights_only=False)
        entries = [i for i, stage in enumerate(record["stages"]) if stage == "MOVE_TO_GRASP"]
        index = entries[0] if entries else len(record["stages"]) - 1
        snapshot = record["pre"] if entries else record["post"]
        root = snapshot["root_states"][index].cuda()
        obj = snapshot["object_state"][index].cuda()
        detail = summary["planning"]["grasp"]
        q = obj.new_tensor(detail["candidate_orientations_world_wxyz"])[0]
        quality = detail["convex_contact_model"]
        fit = torch.tensor(quality["pocket_fit"], device="cuda", dtype=torch.bool)[0]
        fit &= obj.new_tensor(quality["antipodal_margin"])[0] >= 0
        axes = _quat_rotate(q, obj.new_tensor([1, 0, 0]))
        side = fit & (axes[:, 2].abs() <= 0.5)
        towards = obj[:2] - root[:2]
        towards /= towards.norm().clamp_min(1e-6)
        score = (axes[:, :2] * towards).sum(-1).masked_fill(~side, -torch.inf)
        ids = score.topk(min(16, int(side.sum()))).indices
        if not len(ids):
            continue
        shifts = (
            torch.arange(51, device="cuda", dtype=obj.dtype) * 0.002
            if os.environ.get("PICK_SIDE_EDGE_FINE")
            else obj.new_tensor([0, 0.02, 0.04, 0.06, 0.08, 0.10])
        )
        if os.environ.get("PICK_SIDE_EDGE_JOINT"):
            proposal_index = torch.arange(512, device="cuda", dtype=obj.dtype)
            options = torch.where(side)[0]
            ids = options[((proposal_index * 0.618033989) % 1 * len(options)).long()]
            shifts = ((proposal_index * 0.414213562 + 0.5) % 1) * 0.10
            query_orientations = q[ids][:, None]
        else:
            query_orientations = q[ids][None].expand(len(shifts), -1, -1)
        targets = obj[:3].expand(len(shifts), -1).clone()
        targets[:, :2] -= shifts[:, None] * towards
        room = geometry.select(torch.full((len(shifts),), row, device="cuda", dtype=torch.long))
        support = load_grasp_support(episode["asset"]["usd_path"], "cuda")
        goals, counterfactual = pocket_pose_targets(
            query_orientations,
            targets,
            obj[3:7][None].expand(len(shifts), -1),
            support,
            collision,
            room,
        )
        valid = counterfactual["pocket_fit"] & (counterfactual["antipodal_margin"] >= 0)
        valid &= counterfactual["environment_clearance_m"] >= 0.002
        upper, lower = targets[:, None].clone(), targets[:, None].clone()
        upper[..., 2] = episode["support_z"] + 0.05
        lower[..., 2] = episode["support_z"] - 0.05
        hit, normal, _ = room.rays(upper, lower)
        supported = ((hit[:, 0, 2] - episode["support_z"]).abs() <= 0.002) & (normal[:, 0, 2] > 0.9)
        center_supported = supported.clone()
        stencil_margin = None
        polygon_mode = bool(os.environ.get("PICK_SIDE_EDGE_CONTACT_POLYGON"))
        if os.environ.get("PICK_SIDE_EDGE_CONTACT_STENCIL") or polygon_mode:
            from gr00t.rl.pick.support_features import (
                bottom_contact_extrema, bottom_support_stencil, supported_contact_margin,
            )

            world_support = _quat_rotate(obj[None, 3:7], support)
            mask = torch.ones(1, len(support), device="cuda", dtype=torch.bool)
            stencil = (
                bottom_contact_extrema(world_support[None], mask, contact_band_m=0.002)
                if polygon_mode else bottom_support_stencil(world_support[None], mask)
            )[0]
            com = _quat_rotate(obj[3:7], load_grasp_center(episode["asset"]["usd_path"], "cuda"))
            stencil_margin = torch.minimum(
                com[:2] - stencil[:, :2].amin(0), stencil[:, :2].amax(0) - com[:2]
            ).amin()
            stencil = torch.cat((stencil, com[None]))
            upper = targets[:, None].expand(-1, len(stencil), -1).clone()
            upper[..., :2] += stencil[None, :, :2]
            lower = upper.clone()
            upper[..., 2], lower[..., 2] = episode["support_z"] + 0.05, episode["support_z"] - 0.05
            footprint_hit, footprint_normal, _ = room.rays(upper, lower)
            hits_supported = (
                ((footprint_hit[..., 2] - episode["support_z"]).abs() <= 0.002)
                & (footprint_normal[..., 2] > 0.9)
            )
            if polygon_mode:
                margin = supported_contact_margin(
                    stencil[None, :-1].expand(len(shifts), -1, -1), hits_supported[:, :-1],
                    com[None].expand(len(shifts), -1),
                )
                supported = (margin >= 0.001) & hits_supported[:, -1]
                report["support_test"] = "2mm_contact_band_actual_points_convex_support_polygon_COM_margin_1mm"
            else:
                supported = hits_supported.all(-1) & (stencil_margin >= -0.0005)
                report["support_test"] = "actual_bottom_band_9_ray_envelope_plus_convex_centroid_projection"
        support_row, side_id = torch.where(valid & supported[:, None])
        ik_counts = torch.zeros(len(shifts), dtype=torch.long, device="cuda")
        if len(support_row):
            grasp = goals[support_row, side_id]
            lifted = grasp.clone()
            minimum = max(0.30, episode["object_position"][2] + 0.10)
            lifted[:, 2] += max(0, minimum - float(obj[2])) + 0.005
            withdrawn = lifted.clone()
            withdrawn[:, :2] -= towards * 0.16
            positions = torch.stack((grasp, lifted, withdrawn), 1)
            rotations = query_orientations[support_row, side_id, None].expand(-1, 3, -1)
            heights = torch.cat(
                (
                    obj.new_tensor([0.18, 0.22, 0.30, 0.38, 0.46, 0.536]),
                    root[2] + obj.new_tensor([-0.01, 0, 0.01]),
                )
            )
            actual = snapshot["dof_pos"][index].cuda()[collision.arm_joint_ids]
            seeds = torch.stack((actual, limits.mean(-1), limits.sum(-1) - actual))
            body = root[None, None, None].expand(len(heights), 3, len(grasp), -1).clone()
            body[..., 2] = heights[:, None, None]
            result = solve_pose_grid(
                positions,
                rotations,
                body,
                seeds[None, :, None].expand(len(heights), -1, len(grasp), -1),
                limits,
            )
            exact = result.exact.any(0).any(0).all(-1)
            ik_counts.scatter_add_(0, support_row, exact.long())
        report["episodes"].append(
            {
                "seed": summary["seed"],
                "side_proposals": len(ids),
                "shifts_towards_robot_m": shifts.tolist(),
                "clear_side_grasps": valid.sum(-1).tolist(),
                "support_under_center": center_supported.tolist(),
                "support_footprint_accepted": supported.tolist(),
                "centroid_margin_to_contact_bbox_m": (
                    None if stencil_margin is None else float(stencil_margin)
                ),
                "supported_and_endpoint_inverse_available": ik_counts.tolist(),
            }
        )
    (tmp_path / "side_edge_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["episodes"]
