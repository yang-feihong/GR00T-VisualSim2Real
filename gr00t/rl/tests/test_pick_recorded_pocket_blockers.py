"""Diagnose recorded contact geometry and viable body-height preferences on GPU."""

import json
import os
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


@pytest.mark.skipif(not os.environ.get("PICK_POCKET_BLOCKER_ROOT"), reason="Records required")
def test_recorded_pocket_blockers(tmp_path):
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.auto_open.first_pull_wall import _asset_envelopes
    from gr00t.rl.envs.door.b2z1_reset_geometry import _axis_rotation, _arm_offset_tensors
    from gr00t.rl.utils.b2z1_asset import ROBOT_USD

    root = Path(os.environ["PICK_POCKET_BLOCKER_ROOT"])
    state, _, _, _, _, geometry, collision, _ = _recorded_batch(root, count=4, frame=-1)
    paths = sorted(root.glob("logs/rank*-control/seed_*/summary.json"))
    report = []
    for row, path in enumerate(paths):
        summary = json.loads(path.read_text())
        if summary["success"]:
            continue
        detail = summary["planning"]["grasp"]
        q = state["ee_pos"].new_tensor(detail["candidate_orientations_world_wxyz"]).reshape(-1, 4)
        positions = state["ee_pos"].new_tensor(detail["candidate_grasp_origins_world_m"]).reshape(-1, 3)
        quality = detail["convex_contact_model"]
        selected = detail["selected"]
        tensors = {key: q.new_tensor(quality[key]).reshape(-1, width)
                   for key, width in (("fixed_contact_local", 3), ("moving_contact_local", 3),
                                      ("aperture_angle_rad", 1))}
        pinch = _quat_rotate(q, tensors["moving_contact_local"] - tensors["fixed_contact_local"])
        tilt = torch.rad2deg(torch.asin((pinch[:, 2] / pinch.norm(dim=-1).clamp_min(1e-8)).clamp(-1, 1)))
        record = torch.load(path.with_name("control_record.pt"), weights_only=False)
        post = record["post"]
        ids = record["gripper_joint_id"]
        actual_angle = -post["dof_pos"][-1, ids].to(q).flatten()[0]
        aperture = torch.stack((tensors["aperture_angle_rad"][selected, 0], actual_angle))
        p = torch.stack((positions[selected], state["ee_pos"][row]))[None]
        orn = torch.stack((q[selected], state["ee_orn"][row]))[None]
        scene = geometry.select(torch.tensor([row], device=q.device))
        gap = collision.gripper_pose_clearance(scene, p, orn, aperture[None])[0]
        sweep = collision.gripper_closing_clearance(scene, p, orn, aperture[None])[0]
        bounds, first, second, axis = _asset_envelopes(str(ROBOT_USD))
        first, second = q.new_tensor(first), q.new_tensor(second)
        open_r = first[:3, :3] @ _axis_rotation(q.new_tensor([-torch.pi / 3]), axis, q.dtype)[0]
        open_t = first[:3, 3] + open_r @ second[:3, 3]
        open_r = open_r @ second[:3, :3]
        raw = (collision.gripper_surface[1] - open_t) @ open_r
        r = first[:3, :3] @ _axis_rotation(-aperture, axis, q.dtype)
        t = first[:3, 3] + (r @ second[:3, 3, None])[..., 0]
        r = r @ second[:3, :3]
        wrist_points = torch.einsum("bij,nj->bni", r, raw) + t[:, None]
        offsets = _arm_offset_tensors(q.dtype, q.device)
        local_points = (wrist_points - offsets[7] - offsets[8]) @ q.new_tensor(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        )
        exact_gap = geometry.select(torch.tensor([row, row], device=q.device)).rigid_surface_clearance(
            p[0, :, None], orn[0, :, None], local_points, maximum_distance_m=0.1
        )[:, 0]
        raw_bounds = q.new_tensor(bounds[-1])
        corners = torch.cartesian_prod(*[raw_bounds[:, axis] for axis in range(3)])
        box = torch.einsum("bij,nj->bni", r, corners) + t[:, None]
        box = (box - offsets[7] - offsets[8]) @ q.new_tensor(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        )
        world_box = p[0, :, None] + _quat_rotate(orn[0, :, None], box)
        edges = torch.tensor([[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],
                              [4,5],[4,6],[5,7],[6,7]], device=q.device)
        edge_clear = geometry.select(torch.tensor([row, row], device=q.device)).visible(
            world_box[:, edges[:, 0]], world_box[:, edges[:, 1]]
        )
        goal = post["ee_goal_cart_world"][-1].to(q)
        actual_q = state["ee_orn"][row]
        angle_error = 2 * (actual_q * q[selected]).sum().abs().clamp_max(1).acos()
        report.append({
            "seed": summary["seed"], "selected": selected,
            "pinch_tilt_deg": float(tilt[selected]),
            "planned_aperture_rad": float(aperture[0]), "actual_aperture_rad": float(aperture[1]),
            "nominal_and_actual_pose_clearance_m": gap.tolist(),
            "nominal_and_actual_closing_sweep_clearance_m": sweep.tolist(),
            "asset_joint_nominal_and_actual_mover_clearance_m": exact_gap.tolist(),
            "mover_obb_intersecting_edge_count": (~edge_clear).sum(-1).tolist(),
            "hinge_frame_parent": first.tolist(),
            "actual_position_error_m": float((state["ee_pos"][row] - goal).norm()),
            "actual_orientation_error_rad": float(angle_error),
            "selected_position_world": positions[selected].tolist(),
            "actual_position_world": state["ee_pos"][row].tolist(),
            "selected_orientation_world": q[selected].tolist(),
            "actual_orientation_world": actual_q.tolist(),
            "level_pinch_candidates": int((tilt.abs() <= 10).sum()),
        })
    (tmp_path / "recorded_pocket_blockers.json").write_text(json.dumps(report, indent=2))
    print(report)
