"""Compare independent controller response poses to the deployed inverse, offline."""

import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not os.environ.get("PICK_INVERSE_PROBE"), reason="Probe required")
def test_inverse_controller_coverage(tmp_path):
    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
    from gr00t.rl.auto_open.pose_planning import solve_pose_grid
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    path = Path(os.environ["PICK_INVERSE_PROBE"])
    probe = json.loads((path / "summary.json").read_text())
    history = torch.load(path / "trace.pt", weights_only=False, map_location="cuda")
    roots = history[-1, :, :13]
    goals = roots.new_tensor(probe["goals_world"])
    quats = roots.new_tensor(probe["goal_quaternions_wxyz"])
    limits = roots.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    initial = roots.new_tensor(probe["initial_arm_joints_rad"]).expand(len(roots), -1)
    report = []
    if os.environ.get("PICK_RESPONSE_MODEL"):
        from gr00t.rl.pick.reach_calibration import CalibratedBodyResponse

        model = CalibratedBodyResponse(
            os.environ["PICK_RESPONSE_MODEL"], "cuda", allow_unverified=True
        )
        reference = roots.new_tensor(probe["initial_root_states"])
        predicted, valid = model(reference, goals, quats, initial)
        result = solve_pose_grid(
            goals[:, None],
            quats[:, None],
            predicted[None].expand(3, -1, -1),
            torch.stack((initial, limits.mean(-1).expand_as(initial), history[-1, :, 20:26])),
            limits,
        )
        residual = result.position_error.norm(dim=-1)[:, :, 0].amin(0)
        physical = torch.as_tensor(probe["physical_pocket_valid"], device="cuda")
        selected = residual.masked_fill(~valid | ~physical, torch.inf).argsort()[:20]
        report.append(
            {
                "model": os.environ["PICK_RESPONSE_MODEL"],
                "top_indices": selected.tolist(),
                "predicted_errors": residual[selected].tolist(),
                "actual_errors": history[-1, selected, -2].tolist(),
                "predicted_heights": predicted[selected, 2].tolist(),
                "actual_heights": roots[selected, 2].tolist(),
                "success_predicted_error": residual[history[-1, :, -2] < 0.012].tolist(),
            }
        )
    for height in (None, 0.34, 0.36):
        bodies = roots.clone()
        if height is not None:
            bodies[:, 2] = height
            w, x, y, z = roots[:, 3:7].unbind(-1)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            bodies[:, 3:7] = _euler_xyz_quat(
                torch.stack((yaw * 0, yaw * 0 + torch.pi * 8 / 180, yaw), -1)
            )
        seeds = torch.stack((initial, limits.mean(-1).expand_as(initial), history[-1, :, 20:26]))
        result = solve_pose_grid(
            goals[:, None], quats[:, None], bodies[None].expand(3, -1, -1), seeds, limits
        )
        error = result.position_error.norm(dim=-1)[:, :, 0]
        success = history[-1, :, -2] < 0.012
        report.append(
            {
                "height": height,
                "controller_success_indices": torch.where(success)[0].tolist(),
                "success_inverse_errors": error[:, success].tolist(),
                "success_inverse_exact": result.exact[:, success, 0].tolist(),
                "exact_count": result.exact.any(0).sum().item(),
            }
        )
    (tmp_path / "inverse_coverage.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
