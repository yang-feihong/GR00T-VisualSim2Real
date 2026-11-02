"""Compare recorded arm branches against independent calibration metadata."""

import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not os.environ.get("PICK_BRANCH_RECORD"), reason="Records required")
def test_measured_joint_branch(tmp_path):
    run = Path(os.environ["PICK_BRANCH_RECORD"])
    record = torch.load(run / "control_record.pt", weights_only=False)
    actual = record["pre"]
    post = record["post"]
    model = torch.load(run / "collision_model.pt", weights_only=False)
    probes = Path(os.environ["PICK_BRANCH_PROBE"])
    history = torch.load(probes / "trace.pt", weights_only=False)
    index = int(os.environ["PICK_BRANCH_INDEX"])
    report = {
        "calibration_arm": history[-1, index, 20:26].tolist(),
        "calibration_body": history[-1, index, :7].tolist(),
        "actual_arm": actual["dof_pos"][700, model["arm_joint_ids"]].tolist(),
        "actual_body": actual["root_states"][700, :7].tolist(),
        "actual_base_commands": post["base_command"][675:700].tolist(),
        "object_force_at700": post["object_finger_forces"][700].tolist(),
        "object_force_peak": post["object_finger_forces"][600:700].norm(dim=-1).amax(0).tolist(),
    }
    if os.environ.get("PICK_BRANCH_SCENE_ROOT"):
        from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch

        _, _, _, _, _, geometry, _, _ = _recorded_batch(
            Path(os.environ["PICK_BRANCH_SCENE_ROOT"]), count=4, frame=700
        )
        names = record["robot_body_names"]
        ids = [i for i, name in enumerate(names) if name.lower().endswith("foot")]
        points = post["robot_body_state_w"][700, ids, :3].cuda()
        start, end = points[None].clone(), points[None].clone()
        start[..., 2] = 0.15
        end[..., 2] = -0.15
        hit, normal, _ = geometry.select(torch.tensor([2], device="cuda")).rays(start, end)
        report.update(
            foot_positions=points.tolist(), ground_hits=hit.tolist(), ground_normals=normal.tolist()
        )
    (tmp_path / "arm_branches.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
