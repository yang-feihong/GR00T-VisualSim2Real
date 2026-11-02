"""Opt-in physical-target verification of completed compensated debug episodes."""

import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not os.environ.get("PICK_COMMAND_VALIDATION_ROOT"), reason="Records required")
def test_recorded_compensation_preserves_physical_arrival_and_strict_acceptance(tmp_path):
    root = Path(os.environ["PICK_COMMAND_VALIDATION_ROOT"])
    acceptance = json.loads((root / "acceptance_report.json").read_text())
    report = {"accepted": acceptance["accepted_episodes"], "episodes": []}
    records = sorted(root.glob("logs/rank*-control/seed_*/control_record.pt"))
    assert len(records) == 4
    for path in records:
        record = torch.load(path, weights_only=False)
        summary = json.loads(path.with_name("summary.json").read_text())
        post = record["post"]
        intended = post["ee_intended_goal_world"]
        offsets = post["ee_command_height_offset_m"]
        assert torch.isfinite(intended).all() and torch.isfinite(offsets).all()
        assert (offsets >= -0.020001).all() and (offsets <= 0).all()
        torch.testing.assert_close(
            offsets.abs().max(),
            offsets.new_tensor(summary["command_compensation"]["peak_applied_abs_m"]),
        )
        error = (post["ee_pos"] - intended).norm(dim=-1)
        visits = []
        start = None
        for index, stage in enumerate([*record["stages"], None]):
            if stage == "MOVE_TO_GRASP" and start is None:
                start = index
            if stage != "MOVE_TO_GRASP" and start is not None:
                visits.append(
                    {
                        "frames": index - start,
                        "last_physical_error_m": float(error[index - 1]),
                        "minimum_physical_error_m": float(error[start:index].min()),
                        "peak_downward_command_offset_m": float(offsets[start:index].abs().max()),
                    }
                )
                start = None
        report["episodes"].append(
            {
                "seed": summary["seed"],
                "success": summary["success"],
                "failure": summary["failure"],
                "grasp_arrival_visits": visits,
            }
        )
    (tmp_path / "command_validation.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    assert report["accepted"] == 4, report
    for episode in report["episodes"]:
        assert episode["success"] and episode["grasp_arrival_visits"], episode
        assert episode["grasp_arrival_visits"][-1]["last_physical_error_m"] < 0.012001, episode
