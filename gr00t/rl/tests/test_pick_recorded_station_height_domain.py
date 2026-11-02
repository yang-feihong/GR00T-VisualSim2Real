"""Probe station body-height coverage without changing a running collection."""

import json
import os
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


@pytest.mark.skipif(
    not os.environ.get("PICK_INITIAL_STATION_HEIGHT_ROOT"), reason="Recorded probe required"
)
def test_initial_station_height_domain_counterfactual(tmp_path):
    from gr00t.rl.pick.station_batch import select_initial_pick_station_batch

    state, destination, sizes, support, mask, geometry, collision, limits = _recorded_batch(
        Path(os.environ["PICK_INITIAL_STATION_HEIGHT_ROOT"]), count=4
    )
    report = {"seeds": state["episode_seed"].tolist(), "physics_simulated": False, "cases": {}}
    for name, options in (
        ("current", {}),
        ("standing_extension", {"body_heights": (0.30, 0.34, 0.38, 0.536, 0.60, 0.64)}),
        (
            "standing_extension_closer_station",
            {
                "body_heights": (0.30, 0.34, 0.38, 0.536, 0.60, 0.64),
                "forward_offsets_m": (-0.32, -0.16, 0.0, 0.16, 0.32),
            },
        ),
    ):
        with torch.inference_mode():
            p, q, joints, valid, detail = select_initial_pick_station_batch(
                state,
                destination,
                sizes,
                support,
                limits,
                collision,
                geometry,
                support_mask=mask,
                **options,
            )
        report["cases"][name] = {
            "valid": valid.tolist(),
            "station": p.tolist(),
            "orientation": q.tolist(),
            "joint_reference": joints.tolist(),
            "failure_code": detail["planning_failure_code"].tolist(),
            "joint_candidate_budget": detail["joint_candidate_budget"],
            "joint_candidate_count": detail["joint_candidate_count"].tolist(),
            "selected_arrival_coverage": detail["selected_arrival_coverage"].tolist(),
            "exact_endpoint_counts": detail["exact_endpoint_point_counts"].tolist(),
            "safe_endpoint_counts": detail["safe_endpoint_point_counts"].tolist(),
            "maximum_arrival_coverage": detail["arrival_coverage_counts"]
            .flatten(1)
            .amax(-1)
            .tolist(),
        }
    report["limitations"] = "Hypothesis screening is not evidence of whole-body posture tracking."
    (tmp_path / "recorded_station_height_domain.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
