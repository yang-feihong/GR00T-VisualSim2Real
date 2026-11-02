"""Recheck deployed initial station geometry against completed debug records."""

import json
import os
import time
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


@pytest.mark.skipif(not os.environ.get("PICK_INITIAL_STATION_ROOT"), reason="Records required")
@pytest.mark.parametrize(
    "case",
    [
        "original",
        "wider",
        "raised",
        "pitch",
        "height",
        "robust_low",
        "lateral",
    ],
)
def test_recorded_initial_station(tmp_path, case):
    from gr00t.rl.pick.station_batch import select_initial_pick_station_batch

    state, destination, sizes, support, mask, geometry, collision, limits = _recorded_batch(
        Path(os.environ["PICK_INITIAL_STATION_ROOT"]), count=4
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        options = (
            {
                "forward_offsets_m": (-0.32, -0.16, 0.0, 0.16, 0.32),
                "lateral_offsets_m": (-0.24, -0.12, 0.0, 0.12, 0.24),
            }
            if case in ("wider", "raised")
            else {}
        )
        if case == "raised":
            options.update(
                body_heights=(0.30, 0.34, 0.38, 0.536, 0.58),
                body_pitches_deg=(-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0),
            )
        if case == "pitch":
            options.update(body_pitches_deg=(-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0))
        if case == "height":
            options.update(body_heights=(0.30, 0.34, 0.38, 0.536, 0.58))
        if case == "lateral":
            options.update(lateral_offsets_m=(-0.36, -0.12, 0.0, 0.12, 0.36))
        if case == "robust_low":
            options.update(
                body_heights=(0.34, 0.38),
                body_pitches_deg=(4.0, 8.0),
                require_all_postures=True,
                forward_offsets_m=(-0.48, -0.32, -0.16, 0.0, 0.16, 0.32),
                lateral_offsets_m=(-0.24, 0.0, 0.24),
            )
        pose, orientation, joints, valid, detail = select_initial_pick_station_batch(
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
    torch.cuda.synchronize()
    report = {
        "physics_simulated": False,
        "elapsed_s": time.perf_counter() - started,
        "valid": valid.tolist(),
        "station_poses": pose.tolist(),
        "orientations": orientation.tolist(),
        "selected_grasp_orientation": orientation.tolist(),
        "joint_reference": joints.tolist(),
    }
    for key in (
        "planning_failure_code",
        "selected_arrival_coverage",
        "highest_grasp_height_m",
        "selected_endpoint_joints_rad",
        "selected_candidate",
        "exact_endpoint_point_counts",
        "safe_endpoint_point_counts",
    ):
        report[key] = detail[key].tolist()
    (tmp_path / "initial_station_recheck.json").write_text(json.dumps(report, indent=2))
    print(report)
    assert valid[state["on_floor"]].all() if case == "robust_low" else valid.all(), report
