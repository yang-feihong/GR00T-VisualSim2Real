"""Measured proposals are selected by geometry/cost, not scene or seed identities."""

import json

import torch

from gr00t.rl.pick import measured_station


def test_measured_station_chooses_nearest_safe_candidate(monkeypatch, tmp_path):
    catalog = tmp_path / "commands.json"
    catalog.write_text(
        json.dumps({"schema": "pick_measured_low_commands_v1", "checkpoint_sha256": "a" * 64})
    )
    planner = measured_station.MeasuredLowReachPlanner({}, "a" * 64, catalog=catalog)
    bank = {
        "stations": torch.tensor([[0.7, 0, 0], [0.1, 0, 0], [0.5, 0, 0]]),
        "orientations": torch.tensor([[1.0, 0, 0, 0]]).expand(3, -1),
        "depth_preferences": torch.tensor([0.0, 0.5, 0.25]),
        "sample_ids": torch.tensor([10, 20, 30]),
        "feasible": torch.tensor([True, False, True]),
        "measured_command_count": 3,
        "checkpoint_sha256": "a" * 64,
    }
    monkeypatch.setattr(measured_station, "measured_grasp_matches", lambda *a, **kw: bank)
    state = {"ee_pos": torch.zeros(1, 3), "root_states": torch.zeros(1, 13)}
    pose, quat, joints, report = planner.station(state, None, None, None, None, None, None)
    torch.testing.assert_close(pose, bank["stations"][2:3])
    assert joints is None and report["selected_sample_id"] == 30
    assert report["not_a_physical_grasp_guarantee"]
    torch.testing.assert_close(planner.orientation, quat)
    assert planner.depth == 0.25
    assert not hasattr(planner, "grasp")  # Stage-entry GPU batches own grasp planning.


def test_policy_mismatch_does_not_use_measured_data(monkeypatch, tmp_path):
    catalog = tmp_path / "commands.json"
    catalog.write_text(
        json.dumps({"schema": "pick_measured_low_commands_v1", "checkpoint_sha256": "a" * 64})
    )
    planner = measured_station.MeasuredLowReachPlanner({}, "b" * 64, catalog=catalog)
    monkeypatch.setattr(
        measured_station, "select_initial_pick_station", lambda *a, **kw: (None, None, None, {})
    )
    result = planner.station({}, None, None, None, None, None, None)
    assert result[3]["measured_reach_fallback"] == "catalog_missing_or_policy_mismatch"
