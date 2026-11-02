"""Measured proposals retain the world floor reference and remain experiments."""

import json
import math

import pytest
import torch

from gr00t.rl.pick.reach_calibration import export_measured_commands, measured_grasp_matches


def fixture(tmp_path):
    run, probe = tmp_path / "run", tmp_path / "probe"
    run.mkdir()
    probe.mkdir()
    root = torch.zeros(1, 13)
    root[:, 2], root[:, 3] = 0.5, 1
    obj = root.clone()
    obj[:, 2] = 0.05
    state = {"root_states": root, "object_state": obj, "distractor_states": torch.zeros(1, 0, 13)}
    torch.save([state], run / "trajectory.pt")
    torch.save({"gripper_surface": [torch.zeros(1, 3)] * 2}, run / "collision_model.pt")
    (run / "episode.json").write_text(
        json.dumps(
            {
                "support_type": "floor",
                "support_z": 0.0,
                "asset": {"usd_path": "unused"},
                "scene_usd_path": "unused",
                "geometry_cache_root": "unused",
            }
        )
    )
    report = {
        "checkpoint_sha256": "a" * 64,
        "synthetic_count": 1,
        "synthetic_offset": 0,
        "arm_ik_mode": "constrained",
        "gripper_angle_rad": -math.pi / 3,
        "final_root_states": [[2, 3, 0.35, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        "goals_world": [[2.6, 3, 0.1]],
        "goal_quaternions_wxyz": [[1, 0, 0, 0]],
        "final_position_error_m": [0.001],
        "final_orientation_error_rad": [0.001],
        "goal_gripper_floor_clearance_m": [0.005],
    }
    (probe / "summary.json").write_text(json.dumps(report))
    return run, probe, report


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("checkpoint_sha256", None, "fingerprinted"),
        ("synthetic_count", 0, "asset-independent"),
        ("gripper_angle_rad", 0.0, "fully-open"),
    ],
)
def test_measured_grasp_matching_rejects_incompatible_probes(tmp_path, field, value, reason):
    run, probe, report = fixture(tmp_path)
    report[field] = value
    (probe / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match=reason):
        measured_grasp_matches([probe], run, tmp_path / "result.json", device="cpu")


def test_measured_grasp_matching_uses_scene_floor_not_penetrating_object_hull(
    monkeypatch, tmp_path
):
    from gr00t.rl.pick import collision, distractors, grasp_quality, scene_geometry, station

    run, probe, _ = fixture(tmp_path)
    monkeypatch.setattr(
        grasp_quality, "load_grasp_support", lambda *args: torch.tensor([[0.0, 0.0, -0.053]])
    )

    def quality(orientation, *args, **kwargs):
        shape = (*orientation.shape[:2], 17)
        fit = torch.zeros(shape, dtype=torch.bool)
        fit[..., 5] = True
        center = torch.zeros(*shape, 3)
        # The command seating offset adds 0.5 mm to the requested world height.
        center[..., 2] = -0.0495
        return {
            "pocket_fit": fit,
            "antipodal_margin": torch.ones(shape),
            "center_local": center,
            "aperture_angle_rad": torch.ones(shape),
        }

    monkeypatch.setattr(grasp_quality, "convex_grasp_quality", quality)
    monkeypatch.setattr(scene_geometry, "read_scene_triangles", lambda *args: (None, None))
    monkeypatch.setattr(scene_geometry, "SceneGeometry", lambda *args: None)
    monkeypatch.setattr(distractors, "build_distractor_mesh", lambda *args: None)
    monkeypatch.setattr(distractors, "augment_scene_geometry", lambda *args: None)
    monkeypatch.setattr(
        station,
        "station_corridors_clear",
        lambda root, poses, geometry: torch.ones(poses.shape[:2], dtype=torch.bool),
    )

    class Collision:
        def gripper_clearance(self, geometry, poses, quats):
            return poses.new_full(poses.shape[:2], 0.01)

        def gripper_closing_clearance(self, geometry, poses, quats, aperture):
            return poses.new_full(poses.shape[:2], 0.01)

    monkeypatch.setattr(collision, "ArmCollisionSamples", Collision)
    result = measured_grasp_matches([probe], run, tmp_path / "result.json", device="cpu")
    assert result["matched_geometry_count"] == result["scene_safe_candidate_count"] == 24
    assert max(abs(value) for value in result["height_error_m"]) < 1e-7
    assert result["not_collision_certified"] and result["not_integrated_into_collection"]
    assert result["pocket_depth_preferences"] == [5 / 16] * 24
    for pose, goal in zip(result["station_poses"], result["candidate_grasp_origins_world_m"]):
        assert goal[0] - pose[0] == pytest.approx(0.6 * math.cos(pose[2]), abs=1e-6)
        assert goal[1] - pose[1] == pytest.approx(0.6 * math.sin(pose[2]), abs=1e-6)
    state = torch.load(run / "trajectory.pt", weights_only=False)[0]
    episode = json.loads((run / "episode.json").read_text())
    catalog = tmp_path / "commands.json"
    exported = export_measured_commands([probe], catalog, device="cpu")
    assert exported["source_sample_ids"] == [0]
    assert len(exported["source_probes"][0]["sha256"]) == 64

    def unexpected_trace_read(*args, **kwargs):
        raise AssertionError("Runtime matching must use the provided live state/geometry")

    monkeypatch.setattr(torch, "load", unexpected_trace_read)
    online = measured_grasp_matches(
        [probe],
        None,
        None,
        device="cpu",
        episode=episode,
        state=state,
        collision=Collision(),
        geometry=object(),
    )
    torch.testing.assert_close(online["stations"], torch.tensor(result["station_poses"]))
    assert online["feasible"].all()
    assert isinstance(online["orientations"], torch.Tensor)
    assert online["checkpoint_sha256"] == "a" * 64
    compact = measured_grasp_matches(
        [catalog],
        None,
        None,
        device="cpu",
        episode=episode,
        state=state,
        collision=Collision(),
        geometry=object(),
    )
    for key in ("stations", "orientations", "depth_preferences", "sample_ids", "feasible"):
        torch.testing.assert_close(compact[key], online[key])


def test_export_rejects_overlapping_physical_measurements(tmp_path):
    _, probe, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        export_measured_commands([probe, probe], tmp_path / "commands.json", device="cpu")


def test_export_position_budget_round_trips_without_changing_legacy_default(tmp_path):
    from gr00t.rl.pick.measured_commands import load_measured_commands

    _, probe, report = fixture(tmp_path)
    report["final_position_error_m"] = [0.007]
    (probe / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="nonempty"):
        export_measured_commands([probe], tmp_path / "legacy.json", device="cpu")
    catalog = tmp_path / "expanded.json"
    exported = export_measured_commands(
        [probe], catalog, device="cpu", maximum_position_error_m=0.008
    )
    assert exported["maximum_position_error_m"] == 0.008
    commands = load_measured_commands([catalog], torch.zeros(1))
    assert len(commands[0]) == 1
    with pytest.raises(ValueError, match="no accurately"):
        load_measured_commands([catalog], torch.zeros(1), maximum_position_error_m=0.004)


@pytest.mark.parametrize("limit", [0, -0.001, 0.009, float("nan"), float("inf")])
def test_export_rejects_unsafe_position_budgets(tmp_path, limit):
    _, probe, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="within task tolerance"):
        export_measured_commands(
            [probe], tmp_path / "invalid.json", device="cpu", maximum_position_error_m=limit
        )


def test_robust_export_requires_precise_center_and_both_neighbors(tmp_path):
    from gr00t.rl.pick.reach_calibration import export_robust_measured_commands

    _, probe, report = fixture(tmp_path)
    errors = [0.003, 0.002, 0.001, 0.012, 0.020]
    for key in (
        "final_root_states",
        "goals_world",
        "goal_quaternions_wxyz",
        "final_position_error_m",
        "final_orientation_error_rad",
        "goal_gripper_floor_clearance_m",
    ):
        report[key] *= 5
    report.update(
        synthetic_catalog="independent_catalog.json",
        synthetic_count=5,
        source_sample_ids=list(range(5)),
        final_position_error_m=errors,
        final_nonfoot_contact_peak_n=[0] * 5,
    )
    (probe / "summary.json").write_text(json.dumps(report))
    history = torch.zeros(25, 5, 28)
    history[..., :13] = torch.tensor(report["final_root_states"])
    history[..., -2] = torch.tensor(errors)
    torch.save(history, probe / "trace.pt")
    result = export_robust_measured_commands(probe, tmp_path / "robust.json", device="cpu")
    assert result["source_sample_ids"] == [1]
    assert result["maximum_position_error_m"] == 0.004
    assert result["robustness"]["nominal_longitudinal_radius_m"] == 0.03
    history[:, 0, -2] = 0.012
    torch.save(history, probe / "trace.pt")
    with pytest.raises(ValueError, match="verified longitudinal"):
        export_robust_measured_commands(probe, tmp_path / "invalid.json", device="cpu")
