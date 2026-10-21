from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.rl.auto_open.geometry import _quat_axis_angle_np, _quat_mul_np
from gr00t.rl.auto_open.pregrasp_reachability import (
    PREGRASP_SAMPLE_POSITION_STEP_M,
    PREGRASP_SAMPLE_ROTATION_STEP_RAD,
    pregrasp_reachability_samples,
    select_pregrasp_candidate_from_errors,
)


@pytest.fixture(autouse=True)
def lowlevel_import_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "third_party/visual_wholebody_lowlevel"))


def _family(hinge_angle=0.17):
    return pregrasp_reachability_samples(
        [.8, .3, .9], [1., 0, 0, 0], [.65, .3, .9], [1., 0, 0, 0],
        handle_axle_world=[.6, .3, 1.], handle_axis_world=[1., 0, 0],
        hinge_world=[0., 0, 0], hinge_rotation_rad=hinge_angle,
        handle_start_angle_rad=0., handle_target_angle_rad=.74,
    )


@pytest.mark.parametrize("hinge_angle", [-.17, .17])
def test_pose_family_covers_three_rigid_rotations_with_bounded_spacing(hinge_angle):
    family = _family(hinge_angle)
    positions = family["reachability_positions"]
    quats = family["reachability_orientations"]
    labels = family["reachability_samples"]
    np.testing.assert_allclose(positions[0], [.8, .3, .9])
    insertion = [i for i, s in enumerate(labels) if s["stage"] == "insertion"]
    np.testing.assert_allclose(positions[insertion[-1]], [.65, .3, .9])
    assert {s["stage"] for s in labels} == {
        "insertion", "press_handle", "push_pull_door", "return_handle"
    }
    groups = {}
    for index, sample in enumerate(labels):
        groups.setdefault((sample["stage"], sample["press_branch_rad"]), []).append(index)
    assert len([key for key in groups if key[0] == "push_pull_door"]) == 3
    for (stage, branch), indices in groups.items():
        p, q = positions[indices], quats[indices]
        assert np.linalg.norm(np.diff(p, axis=0), axis=1).max() <= PREGRASP_SAMPLE_POSITION_STEP_M + 1e-6
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        steps = 2 * np.arccos(np.clip(abs((q[1:] * q[:-1]).sum(axis=1)), 0, 1))
        assert steps.max() <= PREGRASP_SAMPLE_ROTATION_STEP_RAD + 1e-6
        if stage == "push_pull_door":
            np.testing.assert_allclose(p[:, 2], p[0, 2], atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(p[:, :2], axis=1), np.linalg.norm(p[0, :2]), atol=1e-6)
            rotation = _quat_axis_angle_np([0, 0, 1], hinge_angle)
            np.testing.assert_allclose(q[-1], _quat_mul_np(rotation, q[0]), atol=1e-6)
        if stage == "return_handle":
            np.testing.assert_allclose(q[-1], _quat_axis_angle_np([0, 0, 1], hinge_angle), atol=1e-6)
    np.testing.assert_allclose(
        sorted({s["press_branch_rad"] for s in labels if s["stage"] == "push_pull_door"}),
        [np.deg2rad(37.5), (np.deg2rad(37.5) + .74) / 2, .74],
    )


def _candidate(pitch, count):
    return {
        "pitch_deg": pitch, "end_position": [.03, 0, 0], "orientation": [1., 0, 0, 0],
        "reachability_positions": np.c_[np.arange(count), np.zeros((count, 2))],
        "reachability_orientations": np.array([
            _quat_axis_angle_np([0, 0, 1], index * .01) for index in range(count)
        ]),
        "reachability_samples": [
            {"stage": "push_pull_door", "angle_rad": index * .01, "press_branch_rad": .74}
            for index in range(count)
        ],
    }


def test_ragged_selection_rejects_rotation_failure_and_preserves_pregrasp_target():
    candidates = [_candidate(0, 3), _candidate(5, 5)]
    selected, exact, diagnostics = select_pregrasp_candidate_from_errors(
        candidates, [[0, .002, 0], np.zeros(5)], [np.zeros(3), np.zeros(5)]
    )
    assert exact and selected["pitch_deg"] == 5
    assert selected["end_position"] == [.03, 0, 0]
    assert diagnostics[0]["first_failure"]["stage"] == "push_pull_door"
    assert diagnostics[0]["first_failure"]["sample_index"] == 1
    assert [row["sample_count"] for row in diagnostics] == [3, 5]


def test_nonfinite_inverse_result_cannot_qualify_pitch():
    _, exact, diagnostics = select_pregrasp_candidate_from_errors(
        [_candidate(0, 2)], [[0, np.nan]], [[0, 0]]
    )
    assert not exact
    assert diagnostics[0]["first_failure"]["sample_index"] == 1


def test_gpu_selection_submits_all_ragged_env_candidates_once(monkeypatch):
    from gr00t.rl.auto_open import tensor_planning

    calls = []

    def inverse(positions, orientations, **kwargs):
        calls.append((positions.clone(), orientations.clone(), kwargs))
        return SimpleNamespace(position_error=torch.zeros_like(positions), orientation_error=torch.zeros_like(positions))

    monkeypatch.setattr(tensor_planning, "batched_steady_ee_inverse_no_gravity", inverse)
    coordinator = tensor_planning.TensorPlanningCoordinator.__new__(tensor_planning.TensorPlanningCoordinator)
    env = SimpleNamespace(device="cpu", arm_joint_ids=torch.arange(6), dof_pos_limits=torch.tensor([[-3., 3.]] * 6))
    controllers = [SimpleNamespace(env=env), SimpleNamespace(env=env)]
    coordinator.controllers = controllers
    state = {
        "dof_names": [f"joint{i}" for i in range(1, 7)], "dof_pos": np.zeros((1, 6)),
        "root_states": np.array([[0, 0, .5, 1, 0, 0, 0]]),
    }
    candidates = [[_candidate(0, 3), _candidate(5, 4)], [_candidate(0, 2), _candidate(5, 6)]]
    selected = coordinator._select_pregrasp_gpu_batch([
        (i, controllers[i], state, value) for i, value in enumerate(candidates)
    ])
    assert len(calls) == 1
    assert calls[0][0].shape == (15, 3)
    np.testing.assert_allclose(calls[0][1], np.concatenate([c["reachability_orientations"] for group in candidates for c in group]))
    assert calls[0][2]["joint_limits"].shape == (15, 6, 2)
    assert all(result[1] for result in selected)
    assert [len(result[2]) for result in selected] == [2, 2]


def test_no_reachable_pitch_fails_without_installing_best_effort_path(monkeypatch):
    from gr00t.rl.auto_open import tensor_planning

    coordinator = tensor_planning.TensorPlanningCoordinator.__new__(tensor_planning.TensorPlanningCoordinator)
    controller = SimpleNamespace()
    coordinator.controllers = [controller]
    coordinator._state = lambda *_: {}
    candidate = {"pitch_deg": 0., "end_position": np.zeros(3), "orientation": np.array([1., 0, 0, 0])}
    monkeypatch.setattr(tensor_planning, "_pregrasp_pitch_candidates", lambda _: [candidate])
    monkeypatch.setattr(tensor_planning, "_perturb_ee_target", lambda _, label, p, q: (p, q))
    monkeypatch.setattr(tensor_planning, "_attach_pregrasp_reachability", lambda _, c, *args: c)
    diagnostics = [{"pitch_deg": 0., "exact": False}]
    coordinator._select_pregrasp_gpu_batch = lambda _: [(candidate, False, diagnostics)]
    failures = []
    fsm = SimpleNamespace(device="cpu", fail=lambda rows, code: failures.append((rows.tolist(), code)))
    coordinator.plan_pregrasp(fsm, {}, torch.tensor([0]))
    assert failures == [([0], 47)]
    assert controller.pregrasp_pitch_diagnostics == diagnostics
