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


def _gpu_pregrasp_case(fail_all=False):
    from gr00t.rl.auto_open.tensor_stage_planning import GPUStagePlanningMixin
    from gr00t.rl.auto_open.tensor_fsm import PaddedPathBank

    coordinator = GPUStagePlanningMixin()
    coordinator.path_diagnostics = []
    coordinator.gpu_stage_metadata = []
    coordinator._record_stats = lambda *args, **kwargs: None
    position = torch.tensor([[[.8, .3, .9], [.8, .3, .91]],
                             [[.7, .3, .9], [.7, .3, .91]]])
    quat = torch.tensor([1., 0, 0, 0]).expand(2, 2, -1).clone()
    coordinator.gpu_stage_constants = {
        "position": position, "quat": quat, "grasp": position - torch.tensor([.15, 0, 0]),
        "grasp_q": quat, "selected_pitch": torch.zeros(2, dtype=torch.int64),
        "pitch_deg": torch.tensor([0., 5.]),
        "nominal_q": quat, "nominal": position,
    }
    failures, calls = [], []
    fsm = SimpleNamespace(
        device="cpu", Stage=SimpleNamespace(PREGRASP=1), panel_body_idx=0, handle_body_idx=1,
        handle_joint_idx=0, handle_joint_axis_panel_local=torch.tensor([[1., 0, 0]]).expand(2, -1),
        hinge_axis_sign=torch.tensor([-1., 1.]), turn_gap_angle=torch.tensor([.1, .2]),
        turn_gap_margin=torch.zeros(2), turn_target_margin=torch.zeros(2), hinge_world=torch.zeros((2, 3)),
        arm_align_path=PaddedPathBank(2, 512, "cpu"), arm_align_desired_target_world=torch.zeros((2, 3)),
        arm_align_target_valid=torch.zeros(2, dtype=torch.bool), stage_steps=torch.zeros(2, dtype=torch.int64),
        arm_stability=SimpleNamespace(clear=lambda rows: None), transition=lambda *args: None,
        fail=lambda rows, code: failures.extend((int(row), code) for row in rows))
    bodies = torch.zeros((2, 2, 13))
    bodies[..., 3] = 1
    bodies[:, 1, :3] = torch.tensor([.6, .3, 1.])
    state = {"door_body_state_w": bodies, "door_joint_position": torch.zeros((2, 1)),
             "ee_pos": torch.tensor([[.8, .3, .9], [.7, .3, .9]]), "ee_orn": quat[:, 0]}

    def inverse(fsm, state, rows, positions, orientations, lengths):
        owner, sample = torch.where(torch.arange(positions.shape[1])[None] < lengths[:, None])
        calls.append((rows.clone(), positions.clone(), orientations.clone(), lengths.clone()))
        error = torch.zeros((len(owner), 3))
        if len(calls) == 1:
            # Reject first candidate by one sample; or reject every sample.
            error[:, 0] = .01 if fail_all else 0.
            error[0, 0] = .01
        result = SimpleNamespace(position_error=error, orientation_error=torch.zeros_like(error),
                                 command_position_world=positions[owner, sample],
                                 command_orientation_world_wxyz=orientations[owner, sample])
        return result, owner, sample

    coordinator._inverse_pose_bank = inverse
    return coordinator, fsm, state, calls, failures


def test_gpu_selection_submits_all_ragged_env_candidates_once():
    coordinator, fsm, state, calls, failures = _gpu_pregrasp_case()
    coordinator.plan_pregrasp(fsm, state, torch.tensor([0, 1]))
    assert len(calls) == 2  # One complete family solve, one selected alignment solve.
    family_rows, family_positions, _, _ = calls[0]
    assert family_positions.shape[1:] == (1, 3)
    assert set(family_rows.tolist()) == {0, 1}
    assert coordinator.gpu_stage_constants["selected_pitch"].tolist() == [1, 0]
    torch.testing.assert_close(fsm.arm_align_desired_target_world,
        torch.tensor([[.8, .3, .91], [.7, .3, .9]]))
    assert not failures


def test_no_reachable_pitch_fails_without_installing_best_effort_path():
    coordinator, fsm, state, calls, failures = _gpu_pregrasp_case(fail_all=True)
    coordinator.plan_pregrasp(fsm, state, torch.tensor([0, 1]))
    assert failures == [(0, 47), (1, 47)]
    assert len(calls) == 1
    assert not fsm.arm_align_target_valid.any()
    assert fsm.arm_align_path.length.sum() == 0


def test_pregrasp_controller_metadata_is_materialized_offline_only():
    coordinator, fsm, state, _, _ = _gpu_pregrasp_case()
    controllers = []
    names = ("center_standoff_ee_world", "center_standoff_nominal_world", "center_insert_waypoint_world",
             "center_insert_nominal_world", "grasp_waypoint_world", "grasp_nominal_world")
    for _ in range(2):
        controller = SimpleNamespace(pregrasp_pitch_deg=-1.,
            open_template={"tip_capture_center": np.zeros(3, dtype=np.float32)})
        for name in names:
            setattr(controller, name, np.zeros(3, dtype=np.float32))
        controllers.append(controller)
    coordinator.controllers = controllers
    coordinator.plan_pregrasp(fsm, state, torch.tensor([0, 1]))
    assert [c.pregrasp_pitch_deg for c in controllers] == [-1., -1.]
    coordinator.finalize_gpu_stage_diagnostics()
    assert [c.pregrasp_pitch_deg for c in controllers] == [5., 0.]
    assert controllers[0].pregrasp_pitch_diagnostics[0]["first_failure"]["sample_index"] == 0
    assert not controllers[0].pregrasp_pitch_diagnostics[0]["exact"]
    assert controllers[0].pregrasp_pitch_diagnostics[1]["exact"]
    for controller in controllers:
        for candidate in controller.pregrasp_pitch_diagnostics:
            assert len(candidate["position_errors_m"]) == candidate["sample_count"]
            assert len(candidate["orientation_errors"]) == candidate["sample_count"]
    assert not coordinator.gpu_stage_metadata
