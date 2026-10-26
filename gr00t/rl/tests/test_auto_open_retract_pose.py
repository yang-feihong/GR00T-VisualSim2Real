"""Retract configuration and current GPU candidate selection contracts."""

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.auto_open.arm_model import ARM_JOINT_NAMES, read_arm_joint_limits
from gr00t.rl.auto_open.cli import DEFAULT_ARM_RETRACT_POSE_LOCAL, _parser
from gr00t.rl.auto_open.tensor_stage_planning import GPUStagePlanningMixin
from gr00t.rl.scripts.doorman_auto_open import DoorController


def test_retract_pose_is_one_explicit_six_dof_controller_default():
    configured = np.asarray(DEFAULT_ARM_RETRACT_POSE_LOCAL)
    parameter = inspect.signature(DoorController).parameters["arm_retract_pose_local"]

    assert configured.shape == (6,)
    assert np.isfinite(configured).all()
    np.testing.assert_allclose(
        configured,
        [0.30, 0.0, 0.30, -np.pi / 2.0, 0.0, 0.0],
        atol=1.0e-12,
    )
    assert parameter.default == DEFAULT_ARM_RETRACT_POSE_LOCAL


def test_retract_randomization_defaults_match_offline_checked_range():
    parser = _parser()
    parameters = inspect.signature(DoorController).parameters
    for name, expected in (
        ("arm_retract_position_noise_radius_m", 0.01),
        ("arm_retract_rotation_noise_deg", 1.0),
    ):
        assert parser.get_default(name) == expected
        assert parameters[name].default == expected


def test_static_arm_limits_preserve_urdf_bounds_and_margin():
    path = (
        Path(__file__).parents[1]
        / "third_party/visual_wholebody_lowlevel/resources/robots/b2z1/urdf/b2z1.urdf"
    )
    tree = ET.parse(path)
    expected = np.array(
        [
            [
                float(tree.find(f"joint[@name='{name}']/limit").get(key))
                for key in ("lower", "upper")
            ]
            for name in ARM_JOINT_NAMES
        ]
    )
    expected[:, 0] += 0.001
    expected[:, 1] -= 0.001
    actual = read_arm_joint_limits(path)
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
    with pytest.raises(ValueError, match="Invalid arm joint"):
        read_arm_joint_limits(path, margin_rad=10.0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("reject_all", [False, True])
def test_gpu_retract_selection_preserves_rejection_records_and_never_dispatches_bad_pose(
    monkeypatch, device, reject_all
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    planner = GPUStagePlanningMixin()
    planner.arm_joint_ids = torch.arange(6, device=device)
    planner.gpu_stage_metadata, planner.path_diagnostics = [], []
    base = torch.tensor([DEFAULT_ARM_RETRACT_POSE_LOCAL], device=device)
    planner.gpu_stage_constants = {
        "retract_entries": torch.zeros(1, dtype=torch.int64, device=device),
        "retract_seeds": torch.tensor([1700], device=device),
        "arm_retract_pose_local": base,
        "arm_retract_position_noise_radius_m": torch.tensor([0.01], device=device),
        "arm_retract_rotation_noise_deg": torch.tensor([1.0], device=device),
        "retract_lower": torch.full((1, 6), -3.0, device=device),
        "retract_upper": torch.full((1, 6), 3.0, device=device),
    }
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    env = SimpleNamespace(
        get_goal_reference_quat=lambda: q,
        get_goal_reference_origin=lambda: torch.zeros(1, 3, device=device),
        get_goal_center_offset_local=lambda: torch.zeros(1, 3, device=device),
    )
    failed = []
    fsm = SimpleNamespace(
        env=env,
        device=device,
        stage=torch.ones(1, device=device),
        active_arm_retract_pose_local=torch.full((1, 6), -9.0, device=device),
        fail=lambda rows, code: failed.append((rows.clone(), code)),
    )
    roots = torch.zeros(1, 13, device=device)
    roots[:, 3] = 1.0
    state = {"root_states": roots, "dof_pos": torch.zeros(1, 6, device=device)}

    def ik(*args, **kwargs):
        solution = torch.zeros_like(args[4])
        error = torch.zeros(len(solution), device=device, dtype=solution.dtype)
        error[:] = 0.01 if reject_all else 0.0
        error[0] = 0.01
        return solution, error, torch.zeros_like(error)

    monkeypatch.setattr("gr00t.rl.auto_open.planning_ik.solve_candidate_ik", ik)
    result = planner._sample_gpu_retract(fsm, state, torch.tensor([0], device=device))
    diagnostic = planner.gpu_stage_metadata[0][2]
    assert not diagnostic["reachable"][0, 0]
    assert diagnostic["position_error"][0, 0].item() == pytest.approx(0.01)
    if reject_all:
        assert not len(result)
        assert fsm.active_arm_retract_pose_local.eq(-9.0).all()
        assert failed[-1][1] == 46 and failed[-1][0].tolist() == [0]
    else:
        assert result.tolist() == [0]
        assert planner.gpu_stage_metadata[-1][2]["draws_until_acceptance"].item() == 2
        torch.testing.assert_close(
            fsm.active_arm_retract_pose_local[0], diagnostic["poses"][0, 1].float()
        )
