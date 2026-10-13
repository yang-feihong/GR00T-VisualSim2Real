import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import gr00t.rl.auto_open.planning as retract_planning
from gr00t.rl.auto_open.cli import DEFAULT_ARM_RETRACT_POSE_LOCAL
from gr00t.rl.auto_open.control import _ee_world_orientation_target
from gr00t.rl.auto_open.planning import (
    _sample_ball_uniform_volume,
    _sample_reachable_arm_retract_pose,
)
from gr00t.rl.scripts.doorman_auto_open import DoorController
from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.ee_steady_state import (
    B2Z1SteadyEeForwardModel,
)


class _ReferenceFrame:
    def __init__(self, quaternion):
        self.quaternion = np.asarray(quaternion, dtype=np.float32).reshape(1, 4)

    def get_goal_reference_quat(self):
        return self.quaternion


def test_retract_pose_is_one_explicit_six_dof_controller_default():
    configured = np.asarray(DEFAULT_ARM_RETRACT_POSE_LOCAL)
    parameter = inspect.signature(DoorController).parameters[
        "arm_retract_pose_local"
    ]

    assert configured.shape == (6,)
    assert np.isfinite(configured).all()
    np.testing.assert_allclose(
        configured,
        [0.30, 0.0, 0.30, -np.pi / 2.0, 0.0, 0.0],
        atol=1.0e-12,
    )
    assert parameter.default == DEFAULT_ARM_RETRACT_POSE_LOCAL


def test_retract_orientation_uses_goal_reference_frame_xyz_euler():
    controller = SimpleNamespace(env=_ReferenceFrame([1.0, 0.0, 0.0, 0.0]))

    quaternion = _ee_world_orientation_target(
        controller, [0.0, 0.0, np.pi / 2.0]
    )

    np.testing.assert_allclose(
        quaternion,
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        atol=1.0e-6,
    )


def test_retract_orientation_composes_with_world_reference():
    quarter_turn = np.asarray(
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32
    )
    controller = SimpleNamespace(env=_ReferenceFrame(quarter_turn))

    quaternion = _ee_world_orientation_target(
        controller, [0.0, 0.0, np.pi / 2.0]
    )

    np.testing.assert_allclose(quaternion, [0.0, 0.0, 0.0, 1.0], atol=1.0e-6)


def test_retract_position_sampler_is_uniform_volume_and_bounded():
    offsets = _sample_ball_uniform_volume(
        np.random.default_rng(1700), 100_000, 0.05
    )
    radii = np.linalg.norm(offsets, axis=1)

    assert np.max(radii) <= 0.05
    assert abs(float(np.mean(radii)) - 0.0375) < 3.0e-4
    np.testing.assert_allclose(np.mean(offsets, axis=0), 0.0, atol=3.0e-4)


class _RetractEnvironment:
    device = "cpu"

    def get_goal_reference_origin(self):
        return np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    def get_goal_reference_quat(self):
        return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def get_goal_center_offset_local(self):
        return np.asarray([[0.2, 0.0, 0.7]], dtype=np.float32)


def _retract_selector_fixture(seed):
    repository = Path(__file__).resolve().parents[3]
    model = B2Z1SteadyEeForwardModel(
        repository
        / "gr00t/rl/third_party/visual_wholebody_lowlevel/resources/robots/b2z1/urdf/generated/b2z1_mount_90_x0p22_y0_z0p17.urdf",
        stiffness=1000.0,
    )
    controller = SimpleNamespace(
        arm_retract_entry_count=0,
        arm_retract_random_seed=seed,
        arm_retract_pose_local=np.asarray(
            DEFAULT_ARM_RETRACT_POSE_LOCAL, dtype=np.float32
        ),
        arm_retract_position_noise_radius_m=0.05,
        arm_retract_rotation_noise_deg=5.0,
        arm_retract_randomization_records=[],
        active_arm_retract_pose_local=None,
        ee_steady_model=model,
        env=_RetractEnvironment(),
    )
    state = {
        "dof_names": list(model.arm_joint_names),
        "dof_pos": np.asarray(
            [[-1.57, 1.60, -0.63, -1.10, 0.0, 0.0]], dtype=np.float64
        ),
        "root_states": np.asarray(
            [[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64
        ),
    }
    return controller, state


def test_retract_pose_is_resampled_per_entry_and_exactly_reachable():
    controller, state = _retract_selector_fixture(1700)
    first = _sample_reachable_arm_retract_pose(
        controller, state, "RESTORE_GRIPPER"
    ).copy()
    second = _sample_reachable_arm_retract_pose(
        controller, state, "PUSH_ARM_RETRACT"
    ).copy()

    assert not np.array_equal(first, second)
    for pose in (first, second):
        position_delta = pose[:3] - np.asarray(DEFAULT_ARM_RETRACT_POSE_LOCAL[:3])
        rpy_delta = pose[3:] - np.asarray(DEFAULT_ARM_RETRACT_POSE_LOCAL[3:])
        assert np.linalg.norm(position_delta) <= 0.05 + 1.0e-7
        assert np.max(np.abs(rpy_delta)) <= math.radians(5.0) + 1.0e-7
    assert [record["stage"] for record in controller.arm_retract_randomization_records] == [
        "RESTORE_GRIPPER",
        "PUSH_ARM_RETRACT",
    ]

    duplicate, duplicate_state = _retract_selector_fixture(1700)
    duplicate_first = _sample_reachable_arm_retract_pose(
        duplicate, duplicate_state, "RESTORE_GRIPPER"
    )
    np.testing.assert_array_equal(first, duplicate_first)


def test_unreachable_retract_candidate_is_recorded_and_never_dispatched(
    monkeypatch,
):
    controller, state = _retract_selector_fixture(1700)

    def fake_ik(
        root_position,
        root_quaternion,
        target_position,
        target_rotation,
        initial_joint_position,
        joint_lower,
        joint_upper,
        **_kwargs,
    ):
        count = target_position.shape[0]
        position_error = torch.zeros(count, dtype=target_position.dtype)
        position_error[0] = 0.01
        orientation_error = torch.zeros_like(position_error)
        solutions = initial_joint_position.clone()
        return solutions, position_error, orientation_error

    monkeypatch.setattr(retract_planning, "solve_b2z1_arm_ik", fake_ik)
    accepted = _sample_reachable_arm_retract_pose(
        controller, state, "RESTORE_GRIPPER"
    )
    record = controller.arm_retract_randomization_records[0]

    assert record["unreachable_candidates"] == 1
    assert record["rejected_before_acceptance"] == 1
    assert record["draws_until_acceptance"] == 2
    assert np.isclose(record["rejected"][0]["position_error_m"], 0.01)
    assert not np.array_equal(accepted, record["rejected"][0]["pose_local"])
