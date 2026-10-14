import inspect

import numpy as np

from gr00t.rl.scripts.doorman_auto_open import (
    DoorController,
    _approach_stop_latch,
    _base_pose_accepted,
    _through_stop_latch,
)
from gr00t.rl.auto_open.control import (
    _base_commands_exact,
    _stabilize_base_command,
    _zero_cross_base_reversal,
)
from gr00t.rl.auto_open.geometry import _resolved_door_prim_path


def _state(x, speed):
    roots = np.zeros((len(x), 13), dtype=np.float32)
    roots[:, 0] = np.asarray(x, dtype=np.float32)
    roots[:, 3] = 1.0
    roots[:, 7] = np.asarray(speed, dtype=np.float32)
    return {"root_states": roots}


def test_base_target_keeps_closed_loop_active_until_stage_owns_settling():
    command, distance, yaw_error = _base_commands_exact(
        _state([0.05], [0.20]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float64),
        minimum_speed_toward_target_only=True,
    )
    assert distance[0] < 0.12
    assert yaw_error[0] == 0.0
    assert np.linalg.norm(command[0, :2]) > 0.0


def test_base_target_resumes_closed_loop_after_drifting_outside_tolerance():
    command, distance, _ = _base_commands_exact(
        _state([0.13], [0.0]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float64),
    )
    assert distance[0] > 0.12
    assert np.linalg.norm(command[0, :2]) > 0.0


def test_minimum_base_speed_does_not_amplify_a_braking_command():
    command, _, _ = _base_commands_exact(
        _state([0.20], [-0.25]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float64),
        minimum_speed_toward_target_only=True,
    )
    assert np.isclose(command[0, 0], 0.01)


def test_minimum_yaw_rate_does_not_amplify_a_braking_command():
    roots = _state([0.0], [0.0])["root_states"]
    roots[0, 12] = -0.5
    command, _, _ = _base_commands_exact(
        {"root_states": roots},
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([-0.1], dtype=np.float64),
        minimum_speed_toward_target_only=True,
    )
    assert np.isclose(command[0, 2], 0.025)


def test_dispatched_base_command_rejects_short_reversal():
    controller = type(
        "Controller",
        (),
        {
            "base_command_previous": np.asarray([-0.2, 0.1, 0.0], dtype=np.float32),
            "base_reversal_pending_sign": np.zeros(3, dtype=np.int8),
            "base_reversal_pending_steps": np.zeros(3, dtype=np.int32),
            "base_reversal_lock_steps": np.zeros(3, dtype=np.int32),
        },
    )()
    first = _stabilize_base_command(
        controller, np.asarray([0.2, 0.08, 0.1], dtype=np.float32)
    )
    second = _stabilize_base_command(
        controller, np.asarray([-0.2, 0.06, 0.0], dtype=np.float32)
    )
    assert np.allclose(first, [-0.2, 0.08, 0.1])
    assert np.allclose(second, [-0.2, 0.06, 0.0])


def test_dispatched_base_command_allows_sustained_reversal():
    controller = type(
        "Controller",
        (),
        {
            "base_command_previous": np.asarray([-0.2, 0.0, 0.0], dtype=np.float32),
            "base_reversal_pending_sign": np.zeros(3, dtype=np.int8),
            "base_reversal_pending_steps": np.zeros(3, dtype=np.int32),
            "base_reversal_lock_steps": np.zeros(3, dtype=np.int32),
        },
    )()
    outputs = [
        _stabilize_base_command(
            controller, np.asarray([0.2, 0.0, 0.0], dtype=np.float32)
        )
        for _ in range(3)
    ]
    assert np.allclose([output[0] for output in outputs], (-0.2, -0.2, 0.2))
    assert np.isclose(outputs[-1][0], 0.2)


def test_approach_large_reversal_crosses_zero_for_one_frame():
    controller = type(
        "Controller",
        (),
        {"base_command_previous": np.asarray([0.2, -0.15, 0.2], dtype=np.float32)},
    )()
    first = _zero_cross_base_reversal(
        controller, np.asarray([-0.014, 0.04, -0.02], dtype=np.float32)
    )
    second = _zero_cross_base_reversal(
        controller, np.asarray([-0.014, 0.04, -0.02], dtype=np.float32)
    )
    assert np.allclose(first, [0.0, 0.0, 0.0])
    assert np.allclose(second, [-0.014, 0.04, -0.02])


def test_base_stabilizer_runs_once_at_the_final_dispatch_boundary():
    approach_source = inspect.getsource(DoorController._command_approach)
    command_source = inspect.getsource(DoorController.command)
    assert approach_source.count("_zero_cross_base_reversal(") == 1
    assert command_source.count("_stabilize_base_command(") == 1
    assert "return self._command_approach(state)" in command_source


def test_base_target_remains_a_stateless_feedback_controller():
    command, distance, yaw_error = _base_commands_exact(
        _state([0.05], [0.01]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float64),
    )
    assert distance[0] < 0.12
    assert yaw_error[0] == 0.0
    assert np.linalg.norm(command[0, :2]) > 0.0


def test_terminal_hold_overrides_reversal_filter_with_exact_zero():
    command_source = inspect.getsource(DoorController.command)
    assert "self.stage == Stage.THROUGH and self.through_stop_latched" in command_source
    assert "base_cmd[:] = 0.0" in command_source


def test_approach_owns_stop_latch_until_pose_leaves_acceptance():
    assert not _approach_stop_latch(False, 0.05, 0.01, 0.03, 0.01)
    assert _approach_stop_latch(False, 0.05, 0.01, 0.01, 0.01)
    assert _approach_stop_latch(True, 0.11, 0.07, 0.20, 0.20)
    assert not _approach_stop_latch(True, 0.13, 0.07, 0.01, 0.01)


def test_terminal_pose_acceptance_uses_the_task_position_and_yaw_bounds():
    assert _base_pose_accepted(0.119, 0.079)
    assert not _base_pose_accepted(0.121, 0.0)
    assert not _base_pose_accepted(0.0, 0.081)


def test_through_stop_latch_uses_inner_target_and_outer_acceptance():
    assert not _through_stop_latch(False, 0.10, 0.01)
    assert not _through_stop_latch(False, 0.07, 0.06)
    assert _through_stop_latch(False, 0.07, 0.04)
    assert _through_stop_latch(True, 0.11, 0.07)
    assert not _through_stop_latch(True, 0.13, 0.07)


def test_door_prim_path_uses_scene_resolver_instead_of_guessing_name():
    door = type("Door", (), {"prim_path_for_env": lambda self, env_id: f"/env/{env_id}/door_left"})()
    env = type("Env", (), {"door": door})()
    assert _resolved_door_prim_path(env, 3) == "/env/3/door_left"
