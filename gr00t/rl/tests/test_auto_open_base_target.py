import numpy as np

from gr00t.rl.scripts.doorman_auto_open import (
    _approach_stop_latch,
    _base_pose_accepted,
    _through_stop_latch,
)
from gr00t.rl.auto_open.control import _base_commands_exact
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


def test_base_target_remains_a_stateless_feedback_controller():
    command, distance, yaw_error = _base_commands_exact(
        _state([0.05], [0.01]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float64),
    )
    assert distance[0] < 0.12
    assert yaw_error[0] == 0.0
    assert np.linalg.norm(command[0, :2]) > 0.0


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
