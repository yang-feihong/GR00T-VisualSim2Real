import inspect

import numpy as np
import torch

from gr00t.rl.auto_open import execution
from gr00t.rl.auto_open.control import (
    ALIGN_PANEL_ARRIVAL,
    APPROACH_ARRIVAL,
    THROUGH_TAIL_RELEASE_DISTANCE,
    THROUGH_TAIL_TARGET_DISTANCE,
)
from gr00t.rl.auto_open.control import (
    update_base_target_arrival as tensor_arrival,
)
from gr00t.rl.auto_open.execution import StageExecutionMixin
from gr00t.rl.auto_open.geometry import _resolved_door_prim_path
from gr00t.rl.auto_open.motion_math import track_base_pose
from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM


def _track_target(state, xy, yaw, *, minimum_speed_toward_target_only=False):
    roots = torch.as_tensor(state["root_states"])
    xy = torch.as_tensor(xy)
    yaw = torch.as_tensor(yaw)
    return track_base_pose(
        roots, xy, yaw, xy, yaw, minimum_speed_toward_target_only=minimum_speed_toward_target_only
    )


def _filter_reversal(controller, command):
    for name in (
        "base_command_previous",
        "base_reversal_pending_sign",
        "base_reversal_pending_steps",
        "base_reversal_lock_steps",
    ):
        setattr(controller, name, torch.as_tensor(getattr(controller, name)).reshape(1, 3))
    return StageExecutionMixin._stabilize_base(
        controller, torch.tensor([0]), torch.as_tensor(command).reshape(1, 3)
    )[0]


def _arrival(confirmed, steps, inner, outer, *, required_settle_frames):
    return tuple(
        value.item()
        for value in tensor_arrival(
            torch.as_tensor(confirmed),
            torch.as_tensor(steps),
            torch.as_tensor(inner),
            torch.as_tensor(outer),
            required_settle_frames=required_settle_frames,
        )
    )


def _state(x, speed):
    roots = np.zeros((len(x), 13), dtype=np.float32)
    roots[:, 0] = np.asarray(x, dtype=np.float32)
    roots[:, 3] = 1.0
    roots[:, 7] = np.asarray(speed, dtype=np.float32)
    return {"root_states": roots}


def test_base_target_keeps_closed_loop_active_until_stage_owns_settling():
    command, distance, yaw_error = _track_target(
        _state([0.05], [0.20]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float64),
        minimum_speed_toward_target_only=True,
    )
    assert distance[0] < 0.12
    assert yaw_error[0] == 0.0
    assert np.linalg.norm(command[0, :2]) > 0.0


def test_base_target_resumes_closed_loop_after_drifting_outside_tolerance():
    command, distance, _ = _track_target(
        _state([0.13], [0.0]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float64),
    )
    assert distance[0] > 0.12
    assert np.linalg.norm(command[0, :2]) > 0.0


def test_nonzero_translation_command_keeps_minimum_speed_near_target():
    command, _, _ = _track_target(
        _state([0.04], [0.0]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float64),
        minimum_speed_toward_target_only=True,
    )
    assert np.isclose(np.linalg.norm(command[0, :2]), 0.15)


def test_nonzero_yaw_command_keeps_minimum_rate_near_target():
    command, _, _ = _track_target(
        _state([0.0], [0.0]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.02], dtype=np.float64),
        minimum_speed_toward_target_only=True,
    )
    assert np.isclose(command[0, 2], 0.2)


def test_minimum_base_speed_does_not_amplify_a_braking_command():
    command, _, _ = _track_target(
        _state([0.20], [-0.25]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float64),
        minimum_speed_toward_target_only=True,
    )
    assert np.isclose(command[0, 0], 0.01)


def test_minimum_yaw_rate_does_not_amplify_a_braking_command():
    roots = _state([0.0], [0.0])["root_states"]
    roots[0, 12] = -0.5
    command, _, _ = _track_target(
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
    first = _filter_reversal(controller, np.asarray([0.2, 0.08, 0.1], dtype=np.float32))
    second = _filter_reversal(controller, np.asarray([-0.2, 0.06, 0.0], dtype=np.float32))
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
        _filter_reversal(controller, np.asarray([0.2, 0.0, 0.0], dtype=np.float32))
        for _ in range(3)
    ]
    assert np.allclose([output[0] for output in outputs], (-0.2, -0.2, 0.2))
    assert np.isclose(outputs[-1][0], 0.2)


def test_approach_large_reversal_crosses_zero_for_one_frame(monkeypatch):
    controller = StageExecutionMixin()
    controller.base_command_previous = torch.tensor([[0.2, -0.15, 0.2]])
    desired = torch.tensor([[-0.014, 0.04, -0.02]])
    monkeypatch.setattr(
        execution,
        "track_base_pose",
        lambda *args, **kwargs: (desired.clone(), torch.zeros(1), torch.zeros(1)),
    )
    rows = torch.tensor([0])
    first = controller.base_target(
        {"root_states": torch.zeros(1, 13)},
        rows,
        torch.zeros(1, 2),
        torch.zeros(1),
        zero_crossing=True,
    )[0]
    second = controller.base_target(
        {"root_states": torch.zeros(1, 13)},
        rows,
        torch.zeros(1, 2),
        torch.zeros(1),
        zero_crossing=True,
    )[0]
    torch.testing.assert_close(first, torch.zeros(1, 3))
    torch.testing.assert_close(second, desired)


def test_point_target_stages_use_the_shared_arrival_interface():
    for method_name in ("command_approach_handle", "command_move_to_panel_station"):
        source = inspect.getsource(getattr(TensorDoorFSM, method_name))
        assert "_track_station" in source
    for method_name in ("_track_station", "_base_hold_command"):
        source = inspect.getsource(getattr(TensorDoorFSM, method_name))
        assert "self.base_target(" in source
        assert "arrival=True" in source
    assert "_update_base_arrival" in inspect.getsource(TensorDoorFSM.base_target)


def test_base_target_remains_a_stateless_feedback_controller():
    command, distance, yaw_error = _track_target(
        _state([0.05], [0.01]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float64),
    )
    assert distance[0] < 0.12
    assert yaw_error[0] == 0.0
    assert np.linalg.norm(command[0, :2]) > 0.0


def test_base_arrival_requires_continuous_zero_command_window():
    state = (False, 0)
    for _ in range(24):
        confirmed, steps, zero_command = _arrival(*state, True, True, required_settle_frames=25)
        assert not confirmed
        assert zero_command
        state = (confirmed, steps)
    confirmed, steps, zero_command = _arrival(*state, True, True, required_settle_frames=25)
    assert (confirmed, steps, zero_command) == (True, 25, True)


def test_base_arrival_zero_command_starts_before_robot_has_stopped():
    assert _arrival(False, 0, True, True, required_settle_frames=25) == (False, 1, True)


def test_base_arrival_brakes_across_inner_to_outer_region():
    state = _arrival(False, 7, False, True, required_settle_frames=25)
    assert state == (False, 8, True)


def test_base_arrival_holds_outer_bound_then_requires_fresh_reacquisition():
    assert _arrival(True, 25, False, True, required_settle_frames=25) == (
        True,
        25,
        True,
    )
    released = _arrival(True, 25, False, False, required_settle_frames=25)
    assert released == (False, 0, False)
    assert _arrival(*released[:2], False, True, required_settle_frames=25) == (
        False,
        0,
        False,
    )


def test_base_arrival_single_and_batched_tensors_match():
    single = tensor_arrival(
        torch.tensor(False),
        torch.tensor(24),
        torch.tensor(False),
        torch.tensor(True),
        required_settle_frames=25,
    )
    batch = tensor_arrival(
        torch.tensor([False] * 9),
        torch.tensor([24] * 9),
        torch.tensor([False] * 9),
        torch.tensor([True] * 9),
        required_settle_frames=25,
    )
    for scalar, vector in zip(single, batch):
        assert torch.equal(scalar.expand_as(vector), vector)
    assert tuple(v.item() for v in single) == (True, 25, True)


def test_point_target_stages_own_independent_arrival_profiles():
    profiles = (APPROACH_ARRIVAL, ALIGN_PANEL_ARRIVAL)
    assert len({id(profile) for profile in profiles}) == 2
    assert APPROACH_ARRIVAL.inner_distance_m == ALIGN_PANEL_ARRIVAL.inner_distance_m
    assert APPROACH_ARRIVAL.outer_distance_m == ALIGN_PANEL_ARRIVAL.outer_distance_m
    assert APPROACH_ARRIVAL.inner_yaw_rad == ALIGN_PANEL_ARRIVAL.inner_yaw_rad
    assert APPROACH_ARRIVAL.outer_yaw_rad == ALIGN_PANEL_ARRIVAL.outer_yaw_rad
    assert APPROACH_ARRIVAL.settle_frames == ALIGN_PANEL_ARRIVAL.settle_frames


def test_point_target_profiles_use_benchmarked_thresholds():
    for profile in (APPROACH_ARRIVAL, ALIGN_PANEL_ARRIVAL):
        assert profile.inner_distance_m == 0.02
        assert profile.outer_distance_m == 0.10
        assert profile.inner_yaw_rad == 0.05
        assert profile.outer_yaw_rad == 0.15
        assert profile.settle_frames == 25
        assert profile.max_linear_speed_m_s == 0.02
        assert profile.max_angular_speed_rad_s == 0.05


def test_semantic_target_stages_stop_on_their_physical_condition():
    base_advance = inspect.getsource(TensorDoorFSM.command_advance_to_next_push_station)
    through = inspect.getsource(TensorDoorFSM.command_pass_through_doorway)
    assert "task_stop=reachable" in base_advance
    assert "self.base_target(" in base_advance
    assert "_update_base_arrival(" not in base_advance
    assert "tail_clearance >= THROUGH_TAIL_TARGET_DISTANCE" in through
    assert "tail_clearance >= THROUGH_TAIL_RELEASE_DISTANCE" in through
    assert "_update_base_arrival(" not in through


def test_point_arrival_inner_region_is_strictly_inside_outer_region():
    for profile in (APPROACH_ARRIVAL, ALIGN_PANEL_ARRIVAL):
        assert 0 < profile.inner_distance_m < profile.outer_distance_m
        assert 0 < profile.inner_yaw_rad < profile.outer_yaw_rad


def test_through_tail_hysteresis_has_five_centimeter_release_band():
    assert np.isclose(
        THROUGH_TAIL_TARGET_DISTANCE - THROUGH_TAIL_RELEASE_DISTANCE,
        0.05,
    )


def test_door_prim_path_uses_scene_resolver_instead_of_guessing_name():
    door = type(
        "Door", (), {"prim_path_for_env": lambda self, env_id: f"/env/{env_id}/door_left"}
    )()
    env = type("Env", (), {"door": door})()
    assert _resolved_door_prim_path(env, 3) == "/env/3/door_left"
