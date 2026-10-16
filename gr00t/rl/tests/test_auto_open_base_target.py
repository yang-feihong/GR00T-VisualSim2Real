import inspect

import numpy as np
import torch

from gr00t.rl.scripts.doorman_auto_open import (
    DoorController,
)
from gr00t.rl.auto_open.control import (
    ALIGN_PANEL_ARRIVAL,
    APPROACH_ARRIVAL,
    THROUGH_TAIL_RELEASE_DISTANCE,
    THROUGH_TAIL_TARGET_DISTANCE,
    _base_commands_exact,
    base_pose_arrival_regions,
    _stabilize_base_command,
    _zero_cross_base_reversal,
    update_base_target_arrival,
)
from gr00t.rl.auto_open.geometry import _resolved_door_prim_path
from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM


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


def test_nonzero_translation_command_keeps_minimum_speed_near_target():
    command, _, _ = _base_commands_exact(
        _state([0.04], [0.0]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float64),
        minimum_speed_toward_target_only=True,
    )
    assert np.isclose(np.linalg.norm(command[0, :2]), 0.15)


def test_nonzero_yaw_command_keeps_minimum_rate_near_target():
    command, _, _ = _base_commands_exact(
        _state([0.0], [0.0]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([0.02], dtype=np.float64),
        minimum_speed_toward_target_only=True,
    )
    assert np.isclose(command[0, 2], 0.2)


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


def test_point_target_stages_use_the_shared_arrival_interface():
    assert "_update_base_arrival(" in inspect.getsource(
        DoorController._command_approach
    )
    assert "_update_base_arrival(" in inspect.getsource(DoorController.command)
    for method_name in (
        "command_approach",
        "_base_hold_command",
        "command_align_panel",
    ):
        source = inspect.getsource(getattr(TensorDoorFSM, method_name))
        assert "_update_base_arrival" in source


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
    assert "if stop_for_arrival:" in command_source
    assert "base_cmd[:] = 0.0" in command_source


def test_base_arrival_requires_continuous_zero_command_window():
    state = (False, 0)
    for _ in range(24):
        confirmed, steps, zero_command = update_base_target_arrival(
            *state, True, True, required_settle_frames=25
        )
        assert not confirmed
        assert zero_command
        state = (confirmed, steps)
    confirmed, steps, zero_command = update_base_target_arrival(
        *state, True, True, required_settle_frames=25
    )
    assert (confirmed, steps, zero_command) == (True, 25, True)


def test_base_arrival_zero_command_starts_before_robot_has_stopped():
    assert update_base_target_arrival(
        False, 0, True, True, required_settle_frames=25
    ) == (False, 1, True)


def test_base_arrival_brakes_across_inner_to_outer_region():
    state = update_base_target_arrival(
        False, 7, False, True, required_settle_frames=25
    )
    assert state == (False, 8, True)


def test_base_arrival_holds_outer_bound_then_requires_fresh_reacquisition():
    assert update_base_target_arrival(
        True, 25, False, True, required_settle_frames=25
    ) == (
        True,
        25,
        True,
    )
    released = update_base_target_arrival(
        True, 25, False, False, required_settle_frames=25
    )
    assert released == (False, 0, False)
    assert update_base_target_arrival(
        *released[:2], False, True, required_settle_frames=25
    ) == (
        False,
        0,
        False,
    )


def test_base_arrival_tensor_and_scalar_semantics_match():
    scalar = update_base_target_arrival(
        False, 24, False, True, required_settle_frames=25
    )
    tensor = update_base_target_arrival(
        torch.tensor([False]),
        torch.tensor([24], dtype=torch.int32),
        torch.tensor([False]),
        torch.tensor([True]),
        required_settle_frames=torch.tensor([25], dtype=torch.int32),
    )
    assert tuple(value.item() for value in tensor) == scalar


def test_point_target_stages_own_independent_arrival_profiles():
    profiles = (APPROACH_ARRIVAL, ALIGN_PANEL_ARRIVAL)
    assert len({id(profile) for profile in profiles}) == 2
    assert APPROACH_ARRIVAL.inner_distance_m == ALIGN_PANEL_ARRIVAL.inner_distance_m
    assert APPROACH_ARRIVAL.outer_distance_m == ALIGN_PANEL_ARRIVAL.outer_distance_m
    assert APPROACH_ARRIVAL.inner_yaw_rad == ALIGN_PANEL_ARRIVAL.inner_yaw_rad
    assert APPROACH_ARRIVAL.outer_yaw_rad == ALIGN_PANEL_ARRIVAL.outer_yaw_rad
    assert APPROACH_ARRIVAL.settle_frames == ALIGN_PANEL_ARRIVAL.settle_frames


def test_semantic_target_stages_stop_on_their_physical_condition():
    base_advance = inspect.getsource(TensorDoorFSM.command_base_advance)
    through = inspect.getsource(TensorDoorFSM.command_through)
    assert "command[reachable] = 0.0" in base_advance
    assert "_update_base_arrival(" not in base_advance
    assert "tail_clearance >= THROUGH_TAIL_TARGET_DISTANCE" in through
    assert "tail_clearance >= THROUGH_TAIL_RELEASE_DISTANCE" in through
    assert "_update_base_arrival(" not in through


def test_point_arrival_confirmation_implies_outer_acceptance_geometry():
    for profile in (APPROACH_ARRIVAL, ALIGN_PANEL_ARRIVAL):
        inner, outer = base_pose_arrival_regions(
            profile.inner_distance_m * 0.9,
            profile.inner_yaw_rad * 0.9,
            profile,
        )
        assert inner and outer


def test_through_tail_hysteresis_has_five_centimeter_release_band():
    assert np.isclose(
        THROUGH_TAIL_TARGET_DISTANCE - THROUGH_TAIL_RELEASE_DISTANCE,
        0.05,
    )


def test_door_prim_path_uses_scene_resolver_instead_of_guessing_name():
    door = type("Door", (), {"prim_path_for_env": lambda self, env_id: f"/env/{env_id}/door_left"})()
    env = type("Env", (), {"door": door})()
    assert _resolved_door_prim_path(env, 3) == "/env/3/door_left"
