import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.auto_open.motion_targets import PaddedPathBank
from gr00t.rl.auto_open.execution import TensorCommandBuffers
from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM
from gr00t.rl.auto_open.motion_math import (
    _axis_angle_quat,
    _quat_distance,
    _quat_multiply,
    _signed_quat_rotation_about_axis,
    _slew_quaternion,
    track_base_pose,
)
from gr00t.rl.auto_open.acceptance import update_robot_stability


def test_opposed_grasp_allows_contacts_near_handle_end():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.handle_body_idx = 0
    fsm.handle_center_local = torch.zeros(4, 3)
    fsm.handle_axis_local = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
    fsm.door_normal = torch.tensor([[0.0, 1.0, 0.0]]).repeat(4, 1)
    fsm.handle_radius = torch.full((4,), 0.01)
    fsm.handle_half_length = torch.full((4,), 0.05)
    poses = torch.zeros(4, 1, 7)
    poses[:, :, 3] = 1.0
    points = torch.tensor(
        [
            [[0.049, 0.0, 0.005], [0.049, 0.0, -0.005]],
            [[0.049, 0.0, 0.005], [0.049, 0.0, 0.006]],
            [[0.049, 0.0, 0.001], [0.049, 0.0, -0.001]],
            [[float("nan"), 0.0, 0.005], [0.049, 0.0, -0.005]],
        ]
    )
    actual = fsm._opposed_grasp({"door_body_state_w": poses}, torch.arange(4), points)
    assert actual.tolist() == [True, False, False, False]


def test_shared_station_tracking_preserves_approach_commands():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.base_command_previous = torch.zeros((2, 3))
    fsm._update_base_arrival = lambda state, rows, distance, yaw: torch.tensor([False, True])
    rows = torch.arange(2)
    roots = torch.zeros((2, 13))
    roots[:, 3] = 1.0
    target_xy = torch.tensor([[0.1, 0.3], [0.0, 0.0]])
    target_yaw = torch.tensor([0.01, 0.0])
    previous = torch.zeros((2, 3))
    for velocity in (0.0, 0.2, -0.2, 0.3, -0.1):
        roots[:, 8] = velocity
        roots[:, 12] = velocity
        expected, distance, yaw = track_base_pose(
            roots,
            target_xy,
            target_yaw,
            target_xy,
            target_yaw,
            minimum_speed_toward_target_only=True,
        )
        expected[1] = 0.0
        crossing = (previous * expected < 0.0) & ((expected - previous).abs() >= 0.15)
        expected = torch.where(crossing, torch.zeros_like(expected), expected)
        actual, actual_distance, actual_yaw = fsm._track_station(
            {"root_states": roots},
            rows,
            target_xy,
            target_yaw,
            target_xy,
            target_yaw,
        )
        assert torch.equal(actual, expected)
        assert torch.equal(actual_distance, distance)
        assert torch.equal(actual_yaw, yaw)
        previous = expected.clone()


def test_station_stability_gate_does_not_block_motion_or_shortcut_window():
    from gr00t.rl.auto_open.control import APPROACH_ARRIVAL

    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device = torch.device("cpu")
    fsm.base_command_previous = torch.zeros(1, 3)
    fsm.base_arrival_confirmed = torch.zeros(1, dtype=torch.bool)
    fsm.base_stability_confirmed = torch.zeros(1, dtype=torch.bool)
    for name in (
        "base_arrival_settle_steps",
        "base_arrival_max_settle_steps",
        "base_stability_steps",
    ):
        setattr(fsm, name, torch.zeros(1, dtype=torch.int32))
    fsm.through_min_base_height = torch.tensor([0.4])
    rows = torch.tensor([0])
    # configure_base_arrival writes these per-environment tensors in place.
    for name in (
        "inner_distance",
        "outer_distance",
        "inner_yaw",
        "outer_yaw",
        "max_linear_speed",
        "max_angular_speed",
    ):
        setattr(fsm, "base_arrival_" + name, torch.zeros(1))
    fsm.base_arrival_required_frames = torch.zeros(1, dtype=torch.int32)
    fsm.reset_base_arrival(rows, APPROACH_ARRIVAL)
    root = torch.zeros(1, 13)
    root[:, 2], root[:, 3] = 0.5, 1.0
    root[:, 0] = 0.5
    target, yaw = torch.zeros(1, 2), torch.zeros(1)

    def command(slow):
        return fsm._track_station(
            {"root_states": root},
            rows,
            target,
            yaw,
            target,
            yaw,
            stability_condition=torch.tensor([slow]),
        )[0]

    assert command(False).abs().max() >= 0.15
    root[:, 0] = 0.0
    for _ in range(30):
        assert torch.count_nonzero(command(False)) == 0
    assert not fsm.base_stability_confirmed.item()
    for _ in range(24):
        command(True)
    assert not fsm.base_stability_confirmed.item()
    command(True)
    assert fsm.base_stability_confirmed.item()
    command(False)
    assert not fsm.base_stability_confirmed.item()


def test_acceptance_diagnostics_preserves_unobserved_rows_without_nan_conversion():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device = torch.device("cpu")
    fsm.num_envs = 2
    fsm.stage_steps = torch.ones(2, dtype=torch.int32)
    fsm.stage_visit = torch.ones((2, 3), dtype=torch.int32)
    fsm._acceptance = {}
    fsm.record_acceptance(
        1,
        torch.tensor([0]),
        {"position": (torch.tensor([True]), torch.tensor([0.0]), "<", 0.01)},
        required_streak=10,
    )
    observed = fsm.acceptance_diagnostics(0, 1)
    unobserved = fsm.acceptance_diagnostics(1, 1)
    assert observed["all_conditions"]["observed"]
    assert not observed["all_conditions"]["window_satisfied"]
    assert unobserved["all_conditions"]["observed"] is False
    assert unobserved["all_conditions"]["threshold"] is None
    assert unobserved["position"]["value"] is None
    # Earlier visits must never be mistaken for observations of this visit.
    fsm.stage_visit[0, 1] += 1
    assert fsm.acceptance_diagnostics(0, 1)["position"]["observed"] is False


@pytest.mark.parametrize("blocking", [False, True])
def test_open_gripper_requires_no_blocking_jaw_contact(blocking):
    from enum import IntEnum

    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device = torch.device("cpu")
    fsm.Stage = IntEnum("Stage", {"RELEASE_HANDLE": 1})
    fsm.mask = lambda stage: torch.tensor([True])
    fsm.q_open = torch.tensor([-1.0])
    fsm.release_stable_steps = torch.zeros(1, dtype=torch.int32)
    fsm.stage_steps = torch.ones(1, dtype=torch.int32)
    fsm._release_blocking_handle_contacts = lambda *_: torch.tensor([[blocking, False]])
    fsm.release_open_path = PaddedPathBank(1, 2, "cpu")
    fsm.release_open_path.length[:] = 2
    fsm.release_open_path.index[:] = 1
    fsm.release_open_target_world = torch.zeros(1, 3)
    recorded = []
    fsm.record_acceptance = lambda stage, rows, conditions, **kwargs: recorded.append(conditions)
    state = {"gripper_position": torch.tensor([[-1.0]]), "ee_pos": torch.zeros(1, 3)}
    for _ in range(9):
        assert fsm.advance_release_handle(state).numel() == 0
    assert fsm.advance_release_handle(state).tolist() == ([] if blocking else [0])
    assert all(
        set(conditions)
        == {"gripper_open_error_rad", "fixed_jaw_contact_clear", "moving_jaw_contact_clear",
            "path_complete", "release_height_error_m"}
        for conditions in recorded
    )
    state["gripper_position"][:] = -0.5
    assert fsm.advance_release_handle(state).numel() == 0
    assert fsm.release_stable_steps.item() == 0


def _numpy_reference(
    roots,
    command_xy,
    command_yaw,
    acceptance_xy,
    acceptance_yaw,
):
    roots = np.asarray(roots, dtype=np.float32)
    error = np.asarray(command_xy, dtype=np.float32) - roots[:, :2]
    w, x, y, z = (roots[:, index] for index in range(3, 7))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local_x = cosine * error[:, 0] + sine * error[:, 1]
    local_y = -sine * error[:, 0] + cosine * error[:, 1]
    local_vx = cosine * roots[:, 7] + sine * roots[:, 8]
    local_vy = -sine * roots[:, 7] + cosine * roots[:, 8]
    yaw_error = (command_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
    command = np.stack(
        (
            1.2 * local_x - local_vx,
            1.2 * local_y - local_vy,
            1.5 * yaw_error - 0.35 * roots[:, 12],
        ),
        axis=-1,
    ).astype(np.float32)
    planar_norm = np.linalg.norm(command[:, :2], axis=-1, keepdims=True)
    command[:, :2] *= np.minimum(1.0, 0.4 / np.maximum(planar_norm, 1.0e-8))
    command[:, 2] = np.clip(command[:, 2], -0.6, 0.6)
    planar_norm = np.linalg.norm(command[:, :2], axis=-1)
    deadzone = (
        (planar_norm > 1.0e-6)
        & (planar_norm < 0.15)
        & ((command[:, 0] * local_x + command[:, 1] * local_y) > 0.0)
    )
    command[deadzone, :2] *= (0.15 / planar_norm[deadzone])[:, None]
    yaw_deadzone = (
        (np.abs(command[:, 2]) > 1.0e-6)
        & (np.abs(command[:, 2]) < 0.2)
        & ((command[:, 2] * yaw_error) > 0.0)
    )
    command[yaw_deadzone, 2] = np.copysign(0.2, command[yaw_deadzone, 2])
    distance = np.linalg.norm(acceptance_xy - roots[:, :2], axis=-1)
    acceptance_error = np.abs((acceptance_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi)
    return command, distance, acceptance_error


def test_gpu_base_tracking_matches_reference_batch():
    generator = torch.Generator().manual_seed(731)
    roots = torch.randn((257, 13), generator=generator, dtype=torch.float32)
    yaw = torch.empty(257).uniform_(-math.pi, math.pi, generator=generator)
    roots[:, 3:7] = 0.0
    roots[:, 3] = torch.cos(yaw / 2.0)
    roots[:, 6] = torch.sin(yaw / 2.0)
    command_xy = torch.randn((257, 2), generator=generator)
    command_yaw = torch.empty(257).uniform_(-math.pi, math.pi, generator=generator)
    acceptance_xy = torch.randn((257, 2), generator=generator)
    acceptance_yaw = torch.empty(257).uniform_(-math.pi, math.pi, generator=generator)

    actual = track_base_pose(
        roots,
        command_xy,
        command_yaw,
        acceptance_xy,
        acceptance_yaw,
        minimum_speed_toward_target_only=True,
    )
    expected = _numpy_reference(
        roots.numpy(),
        command_xy.numpy(),
        command_yaw.numpy(),
        acceptance_xy.numpy(),
        acceptance_yaw.numpy(),
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_value.numpy(), expected_value, rtol=2.0e-5, atol=2.0e-6)


def test_robot_stability_requires_every_configured_condition_consecutively():
    base = torch.zeros((2, 13), dtype=torch.float32)
    base[:, 2] = 0.5
    base[:, 3] = 1.0
    base[1, 7] = 0.06

    confirmed, next_steps, stable = update_robot_stability(
        base,
        torch.tensor([2, 2], dtype=torch.int32),
        required_frames=3,
        max_linear_speed_m_s=0.05,
        max_angular_speed_rad_s=0.05,
        position_error_m=torch.tensor([0.04, 0.04]),
        max_position_error_m=0.05,
        yaw_error_rad=torch.tensor([0.02, 0.02]),
        max_yaw_error_rad=0.05,
        min_base_height_m=0.4,
        min_upright_cos=math.cos(math.radians(10.0)),
        door_speed_rad_s=torch.tensor([0.01, 0.01]),
        max_door_speed_rad_s=0.02,
        additional_condition=torch.tensor([True, True]),
    )

    assert stable.tolist() == [True, False]
    assert next_steps.tolist() == [3, 0]
    assert confirmed.tolist() == [True, False]


def test_turn_gap_path_is_only_a_rigid_rotation_about_the_door_hinge():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device = torch.device("cpu")
    fsm.stage_steps = torch.zeros(1, dtype=torch.int32)
    fsm.hinge_idx = 0
    fsm.turn_start_hinge_angle = torch.zeros(1)
    fsm.turn_target_hinge_angle = torch.tensor([math.pi / 2.0])
    fsm.hinge_axis_sign = torch.ones(1)
    fsm.hinge_world = torch.zeros((1, 3))
    fsm.turn_handle_axis_world = torch.tensor([[0.0, 1.0, 0.0]])
    fsm.turn_grasp_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    fsm.turn_gap_gripper_angle = torch.zeros(1)
    fsm.turn_path = PaddedPathBank(1, 512, "cpu")
    start = torch.tensor([[1.0, 0.0, 0.25]])
    start_q = _axis_angle_quat(torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([0.61]))
    state = {
        "ee_pos": torch.tensor([[9.0, 9.0, 9.0]]),
        "ee_orn": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "ee_goal_cart_world": start,
        "ee_goal_orn_quat_world": start_q,
        "door_joint_position": torch.tensor([[0.0, 0.83, 0.0]]),
    }

    fsm._build_turn_gap_path(state, torch.tensor([0]))

    count = int(fsm.turn_path.length[0])
    positions = fsm.turn_path.positions[0, :count]
    orientations = fsm.turn_path.orientations[0, :count]
    torch.testing.assert_close(positions[0], start[0])
    torch.testing.assert_close(positions[-1], torch.tensor([0.0, 1.0, 0.25]))
    torch.testing.assert_close(
        torch.linalg.vector_norm(positions[:, :2], dim=-1), torch.ones(count)
    )
    torch.testing.assert_close(orientations[0], start_q[0])
    assert fsm.turn_gap_gripper_angle.item() == pytest.approx(0.61, abs=1e-6)
    expected_end_q = _axis_angle_quat(
        torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([math.pi / 2.0])
    )
    expected_end_q = _quat_multiply(expected_end_q, start_q)[0]
    assert _quat_distance(orientations[-1:], expected_end_q[None]).item() < 1e-6


def test_return_rotation_uses_gripper_orientation_not_handle_angle():
    axis = torch.tensor([[0.0, 1.0, 0.0]])
    horizontal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    gripper_pressed = _axis_angle_quat(axis, torch.tensor([math.radians(35.0)]))

    return_angle = _signed_quat_rotation_about_axis(gripper_pressed, horizontal, axis)

    assert return_angle.item() == pytest.approx(math.radians(-35.0), abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_quaternion_and_base_control_are_exactly_independent_of_batch_shape():
    generator = torch.Generator(device="cuda").manual_seed(941)
    current = torch.randn((1, 4), generator=generator, device="cuda")
    target = torch.randn((1, 4), generator=generator, device="cuda")
    extra_current = torch.randn((7, 4), generator=generator, device="cuda")
    extra_target = torch.randn((7, 4), generator=generator, device="cuda")
    batch_current = torch.cat((current, extra_current), dim=0)
    batch_target = torch.cat((target, extra_target), dim=0)

    assert torch.equal(
        _slew_quaternion(current, target)[0],
        _slew_quaternion(batch_current, batch_target)[0],
    )
    assert torch.equal(
        _quat_distance(current, target)[0],
        _quat_distance(batch_current, batch_target)[0],
    )

    root = torch.randn((1, 13), generator=generator, device="cuda")
    root[:, 3:7] = current
    command_xy = torch.randn((1, 2), generator=generator, device="cuda")
    command_yaw = torch.randn((1,), generator=generator, device="cuda")
    acceptance_xy = torch.randn((1, 2), generator=generator, device="cuda")
    acceptance_yaw = torch.randn((1,), generator=generator, device="cuda")
    repeats = 8
    single = track_base_pose(
        root,
        command_xy,
        command_yaw,
        acceptance_xy,
        acceptance_yaw,
        minimum_speed_toward_target_only=False,
    )
    batched = track_base_pose(
        root.repeat(repeats, 1),
        command_xy.repeat(repeats, 1),
        command_yaw.repeat(repeats),
        acceptance_xy.repeat(repeats, 1),
        acceptance_yaw.repeat(repeats),
        minimum_speed_toward_target_only=False,
    )
    for single_value, batched_value in zip(single, batched, strict=True):
        assert torch.equal(single_value[0], batched_value[0])


def test_padded_path_bank_advances_rows_independently():
    bank = PaddedPathBank(3, 8, "cpu")
    positions = torch.ones(2, 3, 3)
    positions[0] = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    orientations = torch.zeros(2, 3, 4)
    orientations[0, :, 0] = 1.0
    orientations[1, :, 1] = 1.0
    bank.install_tensor(torch.tensor([0, 2]), positions, orientations, torch.tensor([3, 2]))
    rows = torch.tensor([0, 2])
    first, _ = bank.current(rows, advance=True)
    second, _ = bank.current(rows, advance=True)
    np.testing.assert_array_equal(first.numpy(), [[0, 1, 2], [1, 1, 1]])
    np.testing.assert_array_equal(second.numpy(), [[3, 4, 5], [1, 1, 1]])
    # Match the legacy controller: completion observes the post-command index.
    assert bank.complete(rows).tolist() == [True, True]


def test_command_buffers_use_arm_mode_transition_api():
    env = SimpleNamespace(
        teleop_raw_commands=torch.zeros((3, 3)),
        teleop_raw_ee_goal_cart=torch.zeros((3, 3)),
        teleop_raw_ee_goal_orn_delta_rpy=torch.zeros((3, 3)),
        gripper_pos_targets=torch.zeros((3, 1)),
        teleop_world_ee_goal_orn_quat=torch.zeros((3, 4)),
        teleop_world_ee_goal_orn_quat_enabled=torch.zeros(3, dtype=torch.bool),
        teleop_arm_joint_pos_targets=torch.zeros((3, 6)),
        teleop_arm_joint_control_mask=torch.tensor([True, False, True]),
        _teleop_inputs_dirty=False,
    )
    transitions = []

    def set_mode(mode, rows):
        rows = torch.as_tensor(rows, dtype=torch.int64)
        transitions.append((mode, rows.tolist()))
        env.teleop_arm_joint_control_mask[rows] = mode == "joint"

    env._set_teleop_arm_control_mode = set_mode
    buffers = TensorCommandBuffers.from_env(env)
    buffers.arm_joint_mode[:] = torch.tensor([False, True, True])
    buffers.arm_joint[:] = 0.25

    buffers.flush(env)

    assert transitions == [("joint", [1]), ("ee", [0])]
    assert env.teleop_arm_joint_control_mask.tolist() == [False, True, True]
    torch.testing.assert_close(env.teleop_arm_joint_pos_targets, torch.full((3, 6), 0.25))


def test_push_arc_latches_physical_contact_not_inverse_command():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device = torch.device("cpu")
    fsm.Stage = SimpleNamespace(PUSH_PANEL_OPEN=13)
    fsm.stage = torch.tensor([12], dtype=torch.int64)
    fsm.stage_steps = torch.zeros(1, dtype=torch.int32)
    fsm.hinge_idx = 0
    fsm.panel_contact_target_world = torch.tensor([[1.0, 2.0, 3.0]])
    fsm.panel_push_tip_offset_local = torch.tensor([[0.1, 0.0, 0.0]])
    fsm.panel_push_tip_surface_world = torch.tensor([[1.1, 2.0, 3.0]])
    fsm.panel_push_tip_hinge_angle = torch.tensor([0.25])
    fsm.hinge_world = torch.zeros(1, 3)
    fsm.hinge_axis_sign = torch.ones(1)
    fsm.push_tip_offset_world = torch.zeros(1, 3)
    fsm.panel_contact_command_target_world = torch.tensor([[4.0, 5.0, 6.0]])
    fsm.push_start_hinge_angle = torch.zeros(1)
    fsm.push_last_progress_hinge = torch.zeros(1)
    fsm.push_no_progress_steps = torch.zeros(1, dtype=torch.int32)
    fsm.push_contact_world = torch.zeros((1, 3))
    fsm.push_orientation = torch.zeros((1, 4))
    fsm.push_command_hinge_angle = torch.zeros(1)
    fsm.push_speed_error_integral = torch.zeros(1)
    fsm.push_overspeed_steps = torch.zeros(1, dtype=torch.int32)
    fsm.push_max_observed_angular_speed = torch.zeros(1)
    fsm.push_arm_detached = torch.zeros(1, dtype=torch.bool)
    fsm.push_recovery_settle_steps = torch.zeros(1, dtype=torch.int32)
    fsm.push_reengage_reachable = torch.zeros(1, dtype=torch.bool)
    fsm.push_previous_finger_clearance = torch.zeros(1)
    fsm.push_base_start_xy = torch.zeros((1, 2))
    fsm.push_entry_ee_world = torch.zeros((1, 3))
    fsm.push_entry_command_world = torch.zeros((1, 3))
    fsm.ee_command_world = torch.tensor([[7.0, 8.0, 9.0]])
    state = {
        "door_joint_position": torch.tensor([[0.25]]),
        "door_joint_velocity": torch.tensor([[0.1]]),
        "ee_orn": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "root_states": torch.zeros((1, 13)),
        "ee_pos": torch.tensor([[0.5, 0.6, 0.7]]),
    }

    fsm.begin_push_panel_open(state, torch.tensor([0]), torch.tensor([0.03]))

    torch.testing.assert_close(fsm.push_contact_world, fsm.panel_contact_target_world)
    assert not torch.equal(fsm.push_contact_world, fsm.panel_contact_command_target_world)


def test_base_advance_reentry_preserves_contact_seed_and_actual_ee_origin():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage = SimpleNamespace(ADVANCE_TO_NEXT_PUSH_STATION=15, MOVE_TO_PANEL_STANDOFF=12)
    fsm.stage = torch.tensor([15], dtype=torch.int64)
    fsm.stage_steps = torch.tensor([23], dtype=torch.int32)
    fsm.push_recovery_settle_steps = torch.tensor([10], dtype=torch.int32)
    fsm.base_arrival_confirmed = torch.ones(1, dtype=torch.bool)
    fsm.base_arrival_settle_steps = torch.full((1,), 25, dtype=torch.int32)
    fsm.base_arrival_max_settle_steps = torch.full((1,), 25, dtype=torch.int32)
    fsm.base_arrival_inner_distance = torch.zeros(1)
    fsm.base_arrival_outer_distance = torch.zeros(1)
    fsm.base_arrival_inner_yaw = torch.zeros(1)
    fsm.base_arrival_outer_yaw = torch.zeros(1)
    fsm.base_arrival_required_frames = torch.zeros(1, dtype=torch.int32)
    fsm.hinge_idx = 0
    fsm.hinge_axis_sign = torch.ones(1)
    fsm.push_start_hinge_angle = torch.zeros(1)
    fsm.push_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    fsm.panel_contact_orientation_seed = torch.zeros((1, 4))
    fsm.panel_contact_orientation_seed_valid = torch.zeros(1, dtype=torch.bool)
    fsm.ee_command_world = torch.tensor([[8.0, 8.0, 8.0]])
    fsm.panel_contact_plan_valid = torch.ones(1, dtype=torch.bool)
    fsm.panel_contact_settle_steps = torch.ones(1, dtype=torch.int32)
    state = {
        "door_joint_position": torch.tensor([[math.pi / 2.0]]),
        "ee_pos": torch.tensor([[0.4, -0.2, 1.1]]),
    }

    completed = fsm.advance_advance_to_next_push_station(state)

    assert completed.tolist() == [0]
    assert fsm.stage.tolist() == [12]
    assert fsm.panel_contact_orientation_seed_valid.tolist() == [True]
    torch.testing.assert_close(fsm.ee_command_world, state["ee_pos"])
    torch.testing.assert_close(
        fsm.panel_contact_orientation_seed,
        torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]]),
    )


def test_push_retract_slews_orientation_instead_of_jumping_to_endpoint():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage = SimpleNamespace(STOW_ARM_AFTER_PUSH=14)
    fsm.device = torch.device("cpu")
    fsm.stage = torch.tensor([14], dtype=torch.int64)
    fsm.stage_steps = torch.tensor([1], dtype=torch.int32)
    fsm.push_detached_ee_goal_local = torch.tensor([[0.3, 0.0, 0.3]])
    fsm.active_arm_retract_pose_local = torch.tensor([[0.3, 0.0, 0.3, 0.0, 0.0, math.pi / 2.0]])
    fsm.ee_orientation_command = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    fsm.arm_induced_pitch = torch.zeros(1)
    fsm.q_closed = torch.zeros(1)
    fsm._base_hold_command = lambda state, rows, buffers=None: (
        torch.zeros((len(rows), 3)),
        torch.zeros(len(rows)),
        torch.zeros(len(rows)),
    )
    fsm.env = SimpleNamespace(get_goal_reference_quat=lambda: torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    buffers = SimpleNamespace(
        base=torch.zeros((1, 3)),
        ee_cart=torch.zeros((1, 3)),
        ee_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ee_quat_enabled=torch.zeros(1, dtype=torch.bool),
        ee_rpy=torch.zeros((1, 3)),
        arm_joint_mode=torch.ones(1, dtype=torch.bool),
        gripper=torch.zeros((1, 1)),
    )

    fsm.command_stow_arm_after_push({}, buffers)

    start = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    endpoint = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
    torch.testing.assert_close(_quat_distance(start, buffers.ee_quat), torch.tensor([0.025]))
    assert _quat_distance(buffers.ee_quat, endpoint).item() > 1.5


def test_terminal_hold_dispatches_exact_zero_after_reversal_filter():
    from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory

    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.pull_door = False
    fsm.through_trajectory = BasePoseTrajectory(1, "cpu")
    fsm.through_trajectory.length[:] = 1
    fsm.device = torch.device("cpu")
    fsm.Stage = SimpleNamespace(PASS_THROUGH_DOORWAY=16, SUCCESS=17)
    fsm.stage = torch.tensor([16], dtype=torch.int64)
    fsm.through_command_xy = torch.zeros((1, 2))
    fsm.through_command_yaw = torch.zeros(1)
    fsm.through_station_xy = torch.zeros((1, 2))
    fsm.through_station_yaw = torch.zeros(1)
    fsm.through_stop_latched = torch.zeros(1, dtype=torch.bool)
    fsm.base_arrival_confirmed = torch.ones(1, dtype=torch.bool)
    fsm.base_arrival_settle_steps = torch.full((1,), 25, dtype=torch.int32)
    fsm.base_arrival_max_settle_steps = torch.zeros(1, dtype=torch.int32)
    fsm.base_arrival_required_frames = torch.full((1,), 25, dtype=torch.int32)
    fsm.base_arrival_inner_distance = torch.full((1,), 0.08)
    fsm.base_arrival_outer_distance = torch.full((1,), 0.30)
    fsm.base_arrival_inner_yaw = torch.full((1,), 0.08)
    fsm.base_arrival_outer_yaw = torch.full((1,), 0.20)
    fsm.initial_door_root = torch.tensor([[-1.284, 0.0, 0.0]])
    fsm.door_forward = torch.tensor([[1.0, 0.0, 0.0]])
    fsm.through_motion_ready = torch.ones(1, dtype=torch.bool)
    fsm.base_command_previous = torch.tensor([[0.2, -0.2, 0.2]])
    fsm.base_reversal_pending_sign = torch.zeros((1, 3), dtype=torch.int8)
    fsm.base_reversal_pending_steps = torch.zeros((1, 3), dtype=torch.int32)
    fsm.base_reversal_lock_steps = torch.full((1, 3), 5, dtype=torch.int32)
    fsm.pending_station_distance = torch.zeros(1)
    fsm.pending_yaw_error = torch.zeros(1)
    fsm.middle_arm_joint_target = torch.zeros((1, 6))
    fsm.q_closed = torch.zeros(1)
    state = {"root_states": torch.tensor([[0.0, 0.0, 0.5, 1.0] + [0.0] * 9])}
    buffers = SimpleNamespace(
        base=torch.ones((1, 3)),
        arm_joint_mode=torch.zeros(1, dtype=torch.bool),
        arm_joint=torch.ones((1, 6)),
        gripper=torch.ones((1, 1)),
    )

    fsm.command_pass_through_doorway(state, buffers)

    torch.testing.assert_close(buffers.base, torch.zeros((1, 3)))
    assert fsm.through_stop_latched.tolist() == [True]


def test_control_joint_speed_uses_control_frame_displacement_not_physx_spike():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.dt = 0.02
    fsm.previous_dof_position = torch.zeros((1, 2))
    fsm.control_joint_velocity = torch.zeros((1, 2))
    state = {
        "dof_pos": torch.tensor([[0.001, -0.0004]]),
        "dof_vel": torch.tensor([[12.0, -8.0]]),
    }

    fsm.update_control_joint_velocity(state)
    speed = fsm.control_joint_speed(state, torch.tensor([0]), 0)

    torch.testing.assert_close(speed, torch.tensor([0.05]))
