import math
from types import SimpleNamespace

import numpy as np
import torch

from gr00t.rl.auto_open.tensor_fsm import (
    PaddedPathBank,
    TensorDoorFSM,
    _quat_distance,
    track_base_pose,
)


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
    yaw = np.arctan2(
        2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
    )
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
    command_distance = np.linalg.norm(error, axis=-1)
    planar_norm = np.linalg.norm(command[:, :2], axis=-1)
    deadzone = (
        (command_distance > 0.08)
        & (planar_norm > 1.0e-6)
        & (planar_norm < 0.15)
        & ((command[:, 0] * local_x + command[:, 1] * local_y) > 0.0)
    )
    command[deadzone, :2] *= (0.15 / planar_norm[deadzone])[:, None]
    yaw_deadzone = (
        (np.abs(yaw_error) > 0.08)
        & (np.abs(command[:, 2]) > 1.0e-6)
        & (np.abs(command[:, 2]) < 0.2)
        & ((command[:, 2] * yaw_error) > 0.0)
    )
    command[yaw_deadzone, 2] = np.copysign(0.2, command[yaw_deadzone, 2])
    distance = np.linalg.norm(acceptance_xy - roots[:, :2], axis=-1)
    acceptance_error = np.abs(
        (acceptance_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
    )
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
        np.testing.assert_allclose(
            actual_value.numpy(), expected_value, rtol=2.0e-5, atol=2.0e-6
        )


def test_padded_path_bank_advances_rows_independently():
    bank = PaddedPathBank(3, 8, "cpu")
    bank.install(
        [0, 2],
        [np.arange(9, dtype=np.float32).reshape(3, 3), np.ones((2, 3))],
        [
            np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
            np.tile([0.0, 1.0, 0.0, 0.0], (2, 1)),
        ],
    )
    rows = torch.tensor([0, 2])
    first, _ = bank.current(rows, advance=True)
    second, _ = bank.current(rows, advance=True)
    np.testing.assert_array_equal(first.numpy(), [[0, 1, 2], [1, 1, 1]])
    np.testing.assert_array_equal(second.numpy(), [[3, 4, 5], [1, 1, 1]])
    # Match the legacy controller: completion observes the post-command index.
    assert bank.complete(rows).tolist() == [True, True]


def test_push_arc_latches_physical_contact_not_inverse_command():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage = SimpleNamespace(PUSH_OPEN=13)
    fsm.stage = torch.tensor([12], dtype=torch.int64)
    fsm.stage_steps = torch.zeros(1, dtype=torch.int32)
    fsm.hinge_idx = 0
    fsm.panel_contact_target_world = torch.tensor([[1.0, 2.0, 3.0]])
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

    fsm.begin_push(state, torch.tensor([0]), torch.tensor([0.03]))

    torch.testing.assert_close(
        fsm.push_contact_world, fsm.panel_contact_target_world
    )
    assert not torch.equal(
        fsm.push_contact_world, fsm.panel_contact_command_target_world
    )


def test_base_advance_reentry_preserves_contact_seed_and_actual_ee_origin():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage = SimpleNamespace(BASE_ADVANCE=15, CONTACT_PANEL=12)
    fsm.stage = torch.tensor([15], dtype=torch.int64)
    fsm.stage_steps = torch.tensor([23], dtype=torch.int32)
    fsm.push_recovery_settle_steps = torch.tensor([10], dtype=torch.int32)
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

    completed = fsm.advance_base_advance(state)

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
    fsm.Stage = SimpleNamespace(PUSH_ARM_RETRACT=14)
    fsm.device = torch.device("cpu")
    fsm.stage = torch.tensor([14], dtype=torch.int64)
    fsm.stage_steps = torch.tensor([1], dtype=torch.int32)
    fsm.push_detached_ee_goal_local = torch.tensor([[0.3, 0.0, 0.3]])
    fsm.active_arm_retract_pose_local = torch.tensor(
        [[0.3, 0.0, 0.3, 0.0, 0.0, math.pi / 2.0]]
    )
    fsm.ee_orientation_command = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    fsm.arm_induced_pitch = torch.zeros(1)
    fsm.q_closed = torch.zeros(1)
    fsm._base_hold_command = lambda state, rows: (
        torch.zeros((len(rows), 3)), torch.zeros(len(rows)), torch.zeros(len(rows))
    )
    fsm.env = SimpleNamespace(
        get_goal_reference_quat=lambda: torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    buffers = SimpleNamespace(
        base=torch.zeros((1, 3)),
        ee_cart=torch.zeros((1, 3)),
        ee_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ee_quat_enabled=torch.zeros(1, dtype=torch.bool),
        ee_rpy=torch.zeros((1, 3)),
        arm_joint_mode=torch.ones(1, dtype=torch.bool),
        gripper=torch.zeros((1, 1)),
    )

    fsm.command_push_retract({}, buffers)

    start = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    endpoint = torch.tensor(
        [[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]]
    )
    torch.testing.assert_close(_quat_distance(start, buffers.ee_quat), torch.tensor([0.025]))
    assert _quat_distance(buffers.ee_quat, endpoint).item() > 1.5
