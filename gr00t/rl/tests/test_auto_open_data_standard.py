import math

import numpy as np
import pytest

from gr00t.rl.data_collection.auto_open_standard import (
    ACTION_NAMES,
    OBSERVATION_STATE_NAMES,
    derive_stage_boundaries,
    height_invariant_pose,
    pack_action,
    pack_observation_state,
    transform_dataset_images,
)
from gr00t.rl.data_collection.auto_open_writer import (
    AutoOpenEpisodeData,
    _physical_acceptance_reasons,
)


def _yaw_quaternion(yaw):
    return np.asarray((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))


def _yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def test_fixed_schema_dimensions_and_action_alias():
    assert len(OBSERVATION_STATE_NAMES) == 58
    assert len(set(OBSERVATION_STATE_NAMES)) == 58
    assert len(ACTION_NAMES) == 25
    assert len(set(ACTION_NAMES)) == 25

    control = np.arange(9, dtype=np.float32)
    action = pack_action(
        base_command=np.asarray((0.1, -0.2, 0.3)),
        arm_teleop_inactive=False,
        arm_reset=False,
        control_height_invariant_ee=control,
        gripper_target=-math.pi / 3,
    )
    np.testing.assert_array_equal(action[5:14], action[16:25])
    assert action[15] == 0.0


def test_illegal_arm_mode_is_rejected():
    with pytest.raises(ValueError, match="cannot both be 1"):
        pack_action(
            base_command=np.zeros(3),
            arm_teleop_inactive=True,
            arm_reset=True,
            control_height_invariant_ee=np.zeros(9),
            gripper_target=0.0,
        )


def test_height_invariant_pose_removes_global_xy_and_yaw_only():
    local_position = np.asarray((0.55, -0.08, 0.92))
    local_rotation = _yaw_matrix(0.3)
    reference = height_invariant_pose(
        np.asarray((1.0, 2.0, 0.48)),
        _yaw_quaternion(0.0),
        np.asarray((1.0, 2.0, 0.0)) + local_position,
        local_rotation,
    )
    global_yaw = 1.2
    rotated_position = (
        np.asarray((8.0, -4.0, 0.0))
        + _yaw_matrix(global_yaw) @ local_position
    )
    transformed = height_invariant_pose(
        np.asarray((8.0, -4.0, 0.48)),
        _yaw_quaternion(global_yaw),
        rotated_position,
        _yaw_matrix(global_yaw) @ local_rotation,
    )
    np.testing.assert_allclose(transformed, reference, atol=1.0e-6, rtol=0.0)


def test_observation_packer_is_finite_58_vector():
    state = pack_observation_state(
        arm_position=np.arange(6) * 0.1,
        arm_velocity=np.arange(6) * -0.1,
        gripper_feedback=-0.5,
        b2_position=np.arange(12) * 0.01,
        b2_velocity=np.arange(12) * -0.01,
        root_position_world=np.asarray((1.0, 2.0, 0.5)),
        root_quaternion_wxyz=_yaw_quaternion(0.2),
        root_linear_velocity_world=np.asarray((0.1, 0.2, 0.3)),
        root_angular_velocity_world=np.asarray((-0.1, -0.2, -0.3)),
        actual_tool_position_world=np.asarray((1.4, 2.1, 0.8)),
        actual_tool_rotation_world=np.eye(3),
    )
    assert state.shape == (58,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()


def test_camera_transform_has_exact_keys_shapes_and_rotation():
    base = np.arange(480 * 640 * 3, dtype=np.uint32).reshape(480, 640, 3)
    base = (base % 251).astype(np.uint8)
    wrist = np.zeros((480, 640, 3), dtype=np.uint8)
    transformed = transform_dataset_images(
        {"base_0_rgb": base, "left_wrist_0_rgb": wrist}
    )
    assert transformed["observation.images.base_0_rgb"].shape == (640, 480, 3)
    assert transformed["observation.images.left_wrist_0_rgb"].shape == (480, 640, 3)
    np.testing.assert_array_equal(
        transformed["observation.images.base_0_rgb"], np.rot90(base, 1)
    )


def test_three_stage_boundaries_are_half_open_and_terminal_hold_is_retained():
    frames = 145
    base = np.zeros((frames, 3), dtype=np.float32)
    base[10:40, 0] = 0.2
    base[90:95, 0] = 0.2
    inactive = np.ones(frames, dtype=bool)
    inactive[65:90] = False
    reset = np.zeros(frames, dtype=bool)
    gripper = np.zeros(frames, dtype=np.float32)
    robot_clear = np.zeros(frames, dtype=bool)
    robot_clear[90:] = True

    result = derive_stage_boundaries(
        base_commands=base,
        arm_teleop_inactive=inactive,
        arm_reset=reset,
        gripper_targets=gripper,
        terminal_robot_clear=robot_clear,
    )
    assert result.quarantine_reasons == ()
    assert result.boundaries["approach"].start_frame == 10
    assert result.boundaries["approach"].stop_frame == 65
    assert result.boundaries["handle_press"].start_frame == 65
    assert result.boundaries["handle_press"].stop_frame == 90
    assert result.boundaries["traversal"].start_frame == 90
    assert result.boundaries["traversal"].stop_frame == 145
    assert result.physical_phase[95:] == ("terminal_hold",) * 50


def test_boundary_validator_quarantines_missing_approach_stop():
    frames = 100
    base = np.zeros((frames, 3), dtype=np.float32)
    base[0:30, 0] = 0.2
    inactive = np.ones(frames, dtype=bool)
    inactive[40:60] = False
    result = derive_stage_boundaries(
        base_commands=base,
        arm_teleop_inactive=inactive,
        arm_reset=np.zeros(frames, dtype=bool),
        gripper_targets=np.zeros(frames),
        terminal_robot_clear=np.ones(frames, dtype=bool),
    )
    assert "approach_missing_25_frame_stop" in result.quarantine_reasons


def test_physical_validator_requires_standard_full_open_target():
    frames = 145
    base = np.zeros((frames, 3), dtype=np.float32)
    base[10:40, 0] = 0.2
    base[90:95, 0] = 0.2
    inactive = np.ones(frames, dtype=bool)
    inactive[65:90] = False
    gripper = np.zeros(frames, dtype=np.float32)
    gripper[68:72] = -math.pi / 3
    gripper[72:78] = -0.2
    gripper[78:84] = -math.pi / 3
    robot_clear = np.zeros(frames, dtype=bool)
    robot_clear[90:] = True
    boundary = derive_stage_boundaries(
        base_commands=base,
        arm_teleop_inactive=inactive,
        arm_reset=np.zeros(frames, dtype=bool),
        gripper_targets=gripper,
        terminal_robot_clear=robot_clear,
    )
    action = np.zeros((frames, 25), dtype=np.float32)
    action[:, :3] = base
    action[:, 3] = inactive
    action[:, 14] = gripper
    audit = {
        "sim_time_ns": np.arange(frames, dtype=np.int64) * 20_000_000,
        "base_image_capture_time_ns": np.arange(frames, dtype=np.int64) * 20_000_000,
        "wrist_image_capture_time_ns": np.arange(frames, dtype=np.int64) * 20_000_000,
        "handle_position": np.full(frames, math.radians(25.0)),
        "door_edge_gap_m": np.full(frames, 0.03),
        "arm_joint_position": np.zeros((frames, 6), dtype=np.float32),
        "robot_clear_of_door": robot_clear,
    }
    episode = AutoOpenEpisodeData(
        episode_index=0,
        seed=1700,
        observation_state=np.zeros((frames, 58), dtype=np.float32),
        action=action,
        images={},
        raw_audit=audit,
        event_frames={"handle_unlocked": 70, "door_gap_20mm": 77, "robot_clear": 90},
    )
    assert _physical_acceptance_reasons(episode, boundary) == []

    episode.action[68:72, 14] = -0.785
    assert "invalid_gripper_closed_open_grasp_open_closed_sequence" in (
        _physical_acceptance_reasons(episode, boundary)
    )
