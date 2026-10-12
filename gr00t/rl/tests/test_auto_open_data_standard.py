import math
import json

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
    aggregate_standard_shards,
)
from gr00t.rl.data_collection.auto_open_collector import _door_edge_gap_m


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


def test_terminal_boundary_requires_actual_stability_not_only_zero_command():
    frames = 146
    base = np.zeros((frames, 3), dtype=np.float32)
    base[10:40, 0] = 0.2
    base[90:95, 0] = 0.2
    inactive = np.ones(frames, dtype=bool)
    inactive[65:90] = False
    stable = np.ones(frames, dtype=bool)
    stable[:96] = False
    result = derive_stage_boundaries(
        base_commands=base,
        arm_teleop_inactive=inactive,
        arm_reset=np.zeros(frames, dtype=bool),
        gripper_targets=np.zeros(frames),
        terminal_robot_clear=np.arange(frames) >= 90,
        terminal_actual_stable=stable,
    )
    assert result.quarantine_reasons == ()
    assert result.boundaries["traversal"].stop_frame == frames
    assert result.physical_phase[-50:] == ("terminal_hold",) * 50


def test_door_edge_gap_is_free_edge_distance_from_closed_plane():
    assert _door_edge_gap_m(0.0, 1.0, 0.02) == 0.0
    assert _door_edge_gap_m(0.03, 1.0, 0.02) > 0.02
    assert _door_edge_gap_m(0.20, 1.0, 0.02) > _door_edge_gap_m(0.10, 1.0, 0.02)


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
    gripper[72:78] = 0.0
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
        "arm_joint_velocity": np.zeros((frames, 6), dtype=np.float32),
        "root_linear_velocity_world": np.zeros((frames, 3), dtype=np.float32),
        "root_angular_velocity_world": np.zeros((frames, 3), dtype=np.float32),
        "door_hinge_velocity": np.zeros(frames, dtype=np.float32),
        "gripper_feedback": np.zeros(frames, dtype=np.float32),
        "gripper_target_dispatched": gripper.copy(),
        "gripper_actuator_target": gripper.copy(),
        "arm_joint_target_dispatched": np.zeros((frames, 6), dtype=np.float32),
        "robot_clear_of_door": robot_clear,
    }
    stages = np.full(frames, "THROUGH", dtype=object)
    stages[:65] = "APPROACH"
    stages[65:68] = "PREGRASP"
    stages[68:72] = "OPEN_GRIPPER"
    stages[72:78] = "GRASP_HANDLE"
    stages[78:84] = "OPEN_AND_CLEAR"
    stages[84:90] = "RESTORE_GRIPPER"
    audit["autoopen_stage"] = stages
    raw_contacts = (
        {
            "frame_index": 74,
            "body_a_path": "/World/envs/env_0/Robot/link06",
            "body_b_path": "/World/envs/env_0/door/door_handle",
            "max_contact_force": 4.0,
            "max_normal_impulse": 0.08,
        },
        {
            "frame_index": 74,
            "body_a_path": "/World/envs/env_0/Robot/gripperMover",
            "body_b_path": "/World/envs/env_0/door/door_handle",
            "max_contact_force": 3.0,
            "max_normal_impulse": 0.06,
        },
    )
    episode = AutoOpenEpisodeData(
        episode_index=0,
        seed=1700,
        observation_state=np.zeros((frames, 58), dtype=np.float32),
        action=action,
        images={},
        raw_audit=audit,
        raw_contacts=raw_contacts,
        event_frames={"handle_unlocked": 70, "door_gap_20mm": 77, "robot_clear": 90},
    )
    assert _physical_acceptance_reasons(episode, boundary) == []

    zero_contact_episode = AutoOpenEpisodeData(
        **{
            **episode.__dict__,
            "raw_contacts": tuple(
                {**row, "max_contact_force": 0.0, "max_normal_impulse": 0.0}
                for row in raw_contacts
            ),
        }
    )
    assert any(
        reason.startswith("missing_nonzero_handle_contact_evidence")
        for reason in _physical_acceptance_reasons(zero_contact_episode, boundary)
    )

    moving_audit = dict(audit)
    moving_audit["root_linear_velocity_world"] = np.zeros(
        (frames, 3), dtype=np.float32
    )
    moving_audit["root_linear_velocity_world"][-50:, 0] = 0.08
    moving_episode = AutoOpenEpisodeData(
        **{**episode.__dict__, "raw_audit": moving_audit}
    )
    assert "terminal_base_linear_velocity_not_stable" in _physical_acceptance_reasons(
        moving_episode, boundary
    )

    episode.action[68:72, 14] = -0.785
    assert "invalid_gripper_closed_open_grasp_open_closed_sequence" in (
        _physical_acceptance_reasons(episode, boundary)
    )


def test_standard_shard_aggregation_writes_machine_and_chinese_reports(tmp_path):
    root = tmp_path / "dataset"
    report = {
        "schema_version": "b2_z1_sim_door_v1",
        "total_episodes": 1,
        "accepted_episodes": 1,
        "quarantined_episodes": 0,
        "quarantine": [],
        "exceptions": [],
        "episodes": [
            {
                "seed": 1700,
                "accepted": True,
                "frames": 100,
                "stage_duration_s": {"approach": 1.0},
                "control": {"stage2_entry_q5_target_step_rad": 0.02},
                "contacts": {"nonzero_handle_rows": 20},
                "terminal_actual": {"window_frames": 50},
                "reasons": [],
            }
        ],
    }
    path = root / "shards/episode-seed1700/meta/acceptance_report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))

    aggregate = aggregate_standard_shards(
        root,
        [1700],
        camera_mount_position_noise_m=0.02,
        camera_mount_rotation_noise_deg=15.0,
        collection_wall_time_s=123,
    )

    assert aggregate["accepted_episodes"] == 1
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["collection_wall_time_s"] == 123
    assert manifest["acceptance_report"] == "acceptance_report.json"
    assert (root / "DATASET_ACCEPTANCE_SUMMARY_ZH.md").is_file()
