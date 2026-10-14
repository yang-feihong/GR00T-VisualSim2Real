import math
import json

import numpy as np
import pytest

from gr00t.rl.data_collection.auto_open_standard import (
    ACTION_NAMES,
    OBSERVATION_STATE_NAMES,
    dataset_tool_quaternion_from_gripper_quaternion,
    derive_stage_boundaries,
    height_invariant_pose,
    dataset_tool_pose_from_gripper_pose,
    pack_action,
    pack_observation_state,
    transform_dataset_images,
)
from gr00t.rl.data_collection.auto_open_writer import (
    AutoOpenEpisodeData,
    AutoOpenDatasetWriter,
    StreamingVideoEncoder,
    _contact_acceptance_summary,
    _episode_acceptance_metrics,
    _physical_acceptance_reasons,
    _short_round_trips,
    aggregate_standard_shards,
    _verify_video,
)
from gr00t.rl.data_collection.auto_open_collector import _door_edge_gap_m
from gr00t.rl.data_collection.contact_audit import (
    ContactPairMaximum,
    aggregate_contact_pair_maxima,
)


def _yaw_quaternion(yaw):
    return np.asarray((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))


def _yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _roll_matrix(roll):
    c, s = math.cos(roll), math.sin(roll)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def test_dataset_tool_pose_uses_single_offset_and_no_fixed_mesh_axis_roll():
    expected_rotation = _yaw_matrix(0.37)
    production_rotation = expected_rotation @ _roll_matrix(-math.pi / 2.0)
    production_position = np.asarray((0.42, -0.13, 0.91))

    position, rotation = dataset_tool_pose_from_gripper_pose(
        production_position, production_rotation
    )

    np.testing.assert_array_equal(position, production_position)
    np.testing.assert_allclose(rotation, expected_rotation, atol=1.0e-12)
    production_quaternion = np.asarray(
        (
            math.cos(-math.pi / 4.0),
            math.sin(-math.pi / 4.0),
            0.0,
            0.0,
        )
    )
    dataset_quaternion = dataset_tool_quaternion_from_gripper_quaternion(
        production_quaternion
    )
    np.testing.assert_allclose(dataset_quaternion, (1.0, 0.0, 0.0, 0.0), atol=1.0e-12)


def test_observation_packs_world_velocities_in_body_coordinates():
    packed = pack_observation_state(
        arm_position=np.zeros(6),
        arm_velocity=np.zeros(6),
        gripper_feedback=0.0,
        b2_position=np.zeros(12),
        b2_velocity=np.zeros(12),
        root_position_world=np.zeros(3),
        root_quaternion_wxyz=_yaw_quaternion(math.pi / 2.0),
        root_linear_velocity_world=np.asarray((1.0, 0.0, 0.0)),
        root_angular_velocity_world=np.asarray((0.0, 1.0, 0.0)),
        actual_tool_position_world=np.zeros(3),
        actual_tool_rotation_world=np.eye(3),
    )

    np.testing.assert_allclose(packed[43:46], (0.0, -1.0, 0.0), atol=1.0e-7)
    np.testing.assert_allclose(packed[46:49], (1.0, 0.0, 0.0), atol=1.0e-7)


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


def test_streaming_video_encoder_preserves_frame_count_with_background_worker(tmp_path):
    path = tmp_path / "episode.mp4"
    encoder = StreamingVideoEncoder(path, (16, 20, 3))
    for value in range(12):
        encoder.append(np.full((16, 20, 3), value, dtype=np.uint8))
    encoder.close()

    assert encoder.frame_count == 12
    assert encoder.encode_wall_s > 0.0
    _verify_video(path, (16, 20, 3), 12)


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


def test_short_round_trip_audit_detects_a_jump_returning_within_100ms():
    values = np.zeros((12, 2), dtype=np.float32)
    values[:, 1] = -0.2
    values[4:6, 1] = 0.2
    events = _short_round_trips(
        values,
        jump_threshold=0.15,
        return_tolerance=0.03,
        require_sign_reversal=True,
    )
    assert events == [
        {"frame": 4, "return_frame": 6, "component": 1, "jump": pytest.approx(0.4)}
    ]


def test_arm_round_trip_audit_does_not_flag_sustained_motion_then_retraction():
    values = np.asarray(
        (0.0, -0.043, -0.093, -0.149, -0.212, -0.281, -0.270, -0.261, -0.240, -0.213),
        dtype=np.float32,
    )

    events = _short_round_trips(
        values,
        jump_threshold=0.05,
        return_tolerance=0.015,
        require_isolated_jump=True,
    )

    assert events == []


def test_contact_classifier_rejects_non_foot_ground_contact():
    episode = AutoOpenEpisodeData(
        episode_index=0,
        seed=1700,
        observation_state=np.zeros((10, 58), dtype=np.float32),
        action=np.zeros((10, 25), dtype=np.float32),
        images={},
        raw_audit={},
        raw_contacts=(
            {
                "frame_index": 5,
                "body_a_path": "/World/envs/env_0/Robot/FR_calf",
                "body_b_path": "/World/ground/terrain/GroundPlane/CollisionPlane",
                "max_contact_force": 812.9,
                "max_normal_impulse": 4.0,
                "max_penetration_depth": 0.0,
            },
        ),
    )
    summary = _contact_acceptance_summary(
        episode, np.full(10, "THROUGH", dtype=object)
    )
    assert len(summary["abnormal"]) == 1
    assert summary["abnormal"][0]["reason"] == "non_foot_robot_ground_contact"


def test_contact_aggregation_drops_zero_rows_and_merges_mirrors():
    rows = (
        ContactPairMaximum("/A", "/B", 0.2, 4.0, 0.001, (1, 2, 3), (1, 0, 0)),
        ContactPairMaximum("/B", "/A", 0.3, 3.0, 0.002, (4, 5, 6), (-1, 0, 0)),
        ContactPairMaximum("/C", "/D", 0.0, 0.0, 0.0, (0, 0, 0), (0, 0, 0)),
    )

    aggregated = aggregate_contact_pair_maxima(rows)

    assert len(aggregated) == 1
    assert (aggregated[0].body_a_path, aggregated[0].body_b_path) == ("/A", "/B")
    assert aggregated[0].max_normal_impulse == pytest.approx(0.3)
    assert aggregated[0].max_contact_force == pytest.approx(4.0)
    assert aggregated[0].max_penetration_depth == pytest.approx(0.002)
    assert aggregated[0].contact_point_world == (4.0, 5.0, 6.0)
    assert aggregated[0].contact_normal_world == (1.0, 0.0, 0.0)


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
    metrics = _episode_acceptance_metrics(episode, boundary, [])
    assert metrics["physical_result"]["max_door_gap_before_release_m"] == pytest.approx(0.03)
    assert metrics["physical_result"]["door_gap_at_stage2_end_m"] == pytest.approx(0.03)
    assert metrics["physical_result"]["max_door_gap_episode_m"] == pytest.approx(0.03)
    assert metrics["contacts"]["unique_frame_pair_rows"] == 2

    inconsistent_audit = dict(audit)
    inconsistent_audit["arm_joint_position"] = np.zeros(
        (frames, 6), dtype=np.float32
    )
    inconsistent_audit["arm_joint_position"][80:, 2] = 0.1
    inconsistent_episode = AutoOpenEpisodeData(
        **{**episode.__dict__, "raw_audit": inconsistent_audit}
    )
    assert any(
        reason.startswith("arm_velocity_position_inconsistent")
        for reason in _physical_acceptance_reasons(inconsistent_episode, boundary)
    )

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

    nonzero_terminal_action = action.copy()
    nonzero_terminal_action[-50:, 1] = 0.01
    nonzero_terminal_episode = AutoOpenEpisodeData(
        **{**episode.__dict__, "action": nonzero_terminal_action}
    )
    assert "terminal_base_command_not_zero" in _physical_acceptance_reasons(
        nonzero_terminal_episode, boundary
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
                "contacts": {"nonzero_handle_unique_frame_pair_rows": 20},
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


def test_camera_calibration_covers_renderer_config_and_saved_pixel_transform(tmp_path):
    writer = AutoOpenDatasetWriter(
        tmp_path,
        camera_calibration={"cameras": []},
        simulation_provenance={"camera_calibration_id": "abc123"},
    )

    assert writer.camera_calibration["renderer_runtime_camera_config_id"] == "abc123"
    assert writer.simulation_provenance["renderer_runtime_camera_config_id"] == (
        "abc123"
    )
    base_transform = writer.camera_calibration["dataset_image_transforms"][
        "observation.images.base_0_rgb"
    ]
    assert base_transform["operation"] == "rot90_counterclockwise"
    assert base_transform["source_shape_hwc"] == [480, 640, 3]
    assert base_transform["dataset_shape_hwc"] == [640, 480, 3]
    assert base_transform["source_to_dataset_pixel_homography"] == [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 639.0],
        [0.0, 0.0, 1.0],
    ]
    assert len(writer.camera_calibration["calibration_id"]) == 64
    assert writer.simulation_provenance["camera_calibration_id"] == (
        writer.camera_calibration["calibration_id"]
    )
    assert writer.camera_calibration["calibration_id_source"] == (
        "sha256_of_effective_renderer_config_and_dataset_pixel_transforms"
    )
