"""Diagnostics compare recorded commands without treating paired missing contacts as errors."""

import json

import pytest
import torch

from gr00t.rl.pick.inspect_run import (
    compare_recorded_runs,
    inspect_force_history,
    inspect_goal_bounds,
    inspect_trunk_transform,
    recorded_stage_metrics,
    terminal_inverse_preview,
)
from gr00t.rl.tests.test_pick_collection import dataset_record


def test_batch_comparison_ignores_only_padding_and_matching_missing_contacts(tmp_path):
    paths = [tmp_path / "single", tmp_path / "batch"]
    for index, path in enumerate(paths):
        path.mkdir()
        (path / "episode.json").write_text(json.dumps({"seed": 1700, "distractors": []}))
        state = {
            "root_states": torch.zeros(2, 13),
            "distractor_states": torch.full((2, index, 13), 100.0),
            "object_finger_contact_points": torch.full((2, 2, 3), torch.nan),
        }
        torch.save(
            {
                "pre": state,
                "post": state,
                "observation_state": torch.zeros(2, 58),
                "action": torch.zeros(2, 25),
            },
            path / "control_record.pt",
        )
    report = compare_recorded_runs(*paths)
    assert report["within_tolerance"]
    assert report["maximum_absolute_difference_by_field"]["post.object_finger_contact_points"] == 0
    assert report["first_nonzero_difference_by_field"]["post.object_finger_contact_points"] is None
    action_trace = report["raw_policy_trace_comparison"]["lowlevel_policy_action"]
    assert not action_trace["available_in_both_runs"]


def test_batch_comparison_reports_raw_network_component_values(tmp_path):
    paths = [tmp_path / "single", tmp_path / "batch"]
    for index, path in enumerate(paths):
        path.mkdir()
        (path / "episode.json").write_text(json.dumps({"seed": 1700, "distractors": []}))
        record = dataset_record(3)
        record["post"]["lowlevel_policy_observation"] = torch.zeros(3, 128)
        record["post"]["lowlevel_policy_action"] = torch.zeros(3, 18)
        record["post"]["physics_substep_torque"] = torch.zeros(3, 4, 19)
        if index:
            record["post"]["lowlevel_policy_observation"][1, 7] = 1e-7
            record["post"]["lowlevel_policy_action"][2, 4] = 1e-3
            record["post"]["physics_substep_torque"][1, 2, 5] = 0.125
        torch.save(record, path / "control_record.pt")
    report = compare_recorded_runs(*paths)
    detail = report["raw_policy_trace_comparison"]["lowlevel_policy_observation"]
    assert detail["available_in_both_runs"]
    assert detail["width"] == 128
    assert detail["first_nonzero_frame"] == 1
    assert detail["different_components"] == 1
    values = detail["largest_first_frame_differences"][0]
    assert values["index"] == 7
    assert values["current"] == 0
    assert values["other"] == pytest.approx(1e-7)
    action_trace = report["raw_policy_trace_comparison"]["lowlevel_policy_action"]
    assert action_trace["first_nonzero_frame"] == 2
    physics = report["physics_substep_comparison"]["physics_substep_torque"]
    assert physics["available_in_both_runs"]
    assert physics["first_nonzero_frame"] == 1
    assert physics["first_nonzero_substep"] == 2
    assert physics["different_joint_indices"] == [5]
    assert physics["other_substep_values"][5] == 0.125


def test_batch_comparison_distinguishes_small_first_changes_from_threshold_crossings(tmp_path):
    paths = [tmp_path / "single", tmp_path / "batch"]
    for index, path in enumerate(paths):
        path.mkdir()
        (path / "episode.json").write_text(json.dumps({"seed": 1700, "distractors": []}))
        record = dataset_record(3)
        record["stages"] = ["GRASP_OBJECT"] * 3
        record["post"]["base_command"] = torch.zeros(3, 3)
        if index:
            record["post"]["base_command"][1, 0] = 1e-7
            record["post"]["base_command"][2, 0] = 1e-3
        torch.save(record, path / "control_record.pt")
    report = compare_recorded_runs(*paths)
    key = "post.base_command"
    assert report["first_nonzero_difference_by_field"][key] == 1
    assert report["first_frame_exceeding_tolerance_by_field"][key] == 2


def test_goal_bound_audit_reports_axes_and_first_affected_frame(tmp_path):
    torch.save(
        {"post": {"ee_goal_cart_local": torch.tensor([[0.0, 0.0, 0.0], [-0.6, 0.0, -0.8]])}},
        tmp_path / "control_record.pt",
    )
    report = inspect_goal_bounds(tmp_path, [-0.5, 1, -0.7, 0.7, -0.75, 0.6])
    assert report["outside_frames_by_xyz"] == [1, 0, 1]
    assert report["first_outside_frame"] == 1


def test_stage_metrics_include_contact_gripper_pocket_and_tracking_for_each_stage():
    record = dataset_record(2)
    record["stages"] = ["MOVE_TO_GRASP", "GRASP_OBJECT"]
    record["gripper_joint_id"] = 18
    record["post"]["ee_goal_cart_world"] = record["post"]["ee_pos"] + torch.tensor([0.0, 0.0, 0.01])
    record["post"]["object_finger_forces"][1, :, 0] = torch.tensor([2.5, 4.0])
    metrics = recorded_stage_metrics(record)
    assert set(metrics) == {"MOVE_TO_GRASP", "GRASP_OBJECT"}
    assert metrics["GRASP_OBJECT"]["finger_force_last_n"] == [2.5, 4.0]
    assert abs(metrics["GRASP_OBJECT"]["ee_tracking_error_last_m"] - 0.01) < 1e-6


def test_force_history_selects_actual_stage_labels_and_preserves_force_convention(tmp_path):
    record = dataset_record(4)
    record["stages"] = ["GRASP_OBJECT", "LIFT_OBJECT", "LIFT_OBJECT", "VERIFY_RETENTION"]
    record["gripper_joint_id"] = 18
    record["post"]["object_finger_forces"][1, 0] = torch.tensor([1.0, 2.0, 3.0])
    record["post"]["object_finger_friction_forces"] = torch.zeros(4, 2, 3)
    record["post"]["object_finger_contact_points"] = torch.full((4, 2, 3), torch.nan)
    torch.save(record, tmp_path / "control_record.pt")
    report = inspect_force_history(tmp_path, "LIFT_OBJECT")
    assert report["frames"] == [1, 2]
    assert report["normal_forces_world_n"][0][0] == [1.0, 2.0, 3.0]
    assert len(report["object_center_gripper_local_m"]) == 2
    assert "not forces on object" in report["force_convention"]
    assert (tmp_path / "force_history_LIFT_OBJECT.json").is_file()


def test_trunk_transform_reconstruction_separates_raw_goal_and_filtered_transform(tmp_path):
    record = dataset_record(3)
    for phase in ("pre", "post"):
        record[phase]["root_states"][:, 2] = torch.tensor([0.52, 0.47, 0.47])
    record["post"]["ee_goal_cart_world"] = torch.tensor([[0.4, 0, 0.2]]).repeat(3, 1)
    record["post"]["ee_goal_orn_quat_world"] = torch.tensor([[1.0, 0, 0, 0]]).repeat(3, 1)
    record["post"]["arm_position_target"] = torch.zeros(3, 6)
    torch.save(record, tmp_path / "control_record.pt")
    report = inspect_trunk_transform(tmp_path)
    assert abs(report["trunk_transform_offset_world_m"][0][2]) < 1e-6
    assert report["trunk_transform_offset_world_m"][1][2] < -0.049
    assert report["raw_ee_goal_world_m"][1][2] > 0.19
    assert "warmup history unavailable" in report["limitations"]


@pytest.mark.parametrize("changed", ["length", "stage", "initial_state"])
def test_batch_comparison_rejects_prefix_only_and_reports_first_divergence(tmp_path, changed):
    paths = [tmp_path / "single", tmp_path / "batch"]
    for index, path in enumerate(paths):
        path.mkdir()
        (path / "episode.json").write_text(json.dumps({"seed": 1700, "distractors": []}))
        record = dataset_record(3 if index and changed == "length" else 2)
        record["stages"] = ["GRASP_OBJECT"] * len(record["action"])
        if index and changed == "stage":
            record["stages"][-1] = "LIFT_OBJECT"
        if index and changed == "initial_state":
            record["pre"]["root_states"][0, 0] += 0.001
        torch.save(record, path / "control_record.pt")
    report = compare_recorded_runs(*paths)
    assert not report["within_tolerance"]
    if changed == "length":
        assert report["frame_counts"] == [2, 3]
    elif changed == "stage":
        assert not report["stage_sequences_equal"]
    else:
        assert report["first_frame_exceeding_tolerance_by_field"]["pre.root_states"] == 0
        assert report["initial_frame_absolute_difference_by_field"]["pre.root_states"] > 0.0009


def test_terminal_inverse_preview_recovers_a_known_pose_without_simulation():
    from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

    from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian

    record = dataset_record(1)
    arm = torch.tensor([[-1.57, 2.6, -1.5, 0.4, 0, 0]])
    root = record["pre"]["root_states"]
    p, rotation, _ = b2z1_gripper_pose_and_jacobian(root[:, :3], root[:, [4, 5, 6, 3]], arm)
    record["pre"]["dof_pos"][:, record["arm_joint_ids"]] = arm
    record["post"]["ee_goal_cart_world"] = p
    record["post"]["ee_goal_orn_quat_world"] = _matrix_to_quaternion_wxyz(rotation)
    report = terminal_inverse_preview(record, body_offsets=(0,), seeds=8)
    assert report["joint_seed_count"] == 11
    assert report["successful_seed_counts"][0] >= 1
    assert report["best_position_error_m"][0] < 0.002
    assert report["not_a_body_policy_or_collision_feasibility_guarantee"]
