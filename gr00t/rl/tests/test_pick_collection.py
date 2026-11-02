"""Pick collection must match the deployed main-flow dataset conventions."""

import json

import numpy as np
import pytest
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.data_collection.auto_open_collector import batched_target_height_invariant_poses
from gr00t.rl.data_collection.auto_open_standard import (
    dataset_tool_pose_from_gripper_pose,
    pack_action,
    pack_observation_state,
    quaternion_wxyz_to_matrix,
)
from gr00t.rl.pick.collection import pack_control_frame


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA packing")
        ),
    ],
)
def test_gpu_pick_packing_matches_shared_schema_and_is_batch_independent(device):
    root = torch.zeros(2, 13, device=device)
    root[:, :3] = root.new_tensor([[1, 2, 0.52], [-3, -2, 0.36]])
    root[:, 3:7] = _euler_xyz_quat(root.new_tensor([[0.1, -0.2, 0.7], [-0.1, 0.15, -2.5]]))
    root[:, 7:] = root.new_tensor(
        [[0.2, 0.1, -0.02, 0.01, -0.04, 0.3], [-0.1, 0.2, 0.04, -0.1, 0.1, -0.3]]
    )
    joints = torch.linspace(-0.2, 0.2, 19, device=device)[None].repeat(2, 1)
    arm_ids = torch.arange(12, 18, device=device)
    b2_ids = torch.arange(12, device=device)
    joints[:, arm_ids] = root.new_tensor([-1.57, 2.6, -1.5, 0.4, 0, 0.03])
    velocity = joints * 0.1
    before = {
        "root_states": root,
        "dof_pos": joints,
        "dof_vel": velocity,
        "ee_pos": root.new_tensor([[1.4, 2.1, 0.9], [-3.2, -2.3, 0.5]]),
        "ee_orn": _euler_xyz_quat(root.new_tensor([[0.2, 0.3, 0.4], [0.5, -0.6, 0.7]])),
    }
    after = {
        "arm_position_target": joints[:, arm_ids] + 0.002,
        "base_command": root.new_tensor([[0.15, 0, 0.2], [-0.2, 0.15, -0.3]]),
        "gripper_command": root.new_tensor([[-0.7], [-0.4]]),
    }
    previous = joints[:, arm_ids] - 0.001
    joint_mode = torch.tensor([False, True], device=device)
    state, action = pack_control_frame(
        before,
        after,
        arm_ids=arm_ids,
        b2_ids=b2_ids,
        gripper_id=18,
        previous_arm_position=previous,
        joint_mode=joint_mode,
    )
    assert state.device == root.device and action.device == root.device
    assert state.shape == (2, 58) and action.shape == (2, 25)
    expected_targets = batched_target_height_invariant_poses(
        root.cpu().numpy(), after["arm_position_target"].cpu().numpy()
    )
    for i in range(2):
        r = root[i].cpu().numpy()
        ee_position, ee_rotation = dataset_tool_pose_from_gripper_pose(
            before["ee_pos"][i].cpu().numpy(),
            quaternion_wxyz_to_matrix(before["ee_orn"][i].cpu().numpy()),
        )
        expected = pack_observation_state(
            arm_position=joints[i, arm_ids].cpu().numpy(),
            arm_velocity=((joints[i, arm_ids] - previous[i]) / 0.02).cpu().numpy(),
            gripper_feedback=float(joints[i, 18]),
            b2_position=joints[i, b2_ids].cpu().numpy(),
            b2_velocity=velocity[i, b2_ids].cpu().numpy(),
            root_position_world=r[:3],
            root_quaternion_wxyz=r[3:7],
            root_linear_velocity_world=r[7:10],
            root_angular_velocity_world=r[10:13],
            actual_tool_position_world=ee_position,
            actual_tool_rotation_world=ee_rotation,
        )
        np.testing.assert_allclose(state[i].cpu().numpy(), expected, atol=3e-6, rtol=1e-5)
        expected_action = pack_action(
            base_command=after["base_command"][i].cpu().numpy(),
            arm_teleop_inactive=bool(joint_mode[i]),
            arm_reset=False,
            control_height_invariant_ee=expected_targets[i],
            gripper_target=float(after["gripper_command"][i, 0]),
        )
        np.testing.assert_allclose(action[i].cpu().numpy(), expected_action, atol=3e-6, rtol=1e-5)
        singleton = pack_control_frame(
            {k: v[i : i + 1] for k, v in before.items()},
            {k: v[i : i + 1] for k, v in after.items()},
            arm_ids=arm_ids,
            b2_ids=b2_ids,
            gripper_id=18,
            previous_arm_position=previous[i : i + 1],
            joint_mode=joint_mode[i : i + 1],
        )
        torch.testing.assert_close(singleton[0][0], state[i])
        torch.testing.assert_close(singleton[1][0], action[i])


def test_first_frame_arm_velocity_is_zero_not_instantaneous_dof_velocity():
    root = torch.zeros(1, 13)
    root[:, 2:4] = torch.tensor([0.52, 1])
    joints = torch.zeros(1, 19)
    joints[:, 12:18] = torch.tensor([-1.57, 2.6, -1.5, 0.4, 0, 0])
    state, _ = pack_control_frame(
        {
            "root_states": root,
            "dof_pos": joints,
            "dof_vel": torch.ones_like(joints) * 99,
            "ee_pos": torch.tensor([[0.5, 0, 0.8]]),
            "ee_orn": torch.tensor([[1.0, 0, 0, 0]]),
        },
        {
            "arm_position_target": joints[:, 12:18],
            "base_command": torch.zeros(1, 3),
            "gripper_command": torch.zeros(1, 1),
        },
        arm_ids=torch.arange(12, 18),
        b2_ids=torch.arange(12),
        gripper_id=18,
    )
    assert not state[:, 6:12].any()


def dataset_record(frames=60):
    from gr00t.rl.pick.collection import PickEpisodeCollector

    state = {
        "root_states": torch.zeros(frames, 13),
        "dof_pos": torch.zeros(frames, 19),
        "dof_vel": torch.zeros(frames, 19),
        "ee_pos": torch.zeros(frames, 3),
        "ee_orn": torch.zeros(frames, 4),
        "object_state": torch.zeros(frames, 13),
        "distractor_states": torch.zeros(frames, 0, 13),
        "arm_position_target": torch.zeros(frames, 6),
        "base_command": torch.zeros(frames, 3),
        "gripper_command": torch.zeros(frames, 1),
        "object_finger_forces": torch.zeros(frames, 2, 3),
        "object_finger_friction_forces": torch.zeros(frames, 2, 3),
        "object_finger_contact_points": torch.zeros(frames, 2, 3),
        "contact_forces_w": torch.zeros(frames, 26, 3),
    }
    state["root_states"][:, 2:4] = torch.tensor([0.52, 1])
    state["ee_orn"][:, 0] = 1
    state["ee_pos"][:, 2] = 0.4
    state["object_state"][:, 2:4] = torch.tensor([0.2, 1])
    pre = {
        key: value.clone() for key, value in state.items() if key in PickEpisodeCollector.PRE_KEYS
    }
    state["object_state"][:, 2] = 0.4
    return {
        "observation_state": torch.zeros(frames, 58),
        "action": torch.zeros(frames, 25),
        "pre": pre,
        "post": state,
        "stages": ["VERIFY_RETENTION"] * frames,
        "sim_times": [i * 0.02 for i in range(frames)],
        "contacts": [],
        "arm_joint_ids": torch.arange(12, 18),
        "b2_joint_ids": torch.arange(12),
        "gripper_joint_id": 18,
    }


def test_pick_writer_uses_shared_parquet_contract_without_door_fields(tmp_path):
    import pyarrow.parquet as pq

    from gr00t.rl.data_collection.auto_open_standard import ACTION_NAMES, OBSERVATION_STATE_NAMES
    from gr00t.rl.pick.dataset import write_pick_dataset

    episode = {
        "seed": 1701,
        "target_semantics": {"object_category": "strawberry", "support_category": "table"},
        "asset": {"name": "012_strawberry"},
        "distractors": [],
    }
    summary = {"success": True, "mode": "canonical", "acceptance_diagnostics": {}}
    report = write_pick_dataset(tmp_path, episode, dataset_record(), summary, {"accepted": True})
    assert report["accepted_episodes"] == 1
    table = pq.read_table(tmp_path / "data/chunk-000/episode_000000.parquet")
    assert table.num_rows == 60
    assert table["observation.state"].type.list_size == 58
    assert table["action"].type.list_size == 25
    assert table["timestamp"][1].as_py() == pytest.approx(0.02)
    info = json.loads((tmp_path / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["names"] == list(OBSERVATION_STATE_NAMES)
    assert info["features"]["action"]["names"] == list(ACTION_NAMES)
    raw = pq.read_table(tmp_path / "raw_audit/chunk-000/episode_000000.parquet")
    assert "post_object_state_world" in raw.column_names
    assert not any("door_hinge" in name for name in raw.column_names)
    tasks = json.loads((tmp_path / "meta/tasks.jsonl").read_text())
    assert tasks["task"] == "Pick up the strawberry from the table."
    assert (tmp_path / "raw_contacts/chunk-000/episode_000000.parquet").exists()


def test_pick_writer_quarantines_false_lift_even_when_controller_reports_success(tmp_path):
    from gr00t.rl.pick.dataset import write_pick_dataset

    record = dataset_record()
    record["post"]["object_state"][:, 2] = 0.2
    report = write_pick_dataset(
        tmp_path,
        {
            "seed": 1700,
            "target_semantics": {"object_category": "lemon", "support_category": "floor"},
            "asset": {"name": "014_lemon"},
        },
        record,
        {"success": True, "mode": "canonical", "acceptance_diagnostics": {}},
        {"accepted": True},
    )
    assert report["accepted_episodes"] == 0
    assert "actual_object_final_height_below_0.30m" in report["quarantine"][0]["reasons"]
    assert not (tmp_path / "data").exists()
    assert (tmp_path / "quarantine/raw_audit/chunk-000/episode_000000.parquet").exists()


def test_basket_writer_checks_actual_object_bottom_over_entire_terminal_window(tmp_path):
    from gr00t.rl.pick.dataset import pick_acceptance_reasons, write_pick_dataset

    np.save(
        tmp_path / "collision_support.npy",
        np.array([[0, 0, -0.03], [0, 0, 0.03]], dtype=np.float32),
    )
    episode = {
        "seed": 1700,
        "task_type": "pick_basket",
        "asset": {"usd_path": str(tmp_path / "object.usda")},
        "target_semantics": {"object_category": "plum", "support_category": "basket"},
        "basket_sampling": {"collider_corners_local_m": [[[0, 0, 0.208]] * 8]},
        "lift_above_rim_m": 0.10,
    }
    record = dataset_record()
    record["post"]["object_state"][:, 2] = 0.90
    summary = {"success": True, "mode": "canonical", "acceptance_diagnostics": {}}
    reasons = pick_acceptance_reasons(record, summary, {"accepted": True}, episode=episode)
    assert not reasons
    record["post"]["object_state"][-50, 2] = 0.85
    assert (
        "actual_object_bottom_not_0.10m_above_basket_rim_for_50_frames"
        in pick_acceptance_reasons(record, summary, {"accepted": True}, episode=episode)
    )
    record["post"]["object_state"][-50, 2] = 0.90
    report = write_pick_dataset(tmp_path / "dataset", episode, record, summary, {"accepted": True})
    assert report["accepted_episodes"] == 1
    info = json.loads((tmp_path / "dataset/meta/info.json").read_text())
    assert info["task_type"] == "pick_basket"
    del episode["basket_sampling"]["collider_corners_local_m"]
    assert "missing_recorded_basket_colliders" in pick_acceptance_reasons(
        record, summary, {"accepted": True}, episode=episode
    )


@pytest.mark.parametrize("debug", [False, True])
def test_collector_preserves_pre_control_states_and_dispatched_actions(debug):
    from types import SimpleNamespace

    from gr00t.rl.data_collection.auto_open_standard import B2_JOINT_NAMES
    from gr00t.rl.pick.collection import PickEpisodeCollector

    record = dataset_record(2)
    env = SimpleNamespace(
        num_envs=1,
        dt=0.02,
        device="cpu",
        arm_joint_ids=torch.arange(12, 18),
        gripper_joint_ids=torch.tensor([18]),
        dof_names=[*B2_JOINT_NAMES, *[f"joint{i}" for i in range(1, 7)], "jointGripper"],
    )
    collector = PickEpisodeCollector(env, debug=debug)
    for i in range(2):
        before = {key: value[i : i + 1] for key, value in record["pre"].items()}
        before["sim_time"] = i * 0.02
        after = {key: value[i : i + 1] for key, value in record["post"].items()}
        after["sim_time"] = (i + 1) * 0.02
        after["ee_goal_cart_world"] = torch.full((1, 3), float(i))
        after["lowlevel_policy_observation"] = torch.full((1, 128), float(i))
        after["lowlevel_policy_action"] = torch.full((1, 18), float(i + 10))
        after["physics_substep_torque"] = torch.full((1, 4, 19), float(i))
        before["dof_pos"][:, 12:18] = i * 0.001
        after["base_command"][:, 0] = 0.15 + i * 0.01
        collector.append(before, after, "VERIFY_RETENTION", joint_mode=torch.tensor([False]))
    packed = collector.finish()
    assert packed["observation_state"][0, 6:12].abs().sum() == 0
    torch.testing.assert_close(packed["observation_state"][1, 6:12], torch.full((6,), 0.05))
    torch.testing.assert_close(packed["action"][:, 0], torch.tensor([0.15, 0.16]))
    torch.testing.assert_close(packed["pre"]["object_state"][:, 2], torch.full((2,), 0.2))
    torch.testing.assert_close(packed["post"]["object_state"][:, 2], torch.full((2,), 0.4))
    assert ("ee_goal_cart_world" in packed["post"]) == debug
    assert ("lowlevel_policy_observation" in packed["post"]) == debug
    assert ("lowlevel_policy_action" in packed["post"]) == debug
    assert ("physics_substep_torque" in packed["post"]) == debug
    if debug:
        assert packed["post"]["physics_substep_torque"].shape == (2, 4, 19)
        torch.testing.assert_close(
            packed["post"]["lowlevel_policy_observation"],
            torch.arange(2, dtype=torch.float32)[:, None].expand(2, 128),
        )
        torch.testing.assert_close(
            packed["post"]["lowlevel_policy_action"],
            torch.arange(10, 12, dtype=torch.float32)[:, None].expand(2, 18),
        )
        torch.testing.assert_close(
            packed["post"]["ee_goal_cart_world"], torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        )


def test_initialization_rejection_is_an_aggregatable_zero_frame_shard(tmp_path):
    from gr00t.rl.data_collection.auto_open_writer import aggregate_standard_shards
    from gr00t.rl.pick.dataset import write_initialization_rejection

    for seed in [3, 4]:
        write_initialization_rejection(
            tmp_path / "shards" / f"episode-seed{seed}",
            seed=seed,
            task_type="pick_basket",
            reason="no_valid_basket_layout",
            details={"object_count": 4},
        )
    report = aggregate_standard_shards(
        tmp_path,
        [3, 4],
        camera_mount_position_noise_m=0,
        camera_mount_rotation_noise_deg=0,
        collection_wall_time_s=5,
        task_name="pick_basket",
    )
    assert report["accepted_episodes"] == 0
    assert report["quarantined_episodes"] == 2
    assert report["delivery_metrics"]["total_frames"] == 0
    assert report["reason_distribution"] == {"no_valid_basket_layout": 2}
    assert not list(tmp_path.rglob("*.parquet"))


def test_batch_finalization_packs_once_and_shards_do_not_share_storage(monkeypatch):
    from types import SimpleNamespace

    from gr00t.rl.data_collection.auto_open_standard import B2_JOINT_NAMES
    from gr00t.rl.pick.collection import PickEpisodeCollector

    source = dataset_record(2)
    env = SimpleNamespace(
        num_envs=3,
        dt=0.02,
        device="cpu",
        arm_joint_ids=torch.arange(12, 18),
        gripper_joint_ids=torch.tensor([18]),
        dof_names=[*B2_JOINT_NAMES, *[f"joint{i}" for i in range(1, 7)], "jointGripper"],
    )
    collector = PickEpisodeCollector(env)
    for frame in range(2):
        before = {
            key: value[frame : frame + 1].expand(3, *value.shape[1:]).clone()
            for key, value in source["pre"].items()
        }
        after = {
            key: value[frame : frame + 1].expand(3, *value.shape[1:]).clone()
            for key, value in source["post"].items()
        }
        before["sim_time"], after["sim_time"] = frame * 0.02, (frame + 1) * 0.02
        after["base_command"][:, 0] = torch.tensor([0.15, 0.20, 0.25])
        collector.append(
            before, after, torch.tensor([1, 2, 3]), joint_mode=torch.zeros(3, dtype=torch.bool)
        )
    original = torch.stack
    calls = []

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "stack", tracked)
    first = collector.finish(env_id=0, frame_count=1)
    first_calls = len(calls)
    second = collector.finish(env_id=1)
    third = collector.finish(env_id=2)
    assert first_calls > 0 and len(calls) == first_calls
    for record, speed in ((first, 0.15), (second, 0.20), (third, 0.25)):
        torch.testing.assert_close(
            record["action"][:, 0], torch.full_like(record["action"][:, 0], speed)
        )
        for value in (record["observation_state"], record["action"], *record["pre"].values()):
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    first["pre"]["dof_pos"].fill_(99)
    assert not (second["pre"]["dof_pos"] == 99).any()
    repeated = collector.finish(env_id=0, frame_count=1)
    assert not (repeated["pre"]["dof_pos"] == 99).any()
