"""Render cameras from canonical AutoOpen trajectories without stepping physics."""

from __future__ import annotations

import json
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy

from gr00t.rl.auto_open.geometry import _resolved_door_prim_path
from gr00t.rl.auto_open.io import (
    door_asset_stage_snapshot,
    standard_simulation_provenance,
)
from gr00t.rl.data_collection.auto_open_standard import (
    B2_JOINT_NAMES,
    camera_streams_for_mode,
    transform_collection_images,
)
from gr00t.rl.data_collection.auto_open_writer import (
    AutoOpenDatasetWriter,
    AutoOpenEpisodeData,
    StreamingVideoEncoder,
)


def _column_numpy(column) -> np.ndarray:
    """Convert an Arrow scalar or fixed-list column without changing values."""
    import pyarrow as pa

    column = column.combine_chunks()
    value_type = column.type
    if pa.types.is_fixed_size_list(value_type):
        values = column.values.to_numpy(zero_copy_only=False)
        return values.reshape(len(column), value_type.list_size)
    if pa.types.is_string(value_type) or pa.types.is_large_string(value_type):
        return np.asarray(column.to_pylist(), dtype=object)
    return column.to_numpy(zero_copy_only=False)


def _table_mapping(table) -> dict[str, np.ndarray]:
    return {name: _column_numpy(table[name]) for name in table.column_names}


def _plain(value: Any):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    try:
        return value.item()
    except (AttributeError, ValueError):
        return str(value)


def _joint_index(names: list[str], token: str) -> int:
    matches = [index for index, name in enumerate(names) if token in name.lower()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one door {token} joint, got {names}")
    return matches[0]


def _load_source_episode(source_root: Path, seed: int):
    import pyarrow.parquet as pq

    root = source_root / "shards" / f"episode-seed{seed}"
    paths = {
        "data": root / "data/chunk-000/episode_000000.parquet",
        "audit": root / "raw_audit/chunk-000/episode_000000.parquet",
        "contacts": root / "raw_contacts/chunk-000/episode_000000.parquet",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing replay source table: {path}")
    episodes = [
        json.loads(line)
        for line in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(episodes) != 1 or int(episodes[0]["seed"]) != seed:
        raise ValueError(f"source shard does not uniquely describe seed {seed}: {root}")
    report = json.loads((root / "meta/acceptance_report.json").read_text(encoding="utf-8"))
    if report.get("accepted_episodes") != 1 or report.get("quarantined_episodes") != 0:
        raise ValueError(f"replay source seed {seed} is not accepted")
    tables = {name: pq.read_table(path) for name, path in paths.items()}
    if tables["data"].num_rows != tables["audit"].num_rows:
        raise ValueError(f"source frame counts disagree for seed {seed}")
    return root, paths, tables, episodes[0]


def _metadata_contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _metadata_contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _metadata_contains(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected)
            )
        )
    return actual == expected


def _first_scene_mismatch(actual: Any, expected: Any, path="scene"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path}: expected mapping, got {type(actual).__name__}"
        for key, expected_value in expected.items():
            if key not in actual:
                return f"{path}.{key}: missing"
            mismatch = _first_scene_mismatch(
                actual[key], expected_value, f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            detail = ""
            if isinstance(actual, list) and all(
                isinstance(item, dict) and "path" in item
                for item in actual + expected
            ):
                actual_paths = {item["path"] for item in actual}
                expected_paths = {item["path"] for item in expected}
                detail = (
                    f" missing_paths={sorted(expected_paths - actual_paths)}"
                    f" extra_paths={sorted(actual_paths - expected_paths)}"
                )
            return (
                f"{path}: length/type "
                f"{type(actual).__name__}/{len(actual) if isinstance(actual, list) else 'n/a'} "
                f"!= list/{len(expected)}{detail}"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            mismatch = _first_scene_mismatch(
                actual_item, expected_item, f"{path}[{index}]"
            )
            if mismatch is not None:
                return mismatch
        return None
    if actual != expected:
        return f"{path}: actual={actual!r} expected={expected!r}"
    return None


def _normalize_scene_paths(value: Any, env_id: int):
    if isinstance(value, dict):
        return {
            key: _normalize_scene_paths(item, env_id)
            for key, item in value.items()
            if key != "sha256"
        }
    if isinstance(value, list):
        return [_normalize_scene_paths(item, env_id) for item in value]
    if isinstance(value, str):
        return re.sub(r"/World/envs/env_\d+", "/World/envs/env_0", value)
    return value


def _assert_source_scene(env, env_id: int, source_episode: dict, seed: int):
    prim = env.sim.stage.GetPrimAtPath(_resolved_door_prim_path(env, env_id))
    actual = _plain(dict(prim.GetMetadata("customData") or {}))
    expected = _plain(source_episode["provenance"]["door_randomization"])
    if not _metadata_contains(actual, expected):
        differing = sorted(
            key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
        )
        raise RuntimeError(
            f"seed {seed} replay scene differs from source door randomization: {differing}; "
            f"actual_visual={actual.get('visual')!r}; "
            f"expected_visual={expected.get('visual')!r}"
        )
    actual_asset = _normalize_scene_paths(
        door_asset_stage_snapshot(env.sim.stage, prim), env_id
    )
    expected_asset = _normalize_scene_paths(
        source_episode["provenance"]["door_asset"], 0
    )
    actual_asset["prims"].sort(key=lambda item: item["path"])
    expected_asset["prims"].sort(key=lambda item: item["path"])
    if not _metadata_contains(actual_asset, expected_asset):
        raise RuntimeError(
            f"seed {seed} replay USD door subtree differs from the source scene: "
            f"{_first_scene_mismatch(actual_asset, expected_asset)}"
        )
    return actual


def _contacts_from_table(table) -> list[dict[str, Any]]:
    return [dict(row) for row in table.to_pylist()]


def _write_replayed_episode(
    args,
    runner,
    seed: int,
    source_root: Path,
    source_paths: dict[str, Path],
    source_tables: dict,
    source_episode: dict,
    encoders: dict[str, StreamingVideoEncoder],
):
    import pyarrow.parquet as pq

    data = _table_mapping(source_tables["data"])
    audit = _table_mapping(source_tables["audit"])
    provenance = dict(source_episode.get("provenance", {}))
    provenance.update(
        {
            "images_enabled": True,
            "camera_mode": args.collection_camera_mode,
            "camera_mount_randomization": dict(
                args.camera_mount_randomizations[int(seed)]
            ),
            "trajectory_replay": {
                "source_dataset": str(source_root),
                "recorded_state_written_before_each_render": True,
                "source_door_randomization_verified": True,
                "source_scene_usd_subtree_verified": True,
                "renderer_sync_physics_steps": 0,
                "policy_invocations": 0,
                "fsm_invocations": 0,
            },
        }
    )
    simulation = standard_simulation_provenance(args, runner, seed)
    simulation.update(
        {
            "collector": "auto_open_canonical_trajectory_replay",
            "control_backend": "none; recorded state writeback before each render",
            "source_dataset": str(source_root),
            "physics_steps_during_replay": 0,
            "control_invocations_during_replay": 0,
        }
    )
    shard = Path(args.standard_dataset_dir) / f"episode-seed{seed}"
    writer = AutoOpenDatasetWriter(
        shard,
        camera_calibration=args.episode_camera_configs[int(seed)],
        simulation_provenance=simulation,
    )
    accepted = writer.write_episode(
        AutoOpenEpisodeData(
            episode_index=0,
            seed=seed,
            observation_state=np.asarray(data["observation.state"], dtype=np.float32),
            action=np.asarray(data["action"], dtype=np.float32),
            images={},
            raw_audit=audit,
            staged_videos={key: encoder.path for key, encoder in encoders.items()},
            image_frame_counts={key: encoder.frame_count for key, encoder in encoders.items()},
            raw_contacts=_contacts_from_table(source_tables["contacts"]),
            event_frames=dict(source_episode.get("events", {})),
            acceptance_reasons=(),
            provenance=provenance,
        )
    )
    writer.finalize()
    if not accepted:
        raise RuntimeError(f"replayed seed {seed} failed strict dataset acceptance")

    output_paths = {
        "data": shard / "data/chunk-000/episode_000000.parquet",
        "audit": shard / "raw_audit/chunk-000/episode_000000.parquet",
        "contacts": shard / "raw_contacts/chunk-000/episode_000000.parquet",
    }
    for name, output_path in output_paths.items():
        output = pq.read_table(output_path)
        if not source_tables[name].equals(output):
            raise RuntimeError(
                f"seed {seed} replay changed non-image table {name}: "
                f"{source_paths[name]} != {output_path}"
            )


def run_trajectory_replay(args):
    """Write recorded states frame-by-frame and render the requested cameras."""
    source_root = Path(args.replay_source_dataset_dir).resolve()
    seeds = [int(seed) for seed in args.episode_seeds]
    args.rgb_camera_shared_sidecar = True
    args.rgb_camera_sidecar_pool_size = int(args.replay_render_batch_size)
    args.num_envs = int(args.batched_envs)
    args.env_spacing = 8.0 if args.env_spacing is None else args.env_spacing
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    loaded = [_load_source_episode(source_root, seed) for seed in seeds]
    args.doorman_source_asset_snapshots = [
        episode["provenance"]["door_asset"]
        for _, _, _, episode in loaded
    ]
    runner = ManipLoco_Policy(args)
    env = runner.env
    manager = runner.rgb_camera_manager
    if manager is None or not manager.enabled:
        raise RuntimeError("trajectory replay requires initialized RGB cameras")
    manager.set_episode_camera_configs(
        {
            env_id: args.episode_camera_configs[seed]
            for env_id, seed in enumerate(seeds)
        }
    )
    env.reset()
    env.robot.update(0.0)
    env.door.update(0.0)

    streams = camera_streams_for_mode(args.collection_camera_mode)
    source_names = tuple(streams)
    dof_names = list(env.dof_names)
    arm_indices = [dof_names.index(name) for name in env.arm_joint_names]
    b2_indices = [dof_names.index(name) for name in B2_JOINT_NAMES]
    gripper_index = int(env.gripper_joint_ids[0].item())
    door_names = list(env.door.data.joint_names)
    hinge_index = _joint_index(door_names, "hinge")
    handle_index = _joint_index(door_names, "handle")
    latch_index = _joint_index(door_names, "latch")

    audits = []
    lengths = []
    translations = []
    for env_id, (seed, loaded_episode) in enumerate(zip(seeds, loaded)):
        _, _, tables, episode = loaded_episode
        actual_door_randomization = _assert_source_scene(env, env_id, episode, seed)
        episode["provenance"]["door_randomization"] = actual_door_randomization
        audit = _table_mapping(tables["audit"])
        audits.append(audit)
        lengths.append(int(tables["audit"].num_rows))
        current_initial = env.robot.data.root_pos_w[env_id].detach().cpu().numpy()
        source_initial = np.asarray(audit["root_position_world"][0])
        translations.append(
            np.asarray(
                [current_initial[0] - source_initial[0], current_initial[1] - source_initial[1], 0.0],
                dtype=np.float32,
            )
        )
    render_pool_size = min(int(args.replay_render_batch_size), len(seeds))
    slot_queues = [
        iter(range(slot, len(seeds), render_pool_size))
        for slot in range(render_pool_size)
    ]
    active_by_slot = {}
    frame_by_env = {}
    encoders_by_env = {}
    sampled_frame_hashes = {}

    def start_next(slot):
        env_id = next(slot_queues[slot], None)
        if env_id is None:
            active_by_slot.pop(slot, None)
            return
        active_by_slot[slot] = env_id
        frame_by_env[env_id] = 0
        staging = Path(args.output_dir) / "replay-staging" / f"seed-{seeds[env_id]}"
        encoders_by_env[env_id] = {
            stream["key"]: StreamingVideoEncoder(
                staging / stream["key"] / "episode.mp4",
                stream["shape"],
                codec_options={
                    "preset": "ultrafast",
                    "crf": "18",
                    "threads": "2",
                },
            )
            for stream in streams.values()
        }
        sampled_frame_hashes[env_id] = {
            stream["key"]: [] for stream in streams.values()
        }
        print(
            f"[replay] slot={slot} start seed={seeds[env_id]} "
            f"frames={lengths[env_id]}",
            flush=True,
        )

    def finish_env(slot, env_id):
        for encoder in encoders_by_env[env_id].values():
            encoder.close()
        seed = seeds[env_id]
        for key, samples in sampled_frame_hashes[env_id].items():
            second_half = {
                digest
                for frame, digest in samples
                if frame >= lengths[env_id] // 2
            }
            if len(second_half) < 2:
                raise RuntimeError(
                    f"seed {seed} camera {key} is temporally frozen in the "
                    f"second half ({len(second_half)} unique sampled frames)"
                )
        source_episode_root, source_paths, source_tables, source_episode = loaded[
            env_id
        ]
        _write_replayed_episode(
            args,
            runner,
            seed,
            source_episode_root,
            source_paths,
            source_tables,
            source_episode,
            encoders_by_env[env_id],
        )
        print(
            f"[replay] slot={slot} seed={seed} "
            f"frames={source_tables['data'].num_rows} non_image_tables=exact",
            flush=True,
        )
        del frame_by_env[env_id]
        del encoders_by_env[env_id]
        del sampled_frame_hashes[env_id]
        start_next(slot)

    for slot in range(render_pool_size):
        start_next(slot)

    queue_tick = 0
    try:
        while active_by_slot:
            active_env_ids = list(active_by_slot.values())
            robot_position = env.robot.data.joint_pos.clone()
            robot_velocity = env.robot.data.joint_vel.clone()
            root_pose = env.robot.data.root_state_w[:, :7].clone()
            root_velocity = env.robot.data.root_state_w[:, 7:13].clone()
            door_position = env.door.data.joint_pos.clone()
            door_velocity = env.door.data.joint_vel.clone()
            for env_id in active_env_ids:
                audit = audits[env_id]
                frame = frame_by_env[env_id]
                position = (
                    np.asarray(audit["root_position_world"][frame])
                    + translations[env_id]
                )
                quaternion_xyzw = np.asarray(audit["root_quaternion_xyzw"][frame])
                root_pose[env_id, :3] = torch.as_tensor(position, device=env.device)
                root_pose[env_id, 3:7] = torch.as_tensor(
                    quaternion_xyzw[[3, 0, 1, 2]], device=env.device
                )
                root_velocity[env_id, :3] = torch.as_tensor(
                    audit["root_linear_velocity_world"][frame], device=env.device
                )
                root_velocity[env_id, 3:6] = torch.as_tensor(
                    audit["root_angular_velocity_world"][frame], device=env.device
                )
                robot_position[env_id, arm_indices] = torch.as_tensor(
                    audit["arm_joint_position"][frame], device=env.device
                )
                robot_velocity[env_id, arm_indices] = torch.as_tensor(
                    audit["physx_arm_joint_velocity"][frame], device=env.device
                )
                robot_position[env_id, b2_indices] = torch.as_tensor(
                    audit["b2_joint_position"][frame], device=env.device
                )
                robot_velocity[env_id, b2_indices] = torch.as_tensor(
                    audit["b2_joint_velocity"][frame], device=env.device
                )
                robot_position[env_id, gripper_index] = float(
                    audit["gripper_feedback"][frame]
                )
                robot_velocity[env_id, gripper_index] = 0.0
                door_position[env_id, hinge_index] = float(
                    audit["door_hinge_position"][frame]
                )
                door_velocity[env_id, hinge_index] = float(
                    audit["door_hinge_velocity"][frame]
                )
                door_position[env_id, handle_index] = float(
                    audit["handle_position"][frame]
                )
                door_velocity[env_id, handle_index] = float(
                    audit["handle_velocity"][frame]
                )
                door_position[env_id, latch_index] = float(
                    audit["door_latch_position"][frame]
                )
                door_velocity[env_id, latch_index] = 0.0

            env.robot.write_root_pose_to_sim(root_pose)
            env.robot.write_root_velocity_to_sim(root_velocity)
            env.robot.write_joint_state_to_sim(robot_position, robot_velocity)
            env.door.write_joint_state_to_sim(door_position, door_velocity)
            env.sim.forward()
            env.robot.update(float(env.dt))
            env.door.update(float(env.dt))

            rendered_by_env = manager.render_images_batch(
                env_ids=active_env_ids, rig_names=source_names
            )
            for env_id in active_env_ids:
                frame = frame_by_env[env_id]
                rendered = rendered_by_env[env_id]
                source_images = {}
                for name in source_names:
                    frames = rendered.get(name, ())
                    if len(frames) != 1:
                        raise RuntimeError(
                            f"seed {seeds[env_id]} camera {name} returned "
                            f"{len(frames)} images at frame {frame}"
                        )
                    source_images[name] = np.asarray(frames[0])
                transformed = transform_collection_images(
                    source_images, args.collection_camera_mode
                )
                for key, image in transformed.items():
                    encoders_by_env[env_id][key].append(image)
                    if frame % 50 == 0 or frame == lengths[env_id] - 1:
                        sampled_frame_hashes[env_id][key].append(
                            (frame, hashlib.sha256(image.tobytes()).hexdigest())
                        )
                frame_by_env[env_id] += 1

            completed = [
                (slot, env_id)
                for slot, env_id in list(active_by_slot.items())
                if frame_by_env[env_id] >= lengths[env_id]
            ]
            for slot, env_id in completed:
                finish_env(slot, env_id)

            if queue_tick % 250 == 0:
                print(
                    f"[replay] queue_tick={queue_tick} "
                    f"active_seeds={[seeds[env_id] for env_id in active_by_slot.values()]}",
                    flush=True,
                )
            queue_tick += 1
    finally:
        for encoders in encoders_by_env.values():
            for encoder in encoders.values():
                encoder.close()
    if args.headless:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
