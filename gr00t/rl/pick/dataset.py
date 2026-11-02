"""Task-specific acceptance and shared-format dataset output for pick episodes."""

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from gr00t.rl.data_collection.auto_open_standard import (
    ACTION_NAMES,
    CONTROL_HZ,
    OBSERVATION_STATE_NAMES,
    SCHEMA_VERSION,
    camera_streams_for_mode,
)
from gr00t.rl.data_collection.auto_open_writer import (
    _feature_stats,
    _fixed_list_array,
    _json_dump,
    _jsonl_write,
    _verify_video,
    _write_raw_contacts,
    _write_training_parquet,
)


def episode_dataset_root(run):
    """Resolve both standard shards and previously recorded episode datasets."""
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text())
    recorded = summary.get("data_collection", {}).get("dataset")
    if recorded:
        path = Path(recorded)
        if path.is_absolute():
            return path
        if (run / path).is_dir():
            return run / path
        if path.is_dir():
            return path
    return run / "dataset"


def write_initialization_rejection(root, *, seed, task_type, reason, details):
    """A rejected initialization has metadata, never fabricated simulation frames."""
    root = Path(root)
    row = {"seed": int(seed), "accepted": False, "frames": 0, "reasons": [reason]}
    report = {
        "schema_version": SCHEMA_VERSION,
        "task_type": task_type,
        "total_episodes": 1,
        "accepted_episodes": 0,
        "quarantined_episodes": 1,
        "episodes": [row],
        "quarantine": [{**row, "initialization_details": details}],
        "exceptions": [],
    }
    _json_dump(root / "meta/acceptance_report.json", report)
    _json_dump(root / "quarantine/preparation_rejection.json", {"reason": reason, **details})
    _json_dump(
        root / "meta/info.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_type": task_type,
            "total_episodes": 0,
            "total_frames": 0,
            "physics_started": False,
        },
    )
    return report


def pick_acceptance_reasons(record, summary, initial_checks, *, episode=None):
    state, action = record["observation_state"], record["action"]
    reasons = []
    if not summary["success"]:
        reasons.append(f"task_failed:{summary.get('failure')}")
    if not initial_checks["accepted"]:
        reasons.append("initial_state_admission_failed")
    frames = len(state)
    if state.shape != (frames, 58) or action.shape != (frames, 25) or not frames:
        reasons.append("invalid_training_shapes_or_empty")
    if not bool(torch.isfinite(state).all() & torch.isfinite(action).all()):
        reasons.append("non_finite_training_values")
    if len(record["stages"]) != frames or len(record["post"]["root_states"]) != frames:
        reasons.append("raw_training_frame_count_mismatch")
    if not frames:
        return reasons
    post = record["post"]
    initial_height = record["pre"]["object_state"][0, 2]
    final_height = post["object_state"][-1, 2]
    if float(final_height) < 0.30:
        reasons.append("actual_object_final_height_below_0.30m")
    if float(final_height - initial_height) < 0.10:
        reasons.append("actual_object_net_lift_below_0.10m")
    if float(post["ee_pos"][-1, 2]) < 0.30:
        reasons.append("actual_ee_final_height_below_0.30m")
    if len(record["stages"]) < 50 or record["stages"][-50:] != ["VERIFY_RETENTION"] * 50:
        reasons.append("missing_terminal_retention_window")
    if episode is not None and episode.get("task_type") == "pick_basket":
        from types import SimpleNamespace

        from gr00t.rl.pick.basket_controller import basket_vertical_clearance

        corners = episode.get("basket_sampling", {}).get("collider_corners_local_m")
        if corners is None:
            reasons.append("missing_recorded_basket_colliders")
        else:
            points = post["object_state"].new_tensor(
                np.load(Path(episode["asset"]["usd_path"]).with_name("collision_support.npy"))
            )
            basket = SimpleNamespace(boxes=post["root_states"].new_tensor(corners))
            margin, _ = basket_vertical_clearance(
                basket, post["root_states"][-50:], post["object_state"][-50:], points
            )
            if not bool((margin >= episode.get("lift_above_rim_m", 0.10)).all()):
                reasons.append("actual_object_bottom_not_0.10m_above_basket_rim_for_50_frames")
    return sorted(set(reasons))


def _write_audit(path, values, frames):
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = {}
    for name, value in values.items():
        value = np.asarray(value)
        if len(value) != frames:
            raise ValueError(f"Audit frame count mismatch: {name}")
        if value.ndim > 1:
            flat = value.reshape(frames, -1)
            if flat.shape[1]:
                columns[name] = _fixed_list_array(flat, flat.shape[1], pa.float32())
        else:
            columns[name] = pa.array(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, compression="zstd")


def write_pick_dataset(
    root,
    episode,
    record,
    summary,
    initial_checks,
    *,
    staged_videos=None,
    camera_calibration=None,
    contact_audit=None,
):
    root = Path(root)
    reasons = pick_acceptance_reasons(record, summary, initial_checks, episode=episode)
    frames = len(record["observation_state"])
    videos = staged_videos or {}
    camera_mode = summary.get("camera_mode", "dual" if videos else "none")
    shapes = {
        stream["key"]: stream["shape"]
        for stream in camera_streams_for_mode(camera_mode, task_kind="pick_external").values()
    }
    if videos and set(videos) != set(shapes):
        reasons.append("missing_dual_camera_stream")
    for name, path in videos.items():
        _verify_video(Path(path), shapes[name], frames)
    accepted = not reasons
    raw_root = root if accepted else root / "quarantine"
    pre, post = record["pre"], record["post"]
    arm, b2 = record["arm_joint_ids"], record["b2_joint_ids"]
    raw = {
        "frame_index": np.arange(frames),
        "sim_time_ns": np.rint(np.asarray(record["sim_times"]) * 1e9).astype(np.int64),
        "task_stage": record["stages"],
        "root_position_world": pre["root_states"][:, :3],
        "root_quaternion_xyzw": pre["root_states"][:, [4, 5, 6, 3]],
        "root_linear_velocity_world": pre["root_states"][:, 7:10],
        "root_angular_velocity_world": pre["root_states"][:, 10:13],
        "all_joint_position": pre["dof_pos"],
        "all_joint_velocity": pre["dof_vel"],
        "arm_joint_position": pre["dof_pos"][:, arm],
        "arm_joint_velocity": record["observation_state"][:, 6:12],
        "physx_arm_joint_velocity": pre["dof_vel"][:, arm],
        "b2_joint_position": pre["dof_pos"][:, b2],
        "b2_joint_velocity": pre["dof_vel"][:, b2],
        "gripper_feedback": pre["dof_pos"][:, record["gripper_joint_id"]],
        "ee_pose_world_xyzw": torch.cat((pre["ee_pos"], pre["ee_orn"][:, [1, 2, 3, 0]]), -1),
        "object_state_world": pre["object_state"],
        "distractor_states_world": pre["distractor_states"],
        "base_command_dispatched": post["base_command"],
        "arm_joint_target_dispatched": post["arm_position_target"],
        "gripper_target_dispatched": post["gripper_command"].flatten(),
        "post_root_state": post["root_states"],
        "post_all_joint_position": post["dof_pos"],
        "post_all_joint_velocity": post["dof_vel"],
        "post_ee_pose_world_xyzw": torch.cat((post["ee_pos"], post["ee_orn"][:, [1, 2, 3, 0]]), -1),
        "post_object_state_world": post["object_state"],
        "post_distractor_states_world": post["distractor_states"],
        "object_finger_forces_world": post["object_finger_forces"],
        "object_finger_friction_forces_world": post["object_finger_friction_forces"],
        "object_finger_contact_points_world": post["object_finger_contact_points"],
        "robot_body_contact_forces_world": post["contact_forces_w"],
    }
    _write_audit(raw_root / "raw_audit/chunk-000/episode_000000.parquet", raw, frames)
    _write_raw_contacts(
        raw_root / "raw_contacts/chunk-000/episode_000000.parquet", record["contacts"]
    )
    for key, source in videos.items():
        destination = raw_root / "videos" / key / "chunk-000/episode_000000.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), destination)
    if accepted:
        _write_training_parquet(
            root / "data/chunk-000/episode_000000.parquet",
            0,
            record["observation_state"].numpy(),
            record["action"].numpy(),
            0,
        )
    meta = root / "meta"
    target = episode["target_semantics"]
    task = f"Pick up the {target['object_category']} from the {target['support_category']}."
    _jsonl_write(meta / "tasks.jsonl", [{"task_index": 0, "task": task}])
    episode_meta = {
        "episode_index": 0,
        "seed": episode["seed"],
        "length": frames,
        "frames": frames,
        "duration_s": frames / CONTROL_HZ,
        "stage_duration_s": {
            stage: count / CONTROL_HZ for stage, count in Counter(record["stages"]).items()
        },
        "tasks": [task],
        "target_semantics": target,
        "target_asset": episode["asset"],
        "distractors": episode.get("distractors", []),
    }
    _jsonl_write(meta / "episodes.jsonl", [episode_meta] if accepted else [])
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [58],
            "names": list(OBSERVATION_STATE_NAMES),
        },
        "action": {"dtype": "float32", "shape": [25], "names": list(ACTION_NAMES)},
    }
    features.update({name: {"dtype": "video", "shape": list(shapes[name])} for name in videos})
    _json_dump(
        meta / "info.json",
        {
            "codebase_version": "v2.0",
            "schema_version": SCHEMA_VERSION,
            "task_type": episode.get("task_type", "pick_external"),
            "fps": CONTROL_HZ,
            "total_episodes": int(accepted),
            "total_frames": frames if accepted else 0,
            "total_tasks": 1,
            "chunks_size": 1000,
            "features": features,
            "splits": {"train": f"0:{int(accepted)}"},
        },
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "task_type": episode.get("task_type", "pick_external"),
        "total_episodes": 1,
        "accepted_episodes": int(accepted),
        "quarantined_episodes": int(not accepted),
        "quarantine": [] if accepted else [{**episode_meta, "reasons": reasons}],
        "exceptions": [],
        "episodes": [{**episode_meta, "accepted": accepted, "reasons": reasons}],
    }
    _json_dump(meta / "acceptance_report.json", report)
    _json_dump(meta / "camera_calibration.json", camera_calibration or {"camera_mode": "none"})
    _json_dump(
        meta / "simulation_provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_type": episode.get("task_type", "pick_external"),
            "scene": episode,
            "mode": summary["mode"],
            "physics_hz": 200,
            "control_hz": CONTROL_HZ,
            "contact_audit": contact_audit or {},
            "acceptance": summary["acceptance_diagnostics"],
            "images_enabled": bool(videos),
            "camera_mode": camera_mode,
        },
    )
    _json_dump(
        meta / "stats.json",
        {
            "observation.state": _feature_stats(record["observation_state"].numpy()),
            "action": _feature_stats(record["action"].numpy()),
        }
        if accepted
        else {},
    )
    boundaries = {}
    start = 0
    for stop in range(1, frames + 1):
        if stop == frames or record["stages"][stop] != record["stages"][start]:
            name = record["stages"][start]
            boundaries.setdefault(name, []).append(
                {
                    "start_frame": start,
                    "stop_frame": stop,
                    "start_time_s": start / CONTROL_HZ,
                    "stop_time_s": stop / CONTROL_HZ,
                }
            )
            start = stop
    _json_dump(meta / "stage_boundaries.json", {"0": boundaries})
    return report
