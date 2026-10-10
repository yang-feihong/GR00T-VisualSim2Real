"""LeRobot v2 writer and validator for standard AutoOpen physical episodes."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .auto_open_standard import (
    ACTION_NAMES,
    CONTROL_HZ,
    OBSERVATION_STATE_NAMES,
    SCHEMA_VERSION,
    TASK_PLACEHOLDER,
    derive_stage_boundaries,
)


CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
)
CAMERA_SHAPES = {
    CAMERA_KEYS[0]: (640, 480, 3),
    CAMERA_KEYS[1]: (480, 640, 3),
}


@dataclass
class AutoOpenEpisodeData:
    episode_index: int
    seed: int
    observation_state: np.ndarray
    action: np.ndarray
    images: Mapping[str, Sequence[np.ndarray]]
    raw_audit: Mapping[str, Any]
    staged_videos: Mapping[str, Path] = field(default_factory=dict)
    image_frame_counts: Mapping[str, int] = field(default_factory=dict)
    raw_contacts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    event_frames: Mapping[str, int | None] = field(default_factory=dict)
    acceptance_reasons: Sequence[str] = field(default_factory=tuple)
    provenance: Mapping[str, Any] = field(default_factory=dict)


def _json_dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _jsonl_write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fixed_list_array(values, size, value_type):
    import pyarrow as pa

    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != size:
        raise ValueError(f"fixed list values must have shape [N, {size}], got {array.shape}")
    flat = pa.array(array.reshape(-1), type=value_type)
    return pa.FixedSizeListArray.from_arrays(flat, size)


def _write_training_parquet(path, episode_index, state, action, global_start):
    import pyarrow as pa
    import pyarrow.parquet as pq

    frames = len(state)
    table = pa.table(
        {
            "observation.state": _fixed_list_array(state, 58, pa.float32()),
            "action": _fixed_list_array(action, 25, pa.float32()),
            "episode_index": pa.array(
                np.full(frames, episode_index, dtype=np.int64), type=pa.int64()
            ),
            "frame_index": pa.array(np.arange(frames, dtype=np.int64)),
            "index": pa.array(
                np.arange(global_start, global_start + frames, dtype=np.int64)
            ),
            "timestamp": pa.array(
                np.arange(frames, dtype=np.float64) / CONTROL_HZ, type=pa.float64()
            ),
            "task_index": pa.array(np.zeros(frames, dtype=np.int64)),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


_AUDIT_FIXED_LISTS = {
    "root_position_world": 3,
    "root_quaternion_xyzw": 4,
    "root_linear_velocity_world": 3,
    "root_angular_velocity_world": 3,
    "arm_joint_position": 6,
    "arm_joint_velocity": 6,
    "b2_joint_position": 12,
    "b2_joint_velocity": 12,
    "foot_position_world": 12,
    "foot_contact": 4,
    "foot_normal_force": 4,
    "ee_pose_world_xyzw": 7,
    "arm_joint_position_target": 6,
    "base_command_pre": 3,
    "base_command_dispatched": 3,
    "arm_joint_target_pre": 6,
    "arm_joint_target_dispatched": 6,
}


def _write_raw_audit(path: Path, audit: Mapping[str, Any], frames: int):
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = {}
    for name, values in audit.items():
        values = np.asarray(values)
        if len(values) != frames:
            raise ValueError(f"raw_audit.{name} has {len(values)} rows, expected {frames}")
        if name in _AUDIT_FIXED_LISTS:
            size = _AUDIT_FIXED_LISTS[name]
            value_type = pa.bool_() if name == "foot_contact" else pa.float32()
            columns[name] = _fixed_list_array(values, size, value_type)
        elif values.dtype.kind == "b":
            columns[name] = pa.array(values, type=pa.bool_())
        elif values.dtype.kind in "iu":
            columns[name] = pa.array(values.astype(np.int64), type=pa.int64())
        elif values.dtype.kind in "f":
            columns[name] = pa.array(values.astype(np.float32), type=pa.float32())
        else:
            columns[name] = pa.array(values.astype(str), type=pa.string())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, compression="zstd")


def _write_raw_contacts(path: Path, rows: Sequence[Mapping[str, Any]]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("episode_index", pa.int64()),
            ("frame_index", pa.int64()),
            ("sim_time_ns", pa.int64()),
            ("body_a_path", pa.string()),
            ("body_b_path", pa.string()),
            ("max_normal_impulse", pa.float32()),
            ("max_contact_force", pa.float32()),
            ("max_penetration_depth", pa.float32()),
            ("ccd_triggered", pa.bool_()),
        ]
    )
    table = pa.Table.from_pylist(list(rows), schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


class StreamingVideoEncoder:
    """Bounded-memory H.264 writer for one 50 Hz camera stream."""

    def __init__(self, path: Path, expected_shape):
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "dataset video writing requires PyAV; install gr00t[dataset]"
            ) from exc
        self._av = av
        self.path = Path(path)
        self.expected_shape = tuple(expected_shape)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(self.path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=CONTROL_HZ)
        height, width, channels = self.expected_shape
        if channels != 3:
            raise ValueError("RGB videos require three channels")
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "23"}
        self.frame_count = 0
        self.closed = False

    def append(self, image):
        if self.closed:
            raise RuntimeError("cannot append to a closed video encoder")
        image = np.asarray(image)
        if image.shape != self.expected_shape or image.dtype != np.uint8:
            raise ValueError(
                f"video frame {self.frame_count} must be uint8 "
                f"{self.expected_shape}, got {image.dtype} {image.shape}"
            )
        frame = self._av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frame_count += 1

    def close(self):
        if self.closed:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.closed = True


def _verify_video(path: Path, expected_shape, expected_frames: int):
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("dataset video writing requires PyAV; install gr00t[dataset]") from exc

    height, width, channels = expected_shape
    decoded = 0
    with av.open(str(path), mode="r") as container:
        video_streams = list(container.streams.video)
        if len(video_streams) != 1:
            raise ValueError(f"{path} has {len(video_streams)} video streams")
        stream = video_streams[0]
        if stream.average_rate is None or abs(float(stream.average_rate) - CONTROL_HZ) > 1.0e-6:
            raise ValueError(f"{path} is not {CONTROL_HZ} fps")
        for frame in container.decode(stream):
            if (frame.height, frame.width) != (height, width):
                raise ValueError(f"decoded frame has wrong size: {(frame.height, frame.width)}")
            decoded += 1
    if decoded != expected_frames:
        raise ValueError(f"decoded {decoded} frames from {path}, expected {expected_frames}")


def _encode_and_verify_video(path: Path, frames: Sequence[np.ndarray], expected_shape):
    encoder = StreamingVideoEncoder(path, expected_shape)
    try:
        for image in frames:
            encoder.append(image)
    finally:
        encoder.close()
    _verify_video(path, expected_shape, len(frames))


def _feature_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
    }


def _first_index(mask, start=0):
    indices = np.flatnonzero(np.asarray(mask, dtype=bool)[int(start) :])
    return None if len(indices) == 0 else int(start) + int(indices[0])


def _physical_acceptance_reasons(episode, boundary):
    """Validate physical requirements that are not boundary-only checks."""
    audit = episode.raw_audit
    reasons = []
    frames = len(episode.observation_state)
    required = (
        "sim_time_ns",
        "base_image_capture_time_ns",
        "wrist_image_capture_time_ns",
        "handle_position",
        "door_edge_gap_m",
        "arm_joint_position",
        "robot_clear_of_door",
    )
    missing = [name for name in required if name not in audit]
    if missing:
        return [f"missing_raw_audit_fields:{','.join(missing)}"]

    expected_ns = np.arange(frames, dtype=np.int64) * 20_000_000
    sim_time_ns = np.asarray(audit["sim_time_ns"], dtype=np.int64)
    if sim_time_ns.shape != (frames,) or not np.array_equal(sim_time_ns, expected_ns):
        reasons.append("invalid_50hz_sim_timestamps")
    base_camera_ns = np.asarray(audit["base_image_capture_time_ns"], dtype=np.int64)
    wrist_camera_ns = np.asarray(audit["wrist_image_capture_time_ns"], dtype=np.int64)
    if (
        base_camera_ns.shape != (frames,)
        or wrist_camera_ns.shape != (frames,)
        or np.any(np.abs(base_camera_ns - expected_ns) > 10_000_000)
        or np.any(np.abs(wrist_camera_ns - expected_ns) > 10_000_000)
        or np.any(np.abs(base_camera_ns - wrist_camera_ns) > 10_000_000)
    ):
        reasons.append("camera_timestamps_outside_10ms_tolerance")

    action = np.asarray(episode.action, dtype=np.float32)
    if action.shape == (frames, 25):
        inactive = action[:, 3] > 0.5
        reset = action[:, 4] > 0.5
        if np.any(inactive & reset):
            reasons.append("illegal_arm_mode_1_1")
        gripper = action[:, 14]
        closed = np.abs(gripper) <= 1.0e-4
        fully_open = gripper <= (-np.pi / 3 + 1.0e-3)
        first_open = _first_index(fully_open)
        grasp_close = (
            None
            if first_open is None
            else _first_index(gripper > gripper[first_open] + 0.10, first_open + 1)
        )
        release_open = (
            None
            if grasp_close is None
            else _first_index(fully_open, grasp_close + 1)
        )
        final_close = (
            None
            if release_open is None
            else _first_index(closed, release_open + 1)
        )
        if not closed[0] or None in (first_open, grasp_close, release_open, final_close):
            reasons.append("invalid_gripper_closed_open_grasp_open_closed_sequence")
        elif not np.all(closed[final_close:]):
            reasons.append("gripper_reopened_after_final_close")
    else:
        release_open = None

    handle = boundary.boundaries.get("handle_press")
    traversal = boundary.boundaries.get("traversal")
    if handle is not None:
        handle_angle = np.asarray(audit["handle_position"], dtype=np.float64)[
            handle.start_frame : handle.stop_frame
        ]
        if len(handle_angle) == 0 or np.max(np.abs(handle_angle)) < np.deg2rad(20.0):
            reasons.append("handle_never_reached_20deg")
        gap_frame = episode.event_frames.get("door_gap_20mm")
        if gap_frame is None:
            reasons.append("door_gap_never_reached_20mm")
        elif release_open is not None and int(gap_frame) > int(release_open):
            reasons.append("gripper_released_before_20mm_door_gap")
        if episode.event_frames.get("handle_unlocked") is None:
            reasons.append("handle_never_unlocked_latch")
    if traversal is not None and traversal.start_frame > 0:
        arm_q = np.asarray(audit["arm_joint_position"], dtype=np.float64)
        if abs(float(arm_q[traversal.start_frame - 1, 5])) > 0.2:
            reasons.append("q6_outside_0p2rad_before_traversal")
    if episode.event_frames.get("robot_clear") is None:
        reasons.append("robot_never_cleared_door_plane")
    return reasons


class AutoOpenDatasetWriter:
    """Write independently collected episodes without touching AutoOpen control."""

    def __init__(self, root: Path, *, camera_calibration, simulation_provenance):
        self.root = Path(root)
        self.camera_calibration = dict(camera_calibration)
        self.simulation_provenance = dict(simulation_provenance)
        self.accepted = []
        self.quarantined = []
        self.boundaries = {}
        self._states = []
        self._actions = []
        self._global_frames = 0

    def write_episode(self, episode: AutoOpenEpisodeData):
        state = np.asarray(episode.observation_state, dtype=np.float32)
        action = np.asarray(episode.action, dtype=np.float32)
        reasons = list(episode.acceptance_reasons)
        if state.ndim != 2 or state.shape[1:] != (58,):
            reasons.append(f"observation_state_shape:{state.shape}")
        if action.ndim != 2 or action.shape[1:] != (25,):
            reasons.append(f"action_shape:{action.shape}")
        frames = len(state)
        if len(action) != frames:
            reasons.append("state_action_frame_count_mismatch")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            reasons.append("non_finite_training_values")
        use_staged_videos = bool(episode.staged_videos)
        for key in CAMERA_KEYS:
            if use_staged_videos:
                count = int(episode.image_frame_counts.get(key, -1))
                path = episode.staged_videos.get(key)
                if path is None or not Path(path).is_file():
                    reasons.append(f"missing_camera:{key}")
                elif count != frames:
                    reasons.append(f"camera_frame_count:{key}:{count}:{frames}")
                else:
                    try:
                        _verify_video(Path(path), CAMERA_SHAPES[key], frames)
                    except (RuntimeError, ValueError) as exc:
                        reasons.append(f"camera_video_invalid:{key}:{exc}")
            elif key not in episode.images:
                reasons.append(f"missing_camera:{key}")
            elif len(episode.images[key]) != frames:
                reasons.append(f"camera_frame_count:{key}:{len(episode.images[key])}:{frames}")

        boundary = None
        if state.shape == (frames, 58) and action.shape == (frames, 25):
            boundary = derive_stage_boundaries(
                base_commands=action[:, :3],
                arm_teleop_inactive=action[:, 3].astype(bool),
                arm_reset=action[:, 4].astype(bool),
                gripper_targets=action[:, 14],
                terminal_robot_clear=np.asarray(
                    episode.raw_audit.get("robot_clear_of_door", np.zeros(frames)), dtype=bool
                ),
            )
            reasons.extend(boundary.quarantine_reasons)
            if boundary.boundaries:
                reasons.extend(_physical_acceptance_reasons(episode, boundary))
        if self.simulation_provenance.get("self_collisions_enabled") is not True:
            reasons.append("self_collision_diagnostics_disabled")
        reasons = sorted(set(reasons))
        chunk = episode.episode_index // 1000
        chunk_name = f"chunk-{chunk:03d}"
        episode_name = f"episode_{episode.episode_index:06d}"
        raw_root = self.root if not reasons else self.root / "quarantine"
        audit = dict(episode.raw_audit)
        if boundary is not None:
            audit["physical_phase"] = np.asarray(boundary.physical_phase)
        _write_raw_audit(
            raw_root / "raw_audit" / chunk_name / f"{episode_name}.parquet",
            audit,
            frames,
        )
        _write_raw_contacts(
            raw_root / "raw_contacts" / chunk_name / f"{episode_name}.parquet",
            episode.raw_contacts,
        )
        for key in CAMERA_KEYS:
            destination = raw_root / "videos" / key / chunk_name / f"{episode_name}.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if use_staged_videos:
                source_value = episode.staged_videos.get(key)
                if source_value is not None and Path(source_value).is_file():
                    shutil.move(str(source_value), str(destination))
            else:
                _encode_and_verify_video(
                    destination,
                    episode.images[key],
                    CAMERA_SHAPES[key],
                )
        if reasons:
            self.quarantined.append(
                {
                    "episode_index": episode.episode_index,
                    "seed": episode.seed,
                    "frames": frames,
                    "reasons": reasons,
                    "provenance": dict(episode.provenance),
                }
            )
            return False

        _write_training_parquet(
            self.root / "data" / chunk_name / f"{episode_name}.parquet",
            episode.episode_index,
            state,
            action,
            self._global_frames,
        )
        self.accepted.append(
            {
                "episode_index": episode.episode_index,
                "seed": episode.seed,
                "length": frames,
                "events": dict(episode.event_frames),
                "provenance": dict(episode.provenance),
            }
        )
        self.boundaries[str(episode.episode_index)] = {
            name: {
                "start_frame": item.start_frame,
                "stop_frame": item.stop_frame,
                "start_time_s": item.start_frame / CONTROL_HZ,
                "stop_time_s": item.stop_frame / CONTROL_HZ,
                "trigger": item.trigger,
            }
            for name, item in boundary.boundaries.items()
        }
        self._states.append(state)
        self._actions.append(action)
        self._global_frames += frames
        return True

    def finalize(self):
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        _jsonl_write(meta / "tasks.jsonl", [{"task_index": 0, "task": TASK_PLACEHOLDER}])
        _jsonl_write(meta / "episodes.jsonl", self.accepted)
        _json_dump(meta / "stage_boundaries.json", self.boundaries)
        provenance = {
            "schema_version": SCHEMA_VERSION,
            **self.simulation_provenance,
            "episodes": {
                str(item["episode_index"]): item.get("provenance", {})
                for item in (*self.accepted, *self.quarantined)
            },
        }
        _json_dump(meta / "simulation_provenance.json", provenance)
        _json_dump(meta / "camera_calibration.json", self.camera_calibration)
        _json_dump(
            meta / "acceptance_report.json",
            {
                "schema_version": SCHEMA_VERSION,
                "total_episodes": len(self.accepted) + len(self.quarantined),
                "accepted_episodes": len(self.accepted),
                "quarantined_episodes": len(self.quarantined),
                "quarantine": self.quarantined,
                "exceptions": [],
            },
        )
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [58],
                "names": list(OBSERVATION_STATE_NAMES),
            },
            "action": {
                "dtype": "float32",
                "shape": [25],
                "names": list(ACTION_NAMES),
            },
            CAMERA_KEYS[0]: {"dtype": "video", "shape": [640, 480, 3]},
            CAMERA_KEYS[1]: {"dtype": "video", "shape": [480, 640, 3]},
        }
        _json_dump(
            meta / "info.json",
            {
                "codebase_version": "v2.0",
                "schema_version": SCHEMA_VERSION,
                "fps": CONTROL_HZ,
                "total_episodes": len(self.accepted),
                "total_frames": self._global_frames,
                "total_tasks": 1,
                "chunks_size": 1000,
                "features": features,
                "splits": {"train": f"0:{len(self.accepted)}"},
            },
        )
        stats = {}
        if self._states:
            stats["observation.state"] = _feature_stats(np.concatenate(self._states))
            stats["action"] = _feature_stats(np.concatenate(self._actions))
        _json_dump(meta / "stats.json", stats)
        summary = self.root / "DATASET_ACCEPTANCE_SUMMARY_ZH.md"
        reason_counts = {}
        for item in self.quarantined:
            for reason in item["reasons"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        summary.write_text(
            "# AutoOpen 数据验收摘要\n\n"
            f"- 标准版本：`{SCHEMA_VERSION}`\n"
            f"- 总 episode：`{len(self.accepted) + len(self.quarantined)}`\n"
            f"- 通过：`{len(self.accepted)}`\n"
            f"- 隔离：`{len(self.quarantined)}`\n"
            f"- 总训练帧：`{self._global_frames}`\n"
            f"- 隔离原因：`{json.dumps(reason_counts, ensure_ascii=False)}`\n",
            encoding="utf-8",
        )


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
