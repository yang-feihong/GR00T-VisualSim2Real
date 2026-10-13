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
    TERMINAL_ARM_JOINT_SPEED_RAD_S,
    TERMINAL_BASE_ANGULAR_SPEED_RAD_S,
    TERMINAL_BASE_LINEAR_SPEED_M_S,
    TERMINAL_DOOR_SPEED_RAD_S,
    TERMINAL_GRIPPER_SPEED_RAD_S,
    TERMINAL_STABILITY_FRAMES,
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

ROUND_TRIP_WINDOW_FRAMES = 5
BASE_ROUND_TRIP_JUMP = 0.15
BASE_ROUND_TRIP_RETURN_TOLERANCE = 0.03
ARM_ROUND_TRIP_JUMP = 0.05
ARM_ROUND_TRIP_RETURN_TOLERANCE = 0.015
CONTACT_PENETRATION_LIMITS_M = {
    "foot_ground": 0.005,
    "gripper_handle": 0.0035,
    "gripper_panel": 0.003,
    "door_internal": 0.001,
}
HANDLE_CONTACT_STAGES = frozenset(
    {"GRASP_HANDLE", "TURN_HANDLE", "RETURN_HANDLE", "OPEN_AND_CLEAR"}
)
PANEL_CONTACT_STAGES = frozenset({"PUSH_OPEN", "PUSH_ARM_RETRACT"})


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
            ("contact_point_world", pa.list_(pa.float32(), 3)),
            ("contact_normal_world", pa.list_(pa.float32(), 3)),
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


def _short_round_trips(
    values, *, jump_threshold, return_tolerance, require_sign_reversal=False
):
    """Find a large dispatched-target jump that returns within 100 ms."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    events = []
    for frame in range(1, len(values)):
        jump = values[frame] - values[frame - 1]
        for component in np.flatnonzero(np.abs(jump) >= jump_threshold):
            if require_sign_reversal and not (
                values[frame - 1, component] * values[frame, component] < 0.0
            ):
                continue
            stop = min(frame + ROUND_TRIP_WINDOW_FRAMES + 1, len(values))
            returned = np.flatnonzero(
                np.abs(values[frame + 1 : stop, component] - values[frame - 1, component])
                <= return_tolerance
            )
            if len(returned) == 0:
                continue
            return_frame = frame + 1 + int(returned[0])
            events.append(
                {
                    "frame": int(frame),
                    "return_frame": int(return_frame),
                    "component": int(component),
                    "jump": float(jump[component]),
                }
            )
    return events


def _contact_acceptance_summary(episode, stages):
    abnormal = []
    penetration_violations = []
    class_maxima = {
        name: {"rows": 0, "max_penetration_m": 0.0}
        for name in CONTACT_PENETRATION_LIMITS_M
    }
    for row in episode.raw_contacts:
        force = float(row.get("max_contact_force", 0.0))
        impulse = float(row.get("max_normal_impulse", 0.0))
        if force <= 1.0e-6 and impulse <= 1.0e-9:
            continue
        frame = int(row.get("frame_index", -1))
        stage = str(stages[frame]) if 0 <= frame < len(stages) else "UNKNOWN"
        path_a = str(row.get("body_a_path", ""))
        path_b = str(row.get("body_b_path", ""))
        leaf_a = path_a.rsplit("/", 1)[-1]
        leaf_b = path_b.rsplit("/", 1)[-1]
        leaves = {leaf_a, leaf_b}
        joined = path_a + "|" + path_b
        contact_class = None
        if "ground" in joined.lower():
            robot_leaf = leaf_b if "ground" in path_a.lower() else leaf_a
            if robot_leaf.endswith("_foot"):
                contact_class = "foot_ground"
            else:
                abnormal.append(
                    {
                        "frame": frame,
                        "stage": stage,
                        "pair": [leaf_a, leaf_b],
                        "force_n": force,
                        "impulse_ns": impulse,
                        "reason": "non_foot_robot_ground_contact",
                    }
                )
        elif "door_handle" in leaves and leaves & {"link06", "gripperMover"}:
            if stage in HANDLE_CONTACT_STAGES:
                contact_class = "gripper_handle"
            else:
                abnormal.append(
                    {
                        "frame": frame,
                        "stage": stage,
                        "pair": [leaf_a, leaf_b],
                        "force_n": force,
                        "impulse_ns": impulse,
                        "reason": "handle_contact_outside_allowed_stage",
                    }
                )
        elif "door_panel" in leaves and leaves & {"link06", "gripperMover"}:
            if stage in PANEL_CONTACT_STAGES:
                contact_class = "gripper_panel"
            else:
                abnormal.append(
                    {
                        "frame": frame,
                        "stage": stage,
                        "pair": [leaf_a, leaf_b],
                        "force_n": force,
                        "impulse_ns": impulse,
                        "reason": "panel_contact_outside_push_open",
                    }
                )
        elif "/Robot/" not in joined and "ground" not in joined.lower():
            contact_class = "door_internal"
        else:
            abnormal.append(
                {
                    "frame": frame,
                    "stage": stage,
                    "pair": [leaf_a, leaf_b],
                    "force_n": force,
                    "impulse_ns": impulse,
                    "reason": "unclassified_robot_environment_contact",
                }
            )
        if contact_class is None:
            continue
        penetration = float(row.get("max_penetration_depth", 0.0))
        metrics = class_maxima[contact_class]
        metrics["rows"] += 1
        metrics["max_penetration_m"] = max(
            float(metrics["max_penetration_m"]), penetration
        )
        limit = CONTACT_PENETRATION_LIMITS_M[contact_class]
        if penetration > limit:
            penetration_violations.append(
                {
                    "frame": frame,
                    "stage": stage,
                    "pair": [leaf_a, leaf_b],
                    "class": contact_class,
                    "penetration_m": penetration,
                    "limit_m": limit,
                }
            )
    return {
        "abnormal": abnormal,
        "penetration_violations": penetration_violations,
        "class_maxima": class_maxima,
        "penetration_limits_m": dict(CONTACT_PENETRATION_LIMITS_M),
    }


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
        "arm_joint_velocity",
        "root_linear_velocity_world",
        "root_angular_velocity_world",
        "door_hinge_velocity",
        "gripper_feedback",
        "gripper_target_dispatched",
        "gripper_actuator_target",
        "arm_joint_target_dispatched",
        "autoopen_stage",
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
    stages = np.asarray(audit["autoopen_stage"], dtype=str)
    arm_targets = np.asarray(audit["arm_joint_target_dispatched"], dtype=np.float64)
    gripper_dispatched = np.asarray(
        audit["gripper_target_dispatched"], dtype=np.float64
    )
    stage2_entries = np.flatnonzero(
        (stages[1:] == "PREGRASP") & (stages[:-1] == "APPROACH")
    ) + 1
    if len(stage2_entries) != 1:
        reasons.append(f"stage2_entry_count:{len(stage2_entries)}")
    else:
        stage2_entry = int(stage2_entries[0])
        q5_step = abs(
            float(arm_targets[stage2_entry, 4] - arm_targets[stage2_entry - 1, 4])
        )
        if q5_step > 0.021:
            reasons.append(f"stage2_q5_target_discontinuity:{q5_step:.6f}")
    base_round_trips = _short_round_trips(
        action[:, :3],
        jump_threshold=BASE_ROUND_TRIP_JUMP,
        return_tolerance=BASE_ROUND_TRIP_RETURN_TOLERANCE,
        require_sign_reversal=True,
    )
    arm_round_trips = _short_round_trips(
        arm_targets,
        jump_threshold=ARM_ROUND_TRIP_JUMP,
        return_tolerance=ARM_ROUND_TRIP_RETURN_TOLERANCE,
    )
    if base_round_trips:
        first = base_round_trips[0]
        reasons.append(
            "base_command_short_round_trip:"
            f"count={len(base_round_trips)},frame={first['frame']},"
            f"component={first['component']},jump={first['jump']:.6f}"
        )
    if arm_round_trips:
        first = arm_round_trips[0]
        reasons.append(
            "arm_target_short_round_trip:"
            f"count={len(arm_round_trips)},frame={first['frame']},"
            f"joint={first['component'] + 1},jump={first['jump']:.6f}"
        )
    grasp_hold = np.isin(stages, ("GRASP_HANDLE", "TURN_HANDLE"))
    if np.any(np.abs(gripper_dispatched[grasp_hold]) > 1.0e-6):
        reasons.append("grasp_hold_high_level_target_not_zero")
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

    contact_summary = _contact_acceptance_summary(episode, stages)
    if contact_summary["abnormal"]:
        first = contact_summary["abnormal"][0]
        reasons.append(
            "abnormal_environment_contact:"
            f"count={len(contact_summary['abnormal'])},frame={first['frame']},"
            f"pair={'|'.join(first['pair'])},reason={first['reason']}"
        )
    if contact_summary["penetration_violations"]:
        first = contact_summary["penetration_violations"][0]
        reasons.append(
            "contact_penetration_limit_exceeded:"
            f"count={len(contact_summary['penetration_violations'])},"
            f"frame={first['frame']},class={first['class']},"
            f"depth={first['penetration_m']:.6f},limit={first['limit_m']:.6f}"
        )

    handle_boundary = boundary.boundaries.get("handle_press")
    if handle_boundary is not None:
        physical_fingers = {"link06": False, "gripperMover": False}
        for row in episode.raw_contacts:
            frame = int(row.get("frame_index", -1))
            if not (handle_boundary.start_frame <= frame < handle_boundary.stop_frame):
                continue
            pair = (
                str(row.get("body_a_path", "")),
                str(row.get("body_b_path", "")),
            )
            if not any("door_handle" in path for path in pair):
                continue
            nonzero = (
                float(row.get("max_contact_force", 0.0)) > 1.0e-6
                or float(row.get("max_normal_impulse", 0.0)) > 1.0e-9
            )
            if not nonzero:
                continue
            for finger in physical_fingers:
                if any(finger in path for path in pair):
                    physical_fingers[finger] = True
        missing_fingers = [name for name, seen in physical_fingers.items() if not seen]
        if missing_fingers:
            reasons.append(
                "missing_nonzero_handle_contact_evidence:" + ",".join(missing_fingers)
            )

    if frames < TERMINAL_STABILITY_FRAMES:
        reasons.append("terminal_actual_stability_window_missing")
    else:
        terminal = slice(frames - TERMINAL_STABILITY_FRAMES, frames)
        root_linear = np.asarray(audit["root_linear_velocity_world"], dtype=np.float64)[terminal]
        root_angular = np.asarray(audit["root_angular_velocity_world"], dtype=np.float64)[terminal]
        arm_velocity = np.asarray(audit["arm_joint_velocity"], dtype=np.float64)[terminal]
        hinge_velocity = np.asarray(audit["door_hinge_velocity"], dtype=np.float64)[terminal]
        gripper_feedback = np.asarray(audit["gripper_feedback"], dtype=np.float64)[terminal]
        gripper_velocity = np.abs(np.diff(gripper_feedback, prepend=gripper_feedback[0])) * CONTROL_HZ
        if np.max(np.linalg.norm(root_linear, axis=1)) >= TERMINAL_BASE_LINEAR_SPEED_M_S:
            reasons.append("terminal_base_linear_velocity_not_stable")
        if np.max(np.linalg.norm(root_angular, axis=1)) >= TERMINAL_BASE_ANGULAR_SPEED_RAD_S:
            reasons.append("terminal_base_angular_velocity_not_stable")
        if np.max(np.abs(arm_velocity)) >= TERMINAL_ARM_JOINT_SPEED_RAD_S:
            reasons.append("terminal_arm_velocity_not_stable")
        if np.max(np.abs(hinge_velocity)) >= TERMINAL_DOOR_SPEED_RAD_S:
            reasons.append("terminal_door_velocity_not_stable")
        if np.max(gripper_velocity) >= TERMINAL_GRIPPER_SPEED_RAD_S:
            reasons.append("terminal_gripper_velocity_not_stable")
    return reasons


def _episode_acceptance_metrics(episode, boundary, reasons):
    audit = episode.raw_audit
    frames = len(episode.observation_state)

    def array(name, width=None):
        default = np.zeros((frames, width), dtype=np.float64) if width else np.zeros(frames)
        return np.asarray(audit.get(name, default), dtype=np.float64)

    sim_ns = array("sim_time_ns")
    base_ns = array("base_image_capture_time_ns")
    wrist_ns = array("wrist_image_capture_time_ns")
    arm_targets = array("arm_joint_target_dispatched", 6)
    arm_velocity = array("arm_joint_velocity", 6)
    base_command = array("base_command_dispatched", 3)
    root_linear = array("root_linear_velocity_world", 3)
    root_angular = array("root_angular_velocity_world", 3)
    stages = np.asarray(audit.get("autoopen_stage", [""] * frames), dtype=str)
    contacts = list(episode.raw_contacts)
    base_round_trips = _short_round_trips(
        base_command,
        jump_threshold=BASE_ROUND_TRIP_JUMP,
        return_tolerance=BASE_ROUND_TRIP_RETURN_TOLERANCE,
        require_sign_reversal=True,
    )
    arm_round_trips = _short_round_trips(
        arm_targets,
        jump_threshold=ARM_ROUND_TRIP_JUMP,
        return_tolerance=ARM_ROUND_TRIP_RETURN_TOLERANCE,
    )
    contact_summary = _contact_acceptance_summary(episode, stages)
    handle_contacts = [
        row for row in contacts
        if "door_handle" in str(row.get("body_a_path", ""))
        or "door_handle" in str(row.get("body_b_path", ""))
    ]
    nonzero_handle_contacts = [
        row for row in handle_contacts
        if float(row.get("max_contact_force", 0.0)) > 1.0e-6
        or float(row.get("max_normal_impulse", 0.0)) > 1.0e-9
    ]
    stage_durations = {
        name: (item.stop_frame - item.start_frame) / CONTROL_HZ
        for name, item in (boundary.boundaries.items() if boundary is not None else ())
    }
    target_steps = np.abs(np.diff(arm_targets, axis=0)) if frames > 1 else np.zeros((0, 6))
    stage2_entries = np.flatnonzero(
        (stages[1:] == "PREGRASP") & (stages[:-1] == "APPROACH")
    ) + 1
    stage2_entry = int(stage2_entries[0]) if len(stage2_entries) == 1 else None
    terminal_slice = slice(max(0, frames - CONTROL_HZ), frames)
    return {
        "episode_index": int(episode.episode_index),
        "seed": int(episode.seed),
        "accepted": not bool(reasons),
        "frames": int(frames),
        "duration_s": frames / CONTROL_HZ,
        "stage_duration_s": stage_durations,
        "timing": {
            "nominal_hz": CONTROL_HZ,
            "max_sim_tick_jitter_ns": int(
                np.max(np.abs(np.diff(sim_ns) - 20_000_000)) if frames > 1 else 0
            ),
            "max_camera_skew_ns": int(
                np.max(np.abs(base_ns - wrist_ns)) if frames else 0
            ),
        },
        "control": {
            "max_arm_target_step_rad": (
                np.max(target_steps, axis=0).tolist() if len(target_steps) else [0.0] * 6
            ),
            "max_q5_target_step_rad": float(
                np.max(target_steps[:, 4]) if len(target_steps) else 0.0
            ),
            "stage2_entry_q5_target_step_rad": (
                float(abs(arm_targets[stage2_entry, 4] - arm_targets[stage2_entry - 1, 4]))
                if stage2_entry is not None
                else None
            ),
            "stage2_entry_q5_actual_speed_rad_s": (
                float(abs(arm_velocity[stage2_entry, 4]))
                if stage2_entry is not None
                else None
            ),
            "max_arm_actual_speed_rad_s": float(np.max(np.abs(arm_velocity))) if frames else 0.0,
            "max_base_command_abs": np.max(np.abs(base_command), axis=0).tolist() if frames else [0.0] * 3,
            "base_short_round_trip_count": len(base_round_trips),
            "base_short_round_trips": base_round_trips,
            "arm_short_round_trip_count": len(arm_round_trips),
            "arm_short_round_trips": arm_round_trips,
        },
        "physical_result": {
            "max_handle_angle_deg": float(np.rad2deg(np.max(np.abs(array("handle_position"))))) if frames else 0.0,
            "max_door_gap_m": float(np.max(array("door_edge_gap_m"))) if frames else 0.0,
            "final_door_angle_deg": float(np.rad2deg(array("door_hinge_position")[-1])) if frames else 0.0,
            "robot_clear_frame": episode.event_frames.get("robot_clear"),
        },
        "contacts": {
            "rows": len(contacts),
            "handle_rows": len(handle_contacts),
            "nonzero_handle_rows": len(nonzero_handle_contacts),
            "max_force_n": max((float(row.get("max_contact_force", 0.0)) for row in contacts), default=0.0),
            "max_normal_impulse_ns": max((float(row.get("max_normal_impulse", 0.0)) for row in contacts), default=0.0),
            "max_penetration_m": max((float(row.get("max_penetration_depth", 0.0)) for row in contacts), default=0.0),
            "allowed_class_maxima": contact_summary["class_maxima"],
            "penetration_limits_m": contact_summary["penetration_limits_m"],
            "abnormal_contact_count": len(contact_summary["abnormal"]),
            "abnormal_contacts": contact_summary["abnormal"],
            "penetration_violation_count": len(
                contact_summary["penetration_violations"]
            ),
            "penetration_violations": contact_summary[
                "penetration_violations"
            ],
        },
        "terminal_actual": {
            "window_frames": min(frames, CONTROL_HZ),
            "max_base_linear_speed_m_s": float(np.max(np.linalg.norm(root_linear[terminal_slice], axis=1))) if frames else 0.0,
            "max_base_angular_speed_rad_s": float(np.max(np.linalg.norm(root_angular[terminal_slice], axis=1))) if frames else 0.0,
            "max_arm_speed_rad_s": float(np.max(np.abs(arm_velocity[terminal_slice]))) if frames else 0.0,
        },
        "events": dict(episode.event_frames),
        "reasons": list(reasons),
        "exceptions": [],
    }


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
        self._episode_reports = []

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
            root_linear = np.asarray(
                episode.raw_audit.get("root_linear_velocity_world", np.zeros((frames, 3)))
            )
            root_angular = np.asarray(
                episode.raw_audit.get("root_angular_velocity_world", np.zeros((frames, 3)))
            )
            arm_velocity = np.asarray(
                episode.raw_audit.get("arm_joint_velocity", np.zeros((frames, 6)))
            )
            hinge_velocity = np.asarray(
                episode.raw_audit.get("door_hinge_velocity", np.zeros(frames))
            )
            actual_stable = (
                (np.linalg.norm(root_linear, axis=1) < TERMINAL_BASE_LINEAR_SPEED_M_S)
                & (np.linalg.norm(root_angular, axis=1) < TERMINAL_BASE_ANGULAR_SPEED_RAD_S)
                & (np.max(np.abs(arm_velocity), axis=1) < TERMINAL_ARM_JOINT_SPEED_RAD_S)
                & (np.abs(hinge_velocity) < TERMINAL_DOOR_SPEED_RAD_S)
            )
            boundary = derive_stage_boundaries(
                base_commands=action[:, :3],
                arm_teleop_inactive=action[:, 3].astype(bool),
                arm_reset=action[:, 4].astype(bool),
                gripper_targets=action[:, 14],
                terminal_robot_clear=np.asarray(
                    episode.raw_audit.get("robot_clear_of_door", np.zeros(frames)), dtype=bool
                ),
                terminal_actual_stable=actual_stable,
            )
            reasons.extend(boundary.quarantine_reasons)
            if boundary.boundaries:
                reasons.extend(_physical_acceptance_reasons(episode, boundary))
        if self.simulation_provenance.get("self_collisions_enabled") is True:
            reasons.append("self_collision_dynamics_changed")
        reasons = sorted(set(reasons))
        episode_report = _episode_acceptance_metrics(episode, boundary, reasons)
        self._episode_reports.append(episode_report)
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
                "episodes": self._episode_reports,
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
        stage_values = {}
        for report in self._episode_reports:
            for name, value in report["stage_duration_s"].items():
                stage_values.setdefault(name, []).append(float(value))
        stage_summary = {
            name: {
                "episodes": len(values),
                "min_s": min(values),
                "median_s": float(np.median(values)),
                "max_s": max(values),
            }
            for name, values in stage_values.items()
        }
        rows = [
            "| seed | 结果 | 帧数 | Stage2 入口 q5 目标步进(rad) | 把手最大角(deg) | 门缝最大值(m) | 把手非零接触行 | 原因 |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
        for report in self._episode_reports:
            rows.append(
                f"| {report['seed']} | {'通过' if report['accepted'] else '隔离'} | "
                f"{report['frames']} | {report['control']['stage2_entry_q5_target_step_rad'] or 0.0:.6f} | "
                f"{report['physical_result']['max_handle_angle_deg']:.3f} | "
                f"{report['physical_result']['max_door_gap_m']:.4f} | "
                f"{report['contacts']['nonzero_handle_rows']} | "
                f"{'; '.join(report['reasons']) or '无'} |"
            )
        summary.write_text(
            "# AutoOpen 数据验收摘要\n\n"
            f"- 标准版本：`{SCHEMA_VERSION}`\n"
            f"- 总 episode：`{len(self.accepted) + len(self.quarantined)}`\n"
            f"- 通过：`{len(self.accepted)}`\n"
            f"- 隔离：`{len(self.quarantined)}`\n"
            f"- 总训练帧：`{self._global_frames}`\n"
            f"- 三阶段时长统计：`{json.dumps(stage_summary, ensure_ascii=False)}`\n"
            f"- 隔离原因：`{json.dumps(reason_counts, ensure_ascii=False)}`\n"
            "- 频率：状态、动作及两路相机均为 50 Hz；逐 episode 抖动和跨相机偏差见机器报告。\n"
            "- 接触：逐 episode 报告包含总接触、把手接触、非零力/冲量接触及最大穿透。\n"
            "- 例外：`exceptions: []`\n\n"
            "## Episode 明细\n\n"
            + "\n".join(rows)
            + "\n",
            encoding="utf-8",
        )


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_standard_shards(
    root: Path,
    seeds: Sequence[int],
    *,
    camera_mount_position_noise_m: float,
    camera_mount_rotation_noise_deg: float,
    collection_wall_time_s: int,
    camera_focal_scale_noise_fraction: float = 0.0,
    camera_principal_point_noise_px: float = 0.0,
    camera_distortion_noise_fraction: float = 0.0,
):
    """Write dataset-level delivery reports for independently collected shards."""
    root = Path(root)
    shard_rows = []
    episode_rows = []
    quarantine = []
    exceptions = []
    for seed in seeds:
        name = f"episode-seed{int(seed)}"
        report_path = root / "shards" / name / "meta" / "acceptance_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"missing shard acceptance report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unexpected schema version in {report_path}")
        episodes = report.get("episodes", [])
        if len(episodes) != 1 or int(episodes[0].get("seed", -1)) != int(seed):
            raise ValueError(f"shard report does not uniquely describe seed {seed}")
        shard_rows.append(
            {"name": name, "path": f"shards/{name}", "acceptance": report}
        )
        episode_rows.extend(episodes)
        quarantine.extend(report.get("quarantine", []))
        exceptions.extend(report.get("exceptions", []))
        exceptions.extend(episodes[0].get("exceptions", []))

    reason_counts = {}
    for row in episode_rows:
        for reason in row.get("reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    stage_values = {}
    for row in episode_rows:
        for stage, value in row.get("stage_duration_s", {}).items():
            stage_values.setdefault(stage, []).append(float(value))
    stage_summary = {
        stage: {
            "episodes": len(values),
            "min_s": min(values),
            "median_s": float(np.median(values)),
            "max_s": max(values),
        }
        for stage, values in stage_values.items()
    }
    accepted = sum(bool(row.get("accepted")) for row in episode_rows)
    total_frames = sum(int(row.get("frames", 0)) for row in episode_rows)
    delivery_metrics = {
        "total_frames": total_frames,
        "nominal_hz": CONTROL_HZ,
        "max_sim_tick_jitter_ns": max(
            (int(row.get("timing", {}).get("max_sim_tick_jitter_ns", 0)) for row in episode_rows),
            default=0,
        ),
        "max_camera_skew_ns": max(
            (int(row.get("timing", {}).get("max_camera_skew_ns", 0)) for row in episode_rows),
            default=0,
        ),
        "max_stage2_entry_q5_target_step_rad": max(
            (
                float(row.get("control", {}).get("stage2_entry_q5_target_step_rad") or 0.0)
                for row in episode_rows
            ),
            default=0.0,
        ),
        "min_handle_angle_deg": min(
            (float(row.get("physical_result", {}).get("max_handle_angle_deg", 0.0)) for row in episode_rows),
            default=0.0,
        ),
        "min_door_gap_m": min(
            (float(row.get("physical_result", {}).get("max_door_gap_m", 0.0)) for row in episode_rows),
            default=0.0,
        ),
        "min_final_door_angle_deg": min(
            (float(row.get("physical_result", {}).get("final_door_angle_deg", 0.0)) for row in episode_rows),
            default=0.0,
        ),
        "nonzero_handle_contact_rows": sum(
            int(row.get("contacts", {}).get("nonzero_handle_rows", 0))
            for row in episode_rows
        ),
        "max_penetration_m": max(
            (float(row.get("contacts", {}).get("max_penetration_m", 0.0)) for row in episode_rows),
            default=0.0,
        ),
    }
    machine_report = {
        "schema_version": SCHEMA_VERSION,
        "requested_seeds": [int(seed) for seed in seeds],
        "total_episodes": len(episode_rows),
        "accepted_episodes": accepted,
        "quarantined_episodes": len(episode_rows) - accepted,
        "collection_wall_time_s": int(collection_wall_time_s),
        "delivery_metrics": delivery_metrics,
        "stage_duration_s": stage_summary,
        "reason_distribution": reason_counts,
        "quarantine": quarantine,
        "exceptions": exceptions,
        "episodes": episode_rows,
    }
    _json_dump(root / "acceptance_report.json", machine_report)
    _json_dump(
        root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "requested_seeds": [int(seed) for seed in seeds],
            "requested_episodes": len(seeds),
            "collection_wall_time_s": int(collection_wall_time_s),
            "camera_mount_randomization": {
                "position_radius_m": float(camera_mount_position_noise_m),
                "rotation_angle_deg": float(camera_mount_rotation_noise_deg),
                "focal_scale_fraction": float(camera_focal_scale_noise_fraction),
                "principal_point_radius_px": float(camera_principal_point_noise_px),
                "distortion_relative_fraction": float(camera_distortion_noise_fraction),
                "episode_constant": True,
                "independent_per_camera": True,
            },
            "acceptance_report": "acceptance_report.json",
            "human_report": "DATASET_ACCEPTANCE_SUMMARY_ZH.md",
            "shards": shard_rows,
        },
    )
    rows = [
        "| seed | 结果 | 帧数 | Stage2入口q5步进(rad) | 把手非零接触 | 终态稳定帧 | 原因 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in episode_rows:
        rows.append(
            f"| {row['seed']} | {'通过' if row.get('accepted') else '隔离'} | "
            f"{row.get('frames', 0)} | "
            f"{float(row.get('control', {}).get('stage2_entry_q5_target_step_rad') or 0.0):.6f} | "
            f"{int(row.get('contacts', {}).get('nonzero_handle_rows', 0))} | "
            f"{int(row.get('terminal_actual', {}).get('window_frames', 0))} | "
            f"{'; '.join(row.get('reasons', [])) or '无'} |"
        )
    (root / "DATASET_ACCEPTANCE_SUMMARY_ZH.md").write_text(
        "# AutoOpen 标准数据整批验收摘要\n\n"
        f"- 标准版本：`{SCHEMA_VERSION}`\n"
        f"- 请求/完成 episode：`{len(seeds)}/{len(episode_rows)}`\n"
        f"- 通过：`{accepted}`\n"
        f"- 隔离：`{len(episode_rows) - accepted}`\n"
        f"- 四卡整批墙钟耗时：`{int(collection_wall_time_s)} s`\n"
        f"- 总训练帧：`{total_frames}`；状态、动作及两路相机：`{CONTROL_HZ} Hz`\n"
        f"- 最大仿真时钟抖动：`{delivery_metrics['max_sim_tick_jitter_ns']} ns`；最大跨相机偏差：`{delivery_metrics['max_camera_skew_ns']} ns`\n"
        f"- Stage2入口最大q5目标步进：`{delivery_metrics['max_stage2_entry_q5_target_step_rad']:.6f} rad`\n"
        f"- 把手最小最大按压角：`{delivery_metrics['min_handle_angle_deg']:.3f} deg`；最小门缝：`{delivery_metrics['min_door_gap_m']:.4f} m`；最小最终门角：`{delivery_metrics['min_final_door_angle_deg']:.3f} deg`\n"
        f"- 把手非零接触总行数：`{delivery_metrics['nonzero_handle_contact_rows']}`；最大穿透：`{delivery_metrics['max_penetration_m']:.6f} m`\n"
        f"- 三阶段时长统计：`{json.dumps(stage_summary, ensure_ascii=False)}`\n"
        f"- 隔离原因：`{json.dumps(reason_counts, ensure_ascii=False)}`\n"
        f"- 例外：`{json.dumps(exceptions, ensure_ascii=False)}`\n\n"
        "## Episode 明细\n\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return machine_report
