"""Pure schema and boundary logic for DATA_STANARD.md v1.0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "b2_z1_sim_door_v1"
CONTROL_HZ = 50
PHYSICS_HZ = 200
PHYSICS_SUBSTEPS = 4
CONTROL_DT = 1.0 / CONTROL_HZ
TASK_PLACEHOLDER = "__unannotated_physical_episode__"
TERMINAL_STABILITY_FRAMES = CONTROL_HZ
TERMINAL_BASE_LINEAR_SPEED_M_S = 0.05
TERMINAL_BASE_ANGULAR_SPEED_RAD_S = 0.05
TERMINAL_ARM_JOINT_SPEED_RAD_S = 0.08
TERMINAL_GRIPPER_SPEED_RAD_S = 0.05
TERMINAL_DOOR_SPEED_RAD_S = np.deg2rad(2.0)

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
B2_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)

ROT6D_NAMES = (
    "rot6d_col0_x",
    "rot6d_col0_y",
    "rot6d_col0_z",
    "rot6d_col1_x",
    "rot6d_col1_y",
    "rot6d_col1_z",
)
POSE9_NAMES = ROT6D_NAMES + ("x", "y", "z")

OBSERVATION_STATE_NAMES = (
    *(f"arm_q_{i}" for i in range(1, 7)),
    *(f"arm_qd_{i}" for i in range(1, 7)),
    "arm_gripper_feedback",
    *(f"b2_{name}_position" for name in B2_JOINT_NAMES),
    *(f"b2_{name}_velocity" for name in B2_JOINT_NAMES),
    "b2_trunk_roll",
    "b2_trunk_pitch",
    "b2_body_height",
    "b2_position_x",
    "b2_position_y",
    "b2_yaw",
    "b2_body_vx",
    "b2_body_vy",
    "b2_body_vz",
    "b2_body_wx",
    "b2_body_wy",
    "b2_body_wz",
    *(f"ee_{name}" for name in POSE9_NAMES),
)

ACTION_NAMES = (
    "b2_vx",
    "b2_vy",
    "b2_omega_z",
    "arm_teleop_inactive",
    "arm_reset",
    *(f"height_invariant_ee_{name}" for name in POSE9_NAMES),
    "gripper_target",
    "task_complete",
    *(f"control_height_invariant_ee_{name}" for name in POSE9_NAMES),
)

assert len(OBSERVATION_STATE_NAMES) == 58
assert len(ACTION_NAMES) == 25


def _as_float_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def quaternion_wxyz_to_matrix(quaternion):
    q = _as_float_array(quaternion, (4,), "quaternion_wxyz")
    norm = np.linalg.norm(q)
    if norm < 1.0e-12:
        raise ValueError("quaternion_wxyz has zero norm")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_rot6d(rotation):
    rotation = _as_float_array(rotation, (3, 3), "rotation")
    return np.concatenate((rotation[:, 0], rotation[:, 1]))


def root_rpy(root_quaternion_wxyz):
    rotation = quaternion_wxyz_to_matrix(root_quaternion_wxyz)
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    pitch = np.arctan2(
        -rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])
    )
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.asarray((roll, pitch, yaw), dtype=np.float64)


def height_invariant_pose(
    root_position_world,
    root_quaternion_wxyz,
    tool_position_world,
    tool_rotation_world,
):
    """Remove global trunk XY/yaw while retaining roll, pitch, and height."""
    root_position = _as_float_array(root_position_world, (3,), "root_position_world")
    tool_position = _as_float_array(tool_position_world, (3,), "tool_position_world")
    tool_rotation = _as_float_array(tool_rotation_world, (3, 3), "tool_rotation_world")
    yaw = root_rpy(root_quaternion_wxyz)[2]
    cosine, sine = np.cos(yaw), np.sin(yaw)
    world_to_yaw = np.asarray(
        ((cosine, sine, 0.0), (-sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    translated = tool_position - np.asarray(
        (root_position[0], root_position[1], 0.0), dtype=np.float64
    )
    local_position = world_to_yaw @ translated
    local_position -= np.asarray((0.22, 0.0, 0.7), dtype=np.float64)
    local_rotation = world_to_yaw @ tool_rotation
    return np.concatenate((matrix_to_rot6d(local_rotation), local_position)).astype(
        np.float32
    )


def pack_observation_state(
    *,
    arm_position,
    arm_velocity,
    gripper_feedback,
    b2_position,
    b2_velocity,
    root_position_world,
    root_quaternion_wxyz,
    root_linear_velocity_world,
    root_angular_velocity_world,
    actual_tool_position_world,
    actual_tool_rotation_world,
):
    arm_position = _as_float_array(arm_position, (6,), "arm_position")
    arm_velocity = _as_float_array(arm_velocity, (6,), "arm_velocity")
    b2_position = _as_float_array(b2_position, (12,), "b2_position")
    b2_velocity = _as_float_array(b2_velocity, (12,), "b2_velocity")
    root_position = _as_float_array(root_position_world, (3,), "root_position_world")
    root_quaternion = _as_float_array(root_quaternion_wxyz, (4,), "root_quaternion_wxyz")
    linear_velocity = _as_float_array(
        root_linear_velocity_world, (3,), "root_linear_velocity_world"
    )
    angular_velocity = _as_float_array(
        root_angular_velocity_world, (3,), "root_angular_velocity_world"
    )
    roll, pitch, yaw = root_rpy(root_quaternion)
    ee_pose = height_invariant_pose(
        root_position,
        root_quaternion,
        actual_tool_position_world,
        actual_tool_rotation_world,
    )
    packed = np.concatenate(
        (
            arm_position,
            arm_velocity,
            np.asarray((float(gripper_feedback),)),
            b2_position,
            b2_velocity,
            np.asarray((roll, pitch, root_position[2])),
            np.asarray((root_position[0], root_position[1], yaw)),
            linear_velocity,
            angular_velocity,
            ee_pose,
        )
    ).astype(np.float32)
    if packed.shape != (58,) or not np.isfinite(packed).all():
        raise ValueError("packed observation.state is not a finite 58-vector")
    return packed


def pack_action(
    *,
    base_command,
    arm_teleop_inactive,
    arm_reset,
    control_height_invariant_ee,
    gripper_target,
):
    base_command = _as_float_array(base_command, (3,), "base_command")
    control_pose = _as_float_array(
        control_height_invariant_ee, (9,), "control_height_invariant_ee"
    )
    inactive = float(bool(arm_teleop_inactive))
    reset = float(bool(arm_reset))
    if inactive == 1.0 and reset == 1.0:
        raise ValueError("arm_teleop_inactive and arm_reset cannot both be 1")
    packed = np.concatenate(
        (
            base_command,
            np.asarray((inactive, reset)),
            control_pose,
            np.asarray((float(gripper_target), 0.0)),
            control_pose,
        )
    ).astype(np.float32)
    if packed.shape != (25,) or not np.isfinite(packed).all():
        raise ValueError("packed action is not a finite 25-vector")
    return packed


def transform_dataset_images(images: Mapping[str, np.ndarray]):
    expected = {"base_0_rgb", "left_wrist_0_rgb"}
    if set(images) != expected:
        raise ValueError(f"expected exactly {sorted(expected)}, got {sorted(images)}")
    base = np.asarray(images["base_0_rgb"])
    wrist = np.asarray(images["left_wrist_0_rgb"])
    if base.shape != (480, 640, 3) or wrist.shape != (480, 640, 3):
        raise ValueError(
            f"source camera shapes must both be (480, 640, 3), got {base.shape}, {wrist.shape}"
        )
    if base.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise ValueError("dataset RGB images must be uint8")
    return {
        "observation.images.base_0_rgb": np.ascontiguousarray(np.rot90(base, 1)),
        "observation.images.left_wrist_0_rgb": np.ascontiguousarray(wrist),
    }


@dataclass(frozen=True)
class StageBoundary:
    start_frame: int
    stop_frame: int
    trigger: str


@dataclass(frozen=True)
class BoundaryResult:
    boundaries: Mapping[str, StageBoundary]
    physical_phase: tuple[str, ...]
    quarantine_reasons: tuple[str, ...]


def _first_sustained(mask: np.ndarray, length: int, start: int = 0):
    if length <= 0:
        raise ValueError("sustained length must be positive")
    run = 0
    for index in range(start, len(mask)):
        run = run + 1 if bool(mask[index]) else 0
        if run >= length:
            return index - length + 1
    return None


def derive_stage_boundaries(
    *,
    base_commands,
    arm_teleop_inactive,
    arm_reset,
    gripper_targets,
    terminal_robot_clear,
    terminal_actual_stable=None,
):
    """Derive the three physical half-open stages from dispatched controls."""
    base_commands = np.asarray(base_commands, dtype=np.float64)
    if base_commands.ndim != 2 or base_commands.shape[1] != 3:
        raise ValueError("base_commands must have shape [frames, 3]")
    frames = len(base_commands)
    inactive = np.asarray(arm_teleop_inactive, dtype=bool)
    reset = np.asarray(arm_reset, dtype=bool)
    gripper = np.asarray(gripper_targets, dtype=np.float64)
    robot_clear = np.asarray(terminal_robot_clear, dtype=bool)
    actual_stable = (
        np.ones(frames, dtype=bool)
        if terminal_actual_stable is None
        else np.asarray(terminal_actual_stable, dtype=bool)
    )
    if any(
        array.shape != (frames,)
        for array in (inactive, reset, gripper, robot_clear, actual_stable)
    ):
        raise ValueError("all boundary signals must have the same frame count")

    moving = np.max(np.abs(base_commands), axis=1) > 0.03
    approach_start = _first_sustained(moving, 5)
    reasons = []
    if approach_start is None:
        return BoundaryResult({}, tuple("pre_roll" for _ in range(frames)), ("missing_approach_motion",))

    active_arm = ~inactive & ~reset
    stage2_start = next(
        (index for index in range(approach_start, frames) if active_arm[index]), None
    )
    if stage2_start is None:
        reasons.append("missing_handle_press_start")
        return BoundaryResult({}, tuple("pre_roll" for _ in range(frames)), tuple(reasons))
    if stage2_start < 25 or np.any(moving[stage2_start - 25 : stage2_start]):
        reasons.append("approach_missing_25_frame_stop")
    if np.any(~inactive[approach_start:stage2_start]) or np.any(reset[approach_start:stage2_start]):
        reasons.append("arm_not_inactive_during_approach")
    if np.any(np.abs(gripper[approach_start:stage2_start]) > 1.0e-6):
        reasons.append("gripper_not_closed_during_approach")

    stage3_start = _first_sustained(moving, 5, start=stage2_start + 1)
    if stage3_start is None:
        reasons.append("missing_traversal_start")
        return BoundaryResult({}, tuple("pre_roll" for _ in range(frames)), tuple(reasons))
    if np.any(moving[stage2_start:stage3_start]):
        reasons.append("base_moved_during_handle_press")
    if np.any(reset[stage2_start:stage3_start]):
        reasons.append("arm_reset_during_handle_press")
    if abs(float(gripper[stage3_start - 1])) > 1.0e-6:
        reasons.append("gripper_not_commanded_closed_before_traversal")

    terminal_hold = ~moving & inactive & ~reset & robot_clear & actual_stable
    terminal_start = _first_sustained(terminal_hold, 50, start=stage3_start)
    if terminal_start is None:
        reasons.append("missing_50_frame_terminal_hold")
        stop = frames
    else:
        stop = terminal_start + 50
        if stop != frames:
            reasons.append("frames_after_terminal_hold")

    boundaries = {
        "approach": StageBoundary(
            approach_start, stage2_start, "first_active_arm_after_25_frame_base_stop"
        ),
        "handle_press": StageBoundary(
            stage2_start, stage3_start, "first_sustained_base_motion_after_final_close"
        ),
        "traversal": StageBoundary(
            stage3_start, stop, "robot_clear_and_50_frame_terminal_hold"
        ),
    }
    phase = ["pre_roll"] * frames
    phase[approach_start:stage2_start] = ["approach"] * (stage2_start - approach_start)
    phase[stage2_start:stage3_start] = ["handle_press"] * (stage3_start - stage2_start)
    phase[stage3_start:stop] = ["traversal"] * (stop - stage3_start)
    if terminal_start is not None:
        phase[terminal_start:stop] = ["terminal_hold"] * (stop - terminal_start)
    return BoundaryResult(boundaries, tuple(phase), tuple(reasons))
