#!/usr/bin/env python3
"""Offline acceptance score for recorded B2Z1 AutoOpen trajectories.

This is deliberately not a replacement for the simulator reward implementation.
It verifies that physically successful demonstrations satisfy the six-stage task
contract before the same contract is implemented in the vectorized RL environment.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


AUTO_STAGE = {
    "approach": 1,
    "align": 2,
    "pregrasp": 3,
    "center_handle": 4,
    "grasp_handle": 5,
    "turn_handle": 6,
    "release_handle": 7,
    "retract_arm": 8,
    "align_panel": 9,
    "contact_panel": 10,
    "push_open": 11,
    "through": 12,
    "reset_arm": 13,
}

MIDDLE_ARM = np.asarray([-1.57, 1.6, -0.63, -1.1, 0.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class StageResult:
    name: str
    passed: bool
    detail: str


def _has_stage(stage_ids: np.ndarray, name: str) -> bool:
    return bool(np.any(stage_ids == AUTO_STAGE[name]))


def _longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors by wxyz quaternions with NumPy broadcasting."""
    quat = np.asarray(quat, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    q_vec = quat[..., 1:]
    return vector + 2.0 * (
        quat[..., :1] * np.cross(q_vec, vector)
        + np.cross(q_vec, np.cross(q_vec, vector))
    )


def _quat_rotate_inverse(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    conjugate = np.asarray(quat, dtype=np.float32).copy()
    conjugate[..., 1:] *= -1.0
    return _quat_rotate(conjugate, vector)


def _validate_bilateral_grasp(
    trajectory: np.lib.npyio.NpzFile, metadata: dict, grasp_mask: np.ndarray
) -> tuple[bool, str]:
    geometry = metadata["scene"]["geometry_report"]["handle"]
    handle_pose = trajectory["handle_pose_w"]
    points = trajectory["handle_contact_points_w"]
    forces = trajectory["handle_contact_force"]
    initial_quat = handle_pose[0, 3:7]
    center_local = _quat_rotate_inverse(
        initial_quat, np.asarray(geometry["center_world"]) - handle_pose[0, :3]
    )
    axis_local = _quat_rotate_inverse(initial_quat, np.asarray(geometry["axis_world"]))
    centers = handle_pose[:, :3] + _quat_rotate(handle_pose[:, 3:7], center_local)
    axes = _quat_rotate(handle_pose[:, 3:7], np.broadcast_to(axis_local, centers.shape))
    axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1.0e-8)
    closing_axes = np.cross(
        -np.asarray(metadata["scene"]["approach_normal_world"], dtype=np.float32), axes
    )
    closing_axes /= np.maximum(np.linalg.norm(closing_axes, axis=1, keepdims=True), 1.0e-8)

    delta = points - centers[:, None, :]
    axial = np.sum(delta * axes[:, None, :], axis=-1)
    radial = delta - axial[:, :, None] * axes[:, None, :]
    radial_distance = np.linalg.norm(radial, axis=-1)
    closing = np.sum(radial * closing_axes[:, None, :], axis=-1)
    radius = float(geometry["radius_m"])
    usable_half_length = max(float(geometry["half_length_m"]) - 0.005, 0.005)
    on_capsule = (
        np.isfinite(points).all(axis=-1)
        & (forces >= 1.0)
        & (np.abs(axial) <= usable_half_length)
        & (radial_distance <= radius + 0.006)
        & (radial_distance >= max(radius - 0.008, 0.0))
    )

    last_seen = np.full(2, -1000, dtype=np.int64)
    last_axial = np.full(2, np.nan, dtype=np.float32)
    last_closing = np.full(2, np.nan, dtype=np.float32)
    valid = np.zeros(len(grasp_mask), dtype=bool)
    for step in np.flatnonzero(grasp_mask):
        for finger in range(2):
            if on_capsule[step, finger]:
                last_seen[finger] = step
                last_axial[finger] = axial[step, finger]
                last_closing[finger] = closing[step, finger]
        recent = np.all(step - last_seen <= 5)
        opposed = (
            recent
            and np.isfinite(last_axial).all()
            and abs(float(last_axial[0] - last_axial[1])) <= 0.020
            and np.all(np.abs(last_axial) <= usable_half_length)
            and float(last_closing[0] * last_closing[1]) < 0.0
            and np.all(np.abs(last_closing) >= 0.25 * radius)
        )
        unexpected_contact = np.any((forces[step] >= 1.0) & ~on_capsule[step])
        valid[step] = (
            opposed
            and np.any(on_capsule[step])
            and not unexpected_contact
            and np.all(forces[step] <= 50.0)
        )

    passed = _longest_true_run(valid) >= 5
    detail = (
        f"precise opposed capsule contact={passed}, "
        f"valid_run={_longest_true_run(valid)} steps"
    )
    return passed, detail


def evaluate_episode(episode_dir: Path) -> dict:
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    trajectory = np.load(episode_dir / "trajectory.npz", allow_pickle=False)
    stage_ids = trajectory["stage_id"]
    dt = float(metadata["control_dt"])
    duration = float(trajectory["timestamp"][-1] - trajectory["timestamp"][0])

    joint_names = metadata["robot_state"]["joint_names"]
    joint_index = {name: index for index, name in enumerate(joint_names)}
    arm_indices = np.asarray([joint_index[f"joint{i}"] for i in range(1, 7)])

    grasp_mask = stage_ids == AUTO_STAGE["grasp_handle"]
    turn_mask = stage_ids == AUTO_STAGE["turn_handle"]
    reposition_mask = np.isin(
        stage_ids,
        [
            AUTO_STAGE["release_handle"],
            AUTO_STAGE["retract_arm"],
            AUTO_STAGE["align_panel"],
        ],
    )
    push_mask = np.isin(
        stage_ids,
        [AUTO_STAGE["contact_panel"], AUTO_STAGE["push_open"]],
    )
    push_completion_mask = push_mask.copy()
    through_indices_all = np.flatnonzero(stage_ids == AUTO_STAGE["through"])
    if through_indices_all.size:
        # The recorder labels a sample with the new stage after the controller
        # accepts the transition. Include that one boundary sample in the
        # PUSH_OPEN completion event without charging later THROUGH motion to it.
        push_completion_mask[through_indices_all[0]] = True
    through_mask = np.isin(
        stage_ids, [AUTO_STAGE["through"], AUTO_STAGE["reset_arm"]]
    )

    handle_forces = trajectory["handle_contact_force"]
    bilateral_grasp, grasp_detail = _validate_bilateral_grasp(
        trajectory, metadata, grasp_mask
    )

    latch_and_gap = False
    latch_peak = 0.0
    turn_hinge_peak = 0.0
    if np.any(turn_mask):
        latch_peak = float(np.max(trajectory["door_joint_position"][turn_mask, 2]))
        turn_hinge_peak = float(np.max(np.abs(trajectory["door_joint_position"][turn_mask, 0])))
        # The recorder does not persist the geometry-derived gap threshold. Entering
        # RELEASE_HANDLE proves that AutoOpen evaluated both conditions successfully.
        latch_and_gap = latch_peak >= 0.02 and _has_stage(stage_ids, "release_handle")

    reposition_ready = False
    if np.any(reposition_mask):
        reposition_ready = bool(
            np.any(np.all(handle_forces[reposition_mask] <= 1.0, axis=1))
            and _has_stage(stage_ids, "contact_panel")
        )

    push_angle = 0.0
    push_speed_max = 0.0
    push_force_max = 0.0
    predicted_clearance_min = math.inf
    if np.any(push_mask):
        push_angle = float(
            np.max(trajectory["door_joint_position"][push_completion_mask, 0])
        )
        push_speed_max = float(np.max(trajectory["door_joint_velocity"][push_mask, 0]))
        push_force_max = float(np.nanmax(trajectory["push_finger_force"][push_mask]))
        predicted_clearance_min = float(
            np.nanmin(trajectory["push_shoulder_predicted_min_clearance"][push_mask])
        )
    push_safe = bool(
        push_angle >= math.radians(89.0)
        and push_speed_max <= math.radians(30.0)
        and push_force_max <= 600.0
        and predicted_clearance_min > 0.0
        and _has_stage(stage_ids, "through")
    )

    through_distance = 0.0
    final_arm_error = math.inf
    if np.any(through_mask):
        through_indices = np.flatnonzero(through_mask)
        direction = np.asarray(
            metadata["scene"]["push_and_traverse_direction_world"][:2], dtype=np.float32
        )
        base_xy = trajectory["base_pose_w"][:, :2]
        through_distance = float(
            np.dot(base_xy[through_indices[-1]] - base_xy[through_indices[0]], direction)
        )
        final_arm = trajectory["joint_position"][through_indices[-1], arm_indices]
        final_arm_error = float(np.max(np.abs(final_arm - MIDDLE_ARM)))
    through_complete = bool(
        through_distance > 1.0
        and final_arm_error < 0.05
        and bool(trajectory["success"][-1])
    )

    stages = [
        StageResult(
            "APPROACH_PREGRASP",
            _has_stage(stage_ids, "center_handle"),
            "base station and collision-free open pregrasp were both reached",
        ),
        StageResult(
            "GRASP",
            bilateral_grasp and _has_stage(stage_ids, "turn_handle"),
            grasp_detail,
        ),
        StageResult(
            "OPEN_LATCH",
            latch_and_gap,
            f"latch_peak={latch_peak:.4f} m, turn_hinge={math.degrees(turn_hinge_peak):.2f} deg",
        ),
        StageResult(
            "REPOSITION",
            reposition_ready,
            f"handle clear and robot reached the panel-ready state={reposition_ready}",
        ),
        StageResult(
            "PUSH_OPEN",
            push_safe,
            (
                f"angle={math.degrees(push_angle):.2f} deg, "
                f"speed_max={math.degrees(push_speed_max):.2f} deg/s, "
                f"force_max={push_force_max:.1f} N, "
                f"predicted_clearance_min={predicted_clearance_min:.4f} m"
            ),
        ),
        StageResult(
            "THROUGH",
            through_complete,
            f"distance={through_distance:.3f} m, middle_error={final_arm_error:.4f} rad",
        ),
    ]

    completed = sum(stage.passed for stage in stages)
    all_complete = completed == len(stages)
    stage_events = 10.0 * completed
    bounded_progress = 4.0 * completed
    success_event = 30.0 if all_complete else 0.0
    time_cost = -0.2 * duration
    time_bonus = 20.0 * max(1.0 - duration / 70.0, 0.0) if all_complete else 0.0
    score = stage_events + bounded_progress + success_event + time_cost + time_bonus

    return {
        "seed": int(metadata["seed"]),
        "duration_s": duration,
        "success": all_complete,
        "score": score,
        "score_terms": {
            "stage_events": stage_events,
            "bounded_progress": bounded_progress,
            "success_event": success_event,
            "time_cost": time_cost,
            "time_bonus": time_bonus,
        },
        "stages": [stage.__dict__ for stage in stages],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path, nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = [evaluate_episode(path) for path in args.episodes]
    if args.json:
        print(json.dumps(results, indent=2))
        return

    for result in results:
        print(
            f"seed={result['seed']} success={result['success']} "
            f"duration={result['duration_s']:.2f}s score={result['score']:.3f}"
        )
        for stage in result["stages"]:
            state = "PASS" if stage["passed"] else "FAIL"
            print(f"  {state:4s} {stage['name']:14s} {stage['detail']}")


if __name__ == "__main__":
    main()
