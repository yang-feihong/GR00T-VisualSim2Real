"""Shared reachability qualification for MOVE_TO_PREGRASP wrist-pitch candidates."""

from __future__ import annotations

import numpy as np

from gr00t.rl.auto_open.geometry import (
    _quat_axis_angle_np,
    _quat_mul_np,
    _quat_rotate_np,
)

PREGRASP_SAMPLE_POSITION_STEP_M = 0.004
PREGRASP_SAMPLE_ROTATION_STEP_RAD = np.deg2rad(1.0)
PRESS_HANDLE_LATCH_RELEASE_M = 0.025
PREGRASP_EXACT_POSITION_TOLERANCE_M = 1.0e-4
PREGRASP_EXACT_ORIENTATION_TOLERANCE = 1.0e-4


def pregrasp_reachability_samples(
    pregrasp_position_world,
    pregrasp_orientation_world,
    grasp_position_world,
    grasp_orientation_world,
    *,
    handle_axle_world,
    handle_axis_world,
    hinge_world,
    hinge_rotation_rad,
    handle_start_angle_rad,
    handle_target_angle_rad,
):
    """Sample insertion and three rigid rotations at 4 mm / 1 degree."""
    positions, orientations, samples = [], [], []

    def append(stage, point, quat, angle=None, branch=None):
        positions.append(np.asarray(point, dtype=np.float64))
        orientations.append(np.asarray(quat, dtype=np.float64))
        samples.append({"stage": stage, "angle_rad": angle, "press_branch_rad": branch})

    start = np.asarray(pregrasp_position_world, dtype=np.float64)
    grasp = np.asarray(grasp_position_world, dtype=np.float64)
    start_q = np.asarray(pregrasp_orientation_world, dtype=np.float64).copy()
    end_q = np.asarray(grasp_orientation_world, dtype=np.float64).copy()
    start_q /= np.linalg.norm(start_q)
    end_q /= np.linalg.norm(end_q)
    if np.dot(start_q, end_q) < 0:
        end_q = -end_q
    orientation_distance = 2 * np.arccos(np.clip(np.dot(start_q, end_q), -1, 1))
    count = max(
        1,
        int(np.ceil(np.linalg.norm(grasp - start) / PREGRASP_SAMPLE_POSITION_STEP_M)),
        int(np.ceil(orientation_distance / PREGRASP_SAMPLE_ROTATION_STEP_RAD)),
    )
    for fraction in np.linspace(0, 1, count + 1):
        # Slerp keeps the angular sampling bound when endpoint noise is present.
        if orientation_distance < 1e-8:
            quat = start_q.copy()
        else:
            half_angle = orientation_distance / 2
            quat = (
                np.sin((1 - fraction) * half_angle) * start_q
                + np.sin(fraction * half_angle) * end_q
            ) / np.sin(half_angle)
        append("insertion", start + fraction * (grasp - start), quat)

    def rotate_path(stage, point, quat, pivot, axis, angle, branch=None):
        pivot = np.asarray(pivot, dtype=np.float64)
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        radius = np.linalg.norm(np.cross(point - pivot, axis))
        count = max(
            1,
            int(np.ceil(abs(angle) / PREGRASP_SAMPLE_ROTATION_STEP_RAD)),
            int(np.ceil(abs(angle) * radius / PREGRASP_SAMPLE_POSITION_STEP_M)),
        )
        for partial in np.linspace(0, angle, count + 1):
            rotation = _quat_axis_angle_np(axis, partial)
            rotated_point = pivot + _quat_rotate_np(rotation, point - pivot)
            rotated_quat = _quat_mul_np(rotation, quat)
            append(stage, rotated_point, rotated_quat, float(partial), branch)
        return rotated_point, rotated_quat

    axle = np.asarray(handle_axle_world, dtype=np.float64)
    handle_axis = np.asarray(handle_axis_world, dtype=np.float64)
    handle_axis /= np.linalg.norm(handle_axis)
    hinge = np.asarray(hinge_world, dtype=np.float64)
    target = max(float(handle_target_angle_rad), float(handle_start_angle_rad))
    press_angle = target - handle_start_angle_rad
    rotate_path("press_handle", grasp, end_q, axle, handle_axis, press_angle)
    # The production latch mimic maps 45 degrees of handle rotation to 30 mm.
    release = np.clip(
        np.deg2rad(45.0) * PRESS_HANDLE_LATCH_RELEASE_M / 0.03,
        handle_start_angle_rad,
        target,
    )
    branches = np.unique([release, (release + target) / 2, target])
    door_rotation = _quat_axis_angle_np([0, 0, 1], hinge_rotation_rad)
    return_axle = hinge + _quat_rotate_np(door_rotation, axle - hinge)
    return_axis = _quat_rotate_np(door_rotation, handle_axis)
    for branch in branches:
        rotation = _quat_axis_angle_np(handle_axis, branch - handle_start_angle_rad)
        point = axle + _quat_rotate_np(rotation, grasp - axle)
        quat = _quat_mul_np(rotation, end_q)
        point, quat = rotate_path(
            "push_pull_door",
            point,
            quat,
            hinge,
            [0, 0, 1],
            hinge_rotation_rad,
            float(branch),
        )
        rotate_path(
            "return_handle",
            point,
            quat,
            return_axle,
            return_axis,
            -float(branch - handle_start_angle_rad),
            float(branch),
        )
    return {
        "reachability_positions": np.asarray(positions),
        "reachability_orientations": np.asarray(orientations),
        "reachability_samples": samples,
    }


def select_pregrasp_candidate_from_errors(candidates, position_errors, orientation_errors):
    """Reduce ragged batched inverse errors into the smallest valid pitch."""
    if not candidates:
        raise ValueError("MOVE_TO_PREGRASP requires at least one pitch candidate")
    if len(position_errors) != len(candidates) or len(orientation_errors) != len(candidates):
        raise ValueError("inverse error rows must match pitch candidates")

    diagnostics = []
    exact_indices = []
    for index, candidate in enumerate(candidates):
        positions = np.asarray(position_errors[index], dtype=np.float64)
        rotations = np.asarray(orientation_errors[index], dtype=np.float64)
        sample_count = len(candidate["reachability_positions"])
        if positions.shape != (sample_count,) or rotations.shape != (sample_count,):
            raise ValueError("inverse error count must match candidate samples")
        failed = np.flatnonzero(
            ~np.isfinite(positions)
            | ~np.isfinite(rotations)
            | (positions > PREGRASP_EXACT_POSITION_TOLERANCE_M)
            | (rotations > PREGRASP_EXACT_ORIENTATION_TOLERANCE)
        )
        first_failure = None
        if len(failed):
            failed_index = int(failed[0])
            first_failure = {
                **candidate["reachability_samples"][failed_index],
                "sample_index": failed_index,
                "position_error_m": float(positions[failed_index]),
                "orientation_error": float(rotations[failed_index]),
            }
        else:
            exact_indices.append(index)
        diagnostics.append(
            {
                "pitch_deg": float(candidate["pitch_deg"]),
                "exact": not len(failed),
                "sample_count": sample_count,
                "max_position_error_m": float(np.nan_to_num(positions, nan=np.inf).max()),
                "max_orientation_error": float(np.nan_to_num(rotations, nan=np.inf).max()),
                "first_failure": first_failure,
            }
        )

    if exact_indices:
        selected_index = exact_indices[0]
        exact = True
    else:
        selected_index = min(
            range(len(candidates)),
            key=lambda index: (
                diagnostics[index]["max_position_error_m"]
                + diagnostics[index]["max_orientation_error"]
            ),
        )
        exact = False
    return candidates[selected_index], exact, diagnostics
