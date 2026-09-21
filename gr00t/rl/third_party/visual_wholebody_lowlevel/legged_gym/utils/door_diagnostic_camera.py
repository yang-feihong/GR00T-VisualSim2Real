from __future__ import annotations

import math

import numpy as np


def axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-9:
        raise ValueError("Rotation axis must be non-zero")
    x, y, z = axis / norm
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle_rad) * skew
        + (1.0 - math.cos(angle_rad)) * (skew @ skew)
    )


def infer_push_side_leaf(
    hinge_axis_leaf: np.ndarray,
    hinge_open_direction: float,
    hinge_pivot_leaf: np.ndarray,
    handle_pivot_leaf: np.ndarray,
) -> np.ndarray:
    opening_axis = float(hinge_open_direction) * np.asarray(
        hinge_axis_leaf, dtype=np.float64
    )
    hinge_to_handle = np.asarray(handle_pivot_leaf, dtype=np.float64) - np.asarray(
        hinge_pivot_leaf, dtype=np.float64
    )
    opening_motion = np.cross(opening_axis, hinge_to_handle)
    norm = float(np.linalg.norm(opening_motion))
    if norm <= 1.0e-9:
        raise ValueError("The handle pivot lies on the door hinge axis")
    return -opening_motion / norm


def closed_leaf_pose(
    leaf_position_world: np.ndarray,
    leaf_rotation_world: np.ndarray,
    hinge_pivot_leaf: np.ndarray,
    hinge_axis_leaf: np.ndarray,
    hinge_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    leaf_position_world = np.asarray(leaf_position_world, dtype=np.float64)
    leaf_rotation_world = np.asarray(leaf_rotation_world, dtype=np.float64)
    hinge_pivot_world = leaf_position_world + leaf_rotation_world @ np.asarray(
        hinge_pivot_leaf, dtype=np.float64
    )
    hinge_axis_world = leaf_rotation_world @ np.asarray(
        hinge_axis_leaf, dtype=np.float64
    )
    undo_hinge_rotation = axis_angle_to_matrix(
        hinge_axis_world, -float(hinge_angle_rad)
    )
    closed_position_world = hinge_pivot_world + undo_hinge_rotation @ (
        leaf_position_world - hinge_pivot_world
    )
    closed_rotation_world = undo_hinge_rotation @ leaf_rotation_world
    return closed_position_world, closed_rotation_world
