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


def panel_normal_toward_reference(
    panel_normal_world: np.ndarray,
    panel_point_world: np.ndarray,
    reference_position_world: np.ndarray,
) -> np.ndarray:
    normal = np.asarray(panel_normal_world, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-9:
        raise ValueError("The panel normal must be non-zero")
    normal /= norm
    side = float(
        np.dot(
            np.asarray(reference_position_world, dtype=np.float64)
            - np.asarray(panel_point_world, dtype=np.float64),
            normal,
        )
    )
    if abs(side) <= 1.0e-9:
        raise ValueError("The reference position lies in the door panel plane")
    return normal if side > 0.0 else -normal


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
