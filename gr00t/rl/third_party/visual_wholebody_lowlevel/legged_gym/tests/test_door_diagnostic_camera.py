import math

import numpy as np

from legged_gym.utils.door_diagnostic_camera import (
    axis_angle_to_matrix,
    closed_leaf_pose,
    infer_push_side_leaf,
)


def test_push_side_is_opposite_the_opening_motion_for_mirrored_doors():
    hinge_axis = np.asarray([0.0, 0.0, 1.0])
    hinge_pivot = np.zeros(3)
    handle_pivot = np.asarray([1.0, 0.0, 0.0])

    np.testing.assert_allclose(
        infer_push_side_leaf(hinge_axis, 1.0, hinge_pivot, handle_pivot),
        [0.0, -1.0, 0.0],
    )
    np.testing.assert_allclose(
        infer_push_side_leaf(hinge_axis, -1.0, hinge_pivot, handle_pivot),
        [0.0, 1.0, 0.0],
    )


def test_closed_leaf_pose_undoes_current_hinge_rotation_about_world_pivot():
    angle = math.radians(20.0)
    current_rotation = axis_angle_to_matrix(np.asarray([0.0, 0.0, 1.0]), angle)
    current_position = np.asarray([2.0, 3.0, 0.0])
    hinge_pivot_leaf = np.asarray([-1.0, 0.0, 0.0])
    hinge_pivot_world = current_position + current_rotation @ hinge_pivot_leaf

    closed_position, closed_rotation = closed_leaf_pose(
        current_position,
        current_rotation,
        hinge_pivot_leaf,
        np.asarray([0.0, 0.0, 1.0]),
        angle,
    )

    np.testing.assert_allclose(closed_rotation, np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(
        closed_position + closed_rotation @ hinge_pivot_leaf,
        hinge_pivot_world,
        atol=1.0e-12,
    )
