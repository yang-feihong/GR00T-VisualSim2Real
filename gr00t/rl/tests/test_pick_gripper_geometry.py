"""Pocket kinematics must match the deployed joint frames, not a guessed pivot."""

import math

import numpy as np
import pytest
import torch

from gr00t.rl.auto_open.first_pull_wall import _asset_envelopes
from gr00t.rl.envs.door.b2z1_reset_geometry import _axis_rotation
from gr00t.rl.pick.grasp_quality import load_grasp_center
from gr00t.rl.pick.pocket import _moving_vertices, jaw_distal_limits
from gr00t.rl.utils.b2z1_asset import ROBOT_USD


def test_hull_mass_center_uses_volume_instead_of_vertex_average(tmp_path):
    import trimesh

    points = np.array([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.0, 0.08, 0.0], [0.0, 0.0, 0.10]])
    hull = trimesh.convex.convex_hull(points)
    np.save(tmp_path / "collision_support.npy", hull.vertices.astype(np.float32))
    np.save(tmp_path / "collision_faces.npy", hull.faces.astype(np.int32))
    actual = load_grasp_center(str(tmp_path / "object.usda"), "cpu")
    torch.testing.assert_close(actual, torch.tensor([0.015, 0.020, 0.025]), atol=1e-7, rtol=0)


@pytest.mark.parametrize("angle", [0.0, 0.35, math.pi / 3])
def test_moving_jaw_edge_matches_asset_joint_frames(angle):
    _, first, second, axis = _asset_envelopes(str(ROBOT_USD))
    first, second = (torch.tensor(value, dtype=torch.float32) for value in (first, second))
    rotation = _axis_rotation(torch.tensor([-angle]), axis, torch.float32)[0]
    body_rotation = first[:3, :3] @ rotation @ second[:3, :3]
    body_position = first[:3, 3] + first[:3, :3] @ rotation @ second[:3, 3]
    points = _moving_vertices("cpu", torch.float32)
    wrist_points = points @ body_rotation.T + body_position
    tcp_points = (wrist_points - torch.tensor([0.1, 0, 0])) @ torch.tensor(
        [[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]
    )
    actual = jaw_distal_limits(torch.tensor([angle]))[0, 1]
    torch.testing.assert_close(actual, tcp_points[:, 0].max(), atol=1e-6, rtol=0)
