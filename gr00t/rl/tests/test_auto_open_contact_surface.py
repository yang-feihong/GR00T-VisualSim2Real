import torch

from gr00t.rl.auto_open.contact_surface import separating_rear_contact
from gr00t.rl.auto_open.geometry import ROBOT_COLLIDER_SPECS


def test_contact_geometry_includes_backstop_while_grasp_geometry_stays_distal():
    scopes = {key: tokens for _, key, tokens in ROBOT_COLLIDER_SPECS}
    for path, contact_key, grasp_key in (
        ("/world/robot/link06/collisions/fixed_convex_001", "fixed_contact_surfaces", "fixed_finger"),
        ("/world/robot/grippermover/collisions/moving_convex_015", "moving_contact_surfaces", "moving_finger"),
    ):
        assert any(token in path for token in scopes[contact_key])
        assert not any(token in path for token in scopes[grasp_key])


def _surfaces():
    return torch.tensor([
        [[1., -1, -1], [1, 1, -1], [1, 1, 1]],
        [[1., -1, -1], [1, 1, 1], [1, -1, 1]],
        [[-1., -1, 1], [1, -1, 1], [1, 1, 1]],
        [[-1., -1, 1], [1, 1, 1], [-1, 1, 1]],
        [[-1., -1, -1], [1, 1, -1], [1, -1, -1]],
        [[-1., -1, -1], [-1, 1, -1], [1, 1, -1]],
        [[-1., -1, -1], [-1, -1, 1], [-1, 1, 1]],
        [[-1., -1, -1], [-1, 1, 1], [-1, 1, -1]],
    ])


def test_only_separating_rear_surface_is_allowed():
    points = torch.tensor([
        [1., 0, 0], [0., 0, 1], [0., 0, -1], [-1., 0, 0],
        [1., 0, 1], [float("nan"), 0, 0],
    ])
    pose = torch.tensor([[0., 0, 0, 1, 0, 0, 0]]).expand(len(points), -1)
    shaft = torch.tensor([[1., 0, 0]]).expand(len(points), -1)
    retreat = -shaft
    allowed = separating_rear_contact(points, pose, _surfaces(), shaft, retreat)
    assert allowed.tolist() == [True, False, False, False, False, False]
    assert not separating_rear_contact(points[:1], pose[:1], _surfaces(), shaft[:1], shaft[:1]).item()


def test_world_rotation_does_not_change_surface_classification():
    # Rotate local +X to world +Y and translate the complete configuration.
    points = torch.tensor([[.1, 1.2, .3], [.1, .2, 1.3]])
    root_half = 2 ** -.5
    poses = torch.tensor([[.1, .2, .3, root_half, 0, 0, root_half]]).expand(2, -1)
    shaft = torch.tensor([[0., 1, 0]]).expand(2, -1)
    assert separating_rear_contact(points, poses, _surfaces(), shaft, -shaft).tolist() == [True, False]


def test_unknown_collision_surface_is_not_allowed():
    point = torch.zeros(1, 3)
    pose = torch.tensor([[0., 0, 0, 1, 0, 0, 0]])
    axis = torch.tensor([[1., 0, 0]])
    assert not separating_rear_contact(point, pose, torch.empty(0, 3, 3), axis, -axis).item()
