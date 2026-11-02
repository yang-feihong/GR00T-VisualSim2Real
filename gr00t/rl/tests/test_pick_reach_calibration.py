"""Reach identification features must not depend on scene placement or quaternion sign."""

import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.reach_calibration import reach_features


def test_reach_features_are_yaw_translation_and_quaternion_sign_invariant():
    root = torch.tensor([[0.3, -0.2, 0.52, 1, 0, 0, 0]])
    goals = torch.tensor([[0.8, 0.1, 0.12]])
    quats = _euler_xyz_quat(torch.tensor([[0.2, 1.3, -0.4]]))
    joints = torch.tensor([[-1.57, 2.6, -1.5, 0.4, 0, 0]])
    expected, _ = reach_features(root, goals, quats, joints)
    rotation = _euler_xyz_quat(torch.tensor([[0.0, 0, 1.2]]))
    offset = torch.tensor([[3.0, -4, 0.3]])
    transformed = root.clone()
    transformed[:, :3] = _quat_rotate(rotation, root[:, :3]) + offset
    transformed[:, 3:7] = _quat_multiply(rotation, root[:, 3:7])
    actual, _ = reach_features(
        transformed,
        _quat_rotate(rotation, goals) + offset,
        -_quat_multiply(rotation, quats),
        joints,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


def test_inverse_bank_accepts_constant_and_per_point_body_references(monkeypatch):
    from gr00t.rl.pick import planning

    captures = []

    def solve(p, q, **kwargs):
        captures.append(kwargs)
        return p

    monkeypatch.setattr(planning, "solve_pose_targets", solve)
    positions = torch.zeros(2, 3, 3)
    quats = torch.tensor([1.0, 0, 0, 0]).expand(2, 3, 4)
    roots = torch.zeros(2, 13)
    roots[:, 3] = 1
    joints, limits = torch.zeros(2, 6), torch.zeros(6, 2)
    planning.inverse_bank(positions, quats, roots, joints, limits)
    repeated = roots[:, None].expand(-1, 3, -1).clone()
    planning.inverse_bank(positions, quats, repeated, joints, limits)
    for key in captures[0]:
        torch.testing.assert_close(captures[0][key], captures[1][key])
    repeated[:, :, 2] = torch.tensor([0.3, 0.4, 0.5])
    planning.inverse_bank(positions, quats, repeated, joints, limits)
    torch.testing.assert_close(
        captures[-1]["base_position_world"][:, 2], torch.tensor([0.3, 0.4, 0.5] * 2)
    )
