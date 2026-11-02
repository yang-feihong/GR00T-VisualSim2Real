"""A transit experiment must retain endpoints and reject any uncertified path."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick import planning


@pytest.mark.parametrize("blocked", [False, True])
def test_elevated_transit_endpoints_and_full_path_admission(monkeypatch, blocked):
    root = torch.zeros(1, 13)
    root[:, 2], root[:, 3] = 0.5, 1
    state = {
        "root_states": root,
        "ee_pos": torch.tensor([[0.3, 0.0, 0.4]]),
        "ee_orn": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "dof_pos": torch.zeros(1, 6),
    }
    target = torch.tensor([[0.6, 0.0, 0.15], [0.5, 0.1, 0.2]])
    rotation = state["ee_orn"].expand(2, -1)

    def inverse(positions, quats, roots, initial, limits):
        return SimpleNamespace(
            arm_joint_positions=positions.new_zeros(positions.numel() // 3, 6),
            exact=torch.ones(positions.shape[:2], dtype=torch.bool).flatten(),
        )

    class Collision:
        arm_joint_ids = torch.arange(6)

        def clearance(self, geometry, roots, joints, **kwargs):
            gaps = roots.new_full((len(roots),), 0.01)
            if blocked:
                gaps[:] = -0.01
            return gaps

        def gripper_clearance(self, geometry, positions, quats):
            return positions.new_full(positions.shape[:2], 0.01)

    monkeypatch.setattr(planning, "inverse_bank", inverse)
    p, q, joints, roots, valid = planning.elevated_cartesian_approach(
        state,
        target,
        rotation,
        torch.tensor([0.38, 0.42]),
        torch.tensor([[-3.0, 3.0]] * 6),
        Collision(),
        None,
        31,
    )
    assert p.shape == (2, 31, 3)
    assert joints.shape == (2, 31, 6)
    torch.testing.assert_close(p[:, 0], state["ee_pos"].expand_as(target))
    torch.testing.assert_close(p[:, -1], target)
    torch.testing.assert_close(q[:, -1], rotation)
    torch.testing.assert_close(roots[:, 0, 2], torch.full((2,), 0.5))
    torch.testing.assert_close(roots[:, -1, 2], torch.tensor([0.38, 0.42]))
    assert valid.tolist() == [not blocked] * 2
