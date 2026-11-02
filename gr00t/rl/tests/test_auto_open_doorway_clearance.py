import math

import pytest
import torch

from gr00t.rl.auto_open.doorway_clearance import DoorwayProjection, THROUGH_PROJECTION_CLEARANCE_M


def install_doorway(fsm, count, device='cpu'):
    """One-meter opening with a centered 45.4 cm quadruped."""
    fsm.doorway_projection = DoorwayProjection(
        torch.tensor([[0., 0., 1., 1.]], device=device).expand(count, -1),
        torch.tensor([[-.02, .02, -.02, .02]], device=device).expand(count, -1),
        torch.full((count,), .273, device=device), torch.full((count,), .727, device=device),
    )
    fsm.through_goal_hinge = fsm.doorway_projection.required_angle()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_width_dependent_projection_gate_includes_panel_thickness(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    width = torch.tensor([.8, 1.1], device=device, dtype=torch.float64)
    cosine = width[:, None] * torch.tensor([[0., 0., 1., 1.]], device=device)
    sine = torch.tensor([[-.02, .02, -.02, .02]], device=device).expand(2, -1)
    projection = DoorwayProjection(cosine, sine, width/2-.227, width/2+.227)
    rows = torch.arange(2, device=device)
    angle = projection.required_angle()
    assert angle[0] > angle[1]
    assert torch.all(angle < math.radians(89))
    assert torch.all(projection.gap(angle, rows) >= THROUGH_PROJECTION_CLEARANCE_M)
    assert torch.all(projection.gap(angle-.001, rows) < THROUGH_PROJECTION_CLEARANCE_M)
    at_ninety = projection.gap(torch.full_like(width, math.pi/2), rows)
    torch.testing.assert_close(at_ninety, width/2-.227-.02)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_projection_is_invariant_to_hinge_side_and_world_offset(sign):
    corners = torch.tensor([[[0., 0., 0.], [0., 1., 0.], [.04, 0., 0.], [.04, 1., 0.]]])
    hinge = torch.tensor([[.02, 0., 0.]])
    tangent = torch.tensor([[0., 1., 0.]])
    center = torch.tensor([[.02, .5, 0.]])
    forward = torch.tensor([[-1., 0., 0.]])
    bounds = torch.tensor([[-.227, .227]])
    original = DoorwayProjection.from_world(corners, hinge, tangent, torch.ones(1), center, forward, bounds)
    offset = torch.tensor([[8., -8., 0.]])
    corners[..., 1] *= sign
    hinge[:, 1] *= sign
    center[:, 1] *= sign
    tangent[:, 1] *= sign
    mirrored = DoorwayProjection.from_world(corners+offset[:, None], hinge+offset, tangent,
                                            torch.full((1,), sign), center+offset, forward, bounds)
    angle = torch.tensor([math.radians(82)])
    torch.testing.assert_close(original.gap(angle, torch.tensor([0])),
                               mirrored.gap(angle, torch.tensor([0])), atol=1e-6, rtol=1e-6)


def test_angle_alone_does_not_authorize_a_wide_robot():
    from types import SimpleNamespace
    from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM

    fsm = SimpleNamespace(hinge_idx=0)
    install_doorway(fsm, 1)
    fsm.doorway_projection.robot_lower[:] = 0.08
    fsm.doorway_projection.robot_upper[:] = 0.92
    state = {"door_joint_position": torch.tensor([[math.radians(89.0)]])}
    fsm._doorway_gap = lambda state, rows: TensorDoorFSM._doorway_gap(fsm, state, rows)
    assert not TensorDoorFSM._doorway_open(fsm, state, torch.tensor([0])).item()
