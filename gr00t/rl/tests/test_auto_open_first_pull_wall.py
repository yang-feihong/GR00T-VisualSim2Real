import pytest
import torch

from gr00t.rl.auto_open.first_pull_entry import plan_first_pull_entry
from gr00t.rl.auto_open.first_pull_wall import FirstPullArmWallGPU, retime_first_pull_paths
from gr00t.rl.auto_open.panel_motion import PullPanelConfig
from gr00t.rl.tests.test_auto_open_pull_entry_gpu import fixture


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_first_approach_is_continuous_and_keeps_original_endpoint(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    cache, state, _ = fixture(1., device)
    state['door_joint_position'].fill_(.1)
    state['root_states'][:, :2] = torch.tensor([[.3, .5]], device=device)
    samples = dict(height_m=torch.tensor([[1.]], device=device), insertion_m=torch.tensor([[.04]], device=device))
    kwargs = dict(frame_edge_m=torch.tensor([.5], device=device), wall_front_m=torch.tensor([.01], device=device),
        hardware_height_bounds=torch.tensor([[.75, .85]], device=device), config=PullPanelConfig())
    old = plan_first_pull_entry(cache, state, torch.tensor([0], device=device), samples, **kwargs)
    new = plan_first_pull_entry(cache, state, torch.tensor([0], device=device), samples, include_approach=True, **kwargs)
    torch.testing.assert_close(new['positions'][0, 0, 0], state['ee_pos'][0])
    torch.testing.assert_close(new['orientations'][0, 0, 0], state['ee_orn'][0])
    torch.testing.assert_close(new['endpoint'], old['endpoint'])
    length = int(new['lengths'][0, 0])
    torch.testing.assert_close(new['positions'][0, 0, length-1], new['endpoint'][0, 0])
    delta = new['positions'][0, 0, 1:length]-new['positions'][0, 0, :length-1]
    assert delta.norm(dim=-1).max() <= .002001


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_retiming_preserves_endpoints_and_handles_padding(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    p = torch.zeros(2, 32, 3, device=device)
    p[:, :3, 0] = torch.tensor([0., .002, .004], device=device)
    q = torch.zeros(2, 32, 4, device=device); q[..., 0] = 1.
    j = torch.zeros(2, 32, 6, device=device)
    j[:, 1, 0] = .025; j[:, 2, 0] = .05
    j[:, 3:] = 999.
    n = torch.tensor([3, 3], device=device)
    out, orn, lengths, overflow = retime_first_pull_paths(p, q, j, n)
    assert lengths.tolist() == [7, 7]
    assert not overflow.any()
    torch.testing.assert_close(out[:, 0], p[:, 0])
    torch.testing.assert_close(out[:, 6], p[:, 2])
    assert (out[:, 1:7]-out[:, :6]).norm(dim=-1).max() <= .002
    torch.testing.assert_close(orn[..., 0], torch.ones_like(orn[..., 0]))
    single = retime_first_pull_paths(p[1:], q[1:], j[1:], n[1:])
    for a, b in zip((out, orn, lengths, overflow), single):
        torch.testing.assert_close(a[1:], b, rtol=0., atol=0.)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_arm_envelope_wall_gate_and_batch_independence(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    checker = FirstPullArmWallGPU(device=device)
    root = torch.tensor([[2., 5., .5, 1., 0., 0., 0.], [-1., 5., .5, 1., 0., 0., 0.]], device=device)
    arm = torch.zeros(2, 6, device=device)
    gripper = torch.zeros(2, device=device)
    hinge = torch.zeros(2, 3, device=device)
    normal = torch.tensor([[1., 0., 0.]], device=device).repeat(2, 1)
    tangent = torch.tensor([[0., 1., 0.]], device=device).repeat(2, 1)
    edge, wall = torch.ones(2, device=device), torch.full((2,), .03, device=device)
    values = (root, arm, gripper, hinge, normal, tangent, edge, wall)
    gap = checker.clearance(*values)
    assert gap[0] > .1 and gap[1] < 0.
    single = checker.clearance(*(v[1:] for v in values))
    torch.testing.assert_close(gap[1:], single, rtol=0., atol=1.e-12)
