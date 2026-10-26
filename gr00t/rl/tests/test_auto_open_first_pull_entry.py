"""First insertion uses analytic dimensions, never collider point clouds."""

import math

import pytest
import torch

from gr00t.rl.auto_open.first_pull_entry import plan_first_pull_entry, insertion_endpoint_radius
from gr00t.rl.auto_open.panel_motion import PullPanelConfig
from gr00t.rl.auto_open.pull_entry_gpu import quat_rotate
from gr00t.rl.tests.test_auto_open_pull_entry_gpu import fixture


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_endpoint_limit_uses_goal_centered_xy_not_height_or_world_origin(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    roots = torch.zeros(2, 13, dtype=torch.float64, device=device)
    roots[:, :3] = torch.tensor([[10., -4., .5], [-7., 8., .6]], device=device)
    roots[:, 3] = 1.
    reference = torch.tensor([[1., 0., 0., 0.], [math.sqrt(.5), 0., 0., math.sqrt(.5)]],
                             dtype=torch.float64, device=device)
    local = torch.tensor([[[.6, 0., 2.], [.60001, 0., -.3], [0., .6, 1.]]],
                         dtype=torch.float64, device=device).expand(2, -1, -1)
    offset = roots.new_tensor([[.2, 0., .7]]).expand(2, -1)
    endpoint = roots[:, None, :3] + quat_rotate(reference[:, None], local + offset[:, None])
    radius = insertion_endpoint_radius(endpoint, roots, (roots[:, :3], reference, offset))
    torch.testing.assert_close(radius, local[..., :2].norm(dim=-1))
    single = insertion_endpoint_radius(endpoint[1:], roots[1:],
        (roots[1:, :3], reference[1:], offset[1:]))
    torch.testing.assert_close(single, radius[1:], rtol=0., atol=0.)
    mixed_reference = tuple(value.float() for value in (roots[:, :3], reference, offset))
    mixed = insertion_endpoint_radius(endpoint, roots, mixed_reference)
    explicit = insertion_endpoint_radius(endpoint, roots, tuple(value.double() for value in mixed_reference))
    assert mixed.dtype == torch.float64
    torch.testing.assert_close(mixed, explicit, rtol=0., atol=0.)
    assert (radius > PullPanelConfig().insertion_max_horizontal_radius_m + 1.e-9).tolist() == [[False, True, False]] * 2


@pytest.mark.parametrize('side', [-1., 1.])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_radial_first_entry_without_collider_points(side, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    cache, state, _ = fixture(side, device)
    cache.clouds = cache.parts = None
    state['door_joint_position'].fill_(.1)
    state['root_states'][:, :2] = torch.tensor([[.3, side * .5]], device=device)
    heights, insertion = torch.meshgrid(
        torch.tensor([.8, .85, .9, .95, 1., 1.05, 1.1, 1.15, 1.2], device=device),
        torch.tensor([.02, .04, .06], device=device), indexing='ij')
    samples = dict(height_m=heights.reshape(1, -1), insertion_m=insertion.reshape(1, -1))
    result = plan_first_pull_entry(cache, state, torch.tensor([0], device=device), samples,
        frame_edge_m=torch.tensor([.5], device=device), wall_front_m=torch.tensor([.01], device=device),
        hardware_height_bounds=torch.tensor([[.75, .85]], device=device), config=PullPanelConfig())
    assert result['positions'].shape == (1, 27, 1024, 3)
    assert not result['feasible'][0, :9].any()
    assert result['feasible'][0, 12:].all()
    assert (result['endpoint_horizontal_radius_m'] <= .6).all()
    rejected = plan_first_pull_entry(cache, state, torch.tensor([0], device=device), samples,
        frame_edge_m=torch.tensor([.5], device=device), wall_front_m=torch.tensor([.01], device=device),
        hardware_height_bounds=torch.tensor([[.75, .85]], device=device),
        config=PullPanelConfig(insertion_max_horizontal_radius_m=.1))
    assert not rejected['endpoint_radius_safe'].any()
    assert not rejected['feasible'].any()
    tip = result['endpoint'] + quat_rotate(result['orientation'], result['face_offset_local'])
    torch.testing.assert_close(torch.linalg.vector_norm(tip[..., :2], dim=-1),
                               .5 - samples['insertion_m'])
    for i in range(27):
        length = int(result['lengths'][0, i])
        path = result['positions'][0, i, :length]
        torch.testing.assert_close(torch.linalg.vector_norm(path[0, :2]),
            path.new_tensor(.5 + .03 + .101))
        torch.testing.assert_close(path[:, 2], samples['height_m'][0, i].expand(length))
        torch.testing.assert_close(result['orientations'][0, i, :length],
            result['orientation'][0, i].expand(length, -1))
        torch.testing.assert_close(path[-1], result['endpoint'][0, i])
        assert torch.linalg.vector_norm(path[1:] - path[:-1], dim=-1).max() <= .002001
        # Once alignment is reached, only the fixed inward radial line is used.
        radial = tip[0, i, :2] / torch.linalg.vector_norm(tip[0, i, :2])
        last = path[-10:, :2]
        assert (last[:, 0] * radial[1] - last[:, 1] * radial[0]).abs().max() < 1.e-6
