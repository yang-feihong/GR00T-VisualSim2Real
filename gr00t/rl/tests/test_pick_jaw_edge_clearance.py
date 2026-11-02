"""A jaw edge may cross a table while every sampled corner and TCP ray misses."""

import pytest
import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.collision import ArmCollisionSamples


class HalfTable:
    def visible(self, starts, ends, **kwargs):
        dz = ends[..., 2] - starts[..., 2]
        fraction = -starts[..., 2] / torch.where(dz.abs() > 1e-9, dz, 1)
        x = starts[..., 0] + fraction * (ends[..., 0] - starts[..., 0])
        hit = (dz.abs() > 1e-9) & (fraction > 1e-5) & (fraction < 1 - 1e-5) & (x <= 0)
        return ~hit

    def rigid_surface_clearance(self, p, q, points, **kwargs):
        world = p[..., None, :] + _quat_rotate(q[..., None, :], points)
        distance = torch.sqrt(world[..., 2].square() + world[..., 0].clamp_min(0).square())
        visible = self.visible(p[..., None, :].expand_as(world), world)
        return distance.masked_fill(~visible, -1).amin(-1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("height,blocked", [(0.04, True), (0.12, False)])
def test_moving_jaw_edges_cannot_cross_support_plane(device, height, blocked):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    jaw = torch.cartesian_prod(
        torch.tensor([-0.05, 0.05], device=device),
        torch.tensor([-0.02, 0.02], device=device),
        torch.tensor([-0.06, 0.06], device=device),
    )
    collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
    collision._moving_envelope_corners = jaw
    collision._gripper_local_surfaces = lambda positions: (jaw + jaw.new_tensor([0, 0, 0.2]), jaw)
    p = jaw.new_tensor([[[0.04, 0, height]]])
    q = jaw.new_tensor([[[1, 0, 0, 0]]])
    geometry = HalfTable()
    assert geometry.rigid_surface_clearance(p, q, jaw).item() > 0.002
    result = collision.gripper_pose_clearance(geometry, p, q, p[..., 0] * 0 + torch.pi / 3)
    assert bool(result[0, 0] < 0.002) == blocked
