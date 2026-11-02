"""Ring proposals cover an object without encoding scene/seed-specific poses."""

import math

import pytest
import torch

from gr00t.rl.pick.station import station_pose_bank


def test_ring_radii_heading_and_full_azimuth():
    root = torch.zeros(1, 13)
    root[:, 2], root[:, 3] = 0.53, 1
    target = torch.tensor([[1.0, 2.0, 0.03]])
    destination = torch.tensor([[0.0, 2.0, 0.1]])
    poses, roots = station_pose_bank(
        root,
        destination,
        target,
        ring_radii_m=(0.55, 0.65, 0.75),
        ring_samples=24,
        yaw_offsets_deg=(0,),
    )
    assert poses.shape == (1, 72, 3)
    torch.testing.assert_close(
        (poses[0, :, :2] - target[0, :2]).norm(dim=-1).reshape(3, 24),
        torch.tensor([0.55, 0.65, 0.75])[:, None].expand(-1, 24),
    )
    to_target = target[:, :2] - poses[..., :2]
    desired = torch.atan2(to_target[..., 1], to_target[..., 0]) + 0.1
    error = poses[..., 2] - desired
    torch.testing.assert_close(error.sin(), torch.zeros_like(error), atol=1e-6, rtol=0)
    assert roots.shape[:2] == (1, 72)
    assert (poses[0, :24, :2].amax(0) - poses[0, :24, :2].amin(0) > 1).all()


@pytest.mark.parametrize("radii", [(), (0,), (-1,), (math.nan,)])
def test_invalid_ring_rejected(radii):
    with pytest.raises(ValueError, match="radii"):
        station_pose_bank(
            torch.zeros(1, 13), torch.zeros(1, 3), torch.ones(1, 3), ring_radii_m=radii
        )
