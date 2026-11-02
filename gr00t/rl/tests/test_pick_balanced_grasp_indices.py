"""Indexed joint proposals must match the existing full hinged-jaw bank exactly."""

import math

import pytest
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.pick.planning import balanced_grasp_orientations, grasp_aperture_angle


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_indexed_balanced_bank_matches_original_full_formula(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU orientation bank")
    heading = torch.tensor([-2.1, 0, 1.7], device=device, dtype=dtype)
    sizes = heading.new_tensor([[0.03, 0.05, 0.04], [0.1, 0.06, 0.12], [0.02, 0.02, 0.05]])
    offsets = heading.new_tensor([0, -5, 5, -10, 10]) * (math.pi / 180)
    yaw = heading[:, None, None] + heading.new_tensor([0, -45, 45, -90, 90, -135, 135, 180])[
        None, None
    ] * (math.pi / 180)
    yaw = yaw.expand(-1, len(offsets), -1)
    aperture = grasp_aperture_angle(sizes)[:, None, None]
    first = _euler_xyz_quat(
        torch.stack(
            (
                torch.full_like(yaw, math.pi),
                math.pi / 2 - aperture / 2 + offsets[None, :, None].expand_as(yaw),
                yaw,
            ),
            -1,
        )
    )
    second = _euler_xyz_quat(
        torch.stack(
            (
                torch.zeros_like(yaw),
                math.pi / 2 + aperture / 2 + offsets[None, :, None].expand_as(yaw),
                yaw,
            ),
            -1,
        )
    )
    expected = torch.stack((first, second), 2).flatten(1, 3)
    actual = balanced_grasp_orientations(heading, sizes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    indices = torch.tensor([[0, 19, 79], [51, 12, 1], [24, 33, 62]], device=device)
    indexed = balanced_grasp_orientations(heading, sizes, candidate_indices=indices)
    torch.testing.assert_close(
        indexed, expected.gather(1, indices[..., None].expand(-1, -1, 4)), rtol=0, atol=0
    )
