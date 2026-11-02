"""Sign-only closing scans must match exhaustive scans, including empty contact."""

import json

import pytest
import torch
import warp as wp

from gr00t.rl.pick.contact_profile_kernel import BLOCK_DIM, _point_gap, first_profile_contact
from gr00t.rl.pick.pocket import gripper_contact_profiles


@wp.kernel
def exhaustive_contact(
    points: wp.array2d(dtype=wp.vec3),
    centers: wp.array(dtype=wp.vec3),
    angles: wp.array(dtype=float),
    profile: wp.array3d(dtype=float),
    depths: int,
    output: wp.array(dtype=float),
):
    row, lane = wp.tid()
    orientation = row // depths
    last = int(0)
    for step in range(17):
        angle = angles[row] * (float(step) / 16.0)
        gap = float(wp.inf)
        for sample in range(lane, points.shape[1], BLOCK_DIM):
            gap = wp.min(
                gap, _point_gap(points[orientation, sample] + centers[row], angle, profile)
            )
        minimum = wp.tile_min(wp.tile(gap))[0]
        if minimum < 0.0:
            last = step
    lower = angles[row] * (float(last) / 16.0)
    upper = angles[row] * (float(wp.min(last + 1, 16)) / 16.0)
    for iteration in range(16):
        angle = (lower + upper) / 2.0
        gap = float(wp.inf)
        for sample in range(lane, points.shape[1], BLOCK_DIM):
            gap = wp.min(
                gap, _point_gap(points[orientation, sample] + centers[row], angle, profile)
            )
        minimum = wp.tile_min(wp.tile(gap))[0]
        if minimum < 0.0:
            lower = angle
        else:
            upper = angle
    if lane == 0:
        output[row] = (lower + upper) / 2.0


def exhaustive_profile_contact(points, centers, angles, profile):
    wp.init()
    result = torch.empty_like(angles)
    wp.launch_tiled(
        exhaustive_contact,
        dim=(result.numel(),),
        inputs=[
            wp.from_torch(points.flatten(0, 1).contiguous(), dtype=wp.vec3),
            wp.from_torch(centers.flatten(0, 2).contiguous(), dtype=wp.vec3),
            wp.from_torch(angles.contiguous().flatten()),
            wp.from_torch(profile[0].contiguous()),
            centers.shape[-2],
            wp.from_torch(result.flatten()),
        ],
        stream=wp.stream_from_torch(torch.cuda.current_stream()),
        block_dim=BLOCK_DIM,
    )
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contact kernel")
@pytest.mark.parametrize("count", [97, 1025, 9984])
@pytest.mark.parametrize("offset", [0.0, 0.5])
def test_exhaustive_contact_is_bitwise_equal(count, offset, tmp_path):
    wp.init()
    generator = torch.Generator(device="cuda").manual_seed(1702)
    points = torch.rand(4, 8, count, 3, device="cuda", generator=generator) * 0.07 - 0.035
    points[..., 1] += offset
    centers = torch.zeros(4, 8, 17, 3, device="cuda")
    centers[..., 0] = torch.linspace(0.015, 0.055, 17, device="cuda")
    centers[..., 2] = -0.03
    angles = torch.full(centers.shape[:-1], -torch.pi / 3, device="cuda")
    profile = gripper_contact_profiles("cuda").maps["moving"]

    def reference():
        return exhaustive_profile_contact(points, centers, angles, profile)

    def optimized():
        return first_profile_contact(points, centers, angles, profile)

    torch.testing.assert_close(optimized(), reference(), rtol=0, atol=0)
    timings = {}
    for name, function in (("exhaustive_ms", reference), ("short_circuit_ms", optimized)):
        function()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            function()
        end.record()
        end.synchronize()
        timings[name] = start.elapsed_time(end) / 5
    (tmp_path / "short_circuit.json").write_text(json.dumps(timings))
