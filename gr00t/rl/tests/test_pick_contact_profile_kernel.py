"""Fused angular scans must retain the exact inner-profile sampling semantics."""

import pytest
import torch
import torch.nn.functional as F

from gr00t.rl.pick.contact_profile_kernel import first_profile_contact, moving_profile_minimum
from gr00t.rl.pick.grasp_quality import first_closing_contact


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA bounded contact geometry")
@pytest.mark.parametrize("masked", [False, True])
def test_chunked_contact_preserves_global_extrema_and_patch_statistics(monkeypatch, masked):
    from gr00t.rl.pick import contact_profile_kernel as kernel
    from gr00t.rl.pick.pocket import gripper_contact_profiles

    g = torch.Generator(device="cuda").manual_seed(731)
    local = torch.rand(2, 3, 257, 3, device="cuda", generator=g) * 0.08 - 0.04
    centers = torch.rand(2, 3, 4, 3, device="cuda", generator=g) * 0.02
    angles = torch.full(centers.shape[:-1], torch.pi / 3, device="cuda")
    profiles = gripper_contact_profiles("cuda")
    fixed, moving = profiles.maps["fixed"], profiles.maps["moving"]
    mask = None
    if masked:
        mask = torch.arange(257, device="cuda")[None].expand(2, -1) % 3 != 0

    def evaluate():
        placed, valid = kernel.fixed_profile_centers(local, centers, fixed)
        closed = kernel.first_profile_contact(local, placed, angles, moving)
        gap = kernel.moving_profile_minimum(local, placed, closed, moving)
        contacts, normals, minima = kernel.profile_contact_statistics(
            local, placed, closed, fixed, moving, mask
        )
        return placed, valid, closed, gap, contacts, normals, minima

    expected = evaluate()
    monkeypatch.setattr(kernel, "MAX_CONTACT_POINTS", 97)
    for actual, reference in zip(evaluate(), expected):
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused geometry")
@pytest.mark.parametrize("sweep", [False, True])
@pytest.mark.parametrize("point_count", [97, 257])
def test_fused_minimum_matches_grid_sample_over_all_points_and_angles(sweep, point_count):
    g = torch.Generator(device="cuda:0").manual_seed(719)
    local = torch.rand(2, 3, point_count, 3, device="cuda:0", generator=g) * 0.25 - 0.125
    center = torch.rand(2, 3, 4, 3, device="cuda:0", generator=g) * 0.02
    angles = torch.rand(2, 3, 4, device="cuda:0", generator=g)
    if sweep:
        angles = angles[..., None] * torch.linspace(0, 1, 17, device="cuda:0")
    source = torch.rand(1, 5, 201, 441, device="cuda:0", generator=g)
    source[:, 0] *= 0.03
    source[:, 4] = 1
    source[:, 4, :15] = 0
    p = local[:, :, None] + center[..., None, :]
    if sweep:
        p = p[..., None, :, :]
    c, s = angles.cos()[..., None], angles.sin()[..., None]
    x, z = p[..., 0] * c + p[..., 2] * s, -p[..., 0] * s + p[..., 2] * c
    grid = torch.stack(((x - 0.01) / 0.11, p[..., 1].expand_as(x) / 0.05), -1)
    sampled = F.grid_sample(
        source[:, [0, 4]],
        grid.reshape(1, 1, -1, 2),
        align_corners=True,
        padding_mode="zeros",
    )[0, :, 0].T.reshape(*x.shape, 2)
    expected = (sampled[..., 0] - z).masked_fill(sampled[..., 1] <= 0.99, torch.inf).amin(-1)
    actual = moving_profile_minimum(local, center, angles, source)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused geometry")
@pytest.mark.parametrize("point_count", [97, 257, 1025])
def test_fused_contact_matches_scan_and_bisection(point_count):
    g = torch.Generator(device="cuda:0").manual_seed(719)
    local = torch.rand(2, 3, point_count, 3, device="cuda:0", generator=g) * 0.1 - 0.05
    center = torch.rand(2, 3, 17, 3, device="cuda:0", generator=g) * 0.02
    angles = torch.full(center.shape[:-1], -torch.pi / 3, device="cuda:0")
    source = torch.rand(1, 5, 201, 441, device="cuda:0", generator=g)
    source[:, 0] *= 0.03
    source[:, 4] = 1
    source[:, 4, :15] = 0
    expected = first_closing_contact(
        angles, lambda angle: moving_profile_minimum(local, center, angle, source)
    )
    actual = first_profile_contact(local, center, angles, source)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
    for row in range(2):
        single = first_profile_contact(
            local[row : row + 1], center[row : row + 1], angles[row : row + 1], source
        )
        torch.testing.assert_close(single[0], actual[row], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused geometry")
def test_contact_kernel_reports_same_work_timing():
    local = torch.rand(8, 24, 257, 3, device="cuda") * 0.1 - 0.05
    centers = torch.rand(8, 24, 17, 3, device="cuda") * 0.02
    angles = torch.full(centers.shape[:-1], -torch.pi / 3, device="cuda")
    source = torch.ones(1, 5, 201, 441, device="cuda")
    source[:, 0] = 0.01

    def previous():
        return first_closing_contact(
            angles, lambda angle: moving_profile_minimum(local, centers, angle, source)
        )

    def fused():
        return first_profile_contact(local, centers, angles, source)

    torch.testing.assert_close(fused(), previous(), rtol=0, atol=2e-6)
    timings = {}
    for name, function in (("previous_ms", previous), ("fused_ms", fused)):
        function()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            function()
        end.record()
        end.synchronize()
        timings[name] = start.elapsed_time(end) / 5
    print(timings)
