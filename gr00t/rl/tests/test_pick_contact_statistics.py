"""Fused profile reductions match the unexpanded grid-sampling definition."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.rl.pick.contact_profile_kernel import fixed_profile_centers, profile_contact_statistics
from gr00t.rl.pick.pocket import ContactProfiles


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contact geometry")
@pytest.mark.parametrize("name", ["014_lemon", "017_orange"])
def test_real_asset_quality_matches_dense_torch(name, tmp_path):
    from gr00t.rl.pick.grasp_quality import convex_grasp_quality

    class DenseReference(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(
                isinstance(target, ast.Name) and target.id == "fused" for target in node.targets
            ):
                node.value = ast.Constant(value=False)
            return self.generic_visit(node)

    tree = DenseReference().visit(ast.parse(inspect.getsource(convex_grasp_quality)))
    scope = dict(convex_grasp_quality.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "dense_profile_reference", "exec"), scope)
    path = Path(__file__).parents[1] / "data/pick_assets" / name / "collision_support.npy"
    if not path.is_file():
        pytest.skip("Real asset unavailable")
    hull = torch.as_tensor(np.load(path), device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(923)
    orientations = torch.randn(1, 48, 4, device="cuda", generator=generator)
    orientations /= orientations.norm(dim=-1, keepdim=True)
    object_q = orientations.new_tensor([[1, 0, 0, 0]])
    expected = scope["convex_grasp_quality"](
        orientations, object_q, hull, return_depth_candidates=True
    )
    actual = convex_grasp_quality(orientations, object_q, hull, return_depth_candidates=True)
    report = {}
    for key, value in actual.items():
        if value.dtype == torch.bool:
            report[key] = {"mismatches": int((value != expected[key]).sum())}
        else:
            error = (value - expected[key]).abs()
            finite = torch.isfinite(error)
            report[key] = {
                "maximum_absolute_error": float(error.masked_fill(~finite, 0).max()),
                "finite_mismatches": int(
                    (torch.isfinite(value) != torch.isfinite(expected[key])).sum()
                ),
            }
    (tmp_path / "dense_comparison.json").write_text(__import__("json").dumps(report, indent=2))
    print(report)
    for threshold in (0.0, 0.15):
        torch.testing.assert_close(
            actual["pocket_fit"] & (actual["antipodal_margin"] >= threshold),
            expected["pocket_fit"] & (expected["antipodal_margin"] >= threshold),
            atol=0,
            rtol=0,
        )
    for key in actual:
        # A one-ULP center shift can move a hull sample across the existing
        # 0.3 mm contact-patch boundary. Commands and acceptance stay unchanged.
        tolerance = (
            0.0003
            if key == "moving_contact_local"
            else (0.001 if key == "antipodal_margin" else 2e-5)
        )
        torch.testing.assert_close(
            actual[key], expected[key], atol=tolerance, rtol=1e-4, equal_nan=True, msg=key
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contact geometry")
@pytest.mark.parametrize("masked", [False, True])
def test_fused_contacts_match_dense_reference(masked):
    generator = torch.Generator(device="cuda").manual_seed(914)
    local = torch.rand((2, 3, 257, 3), generator=generator, device="cuda") * 0.08 - 0.04
    centers = local.new_zeros(2, 3, 5, 3)
    centers[..., 0] = torch.linspace(0, 0.04, 5, device="cuda")
    profiles = ContactProfiles.__new__(ContactProfiles)
    profiles.maps = {}
    for name in ("fixed", "moving"):
        source = torch.rand((1, 5, 201, 441), generator=generator, device="cuda")
        source[:, 0] *= 0.005
        source[:, 4] = 1
        source[:, 4, :10] = 0
        profiles.maps[name] = source
    mask = torch.ones((2, 257), dtype=torch.bool, device="cuda")
    if masked:
        mask[1, 191:] = False
        local[1, :, 191:] = local[1, :, :1]
    points = local[:, :, None] + centers[..., None, :]
    height, _, valid = profiles.sample("fixed", points[..., :2])
    valid &= mask[:, None, None]
    required = (height - points[..., 2]).masked_fill(~valid, -torch.inf).amax(-1)
    expected_centers = centers.clone()
    expected_centers[..., 2] = required
    actual_centers, has_fixed = fixed_profile_centers(local, centers, profiles.maps["fixed"])
    torch.testing.assert_close(actual_centers, expected_centers, atol=2e-6, rtol=1e-4)
    assert has_fixed.all()
    angles = local.new_full(centers.shape[:-1], -0.4)
    actual, actual_normals, actual_gaps = profile_contact_statistics(
        local, actual_centers, angles, profiles.maps["fixed"], profiles.maps["moving"], mask
    )
    rows = torch.tensor([1, 0, 1], device="cuda")
    permuted = profile_contact_statistics(
        local[rows],
        actual_centers[rows],
        angles[rows],
        profiles.maps["fixed"],
        profiles.maps["moving"],
        mask[rows],
    )
    for value, reference in zip(permuted, (actual, actual_normals, actual_gaps), strict=True):
        torch.testing.assert_close(value, reference[rows], atol=0, rtol=0)
    points = local[:, :, None] + actual_centers[..., None, :]
    expected, expected_normals, expected_gaps = [], [], []
    for name in ("fixed", "moving"):
        angle = angles if name == "moving" else angles * 0
        c, s = angle.cos()[..., None], angle.sin()[..., None]
        x = points[..., 0] * c + points[..., 2] * s
        z = -points[..., 0] * s + points[..., 2] * c
        height, normal, valid = profiles.sample(name, torch.stack((x, points[..., 1]), -1))
        gap = (height - z) if name == "moving" else (z - height)
        gap = gap.masked_fill(~valid | ~mask[:, None, None], torch.inf)
        minimum = gap.amin(-1)
        patch = torch.isfinite(gap) & (gap <= minimum[..., None] + 0.0003)
        nx, ny, nz = normal.unbind(-1)
        normal = torch.stack((c * nx - s * nz, ny, s * nx + c * nz), -1)
        total = (normal * patch[..., None]).sum(-2, dtype=torch.float64)
        total /= total.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        contact = (points * patch[..., None]).sum(-2, dtype=torch.float64)
        contact /= patch.sum(-1, keepdim=True).clamp_min(1)
        expected.append(contact.float())
        expected_normals.append(total.float())
        expected_gaps.append(minimum)
    torch.testing.assert_close(actual_gaps, torch.stack(expected_gaps, -1), atol=2e-6, rtol=1e-4)
    torch.testing.assert_close(actual, torch.stack(expected, -2), atol=2e-6, rtol=1e-4)
    torch.testing.assert_close(
        actual_normals, torch.stack(expected_normals, -2), atol=2e-4, rtol=1e-4
    )
