"""Ragged hulls preserve each object's full contact geometry when batched."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.rl.pick.grasp_quality import convex_grasp_quality


@pytest.mark.parametrize("device", ["cuda:0"])
def test_padded_hulls_match_independent_full_hull_results(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("GPU contact geometry")
    hulls = [
        torch.tensor(
            [[x, y, z] for x in (-0.015, 0.015) for y in (-0.02, 0.02) for z in (-0.025, 0.025)],
            device=device,
        ),
        torch.tensor(
            [[x, y, z] for x in (-0.02, 0, 0.02) for y in (-0.015, 0.015) for z in (-0.02, 0.02)],
            device=device,
        ),
    ]
    samples = torch.full((2, 12, 3), 100.0, device=device)
    mask = torch.zeros(2, 12, device=device, dtype=torch.bool)
    for i, hull in enumerate(hulls):
        samples[i, : len(hull)] = hull
        mask[i, : len(hull)] = True
    object_q = samples.new_tensor([[1, 0, 0, 0], [1, 0, 0, 0]])
    q = samples.new_tensor([[[1, 0, 0, 0], [0.70710678, 0, 0.70710678, 0]]]).expand(2, -1, -1)
    batch = convex_grasp_quality(q, object_q, samples, support_mask=mask)
    for i, hull in enumerate(hulls):
        single = convex_grasp_quality(q[i : i + 1], object_q[i : i + 1], hull)
        for name, expected in single.items():
            torch.testing.assert_close(batch[name][i : i + 1], expected, equal_nan=True)


def test_real_asset_contact_quality_is_invariant_to_other_hull_padding():
    if not torch.cuda.is_available():
        pytest.skip("GPU contact geometry")
    root = Path(__file__).parents[1] / "data/pick_assets"
    paths = [
        root / name / "collision_support.npy" for name in ("014_lemon", "gso_Android_Figure_Chrome")
    ]
    if not all(path.is_file() for path in paths):
        pytest.skip("downloaded real-asset hulls unavailable")
    hulls = [torch.as_tensor(np.load(path), device="cuda", dtype=torch.float32) for path in paths]
    width = max(map(len, hulls)) + 137
    samples = torch.zeros(2, width, 3, device="cuda")
    mask = torch.zeros(2, width, device="cuda", dtype=torch.bool)
    for index, hull in enumerate(hulls):
        samples[index, : len(hull)] = hull
        mask[index, : len(hull)] = True
    generator = torch.Generator(device="cuda").manual_seed(831)
    q = torch.randn(2, 96, 4, device="cuda", generator=generator)
    q /= q.norm(dim=-1, keepdim=True)
    object_q = samples.new_tensor([[1, 0, 0, 0], [1, 0, 0, 0]])
    batch = convex_grasp_quality(q, object_q, samples, support_mask=mask)
    single = convex_grasp_quality(q[:1], object_q[:1], hulls[0])
    for name, expected in single.items():
        torch.testing.assert_close(
            batch[name][:1], expected, rtol=0, atol=0, equal_nan=True, msg=name
        )


def test_real_asset_accurate_contact_reductions_match_float32_reference_and_report_cost():
    if not torch.cuda.is_available():
        pytest.skip("GPU contact geometry")
    path = Path(__file__).parents[1] / "data/pick_assets/014_lemon/collision_support.npy"
    if not path.is_file():
        pytest.skip("downloaded real-asset hull unavailable")

    class Float32Reductions(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "sum":
                node.keywords = [keyword for keyword in node.keywords if keyword.arg != "dtype"]
            return node

    tree = Float32Reductions().visit(ast.parse(inspect.getsource(convex_grasp_quality)))
    scope = dict(convex_grasp_quality.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "float32_contact_reference", "exec"), scope)
    reference = scope["convex_grasp_quality"]
    hull = torch.as_tensor(np.load(path), device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(932)
    q = torch.randn(1, 96, 4, device="cuda", generator=generator)
    q /= q.norm(dim=-1, keepdim=True)
    obj = q.new_tensor([[1, 0, 0, 0]])
    expected = reference(q, obj, hull)
    actual = convex_grasp_quality(q, obj, hull)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], equal_nan=True)
    milliseconds = []
    for function in (reference, convex_grasp_quality):
        for _ in range(2):
            function(q, obj, hull)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            function(q, obj, hull)
        end.record()
        end.synchronize()
        milliseconds.append(start.elapsed_time(end) / 10)
    print({"float32_reference_ms": milliseconds[0], "accurate_gpu_reduction_ms": milliseconds[1]})


def test_sparse_height_matches_preserve_full_contact_results_and_report_cost():
    if not torch.cuda.is_available():
        pytest.skip("GPU contact geometry")
    path = Path(__file__).parents[1] / "data/pick_assets/014_lemon/collision_support.npy"
    if not path.is_file():
        pytest.skip("downloaded real-asset hull unavailable")
    hull = torch.as_tensor(np.load(path), device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(923)
    q = torch.randn(4, 96, 4, device="cuda", generator=generator)
    q /= q.norm(dim=-1, keepdim=True)
    obj = q.new_tensor([[1, 0, 0, 0]]).expand(4, -1)

    def full():
        return convex_grasp_quality(q, obj, hull, return_depth_candidates=True)

    def sparse():
        centers = convex_grasp_quality(q, obj, hull, centers_only=True)
        # Same world-z test as measured station search, with a representative
        # floor object and measured command height, not a relaxed threshold.
        from gr00t.rl.auto_open.planning_geometry import _quat_rotate

        adjusted = centers["center_local"].clone()
        adjusted[..., 2] -= 0.0005
        height = 0.03 - _quat_rotate(q[:, :, None], adjusted)[..., 2]
        rows, candidates, depths = torch.where(
            centers["pocket_fit"] & ((height - 0.08).abs() <= 0.002)
        )
        values = convex_grasp_quality(
            q[rows, candidates, None],
            obj[rows],
            hull,
            return_depth_candidates=True,
            depth_fractions=(depths.to(q.dtype) / 16)[:, None, None],
        )
        return rows, candidates, depths, values

    expected = full()
    rows, candidates, depths, actual = sparse()
    assert len(rows) > 0
    for key, value in actual.items():
        torch.testing.assert_close(
            value[:, 0, 0],
            expected[key][rows, candidates, depths],
            rtol=0,
            atol=0,
            equal_nan=True,
            msg=key,
        )
    timings = {"total_depths": 4 * 96 * 17, "height_matched_depths": len(rows)}
    for name, function in (("full_ms", full), ("height_screened_ms", sparse)):
        function()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(3):
            function()
        end.record()
        end.synchronize()
        timings[name] = start.elapsed_time(end) / 3
    print(timings)


def test_real_contact_fused_solver_preserves_unfused_quality(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("GPU contact geometry")
    from gr00t.rl.pick import contact_profile_kernel
    from gr00t.rl.pick.grasp_quality import first_closing_contact

    path = Path(__file__).parents[1] / "data/pick_assets/014_lemon/collision_support.npy"
    if not path.is_file():
        pytest.skip("downloaded real-asset hull unavailable")
    hull = torch.as_tensor(np.load(path), device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(923)
    q = torch.randn(2, 96, 4, device="cuda", generator=generator)
    q /= q.norm(dim=-1, keepdim=True)
    obj = q.new_tensor([[1, 0, 0, 0]]).expand(2, -1)
    actual = convex_grasp_quality(q, obj, hull, return_depth_candidates=True)

    def unfused(local, centers, angles, profile):
        return first_closing_contact(
            angles,
            lambda angle: contact_profile_kernel.moving_profile_minimum(
                local, centers, angle, profile
            ),
        )

    monkeypatch.setattr(contact_profile_kernel, "first_profile_contact", unfused)
    expected = convex_grasp_quality(q, obj, hull, return_depth_candidates=True)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)
