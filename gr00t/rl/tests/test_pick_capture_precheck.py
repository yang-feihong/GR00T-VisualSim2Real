"""Necessary endpoint screening must not change feasible paths or branch selection."""

import math

import pytest
import torch

from gr00t.rl.pick import grasp_inverse
from gr00t.rl.pick.grasp_paths import score_default_grasp_paths
from gr00t.rl.tests.test_pick_batched_grasp_inverse import case, fake_inverse
from gr00t.rl.tests.test_pick_batched_grasp_paths import geometry_and_collision

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU inverse precheck")


@pytest.mark.parametrize("all_rejected", [False, True])
@pytest.mark.parametrize("required", [False, True])
def test_capture_gate_keeps_feasible_paths_and_skips_inexact_candidates(
    monkeypatch, all_rejected, required
):
    state, bank, limits = case()
    calls = []

    def solve(p, q, **kwargs):
        calls.append(len(p))
        result = fake_inverse(p, q, **kwargs)
        if all_rejected:
            result.exact.zero_()
        else:
            result.exact.logical_and_(p[:, 0] < 10)
        return result

    monkeypatch.setattr(grasp_inverse, "solve_pose_targets", solve)
    geometry, collision = geometry_and_collision()
    arguments = (state, bank, limits, collision.arm_joint_ids, state["object_state"][:, 2], 0.02)
    baseline = grasp_inverse.solve_default_grasp_inverse(*arguments)
    original_requests = sum(calls)
    calls.clear()
    mask = torch.full((3,), required, device="cuda", dtype=torch.bool)
    with monkeypatch.context() as scope:

        def forbidden(*args, **kwargs):
            raise AssertionError("Capture precheck must not copy GPU tensors to the host")

        for name in ("cpu", "item", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        available = grasp_inverse.realized_grasp_candidate_mask(
            state, bank, limits, collision.arm_joint_ids, mask, math.radians(2)
        )
        filtered = (bank[0], bank[1], {**bank[2], "pocket_fit": available}, bank[3], bank[4])
        gated = grasp_inverse.solve_default_grasp_inverse(
            state, filtered, limits, collision.arm_joint_ids, state["object_state"][:, 2], 0.02
        )
    if required:
        assert sum(calls) < original_requests
        assert (gated["lookup"][1] == -1).all()
    else:
        torch.testing.assert_close(gated["lookup"], baseline["lookup"])
    outputs = []
    for inverse in (baseline, gated):
        outputs.append(
            score_default_grasp_paths(
                state,
                bank,
                inverse,
                limits,
                collision,
                geometry,
                0.02,
                prefer_realized_body=torch.zeros_like(mask),
                realized_body=mask,
                prefer_orientation=False,
                minimum_realized_grasp_margin_rad=math.radians(2),
            )
        )
    left, right = outputs
    torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)
    if left[1].any():
        for a, b in zip(left[0], right[0], strict=True):
            torch.testing.assert_close(a[left[1]], b[left[1]], rtol=0, atol=0)
    torch.testing.assert_close(
        left[2]["selected"][left[1]], right[2]["selected"][left[1]], rtol=0, atol=0
    )
