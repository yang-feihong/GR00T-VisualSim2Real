"""Basket inverse gates preserve every body hypothesis and complete-path seed."""

import math
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open import pose_planning
from gr00t.rl.pick import basket_inverse, planning
from gr00t.rl.pick.basket_inverse import safe_inverse_paths_batch
from gr00t.rl.pick.basket_nudge import safe_inverse_paths
from gr00t.rl.tests.test_pick_batched_grasp_inverse import case, fake_inverse

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU basket inverse checks")


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("margin", [0.0, math.radians(2)])
@pytest.mark.parametrize("limits_dtype", [torch.float32, torch.float64])
def test_basket_inverse_matches_single_scene_predicates_and_joint_travel(
    native, margin, limits_dtype, monkeypatch
):
    state, bank, limits = case()
    limits = limits.to(limits_dtype)
    if not native:
        monkeypatch.setattr(pose_planning, "solve_pose_targets", fake_inverse)
        monkeypatch.setattr(planning, "solve_pose_targets", fake_inverse)
    t = torch.linspace(0, 1, 5, device="cuda")
    positions = state["ee_pos"][:, None, None] + t[None, None, :, None] * (
        bank[1][:, :, None] - state["ee_pos"][:, None, None]
    )
    quats = bank[0][:, :, None].expand(-1, -1, 5, -1)
    mask = bank[2]["pocket_fit"]
    geometry = SimpleNamespace(select=lambda rows: SimpleNamespace(owners=rows))
    calls = []

    def clearance(g, root, arm, **kwargs):
        calls.append(root.shape)
        return torch.where((root[..., 0] > 16) & (root[..., 2] < 0.395), -0.01, 0.02)

    collision = SimpleNamespace(arm_joint_ids=torch.arange(6, device="cuda"), clearance=clearance)
    offsets = (-0.01, 0.0, 0.01)

    def forbidden(*args, **kwargs):
        raise AssertionError("Basket path certification must remain on the GPU")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        result, travel, report = safe_inverse_paths_batch(
            state,
            positions,
            quats,
            limits,
            collision,
            geometry,
            candidate_mask=mask,
            body_height_offsets=offsets,
            final_joint_margin_rad=margin,
        )
    assert not result[2].any()
    assert travel.dtype == positions.dtype
    assert not result[1].any()  # One rejected height rejects the entire candidate.
    for row in range(2):
        ids = torch.where(mask[row])[0]
        single, single_travel = safe_inverse_paths(
            {key: value[row : row + 1] for key, value in state.items()},
            positions[row, ids],
            quats[row, ids],
            limits,
            collision,
            None,
            body_height_offsets=offsets,
            final_joint_margin_rad=margin,
        )
        torch.testing.assert_close(result[row, ids], single)
        torch.testing.assert_close(travel[row, ids], single_travel.to(travel.dtype))
    monkeypatch.setattr(basket_inverse, "MAX_ARM_QUERY_POSES", 3 * 3 * 5)
    calls.clear()
    chunked = safe_inverse_paths_batch(
        state,
        positions,
        quats,
        limits,
        collision,
        geometry,
        candidate_mask=mask,
        body_height_offsets=offsets,
        final_joint_margin_rad=margin,
    )
    assert len(calls) == int(mask.sum())
    torch.testing.assert_close(chunked[0], result)
    torch.testing.assert_close(chunked[1], travel)
    for name, expected in report.items():
        torch.testing.assert_close(chunked[2][name], expected, equal_nan=True)
