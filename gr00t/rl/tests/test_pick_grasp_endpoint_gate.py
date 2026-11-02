"""Do not execute inexact endpoints; permit wholebody height changes between them."""

import pytest
import torch

from gr00t.rl.pick import grasp_inverse
from gr00t.rl.pick.grasp_paths import score_default_grasp_paths
from gr00t.rl.tests.test_pick_batched_grasp_inverse import case, fake_inverse
from gr00t.rl.tests.test_pick_batched_grasp_paths import geometry_and_collision

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU grasp planning")


@pytest.mark.parametrize("reachable", [False, True])
def test_endpoint_gate_allows_crouch_then_raise_but_not_inexact_grasp(monkeypatch, reachable):
    state, bank, limits = case()
    monkeypatch.setattr(grasp_inverse, "solve_pose_targets", fake_inverse)
    geometry, collision = geometry_and_collision()
    inverse = grasp_inverse.solve_default_grasp_inverse(
        state, bank, limits, collision.arm_joint_ids, state["object_state"][:, 2], 0.02
    )
    owner, _ = torch.where(inverse["lookup"] >= 0)
    endpoints = inverse["counts"][owner].cumsum(-1) - 1
    slots = torch.searchsorted(inverse["references"][owner].contiguous(), endpoints.contiguous())
    exact = inverse["result"]["exact"]
    exact.zero_()
    if reachable:
        rows = torch.arange(len(owner), device="cuda")
        # No single fixed body height can realize all phases. This is intentional:
        # the wholebody policy can crouch for grasp and rise during the lift.
        for endpoint in range(4):
            exact[endpoint, :, rows, slots[:, endpoint]] = True
    _, valid, detail = score_default_grasp_paths(
        state,
        bank,
        inverse,
        limits,
        collision,
        geometry,
        0.02,
        prefer_realized_body=torch.zeros(3, device="cuda", dtype=torch.bool),
        prefer_orientation=False,
    )
    assert valid[:2].eq(reachable).all()
    assert not valid[2]
    assert detail["endpoint_inverse_available"][0, 0].eq(reachable)


def test_retry_continuous_realized_residual_can_override_hypothetical_preferred_pose(monkeypatch):
    state, bank, limits = case()
    monkeypatch.setattr(grasp_inverse, "solve_pose_targets", fake_inverse)
    geometry, collision = geometry_and_collision()
    inverse = grasp_inverse.solve_default_grasp_inverse(
        state, bank, limits, collision.arm_joint_ids, state["object_state"][:, 2], 0.02
    )
    owner, candidate = torch.where(inverse["lookup"] >= 0)
    raw = inverse["result"]
    raw["exact"][-3:] = False
    residual = torch.where((owner == 0) & (candidate == 0), 0.04, 0.01)
    raw["position_error"][-3:, ..., 2] = residual[None, None, :, None]
    results = []
    for retry in (False, True):
        _, valid, detail = score_default_grasp_paths(
            state, bank, inverse, limits, collision, geometry, 0.02,
            prefer_realized_body=torch.full((3,), retry, device="cuda", dtype=torch.bool),
            prefer_orientation=True,
        )
        assert valid[:2].all()
        results.append(detail["selected"])
    assert results[0][0] == 0
    assert results[1][0] == 1
