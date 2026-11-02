"""Open-jaw release selection remains equivalent under batch planning."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open import pose_planning
from gr00t.rl.pick import basket_nudge
from gr00t.rl.pick.basket_clear import plan_clear_for_nudge_batch
from gr00t.rl.tests.test_pick_batched_basket_carry import cases
from gr00t.rl.tests.test_pick_batched_grasp_inverse import fake_inverse

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU basket release")


class Geometry:
    def __init__(self, owners):
        self.owners = owners
        self.nonrobot_geometry = self

    def select(self, rows):
        return type(self)(self.owners[rows])

    def rigid_surface_clearance(self, p, q, points, **kwargs):
        return p.new_full(p.shape[:-1], 0.02)


@pytest.mark.parametrize("reject_second", [False, True])
@pytest.mark.parametrize("capacity", [2, 4096])
def test_release_batch_matches_scalar_and_reports_capacity_and_geometry_failures(
    monkeypatch, reject_second, capacity
):
    state, _, _, basket = cases()
    state["dof_pos"] = state["ee_pos"].new_zeros(2, 6)
    monkeypatch.setattr(pose_planning, "solve_pose_targets", fake_inverse)
    geometry = Geometry(torch.arange(2, device="cuda"))
    surfaces = (state["ee_pos"].new_zeros(1, 3), state["ee_pos"].new_zeros(1, 3))
    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6, device="cuda"),
        clearance=lambda g, r, j, **kw: r.new_ones(r.shape[:-1]),
        _gripper_local_surfaces=lambda p: surfaces,
        gripper_clearance=lambda g, p, q, **kw: (
            torch.where(p[..., 0] > 5, -0.01, 0.02)
            if reject_second
            else p.new_full(p.shape[:-1], 0.02)
        ),
    )
    limits = state["ee_pos"].new_tensor([[-3, 3]] * 6)

    def forbidden(*args, **kwargs):
        raise AssertionError("Release planning must not read tensor values on the CPU")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        path, valid, report = plan_clear_for_nudge_batch(
            state, basket, limits, collision, geometry, geometry, geometry, 0.02, capacity=capacity
        )
    if capacity == 2:
        assert not valid.any()
        assert report["planning_failure_code"].tolist() == [3, 1 if reject_second else 3]
        return
    assert valid.tolist() == [True, not reject_second]
    monkeypatch.setattr(
        basket_nudge, "build_distractor_mesh", lambda *a: {"points": None, "faces": None}
    )
    monkeypatch.setattr(basket_nudge, "augment_scene_geometry", lambda g, mesh: g)
    monkeypatch.setattr(basket_nudge, "SceneGeometry", lambda *a, **kw: geometry)
    monkeypatch.setattr(
        basket_nudge,
        "safe_inverse_paths",
        lambda s, p, q, *a, **kw: (
            torch.ones(len(p), device=p.device, dtype=torch.bool),
            p.new_zeros(len(p)),
        ),
    )
    for row in range(2):
        local = {key: value[row : row + 1] for key, value in state.items()}
        single, diagnostics = basket_nudge.plan_clear_for_nudge(
            local, basket, limits, collision, geometry, 0.02, episode={}
        )
        assert bool(valid[row]) == (single is not None)
        if single is not None:
            n = int(path[2][row])
            assert single[0].shape[1] == n
            torch.testing.assert_close(path[0][row, :n], single[0][0])
            torch.testing.assert_close(path[1][row, :n], single[1][0])
            torch.testing.assert_close(
                report["target_world"][row], state["ee_pos"].new_tensor(diagnostics["target_world"])
            )
    reverse = torch.tensor([1, 0], device="cuda")
    reordered = plan_clear_for_nudge_batch(
        {key: value[reverse] for key, value in state.items()},
        basket,
        limits,
        collision,
        geometry.select(reverse),
        geometry.select(reverse),
        geometry.select(reverse),
        0.02,
    )
    for expected, actual in zip(path, reordered[0]):
        torch.testing.assert_close(expected, actual[reverse])
    torch.testing.assert_close(valid, reordered[1][reverse])
