"""Pooled closed-jaw plans preserve direction choice, geometry checks and time law."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open import pose_planning
from gr00t.rl.pick import basket_nudge, basket_push
from gr00t.rl.pick.basket_push import plan_closed_nudge_batch
from gr00t.rl.tests.test_pick_batched_basket_carry import cases
from gr00t.rl.tests.test_pick_batched_grasp_inverse import fake_inverse

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU basket pushing")


class Geometry:
    def __init__(self, owners):
        self.owners = owners
        self.nonrobot_geometry = self

    def select(self, rows):
        return type(self)(self.owners[rows])

    def rays(self, origins, targets):
        delta = targets - origins
        return (
            origins + 0.475 * delta,
            -delta / delta.norm(dim=-1, keepdim=True),
            delta.norm(dim=-1) * 0.475,
        )


@pytest.mark.parametrize("reject_second", [False, True])
@pytest.mark.parametrize("require_stability", [False, True])
def test_closed_nudge_batch_matches_scalar_targets_directions_and_paths(
    monkeypatch, reject_second, require_stability
):
    state, support, support_mask, basket = cases()
    basket.floor_static_friction = 0.0
    state["dof_pos"] = support.new_zeros(2, 6)
    state["distractor_states"] = support.new_zeros(2, 0, 13)
    sizes = support.new_tensor([[[0.04, 0.04, 0.04]], [[0.05, 0.03, 0.04]]])
    geometry = Geometry(torch.arange(2, device="cuda"))
    fixed, moving = support.new_tensor([[0.1, 0, 0]]), support.new_tensor([[0.1, 0.01, 0]])
    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6, device="cuda"),
        clearance=lambda g, r, j, **kw: r.new_full(r.shape[:-1], 0.02),
        _gripper_local_surfaces=lambda p: (fixed, moving),
        gripper_pose_clearance=lambda g, p, q, a, **kw: (
            torch.where(p[..., 0] > 5, -0.01, 0.02)
            if reject_second
            else p.new_full(p.shape[:-1], 0.02)
        ),
    )
    limits = support.new_tensor([[-3, 3]] * 6)
    monkeypatch.setattr(pose_planning, "solve_pose_targets", fake_inverse)
    monkeypatch.setattr(
        basket_push,
        "shifted_grasp_counts_batch",
        lambda s, z, targets, *a, **kw: (
            torch.ones(targets.shape[:2], device=targets.device, dtype=torch.long),
            {},
        ),
    )

    def run(rows, *, unavailable=True, capacity=4096):
        return plan_closed_nudge_batch(
            {key: value[rows] for key, value in state.items()},
            sizes[rows],
            sizes[rows, 0],
            torch.ones(len(rows), 1, device="cuda", dtype=torch.bool),
            basket,
            limits,
            collision,
            geometry.select(rows),
            geometry.select(rows),
            geometry.select(rows),
            0.02,
            support_points=support[rows],
            support_mask=support_mask[rows],
            current_path_unavailable=torch.full(
                (len(rows),), unavailable, device="cuda", dtype=torch.bool
            ),
            capacity=capacity,
            require_sliding_stability=require_stability,
            object_centers_local=support.new_zeros(len(rows), 3),
        )

    rows = torch.arange(2, device="cuda")

    def forbidden(*args, **kwargs):
        raise AssertionError("Closed-jaw planning must remain on the GPU")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        path, valid, report = run(rows)
    if not require_stability:
        assert valid.tolist() == [True, not reject_second]
    else:
        assert valid[0], "Early stability screening must retain a feasible closed-jaw push"
        assert report["sliding_stable_candidate_count"][0] > 0
    if reject_second:
        assert not valid[1]
    from gr00t.rl.pick import grasp_quality

    monkeypatch.setattr(grasp_quality, "load_grasp_center", lambda *a: support.new_zeros(3))
    monkeypatch.setattr(
        basket_nudge,
        "shifted_grasp_counts",
        lambda s, e, t, *a, **kw: torch.ones(len(t), device=t.device, dtype=torch.long),
    )
    monkeypatch.setattr(
        basket_nudge, "build_distractor_mesh", lambda *a: {"points": None, "faces": None}
    )
    monkeypatch.setattr(basket_nudge, "SceneGeometry", lambda *a, **kw: geometry)
    monkeypatch.setattr(basket_nudge, "augment_scene_geometry", lambda g, m: g)
    monkeypatch.setattr(
        basket_nudge,
        "safe_inverse_paths",
        lambda s, p, q, *a, **kw: (
            torch.ones(len(p), device=p.device, dtype=torch.bool),
            p.new_zeros(len(p)),
        ),
    )
    for row in range(2):
        scalar, diagnostic = basket_nudge.plan_closed_nudge(
            {key: value[row : row + 1] for key, value in state.items()},
            {"asset": {"size_m": sizes[row, 0].tolist(), "usd_path": "test.usd"}},
            basket,
            limits,
            collision,
            geometry,
            0.02,
            object_support_points=support[row, support_mask[row]],
            current_path_unavailable=True,
            require_sliding_stability=require_stability,
        )
        assert bool(valid[row]) == (scalar is not None)
        if scalar is not None:
            n = int(path[2][row])
            assert scalar[0].shape[1] == n
            torch.testing.assert_close(path[0][row, :n], scalar[0][0])
            torch.testing.assert_close(path[1][row, :n], scalar[1][0])
            torch.testing.assert_close(
                report["direction_world"][row], support.new_tensor(diagnostic["direction_world"])
            )
            assert float(report["shift_m"][row]) == pytest.approx(diagnostic["shift_m"])
    reverse = rows.flip(0)
    reordered = run(reverse)
    for value, expected in zip(reordered[0], path):
        torch.testing.assert_close(value[reverse], expected)
    torch.testing.assert_close(reordered[1][reverse], valid)
    for key in (
        "direction_world",
        "shift_m",
        "segment_boundaries",
        "selected_bank_index",
        "contact_target_world",
    ):
        torch.testing.assert_close(reordered[2][key][reverse], report[key], equal_nan=True)
    short = run(rows, capacity=2)
    assert not short[1].any()
    if valid[0]:
        assert short[2]["planning_failure_code"][0] == 5
    unnecessary = run(rows, unavailable=False)
    assert unnecessary[2]["not_required"].all()
    assert not unnecessary[1].any()
