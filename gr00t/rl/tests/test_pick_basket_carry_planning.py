"""Basket lift selection and all retimed surface samples survive GPU batching."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open import pose_planning
from gr00t.rl.pick import basket_carry, planning
from gr00t.rl.pick.basket_carry import (
    certify_padded_held_path,
    plan_basket_carry_bank,
    plan_basket_carry_batch,
)
from gr00t.rl.tests.test_pick_batched_basket_carry import cases
from gr00t.rl.tests.test_pick_batched_grasp_inverse import fake_inverse

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU basket lift planner")


def test_cost_bound_keeps_late_original_index_tie_and_never_bypasses_dense_failure(monkeypatch):
    state, points, mask, basket = cases()
    state["dof_pos"] = points.new_zeros(2, 6)
    bank = points.new_zeros(18, 3)
    bank[:, 0] = 0.3
    bank[0, 0], bank[1, 0], bank[5:13, 0] = 0, 0.18, 0.1
    bank[:, 1] = torch.arange(18, device=points.device)
    calls = []

    def candidates(*args, **kwargs):
        return (
            points.new_zeros(2, 18, 3, 3),
            points.new_zeros(2, 18, 3, 4),
            points.new_zeros(2, 3),
            bank,
        )

    def evaluate(local, initial, aperture, limits, collision, geometry, dt, **options):
        columns = options["bank"][:, 1].long()
        owner = geometry.owners
        calls.append((owner.clone(), columns.clone()))
        feasible = torch.where(owner[:, None] == 0, (columns == 1) | (columns == 9), columns == 0)
        cost = points.new_full(feasible.shape, 0.18)
        cost[:, columns == 0] = 0.001
        selected, found = pose_planning.best_feasible(feasible, cost)
        index = columns[selected]
        path = (
            index[:, None, None].to(points.dtype).expand(-1, 4, 3).clone(),
            points.new_zeros(len(owner), 4, 4),
        )
        return (
            path,
            found,
            {
                "geometric_candidate_mask": feasible,
                "robust_inverse_mask": feasible,
                "candidate_cost": cost,
                "candidate_count": len(columns),
                "selected": torch.where(found, selected, -1),
                "selected_tilt_rad_azimuth_rad_retract_m": options["bank"][selected],
                "geometric_candidates": feasible.sum(-1),
            },
        )

    monkeypatch.setattr(basket_carry, "held_pose_bank_batch", candidates)
    monkeypatch.setattr(basket_carry, "_plan_basket_carry_candidates", evaluate)
    monkeypatch.setattr(
        basket_carry,
        "retime_grasp_approach_batch",
        lambda p, q, *a, **kw: (
            (p, q, torch.full((len(p),), 4, device=p.device, dtype=torch.long)),
            torch.ones(len(p), device=p.device, dtype=torch.bool),
        ),
    )
    monkeypatch.setattr(
        basket_carry,
        "certify_padded_held_path",
        lambda state, *a, **kw: (state["root_states"][:, 0] < 5, {}),
    )
    args = (
        state,
        points.new_zeros(2),
        points.new_zeros(2),
        points.new_zeros(6, 2),
        None,
        Geometry(torch.arange(2, device=points.device)),
        0.02,
    )
    options = dict(
        basket=basket,
        support_points=points,
        support_mask=mask,
        minimum_object_height=points.new_ones(2),
    )
    baseline = plan_basket_carry_batch(*args, **options, early_exit=False)
    calls.clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("Candidate scheduling must not read tensors on the host")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist", "__float__"):
            scope.setattr(torch.Tensor, name, forbidden)
        optimized = plan_basket_carry_batch(*args, **options)
    assert len(calls) == 3
    assert calls[0][0].tolist() == [0, 1]
    assert all(owners.tolist() == [0] for owners, _ in calls[1:])
    assert optimized[2]["selected"].tolist() == [1, -1]
    assert optimized[2]["planning_failure_code"].tolist() == [0, 4]
    for a, b in zip(baseline[0], optimized[0], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(baseline[1], optimized[1])


class Geometry:
    def __init__(self, owners):
        self.owners = owners
        self.nonrobot_geometry = self

    def select(self, rows):
        return type(self)(self.owners[rows])

    def rigid_surface_clearance(self, positions, quats, points, **kwargs):
        return positions.new_ones(positions.shape[:-1])


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reject_second", [False, True])
def test_basket_lift_batch_matches_legacy_selection_and_retimed_targets(
    native, reject_second, monkeypatch
):
    state, points, mask, basket = cases()
    state["dof_pos"] = points.new_zeros(2, 6)
    if not native:
        monkeypatch.setattr(pose_planning, "solve_pose_targets", fake_inverse)
        monkeypatch.setattr(planning, "solve_pose_targets", fake_inverse)
    geometry = Geometry(torch.arange(2, device="cuda"))

    def clearance(g, roots, joints, **kwargs):
        return (
            torch.where(roots[..., 0] > 5, -0.01, 0.02)
            if reject_second
            else roots.new_ones(roots.shape[:-1])
        )

    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6, device="cuda"),
        clearance=clearance,
        gripper_pose_clearance=lambda g, p, q, a, **kw: p.new_ones(p.shape[:-1]),
    )
    minimum = state["root_states"][:, 2] + 0.5
    aperture = points.new_full((2,), 0.5)
    limits = points.new_tensor([[-3, 3]] * 6)

    def forbidden(*args, **kwargs):
        raise AssertionError("The basket lift planner must remain on the GPU")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        path, valid, report = plan_basket_carry_batch(
            state,
            state["object_state"][:, 2],
            aperture,
            limits,
            collision,
            geometry,
            0.02,
            basket=basket,
            support_points=points,
            support_mask=mask,
            minimum_object_height=minimum,
            tilt_angles_deg=(5,),
        )
    for row in range(2):
        local = {key: value[row : row + 1] for key, value in state.items()}
        single, diagnostics = plan_basket_carry_bank(
            local,
            local["object_state"][:, 2],
            aperture[row : row + 1],
            limits,
            collision,
            geometry,
            0.02,
            basket=basket,
            support_points=points[row, mask[row]],
            minimum_object_height=minimum[row : row + 1],
            tilt_angles_deg=(5,),
        )
        assert bool(valid[row]) == (single is not None)
        if single is not None:
            assert int(report["selected"][row]) == diagnostics["selected"]
            length = int(path[2][row])
            assert length == int(single[2][0])
            torch.testing.assert_close(path[0][row, :length], single[0][0])
            torch.testing.assert_close(path[1][row, :length], single[1][0])
    if reject_second:
        assert not valid[1] and report["planning_failure_code"][1] == 2


def test_dense_held_path_checks_only_real_frames_and_keeps_release_monotonicity():
    root = torch.zeros(2, 13, device="cuda")
    root[:, 3] = 1
    state = {"root_states": root, "ee_orn": root[:, 3:7], "object_state": root.clone()}
    p = root.new_full((2, 8, 3), torch.nan)
    p[0, :4] = root.new_tensor([[0, 0, 0.001], [0, 0, 0.0015], [0, 0, 0.002], [0, 0, 0.003]])
    p[1, :2] = root.new_tensor([[0, 0, 0.003], [0, 0, 0.004]])
    q = root.new_tensor([1, 0, 0, 0]).expand(2, 8, -1)
    lengths = torch.tensor([4, 2], device="cuda")
    geometry = Geometry(torch.arange(2, device="cuda"))
    calls = []

    def clearance(g, p, q, aperture, **options):
        assert options["maximum_distance_m"] == 0.0021
        calls.append(len(p))
        return p[..., 2]

    collision = SimpleNamespace(gripper_pose_clearance=clearance)
    basket = SimpleNamespace(
        floor_lower=root.new_full((3,), -1), floor_upper=root.new_ones(3), rim_z=1.0
    )
    args = (
        state,
        (p, q, lengths),
        root.new_zeros(2, 3),
        root.new_zeros(1, 3),
        basket,
        collision,
        geometry,
        root.new_zeros(2),
    )
    valid, report = certify_padded_held_path(*args)
    assert (
        valid.all() and sum(calls) == 6
    )  # Neither NaN padding nor repeated terminal frames are queried.
    assert report["dense_gripper_min_clearance_m"][0] < 0.002
    p[0, 2, 2] = 0.0005
    valid, report = certify_padded_held_path(*args)
    assert valid.tolist() == [False, True]
    assert not report["dense_gripper_release_monotone"][0]


@pytest.mark.parametrize("capacity", [2, 4096])
@pytest.mark.parametrize("reject_second", [False, True])
def test_basket_lift_permutation_preserves_paths_and_isolates_failure(
    capacity, reject_second, monkeypatch
):
    state, points, mask, basket = cases()
    state["dof_pos"] = points.new_zeros(2, 6)
    monkeypatch.setattr(pose_planning, "solve_pose_targets", fake_inverse)
    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6, device="cuda"),
        clearance=lambda g, roots, joints, **kw: roots.new_ones(roots.shape[:-1]),
        gripper_pose_clearance=lambda g, p, q, a, **kw: (
            torch.where(g.owners[:, None, None] == 1, -0.01, 0.02).expand(p.shape[:-1])
            if reject_second
            else p.new_ones(p.shape[:-1])
        ),
    )
    limits = points.new_tensor([[-3, 3]] * 6)
    geometry = Geometry(torch.arange(2, device="cuda"))

    def run(order):
        return plan_basket_carry_batch(
            {key: value[order] for key, value in state.items()},
            state["object_state"][order, 2],
            points.new_full((2,), 0.5),
            limits,
            collision,
            geometry.select(order),
            0.02,
            basket=basket,
            support_points=points[order],
            support_mask=mask[order],
            minimum_object_height=state["root_states"][order, 2] + 0.5,
            tilt_angles_deg=(5,),
            capacity=capacity,
        )

    forward = torch.tensor([0, 1], device="cuda")
    reverse = forward.flip(0)
    path, valid, report = run(forward)
    reversed_path, reversed_valid, reversed_report = run(reverse)
    torch.testing.assert_close(valid, reversed_valid[reverse])
    for key in ("selected", "planning_failure_code", "geometric_candidate_mask"):
        torch.testing.assert_close(report[key], reversed_report[key][reverse])
    for expected, actual in zip(path, reversed_path):
        torch.testing.assert_close(expected, actual[reverse], equal_nan=True)
    if capacity == 2:
        assert not valid.any()
        assert report["planning_failure_code"].tolist() == [3, 1 if reject_second else 3]
    else:
        assert valid.tolist() == [True, not reject_second]
        assert report["planning_failure_code"].tolist() == [0, 1 if reject_second else 0]
