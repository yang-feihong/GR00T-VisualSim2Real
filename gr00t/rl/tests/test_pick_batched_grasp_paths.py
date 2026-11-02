"""Default external path scoring uses the same predicates and costs in GPU batches."""

import math
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick import grasp_inverse, planning
from gr00t.rl.pick.grasp_paths import score_default_grasp_paths
from gr00t.rl.tests.test_pick_batched_grasp_inverse import case, fake_inverse

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU path scoring")


def geometry_and_collision():
    geometry = SimpleNamespace(select=lambda rows: SimpleNamespace(owners=rows))
    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6, device="cuda"),
        clearance=lambda g, roots, joints, **kw: roots.new_ones(roots.shape[:-1]),
        gripper_clearance=lambda g, p, q, **kw: p.new_ones(p.shape[:-1]),
    )
    return geometry, collision


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("realized", [False, True])
@pytest.mark.parametrize("obstacle", [False, True])
@pytest.mark.parametrize("margin", [0.0, math.radians(2)])
def test_shared_basket_grasp_options_match_scalar_selection_and_paths(
    monkeypatch, native, realized, obstacle, margin
):
    state, candidates, limits = case()
    if not native:
        monkeypatch.setattr(grasp_inverse, "solve_pose_targets", fake_inverse)
    geometry, collision = geometry_and_collision()
    if obstacle:
        collision.clearance = lambda g, roots, joints, **kw: torch.where(
            roots[..., 0] > 16, 0.004, 0.02
        )
    minimum = state["root_states"][:, 2] + 0.6
    pooled = grasp_inverse.solve_default_grasp_inverse(
        state,
        candidates,
        limits,
        collision.arm_joint_ids,
        state["object_state"][:, 2],
        0.02,
        collider_clearance_m=0.006,
        minimum_object_height_m=minimum,
        withdraw_distance_m=0.0,
    )
    paths, valid, diagnostics = score_default_grasp_paths(
        state,
        candidates,
        pooled,
        limits,
        collision,
        geometry,
        0.02,
        prefer_realized_body=torch.zeros(3, device="cuda", dtype=torch.bool),
        prefer_orientation=False,
        collider_clearance_m=0.006,
        require_safe_arm_path=True,
        validate_carry_preview=False,
        realized_body=torch.full((3,), realized, device="cuda", dtype=torch.bool),
        wrist_motion_weight=1e-5,
        minimum_realized_grasp_margin_rad=margin,
    )
    assert (pooled["withdraw_distance_m"] == 0).all()
    for row in range(2):
        local = {key: value[row : row + 1] for key, value in state.items()}
        bank = (
            candidates[0][row : row + 1],
            candidates[1][row : row + 1],
            {key: value[row : row + 1] for key, value in candidates[2].items()},
            3,
            0,
        )
        legacy, report = planning.plan_grasp(
            local,
            local["object_state"][:, :3],
            [0.04] * 3,
            limits,
            collision,
            None,
            0.02,
            object_support_points=None,
            prepared_candidates=bank,
            prepared_inverse=(pooled, row),
            minimum_object_height_m=float(minimum[row]),
            withdraw_distance_m=0.0,
            collider_clearance_m=0.006,
            require_safe_arm_path=True,
            validate_carry_preview=False,
            realized_body=realized,
            wrist_motion_weight=1e-5,
            minimum_realized_grasp_margin_rad=margin,
        )
        assert bool(valid[row]) == (legacy is not None)
        assert diagnostics["feasible"][row].tolist() == report["feasible"]
        comparable = diagnostics["feasible"][row]
        torch.testing.assert_close(
            diagnostics["candidate_cost"][row][comparable],
            state["ee_pos"].new_tensor(report["candidate_cost"])[comparable],
            atol=1e-5,
            rtol=1e-5,
        )
        assert diagnostics["feasible"][row].tolist() == report["feasible"]
        if legacy is not None:
            assert int(diagnostics["selected"][row]) == report["selected"]
            length = int(paths[2][row])
            assert length == int(legacy["approach"][2][0])
            torch.testing.assert_close(paths[0][row, :length], legacy["approach"][0][0])
            torch.testing.assert_close(paths[1][row, :length], legacy["approach"][1][0])


@pytest.mark.parametrize("prefer_actual", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("obstacle", [False, True])
@pytest.mark.parametrize("elevated", [False, True])
@pytest.mark.parametrize("preferred", [False, True])
def test_batched_scoring_matches_existing_scalar_path_costs_and_execution(
    monkeypatch, prefer_actual, native, obstacle, elevated, preferred
):
    state, candidates, limits = case()
    if not native:
        monkeypatch.setattr(grasp_inverse, "solve_pose_targets", fake_inverse)
    geometry, collision = geometry_and_collision()
    if obstacle:
        collision.clearance = lambda g, roots, joints, **kw: torch.where(
            (roots[..., 0] > 16) & (roots[..., 2] < 0.37), -0.01, 0.02
        )
        collision.gripper_clearance = lambda g, p, q, **kw: torch.where(
            p[..., 0] > 16, p[..., 2] - 0.75, p[..., 2] + 1
        )
    pooled = grasp_inverse.solve_default_grasp_inverse(
        state, candidates, limits, collision.arm_joint_ids, state["object_state"][:, 2], 0.02
    )
    paths, valid, diagnostics = score_default_grasp_paths(
        state,
        candidates,
        pooled,
        limits,
        collision,
        geometry,
        0.02,
        prefer_realized_body=torch.full((3,), prefer_actual, device="cuda", dtype=torch.bool),
        elevated_transit=elevated,
        prefer_orientation=preferred,
    )
    assert not valid[2]
    for row in range(2):
        local = {key: value[row : row + 1] for key, value in state.items()}
        bank = (
            candidates[0][row : row + 1],
            candidates[1][row : row + 1],
            {key: value[row : row + 1] for key, value in candidates[2].items()},
            3,
            0,
        )
        legacy, report = planning.plan_grasp(
            local,
            local["object_state"][:, :3],
            [0.04] * 3,
            limits,
            collision,
            None,
            0.02,
            object_support_points=None,
            prepared_candidates=bank,
            prepared_inverse=(pooled, row),
            preferred_orientation=state["ee_orn"][row : row + 1] if preferred else None,
            prefer_realized_body=prefer_actual,
            elevated_transit=elevated,
        )
        assert bool(valid[row]) == (legacy is not None)
        torch.testing.assert_close(
            diagnostics["candidate_cost"][row][diagnostics["feasible"][row]],
            state["ee_pos"].new_tensor(report["candidate_cost"])[diagnostics["feasible"][row]],
            atol=1e-5,
            rtol=1e-5,
        )
        assert diagnostics["feasible"][row].tolist() == report["feasible"]
        if legacy is not None:
            assert int(diagnostics["selected"][row]) == report["selected"]
            length = int(paths[2][row])
            assert length == int(legacy["approach"][2][0])
            torch.testing.assert_close(
                paths[0][row, :length], legacy["approach"][0][0], atol=2e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                paths[1][row, :length], legacy["approach"][1][0], atol=2e-5, rtol=1e-5
            )


def test_batched_path_scoring_has_no_tensor_host_reads_and_no_rows_borrow_other_scenes(monkeypatch):
    state, bank, limits = case()
    monkeypatch.setattr(grasp_inverse, "solve_pose_targets", fake_inverse)
    geometry, collision = geometry_and_collision()
    pooled = grasp_inverse.solve_default_grasp_inverse(
        state, bank, limits, collision.arm_joint_ids, state["object_state"][:, 2], 0.02
    )

    def clearance(g, p, q, **kw):
        torch._assert_async(
            (p[:, 0, 0] - state["root_states"][g.owners, 0]).abs().max() < 3,
            "A path point queried a different scene",
        )
        return p.new_ones(p.shape[:-1])

    collision.gripper_clearance = clearance

    def forbidden(*args, **kwargs):
        raise AssertionError("Path scoring must not read GPU tensor values on the host")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        paths, valid, diagnostics = score_default_grasp_paths(
            state,
            bank,
            pooled,
            limits,
            collision,
            geometry,
            0.02,
            prefer_realized_body=torch.zeros(3, device="cuda", dtype=torch.bool),
        )
    assert paths[0].is_cuda and paths[2].is_cuda
    assert valid.tolist() == [True, True, False]
    assert diagnostics["selected"][2] == -1
