"""Sparse grasp inverse pooling preserves per-scene paths, hypotheses and candidates."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.pick import grasp_inverse, planning
from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.ee_steady_inverse_gpu import (
    BatchedSteadyEeInverseResult,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU grasp inverse pooling")


def case():
    roots = torch.zeros(3, 13, device="cuda")
    roots[:, 3] = 1
    roots[:, 0] = roots.new_tensor([0, 32, -32])
    roots[:, 2] = roots.new_tensor([0.52, 0.4, 0.3])
    target = roots[:, :3] + roots.new_tensor([[0.6, 0, -0.49], [0.6, 0, 0.3], [0.6, 0, 0.1]])
    q = _euler_xyz_quat(roots.new_tensor([[0, 1.4, 0], [0, 0.4, 0], [0, 0, 0]]))
    obj = roots.clone()
    obj[:, :3] = target
    state = {
        "root_states": roots,
        "ee_orn": q,
        "object_state": obj,
        "ee_pos": target + roots.new_tensor([-0.2, 0, 0.2]),
        "dof_pos": roots.new_zeros(3, 6),
    }
    state["dof_pos"][1] = 0.2
    orientations = q[:, None].expand(-1, 3, -1).clone()
    goals = target[:, None].expand(-1, 3, -1).clone()
    goals[:, :, 2] += roots.new_tensor([-0.01, 0.01, 0.02])
    quality = {
        "pocket_fit": torch.tensor(
            [[True, True, False], [False, True, False], [False] * 3], device="cuda"
        ),
        "antipodal_margin": roots.new_ones(3, 3),
        "environment_clearance_m": roots.new_ones(3, 3),
    }
    return state, (orientations, goals, quality, 3, 0), roots.new_tensor([[-3, 3]] * 6)


def fake_inverse(p, q, **kwargs):
    arm = kwargs["initial_arm_joint_positions"].clone()
    return BatchedSteadyEeInverseResult(
        p.clone(),
        q.clone(),
        arm,
        arm.clone(),
        torch.zeros_like(p),
        torch.zeros_like(p),
        torch.ones(len(p), device=p.device, dtype=torch.bool),
    )


def test_nominal_preflight_height_subset_keeps_identical_inverse_requests(monkeypatch):
    state, bank, limits = case()
    monkeypatch.setattr(grasp_inverse, "solve_pose_targets", fake_inverse)
    ids = torch.arange(6, device="cuda")
    heights = planning.grasp_body_height_hypotheses(state["root_states"])
    full = grasp_inverse.solve_default_grasp_inverse(
        state, bank, limits, ids, state["object_state"][:, 2], 0.02
    )
    local = grasp_inverse.solve_default_grasp_inverse(
        state,
        bank,
        limits,
        ids,
        state["object_state"][:, 2],
        0.02,
        body_heights=heights[:, -3:],
    )
    assert full["body_heights"].shape[1] == 10 and local["body_heights"].shape[1] == 3
    for name, value in full["result"].items():
        torch.testing.assert_close(
            local["result"][name], value[-3:], rtol=0, atol=0, equal_nan=True
        )
    torch.testing.assert_close(full["lookup"], local["lookup"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Body height"):
        grasp_inverse.solve_default_grasp_inverse(
            state,
            bank,
            limits,
            ids,
            state["object_state"][:, 2],
            0.02,
            body_heights=heights[:2, -3:],
        )


def test_pooled_inverse_matches_split_and_keeps_empty_rows_without_host_tensor_reads(monkeypatch):
    state, bank, limits = case()
    calls = []

    def solve(p, q, **kwargs):
        calls.append(len(p))
        return fake_inverse(p, q, **kwargs)

    monkeypatch.setattr(grasp_inverse, "solve_pose_targets", solve)

    def forbidden(*args, **kwargs):
        raise AssertionError("Inverse request pooling must remain on the GPU")

    ids = torch.arange(6, device="cuda")
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        result = grasp_inverse.solve_default_grasp_inverse(
            state, bank, limits, ids, state["object_state"][:, 2], 0.02
        )
    assert len(calls) == 1
    assert (result["lookup"][2] == -1).all()
    for row in range(2):
        local_bank = (
            bank[0][row : row + 1],
            bank[1][row : row + 1],
            {key: value[row : row + 1] for key, value in bank[2].items()},
            3,
            0,
        )
        split = grasp_inverse.solve_default_grasp_inverse(
            {key: value[row : row + 1] for key, value in state.items()},
            local_bank,
            limits,
            ids,
            state["object_state"][row : row + 1, 2],
            0.02,
        )
        selected = torch.where(bank[2]["pocket_fit"][row])[0]
        count = int(result["reference_lengths"][row]) + 7
        together = grasp_inverse.select_grasp_inverse(result, row, selected, count)
        alone = grasp_inverse.select_grasp_inverse(split, 0, selected, count)
        for name in together.__dataclass_fields__:
            torch.testing.assert_close(getattr(together, name), getattr(alone, name))


def test_cached_requests_match_scalar_planner_and_do_not_resolve_per_row(monkeypatch):
    state, bank, limits = case()
    monkeypatch.setattr(grasp_inverse, "solve_pose_targets", fake_inverse)
    pooled = grasp_inverse.solve_default_grasp_inverse(
        state, bank, limits, torch.arange(6, device="cuda"), state["object_state"][:, 2], 0.02
    )
    captured = []

    def solve(p, q, **kwargs):
        captured.append((p.clone(), q.clone()))
        return fake_inverse(p, q, **kwargs)

    monkeypatch.setattr(planning, "solve_pose_targets", solve)
    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6, device="cuda"),
        clearance=lambda g, roots, joints, **kw: roots.new_ones(roots.shape[:-1]),
        gripper_clearance=lambda g, p, q: p.new_ones(p.shape[:-1]),
    )
    for row in range(2):
        local = {key: value[row : row + 1] for key, value in state.items()}
        candidates = (
            bank[0][row : row + 1],
            bank[1][row : row + 1],
            {key: value[row : row + 1] for key, value in bank[2].items()},
            3,
            0,
        )
        args = (local, local["object_state"][:, :3], [0.04] * 3, limits, collision, None, 0.02)
        paths, report = planning.plan_grasp(
            *args, object_support_points=None, prepared_candidates=candidates
        )
        selected = torch.where(bank[2]["pocket_fit"][row])[0]
        length = len(report["planning_path_indices"]) + 7
        cached_result = grasp_inverse.select_grasp_inverse(pooled, row, selected, length)
        torch.testing.assert_close(
            cached_result.command_position_world, captured[-1][0], atol=2e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            cached_result.command_orientation_world_wxyz, captured[-1][1], atol=1e-6, rtol=1e-6
        )
        before = len(captured)
        cached_paths, cached_report = planning.plan_grasp(
            *args,
            object_support_points=None,
            prepared_candidates=candidates,
            prepared_inverse=(pooled, row),
        )
        assert len(captured) == before
        assert paths is not None and cached_paths is not None
        assert report["selected"] == cached_report["selected"]
        for name in paths:
            for expected, actual in zip(paths[name], cached_paths[name]):
                torch.testing.assert_close(actual, expected)


def test_empty_batch_does_not_submit_inverse(monkeypatch):
    state, bank, limits = case()
    bank[2]["pocket_fit"].zero_()
    monkeypatch.setattr(
        grasp_inverse, "solve_pose_targets", lambda *a, **kw: pytest.fail("Empty solve")
    )
    result = grasp_inverse.solve_default_grasp_inverse(
        state, bank, limits, torch.arange(6, device="cuda"), state["object_state"][:, 2], 0.02
    )
    assert result["result"] is None
    assert (result["lookup"] == -1).all()


def test_request_chunking_preserves_the_entire_bank_and_bounds_solver_workspace(monkeypatch):
    state, bank, limits = case()
    ids = torch.arange(6, device="cuda")
    monkeypatch.setattr(grasp_inverse, "solve_pose_targets", fake_inverse)
    whole = grasp_inverse.solve_default_grasp_inverse(
        state, bank, limits, ids, state["object_state"][:, 2], 0.02
    )
    sizes = []

    def solve(p, q, **kwargs):
        sizes.append(len(p))
        return fake_inverse(p, q, **kwargs)

    monkeypatch.setattr(grasp_inverse, "MAX_GRASP_INVERSE_REQUESTS", 128)
    monkeypatch.setattr(grasp_inverse, "solve_pose_targets", solve)
    chunked = grasp_inverse.solve_default_grasp_inverse(
        state, bank, limits, ids, state["object_state"][:, 2], 0.02
    )
    assert len(sizes) > 1 and max(sizes) <= 128
    for name, value in whole["result"].items():
        torch.testing.assert_close(value, chunked["result"][name], equal_nan=True)


def test_retry_exclusions_do_not_reject_another_environments_candidates(monkeypatch):
    state, bank, limits = case()
    monkeypatch.setattr(grasp_inverse, "solve_pose_targets", fake_inverse)
    result = grasp_inverse.solve_default_grasp_inverse(
        state,
        bank,
        limits,
        torch.arange(6, device="cuda"),
        state["object_state"][:, 2],
        0.02,
        excluded=bank[0][:, :1],
        excluded_mask=torch.tensor([[True], [False], [False]], device="cuda"),
    )
    assert (result["lookup"][0] == -1).all()
    assert result["lookup"][1, 1] == 0


def test_native_inverse_results_are_independent_of_other_environment_requests():
    state, bank, limits = case()
    ids = torch.arange(6, device="cuda")
    result = grasp_inverse.solve_default_grasp_inverse(
        state, bank, limits, ids, state["object_state"][:, 2], 0.02
    )
    for row in range(2):
        local_bank = (
            bank[0][row : row + 1],
            bank[1][row : row + 1],
            {key: value[row : row + 1] for key, value in bank[2].items()},
            3,
            0,
        )
        split = grasp_inverse.solve_default_grasp_inverse(
            {key: value[row : row + 1] for key, value in state.items()},
            local_bank,
            limits,
            ids,
            state["object_state"][row : row + 1, 2],
            0.02,
        )
        selected = torch.where(bank[2]["pocket_fit"][row])[0]
        count = int(result["reference_lengths"][row]) + 7
        together = grasp_inverse.select_grasp_inverse(result, row, selected, count)
        alone = grasp_inverse.select_grasp_inverse(split, 0, selected, count)
        for name in together.__dataclass_fields__:
            torch.testing.assert_close(getattr(together, name), getattr(alone, name))
