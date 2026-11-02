"""Gaze admission keeps each scene and padded path independent on device."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick import gaze


@dataclass
class Result:
    arm_joint_positions: torch.Tensor
    position_error: torch.Tensor
    orientation_error: torch.Tensor


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU gaze planning")
    return request.param


def inputs(device):
    roots = torch.zeros(3, 5, 13, device=device)
    roots[..., 3] = 1
    roots[..., 0] = torch.tensor([0, 10, 20], device=device)[:, None]
    roots[..., 2] = 0.53
    initial = roots.new_tensor([[0, 1, -1, 0, 0, 0]]).expand(3, -1).clone()
    target = roots[:, 0, :3] + roots.new_tensor([1, 0, 0.2])
    limits = roots.new_tensor([[-3, 3]] * 6)
    return roots, initial, target, limits


def install_reference_tools(monkeypatch):
    requests = []

    def candidates(roots, initial, target, limits, dt, offsets, roll_step, **kwargs):
        batch, count = roots.shape[:2]
        p = target[:, None, None].expand(-1, 2, count, -1).clone()
        p[..., 2] += 0.1
        p[:, 1, :, 1] += 0.1
        q = roots[:, None, :, 3:7].expand(-1, 2, -1, -1).clone()
        template = initial[:, None, None].expand(-1, 2, count, -1).clone()
        expanded = roots[:, None].expand(-1, 2, -1, -1)
        choice = kwargs.get("candidate_indices")
        if choice is not None:
            p, q, template, expanded = (
                value.gather(1, choice[..., None, None].expand(-1, -1, count, value.shape[-1]))
                for value in (p, q, template, expanded)
            )
        return (
            p.flatten(0, 1),
            q.flatten(0, 1),
            template.flatten(0, 1),
            expanded.flatten(0, 1),
            roots.new_tensor([0, 0.1]),
            roots.new_zeros(2),
        )

    def inverse(p, q, **kwargs):
        requests.append(len(p))
        root = kwargs["base_position_world"]
        # Relative error varies by scene but is invariant to environment ordering.
        error = torch.zeros_like(p)
        error[:, 0] = root[:, 0] * 0.0001
        return Result(kwargs["initial_arm_joint_positions"].clone(), error, torch.zeros_like(p))

    def clearance(geometry, roots, joints, **kwargs):
        assert roots.shape[0] == len(geometry)
        assert torch.isfinite(joints).all()
        # Scene 10 cannot use candidate 0, scene 20 has no feasible candidate.
        value = roots.new_full(roots.shape[:-1], 0.3)
        value[:, 0] = torch.where(geometry[:, None] == 10, 0.0, value[:, 0])
        return torch.where((geometry == 20)[:, None, None], 0.0, value)

    monkeypatch.setattr(gaze, "natural_gaze_candidates", candidates)
    monkeypatch.setattr(
        gaze,
        "gaze_candidate_offsets",
        lambda reference, *a, **kw: (
            reference.new_tensor([0, 0.1]),
            reference.new_zeros(2),
            reference.new_zeros(2),
        ),
    )
    monkeypatch.setattr(gaze, "solve_pose_targets", inverse)
    return SimpleNamespace(clearance=clearance), requests


def test_gaze_batch_matches_split_and_permutation_with_padded_lengths(monkeypatch, device):
    roots, initial, target, limits = inputs(device)
    collision, requests = install_reference_tools(monkeypatch)
    lengths = torch.tensor([5, 2, 4], device=device)

    def solve(rows):
        return gaze.plan_approach_gaze_batch(
            roots[rows],
            initial[rows],
            target[rows],
            limits,
            0.5,
            collision,
            roots[rows, 0, 0],
            roots.new_zeros(len(rows), 3),
            lengths=lengths[rows],
            max_requests=37,
        )

    rows = torch.arange(3, device=device)
    p, q, diagnostics = solve(rows)
    assert diagnostics["valid"].tolist() == [True, True, False]
    assert max(requests) <= 37
    assert sum(requests) == int(lengths.sum()) * 2 * 24
    for row in range(3):
        sp, sq, sd = solve(rows[row : row + 1])
        torch.testing.assert_close(p[row : row + 1], sp)
        torch.testing.assert_close(q[row : row + 1], sq)
        for key in (
            "valid",
            "candidate_valid",
            "maximum_inverse_position_error_m",
            "samples",
            "selected_path_end_m",
            "minimum_clearance_m",
        ):
            torch.testing.assert_close(diagnostics[key][row : row + 1], sd[key])
    permutation = rows.flip(0)
    pp, pq, pd = solve(permutation)
    torch.testing.assert_close(pp, p[permutation])
    torch.testing.assert_close(pq, q[permutation])
    torch.testing.assert_close(pd["candidate_valid"], diagnostics["candidate_valid"][permutation])


def test_batch_gaze_does_not_serialize_diagnostics(monkeypatch, device):
    roots, initial, target, limits = inputs(device)
    collision, _ = install_reference_tools(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU gaze planner must not read tensors on the host")

    for name in ("item", "tolist", "cpu"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    p, q, diagnostics = gaze.plan_approach_gaze_batch(
        roots,
        initial,
        target,
        limits,
        0.5,
        collision,
        roots[:, 0, 0],
        roots.new_zeros(3, 3),
        max_requests=37,
    )
    assert p.shape == (3, 5, 3) and q.shape == (3, 5, 4)
    assert diagnostics["valid"].device == roots.device


def test_gaze_with_native_inverse_preserves_mixed_lengths_and_chunking():
    if not torch.cuda.is_available():
        pytest.skip("Native GPU gaze inverse")
    roots = torch.zeros(2, 3, 13, device="cuda")
    roots[..., 3], roots[..., 2] = 1, 0.536
    roots[1, :, 0] = 3
    initial = roots.new_tensor([[-1.57, 2.6, -1.5, 0.4, 0, 0]]).expand(2, -1)
    limits = roots.new_tensor([[-3.14, 3.14], [0, 2.96], [-2.87, 0], [-2, 2], [-2, 2], [-3, 3]])
    target = roots[:, 0, :3] + roots.new_tensor([1, 0.2, -0.2])
    lengths = torch.tensor([3, 2], device="cuda")
    collision = SimpleNamespace(clearance=lambda g, r, j: r.new_full(r.shape[:-1], 0.3))

    def solve(rows, count, max_requests):
        return gaze.plan_approach_gaze_batch(
            roots[rows, :count],
            initial[rows],
            target[rows],
            limits,
            1.0,
            collision,
            None,
            roots.new_zeros(len(rows), 3),
            lengths=lengths[rows],
            roll_step_deg=180,
            max_requests=max_requests,
        )

    p, q, diagnostics = solve([0, 1], 3, 211)
    for row, count in ((0, 3), (1, 2)):
        sp, sq, sd = solve([row], count, 65536)
        torch.testing.assert_close(p[row : row + 1, :count], sp, atol=1e-6, rtol=0)
        torch.testing.assert_close(q[row : row + 1, :count], sq, atol=1e-6, rtol=0)
        for key in (
            "valid",
            "candidate_valid",
            "maximum_inverse_position_error_m",
            "maximum_settled_position_error_m",
            "selected_path_end_m",
        ):
            torch.testing.assert_close(diagnostics[key][row : row + 1], sd[key], atol=1e-5, rtol=0)


def test_invalid_initial_station_does_not_add_dummy_wrist_route_cost(monkeypatch, device):
    roots, initial, target, limits = inputs(device)
    collision, _ = install_reference_tools(monkeypatch)
    generate = gaze.natural_gaze_candidates

    def candidates(*args):
        p, q, template, expanded, shoulder, rolls = generate(*args)
        template = template.clone().reshape(3, 2, 5, 6)
        template[:, 0, :, 5], template[:, 1, :, 5] = 1, -1
        return p, q, template.flatten(0, 1), expanded, shoulder, rolls

    monkeypatch.setattr(gaze, "natural_gaze_candidates", candidates)
    next_joints = initial.clone()
    next_joints[:, 5] = -1
    p, _, _ = gaze.plan_approach_gaze_batch(
        roots,
        initial,
        target,
        limits,
        0.5,
        collision,
        roots.new_zeros(3),
        roots.new_zeros(3, 3),
        next_grasp_joints=next_joints,
        next_grasp_mask=torch.tensor([False, True, False], device=device),
        max_requests=137,
    )
    torch.testing.assert_close(p[:, 0, 1], target[:, 1] + target.new_tensor([0, 0.1, 0]))
