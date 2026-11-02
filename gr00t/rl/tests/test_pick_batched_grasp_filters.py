"""Grasp selection reductions preserve each environment's candidates and seeds."""

import math

import pytest
import torch

from gr00t.rl.pick.planning import (
    continuous_inverse_seed,
    executed_arm_path_clear,
    geometric_grasp_mask,
    inverse_bank,
    joint_fk_approach,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def test_geometric_filters_batch_matches_split_with_different_supports_and_exclusions(device):
    orientations = torch.tensor(
        [[[1, 0, 0, 0], [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]]] * 2,
        dtype=torch.float32,
        device=device,
    )
    quality = {
        "pocket_fit": torch.ones(2, 2, device=device, dtype=torch.bool),
        "antipodal_margin": torch.ones(2, 2, device=device),
        "environment_clearance_m": torch.ones(2, 2, device=device),
        "fixed_contact_local": torch.zeros(2, 2, 3, device=device),
        "moving_contact_local": torch.tensor([1.0, 0, 0], device=device).expand(2, 2, 3),
    }
    normals = torch.tensor([[0.0, 0, 1], [1.0, 0, 0]], device=device)
    excluded = orientations.clone()
    excluded_mask = torch.tensor([[True, False], [False, False]], device=device)
    mask = geometric_grasp_mask(
        orientations, quality, 0.002, normals, excluded, excluded_mask=excluded_mask
    )
    torch.testing.assert_close(mask, torch.tensor([[False, True], [False, True]], device=device))
    assert quality["support_pinch_compatible"].shape == (2, 2)
    for row in range(2):
        split = geometric_grasp_mask(
            orientations[row : row + 1],
            {key: value[row : row + 1] for key, value in quality.items()},
            0.002,
            normals[row : row + 1],
            excluded[row : row + 1],
            excluded_mask=excluded_mask[row : row + 1],
        )
        torch.testing.assert_close(mask[row : row + 1], split)
    empty = geometric_grasp_mask(orientations, quality, 0.002, normals, excluded[:, :0])
    torch.testing.assert_close(empty[0], torch.ones(2, device=device, dtype=torch.bool))


def test_inverse_seed_selection_is_independent_per_environment(device):
    shape = (2, 2, 3, 4, 5)
    position_error = torch.zeros(shape, device=device)
    orientation_error = torch.zeros_like(position_error)
    joints = torch.zeros(*shape, 6, device=device)
    joints[:, :, 1] = 1
    joints[:, :, 2] = 2
    actual = torch.tensor([[0.0] * 6, [2.0] * 6], device=device)
    position_error[1, :, :, 3] = 0.01
    position_error[1, :, 1, 3] = 0.001
    chosen = continuous_inverse_seed(position_error, orientation_error, joints, actual)
    assert chosen.shape == (2, 2, 1, 4, 5)
    assert (chosen[0] == 0).all()
    assert (chosen[1, :, :, :3] == 2).all()
    assert (chosen[1, :, :, 3] == 1).all()
    for row in range(2):
        split = continuous_inverse_seed(
            position_error[row], orientation_error[row], joints[row], actual[row]
        )
        torch.testing.assert_close(chosen[row], split)


def test_arm_clearance_reduces_postures_not_environment_rows(device):
    indices = torch.tensor([0, 100, 150, 200], device=device)
    options = torch.tensor([0.08, 0.16], device=device)
    selected = torch.tensor([[0.08, 0.16, 0.08], [0.16, 0.08, 0.16]], device=device)
    safe = torch.ones(2, 2, 3, 6, device=device, dtype=torch.bool)
    safe[0, :, :, 3] = False
    safe[1, 0, :, 3] = False
    clearance = torch.full((2, 3), 0.01, device=device)
    result, executed = executed_arm_path_clear(safe, indices, 151, options, selected, clearance)
    assert not result[0].any()
    assert result[1].all()
    for row in range(2):
        split, split_executed = executed_arm_path_clear(
            safe[row], indices, 151, options, selected[row], clearance[row]
        )
        torch.testing.assert_close(result[row], split)
        torch.testing.assert_close(executed[row], split_executed)


def test_joint_fk_transit_preserves_each_rows_root_height_and_finite_joint_travel(device):
    roots = torch.zeros(2, 13, device=device)
    roots[:, 3] = 1
    roots[0, :3] = torch.tensor([1.0, 2, 0.52], device=device)
    roots[1, :3] = torch.tensor([-2.0, 4, 0.3], device=device)
    initial = torch.tensor(
        [[-1.57, 1.75, -0.785, -0.93, 0, 2], [1.57, 1.57, -1.3, 1.1, 0, -2]],
        device=device,
    )
    endpoints = initial[:, None].expand(-1, 3, -1).clone()
    endpoints[:, :, 5] *= -1
    heights = torch.tensor([[0.18, 0.3, 0.52], [0.3, 0.38, 0.46]], device=device)
    path = joint_fk_approach(roots, initial, endpoints, heights, 21)
    for row in range(2):
        split = joint_fk_approach(
            roots[row : row + 1], initial[row], endpoints[row], heights[row], 21
        )
        for result, single in zip(path, split):
            torch.testing.assert_close(result[row], single)
    torch.testing.assert_close(path[2][:, :, 0], initial[:, None].expand(-1, 3, -1))
    torch.testing.assert_close(path[2][:, :, -1], endpoints)
    torch.testing.assert_close(path[3][:, :, -1, 2], heights)
    assert (path[2][..., 5].abs() <= 2).all()


def test_inverse_bank_submits_all_environment_candidate_paths_together(device, monkeypatch):
    from gr00t.rl.pick import planning

    p = torch.arange(2 * 3 * 4 * 3, device=device, dtype=torch.float32).reshape(2, 3, 4, 3)
    q = torch.zeros(2, 3, 4, 4, device=device)
    q[..., 0] = 1
    roots = torch.zeros(2, 3, 13, device=device)
    roots[..., 3] = 1
    roots[1, :, 0] = 32
    joints = torch.arange(2 * 3 * 6, device=device, dtype=torch.float32).reshape(2, 3, 6)
    calls = []

    def solve(positions, orientations, **kwargs):
        calls.append(positions.shape)
        return torch.cat(
            (positions, kwargs["base_position_world"], kwargs["initial_arm_joint_positions"]), -1
        )

    monkeypatch.setattr(planning, "solve_pose_targets", solve)
    result = inverse_bank(p, q, roots, joints, None).reshape(2, 3, 4, -1)
    assert calls == [torch.Size([24, 3])]
    for row in range(2):
        split = inverse_bank(p[row], q[row], roots[row], joints[row], None).reshape(3, 4, -1)
        torch.testing.assert_close(result[row], split)
    dense_roots = roots[:, :, None].expand(-1, -1, 4, -1)
    dense = inverse_bank(p, q, dense_roots, joints, None).reshape_as(result)
    torch.testing.assert_close(dense, result)
