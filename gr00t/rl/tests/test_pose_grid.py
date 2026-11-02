"""The shared GPU request grid keeps paths, body frames and limits aligned."""

import pytest
import torch

from gr00t.rl.auto_open.pose_planning import solve_pose_grid
from gr00t.rl.tests.test_pick_batched_grasp_inverse import fake_inverse

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU request grids")


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("limit_mode", ["shared", "seed", "point"])
def test_pose_grid_keeps_masks_body_frames_and_joint_limits_aligned(
    moving, limit_mode, monkeypatch
):
    p = torch.arange(18, device="cuda", dtype=torch.float32).reshape(2, 3, 3)
    q = p.new_tensor([1, 0, 0, 0]).expand(2, 3, 4)
    seeds = torch.arange(48, device="cuda", dtype=torch.float32).reshape(2, 2, 2, 6)
    root = p.new_zeros(2, 2, 2, 13)
    root[..., 3] = 1
    root[..., 0] = torch.arange(8, device="cuda").reshape(2, 2, 2)
    body = root[..., None, :].expand(-1, -1, -1, 3, -1).clone() if moving else root
    if moving:
        body[..., 2] = torch.arange(3, device="cuda") * 0.01
    mask = torch.tensor([[True, False, True], [False, True, True]], device="cuda")
    limits = p.new_tensor([[-3, 3]] * 6)
    if limit_mode == "seed":
        limits = limits.expand(2, 2, 2, 6, 2).clone()
        limits[..., 0] -= root[..., 0, None]
    elif limit_mode == "point":
        limits = limits.expand(2, 2, 2, 3, 6, 2).clone()
        limits[..., 0] -= torch.arange(3, device="cuda")[None, None, None, :, None]
    calls = []

    def solve(positions, orientations, **kwargs):
        calls.append(
            (
                positions.clone(),
                kwargs["base_position_world"].clone(),
                kwargs["joint_limits"].clone(),
            )
        )
        return fake_inverse(positions, orientations, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Grid dispatch must not read GPU tensor values on the host")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        result = solve_pose_grid(
            p, q, body, seeds, limits, query_mask=mask, max_requests=5, solver=solve
        )
    expected_mask = mask[None, None].expand(2, 2, -1, -1)
    torch.testing.assert_close(result.exact, expected_mask)
    expected = p[None, None].expand(2, 2, -1, -1, -1)
    torch.testing.assert_close(
        result.command_position_world[expected_mask], expected[expected_mask]
    )
    assert max(len(call[0]) for call in calls) <= 5
    expected_body = body if moving else body[..., None, :].expand(-1, -1, -1, 3, -1)
    torch.testing.assert_close(
        torch.cat([call[1] for call in calls]), expected_body[..., :3][expected_mask]
    )
    if limit_mode != "shared":
        expected_limits = (
            limits
            if limit_mode == "point"
            else limits[..., None, :, :].expand(-1, -1, -1, 3, -1, -1)
        )
        torch.testing.assert_close(
            torch.cat([call[2] for call in calls]), expected_limits[expected_mask]
        )


def test_empty_grid_mask_does_not_invoke_the_solver():
    p = torch.zeros(1, 2, 3, device="cuda")
    q = p.new_tensor([1, 0, 0, 0]).expand(1, 2, 4)
    root = p.new_zeros(1, 13)
    root[:, 3] = 1
    assert (
        solve_pose_grid(
            p,
            q,
            root,
            p.new_zeros(1, 6),
            p.new_zeros(6, 2),
            query_mask=torch.zeros(1, 2, device="cuda", dtype=torch.bool),
            solver=lambda *a, **kw: pytest.fail("Empty inverse request"),
        )
        is None
    )


def test_invalid_shared_limits_are_rejected_before_gpu_solver_dispatch():
    p = torch.zeros(1, 2, 3, device="cuda")
    q = p.new_tensor([1, 0, 0, 0]).expand(1, 2, 4)
    with pytest.raises(ValueError, match="limits"):
        solve_pose_grid(
            p,
            q,
            p.new_zeros(1, 13),
            p.new_zeros(1, 6),
            p.new_zeros(5, 2),
            solver=lambda *a, **kw: pytest.fail("Malformed inverse limits"),
        )
