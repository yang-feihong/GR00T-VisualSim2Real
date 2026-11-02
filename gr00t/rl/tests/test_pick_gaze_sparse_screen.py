import pytest
import torch

from gr00t.rl.pick import gaze
from gr00t.rl.tests.test_pick_batched_gaze import inputs, install_reference_tools


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("failure", ["inverse", "clearance", "joint_margin"])
def test_walking_gaze_cannot_hide_an_unsafe_body_height(monkeypatch, device, failure):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from gr00t.rl.tests.test_pick_batched_gaze import Result

    roots, initial, target, limits = inputs(device)
    roots = roots[:, :1].expand(-1, 33, -1).clone()
    collision, requests = install_reference_tools(monkeypatch)

    def inverse(p, q, **kwargs):
        requests.append(len(p))
        bad = kwargs["base_position_world"][:, 2] > 0.60
        error = torch.zeros_like(p)
        if failure == "inverse":
            error[:, 0] = torch.where(bad, 0.04, 0.0)
        joints = kwargs["initial_arm_joint_positions"].clone()
        if failure == "joint_margin":
            joints[:, 3] = torch.where(bad, limits[3, 1] - 0.001, joints[:, 3])
        return Result(joints, error, torch.zeros_like(p))

    def clearance(geometry, body, joints, **kwargs):
        unsafe = (body[..., 2] > 0.60) & (failure == "clearance")
        return torch.where(unsafe, 0.0, 0.03)

    monkeypatch.setattr(gaze, "solve_pose_targets", inverse)
    collision.clearance = clearance
    _, _, diagnostics = gaze.plan_approach_gaze_batch(
        roots,
        initial,
        target,
        limits,
        0.1,
        collision,
        roots[:, 0, 0],
        roots.new_zeros(3, 3),
        max_requests=37,
    )
    assert not diagnostics["valid"].any()
    assert max(requests) <= 37
    if failure == "joint_margin":
        assert (diagnostics["maximum_inverse_position_error_m"] == 0).all()
        assert (diagnostics["minimum_settled_joint_margin_rad"] < 0.002).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_screen_still_checks_every_published_path_point(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    roots, initial, target, limits = inputs(device)
    roots = roots[:, :1].expand(-1, 33, -1).clone()
    roots[..., 1] = torch.arange(33, device=device)
    collision, requests = install_reference_tools(monkeypatch)
    queries = []

    def clearance(geometry, body, joints, *, maximum_distance_m, query_mask=None):
        queries.append(body.shape[2])
        return torch.where(body[..., 1] == 6, 0.0, maximum_distance_m)

    collision.clearance = clearance
    p, q, diagnostics = gaze.plan_approach_gaze_batch(
        roots,
        initial,
        target,
        limits,
        0.02,
        collision,
        roots[:, 0, 0],
        roots.new_zeros(3, 3),
        max_requests=37,
    )
    assert queries == [9, 33]
    assert not (diagnostics["coarse_anchor_indices"] == 6).any()
    assert not diagnostics["valid"].any()
    assert p.shape == (3, 33, 3) and q.shape == (3, 33, 4)
    assert max(requests) <= 37


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_gaze_mixed_lengths_are_split_independent(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    roots, initial, target, limits = inputs(device)
    roots = roots[:, :1].expand(-1, 33, -1).clone()
    collision, _ = install_reference_tools(monkeypatch)
    collision.clearance = lambda g, r, j, **kw: r.new_full(r.shape[:-1], 0.03)
    lengths = torch.tensor([33, 22, 15], device=device)

    def solve(rows):
        return gaze.plan_approach_gaze_batch(
            roots[rows],
            initial[rows],
            target[rows],
            limits,
            0.02,
            collision,
            roots[rows, 0, 0],
            roots.new_zeros(len(rows), 3),
            lengths=lengths[rows],
            max_requests=37,
        )

    result = solve([0, 1, 2])
    for row in range(3):
        one = solve([row])
        torch.testing.assert_close(one[0], result[0][row : row + 1])
        torch.testing.assert_close(one[1], result[1][row : row + 1])
        torch.testing.assert_close(one[2]["valid"], result[2]["valid"][row : row + 1])


def test_sparse_gaze_with_real_gpu_inverse_preserves_chunking_and_split():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from types import SimpleNamespace

    roots = torch.zeros(2, 33, 13, device="cuda")
    roots[..., 3], roots[..., 2] = 1, 0.536
    roots[1, :, 0] = 3
    initial = roots.new_tensor([[-1.57, 2.6, -1.5, 0.4, 0, 0]]).expand(2, -1)
    limits = roots.new_tensor([[-3.14, 3.14], [0, 2.96], [-2.87, 0], [-2, 2], [-2, 2], [-3, 3]])
    target = roots[:, 0, :3] + roots.new_tensor([1, 0.2, -0.2])
    lengths = torch.tensor([33, 22], device="cuda")
    collision = SimpleNamespace(clearance=lambda g, r, j, **kw: r.new_full(r.shape[:-1], 0.3))

    def solve(rows, chunk):
        return gaze.plan_approach_gaze_batch(
            roots[rows],
            initial[rows],
            target[rows],
            limits,
            0.02,
            collision,
            None,
            roots.new_zeros(len(rows), 3),
            lengths=lengths[rows],
            roll_step_deg=180,
            max_requests=chunk,
        )

    p, q, diagnostics = solve([0, 1], 211)
    assert torch.isfinite(p).all() and torch.isfinite(q).all()
    for row in range(2):
        one = solve([row], 65536)
        torch.testing.assert_close(one[0], p[row : row + 1])
        torch.testing.assert_close(one[1], q[row : row + 1])
        torch.testing.assert_close(one[2]["valid"], diagnostics["valid"][row : row + 1])
