"""Trajectory time laws never depend on another environment's path length."""

import pytest
import torch

from gr00t.rl.pick.planning import retime_grasp_approach, retime_grasp_approach_batch


@pytest.fixture(params=["cpu", "cuda"])
def paths(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p = torch.tensor(
        [
            [[0.0, 0, 0.4], [0.1, 0, 0.4], [0.2, 0, 0.4]],
            [[10.0, 1, 0.4], [10.5, 1, 0.4], [11.0, 1, 0.4]],
        ],
        device=request.param,
    )
    q = p.new_tensor([1.0, 0, 0, 0]).expand(2, 3, 4)
    target = p.new_full((2, 3), 100)
    return p, q, target


def test_batch_retiming_matches_split_without_tensor_host_reads(paths, monkeypatch):
    p, q, target = paths

    def forbidden(*args, **kwargs):
        raise AssertionError("Retiming must not read tensor values on the host")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        batch, valid = retime_grasp_approach_batch(p, q, target, 0.02, capacity=512)
    assert valid.all()
    assert batch[2][0] < batch[2][1]
    for row in range(2):
        split, accepted = retime_grasp_approach_batch(
            p[row : row + 1], q[row : row + 1], target[row : row + 1], 0.02, capacity=512
        )
        assert accepted.all()
        for result, single in zip(batch, split):
            torch.testing.assert_close(result[row : row + 1], single)
        count = int(batch[2][row])
        torch.testing.assert_close(batch[0][row, count - 1 :], p[row, -1:].expand(513 - count, -1))
        torch.testing.assert_close(batch[1][row, count - 1 :], q[row, -1:].expand(513 - count, -1))
        legacy_p, legacy_q = retime_grasp_approach(
            p[row : row + 1], q[row : row + 1], target[row : row + 1], 0.02
        )
        torch.testing.assert_close(batch[0][row : row + 1, :count], legacy_p)
        torch.testing.assert_close(batch[1][row : row + 1, :count], legacy_q)


def test_retiming_preserves_quintic_time_law_and_speed_limit(paths):
    p, q, target = paths
    path, valid = retime_grasp_approach_batch(p, q, target, 0.02, capacity=512)
    assert valid.all()
    for row in range(2):
        count = int(path[2][row])
        t = torch.linspace(0, 1, count, device=p.device)
        t = t.pow(3) * (10 + t * (-15 + 6 * t))
        expected = p[row, :1] + t[:, None] * (p[row, -1:] - p[row, :1])
        torch.testing.assert_close(path[0][row, :count], expected)
    assert (path[0].diff(dim=1).norm(dim=-1) / 0.02).max() < 0.201


def test_retiming_capacity_failure_is_per_environment(paths):
    p, q, target = paths
    path, valid = retime_grasp_approach_batch(p, q, target, 0.02, capacity=128)
    assert valid.tolist() == [True, False]
    assert path[2][1] == 128
    assert path[2][0] < 128


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_retiming_rejects_invalid_time_before_device_work(dt):
    p = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="positive"):
        retime_grasp_approach_batch(p, torch.zeros(1, 2, 4), p[:, -1], dt)
