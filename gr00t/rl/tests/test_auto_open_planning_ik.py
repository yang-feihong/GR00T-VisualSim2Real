import time

import pytest
import torch

from gr00t.rl.auto_open.planning_ik import _CandidateIKGraph, solve_candidate_ik
from gr00t.rl.envs.door.b2z1_reset_geometry import (
    b2z1_gripper_pose_and_jacobian,
    solve_b2z1_arm_ik,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [64, 81, 861])
def test_candidate_graph_replays_identical_math_and_refreshes_inputs(dtype, count):
    root = torch.tensor([[16.0, -24.0, 0.55]], dtype=dtype, device="cuda").expand(count, -1)
    quat = torch.tensor([[0.02, -0.01, 0.15, 0.9884]], dtype=dtype, device="cuda").expand(count, -1)
    initial = torch.tensor([[0.0, 1.2, -1.8, -1.0, 0.0, 0.0]], dtype=dtype, device="cuda").expand(
        count, -1
    )
    lower = torch.full((6,), -3.0, dtype=dtype, device="cuda")
    upper = -lower
    prior = None
    for offset in (0.04, -0.06):
        target = initial + offset
        position, rotation, _ = b2z1_gripper_pose_and_jacobian(root, quat, target)
        inputs = (root, quat, position, rotation, initial, lower, upper)
        start = time.perf_counter()
        reference = solve_b2z1_arm_ik(*inputs, iterations=120)
        torch.cuda.synchronize()
        reference_s = time.perf_counter() - start
        actual = solve_candidate_ik(*inputs, iterations=120)
        torch.cuda.synchronize()
        start = time.perf_counter()
        replay = solve_candidate_ik(*inputs, iterations=120)
        torch.cuda.synchronize()
        replay_s = time.perf_counter() - start
        print({"dtype": str(dtype), "reference_s": reference_s, "replay_s": replay_s})
        for expected, result in zip(reference, actual, strict=True):
            torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)
        for expected, result in zip(actual, replay, strict=True):
            torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)
        if prior is not None:
            assert not torch.equal(prior, actual[0])
        prior = actual[0].clone()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_candidate_graph_error_guard_needs_no_host_scalar_read(monkeypatch):
    from types import SimpleNamespace

    entry = _CandidateIKGraph.__new__(_CandidateIKGraph)
    value = torch.ones(1, device="cuda")
    entry.inputs, entry.outputs = (value.clone(),), (value.clone(),)
    entry.status = torch.zeros((), device="cuda", dtype=torch.bool)
    entry.done = torch.cuda.Event()
    entry.done.record()
    entry.graph = SimpleNamespace(replay=lambda: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Candidate IK replay must not read a device scalar")

    with monkeypatch.context() as scope:
        scope.setattr(torch.Tensor, "item", forbidden)
        result = entry.replay((value,))
    torch.cuda.synchronize()
    torch.testing.assert_close(result[0], value)
