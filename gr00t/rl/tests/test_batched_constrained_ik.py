import numpy as np
import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.constrained_ik import (
    batched_constrained_dls_delta,
)


def _scalar_delta(jacobian, pose_error, joint_position, joint_limits, damping):
    lower = joint_limits[:, 0] + 1.0e-3 - joint_position
    upper = joint_limits[:, 1] - 1.0e-3 - joint_position
    delta = np.zeros_like(joint_position)
    free = np.ones(joint_position.shape[0], dtype=bool)
    for _ in range(joint_position.shape[0] + 1):
        if not np.any(free):
            break
        residual = pose_error - jacobian[:, ~free] @ delta[~free]
        jacobian_free = jacobian[:, free]
        normal = jacobian_free.T @ jacobian_free
        normal.flat[:: normal.shape[0] + 1] += damping**2
        candidate = np.linalg.solve(normal, jacobian_free.T @ residual)
        indices = np.flatnonzero(free)
        below = candidate < lower[free]
        above = candidate > upper[free]
        if not np.any(below | above):
            delta[free] = candidate
            break
        violating = indices[below | above]
        values = candidate[below | above]
        delta[violating] = np.where(
            values < lower[violating], lower[violating], upper[violating]
        )
        free[violating] = False
    return delta


def test_batched_constrained_dls_matches_scalar_active_set():
    rng = np.random.default_rng(1700)
    batch_size = 512
    jacobian = rng.normal(size=(batch_size, 6, 6))
    pose_error = rng.normal(scale=0.8, size=(batch_size, 6))
    limits = np.column_stack((-np.ones(6), np.ones(6)))
    positions = rng.uniform(-0.95, 0.95, size=(batch_size, 6))
    expected = np.stack(
        [
            _scalar_delta(jacobian[i], pose_error[i], positions[i], limits, 0.05)
            for i in range(batch_size)
        ]
    )
    actual = batched_constrained_dls_delta(
        torch.as_tensor(jacobian, dtype=torch.float64),
        torch.as_tensor(pose_error, dtype=torch.float64),
        torch.as_tensor(positions, dtype=torch.float64),
        torch.as_tensor(limits, dtype=torch.float64),
        0.05,
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1.0e-10, atol=1.0e-10)


def test_batch_size_one_uses_identical_tensor_semantics():
    rng = np.random.default_rng(1701)
    jacobian = torch.as_tensor(rng.normal(size=(1, 6, 6)), dtype=torch.float64)
    pose_error = torch.as_tensor(rng.normal(size=(1, 6)), dtype=torch.float64)
    positions = torch.as_tensor(rng.uniform(-0.9, 0.9, size=(1, 6)), dtype=torch.float64)
    limits = torch.tensor([[-1.0, 1.0]] * 6, dtype=torch.float64)

    single = batched_constrained_dls_delta(
        jacobian, pose_error, positions, limits, 0.05
    )
    repeated = batched_constrained_dls_delta(
        jacobian.repeat(7, 1, 1),
        pose_error.repeat(7, 1),
        positions.repeat(7, 1),
        limits,
        0.05,
    )
    torch.testing.assert_close(single[0], repeated[0], rtol=0.0, atol=0.0)
