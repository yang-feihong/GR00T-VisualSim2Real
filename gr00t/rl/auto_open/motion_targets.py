"""Shared GPU pose trajectory storage; independent progress per environment."""

import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_dot,
    fixed_normalize,
)


class PaddedPathBank:
    """Fixed-capacity device paths with independent per-env length/index."""

    def __init__(self, num_envs: int, capacity: int, device):
        self.capacity = int(capacity)
        self.positions = torch.zeros((num_envs, capacity, 3), dtype=torch.float32, device=device)
        self.orientations = torch.zeros((num_envs, capacity, 4), dtype=torch.float32, device=device)
        self.length = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.index = torch.zeros(num_envs, dtype=torch.int32, device=device)

    def clear(self, rows: torch.Tensor) -> None:
        self.length[rows] = 0
        self.index[rows] = 0

    def install_tensor(self, rows, positions, orientations, lengths) -> None:
        """Install padded device paths without scalar reads or per-row dispatch."""
        rows = torch.as_tensor(rows, device=self.positions.device, dtype=torch.int64)
        if positions.ndim != 3 or positions.shape[0] != len(rows) or positions.shape[2] != 3:
            raise ValueError("expected positions [B, L, 3]")
        width = positions.shape[1]
        if not 1 <= width <= self.capacity:
            raise ValueError(f"padded path width must be in [1, {self.capacity}]")
        if orientations.shape != (len(rows), width, 4) or lengths.shape != (len(rows),):
            raise ValueError("orientations/lengths disagree with padded path batch")
        torch._assert_async(
            ((lengths >= 1) & (lengths <= width)).all(), "path lengths are outside the padded path"
        )
        self.positions[rows, :width] = positions
        self.orientations[rows, :width] = orientations
        self.length[rows] = lengths.to(dtype=self.length.dtype)
        self.index[rows] = 0

    def current(self, rows: torch.Tensor, *, advance: bool) -> tuple[torch.Tensor, torch.Tensor]:
        length = self.length[rows]
        torch._assert_async((length > 0).all(), "stage entered without an installed path")
        index = torch.minimum(self.index[rows], length - 1).to(torch.int64)
        position = self.positions[rows, index]
        orientation = self.orientations[rows, index]
        if advance:
            self.index[rows] = torch.minimum(index + 1, length - 1).to(torch.int32)
        return position, orientation

    def complete(self, rows: torch.Tensor) -> torch.Tensor:
        return (self.length[rows] > 0) & (self.index[rows] >= self.length[rows] - 1)

    def sample_at(self, rows, index):
        """Sample a task-driven fractional index without a second controller."""
        index = torch.minimum(index, (self.length[rows] - 1).to(index.dtype))
        lower = index.floor().to(torch.int64)
        upper = torch.minimum(lower + 1, (self.length[rows] - 1).to(torch.int64))
        weight = (index - lower.to(index.dtype))[:, None]
        position = (1.0 - weight) * self.positions[rows, lower] + weight * self.positions[
            rows, upper
        ]
        q0, q1 = self.orientations[rows, lower], self.orientations[rows, upper]
        q1 = torch.where((fixed_dot(q0, q1) < 0.0)[:, None], -q1, q1)
        orientation = fixed_normalize((1.0 - weight) * q0 + weight * q1, eps=1e-8)
        return position, orientation
