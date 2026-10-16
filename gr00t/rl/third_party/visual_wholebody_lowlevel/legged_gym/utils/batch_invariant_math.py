"""Batch-shape-invariant tensor math that preserves scalar baseline semantics.

CUDA's native reductions are already batch-shape invariant for the vector widths
used by the controller except for four-component quaternion expressions.  The
four-wide case uses the same paired accumulation as the native single-row
kernel.  Other widths retain the native implementation so this abstraction does
not perturb the validated single-environment trajectory.
"""

from __future__ import annotations

import torch


def fixed_sum(value: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Sum one dimension without changing established scalar numerics."""
    dim = dim if dim >= 0 else value.ndim + dim
    if dim < 0 or dim >= value.ndim:
        raise IndexError(f"dimension {dim} out of range for shape {tuple(value.shape)}")
    if value.shape[dim] != 4 or value.dtype == torch.bool:
        return torch.sum(value, dim=dim, keepdim=keepdim)
    result = (value.select(dim, 0) + value.select(dim, 1)) + (
        value.select(dim, 2) + value.select(dim, 3)
    )
    return result.unsqueeze(dim) if keepdim else result


def fixed_mean(value: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Mean one dimension using :func:`fixed_sum` accumulation order."""
    if value.shape[dim] == 0:
        raise ValueError("fixed_mean does not support an empty reduction dimension")
    return fixed_sum(value, dim=dim, keepdim=keepdim) / value.shape[dim]


def fixed_dot(
    left: torch.Tensor,
    right: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Dot product with fixed accumulation order along ``dim``."""
    return fixed_sum(left * right, dim=dim, keepdim=keepdim)


def fixed_norm(
    value: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Euclidean norm preserving native single-row semantics."""
    dim = dim if dim >= 0 else value.ndim + dim
    if value.shape[dim] != 4:
        return torch.linalg.vector_norm(value, dim=dim, keepdim=keepdim)
    return torch.sqrt(fixed_sum(value.square(), dim=dim, keepdim=keepdim))


def fixed_normalize(
    value: torch.Tensor,
    dim: int = -1,
    eps: float = 1.0e-9,
) -> torch.Tensor:
    """Normalize vectors using fixed-order norm semantics."""
    return value / fixed_norm(value, dim=dim, keepdim=True).clamp_min(eps)
