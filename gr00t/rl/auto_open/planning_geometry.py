"""Shared device-only geometry for direct targets and candidate planning."""

import torch

from gr00t.rl.auto_open.motion_math import _quat_multiply


def _quat_rotate(quaternion, vector):
    q_vector, vector = torch.broadcast_tensors(quaternion[..., 1:], vector)
    cross = torch.cross(q_vector, vector, dim=-1)
    return vector + 2 * (quaternion[..., :1] * cross + torch.cross(q_vector, cross, dim=-1))


def _normalize(q):
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-12)


def _nlerp(start, end, fraction):
    end = torch.where((start * end).sum(-1, keepdim=True) < 0, -end, end)
    return _normalize(start + fraction[..., None] * (end - start))


def _slerp(start, end, fraction):
    start, end = _normalize(start), _normalize(end)
    end = torch.where((start * end).sum(-1, keepdim=True) < 0, -end, end)
    half_angle = torch.acos((start * end).sum(-1).clamp(-1.0, 1.0))
    sine = half_angle.sin().clamp_min(1.0e-12)
    result = (
        torch.sin((1 - fraction) * half_angle)[..., None] * start
        + torch.sin(fraction * half_angle)[..., None] * end
    ) / sine[..., None]
    return torch.where((half_angle < 5.0e-9)[..., None], start, result)


def align_vectors(source, target, base_quat):
    source, target = _normalize(source), _normalize(target)
    cross = torch.cross(source, target, dim=-1)
    dot = (source * target).sum(-1).clamp(-1.0, 1.0)
    x_axis = torch.zeros_like(source)
    x_axis[:, 0] = 1
    y_axis = torch.zeros_like(source)
    y_axis[:, 1] = 1
    axis = torch.cross(source, x_axis, dim=-1)
    axis = torch.where(
        (torch.linalg.vector_norm(axis, dim=-1) < 1.0e-5)[:, None],
        torch.cross(source, y_axis, dim=-1),
        axis,
    )
    opposite = torch.cat((torch.zeros_like(dot[:, None]), _normalize(axis)), -1)
    delta = _normalize(torch.cat(((1 + dot)[:, None], cross), -1))
    return _quat_multiply(torch.where((dot < -0.9999)[:, None], opposite, delta), base_quat)
