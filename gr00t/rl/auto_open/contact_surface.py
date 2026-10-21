"""Classify release contacts using the actual collision-mesh surface."""

import torch

from gr00t.rl.auto_open.geometry import _quat_rotate_points_torch
from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_dot,
)


def separating_rear_contact(
    points_world, body_pose_world, triangles_body, shaft_axis_world, retreat_direction_world
):
    """Allow only axial rear faces that move away from the handle on retreat.

    At a shared mesh edge, every equally near face must allow separation.
    Upper/lower clamping faces and front-facing hooks remain blocking.
    """
    if triangles_body.numel() == 0:
        return torch.zeros(len(points_world), dtype=torch.bool, device=points_world.device)
    conjugate = body_pose_world[:, 3:7].clone()
    conjugate[:, 1:] *= -1
    point = _quat_rotate_points_torch(
        conjugate, (points_world - body_pose_world[:, :3])[:, None, :]
    )
    shaft = _quat_rotate_points_torch(conjugate, shaft_axis_world[:, None, :])
    retreat = _quat_rotate_points_torch(conjugate, retreat_direction_world[:, None, :])
    a, b, c = triangles_body.unbind(dim=1)
    ab, ac = b - a, c - a
    normal = torch.cross(ab, ac, dim=-1)
    normal_sq = fixed_dot(normal, normal).clamp_min(1e-20)
    ap = point - a[None, :, :]
    signed = fixed_dot(ap, normal[None, :, :])
    projected = ap - (signed / normal_sq)[..., None] * normal
    d00, d01, d11 = fixed_dot(ab, ab), fixed_dot(ab, ac), fixed_dot(ac, ac)
    d20, d21 = fixed_dot(projected, ab), fixed_dot(projected, ac)
    denominator = (d00 * d11 - d01 * d01).clamp_min(1e-20)
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    distance_sq = torch.where(inside, signed.square() / normal_sq, torch.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        fraction = (fixed_dot(point - start, edge) / fixed_dot(edge, edge).clamp_min(1e-20)).clamp(0, 1)
        delta = point - (start + fraction[..., None] * edge)
        distance_sq = torch.minimum(distance_sq, fixed_dot(delta, delta))
    distance = torch.sqrt(distance_sq.clamp_min(0))
    nearest = distance.amin(dim=1, keepdim=True)
    near = distance <= nearest + 1e-6
    normal = normal / torch.sqrt(normal_sq)[:, None]
    axial = fixed_dot(normal, shaft)
    transverse_sq = (1 - axial.square()).clamp_min(0)
    rear_face = (axial > 0) & (axial.square() > transverse_sq)
    separates = fixed_dot(normal, retreat) < 0
    allowed = torch.all(~near | (rear_face & separates), dim=1)
    return allowed & torch.isfinite(points_world).all(dim=-1)
