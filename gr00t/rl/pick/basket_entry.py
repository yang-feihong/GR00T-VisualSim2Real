"""Use a verified approach prefix before refreshing the basket insertion target."""

import math

import torch

from gr00t.rl.auto_open.motion_math import _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_dot,
)


def transport_entry_orientation(grasp, previous_object, current_object):
    """Preserve the selected physical pinch when the supported object rotates."""
    inverse = previous_object.clone()
    inverse[..., 1:] *= -1
    return _quat_multiply(_quat_multiply(current_object, inverse), grasp)


def above_object_path_batch(path, destination):
    """Keep each verified approach prefix in the standard padded GPU path format."""
    positions, quats, lengths = path
    if destination.shape != positions[:, 0].shape or quats.shape != (*positions.shape[:2], 4):
        raise ValueError("Basket approach destinations must match the padded pose batch")
    rows = torch.arange(len(positions), device=positions.device)
    final_quat = quats[rows, lengths.long() - 1]
    time = torch.arange(positions.shape[1], device=positions.device)[None]
    distance = (positions - destination[:, None]).norm(dim=-1)
    angle = 2 * fixed_dot(quats, final_quat[:, None]).abs().clamp_max(1).acos()
    cost = (distance + 0.1 * angle).masked_fill(time >= lengths[:, None], torch.inf)
    index = cost.argmin(-1).clamp_min(1)
    index = torch.minimum(index, lengths.long() - 1)
    p = torch.where((time >= index[:, None])[..., None], destination[:, None], positions)
    q = torch.where((time >= index[:, None])[..., None], final_quat[:, None], quats)
    return (p, q, index + 1), final_quat


def above_object_path(path, report):
    positions, quats, _ = path
    candidate = report["inverse_candidate_indices"].index(report["selected"])
    standoff = report["pregrasp_standoff_m"][candidate]
    final_quat = quats[:, -1].clone()
    pitch = report.get("capture_axis_pitch_rad", 0.0)
    axis = _quat_rotate(
        final_quat, positions.new_tensor([[math.cos(pitch), 0.0, -math.sin(pitch)]])
    )
    destination = positions[:, -1] - standoff * axis
    distance = (positions - destination[:, None]).norm(dim=-1)
    angle = 2 * (quats * final_quat[:, None]).sum(-1).abs().clamp_max(1).acos()
    index = max(1, int((distance + 0.1 * angle)[0].argmin()))
    p, q = positions[:, : index + 1].clone(), quats[:, : index + 1].clone()
    p[:, -1], q[:, -1] = destination, final_quat
    lengths = torch.full((1,), len(p[0]), device=p.device, dtype=torch.long)
    return (p, q, lengths), final_quat
