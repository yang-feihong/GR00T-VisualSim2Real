"""Compact permitted orientations before expensive surface/depth contact queries."""

import math

import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.planning import grasp_candidate_bank
from gr00t.rl.pick.support_features import uniform_surface_support


def cone_grasp_candidate_bank(
    state,
    sizes,
    collision,
    geometry,
    support,
    support_mask,
    orientations,
    maximum_tool_tilt_deg,
    normal,
    *,
    collider_clearance_m,
    minimum_center_depth_m,
    depth_fractions=None,
):
    target = state["object_state"][:, :3]
    batch, width = orientations.shape[:2]
    axis = _quat_rotate(orientations, target.new_tensor([1, 0, 0]))
    allowed = axis[..., 2] <= -math.cos(math.radians(maximum_tool_tilt_deg))
    owner, index = torch.where(allowed)
    if not len(owner):
        owner = torch.arange(batch, device=target.device)
        index = torch.zeros_like(owner)
    if support_mask is None:
        support_mask = torch.ones(support.shape[:-1], device=support.device, dtype=torch.bool)
    if support.shape[-2] > 9984:
        support, support_mask = uniform_surface_support(support, support_mask, count=9984)
    packed = grasp_candidate_bank(
        {key: value[owner] for key, value in state.items()},
        target[owner],
        sizes[owner],
        collision,
        geometry.select(owner),
        support[owner],
        candidate_orientations=orientations[owner, index, None],
        collider_clearance_m=collider_clearance_m,
        support_normal_world=normal[owner],
        object_frame_fallback=False,
        retain_depth_candidates=True,
        minimum_center_depth_m=minimum_center_depth_m,
        support_mask=support_mask[owner],
        **(
            {"depth_fractions": depth_fractions[owner, index, None]}
            if depth_fractions is not None
            else {}
        ),
    )
    depths = packed[0].shape[1]
    slots = index[:, None] * depths + torch.arange(depths, device=target.device)[None]
    quality = {}
    for key, value in packed[2].items():
        fill = torch.nan if value.is_floating_point() else 0
        expanded = torch.full(
            (batch, width * depths, *value.shape[2:]), fill, device=value.device, dtype=value.dtype
        )
        expanded[owner[:, None], slots] = value
        quality[key] = expanded
    quality["pocket_fit"] &= allowed.repeat_interleave(depths, 1)
    goals = target.new_full((batch, width * depths, 3), torch.nan)
    goals[owner[:, None], slots] = packed[1]
    return orientations.repeat_interleave(depths, 1), goals, quality, width * depths, 0
