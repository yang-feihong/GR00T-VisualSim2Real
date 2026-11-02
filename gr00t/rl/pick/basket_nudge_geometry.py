"""Necessary basket sliding geometry before any arm-path search."""

import math

import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.basket_access import (
    bbox_space_batch,
    object_boxes_in_trunk_batch,
    slide_stability_batch,
)
from gr00t.rl.pick.basket_nudge import improving_shift_mask, nudge_shift_bank
from gr00t.rl.pick.pocket import GRIP_STATIC_FRICTION


def closed_nudge_geometry_precheck(
    state,
    sizes,
    payload_mask,
    basket,
    target_surface,
    *,
    support_points,
    support_mask,
    object_centers_local=None,
    contact_height_fractions=(0.35, 0.70, 0.90),
    shift_samples=24,
    require_sliding_stability=False,
):
    if not contact_height_fractions or any(
        not math.isfinite(value) or abs(value) >= 1 for value in contact_height_fractions
    ):
        raise ValueError("Contact heights must be nonempty finite fractions inside (-1, 1)")
    root = state["root_states"]
    batch = len(root)
    if require_sliding_stability and (
        object_centers_local is None or object_centers_local.shape != (batch, 3)
    ):
        raise ValueError("Sliding checks require each object's local center of mass")
    centers, extents = object_boxes_in_trunk_batch(state, sizes)

    def space_at(points):
        return bbox_space_batch(
            points,
            extents[:, 0, :2],
            basket,
            centers[:, 1:, :2],
            extents[:, 1:, :2],
            payload_mask[:, 1:],
        )

    local_d, shifts = nudge_shift_bank(root, shift_samples)
    before = space_at(centers[:, 0, :2])
    after = space_at(centers[:, :1, :2] + local_d[None] * shifts[None, :, None])
    sweep = centers[:, None, :1, :2] + (
        torch.linspace(0, 1, 9, device=root.device)[None, None, :, None]
        * local_d[None, :, None]
        * shifts[None, :, None, None]
    )
    sweep_space = space_at(sweep).amin(-1)
    good = (after >= 0.006) & (
        sweep_space >= torch.minimum(before[:, None], before.new_tensor(-0.001)) - 1e-6
    )
    improving = improving_shift_mask(before[:, None], after, sweep_space)
    direction = _quat_rotate(
        root[:, None, 3:7], torch.cat((local_d, local_d.new_zeros(len(local_d), 1)), -1)[None]
    )
    bbox_good = good.clone()
    sliding_precheck = None
    probe_count = 0
    if require_sliding_stability:
        target = state["object_state"][:, :3]
        height_count = len(contact_height_fractions)
        probe_owner, probe_shift = torch.where(good)
        probe_owner = probe_owner.repeat_interleave(height_count)
        probe_shift = probe_shift.repeat_interleave(height_count)
        probe_count = len(probe_owner)
        probe_height = torch.arange(height_count, device=root.device).repeat(
            probe_count // height_count
        )
        probe_direction = direction[probe_owner, probe_shift]
        origins = target[probe_owner] - 0.20 * probe_direction
        origins[:, 2] = (
            target[probe_owner, 2]
            + sizes[probe_owner, 0, 2] / 2 * root.new_tensor(contact_height_fractions)[probe_height]
        )
        probe_hits, probe_normals, _ = target_surface.select(probe_owner).rays(
            origins[:, None], (origins + 0.40 * probe_direction)[:, None]
        )
        friction = (GRIP_STATIC_FRICTION + basket.floor_static_friction) / 2
        values = slide_stability_batch(
            root[probe_owner],
            state["object_state"][probe_owner],
            support_points[probe_owner],
            object_centers_local[probe_owner],
            probe_hits[:, 0],
            probe_normals[:, 0],
            probe_direction,
            friction,
            support_mask=support_mask[probe_owner],
        )
        slot = (probe_owner * len(shifts) + probe_shift) * height_count + probe_height
        sliding_precheck = []
        for value in values:
            field = torch.full(
                (batch * len(shifts) * height_count,),
                False if value.dtype == torch.bool else torch.nan,
                device=root.device,
                dtype=value.dtype,
            )
            field[slot] = value
            sliding_precheck.append(field)
        good &= sliding_precheck[0].reshape(batch, len(shifts), height_count).any(-1)
    return {
        "shifts": shifts,
        "before": before,
        "after": after,
        "sweep_space": sweep_space,
        "good": good,
        "bbox_good": bbox_good,
        "improving": improving,
        "direction": direction,
        "sliding_precheck": sliding_precheck,
        "sliding_probe_count": probe_count,
    }
