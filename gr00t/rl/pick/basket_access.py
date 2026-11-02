"""Device-side basket shift screening and future grasp-access checks."""

import math

import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.basket import BASKET_GRASP_CLEARANCE_M
from gr00t.rl.pick.basket_inverse import safe_inverse_paths_batch
from gr00t.rl.pick.grasp_candidates import cone_grasp_candidate_bank
from gr00t.rl.pick.planning import (
    grasp_approach_origin,
    grasp_pose_bank,
    object_frame_grasp_bank,
)
from gr00t.rl.pick.pocket import BASKET_MINIMUM_CENTER_DEPTH_M

MAX_ACCESS_POSES_PER_BLOCK = 4096


def slide_stability_batch(
    root, object_state, support, center, hits, normals, directions, friction, *, support_mask=None
):
    inverse = root[:, 3:7].clone()
    inverse[:, 1:] *= -1
    valid = (
        torch.ones(support.shape[:2], device=support.device, dtype=torch.bool)
        if support_mask is None
        else support_mask
    )
    support = torch.where(valid[..., None], support, 0)
    world = object_state[:, None, :3] + _quat_rotate(object_state[:, None, 3:7], support)
    points = _quat_rotate(inverse[:, None], world - root[:, None, :3])
    com = _quat_rotate(
        inverse, object_state[:, :3] + _quat_rotate(object_state[:, 3:7], center) - root[:, :3]
    )
    contact = _quat_rotate(inverse, hits - root[:, :3])
    force = -_quat_rotate(inverse, normals)
    push = _quat_rotate(inverse, directions)
    horizontal = force[:, :2].norm(dim=-1).clamp_min(1e-8)
    axis = force[:, :2] / horizontal[:, None]
    bottom = points[..., 2].masked_fill(~valid, torch.inf).amin(-1)
    supported = valid & (points[..., 2] <= bottom[:, None] + 0.001)
    lever = ((points[..., :2] - com[:, None, :2]) * axis[:, None]).sum(-1)
    radius = lever.masked_fill(~supported, -torch.inf).amax(-1)
    vertical_ratio = force[:, 2] / horizontal
    height = contact[:, 2] - bottom
    offset = ((contact[:, :2] - com[:, :2]) * axis).sum(-1)
    effective_height = height - vertical_ratio * offset
    reserve = radius - friction * effective_height
    safe = ((axis * push[:, :2]).sum(-1) >= 0.9) & (vertical_ratio.abs() <= 0.25)
    safe &= (effective_height >= 0) & (reserve >= 0.006) & valid.any(-1)
    return safe, reserve, radius, effective_height


def object_boxes_in_trunk_batch(state, sizes):
    poses = torch.cat((state["object_state"][:, None], state["distractor_states"]), 1)
    if sizes.shape != (*poses.shape[:2], 3):
        raise ValueError("Basket payload sizes must match the padded rigid states")
    signs = torch.cartesian_prod(*[sizes.new_tensor([-1, 1])] * 3)
    corners = _quat_rotate(poses[:, :, None, 3:7], signs[None, None] * sizes[:, :, None] / 2)
    corners += poses[:, :, None, :3]
    root = state["root_states"]
    inverse = root[:, 3:7].clone()
    inverse[:, 1:] *= -1
    local = _quat_rotate(inverse[:, None, None], corners - root[:, None, None, :3])
    low, high = local.amin(-2), local.amax(-2)
    return (low + high) / 2, (high - low) / 2


def bbox_space_batch(centers, half_extents, basket, other_centers, other_extents, other_mask):
    extra = (1,) * (centers.ndim - 2)
    extent = half_extents.reshape(len(centers), *extra, 2)
    wall = torch.minimum(
        centers - extent - basket.floor_lower[:2],
        basket.floor_upper[:2] - centers - extent,
    ).amin(-1)
    if other_centers.shape[1]:
        shape = (len(centers), *extra, other_centers.shape[1], 2)
        gap = (centers[..., None, :] - other_centers.reshape(shape)).abs()
        gap -= extent[..., None, :] + other_extents.reshape(shape)
        separation = gap.amax(-1).masked_fill(
            ~other_mask.reshape(len(centers), *extra, other_centers.shape[1]), torch.inf
        )
        wall = torch.minimum(wall, separation.amin(-1))
    return wall


def shifted_grasp_counts_batch(
    state,
    sizes,
    targets,
    collision,
    geometry,
    support_points,
    limits,
    *,
    support_mask=None,
    target_mask=None,
    candidate_orientations=None,
    return_candidate_bank=False,
):
    """Pool every eligible shifted target/pose; bound inverse work by pose count."""
    root, actual_target = state["root_states"], state["object_state"][:, :3]
    batch, count = targets.shape[:2]
    if return_candidate_bank and (count != 1 or target_mask is not None):
        raise ValueError("Candidate reuse requires one unmasked target per environment")
    mask = (
        torch.ones(batch, count, device=root.device, dtype=torch.bool)
        if target_mask is None
        else target_mask
    )
    owner, shifted = torch.where(mask)
    geometric_counts = torch.zeros(batch, count, device=root.device, dtype=torch.long)
    reachable_counts = torch.zeros_like(geometric_counts)
    if not len(owner):
        return reachable_counts, {"geometric_grasp_counts": geometric_counts}
    if candidate_orientations is None:
        orientation = object_frame_grasp_bank(state["object_state"][:, 3:7], sizes)
        heading = torch.atan2(actual_target[:, 1] - root[:, 1], actual_target[:, 0] - root[:, 0])
        _, natural, _ = grasp_pose_bank(
            actual_target,
            heading,
            sizes,
            grasp_approach_origin(root),
            reference_orientation=state["ee_orn"],
        )
        orientation = torch.cat((state["ee_orn"][:, None], natural, orientation), 1)
    else:
        orientation = candidate_orientations
    top_down = _quat_rotate(orientation, targets.new_tensor([1, 0, 0]))[..., 2]
    top_down = top_down <= -math.cos(math.radians(30))
    normal = _quat_rotate(root[:, 3:7], targets.new_tensor([0, 0, 1]))
    local = {key: value[owner] for key, value in state.items()}
    local["object_state"] = local["object_state"].clone()
    local["object_state"][:, :3] = targets[owner, shifted]
    orientation, goals, quality, _, _ = cone_grasp_candidate_bank(
        local,
        sizes[owner],
        collision,
        geometry.select(owner),
        support_points[owner],
        None if support_mask is None else support_mask[owner],
        orientation[owner],
        30.0,
        normal[owner],
        collider_clearance_m=BASKET_GRASP_CLEARANCE_M,
        minimum_center_depth_m=BASKET_MINIMUM_CENTER_DEPTH_M,
    )
    depth_count = goals.shape[1] // top_down.shape[1]
    feasible = quality["pocket_fit"] & (quality["antipodal_margin"] >= 0)
    feasible &= quality["support_pinch_compatible"] & (
        quality["environment_clearance_m"] >= BASKET_GRASP_CLEARANCE_M
    )
    feasible &= top_down[owner].repeat_interleave(depth_count, 1)
    geometric_counts[owner, shifted] = feasible.sum(-1)
    target_id, pose_id = torch.where(feasible)
    reachable = torch.zeros_like(feasible)
    for start in range(0, len(target_id), MAX_ACCESS_POSES_PER_BLOCK):
        target_rows = target_id[start : start + MAX_ACCESS_POSES_PER_BLOCK]
        pose_rows = pose_id[start : start + MAX_ACCESS_POSES_PER_BLOCK]
        source = owner[target_rows]
        q = orientation[target_rows, pose_rows]
        end = goals[target_rows, pose_rows]
        axis = _quat_rotate(q, q.new_tensor([math.cos(-math.pi / 6), 0, -math.sin(-math.pi / 6)]))
        standoffs = end.new_tensor([0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32])
        pre = end[:, None] - standoffs[None, :, None] * axis[:, None]
        p = torch.stack((pre, end[:, None].expand_as(pre)), 2)
        quats = q[:, None, None].expand(-1, len(standoffs), 2, -1)
        safe, _, _ = safe_inverse_paths_batch(
            {key: state[key][source] for key in ("root_states", "dof_pos")},
            p,
            quats,
            limits,
            collision,
            geometry.select(source),
            body_height_offsets=(-0.01, 0.0, 0.01),
            final_joint_margin_rad=math.radians(2),
            minimum_arm_clearance_m=BASKET_GRASP_CLEARANCE_M,
            maximum_arm_query_distance_m=BASKET_GRASP_CLEARANCE_M + 0.0001,
        )
        reachable[target_rows, pose_rows] = safe.any(-1)
    reachable_counts[owner, shifted] = reachable.sum(-1)
    report = {
        "geometric_grasp_counts": geometric_counts,
        "reachable_grasp_counts": reachable_counts,
        "reachability_definition": (
            "pregrasp_and_capture_at_measured_trunk_height_plus_minus_1cm; "
            "2mm_inverse_error_2deg_capture_joint_margin_6mm_arm_clearance"
        ),
    }
    if return_candidate_bank:
        report["candidate_bank"] = (orientation, goals, quality, goals.shape[1], 0)
    return reachable_counts, report
