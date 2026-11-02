"""One sparse GPU inverse submission for heterogeneous default grasp path banks."""

import math
from dataclasses import fields

import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate, _slerp
from gr00t.rl.auto_open.pose_planning import solve_pose_grid, solve_pose_targets
from gr00t.rl.pick.planning import (
    GRASP_LIFT_SPEED_M_S,
    GRASP_WITHDRAW_SPEED_M_S,
    continuous_inverse_seed,
    geometric_grasp_mask,
    grasp_body_height_hypotheses,
)
from gr00t.rl.pick.pocket import gripper_pocket_geometry

MAX_GRASP_INVERSE_REQUESTS = 65536


def _realized_capture_candidates(goals, orientations, roots, seeds, actual, limits, margin):
    """Check the same necessary capture predicate before solving the whole path."""
    result = solve_pose_grid(
        goals[:, None],
        orientations[:, None],
        roots[-3:],
        seeds[-3:],
        limits,
        max_requests=MAX_GRASP_INVERSE_REQUESTS,
        solver=solve_pose_targets,
    )
    position = result.position_error.norm(dim=-1)
    rotation = 2 * result.orientation_error.norm(dim=-1).clamp_max(1).asin()
    choice = continuous_inverse_seed(
        position, rotation, result.arm_joint_positions, actual[None, None, :, None]
    )
    joints = result.arm_joint_positions.gather(
        1, choice[..., None].expand(-1, -1, -1, -1, 6)
    ).squeeze(1)
    joint_margin = torch.minimum(joints - limits[:, 0], limits[:, 1] - joints).amin(-1)
    valid = result.exact.gather(1, choice).squeeze(1)
    valid &= position.gather(1, choice).squeeze(1) <= 0.002
    valid &= rotation.gather(1, choice).squeeze(1) <= 0.02
    valid &= joint_margin >= margin
    return valid.all(0).squeeze(-1)


def realized_grasp_candidate_mask(state, bank, limits, arm_joint_ids, required, margin):
    """Batch necessary capture tests once, before sequential candidate cohorts."""
    orientations, goals, quality, _, _ = bank
    available = quality["pocket_fit"].clone()
    owner, candidate = torch.where(available & required[:, None])
    if not len(owner):
        return available
    root = state["root_states"]
    actual = state["dof_pos"][owner][:, arm_joint_ids]
    initial = torch.stack((actual, limits.mean(-1).expand_as(actual), limits.sum(-1) - actual))
    bodies = root[owner][None, None].expand(3, 3, -1, -1).clone()
    bodies[..., 2] += root.new_tensor([-0.01, 0.0, 0.01])[:, None, None]
    available[owner, candidate] = _realized_capture_candidates(
        goals[owner, candidate],
        orientations[owner, candidate],
        bodies,
        initial[None].expand(3, -1, -1, -1),
        actual,
        limits,
        margin,
    )
    return available


def path_reference_indices(counts):
    """Sorted unique 24 uniform samples plus endpoints, padded independently per row."""
    endpoints = counts.cumsum(-1) - 1
    uniform = (
        ((endpoints[:, -1:] * torch.arange(24, device=counts.device)[None]).double() / 23)
        .floor()
        .long()
    )
    values = torch.cat((uniform, endpoints), -1).sort(-1).values
    first = torch.cat(
        (torch.ones_like(values[:, :1], dtype=torch.bool), values[:, 1:] != values[:, :-1]), -1
    )
    rank = first.long().cumsum(-1) - 1
    padded = endpoints[:, -1:].expand_as(values).clone()
    padded.scatter_reduce_(1, rank, values, reduce="amin", include_self=True)
    return padded, first.sum(-1)


def solve_default_grasp_inverse(
    state,
    candidates,
    limits,
    arm_joint_ids,
    initial_height,
    dt,
    *,
    excluded=None,
    excluded_mask=None,
    collider_clearance_m=0.002,
    support_normal_world=None,
    minimum_object_height_m=None,
    withdraw_distance_m=None,
    body_heights=None,
):
    """Preserve all geometric candidates, body hypotheses, seeds and path samples."""
    orientations, goals, quality, _, _ = candidates
    batch, width = orientations.shape[:2]
    target = state["object_state"][:, :3]
    root = state["root_states"]
    if body_heights is not None and (
        body_heights.ndim != 2
        or body_heights.shape[0] != len(root)
        or body_heights.shape[1] < 3
        or body_heights.device != root.device
        or body_heights.dtype != root.dtype
    ):
        raise ValueError("Body height hypotheses require matching device/dtype and (batch, >=3)")
    feasible = geometric_grasp_mask(
        orientations,
        quality,
        collider_clearance_m,
        support_normal_world,
        excluded,
        excluded_mask=excluded_mask,
    )
    owner, index = torch.where(feasible)
    lookup = torch.full((batch, width), -1, device=root.device, dtype=torch.long)
    lookup[owner, index] = torch.arange(len(owner), device=root.device)
    minimum = torch.maximum(initial_height + 0.10, target.new_full((batch,), 0.30))
    if minimum_object_height_m is not None:
        minimum = torch.maximum(
            minimum, torch.as_tensor(minimum_object_height_m, device=root.device)
        )
    withdraw_distance = torch.where(target[:, 2] < 0.12, 0.0, 0.16)
    if withdraw_distance_m is not None:
        withdraw_distance = torch.as_tensor(
            withdraw_distance_m, device=root.device, dtype=target.dtype
        ).expand(batch)
    if not len(owner):
        return {
            "result": None,
            "lookup": lookup,
            "minimum_object_height": minimum,
            "withdraw_distance_m": withdraw_distance,
        }
    goal, q = goals[owner, index], orientations[owner, index]
    pitch = -float(gripper_pocket_geometry()["open_angle"]) / 2
    axis = _quat_rotate(q, root.new_tensor([math.cos(pitch), 0, -math.sin(pitch)]))
    pregrasp = goal - 0.08 * axis
    lift_distance = (minimum - target[:, 2]).clamp_min(0) + 0.005
    lifted = goal.clone()
    lifted[:, 2] = (goal[:, 2] + lift_distance[owner]).clamp_min(0.305)
    maximum_lift = target.new_zeros(batch)
    maximum_lift.scatter_reduce_(0, owner, (lifted - goal).norm(dim=-1), reduce="amax")
    forward = target[:, :2] - root[:, :2]
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    withdrawn = lifted.clone()
    withdrawn[:, :2] -= withdraw_distance[owner, None] * forward[owner]
    lift_samples = (1.875 * maximum_lift.double() / (GRASP_LIFT_SPEED_M_S * dt)).ceil().long() + 1
    withdraw_samples = (
        1.875 * withdraw_distance.double() / (GRASP_WITHDRAW_SPEED_M_S * dt)
    ).ceil().long() + 1
    counts = torch.stack(
        (
            torch.full_like(lift_samples, 151),
            torch.full_like(lift_samples, 101),
            lift_samples.clamp_min(2),
            withdraw_samples.clamp_min(2),
        ),
        -1,
    )
    references, reference_lengths = path_reference_indices(counts)
    samples = references[owner]
    cumulative = counts.cumsum(-1)
    phase = (samples[..., None] >= cumulative[owner, None, :-1]).sum(-1)
    starts = torch.cat((torch.zeros_like(counts[:, :1]), cumulative[:, :-1]), -1)
    fraction = (
        (samples - starts[owner].gather(1, phase)) / (counts[owner].gather(1, phase) - 1)
    ).to(goal.dtype)
    fraction = torch.where(
        phase >= 2, fraction.pow(3) * (10 + fraction * (-15 + 6 * fraction)), fraction
    )
    knots = torch.stack((state["ee_pos"][owner], pregrasp, goal, lifted, withdrawn), 1)
    rotations = torch.cat((state["ee_orn"][owner, None], q[:, None].expand(-1, 4, -1)), 1)
    left = knots.gather(1, phase[..., None].expand(-1, -1, 3))
    right = knots.gather(1, (phase + 1)[..., None].expand(-1, -1, 3))
    p = left + fraction[..., None] * (right - left)
    start_q = rotations.gather(1, phase[..., None].expand(-1, -1, 4))
    end_q = rotations.gather(1, (phase + 1)[..., None].expand(-1, -1, 4))
    sample_q = _slerp(start_q, end_q, fraction)
    standoffs = root.new_tensor([0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32])
    capacity = references.shape[1] + len(standoffs)
    p = torch.cat((p, goal.new_zeros(len(goal), len(standoffs), 3)), 1)
    sample_q = torch.cat((sample_q, q[:, None].expand(-1, len(standoffs), -1)), 1)
    slots = reference_lengths[owner, None] + torch.arange(len(standoffs), device=root.device)[None]
    p.scatter_(
        1,
        slots[..., None].expand(-1, -1, 3),
        goal[:, None] - standoffs[None, :, None] * axis[:, None],
    )
    sample_q.scatter_(
        1, slots[..., None].expand(-1, -1, 4), q[:, None].expand(-1, len(standoffs), -1)
    )
    heights = grasp_body_height_hypotheses(root) if body_heights is None else body_heights
    actual = state["dof_pos"][:, arm_joint_ids]
    seeds = torch.stack((actual, limits.mean(-1).expand_as(actual), limits.sum(-1) - actual), 1)
    shape = (heights.shape[1], seeds.shape[1], len(owner))
    roots = root[owner][None, None].expand(*shape, -1).clone()
    roots[..., 2] = heights[owner].T[:, None]
    joints = seeds[owner].transpose(0, 1)[None].expand(*shape, -1)
    active = torch.arange(capacity, device=root.device)[None] < (
        reference_lengths[owner, None] + len(standoffs)
    )
    result = solve_pose_grid(
        p,
        sample_q,
        roots,
        joints,
        limits,
        query_mask=active,
        max_requests=MAX_GRASP_INVERSE_REQUESTS,
        solver=solve_pose_targets,
    )
    dense = {field.name: getattr(result, field.name) for field in fields(result)}
    return {
        "result": dense,
        "result_type": type(result),
        "body_heights": heights,
        "lookup": lookup,
        "reference_lengths": reference_lengths,
        "references": references,
        "counts": counts,
        "minimum_object_height": minimum,
        "withdraw_distance_m": withdraw_distance,
    }


def select_grasp_inverse(bank, row, candidate_indices, path_length):
    ids = bank["lookup"][row, candidate_indices]
    torch._assert_async((ids >= 0).all(), "Grasp inverse cache omitted an executed candidate")
    selected = {}
    for name, value in bank["result"].items():
        local = value[:, :, ids, :path_length]
        selected[name] = local.reshape(-1, *value.shape[4:])
    return bank["result_type"](**selected)
