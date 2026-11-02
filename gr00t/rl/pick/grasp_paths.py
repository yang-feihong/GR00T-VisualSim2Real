"""Device-side scoring and execution paths for the default external grasp bank."""

import math

import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.auto_open.pose_planning import best_feasible
from gr00t.rl.pick.grasp_clearance import GRASP_CLEARANCE_M
from gr00t.rl.pick.planning import (
    continuous_inverse_seed,
    elevated_cartesian_approach,
    grasp_approach_origin,
    grasp_body_height_hypotheses,
    joint_fk_approach,
    retime_grasp_approach_batch,
)
from gr00t.rl.pick.pocket import gripper_pocket_geometry


def score_default_grasp_paths(
    state,
    candidates,
    inverse,
    limits,
    collision,
    geometry,
    dt,
    *,
    prefer_realized_body,
    capacity=4096,
    elevated_transit=False,
    prefer_orientation=True,
    collider_clearance_m=GRASP_CLEARANCE_M,
    require_safe_arm_path=False,
    validate_carry_preview=True,
    realized_body=None,
    wrist_motion_weight=0.02,
    minimum_realized_grasp_margin_rad=0.0,
):
    """Return standard padded EE paths, validity and tensor-only candidate diagnostics."""
    orientations, goals, quality, _, _ = candidates
    batch, width = orientations.shape[:2]
    root, target = state["root_states"], state["object_state"][:, :3]
    owner, candidate = torch.where(inverse["lookup"] >= 0)
    cost_bank = target.new_full((batch, width), torch.inf)
    feasible_bank = torch.zeros(batch, width, device=target.device, dtype=torch.bool)
    diagnostics = {
        "candidate_cost": cost_bank,
        "feasible": feasible_bank,
        "endpoint_inverse_available": torch.zeros_like(feasible_bank),
        "preferred_orientation_preservable": torch.zeros_like(feasible_bank),
        "candidate_orientations_world_wxyz": orientations,
        "withdraw_displacement_m": target.new_zeros(batch, 3),
        "candidate_grasp_origins_world_m": goals,
        "minimum_pregrasp_standoff_m": 0.08,
        "retain_depth_candidates": False,
        "pocket_depth_preference": 0.0,
        "preferred_grasp_orientation_supplied": prefer_orientation,
        "inverse_wrist_band_rad": None,
        "published_pose": "original_world_target_not_fixed_base_projection",
        "required_object_lift_m": 0.10,
        "minimum_object_height_world_m": inverse["minimum_object_height"],
        "dt": dt,
        "clearance_query_cap_m": collider_clearance_m + 0.0001,
        "approach_definition": "elevated_cartesian_lift_turn_descend_experiment"
        if elevated_transit
        else "joint_limit_respecting_fk_arc_published_as_ee",
    }
    diagnostics.update({f"convex_contact_model.{name}": value for name, value in quality.items()})
    if not len(owner):
        p = state["ee_pos"][:, None]
        q = state["ee_orn"][:, None]
        diagnostics["selected"] = torch.full((batch,), -1, device=root.device, dtype=torch.long)
        return (
            (p, q, torch.ones(batch, device=root.device, dtype=torch.long)),
            feasible_bank.any(-1),
            diagnostics,
        )
    count = len(owner)
    q, goal = orientations[owner, candidate], goals[owner, candidate]
    raw = inverse["result"]
    position_error = raw["position_error"].norm(dim=-1)
    orientation_error = 2 * raw["orientation_error"].norm(dim=-1).clamp_max(1).asin()
    actual = state["dof_pos"][owner][:, collision.arm_joint_ids]
    seed = continuous_inverse_seed(
        position_error, orientation_error, raw["arm_joint_positions"], actual[None, None, :, None]
    )
    position_error = position_error.gather(1, seed).squeeze(1)
    orientation_error = orientation_error.gather(1, seed).squeeze(1)
    exact = raw["exact"].gather(1, seed).squeeze(1)
    joints = (
        raw["arm_joint_positions"].gather(1, seed[..., None].expand(-1, -1, -1, -1, 6)).squeeze(1)
    )
    heights = (
        inverse["body_heights"] if "body_heights" in inverse else grasp_body_height_hypotheses(root)
    )[owner].T
    references = inverse["references"][owner]
    reference_length = inverse["reference_lengths"][owner]
    counts = inverse["counts"][owner]
    endpoints = counts.cumsum(-1) - 1
    endpoint_slots = torch.searchsorted(references.contiguous(), endpoints.contiguous())
    standoffs = root.new_tensor([0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32])
    slots = reference_length[:, None] + torch.arange(len(standoffs), device=root.device)[None]
    approach_exact = exact.gather(2, slots[None].expand(len(heights), -1, -1))
    approach_joints = joints.gather(2, slots[None, ..., None].expand(len(heights), -1, -1, 6))
    pitch = -float(gripper_pocket_geometry()["open_angle"]) / 2
    axes = _quat_rotate(q, root.new_tensor([math.cos(pitch), 0, -math.sin(pitch)]))
    packed_geometry = geometry.select(owner)
    entry_distances = torch.arange(33, device=root.device, dtype=root.dtype) * 0.01
    entry_gap = collision.gripper_clearance(
        packed_geometry,
        goal[:, None] - entry_distances[None, :, None] * axes[:, None],
        q[:, None].expand(-1, 33, -1),
        maximum_distance_m=collider_clearance_m + 0.0001,
    )
    entry_slots = (standoffs / 0.01).round().long()
    insertion_clearance = entry_gap.cummin(-1).values[:, entry_slots]
    approach_exact &= (insertion_clearance >= collider_clearance_m)[None]
    choice_cost = 0.001 * (heights[:, :, None] - root[owner, 2][None, :, None]).abs()
    choice_cost = choice_cost + standoffs[None, None]
    choice_cost = choice_cost + 0.00001 * (approach_joints - actual[None, :, None]).norm(dim=-1)
    choice = (
        choice_cost.masked_fill(~approach_exact, torch.inf).permute(0, 2, 1).flatten(0, 1).argmin(0)
    )
    residual = position_error + 0.1 * orientation_error
    fallback = residual.gather(2, slots[None].expand(len(heights), -1, -1))
    fallback = fallback.permute(0, 2, 1).flatten(0, 1).argmin(0)
    choice = torch.where(approach_exact.any(0).any(-1), choice, fallback)
    h, s = choice // len(standoffs), choice % len(standoffs)
    rows = torch.arange(count, device=root.device)
    end_joints, approach_height = approach_joints[h, rows, s], heights[h, rows]
    selected_standoff = standoffs[s]
    pregrasp = goal - selected_standoff[:, None] * axes
    transit_valid = torch.ones(count, device=root.device, dtype=torch.bool)
    if elevated_transit:
        approach_p, approach_q, transit_joints, transit_roots, transit_valid = (
            elevated_cartesian_approach(
                {key: state[key][owner] for key in ("root_states", "ee_pos", "ee_orn", "dof_pos")},
                pregrasp,
                q,
                approach_height,
                limits,
                collision,
                packed_geometry,
                151,
            )
        )
    else:
        approach_p, approach_q, transit_joints, transit_roots = joint_fk_approach(
            root[owner], actual, end_joints, approach_height, 151
        )
    approach_p[:, -1], approach_q[:, -1] = pregrasp, q
    approach_clearance = collision.clearance(
        packed_geometry,
        transit_roots,
        transit_joints,
        include_gripper=False,
        maximum_distance_m=collider_clearance_m + 0.0001,
    ).amin(-1)
    k = torch.arange(exact.shape[-1], device=root.device)[None]
    request_valid = k < reference_length[:, None] + len(standoffs)
    envelope = root[owner, None, None].expand(-1, len(heights), exact.shape[-1], -1).clone()
    envelope[..., 2] = heights.T[:, :, None]
    safe_joints = joints.permute(1, 0, 2, 3).masked_fill(~request_valid[:, None, :, None], 0)
    clearance = collision.clearance(
        packed_geometry,
        envelope,
        safe_joints,
        include_gripper=False,
        query_mask=request_valid[:, None].expand(-1, len(heights), -1),
        maximum_distance_m=collider_clearance_m + 0.0001,
    ).transpose(0, 1)
    endpoint_support = exact.gather(2, endpoint_slots[None].expand(len(heights), -1, -1)) & (
        clearance.gather(2, endpoint_slots[None].expand(len(heights), -1, -1))
        >= collider_clearance_m
    )
    endpoint_inverse_available = endpoint_support.any(0).all(-1)
    reference_indices = torch.cat(
        (references, torch.zeros(count, len(standoffs), device=root.device, dtype=torch.long)), -1
    )
    is_reference = k < reference_length[:, None]
    stand_index = (k - reference_length[:, None]).clamp(0, len(standoffs) - 1)
    executed = (is_reference & (reference_indices >= 151)) | (
        ~is_reference
        & request_valid
        & (standoffs[stand_index] <= selected_standoff[:, None] + 1e-6)
    )
    if not validate_carry_preview:
        executed &= ~is_reference | (reference_indices <= endpoints[:, 1:2])
    arm_safe = ((clearance >= collider_clearance_m).any(0) | ~executed).all(-1) & (
        approach_clearance >= collider_clearance_m
    )
    total = counts.sum(-1)
    path_owner, time = torch.where(
        (torch.arange(capacity, device=root.device)[None] < total[:, None])
        & endpoint_inverse_available[:, None]
    )
    cumulative = counts.cumsum(-1)
    phase = (time[:, None] >= cumulative[path_owner, :-1]).sum(-1)
    starts = torch.cat((torch.zeros_like(counts[:, :1]), cumulative[:, :-1]), -1)
    fraction = ((time - starts[path_owner, phase]) / (counts[path_owner, phase] - 1)).to(goal.dtype)
    fraction = torch.where(
        phase >= 2, fraction.pow(3) * (10 + fraction * (-15 + 6 * fraction)), fraction
    )
    lifted = goal.clone()
    required = torch.maximum(
        state["object_state"][:, 2].new_tensor(0.30), inverse["minimum_object_height"]
    )[owner]
    lifted[:, 2] = (goal[:, 2] + (required - target[owner, 2]).clamp_min(0) + 0.005).clamp_min(
        0.305
    )
    forward = target[:, :2] - root[:, :2]
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    withdrawal = inverse.get("withdraw_distance_m")
    if withdrawal is None:
        withdrawal = torch.where(target[:, 2] < 0.12, 0.0, 0.16)
    withdrawn = lifted.clone()
    withdrawn[:, :2] -= withdrawal[owner, None] * forward[owner]
    knots = torch.stack((state["ee_pos"][owner], pregrasp, goal, lifted, withdrawn), 1)
    start, end = knots[path_owner, phase], knots[path_owner, phase + 1]
    preview_p = start + fraction[:, None] * (end - start)
    preview_q = q[path_owner].clone()
    transit_time = time.clamp_max(150)
    preview_p = torch.where((phase == 0)[:, None], approach_p[path_owner, transit_time], preview_p)
    preview_q = torch.where((phase == 0)[:, None], approach_q[path_owner, transit_time], preview_q)
    checked = torch.ones_like(phase, dtype=torch.bool) if validate_carry_preview else phase <= 1
    gap = collision.gripper_clearance(
        geometry.select(owner[path_owner[checked]]),
        preview_p[checked, None],
        preview_q[checked, None],
        maximum_distance_m=collider_clearance_m + 0.0001,
    )[:, 0]
    gripper_clearance = goal.new_full((count,), torch.inf)
    gripper_clearance.scatter_reduce_(0, path_owner[checked], gap, reduce="amin")
    phase_clearance = goal.new_full((count * 4,), torch.inf)
    phase_clearance.scatter_reduce_(0, path_owner[checked] * 4 + phase[checked], gap, reduce="amin")
    phase_bank = goal.new_full((batch, width, 4), torch.nan)
    phase_bank[owner, candidate] = phase_clearance.reshape(count, 4)
    diagnostics["gripper_clearance_by_phase_m"] = phase_bank
    pregrasp_bank = goal.new_full((batch, width, 3), torch.nan)
    pregrasp_bank[owner, candidate] = pregrasp
    diagnostics["pregrasp_positions_world_m"] = pregrasp_bank
    approach_height_bank = goal.new_full((batch, width), torch.nan)
    approach_height_bank[owner, candidate] = approach_p[..., 2].amin(-1)
    diagnostics["approach_minimum_tcp_height_world_m"] = approach_height_bank
    collision_clear = (
        endpoint_inverse_available
        & (gripper_clearance >= collider_clearance_m)
        & (approach_clearance >= collider_clearance_m)
    )
    safe = collision_clear & (total <= capacity)
    safe &= transit_valid
    safe &= quality["antipodal_margin"][owner, candidate] >= 0
    safe &= quality["pocket_fit"][owner, candidate]
    if require_safe_arm_path:
        safe &= arm_safe
    joint_margin = torch.minimum(joints - limits[:, 0], limits[:, 1] - joints).amin(-1)

    def at_end(value):
        return value.gather(2, endpoint_slots[None].expand(len(heights), -1, -1))

    end_exact = at_end(exact)
    endpoint_support = end_exact & (at_end(clearance) >= collider_clearance_m)
    # The body may change height between phases, but each endpoint still needs
    # an exact, collision-clear solution at at least one admissible height.
    endpoint_inverse_available = endpoint_support.any(0).all(-1)
    safe &= endpoint_inverse_available
    actual_body = (
        end_exact[-3:, :, 1]
        & (at_end(position_error)[-3:, :, 1] <= 0.002)
        & (at_end(orientation_error)[-3:, :, 1] <= 0.02)
        & (at_end(clearance)[-3:, :, 1] >= collider_clearance_m)
        & (at_end(joint_margin)[-3:, :, 1] >= minimum_realized_grasp_margin_rad)
    ).all(0)
    has_actual = torch.zeros(batch, device=root.device, dtype=torch.int32)
    has_actual.scatter_reduce_(0, owner, (safe & actual_body).int(), reduce="amax")
    safe &= ~prefer_realized_body[owner] | ~has_actual[owner].bool() | actual_body
    if realized_body is not None:
        safe &= ~realized_body[owner] | actual_body
    body_shift = (
        (heights - root[owner, 2][None])
        .abs()
        .masked_fill(~endpoint_support[..., 1], torch.inf)
        .amin(0)
    )
    realized_residual = (
        at_end(position_error)[-3:, :, 1] + 0.1 * at_end(orientation_error)[-3:, :, 1]
    ).amax(0)
    best_residual = root.new_full((batch,), torch.inf)
    best_residual.scatter_reduce_(
        0, owner, realized_residual.masked_fill(~safe, torch.inf), reduce="amin"
    )
    needs_adaptation = prefer_realized_body[owner] & ~has_actual[owner].bool()
    safe &= ~needs_adaptation | (realized_residual <= best_residual[owner] + 1e-6)
    point_cost = position_error + 0.1 * orientation_error + (0.1 - joint_margin).clamp_min(0) * 0.1
    point_cost += 0.02 * (heights[:, :, None] - root[owner, 2][None, :, None]).abs()
    end_cost = at_end(point_cost)[..., 1:].amin(0).amax(-1)
    cost = end_cost + end_cost * 0.1 + (~arm_safe).float() * 0.1
    if realized_body is not None:
        cost += torch.where(realized_body[owner], at_end(point_cost)[-3:, :, 1].amax(0), 0.0)
    rotation = (q * state["ee_orn"][owner]).sum(-1).abs().clamp_max(1)
    cost += 0.001 * 2 * rotation.acos()
    wrist = at_end(joints[..., 5])[..., 1:] - actual[None, :, 5, None]
    wrist = wrist.abs().amax(-1)
    motion_exact = end_exact[..., 1:].all(-1)
    wrist_motion = wrist.masked_fill(~motion_exact, torch.inf).amin(0)
    wrist_motion = torch.where(motion_exact.any(0), wrist_motion, wrist.amin(0))
    wrist_motion = torch.maximum(wrist_motion, (end_joints[:, 5] - actual[:, 5]).abs())
    cost += (0.05 - quality["antipodal_margin"][owner, candidate]).clamp_min(0) * 0.1
    preferred_axis = target - grasp_approach_origin(root)
    preferred_axis /= preferred_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cost += 0.01 * (1 - (axes * preferred_axis[owner]).sum(-1))
    cost += wrist_motion_weight * wrist_motion
    preserve = (candidate == 0) & safe & motion_exact.any(0)
    preservable = torch.zeros_like(feasible_bank)
    preservable[owner, candidate] = preserve
    diagnostics["preferred_orientation_preservable"] = preservable
    has_preferred = torch.zeros(batch, device=root.device, dtype=torch.int32)
    has_preferred.scatter_reduce_(0, owner, preserve.int(), reduce="amax")
    if prefer_orientation:
        safe &= ~has_preferred[owner].bool() | preserve
    cost_bank[owner, candidate] = cost
    feasible_bank[owner, candidate] = safe
    selected, valid = best_feasible(feasible_bank, cost_bank)
    chosen = inverse["lookup"][torch.arange(batch, device=root.device), selected].clamp_min(0)
    insert = (
        pregrasp[:, None]
        + torch.linspace(0, 1, 101, device=root.device)[None, :, None] * (goal - pregrasp)[:, None]
    )
    p = torch.cat((approach_p, insert), 1)[chosen]
    orientation = torch.cat((approach_q, q[:, None].expand(-1, 101, -1)), 1)[chosen]
    path, timed = retime_grasp_approach_batch(p, orientation, goal[chosen], dt, capacity=capacity)
    valid &= timed
    diagnostics["selected"] = torch.where(valid, selected, -1)
    support_counts = torch.zeros(batch, width, 4, device=root.device, dtype=torch.long)
    support_counts[owner, candidate] = endpoint_support.sum(0)
    diagnostics["endpoint_inverse_body_support_counts"] = support_counts
    diagnostics["selected_pregrasp_position_world_m"] = pregrasp[chosen]
    diagnostics["withdraw_displacement_m"] = torch.cat(
        (-withdrawal[:, None] * forward, target.new_zeros(batch, 1)), -1
    )
    for name, values, fill in (
        ("endpoint_inverse_available", endpoint_inverse_available, False),
        ("collision_clear", collision_clear, False),
        ("arm_posture_envelope_clear", arm_safe, False),
        ("gripper_min_clearance_m", gripper_clearance, torch.nan),
        ("approach_arm_minimum_surface_distance_m", approach_clearance, torch.nan),
        ("approach_nominal_base_height_m", approach_height, torch.nan),
        ("pregrasp_standoff_m", selected_standoff, torch.nan),
        ("wrist_travel_cost_rad", wrist_motion, torch.nan),
        ("transit_path_valid", transit_valid, False),
        ("preview_within_capacity", total <= capacity, False),
        ("realized_body_grasp_candidate_mask", actual_body, False),
        ("minimum_supported_grasp_body_shift_m", body_shift, torch.nan),
        ("realized_body_grasp_residual_m", realized_residual, torch.nan),
        (
            "max_position_error_m",
            position_error.amin(0).masked_fill(~request_valid, -torch.inf).amax(-1),
            torch.nan,
        ),
        (
            "max_orientation_error",
            orientation_error.amin(0).masked_fill(~request_valid, -torch.inf).amax(-1),
            torch.nan,
        ),
    ):
        value = torch.full((batch, width), fill, device=values.device, dtype=values.dtype)
        value[owner, candidate] = values
        diagnostics[name] = value
    return path, valid, diagnostics
