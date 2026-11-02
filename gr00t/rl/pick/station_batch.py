"""Scene-owned GPU station screening with the existing pocket and pose planners."""

import torch

from gr00t.rl.auto_open.control import APPROACH_ARRIVAL
from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply
from gr00t.rl.auto_open.pose_planning import solve_pose_grid
from gr00t.rl.pick.planning import (
    grasp_approach_origin,
    grasp_pose_bank,
    object_frame_grasp_bank,
    pocket_pose_targets,
)
from gr00t.rl.pick.station import (
    station_arrival_hypotheses,
    station_corridors_clear,
    station_pose_bank,
)
from gr00t.rl.pick.station_sampling import joint_station_requests, sample_station_pockets

STATION_POLICIES = (
    "arrival_exists",
    "body_volume",
    "all_postures_arrival",
    "standing_arrival",
    "posture_volume_without_standing_solution",
    "highest_reachable_grasp_posture_then_volume",
)


def _station_gripper_clearance(geometry, collision, owner, points, quats, active):
    """Collapse exactly repeated scene/pose queries across body hypotheses on GPU."""
    shape = points.shape[:-1]
    orientation = quats[:, None].expand(-1, shape[1], -1)
    identity = torch.cat(
        (owner[:, None, None].expand(-1, shape[1], 1).to(points.dtype), points, orientation), -1
    ).flatten(0, 1)
    unique, inverse = torch.unique(identity, dim=0, return_inverse=True)
    required = torch.zeros(len(unique), device=points.device, dtype=torch.int32)
    required.scatter_reduce_(0, inverse, active.flatten().to(torch.int32), reduce="amax")
    options = (
        {"query_mask": required[:, None].bool()}
        if getattr(geometry, "supports_query_mask", False)
        else {}
    )
    distance = collision.gripper_clearance(
        geometry.select(unique[:, 0].long()),
        unique[:, None, 1:4],
        unique[:, None, 4:8],
        maximum_distance_m=0.0021,
        **options,
    ).flatten()
    return torch.where(active, distance[inverse].reshape(shape), points.new_tensor(0.0021))


def rank_station_candidates_batch(
    body_cost,
    nominal_arrival,
    *,
    body_volume=False,
    standing_postures=None,
    require_all_postures=False,
    posture_heights=None,
):
    """Apply the scalar station policy independently to [scene, station, body, arrival, grasp]."""
    batch = len(body_cost)
    shape = body_cost.shape
    if body_cost.ndim != 5 or not 0 <= nominal_arrival < shape[3]:
        raise ValueError("Station ranking requires scene/station/body/arrival/grasp costs")
    policy = torch.full((batch,), int(body_volume), device=body_cost.device, dtype=torch.long)
    body_support = torch.isfinite(body_cost).sum((2, 3))
    arrival_cost = body_cost.amin(2)
    use_volume = torch.full_like(policy, body_volume, dtype=torch.bool)
    use_standing = torch.zeros_like(use_volume)
    use_height = torch.zeros_like(use_volume)
    best_height = body_cost.new_full((batch,), -torch.inf)
    highest_nominal = body_cost.new_full((batch, shape[1], shape[4]), -torch.inf)
    height_eligible = torch.ones_like(highest_nominal, dtype=torch.bool)
    if require_all_postures:
        arrival_cost = body_cost.amax(2)
        use_volume.fill_(False)
        policy.fill_(2)
    elif standing_postures is not None:
        if standing_postures.shape != (batch, shape[2]):
            raise ValueError("Standing posture masks must be scene-owned")
        standing_cost = body_cost.masked_fill(
            ~standing_postures[:, None, :, None, None], torch.inf
        ).amin(2)
        use_standing = torch.isfinite(standing_cost[:, :, nominal_arrival]).any((1, 2))
        arrival_cost = torch.where(use_standing[:, None, None, None], standing_cost, arrival_cost)
        use_volume = ~use_standing
        policy = torch.where(use_standing, 3, 4)
        if posture_heights is not None:
            if posture_heights.shape != (batch, shape[2]):
                raise ValueError("Posture heights must be scene-owned")
            nominal = torch.isfinite(body_cost[:, :, :, nominal_arrival])
            highest_nominal = (
                posture_heights[:, None, :, None]
                .expand_as(nominal)
                .masked_fill(~nominal, -torch.inf)
                .amax(2)
            )
            best_height = highest_nominal.amax((1, 2))
            use_height = ~use_standing
            height_eligible = (~use_height[:, None, None]) | (
                highest_nominal == best_height[:, None, None]
            )
            restricted = body_cost.masked_fill(
                (posture_heights < best_height[:, None])[:, None, :, None, None], torch.inf
            )
            body_support = torch.where(
                use_height[:, None, None], torch.isfinite(restricted).sum((2, 3)), body_support
            )
            arrival_cost = torch.where(
                use_height[:, None, None, None], restricted.amin(2), arrival_cost
            )
            policy = torch.where(use_height, 5, policy)
    available = torch.isfinite(arrival_cost)
    coverage = torch.where(available[:, :, nominal_arrival], available.sum(2), 0)
    worst = torch.where(available, arrival_cost, 0).amax(2)
    score = torch.where(use_volume[:, None, None], body_support, coverage)
    score = torch.where(available[:, :, nominal_arrival] & height_eligible, score, 0)
    eligible = (score == score.amax((1, 2), keepdim=True)) & (score > 0)
    return (
        eligible,
        worst,
        {
            "arrival_coverage_counts": coverage,
            "arrival_worst_cost": worst,
            "body_arrival_support_counts": body_support,
            "selection_policy_code": policy,
            "highest_nominal_grasp_height_m": highest_nominal,
            "highest_grasp_height_m": best_height,
            "restrict_to_standing": use_standing,
            "restrict_to_highest": use_height,
        },
    )


def _select_station_chunk(
    state,
    destination,
    object_size,
    support_points,
    limits,
    collision,
    geometry,
    *,
    support_mask=None,
    pitch_step_deg=15.0,
    body_volume=False,
    body_heights=(0.30, 0.34, 0.38, 0.536, 0.58),
    body_pitches_deg=(0.0, 4.0, 8.0, 12.0),
    yaw_offsets_deg=(-15.0, 0.0, 15.0),
    prefer_standing=True,
    require_all_postures=False,
    forward_offsets_m=None,
    lateral_offsets_m=None,
    max_requests=65536,
):
    """Default initial stations, preserving independent grasp/lift/withdraw body hypotheses.

    This is a GPU planner, not a simulation loop. Low targets remain commands for
    the whole-body policy; posture screening never requires the actual standing
    body to have a fixed-base arm solution.
    """
    root, target = state["root_states"], state["object_state"][:, :3]
    batch = len(root)
    stations, posture = station_pose_bank(
        root,
        destination,
        target,
        body_heights=body_heights,
        body_pitches_deg=body_pitches_deg,
        yaw_offsets_deg=yaw_offsets_deg,
        forward_offsets_m=forward_offsets_m,
        lateral_offsets_m=lateral_offsets_m,
    )
    corridor = station_corridors_clear(root, stations, geometry)
    arrivals, nominal = station_arrival_hypotheses(posture)
    station_count, body_count, arrival_count = arrivals.shape[1:4]
    roots = arrivals.flatten(1, 3)
    posture_count = roots.shape[1]
    station_owner = torch.arange(station_count, device=root.device).repeat_interleave(
        body_count * arrival_count
    )
    reference = root.clone()
    reference[:, :2] = destination[:, :2]
    reference[:, 3:7] = _euler_xyz_quat(
        torch.stack((destination[:, 2] * 0, destination[:, 2] * 0, destination[:, 2]), -1)
    )
    w, x, y, z = root[:, 3:7].unbind(-1)
    angle = destination[:, 2] - torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    rotation = torch.stack(((angle / 2).cos(), angle * 0, angle * 0, (angle / 2).sin()), -1)
    _, orientations, _ = grasp_pose_bank(
        target,
        destination[:, 2],
        object_size,
        grasp_approach_origin(reference),
        pitch_step_deg=pitch_step_deg,
        reference_orientation=_quat_multiply(rotation, state["ee_orn"]),
    )
    object_pose_offset = orientations.shape[1]
    orientations = torch.cat(
        (orientations, object_frame_grasp_bank(state["object_state"][:, 3:7], object_size)), 1
    )
    if orientations.shape[1] > 64:
        orientations, goals, quality = sample_station_pockets(
            orientations,
            state,
            support_points,
            support_mask,
            collision,
            geometry,
            object_pose_offset=object_pose_offset,
        )
    else:
        goals, quality = pocket_pose_targets(
            orientations,
            target,
            state["object_state"][:, 3:7],
            support_points,
            collision,
            geometry,
            support_mask=support_mask,
        )
    grasp_count = orientations.shape[1]
    physical = (
        quality["pocket_fit"]
        & (quality["antipodal_margin"] >= 0)
        & (quality["environment_clearance_m"] >= 0.002)
    )
    owner, body, candidate = joint_station_requests(
        physical,
        corridor,
        body_count,
        arrival_count,
        state.get("episode_seed", torch.zeros(batch, device=root.device, dtype=torch.long)),
        station_priors=torch.stack(
            (
                (
                    (stations[..., :2] - destination[:, None, :2]).norm(dim=-1)
                    + 0.05 * (stations[..., 2] - destination[:, None, 2]).abs()
                )
                .masked_fill(~corridor, torch.inf)
                .argmin(-1),
                (
                    (stations[..., :2] - target[:, None, :2]).norm(dim=-1)
                    + 0.05 * (stations[..., 2] - destination[:, None, 2]).abs()
                )
                .masked_fill(~corridor, torch.inf)
                .argmin(-1),
            ),
            -1,
        ),
    )
    sampled_counts = torch.zeros(batch, posture_count, device=root.device, dtype=torch.long)
    sampled_counts.view(-1).index_add_(0, owner * posture_count + body, torch.ones_like(owner))
    actual = state["dof_pos"][:, collision.arm_joint_ids]
    diagnostics = {
        "station_poses": stations,
        "base_corridor_clear": corridor,
        "physical_pocket_counts": sampled_counts,
        "joint_candidate_budget": 1000,
        "joint_candidate_count": sampled_counts.sum(-1),
        "posture_hypotheses": roots[..., :7],
        "orientation_definition": "joint_sampled_station_posture_grasp_with_arrival_stencil",
        "orientation_family": "parallel_transport",
        "pitch_step_deg": pitch_step_deg,
        "posture_hypotheses_are_not_execution_guarantees": True,
        "arrival_sample_count": arrival_count,
        "arrival_position_radius_m": APPROACH_ARRIVAL.outer_distance_m,
        "arrival_yaw_radius_rad": APPROACH_ARRIVAL.outer_yaw_rad,
        "endpoint_posture_policy": "independent_grasp_lift_withdraw_postures",
    }
    if not len(owner):
        empty_cost = root.new_full(
            (batch, station_count, body_count, arrival_count, grasp_count), torch.inf
        )
        change = (posture[:, 0, :, 2] - root[:, 2:3]).abs()
        _, _, ranking = rank_station_candidates_batch(
            empty_cost,
            nominal,
            body_volume=body_volume,
            standing_postures=(change == change.amin(1, keepdim=True)) if prefer_standing else None,
            require_all_postures=require_all_postures,
            posture_heights=posture[:, 0, :, 2],
        )
        diagnostics.update(ranking)
        zeros = torch.zeros(batch, device=root.device, dtype=torch.long)
        diagnostics.update(
            planning_failure_code=torch.ones_like(zeros),
            selected=zeros - 1,
            selected_candidate=zeros - 1,
            exact_endpoint_point_counts=zeros[:, None].expand(-1, 3),
            safe_endpoint_point_counts=zeros[:, None].expand(-1, 3),
            selected_orientation=state["ee_orn"].clone(),
            selected_endpoint_joints_rad=root.new_full((batch, 3, 6), torch.nan),
            predicted_q6_departure_rad=root.new_full((batch,), torch.nan),
            predicted_joint_margin_rad=root.new_full((batch,), torch.nan),
            selected_arrival_coverage=zeros.clone(),
            selected_body_arrival_support=zeros.clone(),
            q6_departure_reference="episode_initial_joint_position",
        )
        return (
            destination.clone(),
            state["ee_orn"].clone(),
            actual.clone(),
            torch.zeros(batch, device=root.device, dtype=torch.bool),
            diagnostics,
        )
    grasp = goals[owner, candidate]
    lift = grasp.clone()
    minimum = torch.maximum(target[:, 2] + 0.10, target.new_full((batch,), 0.30))
    lift[:, 2] = (lift[:, 2] + minimum[owner] - target[owner, 2] + 0.005).clamp_min(0.305)
    withdraw = lift.clone()
    direction = target[owner, :2] - roots[owner, body, :2]
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    withdraw[:, :2] -= direction * torch.where(target[owner, 2:3] < 0.12, 0.0, 0.16)
    points = torch.stack((grasp, lift, withdraw), 1)
    quats = orientations[owner, candidate]
    seeds = torch.stack((actual, limits.mean(-1).expand_as(actual), limits.sum(-1) - actual), 0)
    joints = root.new_empty(len(owner), 3, 3, 6)
    exact = torch.empty(len(owner), 3, 3, device=root.device, dtype=torch.bool)
    error = root.new_empty(len(owner), 3, 3)
    for start in range(0, len(owner), max(1, max_requests // 9)):
        stop = min(start + max(1, max_requests // 9), len(owner))
        indices = slice(start, stop)
        result = solve_pose_grid(
            points[indices],
            quats[indices, None].expand(-1, 3, -1),
            roots[owner[indices], body[indices]][None].expand(3, -1, -1),
            seeds[:, owner[indices]],
            limits,
            max_requests=max_requests,
        )
        joints[indices] = result.arm_joint_positions.permute(1, 0, 2, 3)
        exact[indices] = result.exact.permute(1, 0, 2)
        error[indices] = result.position_error.norm(dim=-1).permute(1, 0, 2)
    margin = torch.minimum(joints - limits[:, 0], limits[:, 1] - joints).amin(-1)
    wrist = (joints[..., 5] - actual[owner, None, None, 5]).abs()
    initial_exact = exact.clone()
    record, seed, phase = torch.where(exact)
    if len(record):
        # Scene selection duplicates BVH handles on device, not scene geometry.
        for start in range(0, len(record), max_requests):
            chosen = slice(start, start + max_requests)
            r, s, t = record[chosen], seed[chosen], phase[chosen]
            clear = collision.clearance(
                geometry.select(owner[r]),
                roots[owner[r], body[r]],
                joints[r, s, t],
                maximum_distance_m=0.0021,
                include_gripper=False,
            )
            exact[r, s, t] &= clear >= 0.002
    # Withdrawal direction depends on the body's xy, so query the sparse records.
    gripper_clear = _station_gripper_clearance(
        geometry, collision, owner, points, quats, exact.any(1)
    )
    exact &= (gripper_clear >= 0.002)[:, None]
    point_cost = 10 * error + 5 * (0.08 - margin).clamp_min(0) + 0.05 * wrist
    cost = point_cost[..., 0].clone()
    cost += 1.5 * (roots[owner, body, 2:3] - root[owner, 2:3]).abs()
    cost += 5 * (0.05 - quality["antipodal_margin"][owner, candidate, None]).clamp_min(0)
    cost += (
        0.1
        * (stations[owner, station_owner[body], :2] - destination[owner, :2]).norm(dim=-1)[:, None]
    )
    point_cost += (cost - point_cost[..., 0])[..., None]
    dense = root.new_full((batch, posture_count, grasp_count, 3), torch.inf)
    dense[owner, body, candidate] = point_cost.masked_fill(~exact, torch.inf).amin(1)
    dense = dense.reshape(batch, station_count, body_count, arrival_count, grasp_count, 3)
    carry = torch.isfinite(dense[..., 1:]).any(2).all(-1)
    body_cost = dense[..., 0].masked_fill(~carry[:, :, None], torch.inf)
    height_change = (posture[:, 0, :, 2] - root[:, 2:3]).abs()
    standing = height_change == height_change.amin(1, keepdim=True)
    eligible, worst, ranking = rank_station_candidates_batch(
        body_cost,
        nominal,
        body_volume=body_volume,
        standing_postures=standing if prefer_standing else None,
        require_all_postures=require_all_postures,
        posture_heights=posture[:, 0, :, 2],
    )
    station_id = station_owner[body]
    body_id = (body // arrival_count).remainder(body_count)
    posture_allowed = ~ranking["restrict_to_standing"][owner] | standing[owner, body_id]
    posture_allowed &= ~ranking["restrict_to_highest"][owner] | (
        roots[owner, body, 2] >= ranking["highest_grasp_height_m"][owner]
    )
    feasible = (
        exact[..., 0] & carry[owner, station_id, body.remainder(arrival_count), candidate, None]
    )
    nominal_mask = (body.remainder(arrival_count) == nominal) & posture_allowed
    if require_all_postures:
        nominal_mask &= eligible[owner, station_id, candidate]
    seed_cost = cost.masked_fill(~(feasible & nominal_mask[:, None]), torch.inf)
    value, best_seed = seed_cost.min(-1)
    table = root.new_full((batch, len(owner)), torch.inf)
    record = torch.arange(len(owner), device=root.device)
    table[owner, record] = value
    nominee_row = table.argsort(dim=-1, stable=True)[:, : min(4, len(owner))]
    nominee_cost = table.gather(1, nominee_row)
    nominee_valid = torch.isfinite(nominee_cost)
    nominees = nominee_row.shape[1]
    verify_owner = torch.arange(batch, device=root.device).repeat_interleave(nominees)
    nominal_roots = roots[owner[nominee_row], body[nominee_row]].flatten(0, 1)
    nominee_record = nominee_row.flatten()
    nominee_station = station_id[nominee_record]
    nominee_grasp = candidate[nominee_record]
    carry_table = dense[verify_owner, nominee_station][:, :, nominal]
    carry_table = carry_table.gather(
        2, nominee_grasp[:, None, None, None].expand(-1, body_count, 1, 3)
    )[:, :, 0]
    endpoint_body = carry_table.argmin(1)
    nominee_postures = posture[verify_owner, nominee_station]
    endpoint_roots = nominee_postures[
        torch.arange(len(verify_owner), device=root.device)[:, None], endpoint_body
    ].clone()
    endpoint_roots[:, 0] = nominal_roots
    arrival_roots, _ = station_arrival_hypotheses(endpoint_roots)
    verified = solve_pose_grid(
        points[nominee_row].flatten(0, 1),
        quats[nominee_row][..., None, :].expand(-1, -1, 3, -1).flatten(0, 1),
        arrival_roots.permute(2, 0, 1, 3)[None].expand(3, -1, -1, -1, -1),
        seeds[:, verify_owner][:, None].expand(-1, arrival_count, -1, -1),
        limits,
        query_mask=nominee_valid.flatten()[:, None].expand(-1, 3),
        max_requests=max_requests,
    )
    coverage = torch.zeros_like(nominee_row)
    if verified is not None:
        robust = verified.exact.clone()
        hypothesis, arrival, proposal, phase = torch.where(robust)
        for first in range(0, len(proposal), max_requests):
            sl = slice(first, first + max_requests)
            h, a, p, t = hypothesis[sl], arrival[sl], proposal[sl], phase[sl]
            gap = collision.clearance(
                geometry.select(verify_owner[p]),
                arrival_roots[p, t, a],
                verified.arm_joint_positions[h, a, p, t],
                maximum_distance_m=0.0021,
                include_gripper=False,
            )
            robust[h, a, p, t] &= gap >= 0.002
        coverage = robust.any(0).all(-1).sum(0).reshape(batch, nominees)
    supported = nominee_valid & (coverage > 0)
    greatest = coverage.masked_fill(~supported, -1).amax(-1, keepdim=True)
    final_cost = nominee_cost.masked_fill(~supported | (coverage != greatest), torch.inf)
    choice = final_cost.argmin(-1)
    own_rows = torch.arange(batch, device=root.device)
    row = nominee_row[own_rows, choice]
    seed = best_seed[row]
    selected_station, selected_candidate = station_id[row], candidate[row]
    valid = supported.any(-1)
    ranking["arrival_coverage_counts"][own_rows, selected_station, selected_candidate] = coverage[
        own_rows, choice
    ]
    diagnostics["nominal_joint_sample_budget"] = 1000
    diagnostics["robust_nominee_count"] = nominees
    pose = torch.where(valid[:, None], stations[own_rows, selected_station], destination)
    orientation = torch.where(valid[:, None], quats[row], state["ee_orn"])
    joint_reference = torch.where(valid[:, None], joints[row, seed, 0], actual)
    exact_counts = torch.zeros(batch, 3, device=root.device, dtype=torch.long)
    safe_counts = torch.zeros_like(exact_counts)
    exact_counts.index_add_(0, owner, initial_exact.any(1).long())
    safe_counts.index_add_(0, owner, exact.any(1).long())
    diagnostics.update(ranking)
    diagnostics.update(
        planning_failure_code=torch.where(
            valid,
            0,
            torch.where(
                ~physical.flatten(1).any(1),
                1,
                torch.where(safe_counts.sum(1) == 0, 2, 3),
            ),
        ),
        selected=torch.where(valid, selected_station, -1),
        selected_candidate=torch.where(valid, selected_candidate, -1),
        exact_endpoint_point_counts=exact_counts,
        safe_endpoint_point_counts=safe_counts,
        selected_orientation=orientation,
        selected_endpoint_joints_rad=torch.where(
            valid[:, None, None], joints[row, seed], torch.nan
        ),
        predicted_q6_departure_rad=torch.where(valid, wrist[row, seed, 0], torch.nan),
        predicted_joint_margin_rad=torch.where(valid, margin[row, seed, 0], torch.nan),
        selected_arrival_coverage=torch.where(
            valid,
            ranking["arrival_coverage_counts"][own_rows, selected_station, selected_candidate],
            0,
        ),
        selected_body_arrival_support=torch.where(
            valid,
            ranking["body_arrival_support_counts"][own_rows, selected_station, selected_candidate],
            0,
        ),
        q6_departure_reference="episode_initial_joint_position",
    )
    return pose, orientation, joint_reference, valid, diagnostics


def select_initial_pick_station_batch(
    state,
    destination,
    object_size,
    support_points,
    limits,
    collision,
    geometry,
    *,
    support_mask=None,
    max_scenes=100,
    **options,
):
    """Bound dense station tables by scene groups, and inverse work by request count.

    Grouping is a memory bound, not an environment-specific planning loop. All
    candidates and hypotheses are retained, including for heterogeneous scenes.
    """
    count = len(destination)
    if count < 1 or max_scenes < 1:
        raise ValueError("Station planning requires a nonempty batch and positive group size")
    if count <= max_scenes:
        return _select_station_chunk(
            state,
            destination,
            object_size,
            support_points,
            limits,
            collision,
            geometry,
            support_mask=support_mask,
            **options,
        )
    parts = []
    for start in range(0, count, max_scenes):
        stop = min(start + max_scenes, count)
        rows = torch.arange(start, stop, device=destination.device)
        parts.append(
            _select_station_chunk(
                {key: value[start:stop] for key, value in state.items()},
                destination[start:stop],
                object_size[start:stop]
                if torch.is_tensor(object_size) and object_size.ndim > 1
                else object_size,
                support_points[start:stop] if support_points.ndim == 3 else support_points,
                limits,
                collision,
                geometry.select(rows),
                support_mask=(
                    support_mask[start:stop]
                    if support_mask is not None and support_mask.ndim > 1
                    else support_mask
                ),
                **options,
            )
        )
    diagnostics = {}
    for key, example in parts[0][-1].items():
        if not torch.is_tensor(example):
            diagnostics[key] = example
            continue
        diagnostics[key] = torch.cat([part[-1][key] for part in parts])
    return tuple(torch.cat([part[i] for part in parts]) for i in range(4)) + (diagnostics,)
