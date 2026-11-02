"""Stage-entry station proposals with shared GPU pocket and inverse planning."""

import math

import torch

from gr00t.rl.auto_open.control import APPROACH_ARRIVAL
from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply
from gr00t.rl.pick.grasp_clearance import GRASP_CLEARANCE_M
from gr00t.rl.pick.planning import (
    grasp_approach_origin,
    grasp_pose_bank,
    inverse_bank,
    object_frame_grasp_bank,
    pocket_pose_targets,
    refine_grasp_candidates,
)


def station_pose_bank(
    root,
    destination,
    target,
    *,
    body_heights=None,
    body_pitches_deg=None,
    yaw_offsets_deg=None,
    forward_offsets_m=None,
    lateral_offsets_m=None,
    ring_radii_m=None,
    ring_samples=24,
):
    """Joint position/yaw samples around the nominal approach station."""
    toward = target[:, :2] - destination[:, :2]
    toward /= toward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    forward, lateral = torch.meshgrid(
        root.new_tensor([-0.16, 0, 0.16] if forward_offsets_m is None else forward_offsets_m),
        root.new_tensor([-0.12, 0, 0.12] if lateral_offsets_m is None else lateral_offsets_m),
        indexing="ij",
    )
    forward, lateral = forward.flatten(), lateral.flatten()
    side = torch.stack((-toward[:, 1], toward[:, 0]), -1)
    stations = destination[:, None].expand(-1, len(forward), -1).clone()
    stations[..., :2] += forward[None, :, None] * toward[:, None]
    stations[..., :2] += lateral[None, :, None] * side[:, None]
    heading_error = destination[:, 2] - torch.atan2(toward[:, 1], toward[:, 0])
    delta = target[:, None, :2] - stations[..., :2]
    stations[..., 2] = torch.atan2(delta[..., 1], delta[..., 0]) + heading_error[:, None]
    if ring_radii_m is not None:
        radii = root.new_tensor(ring_radii_m)
        if (
            radii.ndim != 1
            or not len(radii)
            or not torch.isfinite(radii).all()
            or (radii <= 0).any()
        ):
            raise ValueError("Station ring radii must be a nonempty positive finite sequence")
        if not isinstance(ring_samples, int) or not 4 <= ring_samples <= 72:
            raise ValueError("Station ring sample count must be an integer in [4, 72]")
        angles = torch.arange(ring_samples, device=root.device, dtype=root.dtype) * (
            2 * math.pi / ring_samples
        )
        reference = torch.atan2(toward[:, 1], toward[:, 0])
        angles = reference[:, None] + angles[None]
        direction = torch.stack((angles.cos(), angles.sin()), -1)
        xy = target[:, None, None, :2] - radii[None, :, None, None] * direction[:, None]
        yaw = (angles + heading_error[:, None])[:, None].expand(-1, len(radii), -1)
        stations = torch.cat((xy, yaw[..., None]), -1).flatten(1, 2)
    if yaw_offsets_deg is not None:
        yaw_offsets = root.new_tensor(yaw_offsets_deg) * (math.pi / 180)
        if not len(yaw_offsets) or not torch.isfinite(yaw_offsets).all():
            raise ValueError("Station yaw offsets must be a nonempty finite sequence")
        stations = stations[:, :, None].expand(-1, -1, len(yaw_offsets), -1).clone()
        stations[..., 2] += yaw_offsets
        stations = stations.flatten(1, 2)
    heights, pitch = torch.meshgrid(
        root.new_tensor([0.26, 0.36, 0.46, 0.536] if body_heights is None else body_heights),
        root.new_tensor([0, 8] if body_pitches_deg is None else body_pitches_deg) * (math.pi / 180),
        indexing="ij",
    )
    heights, pitch = heights.flatten(), pitch.flatten()
    roots = root[:, None, None].expand(-1, stations.shape[1], len(heights), -1).clone()
    roots[..., :2] = stations[:, :, None, :2]
    roots[..., 2] = heights[None, None]
    rpy = torch.zeros_like(roots[..., :3])
    rpy[..., 1] = pitch
    rpy[..., 2] = stations[:, :, None, 2]
    roots[..., 3:7] = _euler_xyz_quat(rpy)
    return stations, roots


def station_arrival_hypotheses(posture):
    radius = APPROACH_ARRIVAL.outer_distance_m
    delta = posture.new_tensor([[0, 0], [radius, 0], [-radius, 0], [0, radius], [0, -radius]])
    yaw = posture.new_tensor(
        [-APPROACH_ARRIVAL.outer_yaw_rad, 0, APPROACH_ARRIVAL.outer_yaw_rad]
    ).repeat(len(delta))
    delta = delta.repeat_interleave(3, 0)
    roots = posture[..., None, :].expand(*posture.shape[:-1], len(yaw), posture.shape[-1]).clone()
    roots[..., :2] += delta
    rotation = torch.stack(((yaw / 2).cos(), yaw * 0, yaw * 0, (yaw / 2).sin()), -1)
    roots[..., 3:7] = _quat_multiply(rotation, roots[..., 3:7])
    return roots, 1


def station_corridors_clear(root, stations, geometry):
    t = torch.linspace(0, 1, 40, device=root.device)
    angle = torch.arange(32, device=root.device) * (math.pi / 16)
    z = root.new_tensor([0.20, 0.50, 0.80])
    xy = root[:, None, None, :2] + t[None, None, :, None] * (
        stations[:, :, None, :2] - root[:, None, None, :2]
    )
    origin = root.new_zeros(*xy.shape[:-1], len(z), len(angle), 3)
    origin[..., :2] = xy[..., None, None, :]
    origin[..., 2] = z[:, None]
    end = origin.clone()
    radius = (0.48 + t * APPROACH_ARRIVAL.outer_distance_m)[None, None, :, None, None]
    end[..., 0] += radius * angle.cos()
    end[..., 1] += radius * angle.sin()
    return geometry.visible(origin, end).all(dim=(-1, -2, -3))


def rank_station_candidates(
    body_cost,
    nominal_arrival,
    *,
    body_volume=False,
    standing_postures=None,
    require_all_postures=False,
    posture_heights=None,
):
    """Rank feasible arrival/posture support before comparing motion costs."""
    body_support = torch.isfinite(body_cost).sum((1, 2))
    arrival_cost = body_cost.amin(1)
    height_eligible = None
    highest_nominal = None
    policy = "body_volume" if body_volume else "arrival_exists"
    if require_all_postures:
        arrival_cost = body_cost.amax(1)
        body_volume = False
        policy = "all_postures_arrival"
    elif standing_postures is not None:
        standing_cost = body_cost[:, standing_postures].amin(1)
        if torch.isfinite(standing_cost[:, nominal_arrival]).any():
            arrival_cost = standing_cost
            body_volume = False
            policy = "standing_arrival"
        else:
            body_volume = True
            policy = "posture_volume_without_standing_solution"
            if posture_heights is not None:
                nominal = torch.isfinite(body_cost[:, :, nominal_arrival])
                highest_nominal = (
                    posture_heights[None, :, None]
                    .expand_as(nominal)
                    .masked_fill(~nominal, -torch.inf)
                    .amax(1)
                )
                best_height = highest_nominal.amax()
                height_eligible = highest_nominal == best_height
                body_cost = body_cost.masked_fill(
                    (posture_heights < best_height)[None, :, None, None], torch.inf
                )
                body_support = torch.isfinite(body_cost).sum((1, 2))
                arrival_cost = body_cost.amin(1)
                policy = "highest_reachable_grasp_posture_then_volume"
    available = torch.isfinite(arrival_cost)
    coverage = available.sum(1)
    coverage = torch.where(available[:, nominal_arrival], coverage, 0)
    worst = torch.where(available, arrival_cost, 0).amax(1)
    support_score = body_support if body_volume else coverage
    support_score = torch.where(available[:, nominal_arrival], support_score, 0)
    if height_eligible is not None:
        support_score = torch.where(height_eligible, support_score, 0)
    eligible = (support_score == support_score.max()) & (support_score > 0)
    return (
        eligible,
        worst,
        {
            "arrival_coverage_counts": coverage,
            "arrival_worst_cost": worst,
            "body_arrival_support_counts": body_support,
            "selection_policy": policy,
            **(
                {}
                if highest_nominal is None
                else {"highest_nominal_grasp_height_m": highest_nominal}
            ),
        },
    )


def select_initial_pick_station(
    state,
    destination,
    object_size,
    support_points,
    limits,
    collision,
    geometry,
    *,
    pitch_step_deg=15.0,
    body_volume=False,
    body_heights=(0.30, 0.34, 0.38, 0.536),
    body_pitches_deg=(0.0, 4.0, 8.0, 12.0),
    refine_orientations=False,
    natural_orientations=True,
    yaw_offsets_deg=(-15.0, 0.0, 15.0),
    prefer_standing=True,
    require_all_postures=False,
    forward_offsets_m=None,
    lateral_offsets_m=None,
    retain_depth_candidates=False,
    response_model=None,
    ring_radii_m=None,
    ring_samples=24,
):
    """One batched inverse for grasp/lift/withdraw endpoints at initial stations.

    Posture hypotheses rank station proposals; they never suppress low EE goals
    or claim that the whole-body policy will realize a particular body height.
    """
    root = state["root_states"]
    if len(root) != 1:
        raise ValueError("Station planning requires a scene-scoped request")
    target = state["object_state"][:, :3]
    stations, posture = station_pose_bank(
        root,
        destination,
        target,
        body_heights=body_heights,
        body_pitches_deg=body_pitches_deg,
        yaw_offsets_deg=yaw_offsets_deg,
        forward_offsets_m=forward_offsets_m,
        lateral_offsets_m=lateral_offsets_m,
        ring_radii_m=ring_radii_m,
        ring_samples=ring_samples,
    )
    corridor_clear = station_corridors_clear(root, stations, geometry)[0]
    arrivals, nominal_arrival = station_arrival_hypotheses(posture)
    arrival_count = arrivals.shape[-2]
    roots = arrivals.flatten(0, 3)
    owner = torch.arange(stations.shape[1], device=root.device).repeat_interleave(
        posture.shape[2] * arrival_count
    )
    reference = root.clone()
    reference[:, :2] = destination[:, :2]
    reference[:, 3:7] = _euler_xyz_quat(
        torch.stack((destination[:, 2] * 0, destination[:, 2] * 0, destination[:, 2]), -1)
    )
    bank_options = {}
    if natural_orientations:
        w, x, y, z = root[:, 3:7].unbind(-1)
        initial_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        angle = destination[:, 2] - initial_yaw
        yaw_rotation = torch.stack(((angle / 2).cos(), angle * 0, angle * 0, (angle / 2).sin()), -1)
        bank_options["reference_orientation"] = _quat_multiply(yaw_rotation, state["ee_orn"])
    _, task_orientations, _ = grasp_pose_bank(
        target,
        destination[:, 2],
        object_size,
        grasp_approach_origin(reference),
        pitch_step_deg=pitch_step_deg,
        **bank_options,
    )
    task_orientations = torch.cat(
        (task_orientations, object_frame_grasp_bank(state["object_state"][:, 3:7], object_size)),
        1,
    )
    task_goals, task_quality = pocket_pose_targets(
        task_orientations,
        target,
        state["object_state"][:, 3:7],
        support_points,
        collision,
        geometry,
        **({"retain_depth_candidates": True} if retain_depth_candidates else {}),
    )
    if retain_depth_candidates:
        task_orientations = task_orientations.repeat_interleave(
            task_goals.shape[1] // task_orientations.shape[1], 1
        )
    coarse_count = task_orientations.shape[1]
    if refine_orientations:
        task_orientations, task_goals, task_quality = refine_grasp_candidates(
            task_orientations,
            task_goals,
            task_quality,
            state["ee_orn"],
            target,
            state["object_state"][:, 3:7],
            support_points,
            collision,
            geometry,
        )
    orientations = task_orientations.expand(len(roots), -1, -1)
    goals = task_goals.expand(len(roots), -1, -1)
    quality = {
        key: value.expand(len(roots), *value.shape[1:]) for key, value in task_quality.items()
    }
    physical = (
        quality["pocket_fit"]
        & (quality["antipodal_margin"] >= 0)
        & (quality["environment_clearance_m"] >= GRASP_CLEARANCE_M)
    )
    physical &= corridor_clear[owner, None]
    posture_row, candidate = torch.where(physical)
    diagnostics = {
        "orientation_definition": "nominal_station_and_object_frame_task_pose_bank",
        "orientation_family": "parallel_transport" if natural_orientations else "euler_grid",
        "pitch_step_deg": pitch_step_deg,
        "retain_depth_candidates": retain_depth_candidates,
        "coarse_orientation_count": coarse_count,
        "refined_orientation_count": task_orientations.shape[1] - coarse_count,
        "station_poses": stations[0],
        "station_yaw_offsets_deg": [0.0] if yaw_offsets_deg is None else list(yaw_offsets_deg),
        "posture_hypotheses": roots[:, :7],
        "physical_pocket_counts": physical.sum(-1),
        "base_corridor_clear": corridor_clear,
        "posture_hypotheses_are_not_execution_guarantees": True,
        "arrival_sample_count": arrival_count,
        "arrival_position_radius_m": APPROACH_ARRIVAL.outer_distance_m,
        "arrival_yaw_radius_rad": APPROACH_ARRIVAL.outer_yaw_rad,
    }
    if not len(posture_row):
        return None, None, None, {**diagnostics, "failure": "no_environment_safe_pocket"}
    grasp = goals[posture_row, candidate]
    lift = grasp.clone()
    minimum_object_z = torch.maximum(target[0, 2] + 0.10, target.new_tensor(0.30))
    lift[:, 2] = (lift[:, 2] + minimum_object_z - target[0, 2] + 0.005).clamp_min(0.305)
    withdraw = lift.clone()
    direction = target[:, :2] - roots[posture_row, :2]
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    withdraw[:, :2] -= direction * torch.where(target[:, 2:3] < 0.12, 0.0, 0.16)
    points = torch.stack((grasp, lift, withdraw), 1)
    quats = orientations[posture_row, candidate]
    actual = state["dof_pos"][:, collision.arm_joint_ids][0]
    seeds = torch.stack((actual, limits.mean(-1), limits.sum(-1) - actual))
    grasp_roots = roots[posture_row]
    response_valid = torch.ones(len(grasp_roots), device=root.device, dtype=torch.bool)
    inverse_roots = grasp_roots
    if response_model is not None:
        reference = grasp_roots.clone()
        reference[:, 2] = root[0, 2]
        grasp_roots, response_valid = response_model(
            reference, grasp, quats, actual.expand(len(grasp), -1)
        )
        inverse_roots = roots[posture_row, None].expand(-1, 3, -1).clone()
        inverse_roots[:, 0] = grasp_roots
    result = inverse_bank(
        points[:, None].expand(-1, len(seeds), -1, -1).flatten(0, 1),
        quats[:, None, None].expand(-1, len(seeds), 3, -1).flatten(0, 1),
        inverse_roots[:, None].expand(-1, len(seeds), *inverse_roots.shape[1:]).flatten(0, 1),
        seeds[None].expand(len(posture_row), -1, -1).flatten(0, 1),
        limits,
    )
    joints = result.arm_joint_positions.reshape(-1, len(seeds), 3, 6)
    point_exact = result.exact.reshape(-1, len(seeds), 3).clone()
    point_exact[..., 0] &= response_valid[:, None]
    point_margin = torch.minimum(joints - limits[:, 0], limits[:, 1] - joints).amin(-1)
    point_wrist = (joints[..., 5] - actual[5]).abs()
    point_error = result.position_error.norm(dim=-1).reshape(-1, len(seeds), 3)
    margin, wrist, error = point_margin[..., 0], point_wrist[..., 0], point_error[..., 0]
    cost = 10 * error + 5 * (0.08 - margin).clamp_min(0) + 0.05 * wrist
    cost += 1.5 * (grasp_roots[:, 2:3] - root[0, 2]).abs()
    cost += 5 * (0.05 - quality["antipodal_margin"][posture_row, candidate, None]).clamp_min(0)
    cost += 0.1 * (stations[0, owner[posture_row], :2] - destination[:, :2]).norm(dim=-1)[:, None]
    endpoint_exact_counts = point_exact.all(-1).sum(-1)
    pocket_index, seed_index, phase_index = torch.where(point_exact)
    if len(pocket_index):
        clearance_roots = roots[posture_row[pocket_index]].clone()
        is_grasp = phase_index == 0
        clearance_roots[is_grasp] = grasp_roots[pocket_index[is_grasp]]
        arm_clear = collision.clearance(
            geometry,
            clearance_roots,
            joints[pocket_index, seed_index, phase_index],
            maximum_distance_m=0.0021,
            include_gripper=False,
        )
        packed = torch.cat((points[pocket_index, phase_index], quats[pocket_index]), -1)
        unique_poses, pose_inverse = torch.unique(packed, dim=0, return_inverse=True)
        unique_clear = collision.gripper_clearance(
            geometry, unique_poses[None, :, :3], unique_poses[None, :, 3:]
        )[0]
        gripper_clear = unique_clear[pose_inverse]
        point_exact[pocket_index, seed_index, phase_index] &= (arm_clear >= 0.002) & (
            gripper_clear >= 0.002
        )
    diagnostics.update(
        exact_endpoint_seed_counts=endpoint_exact_counts,
        exact_endpoint_point_counts=result.exact.reshape(-1, len(seeds), 3).any(1).sum(0),
        safe_endpoint_seed_counts=point_exact.sum(1),
        inverse_samples=len(posture_row) * len(seeds) * 3,
        arm_clearance_query_cap_m=0.0021,
    )
    if not point_exact.any():
        return (
            None,
            None,
            None,
            {**diagnostics, "failure": "no_safe_exact_grasp_lift_withdraw_posture"},
        )
    dense_points = root.new_full((*orientations.shape[:-1], 3), torch.inf)
    point_cost = 10 * point_error + 5 * (0.08 - point_margin).clamp_min(0) + 0.05 * point_wrist
    posture_cost = cost - (10 * error + 5 * (0.08 - margin).clamp_min(0) + 0.05 * wrist)
    point_cost += posture_cost[..., None]
    dense_points[posture_row, candidate] = point_cost.masked_fill(~point_exact, torch.inf).amin(1)
    body_points = dense_points.reshape(stations.shape[1], posture.shape[2], arrival_count, -1, 3)
    carry_available = torch.isfinite(body_points[..., 1:]).any(1).all(-1)
    body_cost = body_points[..., 0].masked_fill(~carry_available[:, None], torch.inf)
    exact = (
        point_exact[..., 0]
        & carry_available[owner[posture_row], posture_row.remainder(arrival_count), candidate][
            :, None
        ]
    )
    # A candidate must remain the same world pose across arrival errors; body
    # posture/IK seed may vary. Coverage is a sampled robustness score, not proof.
    height_change = (posture[0, 0, :, 2] - root[0, 2]).abs()
    standing_postures = height_change == height_change.min()
    eligible, worst, ranking = rank_station_candidates(
        body_cost,
        nominal_arrival,
        body_volume=body_volume,
        standing_postures=standing_postures if prefer_standing else None,
        require_all_postures=require_all_postures,
        posture_heights=posture[0, 0, :, 2],
    )
    diagnostics.update(ranking)
    diagnostics["endpoint_posture_policy"] = "independent_grasp_lift_withdraw_postures"
    coverage = ranking["arrival_coverage_counts"]
    body_support = ranking["body_arrival_support_counts"]
    if not eligible.any():
        return None, None, None, {**diagnostics, "failure": "no_safe_nominal_station_candidate"}
    station_candidate = worst.masked_fill(~eligible, torch.inf).argmin(-1)
    station_owner = owner[posture_row]
    nominal_all = (candidate == station_candidate[station_owner]) & (
        posture_row.remainder(arrival_count) == nominal_arrival
    )
    nominal_all &= eligible[station_owner, candidate]
    if ranking["selection_policy"] == "standing_arrival":
        nominal_all &= standing_postures[(posture_row // arrival_count).remainder(posture.shape[2])]
    elif ranking["selection_policy"] == "highest_reachable_grasp_posture_then_volume":
        nominal_all &= roots[posture_row, 2] >= ranking["highest_nominal_grasp_height_m"].amax()
    nominal_cost = cost.masked_fill(~(exact & nominal_all[:, None]), torch.inf).flatten()
    flat_owner = station_owner[:, None].expand_as(cost).flatten()
    per_station_cost = root.new_full((stations.shape[1],), torch.inf)
    per_station_cost.scatter_reduce_(0, flat_owner, nominal_cost, reduce="amin")
    flat_index = torch.arange(cost.numel(), device=root.device)
    best_index = flat_index.masked_fill(
        ~torch.isfinite(nominal_cost) | (nominal_cost != per_station_cost[flat_owner]),
        cost.numel(),
    )
    per_station_index = best_index.new_full((stations.shape[1],), cost.numel())
    per_station_index.scatter_reduce_(0, flat_owner, best_index, reduce="amin")
    valid_station = per_station_index < cost.numel()
    per_station_index = per_station_index.clamp_max(cost.numel() - 1)
    station_row, station_seed = per_station_index // len(seeds), per_station_index % len(seeds)
    diagnostics.update(
        station_nominal_valid=valid_station,
        station_nominal_candidate=station_candidate,
        station_nominal_cost=per_station_cost,
        station_nominal_body_pose=grasp_roots[station_row, :7],
        station_nominal_endpoint_joints_rad=joints[station_row, station_seed],
        station_nominal_joint_margin_rad=margin[station_row, station_seed],
        station_nominal_q6_departure_rad=wrist[station_row, station_seed],
    )
    selected_pair = worst.masked_fill(~eligible, torch.inf).flatten().argmin()
    selected = selected_pair // orientations.shape[1]
    selected_candidate = selected_pair % orientations.shape[1]
    nominal = (
        (owner[posture_row] == selected)
        & (candidate == selected_candidate)
        & (posture_row.remainder(arrival_count) == nominal_arrival)
    )
    if ranking["selection_policy"] == "standing_arrival":
        posture_id = (posture_row // arrival_count).remainder(posture.shape[2])
        nominal &= standing_postures[posture_id]
    elif ranking["selection_policy"] == "highest_reachable_grasp_posture_then_volume":
        nominal &= roots[posture_row, 2] >= ranking["highest_nominal_grasp_height_m"].amax()
    chosen = cost.masked_fill(~(exact & nominal[:, None]), torch.inf).flatten().argmin()
    row, seed = chosen // len(seeds), chosen % len(seeds)
    diagnostics.update(
        selected=selected,
        selected_candidate=selected_candidate,
        selected_posture=grasp_roots[row, :7],
        response_model=None if response_model is None else response_model.path,
        selected_station_body_hypotheses=roots.reshape(
            stations.shape[1], posture.shape[2], arrival_count, 13
        )[selected, :, nominal_arrival, :7],
        predicted_q6_departure_rad=wrist[row, seed],
        predicted_grasp_q6_rad=joints[row, seed, 0, 5],
        q6_departure_reference="episode_initial_joint_position",
        predicted_joint_margin_rad=margin[row, seed],
        selected_orientation=quats[row],
        selected_endpoint_joints_rad=joints[row, seed],
        selected_arrival_coverage=coverage[selected, selected_candidate],
        selected_body_arrival_support=body_support[selected, selected_candidate],
        selected_posture_arrival_supported=torch.isfinite(
            body_cost[selected, :, :, selected_candidate]
        ),
    )
    return stations[:, selected], quats[row][None], joints[row, seed, 0][None], diagnostics
