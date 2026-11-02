"""Choose a compact held-object lift with robust actual-body IK, on the GPU."""

import math
from types import SimpleNamespace

import torch

from gr00t.rl.auto_open.motion_math import _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate, _slerp
from gr00t.rl.auto_open.pose_planning import best_feasible
from gr00t.rl.pick.basket_inverse import safe_inverse_paths_batch
from gr00t.rl.pick.basket_nudge import pose_segments, safe_inverse_paths, transverse_tilts
from gr00t.rl.pick.planning import retime_grasp_approach, retime_grasp_approach_batch
from gr00t.rl.utils.b2z1_asset import read_arm_mount

DEFAULT_CARRY_TILT_ANGLES_DEG = (5, 10, 15, 20, 30, 40, 50, 60)
CARRY_BODY_HEIGHT_OFFSETS_M = (-0.01, 0.0, 0.01)
MAX_HELD_SAMPLE_EVALUATIONS = 1_000_000
HELD_CLEARANCE_QUERY_CAP_M = 0.0021


def held_pose_bank_batch(
    state,
    basket,
    support_points,
    minimum_height,
    *,
    tilt_angles_deg=DEFAULT_CARRY_TILT_ANGLES_DEG,
    support_mask=None,
):
    root = state["root_states"]
    position, orientation = state["ee_pos"], state["ee_orn"]
    obj = state["object_state"]
    inverse = orientation.clone()
    inverse[:, 1:] *= -1
    relative = _quat_rotate(inverse, obj[:, :3] - position)
    world, _ = basket.world_mesh(root)
    if world.ndim == 2:
        if len(root) != 1:
            raise ValueError("Basket world mesh must preserve the environment dimension")
        world = world[None]
    rim = world[..., 2].amax(-1)
    radii = support_points.norm(dim=-1)
    if support_mask is not None:
        radii = radii.masked_fill(~support_mask, -torch.inf)
    radius = radii.amax(-1)
    height = torch.maximum(obj[:, 2], minimum_height.reshape(-1) + 0.005)
    clear_height = torch.maximum(obj[:, 2], torch.minimum(rim + 0.03 + radius, height))
    cone, tilt_rotation = transverse_tilts(root, angles_deg=tilt_angles_deg)
    offsets = root.new_tensor([0, 0.04, 0.08, 0.12])
    bank = torch.cat(
        (cone.repeat_interleave(len(offsets), dim=0), offsets.repeat(len(cone))[:, None]), -1
    )
    rotation = tilt_rotation.repeat_interleave(len(offsets), dim=0)
    end_q = _quat_multiply(orientation[:, None], rotation[None])
    mid_q = _slerp(
        orientation[:, None].expand_as(end_q), end_q, root.new_full((len(root), len(bank)), 0.5)
    )
    mount = root[:, :3] + _quat_rotate(root[:, 3:7], root.new_tensor(read_arm_mount()[0]))
    direction = mount - obj[:, :3]
    direction[:, 2] = 0
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    mid = obj[:, None, :3].expand(-1, len(bank), -1).clone()
    mid[:, :, 2] = clear_height[:, None]
    end = obj[:, None, :3] + bank[None, :, 2, None] * direction[:, None]
    end[:, :, 2] = height[:, None]
    centers = torch.stack((obj[:, None, :3].expand_as(mid), mid, end), 2)
    quats = torch.stack((orientation[:, None].expand_as(end_q), mid_q, end_q), 2)
    return centers, quats, relative, bank


def held_pose_bank(state, basket, support_points, minimum_height, **kwargs):
    """Single-scene adapter to the common tensor candidate geometry."""
    if len(state["root_states"]) != 1:
        raise ValueError("Use held_pose_bank_batch for multiple environments")
    centers, quats, relative, bank = held_pose_bank_batch(
        state, basket, support_points, minimum_height, **kwargs
    )
    return centers[0], quats[0], relative[0], bank


def held_geometry_checks_batch(
    state,
    positions,
    quats,
    relative,
    support_points,
    basket,
    collision,
    geometry,
    aperture,
    *,
    support_mask=None,
):
    """All held surfaces and basket bounds, with environment-local padded hulls."""
    batch, candidates, length = positions.shape[:3]
    vertices = support_points.shape[-2]
    row_capacity = max(1, MAX_HELD_SAMPLE_EVALUATIONS // (length * vertices))
    if batch > row_capacity:
        results = []
        for start in range(0, batch, row_capacity):
            stop = min(start + row_capacity, batch)
            rows = torch.arange(start, stop, device=positions.device)
            results.append(
                held_geometry_checks_batch(
                    {
                        key: state[key][start:stop]
                        for key in ("root_states", "ee_orn", "object_state")
                    },
                    positions[start:stop],
                    quats[start:stop],
                    relative[start:stop],
                    support_points[start:stop] if support_points.ndim == 3 else support_points,
                    basket,
                    collision,
                    geometry.select(rows),
                    aperture[start:stop],
                    support_mask=(
                        support_mask[start:stop]
                        if support_mask is not None and support_mask.ndim == 2
                        else support_mask
                    ),
                )
            )
        return tuple(torch.cat(values, 0) for values in zip(*results))
    candidate_capacity = max(1, MAX_HELD_SAMPLE_EVALUATIONS // (batch * length * vertices))
    if candidates > candidate_capacity:
        results = [
            held_geometry_checks_batch(
                state,
                positions[:, start : start + candidate_capacity],
                quats[:, start : start + candidate_capacity],
                relative,
                support_points,
                basket,
                collision,
                geometry,
                aperture,
                support_mask=support_mask,
            )
            for start in range(0, candidates, candidate_capacity)
        ]
        return tuple(torch.cat(values, 1) for values in zip(*results))
    centers = positions + _quat_rotate(quats, relative[:, None, None])
    initial_q = state["ee_orn"].clone()
    initial_q[:, 1:] *= -1
    object_relative_q = _quat_multiply(initial_q, state["object_state"][:, 3:7])
    object_q = _quat_multiply(quats, object_relative_q[:, None, None])
    # Acceptance uses only positivity, the 2 mm gate and recovery clipped to 2 mm.
    options = {"maximum_distance_m": HELD_CLEARANCE_QUERY_CAP_M}
    if support_mask is not None:
        options["local_point_mask"] = support_mask
    object_clear = geometry.nonrobot_geometry.rigid_surface_clearance(
        centers, object_q, support_points, **options
    )
    gripper_clear = collision.gripper_pose_clearance(
        geometry,
        positions,
        quats,
        aperture.reshape(batch, 1, 1).expand(batch, candidates, length),
        maximum_distance_m=HELD_CLEARANCE_QUERY_CAP_M,
    )
    root = state["root_states"]
    inverse = root[:, 3:7].clone()
    inverse[:, 1:] *= -1
    points = (
        support_points[None].expand(batch, -1, -1) if support_points.ndim == 2 else support_points
    )
    valid = (
        torch.ones(points.shape[:2], device=positions.device, dtype=torch.bool)
        if support_mask is None
        else support_mask.expand(points.shape[:2])
    )
    first = valid.long().argmax(-1)
    fallback = points[torch.arange(batch, device=positions.device), first][:, None]
    points = torch.where(valid[..., None], points, fallback)
    world = centers[..., None, :] + _quat_rotate(object_q[..., None, :], points[:, None, None])
    local = _quat_rotate(inverse[:, None, None, None], world - root[:, None, None, None, :3])
    obj = state["object_state"]
    initial_world = obj[:, None, :3] + _quat_rotate(obj[:, None, 3:7], points)
    initial_bottom = _quat_rotate(inverse[:, None], initial_world - root[:, None, :3])[..., 2]
    initial_bottom = initial_bottom.masked_fill(~valid, torch.inf).amin(-1)
    bottom = local[..., 2].masked_fill(~valid[:, None, None], torch.inf).amin(-1)
    bounds = bottom >= initial_bottom[:, None, None] - 1e-5
    low = local.masked_fill(~valid[:, None, None, :, None], torch.inf).amin(-2)
    high = local.masked_fill(~valid[:, None, None, :, None], -torch.inf).amax(-2)
    inside = (low[..., :2] >= basket.floor_lower[:2]).all(-1) & (
        high[..., :2] <= basket.floor_upper[:2]
    ).all(-1)
    bounds &= inside | (low[..., 2] >= basket.rim_z + 0.002)
    bounds &= valid.any(-1)[:, None, None]
    height_change = bottom - initial_bottom[:, None, None]
    floor_margin = torch.minimum(
        low[..., :2] - basket.floor_lower[:2], basket.floor_upper[:2] - high[..., :2]
    ).amin(-1)
    return (
        bounds,
        gripper_clear,
        object_clear,
        height_change,
        floor_margin,
        low[..., 2] - basket.rim_z,
    )


def held_geometry_acceptance(checks):
    bounds, gripper, payload = checks[:3]
    recovery = gripper.clamp_max(0.002)
    metrics = {
        "bounds_safe": bounds.all(-1),
        "gripper_positive_clearance": (gripper > 0).all(-1),
        "gripper_release_monotone": (recovery.diff(dim=-1) >= -1e-6).all(-1),
        "gripper_terminal_clearance_m": gripper[..., -1],
        "gripper_min_clearance_m": gripper.amin(-1),
        "object_min_clearance_m": payload.amin(-1),
        "minimum_local_object_height_change_m": checks[3].amin(-1),
        "minimum_floor_footprint_margin_m": checks[4].amin(-1),
        "outside_footprint_below_rim": ((checks[4] < 0) & (checks[5] < 0.002)).any(-1),
    }
    safe = (
        metrics["bounds_safe"]
        & metrics["gripper_positive_clearance"]
        & metrics["gripper_release_monotone"]
        & (gripper[..., -1] >= 0.002)
        & (payload >= 0.002).all(-1)
    )
    return safe, metrics


def held_geometry_batch(*args, **kwargs):
    safe, metrics = held_geometry_acceptance(held_geometry_checks_batch(*args, **kwargs))
    return safe, metrics["gripper_min_clearance_m"], metrics["object_min_clearance_m"]


def certify_padded_held_path(
    state,
    path,
    relative,
    support_points,
    basket,
    collision,
    geometry,
    aperture,
    *,
    support_mask=None,
    active_rows=None,
):
    """Query actual path frames together; terminal padding never expands surface work."""
    p, q, lengths = path
    batch, capacity = p.shape[:2]
    mask = torch.arange(capacity, device=p.device)[None] < lengths[:, None]
    if active_rows is not None:
        mask &= active_rows[:, None]
    owner, time = torch.where(mask)
    bounds = torch.ones(batch, capacity, device=p.device, dtype=torch.bool)
    gripper = p.new_full((batch, capacity), torch.inf)
    payload = gripper.clone()
    height_change = p.new_full((batch,), torch.inf)
    footprint_margin = height_change.clone()
    outside_below_rim = torch.zeros(batch, device=p.device, dtype=torch.int32)
    if len(owner):
        local = {key: state[key][owner] for key in ("root_states", "ee_orn", "object_state")}
        points = support_points[owner] if support_points.ndim == 3 else support_points
        point_mask = (
            support_mask[owner]
            if support_mask is not None and support_mask.ndim == 2
            else support_mask
        )
        checks = held_geometry_checks_batch(
            local,
            p[owner, time][:, None, None],
            q[owner, time][:, None, None],
            relative[owner],
            points,
            basket,
            collision,
            geometry.select(owner),
            aperture[owner],
            support_mask=point_mask,
        )
        bounds[owner, time] = checks[0][:, 0, 0]
        gripper[owner, time] = checks[1][:, 0, 0]
        payload[owner, time] = checks[2][:, 0, 0]
        height_change.scatter_reduce_(0, owner, checks[3][:, 0, 0], reduce="amin")
        footprint_margin.scatter_reduce_(0, owner, checks[4][:, 0, 0], reduce="amin")
        outside_below_rim.scatter_reduce_(
            0, owner, ((checks[4][:, 0, 0] < 0) & (checks[5][:, 0, 0] < 0.002)).int(), reduce="amax"
        )
    end = gripper.gather(1, (lengths.long() - 1)[:, None]).squeeze(1)
    recovery = gripper.clamp_max(0.002)
    valid = (
        bounds.all(-1)
        & (gripper > 0).all(-1)
        & (payload >= 0.002).all(-1)
        & (recovery.diff(dim=-1) >= -1e-6).all(-1)
        & (end >= 0.002)
    )
    return valid, {
        "dense_gripper_min_clearance_m": gripper.amin(-1),
        "dense_object_min_clearance_m": payload.amin(-1),
        "dense_bounds_safe": bounds.all(-1),
        "dense_gripper_terminal_clearance_m": end,
        "dense_gripper_release_monotone": (recovery.diff(dim=-1) >= -1e-6).all(-1),
        "dense_minimum_local_object_height_change_m": height_change,
        "dense_minimum_floor_footprint_margin_m": footprint_margin,
        "dense_outside_footprint_below_rim": outside_below_rim.bool(),
    }


def held_geometry(
    state, positions, quats, relative, support_points, basket, collision, geometry, aperture
):
    """Single-scene adapter; geometry mathematics is shared with the batch planner."""
    if len(state["root_states"]) != 1:
        raise ValueError("Use held_geometry_batch for multiple environments")
    view = SimpleNamespace(
        nonrobot_geometry=SimpleNamespace(
            rigid_surface_clearance=lambda p, q, points, **options: (
                geometry.nonrobot_geometry.rigid_surface_clearance(p[0], q[0], points, **options)[
                    None
                ]
            )
        )
    )
    jaws = SimpleNamespace(
        gripper_pose_clearance=lambda g, p, q, a, **options: collision.gripper_pose_clearance(
            geometry, p[0], q[0], a[0], **options
        )[None]
    )
    result = held_geometry_batch(
        state,
        positions[None],
        quats[None],
        relative[None],
        support_points,
        basket,
        jaws,
        view,
        aperture,
    )
    return tuple(value[0] for value in result)


def plan_basket_carry_bank(
    state,
    initial_height,
    aperture,
    limits,
    collision,
    geometry,
    dt,
    *,
    basket,
    support_points,
    minimum_object_height,
    tilt_angles_deg=DEFAULT_CARRY_TILT_ANGLES_DEG,
):
    rim_reserve = max(CARRY_BODY_HEIGHT_OFFSETS_M)
    centers, orientations, relative, bank = held_pose_bank(
        state,
        basket,
        support_points,
        minimum_object_height + rim_reserve,
        tilt_angles_deg=tilt_angles_deg,
    )
    center_path, quats, _ = pose_segments(centers, orientations, dt, samples=17)
    positions = center_path - _quat_rotate(quats, relative)
    safe, gripper_clear, object_clear = held_geometry(
        state, positions, quats, relative, support_points, basket, collision, geometry, aperture
    )
    eligible = torch.where(safe)[0]
    diagnostics = {
        "candidate_count": len(bank),
        "candidate_tilt_rad_azimuth_rad_retract_m": bank.tolist(),
        "geometric_candidates": len(eligible),
        "inverse_calls": 0,
        "gripper_min_clearance_m": gripper_clear.tolist(),
        "object_min_clearance_m": object_clear.tolist(),
    }
    if not len(eligible):
        diagnostics["failure"] = "no_collision_free_held_object_lift"
        return None, diagnostics
    exact, travel = safe_inverse_paths(
        state,
        positions[eligible],
        quats[eligible],
        limits,
        collision,
        geometry,
        body_height_offsets=CARRY_BODY_HEIGHT_OFFSETS_M,
        final_joint_margin_rad=math.radians(2),
        diagnostics=diagnostics,
    )
    diagnostics.update(
        inverse_calls=1,
        robust_inverse_mask=exact.tolist(),
        body_height_offsets_m=list(CARRY_BODY_HEIGHT_OFFSETS_M),
        moving_basket_rim_reserve_m=rim_reserve,
        final_joint_margin_deg=2,
    )
    if not bool(exact.any()):
        diagnostics["failure"] = "no_robust_actual_body_lift_path"
        return None, diagnostics
    cost = bank[eligible, 0] + 3 * bank[eligible, 2] + 0.001 * travel
    selected = eligible[cost.masked_fill(~exact, torch.inf).argmin()]
    p, q = positions[selected : selected + 1], quats[selected : selected + 1]
    p, q = retime_grasp_approach(p, q, p[:, -1], dt, transit_speed_m_s=0.15, capture_speed_m_s=0.15)
    dense_ok, _, _ = held_geometry(
        state, p, q, relative, support_points, basket, collision, geometry, aperture
    )
    if not bool(dense_ok.all()):
        diagnostics["failure"] = "held_lift_dense_geometry_failed"
        return None, diagnostics
    diagnostics.update(
        selected=int(selected),
        end_ee_world=p[:, -1].tolist(),
        selected_tilt_rad_azimuth_rad_retract_m=bank[selected].tolist(),
        reference_samples=p.shape[1],
        speed_limit_m_s=0.15,
        required_object_height_world_m=minimum_object_height.tolist(),
    )
    return (p, q, torch.tensor([p.shape[1]], device=p.device)), diagnostics


def plan_basket_carry_batch(
    state,
    initial_height,
    aperture,
    limits,
    collision,
    geometry,
    dt,
    *,
    basket,
    support_points,
    minimum_object_height,
    support_mask=None,
    tilt_angles_deg=DEFAULT_CARRY_TILT_ANGLES_DEG,
    capacity=4096,
    early_exit=True,
):
    """Plan all stationary basket lifts through the common GPU inverse and path cache."""
    centers, rotations, relative, bank = held_pose_bank_batch(
        state,
        basket,
        support_points,
        minimum_object_height + max(CARRY_BODY_HEIGHT_OFFSETS_M),
        tilt_angles_deg=tilt_angles_deg,
        support_mask=support_mask,
    )
    batch, candidates = centers.shape[:2]
    # Interpolate the small pose bank once, so shortlist shape cannot alter quaternion rounding.
    p, q, _ = pose_segments(centers.flatten(0, 1), rotations.flatten(0, 1), dt, samples=17)
    p, q = p.reshape(batch, candidates, -1, 3), q.reshape(batch, candidates, -1, 4)
    p = p - _quat_rotate(q, relative[:, None, None])
    rows = torch.arange(batch, device=centers.device)

    def evaluate(actors, columns):
        return _plan_basket_carry_candidates(
            {
                key: state[key][actors]
                for key in ("root_states", "dof_pos", "ee_pos", "ee_orn", "object_state")
            },
            initial_height[actors],
            aperture[actors],
            limits,
            collision,
            geometry.select(actors),
            dt,
            basket=basket,
            support_points=support_points[actors] if support_points.ndim == 3 else support_points,
            minimum_object_height=minimum_object_height[actors],
            support_mask=(
                support_mask[actors]
                if support_mask is not None and support_mask.ndim == 2
                else support_mask
            ),
            capacity=capacity,
            positions=p[actors][:, columns],
            quats=q[actors][:, columns],
            relative=relative[actors],
            bank=bank[columns],
        )

    def finish(chosen, found, report):
        chosen_p, chosen_q = chosen
        path, timed = retime_grasp_approach_batch(
            chosen_p,
            chosen_q,
            chosen_p[:, -1],
            dt,
            transit_speed_m_s=0.15,
            capture_speed_m_s=0.15,
            capacity=capacity,
        )
        dense, details = certify_padded_held_path(
            state,
            path,
            relative,
            support_points,
            basket,
            collision,
            geometry,
            aperture,
            support_mask=support_mask,
            active_rows=found & timed,
        )
        valid = found & timed & dense
        failure = torch.where(report["geometric_candidate_mask"].any(-1), 2, 1)
        failure = torch.where(found & ~timed, 3, failure)
        failure = torch.where(found & timed & ~dense, 4, failure)
        report.update(
            retimed_path_within_capacity=timed,
            dense_geometry_checked=found & timed,
            dense_geometry_safe=dense,
            planning_failure_code=torch.where(valid, 0, failure),
            selected=torch.where(valid, report["selected"], -1),
            **details,
        )
        return path, valid, report

    if not early_exit:
        path, valid, report = evaluate(rows, slice(None))
        report["candidate_evaluated_mask"] = torch.ones(
            batch, candidates, device=centers.device, dtype=torch.bool
        )
        report["cost_bound_early_exit"] = torch.zeros_like(valid)
        return finish(path, valid, report)

    # Every cost is tilt + 3*retract + nonnegative joint travel.
    lower_bound = bank[:, 0] + 3 * bank[:, 2]
    order = torch.cat((rows.new_zeros(1), lower_bound[1:].argsort(stable=True) + 1))
    best_cost = centers.new_full((batch,), torch.inf)
    best_index = rows.new_full((batch,), candidates)
    resolved = torch.zeros(batch, device=centers.device, dtype=torch.bool)
    path, valid = None, torch.zeros_like(resolved)
    report = {}
    candidate_fields = {
        "geometric_candidate_mask",
        "robust_inverse_mask",
        "candidate_cost",
        "candidate_tilt_rad_azimuth_rad_retract_m",
        "gripper_min_clearance_m",
        "object_min_clearance_m",
        "geometry_eligible",
        "inverse_exact_path_by_height",
        "inverse_safe_path_by_height",
        "best_max_position_error_by_height_m",
        "best_minimum_arm_clearance_by_height_m",
        "arm_clearance_query_evaluated_by_height",
        "best_final_half_joint_margin_by_height_rad",
    }
    evaluated = torch.zeros(batch, candidates, device=centers.device, dtype=torch.bool)
    cohorts = [(0, 1), *((start, min(start + 8, candidates)) for start in range(1, candidates, 8))]
    for start, stop in cohorts:
        actors = torch.where(~resolved)[0]
        if not len(actors):
            break
        # Preserve the original index tie-break within each cost-ordered cohort.
        columns = order[start:stop].sort().values
        local_path, local_valid, local_report = evaluate(actors, columns)
        selected, found = best_feasible(
            local_report["geometric_candidate_mask"] & local_report["robust_inverse_mask"],
            local_report["candidate_cost"],
        )
        score = local_report["candidate_cost"].gather(1, selected[:, None]).squeeze(1)
        global_index = columns[selected]
        improved = found & (
            (score < best_cost[actors])
            | ((score == best_cost[actors]) & (global_index < best_index[actors]))
        )
        if path is None:
            path = local_path
        else:
            for destination, source in zip(path, local_path, strict=True):
                destination[actors[improved]] = source[improved]
        valid[actors[improved]] = local_valid[improved]
        best_cost[actors[improved]] = score[improved]
        best_index[actors[improved]] = global_index[improved]
        for key, value in local_report.items():
            if torch.is_tensor(value):
                candidate_field = key in candidate_fields or key.startswith("candidate_checks.")
                if key not in report:
                    fill = torch.nan if value.is_floating_point() else False
                    shape = (
                        (batch, candidates, *value.shape[2:])
                        if candidate_field
                        else (batch, *value.shape[1:])
                    )
                    report[key] = value.new_full(shape, fill)
                if candidate_field:
                    report[key][actors[:, None], columns[None]] = value
                else:
                    if key == "selected":
                        value = torch.where(value >= 0, columns[value.clamp_min(0)], -1)
                    take = torch.ones_like(improved) if start == 0 else improved
                    report[key][actors[take]] = value[take]
            else:
                report[key] = value
        evaluated[actors[:, None], columns[None]] = True
        next_bound = lower_bound[order[stop]] if stop < candidates else torch.inf
        resolved |= torch.isfinite(best_cost) & (best_cost < next_bound)
    report["geometric_candidates"] = report["geometric_candidate_mask"].sum(-1)
    report["candidate_count"] = candidates
    report["candidate_tilt_rad_azimuth_rad_retract_m"] = bank[None].expand(batch, -1, -1)
    report["candidate_evaluated_mask"] = evaluated
    report["cost_bound_early_exit"] = resolved & ~evaluated.all(-1)
    report["candidate_diagnostics_scope"] = "evaluated_candidates_only"
    return finish(path, valid, report)


def _plan_basket_carry_candidates(
    state,
    initial_height,
    aperture,
    limits,
    collision,
    geometry,
    dt,
    *,
    basket,
    support_points,
    minimum_object_height,
    support_mask,
    capacity,
    positions,
    quats,
    relative,
    bank,
):
    p, q = positions, quats
    batch, candidates = p.shape[:2]
    checks = held_geometry_checks_batch(
        state,
        p,
        q,
        relative,
        support_points,
        basket,
        collision,
        geometry,
        aperture,
        support_mask=support_mask,
    )
    geometric, geometric_diagnostics = held_geometry_acceptance(checks)
    jaw_gap, object_gap = (
        geometric_diagnostics["gripper_min_clearance_m"],
        geometric_diagnostics["object_min_clearance_m"],
    )
    robust, travel, inverse_diagnostics = safe_inverse_paths_batch(
        state,
        p,
        q,
        limits,
        collision,
        geometry,
        candidate_mask=geometric,
        body_height_offsets=CARRY_BODY_HEIGHT_OFFSETS_M,
        final_joint_margin_rad=math.radians(2),
        maximum_arm_query_distance_m=HELD_CLEARANCE_QUERY_CAP_M,
    )
    cost = bank[None, :, 0] + 3 * bank[None, :, 2] + 0.001 * travel
    selected, found = best_feasible(geometric & robust, cost)
    rows = torch.arange(batch, device=p.device)
    chosen_p, chosen_q = p[rows, selected], q[rows, selected]
    diagnostics = {
        "geometric_candidate_mask": geometric,
        "robust_inverse_mask": robust,
        "candidate_cost": cost,
        "candidate_count": candidates,
        "candidate_tilt_rad_azimuth_rad_retract_m": bank[None].expand(batch, -1, -1),
        "selected": torch.where(found, selected, -1),
        "selected_tilt_rad_azimuth_rad_retract_m": bank[selected],
        "gripper_min_clearance_m": jaw_gap,
        "object_min_clearance_m": object_gap,
        "required_object_height_world_m": minimum_object_height,
        "grasp_object_relative_m": relative,
        "moving_basket_rim_reserve_m": max(CARRY_BODY_HEIGHT_OFFSETS_M),
        "body_height_offsets_m": list(CARRY_BODY_HEIGHT_OFFSETS_M),
        "final_joint_margin_deg": 2,
        "speed_limit_m_s": 0.15,
        "clearance_query_cap_m": HELD_CLEARANCE_QUERY_CAP_M,
        "initial_object_height_world_m": initial_height,
        "geometric_candidates": geometric.sum(-1),
        **{f"candidate_checks.{name}": value for name, value in geometric_diagnostics.items()},
        **inverse_diagnostics,
    }
    return (chosen_p, chosen_q), found, diagnostics
