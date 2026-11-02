"""GPU closed-jaw repositioning, using the existing pose inverse and collider queries."""

import math

import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate, _slerp
from gr00t.rl.pick.basket import BASKET_GRASP_CLEARANCE_M
from gr00t.rl.pick.distractors import augment_scene_geometry, build_distractor_mesh
from gr00t.rl.pick.grasp_quality import load_grasp_support
from gr00t.rl.pick.planning import (
    grasp_approach_origin,
    grasp_pose_bank,
    inverse_bank,
    object_frame_grasp_bank,
    pocket_pose_targets,
)
from gr00t.rl.pick.scene_geometry import SceneGeometry
from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_dot,
)
from gr00t.rl.utils.b2z1_asset import read_arm_mount


def transverse_tilts(reference, *, angles_deg=(5, 10, 15, 20)):
    """Small tool-axis tilts without adding rotation about the tool axis."""
    radii = reference.new_tensor(angles_deg) * (math.pi / 180)
    azimuths = torch.arange(8, device=reference.device, dtype=reference.dtype) * (math.pi / 4)
    cone = torch.cat((reference.new_zeros(1, 2), torch.cartesian_prod(radii, azimuths)))
    half_sine = (cone[:, 0] / 2).sin()
    rotation = torch.stack(
        (
            (cone[:, 0] / 2).cos(),
            cone[:, 0] * 0,
            -cone[:, 1].sin() * half_sine,
            cone[:, 1].cos() * half_sine,
        ),
        -1,
    )
    return cone, rotation


def pose_segments(positions, orientations, dt, *, contact_segment=None, samples=None):
    """Same geometric segments for candidate screening and the timed GPU command bank."""
    points, quats, boundaries = [], [], [0]
    for segment in range(positions.shape[1] - 1):
        p0, p1 = positions[:, segment], positions[:, segment + 1]
        q0, q1 = orientations[:, segment], orientations[:, segment + 1]
        if samples is None:
            speed = 0.025 if segment == contact_segment else 0.15
            angle = 2 * fixed_dot(q0, q1).abs().clamp_max(1).acos()
            duration = torch.maximum((p1 - p0).norm(dim=-1) / speed, angle)
            count = max(2, math.ceil(1.875 * float(duration.max()) / dt) + 1)
        else:
            count = samples
        time = torch.linspace(0, 1, count, device=positions.device, dtype=positions.dtype)
        time = time.pow(3) * (10 + time * (-15 + 6 * time))
        points.append(p0[:, None] + time[None, :, None] * (p1 - p0)[:, None])
        quats.append(
            _slerp(q0[:, None].expand(-1, count, -1), q1[:, None].expand(-1, count, -1), time[None])
        )
        boundaries.append(boundaries[-1] + count)
    return torch.cat(points, 1), torch.cat(quats, 1), boundaries


def pose_segments_batch(positions, orientations, dt, *, contact_segment=None, capacity=4096):
    """Per-row segment durations in the shared padded GPU path representation."""
    if positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[1] < 2 or capacity < 2:
        raise ValueError("Segment paths require [batch, at_least_two_knots, xyz] and capacity >= 2")
    if orientations.shape != (*positions.shape[:-1], 4) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Segment orientations must match positions and timestep must be positive")
    delta = positions[:, 1:] - positions[:, :-1]
    angle = 2 * fixed_dot(orientations[:, 1:], orientations[:, :-1]).abs().clamp_max(1).acos()
    segment = torch.arange(delta.shape[1], device=positions.device)
    speed = (
        torch.where(segment == contact_segment, 0.025, 0.15)
        if contact_segment is not None
        else 0.15
    )
    duration = torch.maximum(delta.norm(dim=-1) / speed, angle)
    counts = (1.875 * duration.double() / dt).ceil().long().add(1).clamp_min(2)
    boundaries = torch.cat((torch.zeros_like(counts[:, :1]), counts.cumsum(-1)), -1)
    total = boundaries[:, -1]
    valid = (total <= capacity) & torch.isfinite(duration).all(-1)
    lengths = total.clamp(2, capacity)
    time = torch.arange(capacity, device=positions.device)[None]
    phase = (time[..., None] >= boundaries[:, None, 1:]).sum(-1).clamp_max(counts.shape[1] - 1)
    fraction = (
        ((time - boundaries.gather(1, phase)) / (counts.gather(1, phase) - 1))
        .to(positions.dtype)
        .clamp(0, 1)
    )
    fraction = fraction.pow(3) * (10 + fraction * (-15 + 6 * fraction))
    p0 = positions.gather(1, phase[..., None].expand(-1, -1, 3))
    p1 = positions.gather(1, (phase + 1)[..., None].expand(-1, -1, 3))
    q0 = orientations.gather(1, phase[..., None].expand(-1, -1, 4))
    q1 = orientations.gather(1, (phase + 1)[..., None].expand(-1, -1, 4))
    p = p0 + fraction[..., None] * (p1 - p0)
    q = _slerp(q0, q1, fraction)
    padding = time >= lengths[:, None] - 1
    p = torch.where(padding[..., None], positions[:, -1:], p)
    q = torch.where(padding[..., None], orientations[:, -1:], q)
    return (p, q, lengths), valid, boundaries


def bbox_space(centers, half_extents, basket, other_centers, other_extents):
    wall = torch.minimum(
        centers - half_extents - basket.floor_lower[:2],
        basket.floor_upper[:2] - centers - half_extents,
    ).amin(-1)
    if len(other_centers):
        gap = (
            (centers[..., None, :] - other_centers).abs()
            - half_extents[..., None, :]
            - other_extents
        )
        separation = gap.amax(-1)
        wall = torch.minimum(wall, separation.amin(-1))
    return wall


def improving_shift_mask(before, after, swept_minimum):
    """Existing conservative box overlap may clear, but must not deepen."""
    initial_limit = torch.minimum(before, before.new_tensor(-0.001))
    return (after >= 0.006) & (after >= before + 0.005) & (swept_minimum >= initial_limit - 1e-6)


def monotone_release_mask(clearance, required_clearance=0.002):
    """Disengage an existing contact without increasing or renewing overlap."""
    progress = clearance.clamp_max(required_clearance)
    best = progress.cummax(-1).values
    return (clearance[..., -1] >= required_clearance) & (
        progress[..., 1:] >= best[..., :-1] - 1e-6
    ).all(-1)


def object_boxes_in_trunk(episode, state, *, sizes=None):
    if sizes is None:
        descriptors = [episode, *episode.get("distractors", [])]
        sizes = state["ee_pos"].new_tensor([item["asset"]["size_m"] for item in descriptors])
    poses = torch.cat((state["object_state"][:, None], state["distractor_states"]), 1)[0]
    signs = torch.cartesian_prod(*[sizes.new_tensor([-1, 1])] * 3)
    corners = (
        _quat_rotate(poses[:, None, 3:7], signs[None] * sizes[:, None] / 2) + poses[:, None, :3]
    )
    root = state["root_states"][0]
    inverse = root[3:7].clone()
    inverse[1:] *= -1
    local = _quat_rotate(inverse, corners - root[:3])
    low, high = local.amin(1), local.amax(1)
    return (low + high) / 2, (high - low) / 2


def safe_inverse_paths(
    state,
    positions,
    quats,
    limits,
    collision,
    geometry,
    *,
    body_height_offsets=(0.0,),
    final_joint_margin_rad=0.0,
    diagnostics=None,
    minimum_arm_clearance_m=0.002,
):
    root = state["root_states"]
    actual = state["dof_pos"][:, collision.arm_joint_ids][0]
    seeds = torch.stack((actual, limits.mean(-1), limits.sum(-1) - actual))
    candidates, length = positions.shape[:2]
    heights = root[0, 2] + root.new_tensor(body_height_offsets)
    shape = (len(heights), len(seeds), candidates)
    p = positions[None, None].expand(*shape, length, 3).flatten(0, 2)
    q = quats[None, None].expand(*shape, length, 4).flatten(0, 2)
    roots = root.expand(*shape, 13).clone()
    roots[..., 2] = heights[:, None, None]
    roots = roots.flatten(0, 2)
    joints = seeds[None, :, None].expand(*shape, 6).flatten(0, 2)
    result = inverse_bank(p, q, roots, joints, limits)
    arm_clear = collision.clearance(
        geometry,
        roots[:, None].expand(-1, length, -1).reshape(-1, 13),
        result.arm_joint_positions.reshape(-1, 6),
        # Callers check actual jaw surfaces at the intended aperture separately.
        # The arm's filled link06 box would falsely obstruct the open pocket.
        include_gripper=False,
    ).reshape(*shape, length)
    exact = result.exact.reshape(*shape, length)
    arm = result.arm_joint_positions.reshape(*shape, length, 6)
    margin = torch.minimum(arm - limits[:, 0], limits[:, 1] - arm).amin(-1)
    clearance_ok = (arm_clear >= minimum_arm_clearance_m).all(-1)
    robust = exact.all(-1) & clearance_ok
    if final_joint_margin_rad > 0:
        robust &= (margin[..., length // 2 :] >= final_joint_margin_rad).all(-1)
        robust &= (result.position_error.norm(dim=-1).reshape(*shape, length) <= 0.002).all(-1)
    safe = robust.any(1).all(0)
    travel = (arm - actual).norm(dim=-1).amax(-1)
    if diagnostics is not None:
        residual = result.position_error.norm(dim=-1).reshape(*shape, length).amax(-1)
        best_seed = residual.argmin(1, keepdim=True)
        per_joint_margin = torch.minimum(arm - limits[:, 0], limits[:, 1] - arm)
        best_joint_margin = (
            per_joint_margin[..., length // 2 :, :]
            .amin(-2)
            .gather(1, best_seed[..., None].expand(-1, -1, -1, 6))
            .squeeze(1)
        )
        diagnostics.update(
            actual_arm_joints_rad=actual.tolist(),
            arm_joint_limits_rad=limits.tolist(),
            closest_path_final_half_joint_margin_by_height_rad=best_joint_margin.tolist(),
            inverse_exact_path_by_height=exact.all(-1).any(1).tolist(),
            inverse_exact_points_by_height=exact.any(1).tolist(),
            inverse_best_position_error_points_by_height_m=result.position_error.norm(dim=-1)
            .reshape(*shape, length)
            .amin(1)
            .tolist(),
            inverse_safe_path_by_height=robust.any(1).tolist(),
            best_max_position_error_by_height_m=result.position_error.norm(dim=-1)
            .reshape(*shape, length)
            .amax(-1)
            .amin(1)
            .tolist(),
            best_minimum_arm_clearance_by_height_m=arm_clear.amin(-1).amax(1).tolist(),
            best_final_arm_clearance_by_height_m=arm_clear[..., -1].amax(1).tolist(),
            best_final_half_joint_margin_by_height_rad=margin[..., length // 2 :]
            .amin(-1)
            .amax(1)
            .tolist(),
            exact_path_candidate_counts_by_height=exact.all(-1).any(1).sum(-1).tolist(),
            joint_margin_candidate_counts_by_height=(
                margin[..., length // 2 :].amin(-1) >= final_joint_margin_rad
            )
            .any(1)
            .sum(-1)
            .tolist(),
        )
    return safe, travel.amin(1).amax(0)


def plan_clear_for_nudge(state, basket, limits, collision, geometry, dt, *, episode):
    world, _ = basket.world_mesh(state["root_states"][0])
    start = state["ee_pos"]
    cone, rotation = transverse_tilts(start)
    offsets = start.new_tensor([0, 0.04, 0.08, 0.12])
    bank = torch.cat(
        (cone.repeat_interleave(len(offsets), dim=0), offsets.repeat(len(cone))[:, None]), -1
    )
    end_q = _quat_multiply(state["ee_orn"], rotation.repeat_interleave(len(offsets), dim=0))
    root = state["root_states"][0]
    mount = root[:3] + _quat_rotate(root[3:7], start.new_tensor(read_arm_mount()[0]))
    direction = mount - start[0]
    direction[2] = 0
    direction /= direction.norm().clamp_min(1e-6)
    end = start + bank[:, 2, None] * direction
    end[:, 2] = torch.maximum(start[:, 2], world[:, 2].amax() + 0.16)
    endpoints = torch.stack((start.expand_as(end), end), 1)
    rotations = torch.stack((state["ee_orn"].expand_as(end_q), end_q), 1)
    target_mesh = build_distractor_mesh([episode], state["object_state"][:, None])
    target_geometry = augment_scene_geometry(geometry, target_mesh)
    target_geometry.nonrobot_geometry = augment_scene_geometry(
        geometry.nonrobot_geometry, target_mesh
    )
    target_geometry.basket_model = basket
    surface = SceneGeometry(target_mesh["points"], target_mesh["faces"], device=str(start.device))
    local = torch.cat(collision._gripper_local_surfaces(start))

    def release_clearance(points, quats):
        scene = collision.gripper_clearance(geometry, points, quats)
        target = surface.rigid_surface_clearance(
            points, quats, local, check_rays=False, signed_distance=True
        )
        return scene, target, (scene.amin(-1) >= 0.002) & monotone_release_mask(target)

    points, quats, _ = pose_segments(endpoints, rotations, dt, samples=17)
    scene_clearance, target_clearance, clear = release_clearance(points, quats)
    eligible = torch.where(clear)[0]
    diagnostics = {"valid": False, "candidate_count": len(bank), "inverse_calls": 0}
    if not len(eligible):
        diagnostics.update(
            scene_start_clearance_m=scene_clearance[:, 0].tolist(),
            scene_path_clearance_m=scene_clearance.amin(-1).tolist(),
            scene_end_clearance_m=scene_clearance[:, -1].tolist(),
            scene_clear_candidate_count=int((scene_clearance.amin(-1) >= 0.002).sum()),
            target_surface_start_clearance_m=target_clearance[:, 0].tolist(),
            target_surface_path_clearance_m=target_clearance.amin(-1).tolist(),
            target_surface_end_clearance_m=target_clearance[:, -1].tolist(),
        )
        return None, {**diagnostics, "failure": "no_open_jaw_clear_geometry"}
    inverse_ok, travel = safe_inverse_paths(
        state,
        points[eligible],
        quats[eligible],
        limits,
        collision,
        target_geometry,
        diagnostics=diagnostics,
    )
    diagnostics.update(inverse_calls=1, geometric_candidates=len(eligible))
    if not bool(inverse_ok.any()):
        return None, {**diagnostics, "failure": "no_open_jaw_clear_inverse_path"}
    cost = bank[eligible, 0] + 3 * bank[eligible, 2] + 0.001 * travel
    selected = eligible[cost.masked_fill(~inverse_ok, torch.inf).argmin()]
    points, quats, _ = pose_segments(
        endpoints[selected : selected + 1], rotations[selected : selected + 1], dt
    )
    scene_clearance, target_clearance, dense_safe = release_clearance(points, quats)
    if not bool(dense_safe.all()):
        return None, {**diagnostics, "failure": "open_jaw_clear_dense_geometry"}
    diagnostics.update(
        valid=True,
        reference_samples=points.shape[1],
        target_world=end[selected].tolist(),
        selected_tilt_rad_azimuth_rad_retract_m=bank[selected].tolist(),
        minimum_clearance_m=float(scene_clearance.min()),
        initial_target_surface_clearance_m=float(target_clearance[0, 0]),
        final_target_surface_clearance_m=float(target_clearance[0, -1]),
        target_contact_release_policy="monotone_signed_surface_separation",
    )
    return (points, quats), diagnostics


def shifted_grasp_counts(
    state, episode, targets, collision, geometry, support_points, *, limits=None, diagnostics=None
):
    """Evaluate all proposed shifts against the actual open/closing jaw geometry."""
    from gr00t.rl.pick.pocket import BASKET_MINIMUM_CENTER_DEPTH_M

    orientation = object_frame_grasp_bank(
        state["object_state"][:, 3:7],
        episode["asset"].get("intrinsic_size_m", episode["asset"]["size_m"]),
    )
    root = state["root_states"]
    target = state["object_state"][:, :3]
    heading = torch.atan2(target[:, 1] - root[:, 1], target[:, 0] - root[:, 0])
    _, natural, _ = grasp_pose_bank(
        target,
        heading,
        episode["asset"].get("intrinsic_size_m", episode["asset"]["size_m"]),
        grasp_approach_origin(root),
        reference_orientation=state["ee_orn"],
    )
    orientation = torch.cat((state["ee_orn"][:, None], natural, orientation), 1)
    axis = _quat_rotate(orientation, targets.new_tensor([1, 0, 0]))
    orientation = orientation[:, axis[0, :, 2] <= -math.cos(math.radians(30))]
    normal = _quat_rotate(state["root_states"][:, 3:7], targets.new_tensor([[0, 0, 1]]))
    goals, quality = pocket_pose_targets(
        orientation.expand(len(targets), -1, -1),
        targets,
        state["object_state"][:, 3:7].expand(len(targets), -1),
        support_points,
        collision,
        geometry,
        minimum_environment_clearance_m=BASKET_GRASP_CLEARANCE_M,
        minimum_center_depth_m=BASKET_MINIMUM_CENTER_DEPTH_M,
        support_normal_world=normal.expand(len(targets), -1),
        shared_object_geometry=True,
        retain_depth_candidates=limits is not None,
    )
    feasible = quality["pocket_fit"] & (quality["antipodal_margin"] >= 0)
    feasible &= quality["support_pinch_compatible"] & (
        quality["environment_clearance_m"] >= BASKET_GRASP_CLEARANCE_M
    )
    if diagnostics is not None:
        diagnostics["geometric_grasp_counts"] = feasible.sum(-1).tolist()
    if limits is not None:
        orientation = orientation.repeat_interleave(goals.shape[1] // orientation.shape[1], 1)
        target_ids, pose_ids = feasible.nonzero(as_tuple=True)
        reachable = torch.zeros_like(feasible)
        if len(target_ids):
            q = orientation[0, pose_ids]
            end = goals[target_ids, pose_ids]
            pitch = -math.pi / 6
            axis = _quat_rotate(q, q.new_tensor([math.cos(pitch), 0, -math.sin(pitch)]))
            standoffs = end.new_tensor([0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32])
            pre = end[:, None] - standoffs[None, :, None] * axis[:, None]
            endpoints = torch.stack((pre, end[:, None].expand_as(pre)), 2).flatten(0, 1)
            quats = q[:, None, None].expand(-1, len(standoffs), 2, -1).flatten(0, 1)
            safe, _ = safe_inverse_paths(
                state,
                endpoints,
                quats,
                limits,
                collision,
                geometry,
                body_height_offsets=(-0.01, 0.0, 0.01),
                final_joint_margin_rad=math.radians(2),
                minimum_arm_clearance_m=BASKET_GRASP_CLEARANCE_M,
            )
            reachable[target_ids, pose_ids] = safe.reshape(-1, len(standoffs)).any(-1)
        feasible &= reachable
        if diagnostics is not None:
            diagnostics["reachable_grasp_counts"] = feasible.sum(-1).tolist()
            diagnostics["reachability_definition"] = (
                "pregrasp_and_capture_at_measured_trunk_height_plus_minus_1cm; "
                "2mm_inverse_error_2deg_capture_joint_margin_6mm_arm_clearance"
            )
    return feasible.sum(-1)


def nudge_shift_bank(reference, samples=24):
    """Legacy rays plus a bounded, near-origin-dense disk cover for small corrections."""
    if samples < 24 or samples > 256:
        raise ValueError("Nudge shift sample count must be in [24, 256]")
    angle = torch.arange(8, device=reference.device, dtype=reference.dtype).repeat_interleave(3)
    angle = angle * (math.pi / 4)
    radius = reference.new_tensor([0.03, 0.06, 0.09]).repeat(8)
    if samples > 24:
        index = torch.arange(samples - 24, device=reference.device, dtype=reference.dtype)
        angle = torch.cat((angle, index * (math.pi * (3 - math.sqrt(5)))))
        radius = torch.cat((radius, 0.005 + 0.085 * (index + 0.5) / len(index)))
    return torch.stack((angle.cos(), angle.sin()), -1), radius


def slide_stability(root, object_state, support, center, hits, normals, directions, friction):
    """Compare sliding force and tipping moment about the actual support footprint."""
    inverse = root[3:7].clone()
    inverse[1:] *= -1
    world = object_state[:3] + _quat_rotate(object_state[3:7], support)
    points = _quat_rotate(inverse, world - root[:3])
    com = _quat_rotate(
        inverse, object_state[:3] + _quat_rotate(object_state[3:7], center) - root[:3]
    )
    contact = _quat_rotate(inverse, hits - root[:3])
    force = -_quat_rotate(inverse, normals)
    push = _quat_rotate(inverse, directions)
    horizontal = force[:, :2].norm(dim=-1).clamp_min(1e-8)
    axis = force[:, :2] / horizontal[:, None]
    bottom = points[:, 2].min()
    supported = points[:, 2] <= bottom + 0.001
    lever = ((points[None, :, :2] - com[:2]) * axis[:, None]).sum(-1)
    radius = lever.masked_fill(~supported[None], -torch.inf).amax(-1)
    vertical_ratio = force[:, 2] / horizontal
    height = contact[:, 2] - bottom
    offset = ((contact[:, :2] - com[:2]) * axis).sum(-1)
    effective_height = height - vertical_ratio * offset
    reserve = radius - friction * effective_height
    safe = ((axis * push[:, :2]).sum(-1) >= 0.9) & (vertical_ratio.abs() <= 0.25)
    safe &= (effective_height >= 0) & (reserve >= 0.006)
    return safe, reserve, radius, effective_height


def plan_closed_nudge(
    state,
    episode,
    basket,
    limits,
    collision,
    geometry,
    dt,
    *,
    object_support_points=None,
    contact_height_fractions=(0.35, 0.70, 0.90),
    current_path_unavailable=False,
    shift_samples=24,
    require_sliding_stability=False,
):
    """Choose an improving free-space direction before doing any physical pushing."""
    if not contact_height_fractions or any(
        not math.isfinite(value) or abs(value) >= 1 for value in contact_height_fractions
    ):
        raise ValueError("Contact heights must be nonempty finite fractions inside (-1, 1)")
    root = state["root_states"][0]
    device, dtype = root.device, root.dtype
    centers, extents = object_boxes_in_trunk(episode, state)
    local_d, shifts = nudge_shift_bank(root, shift_samples)
    end_center = centers[0, :2] + local_d * shifts[:, None]
    before = bbox_space(centers[0, :2], extents[0, :2], basket, centers[1:, :2], extents[1:, :2])
    space = bbox_space(end_center, extents[0, :2], basket, centers[1:, :2], extents[1:, :2])
    sweep = (
        centers[0, :2]
        + torch.linspace(0, 1, 9, device=device)[None, :, None]
        * local_d[:, None]
        * shifts[:, None, None]
    )
    sweep_space = bbox_space(sweep, extents[0, :2], basket, centers[1:, :2], extents[1:, :2]).amin(
        -1
    )
    good_direction = (space >= 0.006) & (
        sweep_space >= torch.minimum(before, before.new_tensor(-0.001)) - 1e-6
    )
    improving = improving_shift_mask(before, space, sweep_space)
    ids = torch.where(good_direction)[0]
    if not len(ids):
        return None, {
            "failure": "no_improving_collision_free_basket_shift",
            "not_required": bool(before >= 0.006) and not current_path_unavailable,
            "space_before_m": float(before),
            "space_after_m": space.tolist(),
        }
    direction = _quat_rotate(
        root[None, 3:7], torch.cat((local_d[ids], local_d.new_zeros(len(ids), 1)), -1)
    )
    target = state["object_state"][0, :3]
    if object_support_points is None:
        object_support_points = load_grasp_support(episode["asset"]["usd_path"], device)
    access_counts = shifted_grasp_counts(
        state,
        episode,
        torch.cat((target[None], target[None] + shifts[ids, None] * direction)),
        collision,
        geometry,
        object_support_points,
        limits=limits,
    )
    current_grasp_count, grasp_counts = access_counts[0], access_counts[1:]
    worthwhile = (grasp_counts > 0) & (
        (current_grasp_count == 0) | improving[ids] | current_path_unavailable
    )
    if not bool(worthwhile.any()):
        return None, {
            "failure": "no_shift_improves_grasp_access_or_space",
            "not_required": bool(before >= 0.006) and not current_path_unavailable,
            "space_before_m": float(before),
            "current_geometric_grasp_count": int(current_grasp_count),
            "current_grasp_path_known_unavailable": current_path_unavailable,
            "predicted_grasp_counts_by_shift": grasp_counts.tolist(),
            "candidate_shift_distances_m": shifts[ids].tolist(),
        }
    direction, grasp_counts, ids = direction[worthwhile], grasp_counts[worthwhile], ids[worthwhile]
    target_mesh = build_distractor_mesh([episode], state["object_state"][:, None])
    surface = SceneGeometry(target_mesh["points"], target_mesh["faces"], device=str(device))
    # Contact the upper flank, reaching from above the basket's interior.
    combinations = torch.cartesian_prod(
        torch.arange(len(ids), device=device),
        torch.arange(4, device=device),
        torch.arange(8, device=device),
        torch.arange(len(contact_height_fractions), device=device),
    )
    d = direction[combinations[:, 0]]
    tilt = root.new_tensor([0, 20, 35, 50])[combinations[:, 1]] * (math.pi / 180)
    yaw = torch.atan2(-d[:, 1], -d[:, 0])
    rotation = _euler_xyz_quat(
        torch.stack((combinations[:, 2].to(dtype) * (math.pi / 4), math.pi / 2 - tilt, yaw), -1)
    )
    contact_z = (
        target[2]
        + root.new_tensor(episode["asset"]["size_m"][2])
        / 2
        * root.new_tensor(contact_height_fractions)[combinations[:, 3]]
    )
    origins = target[None].expand(len(d), -1).clone() - 0.20 * d
    origins[:, 2] = contact_z
    hits, normal, _ = surface.rays(origins, origins + 0.40 * d)
    hit_ok = normal.norm(dim=-1) > 0.9
    fixed, moving = collision._gripper_local_surfaces(state["ee_pos"])
    close_q = root.new_tensor([math.cos(math.pi / 6), 0, math.sin(math.pi / 6), 0])
    closed = torch.cat((fixed, _quat_rotate(close_q, moving)))
    tip = closed[closed[:, 0] >= closed[:, 0].max() - 0.01]
    rotated = _quat_rotate(rotation[:, None], tip[None])
    leading = (rotated * d[:, None]).sum(-1).argmax(-1)
    contact = hits - rotated[torch.arange(len(d), device=device), leading]
    pre = contact - d * 0.008
    end = contact + d * shifts[ids[combinations[:, 0]], None]
    world, _ = basket.world_mesh(root)
    hover = pre.clone()
    hover[:, 2] = torch.maximum(hover[:, 2], world[:, 2].amax() + 0.16)
    # Withdraw opposite the push while rising to the already checked hover pose.
    away = hover.clone()
    endpoints = torch.stack((state["ee_pos"].expand(len(d), -1), hover, pre, end, away), 1)
    rotations = torch.cat(
        (state["ee_orn"][:, None].expand(len(d), 1, -1), rotation[:, None].expand(-1, 4, -1)), 1
    )
    points, quats, bounds = pose_segments(endpoints, rotations, dt, samples=9)
    clear = collision.gripper_pose_clearance(
        geometry, points, quats, points.new_zeros(points.shape[:2])
    ).amin(-1)
    target_geometry = augment_scene_geometry(geometry, target_mesh)
    before_contact = collision.gripper_pose_clearance(
        target_geometry,
        points[:, : bounds[2]],
        quats[:, : bounds[2]],
        points.new_zeros(len(d), bounds[2]),
    ).amin(-1)
    geometry_ok = hit_ok & (clear >= 0.003) & (before_contact >= 0.002)
    stability_report = {"sliding_stability_required": require_sliding_stability}
    if require_sliding_stability:
        from gr00t.rl.pick.grasp_quality import load_grasp_center
        from gr00t.rl.pick.pocket import GRIP_STATIC_FRICTION

        center = load_grasp_center(episode["asset"]["usd_path"], str(device))
        friction = (GRIP_STATIC_FRICTION + basket.floor_static_friction) / 2
        stable, reserve, radius, height = slide_stability(
            root, state["object_state"][0], object_support_points, center, hits, normal, d, friction
        )
        stability_report.update(
            effective_floor_static_friction=friction,
            sliding_stable_candidate_count=int((geometry_ok & stable).sum()),
            tipping_lever_reserve_m=reserve.tolist(),
            support_footprint_radius_m=radius.tolist(),
            effective_contact_height_m=height.tolist(),
        )
        geometry_ok &= stable
    eligible = torch.where(geometry_ok)[0]
    if not len(eligible):
        return None, {
            **stability_report,
            "failure": "no_closed_jaw_flank_entry",
            "predicted_grasp_counts_by_shift": grasp_counts.tolist(),
            "candidate_shift_distances_m": shifts[ids].tolist(),
            "candidate_count": len(d),
            "surface_hit_candidate_count": int(hit_ok.sum()),
            "scene_clear_candidate_count": int((clear >= 0.003).sum()),
            "target_entry_clear_candidate_count": int((before_contact >= 0.002).sum()),
            "scene_clearance_m": clear.tolist(),
            "target_entry_clearance_m": before_contact.tolist(),
        }
    inverse_diagnostics = {}
    inverse_ok, travel = safe_inverse_paths(
        state,
        points[eligible],
        quats[eligible],
        limits,
        collision,
        geometry,
        diagnostics=inverse_diagnostics,
    )
    future_feasible = grasp_counts[combinations[eligible, 0]] > 0
    score = -space[ids[combinations[eligible, 0]]] + 0.001 * travel
    score += shifts[ids[combinations[eligible, 0]]]
    inverse_ok &= future_feasible
    found = bool(inverse_ok.any())
    if not found:
        return None, {
            **stability_report,
            "failure": "no_closed_nudge_inverse_path",
            "predicted_grasp_counts_by_shift": grasp_counts.tolist(),
            "candidate_shift_distances_m": shifts[ids].tolist(),
            "inverse_calls": 1,
            "geometric_candidates": len(eligible),
            "inverse_path": inverse_diagnostics,
            "planning_segment_boundaries": bounds,
        }
    selected = eligible[score.masked_fill(~inverse_ok, torch.inf).argmin()]
    points, quats, bounds = pose_segments(
        endpoints[selected : selected + 1],
        rotations[selected : selected + 1],
        dt,
        contact_segment=2,
    )
    dense_clear = collision.gripper_pose_clearance(
        geometry, points, quats, points.new_zeros(points.shape[:2])
    ).amin()
    if not bool(dense_clear >= 0.003):
        return None, {
            "failure": "closed_nudge_dense_clearance_failed",
            "clearance_m": float(dense_clear),
            "inverse_calls": 1,
        }
    selected_direction = ids[combinations[selected, 0]]
    diagnostics = {
        **stability_report,
        "selected": int(selected),
        "inverse_calls": 1,
        "candidate_count": len(d),
        "direction_world": d[selected].tolist(),
        "shift_m": float(shifts[selected_direction]),
        "predicted_grasp_counts_by_shift": grasp_counts.tolist(),
        "selected_predicted_grasp_count": int(grasp_counts[combinations[selected, 0]]),
        "current_geometric_grasp_count": int(current_grasp_count),
        "current_grasp_path_known_unavailable": current_path_unavailable,
        "shift_selection_policy": "future_open_and_closing_jaw_clearance_before_bbox_gap",
        "space_before_m": float(before),
        "space_after_m": float(space[selected_direction]),
        "closed_gripper_clearance_m": float(dense_clear),
        "reference_samples": points.shape[1],
        "segment_boundaries": bounds,
        "contact_target_world": hits[selected].tolist(),
        "selected_tool_tilt_deg": float(torch.rad2deg(tilt[selected])),
        "selected_tool_roll_deg": int(combinations[selected, 2]) * 45,
        "selected_contact_height_fraction": float(
            root.new_tensor(contact_height_fractions)[combinations[selected, 3]]
        ),
    }
    return (points, quats), diagnostics
