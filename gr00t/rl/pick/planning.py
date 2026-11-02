"""Batched grasp candidates and collision-checked trajectories through shared inverse."""

import math

import torch
from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate, _slerp, align_vectors
from gr00t.rl.auto_open.pose_planning import best_feasible, solve_pose_targets
from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
from gr00t.rl.pick.grasp_clearance import GRASP_CLEARANCE_M
from gr00t.rl.pick.grasp_quality import convex_grasp_quality
from gr00t.rl.pick.pocket import GRASP_FRICTION, gripper_pocket_geometry
from gr00t.rl.utils.b2z1_asset import read_arm_mount

BALANCED_GRASP_PITCH_OFFSETS_DEG = (0.0, -5.0, 5.0, -10.0, 10.0)
GRASP_BODY_HEIGHTS_M = (0.18, 0.22, 0.30, 0.38, 0.46, 0.536, 0.58)
GRASP_TRANSIT_SPEED_M_S = 0.20
GRASP_CAPTURE_SPEED_M_S = 0.08
GRASP_LIFT_SPEED_M_S = 0.15
GRASP_WITHDRAW_SPEED_M_S = 0.15
MAX_POCKET_SAMPLE_EVALUATIONS = 4_000_000


def grasp_body_height_hypotheses(roots):
    return torch.cat(
        (
            roots.new_tensor(GRASP_BODY_HEIGHTS_M).expand(len(roots), -1),
            roots[:, 2:3] + roots.new_tensor([-0.01, 0, 0.01]),
        ),
        -1,
    )


def object_frame_grasp_bank(object_orientation, object_size=None):
    """Scan-relative orientations keep narrow feasible pinches independent of world yaw."""
    reference = object_orientation
    roll, pitch, yaw = torch.meshgrid(
        torch.arange(8, device=reference.device, dtype=reference.dtype) * (math.pi / 4),
        torch.arange(-3, 4, device=reference.device, dtype=reference.dtype) * (math.pi / 6),
        torch.arange(8, device=reference.device, dtype=reference.dtype) * (math.pi / 4),
        indexing="ij",
    )
    local = _euler_xyz_quat(torch.stack((roll, pitch, yaw), -1).reshape(-1, 3))
    local = local[None].expand(len(reference), -1, -1)
    if object_size is not None:
        center = reference.new_zeros(len(reference), 3)
        _, balanced, _ = grasp_pose_bank(
            center,
            center[:, 0],
            object_size,
            center + reference.new_tensor([0, 0, 0.5]),
            reference_orientation=reference.new_tensor([[1, 0, 0, 0]]).expand(len(reference), -1),
        )
        local = torch.cat((local, balanced), 1)
    return _quat_multiply(reference[:, None], local)


def geometric_grasp_mask(
    orientations, quality, clearance, support_normal, excluded, *, annotate=True, excluded_mask=None
):
    mask = (
        quality["pocket_fit"]
        & (quality["antipodal_margin"] >= 0)
        & (quality["environment_clearance_m"] >= clearance)
    )
    if support_normal is not None:
        from gr00t.rl.pick.grasp_quality import supported_pinch_alignment

        fit, sine = supported_pinch_alignment(
            orientations,
            quality["fixed_contact_local"],
            quality["moving_contact_local"],
            support_normal,
        )
        if annotate:
            quality["support_pinch_compatible"] = fit
            quality["support_contact_line_sine"] = sine
        mask &= fit
    if excluded is not None and excluded.shape[-2]:
        similarity = (orientations @ excluded.transpose(-1, -2)).abs()
        rejected = similarity > math.cos(math.radians(0.5))
        if excluded_mask is not None:
            rejected &= excluded_mask[..., None, :]
        mask &= ~rejected.any(-1)
    return mask


def retime_grasp_approach_batch(
    positions,
    orientations,
    target,
    dt,
    *,
    transit_speed_m_s=GRASP_TRANSIT_SPEED_M_S,
    capture_speed_m_s=GRASP_CAPTURE_SPEED_M_S,
    capacity=4096,
):
    """GPU rest-to-rest paths with independent lengths and terminal padding."""
    if capacity < 2 or positions.ndim != 3 or positions.shape[1] < 2:
        raise ValueError("Retiming requires a capacity and path of at least two points")
    if not all(
        math.isfinite(value) and value > 0 for value in (dt, transit_speed_m_s, capture_speed_m_s)
    ):
        raise ValueError("Retiming timestep and speed limits must be positive")
    if orientations.shape != (*positions.shape[:-1], 4) or target.shape != (len(positions), 3):
        raise ValueError("Retiming pose and target batches must agree")
    distance = (positions[:, 1:] - positions[:, :-1]).norm(dim=-1)
    near = (positions[:, 1:] - target[:, None]).norm(dim=-1) < 0.05
    speed = torch.where(near, capture_speed_m_s, transit_speed_m_s)
    angle = 2 * (orientations[:, 1:] * orientations[:, :-1]).sum(-1).abs().clamp_max(1).acos()
    segment_time = torch.maximum(distance / speed, angle).clamp_min(1e-6)
    timeline = torch.cat((segment_time.new_zeros(len(positions), 1), segment_time.cumsum(-1)), -1)
    lengths = (1.875 * timeline[:, -1].double() / dt).ceil().long().clamp_min(1) + 1
    valid = (lengths <= capacity) & torch.isfinite(timeline[:, -1])
    lengths = lengths.clamp(2, capacity)
    indices = torch.arange(capacity, device=positions.device)[None]
    t = (indices / (lengths[:, None] - 1)).clamp_max(1).to(positions.dtype)
    t = t.pow(3) * (10 + t * (-15 + 6 * t))
    times = (timeline[:, -1:] * t).contiguous()
    right = torch.searchsorted(timeline.contiguous(), times, right=True).clamp(
        1, positions.shape[1] - 1
    )
    left = right - 1
    fraction = (times - timeline.gather(1, left)) / segment_time.gather(1, left)
    p0 = positions.gather(1, left[..., None].expand(-1, -1, 3))
    p1 = positions.gather(1, right[..., None].expand(-1, -1, 3))
    q0 = orientations.gather(1, left[..., None].expand(-1, -1, 4))
    q1 = orientations.gather(1, right[..., None].expand(-1, -1, 4))
    p = p0 + fraction[..., None] * (p1 - p0)
    q = _slerp(q0, q1, fraction)
    terminal = indices >= lengths[:, None] - 1
    p = torch.where(terminal[..., None], positions[:, -1:], p)
    q = torch.where(terminal[..., None], orientations[:, -1:], q)
    p[:, 0], q[:, 0] = positions[:, 0], orientations[:, 0]
    return (p, q, lengths), valid


def retime_grasp_approach(positions, orientations, target, dt, **options):
    """Single-scene planner adapter; multi-environment execution uses padded paths."""
    if len(positions) != 1:
        raise ValueError("Multi-environment retiming requires retime_grasp_approach_batch")
    path, valid = retime_grasp_approach_batch(positions, orientations, target, dt, **options)
    # This adapter is used by the remaining scalar planners, not a batch time law.
    status = torch.stack((path[2][0], valid[0].long())).tolist()
    if not status[1]:
        raise ValueError("Retimed path exceeds its capacity or has nonfinite duration")
    return path[0][:, : status[0]], path[1][:, : status[0]]


def grasp_aperture_angle(object_size):
    if torch.is_tensor(object_size):
        pocket = gripper_pocket_geometry(str(object_size.device), object_size.dtype)
        radius = object_size[..., :2].amin(-1) / 2
        x = pocket["front_x"] - radius - 0.0015
        z = radius + pocket["fixed_plane_z"]
        return (
            torch.atan2(z, x)
            + ((radius + pocket["moving_plane_offset"]) / torch.hypot(x, z)).clamp_max(0.95).asin()
        )
    pocket = gripper_pocket_geometry()
    radius = min(object_size[:2]) / 2
    center_x = float(pocket["front_x"]) - radius - 0.0015
    center_z = radius + float(pocket["fixed_plane_z"])
    lever_radius = math.hypot(center_x, center_z)
    return math.atan2(center_z, center_x) + math.asin(
        min(0.95, (radius + float(pocket["moving_plane_offset"])) / lever_radius)
    )


def minimum_grip_normal_force(
    object_size, mass_kg, friction=GRASP_FRICTION, safety_factor=2.0, *, aperture_angle_rad=None
):
    """Per-jaw normal force for a symmetric hinged grasp resisting gravity."""
    half_angle = (
        grasp_aperture_angle(object_size) if aperture_angle_rad is None else aperture_angle_rad
    ) / 2
    uplift_per_normal = friction * math.cos(half_angle) - math.sin(half_angle)
    if uplift_per_normal <= 0:
        raise ValueError("Object aperture is outside the configured gripping friction cone")
    return max(3.0, safety_factor * mass_kg * 9.81 / (2 * uplift_per_normal))


def grasp_approach_origin(roots):
    return roots[:, :3] + _quat_rotate(
        roots[:, 3:7], roots.new_tensor(read_arm_mount()[0]).expand(len(roots), -1)
    )


def balanced_grasp_orientations(
    headings, sizes, pitch_offsets_deg=BALANCED_GRASP_PITCH_OFFSETS_DEG, *, candidate_indices=None
):
    """Shared hinged-jaw bisectors; optional indices avoid expanding the full bank."""
    offsets = headings.new_tensor(pitch_offsets_deg) * (math.pi / 180)
    yaw_offsets = headings.new_tensor([0, -45, 45, -90, 90, -135, 135, 180]) * (math.pi / 180)
    if candidate_indices is None:
        candidate_indices = torch.arange(len(offsets) * 16, device=headings.device)[None].expand(
            len(headings), -1
        )
    if candidate_indices.ndim != 2 or candidate_indices.shape[0] != len(headings):
        raise ValueError("Balanced orientation indices require (environments, choices)")
    aperture = grasp_aperture_angle(sizes)[:, None]
    opposite = candidate_indices.div(8, rounding_mode="floor").remainder(2).bool()
    pitch = torch.where(opposite, math.pi / 2 + aperture / 2, math.pi / 2 - aperture / 2)
    pitch = pitch + offsets[candidate_indices.div(16, rounding_mode="floor")]
    yaw = headings[:, None] + yaw_offsets[candidate_indices.remainder(8)]
    roll = torch.where(opposite, torch.zeros_like(pitch), torch.full_like(pitch, math.pi))
    return _euler_xyz_quat(torch.stack((roll, pitch, yaw), -1))


def grasp_pose_bank(
    object_centers,
    headings,
    object_size,
    approach_origins,
    balanced_pitch_offsets_deg=BALANCED_GRASP_PITCH_OFFSETS_DEG,
    *,
    pitch_step_deg=15.0,
    reference_orientation=None,
):
    """Candidate directions vary continuously with arm-to-object elevation."""
    toward = object_centers - approach_origins
    yaw_options = object_centers.new_tensor([0, -45, 45, -90, 90, -135, 135, 180]) * (math.pi / 180)
    if reference_orientation is None:
        roll = object_centers.new_tensor([0, -45, 45, -90, 90, -135, 135, 180]) * (math.pi / 180)
        preferred_pitch = torch.atan2(-toward[:, 2], toward[:, :2].norm(dim=-1))
        pitch = torch.arange(
            -60, 60.0001, pitch_step_deg, device=object_centers.device, dtype=object_centers.dtype
        ) * (math.pi / 180)
        roll, pitch, yaw_offsets = torch.meshgrid(roll, pitch, yaw_options, indexing="ij")
        roll, pitch, yaw_offsets = roll.flatten(), pitch.flatten(), yaw_offsets.flatten()
        yaw = headings[:, None] + yaw_offsets
        pitch = (preferred_pitch[:, None] + pitch).clamp(-math.pi / 6, 5 * math.pi / 6)
        orientation = _euler_xyz_quat(torch.stack((roll[None].expand_as(yaw), pitch, yaw), -1))
    else:
        source = _quat_rotate(
            reference_orientation, object_centers.new_tensor([1, 0, 0]).expand_as(toward)
        )
        center = align_vectors(source, toward, reference_orientation)
        radius, azimuth = torch.meshgrid(
            torch.arange(
                pitch_step_deg,
                60.0001,
                pitch_step_deg,
                device=object_centers.device,
                dtype=object_centers.dtype,
            )
            * (math.pi / 180),
            torch.arange(8, device=object_centers.device, dtype=object_centers.dtype)
            * (math.pi / 4),
            indexing="ij",
        )
        radius, azimuth = radius.flatten(), azimuth.flatten()
        local = torch.stack(
            (radius.cos(), radius.sin() * azimuth.cos(), radius.sin() * azimuth.sin()), -1
        )
        local = torch.cat((local.new_tensor([[1, 0, 0]]), local))
        axes = _quat_rotate(center[:, None], local[None])
        axes = torch.cat((axes, axes.new_tensor([[[0, 0, -1]]]).expand(len(axes), -1, -1)), 1)
        reference = reference_orientation[:, None].expand(-1, axes.shape[1], -1)
        aligned = align_vectors(
            source[:, None].expand_as(axes).flatten(0, 1),
            axes.flatten(0, 1),
            reference.flatten(0, 1),
        ).reshape_as(reference)
        roll = object_centers.new_tensor(
            [
                0,
                -22.5,
                22.5,
                -45,
                45,
                -67.5,
                67.5,
                -90,
                90,
                -112.5,
                112.5,
                -135,
                135,
                -157.5,
                157.5,
                180,
            ]
        ) * (math.pi / 180)
        twist = torch.stack(((roll / 2).cos(), (roll / 2).sin(), roll * 0, roll * 0), -1)
        orientation = _quat_multiply(aligned[:, :, None], twist[None, None]).flatten(1, 2)
    # The moving jaw is hinged, not parallel. Place the closing-plane bisector
    # horizontally; friction still has to resist the wedge's ejection force.
    sizes = torch.as_tensor(object_size, device=object_centers.device, dtype=object_centers.dtype)
    if sizes.ndim == 1:
        sizes = sizes[None].expand(len(object_centers), -1)
    if sizes.shape != (len(object_centers), 3):
        raise ValueError("Object sizes must have shape (3,) or (environments, 3)")
    balanced = balanced_grasp_orientations(headings, sizes, balanced_pitch_offsets_deg)
    orientation = torch.cat((balanced, orientation), dim=1)
    if reference_orientation is not None:
        # A top-down pinch has a continuous axial degree of freedom. Refine it
        # uniformly without multiplying every other approach-direction sample.
        roll_index = torch.arange(1, 64, device=object_centers.device)
        roll_index = roll_index[roll_index.remainder(4) != 0]
        fine_roll = roll_index.to(object_centers.dtype) * (2 * math.pi / 64)
        fine_twist = torch.stack(
            ((fine_roll / 2).cos(), (fine_roll / 2).sin(), fine_roll * 0, fine_roll * 0), -1
        )
        fine_vertical = _quat_multiply(aligned[:, -1:, None], fine_twist[None, None]).flatten(1, 2)
        orientation = torch.cat((orientation, fine_vertical), 1)
    pocket = gripper_pocket_geometry(str(object_centers.device), object_centers.dtype)
    radius = sizes[:, :2].amin(-1) / 2
    offset = torch.stack(
        (pocket["back_x"] + radius + 0.0015, radius * 0, pocket["fixed_plane_z"] + radius),
        -1,
    )
    offset = offset[:, None].expand(*orientation.shape[:-1], 3)
    position = object_centers[:, None] - _quat_rotate(orientation, offset)
    axis = _quat_rotate(orientation, object_centers.new_tensor([1, 0, 0]).expand_as(position))
    return position, orientation, axis


def inverse_bank(positions, orientations, roots, initial_joints, limits):
    if positions.ndim < 3 or positions.shape[-1] != 3:
        raise ValueError("Inverse positions require candidate/path dimensions ending in xyz")
    if orientations.shape != (*positions.shape[:-1], 4):
        raise ValueError("Inverse orientations must match candidate/path dimensions")
    if roots.ndim == positions.ndim - 1:
        roots = roots[..., None, :].expand(*positions.shape[:-1], 13)
    if roots.shape != (*positions.shape[:-1], 13):
        raise ValueError("Inverse body poses must match the candidate/path dimensions")
    if initial_joints.shape != (*positions.shape[:-2], 6):
        raise ValueError("Inverse joint seeds must match the candidate dimensions")
    return solve_pose_targets(
        positions.reshape(-1, 3),
        orientations.reshape(-1, 4),
        base_position_world=roots[..., :3].reshape(-1, 3),
        base_orientation_world_xyzw=roots[..., [4, 5, 6, 3]].reshape(-1, 4),
        initial_arm_joint_positions=initial_joints[..., None, :]
        .expand(*positions.shape[:-1], 6)
        .reshape(-1, 6),
        joint_limits=limits,
    )


def plan_carry_batch(
    state,
    displacement,
    initial_object_height,
    aperture,
    limits,
    collision,
    geometry,
    dt,
    *,
    minimum_object_height_m=None,
    capacity=4096,
):
    """One GPU submission for every row's seated carry, without host decisions."""
    start, orientation = state["ee_pos"], state["ee_orn"]
    root = state["root_states"]
    batch = len(root)
    object_height = state["object_state"][:, 2]
    required = torch.maximum(
        initial_object_height + 0.10, object_height.new_full(object_height.shape, 0.30)
    )
    if minimum_object_height_m is not None:
        required = torch.maximum(
            required,
            torch.as_tensor(minimum_object_height_m, device=required.device, dtype=required.dtype),
        )
    end = start + displacement
    raise_by = (required + 0.005 - object_height).clamp_min(0)
    end[:, 2] = torch.maximum(start[:, 2] + raise_by, start.new_full((len(start),), 0.305))
    lengths = (1.875 * (end - start).norm(dim=-1) / (GRASP_LIFT_SPEED_M_S * dt)).ceil().long() + 1
    lengths = lengths.clamp_min(2)
    within_capacity = lengths <= capacity
    lengths = lengths.clamp_max(capacity)
    indices = torch.arange(capacity, device=start.device)[None]
    active = indices < lengths[:, None]
    time = (indices / (lengths[:, None] - 1)).clamp_max(1).to(start.dtype)
    time = time.pow(3) * (10 + time * (-15 + 6 * time))
    positions = start[:, None] + time[..., None] * (end - start)[:, None]
    quats = orientation[:, None].expand(-1, capacity, -1)
    sample = (torch.linspace(0, 1, 17, device=start.device)[None] * (lengths[:, None] - 1)).long()
    heights = grasp_body_height_hypotheses(root)
    roots = root[:, None].expand(-1, heights.shape[1], -1).clone()
    roots[..., 2] = heights
    actual = state["dof_pos"][:, collision.arm_joint_ids]
    seeds = torch.stack((actual, limits.mean(-1).expand_as(actual), limits.sum(-1) - actual), 1)
    rows = torch.arange(batch, device=start.device)[:, None]
    hypotheses = heights.shape[1] * seeds.shape[1]
    poses = positions[rows, sample][:, None].expand(-1, hypotheses, -1, -1)
    rotations = quats[rows, sample][:, None].expand(-1, hypotheses, -1, -1)
    inverse_roots = (
        roots[:, :, None].expand(-1, -1, seeds.shape[1], -1).reshape(batch, hypotheses, 13)
    )
    result = inverse_bank(
        poses.reshape(-1, 17, 3),
        rotations.reshape(-1, 17, 4),
        inverse_roots.reshape(-1, 13),
        seeds[:, None].expand(-1, heights.shape[1], -1, -1).reshape(-1, 6),
        limits,
    )
    joints = result.arm_joint_positions.reshape(batch, -1, 6)
    arm_roots = inverse_roots[:, :, None].expand(-1, -1, 17, -1).reshape(batch, -1, 13)
    arm_clear = collision.clearance(
        geometry, arm_roots, joints, maximum_distance_m=0.0021, include_gripper=False
    )
    safe = result.exact.reshape(batch, -1) & (arm_clear >= 0.002)
    coverage = safe.reshape(batch, hypotheses, 17).any(1)
    clearance = collision.gripper_pose_clearance(
        geometry,
        positions,
        quats,
        aperture[:, None].expand(-1, capacity),
        **({"query_mask": active} if getattr(geometry, "supports_query_mask", False) else {}),
    )
    # A seated jaw can already be closer than the planning margin to its support.
    # Permit only a collision-free escape with nondecreasing clearance until 2 mm.
    recovery_clearance = clearance.clamp_max(0.002)
    clearance_safe = (
        ((clearance > 0) | ~active).all(-1)
        & ((recovery_clearance.diff(dim=1) >= -1e-6) | ~active[:, 1:]).all(-1)
        & (clearance.gather(1, lengths[:, None] - 1)[:, 0] >= 0.002)
    )
    diagnostics = {
        "start_ee_world": start.clone(),
        "end_ee_world": end,
        "seated_object_height_world_m": object_height.clone(),
        "required_object_height_world_m": required,
        "target_object_height_world_m": object_height + end[:, 2] - start[:, 2],
        "measured_object_ee_offset_world_m": state["object_state"][:, :3] - start,
        "reference_samples": lengths,
        "inverse_calls": 1,
        "inverse_samples": hypotheses * 17,
        "safe_inverse_sample_mask": coverage,
        "gripper_min_clearance_m": clearance.masked_fill(~active, torch.inf).amin(-1),
        "gripper_initial_clearance_m": clearance[:, 0],
        "gripper_final_clearance_m": clearance.gather(1, lengths[:, None] - 1)[:, 0],
        "gripper_clearance_safe": clearance_safe,
        "within_path_capacity": within_capacity,
        "gripper_collision_configuration": "measured_held_aperture",
        "speed_limit_m_s": GRASP_LIFT_SPEED_M_S,
    }
    valid = coverage.all(-1) & clearance_safe & within_capacity
    return (positions, quats, lengths), valid, diagnostics


def plan_carry_path(*args, **kwargs):
    """Single-scene inspection tools serialize the same GPU planner's result."""
    path, valid, tensors = plan_carry_batch(*args, **kwargs)
    if len(valid) != 1:
        raise ValueError("Use plan_carry_batch for multi-environment planning")
    scalar_keys = {"reference_samples", "gripper_min_clearance_m", "gripper_clearance_safe"}
    diagnostics = {
        key: (value[0].item() if key in scalar_keys else value.cpu().tolist())
        if torch.is_tensor(value)
        else value
        for key, value in tensors.items()
    }
    if not bool(valid[0]):
        diagnostics["failure"] = "seated_carry_path_geometry_or_inverse_failed"
        return None, diagnostics
    count = int(path[2][0])
    return (path[0][:, :count], path[1][:, :count], path[2]), diagnostics


def joint_fk_approach(root, initial_joints, endpoints, heights, samples):
    """Respect finite joint travel, rather than wrapping wrist rotation through its stop."""
    t = torch.linspace(0, 1, samples, device=root.device, dtype=root.dtype)
    t = t.pow(3) * (10 + t * (-15 + 6 * t))
    while initial_joints.ndim < endpoints.ndim:
        initial_joints = initial_joints.unsqueeze(-2)
    while root.ndim < endpoints.ndim:
        root = root.unsqueeze(-2)
    root = root.expand(*endpoints.shape[:-1], 13)
    joints = initial_joints[..., None, :] + t[:, None] * (endpoints - initial_joints)[..., None, :]
    roots = root[..., None, :].expand(*endpoints.shape[:-1], samples, 13).clone()
    roots[..., 2] += t * (heights - root[..., 2])[..., None]
    p, rotation, _ = b2z1_gripper_pose_and_jacobian(
        roots[..., :3], roots[..., [4, 5, 6, 3]], joints
    )
    return p, _matrix_to_quaternion_wxyz(rotation), joints, roots


def elevated_cartesian_approach(
    state, pregrasp, orientation, heights, limits, collision, geometry, samples
):
    """Bounded lift/turn/descend alternatives, certified before physical probing."""
    candidates = len(pregrasp)
    t = torch.linspace(0, 3, samples, device=pregrasp.device, dtype=pregrasp.dtype)
    fraction = (t % 1)[None, :, None]
    fraction[:, -1] = 1
    stage = t.long().clamp_max(2)
    levels = pregrasp.new_tensor([0.35, 0.45, 0.55, 0.65])
    start = state["ee_pos"].expand(candidates, -1)
    start_q = state["ee_orn"].expand(candidates, -1)
    high_start = start[None].expand(len(levels), -1, -1).clone()
    high_start[..., 2] = torch.maximum(high_start[..., 2], levels[:, None])
    high_end = pregrasp[None].expand_as(high_start).clone()
    high_end[..., 2] = high_start[..., 2]
    knots = torch.stack(
        (start.expand_as(high_start), high_start, high_end, pregrasp.expand_as(high_start)), 2
    )
    rotations = torch.stack((start_q, start_q, orientation, orientation), 1)
    p = knots[:, :, stage] + fraction * (knots[:, :, stage + 1] - knots[:, :, stage])
    q = _slerp(rotations[:, stage], rotations[:, stage + 1], fraction[..., 0])
    q = q[None].expand(len(levels), -1, -1, -1)
    roots = state["root_states"][:, None].expand(candidates, samples, -1).clone()
    descent = (t - 2).clamp(0, 1)
    roots[..., 2] += descent[None] * (heights - roots[:, 0, 2])[:, None]
    initial = state["dof_pos"][:, collision.arm_joint_ids].expand(candidates, -1)
    seeds = torch.stack((initial, limits.mean(-1).expand_as(initial), limits.sum(-1) - initial))
    shape = (len(seeds), len(levels), candidates, samples)
    all_roots = roots[None, None].expand(*shape, 13).reshape(-1, samples, 13)
    result = inverse_bank(
        p[None].expand(*shape, 3).reshape(-1, samples, 3),
        q[None].expand(*shape, 4).reshape(-1, samples, 4),
        all_roots,
        seeds[:, None].expand(*shape[:3], 6).reshape(-1, 6),
        limits,
    )
    joints = result.arm_joint_positions.reshape(*shape, 6)
    if hasattr(geometry, "select"):
        if len(state["root_states"]) == 1 and candidates > 1:
            geometry = geometry.select(torch.zeros(candidates, device=p.device, dtype=torch.long))
        arm_gap = collision.clearance(
            geometry,
            all_roots.reshape(*shape, 13).permute(2, 0, 1, 3, 4),
            joints.permute(2, 0, 1, 3, 4),
            include_gripper=False,
        ).permute(1, 2, 0, 3)
        jaw_gap = collision.gripper_clearance(
            geometry, p.transpose(0, 1), q.transpose(0, 1)
        ).transpose(0, 1)
    else:
        arm_gap = collision.clearance(
            geometry,
            all_roots.reshape(-1, 13),
            joints.reshape(-1, 6),
            include_gripper=False,
        ).reshape(shape)
        jaw_gap = collision.gripper_clearance(geometry, p.flatten(0, 1), q.flatten(0, 1)).reshape(
            shape[1:]
        )
    valid = (result.exact.reshape(shape) & (arm_gap >= 0.002)).all(-1)
    valid &= (jaw_gap >= 0.002).all(-1)[None]
    cost = levels[None, :, None].expand(*shape[:3]).masked_fill(~valid, torch.inf)
    choice = cost.flatten(0, 1).argmin(0)
    row = torch.arange(candidates, device=p.device)
    chosen_level = choice % len(levels)
    return (
        p[chosen_level, row],
        q[chosen_level, row],
        joints.flatten(0, 1)[choice, row],
        roots,
        valid.flatten(0, 1).any(0),
    )


def grasp_environment_clearance(
    collision,
    geometry,
    positions,
    orientations,
    aperture,
    *,
    query_mask=None,
    minimum_clearance_m=GRASP_CLEARANCE_M,
):
    """Query only packed active poses, preserving each pose's scene owner."""
    cap = minimum_clearance_m + 0.0001
    if query_mask is not None:
        if query_mask.shape != positions.shape[:-1]:
            raise ValueError("Grasp query mask must match the candidate pose dimensions")
        selected = torch.where(query_mask.reshape(-1))[0]
        result = positions.new_full(query_mask.shape, torch.nan)
        if not len(selected):
            return result
        if hasattr(geometry, "select"):
            owners = selected // math.prod(positions.shape[1:-1])
            view = geometry.select(owners)
            p = positions.reshape(-1, 3)[selected, None]
            q = orientations.reshape(-1, 4)[selected, None]
            angles = aperture.reshape(-1)[selected, None]
        else:
            view = geometry
            p = positions.reshape(-1, 3)[selected][None]
            q = orientations.reshape(-1, 4)[selected][None]
            angles = aperture.reshape(-1)[selected][None]
        values = grasp_environment_clearance(
            collision, view, p, q, angles, minimum_clearance_m=minimum_clearance_m
        )
        result.reshape(-1)[selected] = values.reshape(-1)
        return result
    opened = collision.gripper_clearance(geometry, positions, orientations, maximum_distance_m=cap)
    closing = collision.gripper_closing_clearance(
        geometry, positions, orientations, aperture, maximum_distance_m=cap
    )
    return torch.minimum(opened, closing)


def pocket_pose_targets(
    orientations,
    target,
    object_orientation,
    support_points,
    collision,
    geometry,
    *,
    depth_preference=0.0,
    minimum_environment_clearance_m=GRASP_CLEARANCE_M,
    minimum_center_depth_m=0.0015,
    compact_environment_queries=True,
    support_normal_world=None,
    shared_object_geometry=False,
    retain_depth_candidates=False,
    support_mask=None,
    depth_fractions=None,
):
    """Shared convex-pocket and closing-sweep geometry for station and grasp planning."""
    if support_points.shape[-2] > 9984:
        from gr00t.rl.pick.support_features import uniform_surface_support

        if support_points.ndim == 2:
            support_points = support_points[None].expand(len(target), -1, -1)
        if support_mask is None:
            support_mask = torch.ones(
                support_points.shape[:-1], device=support_points.device, dtype=torch.bool
            )
        support_points, support_mask = uniform_surface_support(
            support_points, support_mask, count=9984
        )
    geometry_batch = 1 if shared_object_geometry else len(target)
    point_factor = (
        1
        if (
            support_points.is_cuda
            and support_points.dtype == torch.float32
            and orientations.dtype == torch.float32
        )
        else 17
    )
    row_capacity = max(
        1, MAX_POCKET_SAMPLE_EVALUATIONS // (point_factor * support_points.shape[-2])
    )
    if geometry_batch > row_capacity:
        if not hasattr(geometry, "select"):
            raise ValueError("Batched pocket geometry must support device-side row selection")
        chunks = []
        for start in range(0, len(target), row_capacity):
            stop = min(start + row_capacity, len(target))
            rows = torch.arange(start, stop, device=target.device)
            preference = torch.as_tensor(depth_preference, device=target.device)
            chunks.append(
                pocket_pose_targets(
                    orientations[start:stop],
                    target[start:stop],
                    object_orientation[start:stop],
                    support_points[start:stop] if support_points.ndim == 3 else support_points,
                    collision,
                    geometry.select(rows),
                    depth_preference=preference[start:stop] if preference.ndim else preference,
                    minimum_environment_clearance_m=minimum_environment_clearance_m,
                    minimum_center_depth_m=minimum_center_depth_m,
                    compact_environment_queries=compact_environment_queries,
                    support_normal_world=(
                        None if support_normal_world is None else support_normal_world[start:stop]
                    ),
                    shared_object_geometry=False,
                    retain_depth_candidates=retain_depth_candidates,
                    support_mask=None if support_mask is None else support_mask[start:stop],
                    depth_fractions=None
                    if depth_fractions is None
                    else depth_fractions[start:stop],
                )
            )
        return torch.cat([part[0] for part in chunks], 0), {
            key: torch.cat([part[1][key] for part in chunks], 0) for key in chunks[0][1]
        }
    chunk_size = max(
        1,
        MAX_POCKET_SAMPLE_EVALUATIONS // (geometry_batch * point_factor * support_points.shape[-2]),
    )
    if orientations.shape[1] > chunk_size:
        preference = torch.as_tensor(depth_preference, device=orientations.device)
        chunks = [
            pocket_pose_targets(
                chunk,
                target,
                object_orientation,
                support_points,
                collision,
                geometry,
                depth_preference=(
                    preference[..., index * chunk_size : (index + 1) * chunk_size]
                    if preference.ndim and preference.shape[-1] == orientations.shape[1]
                    else preference
                ),
                minimum_environment_clearance_m=minimum_environment_clearance_m,
                minimum_center_depth_m=minimum_center_depth_m,
                compact_environment_queries=compact_environment_queries,
                support_normal_world=support_normal_world,
                shared_object_geometry=shared_object_geometry,
                retain_depth_candidates=retain_depth_candidates,
                support_mask=support_mask,
                depth_fractions=(
                    None
                    if depth_fractions is None
                    else depth_fractions[:, index * chunk_size : (index + 1) * chunk_size]
                ),
            )
            for index, chunk in enumerate(orientations.split(chunk_size, dim=1))
        ]
        return torch.cat([part[0] for part in chunks], 1), {
            key: torch.cat([part[1][key] for part in chunks], 1) for key in chunks[0][1]
        }

    # Candidate ranking uses pocket depth, not clearance above this acceptance gate.
    # Keep crossing rays unbounded; only cap the nearest-distance search radius.
    clearance_cap = minimum_environment_clearance_m + 0.0001

    def depth_clearance(centers, aperture, feasible=None):
        if shared_object_geometry:
            centers = centers.expand(len(target), -1, -1, -1)
            aperture = aperture.expand(len(target), -1, -1)
            if feasible is not None:
                feasible = feasible.expand(len(target), -1, -1)
        commanded = centers.clone()
        commanded[..., 2] -= 0.0005
        positions = target[:, None, None] - _quat_rotate(orientations[:, :, None], commanded)
        quats = orientations[:, :, None].expand(*centers.shape[:-1], 4)
        flat_positions, flat_quats = positions.flatten(1, 2), quats.flatten(1, 2)
        return grasp_environment_clearance(
            collision,
            geometry,
            flat_positions,
            flat_quats,
            aperture.flatten(1, 2),
            query_mask=None if feasible is None else feasible.flatten(1, 2),
            minimum_clearance_m=minimum_environment_clearance_m,
        ).reshape(centers.shape[:-1])

    quality = convex_grasp_quality(
        orientations[:1] if shared_object_geometry else orientations,
        object_orientation[:1] if shared_object_geometry else object_orientation,
        support_points[:1]
        if shared_object_geometry and support_points.ndim == 3
        else support_points,
        depth_clearance=depth_clearance,
        depth_preference=depth_preference,
        minimum_environment_clearance_m=minimum_environment_clearance_m,
        minimum_center_depth_m=minimum_center_depth_m,
        masked_depth_clearance=compact_environment_queries,
        support_normal_world=(
            support_normal_world[:1]
            if shared_object_geometry and support_normal_world is not None
            else support_normal_world
        ),
        return_depth_candidates=retain_depth_candidates,
        support_mask=(
            support_mask[:1]
            if shared_object_geometry and support_mask is not None
            else support_mask
        ),
        depth_fractions=(
            depth_fractions[:1]
            if shared_object_geometry and depth_fractions is not None
            else depth_fractions
        ),
    )
    quality["environment_clearance_query_cap_m"] = torch.full_like(
        quality["environment_clearance_m"], clearance_cap
    )
    if retain_depth_candidates:
        depth_count = quality["center_local"].shape[2]
        orientations = orientations.repeat_interleave(depth_count, dim=1)
        quality = {key: value.flatten(1, 2) for key, value in quality.items()}
    center = quality["center_local"].clone()
    center[..., 2] -= 0.0005
    quality["command_center_local"] = center
    return target[:, None] - _quat_rotate(orientations, center), quality


def refine_grasp_candidates(
    orientations,
    goals,
    quality,
    reference,
    target,
    object_orientation,
    support_points,
    collision,
    geometry,
    *,
    support_normal_world=None,
    retain_depth_candidates=False,
    minimum_center_depth_m=0.0015,
    depth_preference=0.0,
    support_mask=None,
):
    """Bounded SO(3) refinement; original candidate indices remain unchanged."""
    if orientations.shape[1] < 2:
        return orientations, goals, quality
    feasible = quality["pocket_fit"] & (quality["antipodal_margin"] >= 0)
    feasible &= quality["environment_clearance_m"] >= GRASP_CLEARANCE_M
    similarity = (orientations * reference[:, None]).sum(-1).abs()
    index = similarity.masked_fill(~feasible, -torch.inf).argmax(-1)
    rows = torch.arange(len(orientations), device=orientations.device)
    anchors, supported = [], []
    available = feasible.clone()
    separation = torch.ones_like(similarity)
    for _ in range(min(8, orientations.shape[1])):
        anchor = orientations[rows, index]
        anchors.append(anchor)
        supported.append(available[rows, index])
        separation = torch.minimum(separation, 1 - (orientations * anchor[:, None]).sum(-1).abs())
        available &= (orientations * anchor[:, None]).sum(-1).abs() < math.cos(math.radians(0.5))
        index = separation.masked_fill(~available, -torch.inf).argmax(-1)
    anchors, supported = torch.stack(anchors, 1), torch.stack(supported, 1)
    pairs = torch.triu_indices(anchors.shape[1], anchors.shape[1], 1, device=orientations.device)
    left, right = anchors[:, pairs[0]], anchors[:, pairs[1]]
    distinct = (left * right).sum(-1).abs() < math.cos(math.radians(0.5))
    distinct &= supported[:, pairs[0]] & supported[:, pairs[1]]
    weights = torch.linspace(0.1, 0.9, 9, device=orientations.device, dtype=orientations.dtype)
    extra = _slerp(left[:, :, None], right[:, :, None], weights[None, None])
    # Keep a bounded common shape. Invalid anchor pairs remain masked on-device,
    # so batching cannot add another environment's orientations to this row.
    extra = extra.flatten(1, 2)
    extra_supported = distinct[:, :, None].expand(-1, -1, len(weights)).flatten(1, 2)
    extra_goals, extra_quality = pocket_pose_targets(
        extra,
        target,
        object_orientation,
        support_points,
        collision,
        geometry,
        retain_depth_candidates=retain_depth_candidates,
        minimum_center_depth_m=minimum_center_depth_m,
        depth_preference=depth_preference,
        support_mask=support_mask,
        **({} if support_normal_world is None else {"support_normal_world": support_normal_world}),
    )
    if retain_depth_candidates:
        depth_count = extra_goals.shape[1] // extra.shape[1]
        extra = extra.repeat_interleave(depth_count, 1)
        extra_supported = extra_supported.repeat_interleave(depth_count, 1)
    extra_quality["pocket_fit"] &= extra_supported
    return (
        torch.cat((orientations, extra), 1),
        torch.cat((goals, extra_goals), 1),
        {key: torch.cat((value, extra_quality[key]), 1) for key, value in quality.items()},
    )


def executed_arm_path_clear(
    safe_points,
    planning_indices,
    approach_samples,
    standoff_options,
    selected_standoff,
    approach_clearance,
    execution_stop_sample=None,
):
    """Exclude superseded Cartesian transit and unselected longer insertions."""
    carry = (planning_indices >= approach_samples).expand(*selected_standoff.shape, -1)
    if execution_stop_sample is not None:
        carry = carry & (planning_indices <= execution_stop_sample)[None]
    insertion = standoff_options <= selected_standoff[..., None] + 1e-6
    executed = torch.cat((carry, insertion), -1)
    clear = (safe_points.any(-3) | ~executed).all(-1) & (approach_clearance >= 0.002)
    return clear, executed


def continuous_inverse_seed(position_error, orientation_error, joints, actual):
    """Keep numerically precise solutions on the nearest finite-joint branch."""
    residual = position_error + 0.1 * orientation_error
    precise = (position_error <= 1e-4) & (orientation_error <= 1e-3)
    while actual.ndim < joints.ndim:
        actual = actual.unsqueeze(-2)
    travel = (joints - actual).square().sum(-1)
    seed_dim = residual.ndim - 3
    nearest = travel.masked_fill(~precise, torch.inf).argmin(seed_dim, keepdim=True)
    return torch.where(
        precise.any(seed_dim, keepdim=True),
        nearest,
        residual.argmin(seed_dim, keepdim=True),
    )


def grasp_candidate_bank(
    state,
    target,
    object_size,
    collision,
    geometry,
    object_support_points,
    *,
    balanced_pitch_offsets_deg=BALANCED_GRASP_PITCH_OFFSETS_DEG,
    preferred_orientation=None,
    candidate_orientations=None,
    natural_orientations=True,
    object_frame_fallback=None,
    fallback_tool_tilt_deg=None,
    refine_orientations=False,
    collider_clearance_m=GRASP_CLEARANCE_M,
    support_normal_world=None,
    minimum_center_depth_m=0.0015,
    pocket_depth_preference=0.0,
    retain_depth_candidates=False,
    support_mask=None,
    depth_fractions=None,
):
    """Shared GPU orientation/pocket bank for heterogeneous environment rows."""
    root = state["root_states"]
    if candidate_orientations is None:
        heading = torch.atan2(target[:, 1] - root[:, 1], target[:, 0] - root[:, 0])
        _, orientations, _ = grasp_pose_bank(
            target,
            heading,
            object_size,
            grasp_approach_origin(root),
            balanced_pitch_offsets_deg,
            **({"reference_orientation": state["ee_orn"]} if natural_orientations else {}),
        )
        orientations = torch.cat((state["ee_orn"][:, None], orientations), 1)
        if preferred_orientation is not None:
            orientations = torch.cat((preferred_orientation[:, None], orientations), 1)
    else:
        if (
            candidate_orientations.ndim != 3
            or candidate_orientations.shape[0] != len(target)
            or candidate_orientations.shape[-1] != 4
            or candidate_orientations.shape[1] == 0
        ):
            raise ValueError(
                "Candidate orientations must have shape (batch, nonempty_candidates, 4)"
            )
        orientations = candidate_orientations
    options = {}
    if depth_fractions is not None:
        if depth_fractions.shape[:2] != orientations.shape[:2] or depth_fractions.ndim != 3:
            raise ValueError("Depth fractions must match the orientation candidate rows")
        if object_frame_fallback is not False or refine_orientations:
            raise ValueError("Explicit depths require a fixed orientation bank without refinement")
        options["depth_fractions"] = depth_fractions
    if support_normal_world is not None:
        options["support_normal_world"] = support_normal_world
    if minimum_center_depth_m != 0.0015:
        options["minimum_center_depth_m"] = minimum_center_depth_m
    if torch.is_tensor(pocket_depth_preference) or pocket_depth_preference != 0:
        preference = pocket_depth_preference
        if torch.is_tensor(preference) and preference.shape == (len(target),):
            preference = preference[:, None]
        options["depth_preference"] = preference
    if retain_depth_candidates:
        options["retain_depth_candidates"] = True
    if support_mask is not None:
        options["support_mask"] = support_mask
    goals, quality = pocket_pose_targets(
        orientations,
        target,
        state["object_state"][:, 3:7],
        object_support_points,
        collision,
        geometry,
        minimum_environment_clearance_m=collider_clearance_m,
        **options,
    )
    if retain_depth_candidates:
        orientations = orientations.repeat_interleave(goals.shape[1] // orientations.shape[1], 1)
    fallback_count = 0
    allow_fallback = (
        candidate_orientations is None if object_frame_fallback is None else object_frame_fallback
    )
    if allow_fallback:
        extra = object_frame_grasp_bank(state["object_state"][:, 3:7], object_size)
        extra_allowed = torch.ones(extra.shape[:2], device=extra.device, dtype=torch.bool)
        if fallback_tool_tilt_deg is not None:
            axis = _quat_rotate(extra, target.new_tensor([1, 0, 0]))
            extra_allowed = axis[..., 2] <= -math.cos(math.radians(fallback_tool_tilt_deg))
        extra_goals, extra_quality = pocket_pose_targets(
            extra,
            target,
            state["object_state"][:, 3:7],
            object_support_points,
            collision,
            geometry,
            minimum_environment_clearance_m=collider_clearance_m,
            **options,
        )
        if retain_depth_candidates:
            depth_count = extra_goals.shape[1] // extra.shape[1]
            extra = extra.repeat_interleave(depth_count, 1)
            extra_allowed = extra_allowed.repeat_interleave(depth_count, 1)
        extra_quality["pocket_fit"] &= extra_allowed
        fallback_count = extra.shape[1]
        orientations = torch.cat((orientations, extra), 1)
        goals = torch.cat((goals, extra_goals), 1)
        quality = {key: torch.cat((value, extra_quality[key]), 1) for key, value in quality.items()}
    coarse_count = orientations.shape[1]
    if refine_orientations:
        orientations, goals, quality = refine_grasp_candidates(
            orientations,
            goals,
            quality,
            state["ee_orn"],
            target,
            state["object_state"][:, 3:7],
            object_support_points,
            collision,
            geometry,
            **options,
        )
    return orientations, goals, quality, coarse_count, fallback_count


def plan_grasp(
    state,
    target,
    object_size,
    limits,
    collision,
    geometry,
    dt,
    *,
    object_support_points,
    initial_object_height=None,
    balanced_pitch_offsets_deg=BALANCED_GRASP_PITCH_OFFSETS_DEG,
    excluded_orientations=None,
    realized_body=False,
    prefer_realized_body=False,
    realized_body_tracking_weight=0.0,
    minimum_realized_grasp_margin_rad=0.0,
    wrist_band_rad=None,
    preferred_orientation=None,
    refine_orientations=False,
    candidate_orientations=None,
    natural_orientations=True,
    minimum_object_height_m=None,
    withdraw_distance_m=None,
    require_precise_carry=False,
    require_safe_arm_path=False,
    collider_clearance_m=GRASP_CLEARANCE_M,
    support_normal_world=None,
    wrist_motion_weight=0.02,
    validate_carry_preview=True,
    object_frame_fallback=None,
    fallback_tool_tilt_deg=None,
    minimum_pregrasp_standoff_m=0.08,
    capture_axis_pitch_rad=None,
    retain_depth_candidates=False,
    minimum_center_depth_m=0.0015,
    pocket_depth_preference=0.0,
    elevated_transit=False,
    prepared_candidates=None,
    prepared_inverse=None,
):
    """One inverse submission for all candidate paths, including lift/withdrawal."""
    if not math.isfinite(realized_body_tracking_weight) or realized_body_tracking_weight < 0:
        raise ValueError("Realized-body tracking weight must be finite and nonnegative")
    if (
        not math.isfinite(minimum_realized_grasp_margin_rad)
        or minimum_realized_grasp_margin_rad < 0
    ):
        raise ValueError("Realized grasp margin must be finite and nonnegative")
    if not math.isfinite(wrist_motion_weight) or wrist_motion_weight < 0:
        raise ValueError("wrist_motion_weight must be finite and nonnegative")
    if not math.isfinite(pocket_depth_preference) or not 0 <= pocket_depth_preference <= 1:
        raise ValueError("Pocket depth preference must be finite and in [0, 1]")
    if not 0.08 <= minimum_pregrasp_standoff_m <= 0.32:
        raise ValueError("Minimum pregrasp standoff must be in [0.08, 0.32] m")
    if capture_axis_pitch_rad is None:
        # Enter between the open jaws, not along the curved fixed finger's axis.
        capture_axis_pitch_rad = -float(gripper_pocket_geometry()["open_angle"]) / 2
    if not math.isfinite(capture_axis_pitch_rad) or abs(capture_axis_pitch_rad) > math.pi / 3:
        raise ValueError("Capture-axis pitch must be finite and within 60 degrees")
    root = state["root_states"]
    initial_height = (
        target[0, 2] if initial_object_height is None else initial_object_height.reshape(-1)[0]
    )
    minimum_object_height = torch.maximum(initial_height + 0.10, target.new_tensor(0.30))
    if minimum_object_height_m is not None:
        minimum_object_height = torch.maximum(
            minimum_object_height, target.new_tensor(minimum_object_height_m)
        )
    lift_distance = (minimum_object_height - target[0, 2]).clamp_min(0.0) + 0.005
    approach_origin = grasp_approach_origin(root)
    bank = prepared_candidates
    if bank is None:
        bank = grasp_candidate_bank(
            state,
            target,
            object_size,
            collision,
            geometry,
            object_support_points,
            balanced_pitch_offsets_deg=balanced_pitch_offsets_deg,
            preferred_orientation=preferred_orientation,
            candidate_orientations=candidate_orientations,
            natural_orientations=natural_orientations,
            object_frame_fallback=object_frame_fallback,
            fallback_tool_tilt_deg=fallback_tool_tilt_deg,
            refine_orientations=refine_orientations,
            collider_clearance_m=collider_clearance_m,
            support_normal_world=support_normal_world,
            minimum_center_depth_m=minimum_center_depth_m,
            pocket_depth_preference=pocket_depth_preference,
            retain_depth_candidates=retain_depth_candidates,
        )
    orientations, goals, quality, coarse_count, fallback_count = bank
    axes = _quat_rotate(
        orientations,
        target.new_tensor(
            [math.cos(capture_axis_pitch_rad), 0, -math.sin(capture_axis_pitch_rad)]
        ).expand(*orientations.shape[:-1], 3),
    )
    all_orientations = orientations
    all_quality = quality
    total_candidates = goals.shape[1]
    geometric = geometric_grasp_mask(
        orientations, quality, collider_clearance_m, support_normal_world, excluded_orientations
    )[0]
    candidate_indices = torch.where(geometric)[0]
    if not len(candidate_indices):
        return None, {
            "selected": None,
            "failure": "no_collision_free_pocket_closing_pose",
            "candidate_cost": [float("inf")] * total_candidates,
            "candidate_orientations_world_wxyz": orientations.cpu().tolist(),
            "candidate_grasp_origins_world_m": goals.cpu().tolist(),
            "feasible": geometric.cpu().tolist(),
            "convex_contact_model": {
                key: value.cpu().tolist() for key, value in all_quality.items()
            },
            "lift_distance_m": lift_distance.item(),
            "required_object_lift_m": 0.10,
            "minimum_object_height_world_m": minimum_object_height.item(),
            "object_frame_fallback_candidates": fallback_count,
        }
    goals = goals[0, candidate_indices]
    orientations, axes = orientations[0, candidate_indices], axes[0, candidate_indices]
    quality = {key: value[:, candidate_indices] for key, value in quality.items()}
    candidates = len(candidate_indices)
    standoff = torch.full_like(goals[:, 2], minimum_pregrasp_standoff_m)
    pregrasp = goals - standoff[:, None] * axes
    lifted = goals.clone()
    lifted[:, 2] = (lifted[:, 2] + lift_distance).clamp_min(0.305)
    withdrawn = lifted.clone()
    forward = target[0, :2] - root[0, :2]
    forward = forward / forward.norm().clamp_min(1e-6)
    withdraw_distance = torch.where(
        target[0, 2] < 0.12, root.new_tensor(0.0), root.new_tensor(0.16)
    )
    if withdraw_distance_m is not None:
        withdraw_distance = root.new_tensor(withdraw_distance_m)
    withdrawn[:, :2] -= withdraw_distance * forward
    # Speed-bounded lengths include quaternion rotation, shared across candidates.
    lift_samples = max(
        2,
        math.ceil(1.875 * float((lifted - goals).norm(dim=-1).max()) / (GRASP_LIFT_SPEED_M_S * dt))
        + 1,
    )
    withdraw_samples = max(
        2, math.ceil(1.875 * float(withdraw_distance) / (GRASP_WITHDRAW_SPEED_M_S * dt)) + 1
    )
    samples = (151, 101, lift_samples, withdraw_samples)
    points, quats = [], []
    current = state["ee_pos"].expand(candidates, -1)
    current_q = state["ee_orn"].expand(candidates, -1)
    for phase, (endpoint, end_q, count) in enumerate(
        zip(
            (pregrasp, goals, lifted, withdrawn),
            (orientations, orientations, orientations, orientations),
            samples,
        )
    ):
        t = torch.linspace(0, 1, count, device=root.device)
        if phase >= 2:
            t = t.pow(3) * (10 + t * (-15 + 6 * t))
        points.append(current[:, None] + t[None, :, None] * (endpoint - current)[:, None])
        quats.append(
            _slerp(
                current_q[:, None].expand(-1, count, -1),
                end_q[:, None].expand(-1, count, -1),
                t[None, :],
            )
        )
        current, current_q = endpoint, end_q
    desired = torch.cat(points, 1)
    desired_q = torch.cat(quats, 1)
    heights = grasp_body_height_hypotheses(root)[0]
    actual_height_slice = slice(len(heights) - 3, len(heights))
    endpoint_index = torch.tensor(samples, device=root.device).cumsum(0) - 1
    from gr00t.rl.pick.grasp_inverse import path_reference_indices, select_grasp_inverse

    references, reference_lengths = path_reference_indices(
        torch.tensor([samples], device=root.device)
    )
    planning_index = references[
        0, torch.arange(references.shape[1], device=root.device) < reference_lengths[0]
    ]
    reference_length = len(planning_index)
    standoff_options = root.new_tensor([0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32])
    standoff_options = standoff_options[standoff_options >= minimum_pregrasp_standoff_m - 1e-6]
    approach_targets = goals[:, None] - standoff_options[None, :, None] * axes[:, None]
    inverse_positions = torch.cat((desired[:, planning_index], approach_targets), 1)
    inverse_quats = torch.cat(
        (desired_q[:, planning_index], orientations[:, None].expand(-1, len(standoff_options), -1)),
        1,
    )
    planned_length = reference_length + len(standoff_options)
    endpoint_sample = torch.searchsorted(planning_index, endpoint_index)
    actual_joints = state["dof_pos"][:, collision.arm_joint_ids][0]
    seeds = torch.stack((actual_joints, limits.mean(-1), limits.sum(-1) - actual_joints))
    shape = (len(heights), len(seeds), candidates)
    roots = root.expand(*shape, -1).clone()
    roots[..., 2] = heights[:, None, None]
    roots = roots.reshape(-1, 13)
    if prepared_inverse is None:
        arm_joints = seeds[None, :, None].expand(*shape, -1).reshape(-1, 6)
        all_desired = (
            inverse_positions[None, None].expand(*shape, -1, -1).reshape(-1, planned_length, 3)
        )
        all_q = inverse_quats[None, None].expand(*shape, -1, -1).reshape(-1, planned_length, 4)
        inverse_limits = limits
        if wrist_band_rad is not None:
            inverse_limits = limits.expand(len(roots), planned_length, -1, -1).clone()
            inverse_limits[..., 5, 0] = torch.maximum(
                limits[5, 0], actual_joints[5] - wrist_band_rad
            )
            inverse_limits[..., 5, 1] = torch.minimum(
                limits[5, 1], actual_joints[5] + wrist_band_rad
            )
            inverse_limits = inverse_limits.flatten(0, 1)
        result = inverse_bank(all_desired, all_q, roots, arm_joints, inverse_limits)
    else:
        inverse_cache, cache_row = prepared_inverse
        torch._assert_async(
            (inverse_cache["counts"][cache_row] == torch.tensor(samples, device=root.device)).all(),
            "Cached grasp inverse path lengths differ from the executing planner",
        )
        result = select_grasp_inverse(inverse_cache, cache_row, candidate_indices, planned_length)
    position_error = result.position_error.norm(dim=-1).reshape(*shape, planned_length)
    orientation_error = (2 * result.orientation_error.norm(dim=-1).clamp_max(1).asin()).reshape(
        *shape, planned_length
    )
    all_joints = result.arm_joint_positions.reshape(*shape, planned_length, 6)
    best_seed = continuous_inverse_seed(
        position_error, orientation_error, all_joints, actual_joints
    )
    position_error = position_error.gather(1, best_seed).squeeze(1)
    orientation_error = orientation_error.gather(1, best_seed).squeeze(1)
    exact_points = result.exact.reshape(*shape, planned_length).gather(1, best_seed).squeeze(1)
    planned_joints = (
        result.arm_joint_positions.reshape(*shape, planned_length, 6)
        .gather(1, best_seed[..., None].expand(-1, -1, -1, -1, 6))
        .squeeze(1)
    )
    approach_exact = exact_points[..., reference_length:]
    approach_q = planned_joints[..., reference_length:, :]
    choice_cost = 0.001 * (heights[:, None, None] - root[0, 2]).abs()
    choice_cost = choice_cost + standoff_options[None, None]
    choice_cost = choice_cost + 0.00001 * (approach_q - actual_joints).norm(dim=-1)
    shape_choice = (len(heights) * len(standoff_options), candidates)
    choice = (
        choice_cost.masked_fill(~approach_exact, torch.inf)
        .permute(0, 2, 1)
        .reshape(shape_choice)
        .argmin(0)
    )
    fallback_cost = (
        position_error[..., reference_length:] + 0.1 * orientation_error[..., reference_length:]
    )
    fallback = fallback_cost.permute(0, 2, 1).reshape(shape_choice).argmin(0)
    choice = torch.where(approach_exact.any(0).any(-1), choice, fallback)
    height_choice, standoff_choice = choice // len(standoff_options), choice % len(standoff_options)
    candidate_row = torch.arange(candidates, device=root.device)
    approach_end = approach_q[height_choice, candidate_row, standoff_choice]
    approach_height = heights[height_choice]
    pregrasp = approach_targets[candidate_row, standoff_choice]
    standoff = standoff_options[standoff_choice]
    approach_count = samples[0]
    approach_p, approach_q, approach_joints, approach_roots = joint_fk_approach(
        root, actual_joints, approach_end, approach_height, approach_count
    )
    transit_valid = None
    if elevated_transit:
        approach_p, approach_q, approach_joints, approach_roots, transit_valid = (
            elevated_cartesian_approach(
                state,
                pregrasp,
                orientations,
                approach_height,
                limits,
                collision,
                geometry,
                approach_count,
            )
        )
    # Retain the original endpoint, including low goals that initiate crouching.
    approach_p[:, -1] = pregrasp
    approach_q[:, -1] = desired_q[:, approach_count - 1]
    desired = torch.cat((approach_p, desired[:, approach_count:]), 1)
    desired_q = torch.cat((approach_q, desired_q[:, approach_count:]), 1)
    insert_t = torch.linspace(0, 1, samples[1], device=root.device)
    desired[:, approach_count : approach_count + samples[1]] = (
        pregrasp[:, None] + insert_t[None, :, None] * (goals - pregrasp)[:, None]
    )
    approach_arm_clearance = (
        collision.clearance(
            geometry,
            approach_roots.reshape(-1, 13),
            approach_joints.reshape(-1, 6),
            include_gripper=False,
        )
        .reshape(candidates, approach_count)
        .amin(-1)
    )
    exact = exact_points.any(0).all(-1)
    envelope_roots = roots.reshape(*shape, 13)[:, 0]
    clearance = collision.clearance(
        geometry,
        envelope_roots[:, :, None].expand(-1, -1, planned_length, -1).reshape(-1, 13),
        planned_joints.reshape(-1, 6),
        include_gripper=False,
    ).reshape(len(heights), candidates, planned_length)
    safe_points = clearance >= collider_clearance_m
    arm_safe, executed_arm_points = executed_arm_path_clear(
        safe_points,
        planning_index,
        approach_count,
        standoff_options,
        standoff,
        approach_arm_clearance,
        execution_stop_sample=None if validate_carry_preview else samples[0] + samples[1] - 1,
    )
    gripper_clearance = collision.gripper_clearance(geometry, desired, desired_q)
    execution_clearance = (
        gripper_clearance
        if validate_carry_preview
        else gripper_clearance[:, : samples[0] + samples[1]]
    )
    safe = execution_clearance.amin(-1) >= collider_clearance_m
    safe &= approach_arm_clearance >= collider_clearance_m
    collision_clear = safe.clone()
    if transit_valid is not None:
        safe &= transit_valid
    safe &= quality["antipodal_margin"][0] >= 0
    safe &= quality["pocket_fit"][0]
    carry_points = planning_index >= (samples[0] + samples[1] - 1)
    precise_carry = exact_points[..., :reference_length][..., carry_points].all(-1).any(0)
    if require_precise_carry:
        safe &= precise_carry
    if require_safe_arm_path:
        safe &= arm_safe
    safe &= (exact_points[..., endpoint_sample] & safe_points[..., endpoint_sample]).any(0).all(-1)
    grasp_sample = endpoint_sample[1]
    joint_margin = torch.minimum(planned_joints - limits[:, 0], limits[:, 1] - planned_joints).amin(
        -1
    )
    actual_body_grasp = (
        exact_points[actual_height_slice, :, grasp_sample]
        & (position_error[actual_height_slice, :, grasp_sample] <= 0.002)
        & (orientation_error[actual_height_slice, :, grasp_sample] <= 0.02)
        & (clearance[actual_height_slice, :, grasp_sample] >= collider_clearance_m)
        & (joint_margin[actual_height_slice, :, grasp_sample] >= minimum_realized_grasp_margin_rad)
    ).all(0)
    # Only retries use the posture already realized by the whole-body policy.
    # Initial low targets must still be published to initiate crouching.
    if realized_body:
        safe &= actual_body_grasp
    elif prefer_realized_body and (safe & actual_body_grasp).any():
        safe &= actual_body_grasp
    if prefer_realized_body and not (safe & actual_body_grasp).any():
        realized_residual = (
            position_error[-3:, :, grasp_sample] + 0.1 * orientation_error[-3:, :, grasp_sample]
        ).amax(0)
        safe &= realized_residual <= realized_residual.masked_fill(~safe, torch.inf).amin() + 1e-6
    standing_pregrasp = ((approach_height - root[0, 2]).abs() < 0.001) & approach_exact[
        height_choice, candidate_row, standoff_choice
    ]
    preferred_safe = safe & standing_pregrasp
    # Fixed-base inverse results rank candidates, never suppress the low world
    # target that the whole-body policy needs in order to initiate crouching.
    point_cost = position_error + 0.1 * orientation_error + (0.1 - joint_margin).clamp_min(0) * 0.1
    point_cost += 0.02 * (heights[:, None, None] - root[0, 2]).abs()
    cost = point_cost[..., endpoint_sample[1:]].amin(0).amax(-1)
    cost += point_cost[..., endpoint_sample[1:]].amin(0).amax(-1) * 0.1
    if realized_body:
        cost += (
            point_cost[actual_height_slice]
            .index_select(-1, endpoint_sample[1:2])
            .squeeze(-1)
            .amax(0)
        )
    if realized_body_tracking_weight:
        # Prefer a small posture adaptation over a far-away hypothetical solution.
        residual = (position_error[-2, :, grasp_sample] - 0.002).clamp_min(0)
        residual += 0.1 * (orientation_error[-2, :, grasp_sample] - 0.02).clamp_min(0)
        cost += realized_body_tracking_weight * residual
    cost += (~arm_safe).float() * 0.1
    start_rotation = (desired_q[:, approach_count - 1] * state["ee_orn"]).sum(-1).abs().clamp_max(1)
    cost += 0.001 * 2 * start_rotation.acos()
    wrist_travel = (planned_joints[..., endpoint_sample[1:], 5] - actual_joints[5]).abs()
    motion_exact = exact_points[..., endpoint_sample[1:]].all(-1)
    wrist_motion = wrist_travel.amax(-1).masked_fill(~motion_exact, torch.inf).amin(0)
    wrist_motion = torch.where(motion_exact.any(0), wrist_motion, wrist_travel.amax(-1).amin(0))
    wrist_motion = torch.maximum(wrist_motion, (approach_end[:, 5] - actual_joints[5]).abs())
    cost += (0.05 - quality["antipodal_margin"][0]).clamp_min(0) * 0.1
    preferred_axis = target[0] - approach_origin[0]
    preferred_axis /= preferred_axis.norm().clamp_min(1e-6)
    cost += 0.01 * (1 - (axes * preferred_axis).sum(-1))
    unconstrained_preference_safe = safe.clone()
    preferred_precise = exact_points[..., endpoint_sample[1:]].all(-1).any(0)
    cost += wrist_motion_weight * wrist_motion
    preserve = (candidate_indices == 0) & safe & (cost <= cost.masked_fill(~safe, torch.inf).min())
    if preferred_orientation is not None:
        preserve = (candidate_indices == 0) & safe & preferred_precise
        safe = torch.where(preserve.any(), preserve, safe)
    selected, found = best_feasible(safe[None], cost[None])
    full_cost = cost.new_full((total_candidates,), torch.inf)
    full_cost[candidate_indices] = cost
    full_feasible = torch.zeros(total_candidates, device=root.device, dtype=torch.bool)
    full_feasible[candidate_indices] = safe
    diagnostics = {
        "lift_distance_m": lift_distance.item(),
        "require_precise_carry": require_precise_carry,
        "require_safe_arm_path": require_safe_arm_path,
        "retain_depth_candidates": retain_depth_candidates,
        "minimum_center_depth_m": minimum_center_depth_m,
        "pocket_depth_preference": pocket_depth_preference,
        "validate_carry_preview": validate_carry_preview,
        "precise_carry_candidate_mask": precise_carry.cpu().tolist(),
        "inverse_wrist_band_rad": wrist_band_rad,
        "balanced_pitch_offsets_deg": list(balanced_pitch_offsets_deg),
        "orientation_family": "supplied"
        if candidate_orientations is not None
        else ("parallel_transport" if natural_orientations else "euler_grid"),
        "required_object_lift_m": 0.10,
        "minimum_object_height_world_m": minimum_object_height.item(),
        "lift_settling_margin_m": 0.005,
        "lift_speed_limit_m_s": GRASP_LIFT_SPEED_M_S,
        "approach_speed_limit_m_s": GRASP_TRANSIT_SPEED_M_S,
        "capture_speed_limit_m_s": GRASP_CAPTURE_SPEED_M_S,
        "preserved_entry_orientation": bool(preserve.any()),
        "preferred_grasp_orientation_supplied": preferred_orientation is not None,
        "feasible_before_orientation_preference": unconstrained_preference_safe.cpu().tolist(),
        "continuous_body_wrist_departure_rad": wrist_travel.amax(-1).amin(0).cpu().tolist(),
        "coarse_orientation_count": coarse_count,
        "object_frame_fallback_candidates": fallback_count,
        "refined_orientation_count": total_candidates - coarse_count,
        "preferred_grasp_orientation_selected": bool(
            preferred_orientation is not None and found[0] and candidate_indices[selected[0]] == 0
        ),
        "candidate_orientations_world_wxyz": all_orientations.cpu().tolist(),
        "wrist_travel_cost_rad": wrist_motion.cpu().tolist(),
        "wrist_motion_weight": wrist_motion_weight,
        "pregrasp_candidate_q6_rad": approach_end[:, 5].cpu().tolist(),
        "orientation_preference_policy": "matched_station_pose_else_finite_joint_motion_cost",
        "carrying_time_law": "quintic_rest_to_rest",
        "path_samples": samples,
        "withdraw_distance_m": withdraw_distance.item(),
        "realized_body_ranking": realized_body,
        "prefer_realized_body": prefer_realized_body,
        "realized_body_tracking_weight": realized_body_tracking_weight,
        "minimum_realized_grasp_margin_rad": minimum_realized_grasp_margin_rad,
        "realized_body_grasp_candidate_mask": actual_body_grasp.cpu().tolist(),
        "realized_body_grasp_arm_clearance_m": clearance[actual_height_slice, :, grasp_sample]
        .cpu()
        .tolist(),
        "approach_definition": (
            "elevated_cartesian_lift_turn_descend_experiment"
            if elevated_transit
            else "joint_limit_respecting_fk_arc_published_as_ee"
        ),
        "elevated_transit_certified": None if transit_valid is None else transit_valid.tolist(),
        "approach_nominal_base_height_m": approach_height.cpu().tolist(),
        "approach_arm_minimum_surface_distance_m": approach_arm_clearance.cpu().tolist(),
        "approach_endpoint_exact": approach_exact[height_choice, candidate_row, standoff_choice]
        .cpu()
        .tolist(),
        "pregrasp_standoff_m": standoff.cpu().tolist(),
        "minimum_pregrasp_standoff_m": minimum_pregrasp_standoff_m,
        "capture_axis_pitch_rad": capture_axis_pitch_rad,
        "standing_pregrasp_available": bool(preferred_safe.any().item()),
        "standing_pregrasp_candidate_mask": standing_pregrasp.cpu().tolist(),
        "pregrasp_posture_policy": "soft_body_height_cost_only",
        "inverse_pregrasp_standoffs_m": standoff_options.cpu().tolist(),
        "inverse_reference_definition": (
            "Cartesian carry path and nominal waypoints; approach FK arc is checked directly"
        ),
        "feasible": full_feasible.cpu().tolist(),
        "inverse_candidate_indices": candidate_indices.cpu().tolist(),
        "antipodal_feasible": (all_quality["antipodal_margin"][0] >= 0).cpu().tolist(),
        "pocket_feasible": all_quality["pocket_fit"][0].cpu().tolist(),
        "grasp_friction": GRASP_FRICTION,
        "preferred_approach_axis_world": preferred_axis.cpu().tolist(),
        "convex_contact_model": {
            key: value.detach().cpu().tolist() for key, value in all_quality.items()
        },
        "inverse_samples": len(roots) * planned_length,
        "inverse_seed_count": len(seeds),
        "initial_arm_joints_rad": actual_joints.cpu().tolist(),
        "selected_all_seed_endpoint_joints_rad": result.arm_joint_positions.reshape(
            *shape, planned_length, 6
        )[:, :, selected[0]]
        .index_select(2, endpoint_sample)
        .cpu()
        .tolist(),
        "selected_all_seed_endpoint_position_error_m": result.position_error.norm(dim=-1)
        .reshape(*shape, planned_length)[:, :, selected[0]]
        .index_select(2, endpoint_sample)
        .cpu()
        .tolist(),
        "selected_pregrasp_joints_rad": approach_end[selected[0]].cpu().tolist(),
        "planning_path_indices": planning_index.cpu().tolist(),
        "candidate_cost": full_cost.cpu().tolist(),
        "endpoint_position_error_m": position_error[..., endpoint_sample].detach().cpu().tolist(),
        "endpoint_orientation_error_rad": orientation_error[..., endpoint_sample]
        .detach()
        .cpu()
        .tolist(),
        "endpoint_joint_margin_rad": joint_margin[..., endpoint_sample].detach().cpu().tolist(),
        "exact": exact.detach().cpu().tolist(),
        "collision_clear": collision_clear.detach().cpu().tolist(),
        "arm_posture_envelope_clear": arm_safe.detach().cpu().tolist(),
        "executed_arm_reference_mask": executed_arm_points.cpu().tolist(),
        "gripper_min_clearance_m": gripper_clearance.amin(-1).detach().cpu().tolist(),
        "gripper_execution_min_clearance_m": execution_clearance.amin(-1).detach().cpu().tolist(),
        "gripper_clearance_by_segment_m": {
            name: segment.amin(-1).detach().cpu().tolist()
            for name, segment in zip(
                ("approach", "insert", "lift", "withdraw"),
                torch.split(gripper_clearance, samples, dim=1),
            )
        },
        "min_clearance_m": clearance.masked_fill(~exact_points, -torch.inf)
        .amax(0)
        .amin(-1)
        .detach()
        .cpu()
        .tolist(),
        "max_position_error_m": position_error.amin(0).amax(-1).detach().cpu().tolist(),
        "max_orientation_error": orientation_error.amin(0).amax(-1).detach().cpu().tolist(),
        "selected": int(candidate_indices[selected].item()) if found.item() else None,
        "dt": dt,
        "body_height_hypotheses_m": heights.detach().cpu().tolist(),
        "collider_clearance_m": collider_clearance_m,
        "published_pose": "original_world_target_not_fixed_base_projection",
    }
    if not bool(found.item()):
        return None, diagnostics
    index = int(selected.item())
    # Whole-body posture adapts to the original world target. A fixed-base IK
    # projection would raise an unreachable low target and prevent crouching.
    commands = desired[index]
    command_q = desired_q[index]
    boundaries = [0]
    for count in samples:
        boundaries.append(boundaries[-1] + count)
    paths = {}
    for name, a, b in zip(("approach", "insert", "lift", "withdraw"), boundaries, boundaries[1:]):
        lengths = torch.full((1,), b - a, device=root.device, dtype=torch.int64)
        if name == "withdraw":
            lengths = torch.where(target[:, 2] < 0.12, torch.ones_like(lengths), lengths)
        paths[name] = (commands[None, a:b], command_q[None, a:b], lengths)
    positions = torch.cat((paths["approach"][0], paths["insert"][0]), 1)
    quaternions = torch.cat((paths["approach"][1], paths["insert"][1]), 1)
    positions, quaternions = retime_grasp_approach(positions, quaternions, goals[index][None], dt)
    paths["approach"] = (
        positions,
        quaternions,
        torch.tensor([positions.shape[1]], device=root.device),
    )
    del paths["insert"]
    diagnostics["execution_approach_samples"] = positions.shape[1]
    return paths, diagnostics
