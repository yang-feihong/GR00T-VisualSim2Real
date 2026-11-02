"""Construct EE gaze paths from the initial arm's q3 rotation, entirely on device."""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate, _slerp
from gr00t.rl.auto_open.pose_planning import solve_pose_grid, solve_pose_targets
from gr00t.rl.envs.door.b2z1_reset_geometry import (
    b2z1_gripper_pose_and_jacobian,
    quaternion_xyzw_to_matrix,
)


@dataclass(frozen=True)
class ApproachGazeConfig:
    base_heading_deg: float = 5.0
    optical_yaw_deg: float = 3.0
    optical_pitch_deg: float = 3.0
    optical_roll_deg: float = 10.0
    ee_object_height_clearance_m: float = 0.0


def sample_approach_offsets(seed, device, config=ApproachGazeConfig()):
    return sample_approach_offsets_batch([seed], device, config)[0]


def sample_approach_offsets_batch(seeds, device, config=ApproachGazeConfig()):
    from gr00t.rl.auto_open.tensor_random import episode_uniform

    scale = torch.tensor(
        [
            config.base_heading_deg,
            config.optical_yaw_deg,
            config.optical_pitch_deg,
            config.optical_roll_deg,
        ],
        device=device,
    ) * (math.pi / 180)
    seed_tensor = torch.tensor(seeds, device=device, dtype=torch.int64)
    return (episode_uniform(seed_tensor, 0x475A4531, 1, 4)[:, 0] * 2 - 1) * scale


@lru_cache(maxsize=1)
def wrist_mount():
    config = json.loads(Path(__file__).with_name("cameras.json").read_text())
    return next(camera for camera in config["cameras"] if camera["name"] == "left_wrist_0_rgb")


def camera_mount_in_ee(reference):
    mount = wrist_mount()
    # FK ends at gripper_link: link06 -> stator (51 mm, roll -90 deg) -> EE (49 mm).
    link_to_ee = _euler_xyz_quat(reference.new_tensor([math.pi / 2, 0, 0]))
    offset = _quat_rotate(link_to_ee, reference.new_tensor(mount["position"]))
    offset = offset - reference.new_tensor([0.1, 0, 0])
    quaternion = _quat_multiply(
        link_to_ee, _euler_xyz_quat(reference.new_tensor(mount["rotation"]))
    )
    return offset, quaternion_xyzw_to_matrix(quaternion[[1, 2, 3, 0]])


def camera_pose(position, rotation):
    offset, camera_rotation = camera_mount_in_ee(position)
    optical = camera_rotation[:, 0]
    return position + (rotation @ offset[..., None]).squeeze(-1), (
        rotation @ optical[..., None]
    ).squeeze(-1)


def aimed_rotation(original_rotation, direction):
    axis = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    optical = original_rotation[..., :, 0]
    cosine = (optical * axis).sum(-1)
    delta = torch.cat(((1 + cosine)[..., None], torch.cross(optical, axis, dim=-1)), -1)
    opposite = torch.cat((torch.zeros_like(cosine)[..., None], original_rotation[..., :, 1]), -1)
    delta = torch.where((cosine < -0.99999)[..., None], opposite, delta)
    delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return _quat_multiply(delta, _matrix_to_quaternion_wxyz(original_rotation))


def gaze_from_actual_camera(
    position, orientation, actual_position, actual_orientation, target, offsets, blend
):
    """Retain the planned optical-center path; point using the measured optical center."""
    mount_offset, mount_rotation = camera_mount_in_ee(position)
    planned_rotation = quaternion_xyzw_to_matrix(orientation[..., [1, 2, 3, 0]])
    planned_camera, _ = camera_pose(position, planned_rotation)
    actual_camera, _ = camera_pose(
        actual_position, quaternion_xyzw_to_matrix(actual_orientation[..., [1, 2, 3, 0]])
    )
    original = planned_rotation @ mount_rotation
    q = aimed_rotation(original, target - actual_camera)
    rotation = quaternion_xyzw_to_matrix(q[..., [1, 2, 3, 0]])
    # Roll is already in the planned reference. Reapply only the fixed yaw/pitch offset.
    angles = torch.stack((offsets[:, 0] * 0, offsets[:, 1], offsets[:, 0]), -1)
    noise = quaternion_xyzw_to_matrix(_euler_xyz_quat(angles)[..., [1, 2, 3, 0]])
    q = _slerp(
        _matrix_to_quaternion_wxyz(original), _matrix_to_quaternion_wxyz(rotation @ noise), blend
    )
    rotation = quaternion_xyzw_to_matrix(q[..., [1, 2, 3, 0]]) @ mount_rotation.T
    return planned_camera - (rotation @ mount_offset[:, None]).squeeze(
        -1
    ), _matrix_to_quaternion_wxyz(rotation)


def perturb_camera_aim(position, orientation, offsets, dt, blend_s=1.5, sample_times=None):
    """Rotate about the optical center, including genuine optical-axis roll."""
    count = position.shape[1]
    times = (
        (torch.arange(count, device=position.device) * dt)[None]
        if sample_times is None
        else sample_times
    )
    blend = (times / blend_s).clamp_max(1)
    angles = offsets[:, None, [2, 1, 0]] * blend[..., None]
    noise = quaternion_xyzw_to_matrix(_euler_xyz_quat(angles)[..., [1, 2, 3, 0]])
    rotation = quaternion_xyzw_to_matrix(orientation[..., [1, 2, 3, 0]])
    mount_offset, mount_rotation = camera_mount_in_ee(position)
    camera, _ = camera_pose(position, rotation)
    rotation = rotation @ mount_rotation @ noise @ mount_rotation.T
    position = camera - (rotation @ mount_offset[:, None]).squeeze(-1)
    return position, _matrix_to_quaternion_wxyz(rotation)


def q3_gaze_path(roots, initial_joints, target, limits, dt, blend_s=1.5):
    """Keep all virtual joints except q3 unchanged, then publish their FK as EE poses."""
    batch, count = roots.shape[:2]
    joints = (
        initial_joints[:, None].expand(batch, count, 6)
        if initial_joints.ndim == 2
        else initial_joints
    ).clone()
    position, rotation, jacobian = b2z1_gripper_pose_and_jacobian(
        roots[..., :3], roots[..., [4, 5, 6, 3]], joints
    )
    camera, optical = camera_pose(position, rotation)
    axis = jacobian[..., 3:, 2]
    pivot = position + torch.cross(axis, jacobian[..., :3, 2], dim=-1)
    radius, destination = camera - pivot, target[:, None] - pivot
    a = (axis * torch.cross(optical, destination, dim=-1)).sum(-1)
    b = (axis * torch.cross(torch.cross(axis, optical, dim=-1), destination, dim=-1)).sum(-1)
    c = (axis * torch.cross(optical, radius, dim=-1)).sum(-1)
    phase = torch.atan2(b, a)
    spread = torch.acos((c / torch.hypot(a, b).clamp_min(1e-8)).clamp(-1, 1))
    delta = torch.stack((phase + spread, phase - spread), -1)
    delta = torch.atan2(delta.sin(), delta.cos())
    q3 = (joints[..., 2, None] + delta).clamp(limits[2, 0], limits[2, 1])
    q3 = torch.cat((q3, joints[..., 2, None]), -1)
    candidates = joints[..., None, :].expand(batch, count, 3, 6).clone()
    candidates[..., 2] = q3
    candidate_roots = roots[..., None, :].expand(batch, count, 3, 13)
    p, r, _ = b2z1_gripper_pose_and_jacobian(
        candidate_roots[..., :3], candidate_roots[..., [4, 5, 6, 3]], candidates
    )
    camera, optical = camera_pose(p, r)
    direction = target[:, None, None] - camera
    score = (direction * optical).sum(-1) / direction.norm(dim=-1).clamp_min(1e-8)
    selected = score.argmax(-1, keepdim=True)
    desired_q3 = q3.gather(-1, selected).squeeze(-1)
    blend = (
        (torch.arange(count, device=roots.device) * dt / blend_s).clamp_max(1)
        if blend_s > 0
        else torch.ones(count, device=roots.device)
    )
    joints[..., 2] += blend[None] * (desired_q3 - joints[..., 2])
    position, rotation, _ = b2z1_gripper_pose_and_jacobian(
        roots[..., :3], roots[..., [4, 5, 6, 3]], joints
    )
    return position, _matrix_to_quaternion_wxyz(rotation), joints


def gaze_candidate_offsets(reference, roll_step_deg=45, joint_sampling=False):
    roll_options = torch.arange(
        -180, 180, roll_step_deg, device=reference.device, dtype=reference.dtype
    )
    roll_options = roll_options[roll_options.abs().argsort(stable=True)] * (math.pi / 180)
    shoulder, rolls = torch.meshgrid(
        reference.new_tensor([0, -10, 10, -20, 20, -30, 30, -40, -50, -60, -70, -80, -90])
        * (math.pi / 180),
        roll_options,
        indexing="ij",
    )
    shoulder, rolls = shoulder.flatten(), rolls.flatten()
    elbow = torch.zeros_like(shoulder)
    if joint_sampling:
        index = torch.arange(512, device=reference.device, dtype=reference.dtype)
        unit = (index[:, None] * reference.new_tensor([0.618033989, 0.414213562, 0.732050808])) % 1
        delta = (unit * 2 - 1) * math.pi
        shoulder = torch.cat((shoulder, delta[:, 0]))
        elbow = torch.cat((elbow, delta[:, 1]))
        rolls = torch.cat((rolls, delta[:, 2]))
    return shoulder, elbow, rolls


def natural_gaze_candidates(
    roots,
    initial_joints,
    target,
    limits,
    dt,
    offsets,
    roll_step_deg=45,
    *,
    sample_times=None,
    candidate_indices=None,
    joint_sampling=False,
):
    """Raise through q2 and aim through q3; optical centers follow their FK arcs."""
    batch, count = roots.shape[:2]
    shoulder, elbow, rolls = gaze_candidate_offsets(roots, roll_step_deg, joint_sampling)
    if candidate_indices is None:
        selected_shoulder, selected_elbow, selected_rolls = (
            value[None].expand(batch, -1) for value in (shoulder, elbow, rolls)
        )
    else:
        selected_shoulder, selected_elbow, selected_rolls = (
            value[candidate_indices] for value in (shoulder, elbow, rolls)
        )
    candidates = selected_shoulder.shape[1]
    expanded_roots = roots[:, None].expand(-1, candidates, -1, -1).reshape(-1, count, 13)
    joints = initial_joints[:, None, None].expand(-1, candidates, count, -1).clone()
    joints[..., 1] = (joints[..., 1] + selected_shoulder[..., None]).clamp(*limits[1])
    joints[..., 3] = (joints[..., 3] + selected_elbow[..., None]).clamp(*limits[3])
    p, q, template = q3_gaze_path(
        expanded_roots,
        joints.reshape(-1, count, 6),
        target[:, None].expand(-1, candidates, -1).reshape(-1, 3),
        limits,
        dt,
        blend_s=0,
    )
    rotation = quaternion_xyzw_to_matrix(q[..., [1, 2, 3, 0]])
    mount_offset, mount_rotation = camera_mount_in_ee(p)
    camera, _ = camera_pose(p, rotation)
    aimed = aimed_rotation(
        rotation @ mount_rotation,
        target[:, None, None].expand(-1, candidates, count, -1).reshape(-1, count, 3) - camera,
    )
    rotation = quaternion_xyzw_to_matrix(aimed[..., [1, 2, 3, 0]]) @ mount_rotation.T
    p = camera - (rotation @ mount_offset[:, None]).squeeze(-1)
    q = _matrix_to_quaternion_wxyz(rotation)
    candidate_offsets = offsets[:, None].expand(-1, candidates, -1).clone()
    candidate_offsets[..., 2] += selected_rolls
    times = (
        None
        if sample_times is None
        else sample_times[:, None].expand(-1, candidates, -1).reshape(-1, count)
    )
    p, q = perturb_camera_aim(p, q, candidate_offsets.reshape(-1, 3), dt, sample_times=times)
    return p, q, template, expanded_roots, shoulder, rolls


def wrist_route_length(endpoint_joints, initial_joints, next_joints):
    """Finite joint travel, without wrapping through the wrist's physical stop."""
    return (endpoint_joints[..., 5] - initial_joints[:, None, 5]).abs() + (
        next_joints[:, None, 5] - endpoint_joints[..., 5]
    ).abs()


def _plan_approach_gaze_dense(
    roots,
    initial_joints,
    target,
    limits,
    dt,
    collision,
    geometry,
    offsets,
    *,
    roll_step_deg=45,
    extra_joint_seeds=0,
    next_grasp_joints=None,
    next_grasp_mask=None,
    config=ApproachGazeConfig(),
    lengths=None,
    max_requests=65536,
):
    """Plan padded scene-owned gaze paths without serializing diagnostics to the host."""
    batch, count = roots.shape[:2]
    if lengths is None:
        lengths = torch.full((batch,), count, device=roots.device, dtype=torch.long)
    if lengths.shape != (batch,) or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("Gaze path lengths must have one entry per environment")
    torch._assert_async(((lengths > 0) & (lengths <= count)).all(), "Invalid gaze path lengths")
    active = torch.arange(count, device=roots.device)[None] < lengths[:, None]
    p, q, template, expanded_roots, shoulder, rolls = natural_gaze_candidates(
        roots, initial_joints, target, limits, dt, offsets, roll_step_deg
    )
    candidates = len(shoulder)
    pose_count = batch * candidates * count
    template_seed = template.reshape(-1, 6)
    actual_seed = initial_joints[:, None, None].expand(-1, candidates, count, -1).reshape(-1, 6)
    midpoint_seed = limits.mean(-1)[None].expand(pose_count, -1)
    reflected_seed = limits.sum(-1)[None] - actual_seed
    sample = torch.arange(1, extra_joint_seeds + 1, device=roots.device, dtype=roots.dtype)[:, None]
    fractions = torch.frac(
        sample * roots.new_tensor([0.618034, 0.414214, 0.732051, 0.236068, 0.645751, 0.316625])
    )
    spread_seeds = limits[:, 0] + (0.1 + 0.8 * fractions) * (limits[:, 1] - limits[:, 0])
    height_offset = roots.new_tensor([-0.25, -0.18, -0.12, 0, 0.12, 0.18]).repeat_interleave(
        4 + extra_joint_seeds
    )
    seed_count = len(height_offset)
    active_poses = active[:, None].expand(-1, candidates, -1).reshape(-1)
    best_roots = expanded_roots.reshape(-1, 13).clone()
    joints = actual_seed.clone()
    error = roots.new_full((pose_count,), torch.inf)
    angle = torch.full_like(error, torch.inf)
    # Reduce seed hypotheses as each request block finishes, rather than retaining
    # every inverse output for all scenes/candidates/path samples at once.
    for start in range(0, pose_count, max(1, max_requests // seed_count)):
        end = min(start + max(1, max_requests // seed_count), pose_count)
        seeds = torch.cat(
            (
                torch.stack(
                    (
                        template_seed[start:end],
                        actual_seed[start:end],
                        midpoint_seed[start:end],
                        reflected_seed[start:end],
                    ),
                    1,
                ),
                spread_seeds[None].expand(end - start, -1, -1),
            ),
            1,
        ).repeat(1, 6, 1)
        r = expanded_roots.reshape(-1, 13)[start:end, None].expand(-1, seed_count, -1).clone()
        ramp = ((torch.arange(start, end, device=roots.device) % count) * dt / 1.5).clamp_max(1)
        r[..., 2] += height_offset[None] * ramp[:, None]
        solved = solve_pose_grid(
            p.reshape(-1, 1, 3)[start:end],
            q.reshape(-1, 1, 4)[start:end],
            r.transpose(0, 1),
            seeds.transpose(0, 1),
            limits,
            query_mask=active_poses[start:end, None],
            max_requests=max_requests,
            solver=solve_pose_targets,
        )
        if solved is None:
            continue
        seed_error = solved.position_error.squeeze(-2).norm(dim=-1).T.nan_to_num(nan=torch.inf)
        seed_angle = solved.orientation_error.squeeze(-2).norm(dim=-1).T.nan_to_num(nan=torch.inf)
        best = (seed_error + 0.1 * seed_angle).argmin(-1)
        indices = torch.arange(end - start, device=roots.device)
        selected_joints = solved.arm_joint_positions.squeeze(-2).transpose(0, 1)[indices, best]
        joints[start:end] = torch.where(
            active_poses[start:end, None], selected_joints, actual_seed[start:end]
        )
        best_roots[start:end] = r[indices, best]
        error[start:end] = seed_error[indices, best]
        angle[start:end] = seed_angle[indices, best]
    clearance = collision.clearance(
        geometry,
        best_roots.reshape(batch, candidates, count, 13),
        joints.reshape(batch, candidates, count, 6),
    )
    error = error.reshape(batch, candidates, count)
    angle = angle.reshape(batch, candidates, count)
    settled_start = min(count - 1, int(1.5 / dt))
    settled = active & (
        torch.arange(count, device=roots.device)[None]
        >= torch.minimum(lengths - 1, lengths.new_full((batch,), settled_start))[:, None]
    )
    valid = (
        ((clearance > 0.02) | ~active[:, None]).all(-1)
        & ((error < 0.02) | ~settled[:, None]).all(-1)
        & ((angle < 0.15) | ~settled[:, None]).all(-1)
        & (
            (
                p.reshape(batch, candidates, count, 3)[..., 2]
                >= target[:, None, None, 2] + config.ee_object_height_clearance_m
            )
            | ~active[:, None]
        ).all(-1)
    )
    cost = (
        shoulder.abs()[None]
        + 0.01 * rolls.abs()[None]
        + (error.masked_fill(~active[:, None], 0).sum(-1) / lengths[:, None]) * 10
    )
    rows = torch.arange(batch, device=roots.device)
    endpoint_joints = joints.reshape(batch, candidates, count, 6)[rows, :, lengths - 1]
    route = None
    if next_grasp_joints is not None:
        route = wrist_route_length(endpoint_joints, initial_joints, next_grasp_joints)
        cost += (
            route if next_grasp_mask is None else torch.where(next_grasp_mask[:, None], route, 0)
        )
    selected = torch.where(valid, cost, torch.inf).argmin(-1)
    return (
        p.reshape(batch, candidates, count, 3)[rows, selected],
        q.reshape(batch, candidates, count, 4)[rows, selected],
        {
            "valid": valid.any(-1),
            "target_definition": "q2_raise_q3_gaze_fk",
            "selected_q2_offset_deg": torch.rad2deg(shoulder[selected]),
            "selected_optical_roll_offset_deg": torch.rad2deg(rolls[selected]),
            "predicted_gaze_end_joints_rad": endpoint_joints[rows, selected],
            "wrist_route_length_rad": route,
            "candidate_q2_offset_deg": torch.rad2deg(shoulder),
            "candidate_optical_roll_offset_deg": torch.rad2deg(rolls),
            "candidate_valid": valid,
            "samples": lengths,
            "inverse_settled_validation_start_step": settled_start,
            "minimum_clearance_m": clearance.masked_fill(~active[:, None], torch.inf).amin(-1),
            "maximum_inverse_position_error_m": error.masked_fill(~active[:, None], 0).amax(-1),
            "maximum_inverse_orientation_error_rad": angle.masked_fill(~active[:, None], 0).amax(
                -1
            ),
            "maximum_settled_position_error_m": error.masked_fill(~settled[:, None], 0).amax(-1),
            "per_step_best_position_error_m": error.amin(1),
            "per_step_feasible_count": ((clearance > 0.02) & (error < 0.02) & (angle < 0.15))
            .logical_and(active[:, None])
            .sum(1),
            "worst_position_error_step": error.masked_fill(~active[:, None], -torch.inf).argmax(-1),
            "selected_path_start_m": p.reshape(batch, candidates, count, 3)[rows, selected, 0],
            "selected_path_end_m": p.reshape(batch, candidates, count, 3)[
                rows, selected, lengths - 1
            ],
            "base_path_start_m": roots[:, 0, :3],
            "base_path_end_m": roots[rows, lengths - 1, :3],
            "minimum_ee_height_above_object_m": (
                p.reshape(batch, candidates, count, 3)[rows, selected, ..., 2] - target[:, None, 2]
            )
            .masked_fill(~active, torch.inf)
            .amin(-1),
        },
    )


def _gaze_inverse_samples(
    p, q, template, roots, times, active, limits, max_requests, collision, geometry, settled
):
    shape = p.shape[:-1]
    body = roots[None].expand(3, *roots.shape).clone()
    height = p.new_tensor([-0.03, 0.04, 0.10])
    body[..., 2] += height[:, None, None, None] * (times / 1.5).clamp_max(1)[None, :, None]
    seeds = template[None].expand(3, *template.shape).reshape(3, -1, 6)
    result = solve_pose_grid(
        p.reshape(-1, 1, 3),
        q.reshape(-1, 1, 4),
        body.reshape(3, -1, 13),
        seeds,
        limits,
        query_mask=active.reshape(-1, 1),
        max_requests=max_requests,
        solver=solve_pose_targets,
    )
    error = result.position_error[:, :, 0].norm(dim=-1).nan_to_num(nan=torch.inf)
    angle = result.orientation_error[:, :, 0].norm(dim=-1).nan_to_num(nan=torch.inf)
    all_joints = result.arm_joint_positions[:, :, 0]
    margin = torch.minimum(all_joints - limits[:, 0], limits[:, 1] - all_joints).amin(-1)
    batch, candidates, count = shape
    inverse_ok = (
        ((error < 0.02) & (angle < 0.15) & (margin >= math.radians(2))).reshape(3, *shape).all(0)
    )
    admissible = (inverse_ok | ~settled).all(-1)
    query_mask = (active & admissible[..., None])[:, :, None].expand(-1, -1, 3, -1)
    query_mask = query_mask.flatten(1, 2)
    checked_roots = body.permute(1, 2, 0, 3, 4).flatten(1, 2)
    checked_joints = all_joints.reshape(3, *shape, 6).permute(1, 2, 0, 3, 4).flatten(1, 2)
    checked_joints = torch.where(
        active[:, :, None].expand(-1, -1, 3, -1).flatten(1, 2)[..., None],
        checked_joints,
        template[:, :, None].expand(-1, -1, 3, -1, -1).flatten(1, 2),
    )
    clearance = (
        collision.clearance(
            geometry,
            checked_roots,
            checked_joints,
            maximum_distance_m=0.0201,
            query_mask=query_mask,
        )
        .reshape(batch, candidates, 3, count)
        .amin(2)
    )
    clearance = torch.where(active & admissible[..., None], clearance, torch.nan)
    joints = torch.where(active[..., None], all_joints[1].reshape(*shape, 6), template)
    return (
        joints,
        body[1],
        error.amax(0).reshape(shape),
        angle.amax(0).reshape(shape),
        clearance,
        margin.amin(0).reshape(shape),
    )


def plan_approach_gaze_batch(
    roots,
    initial_joints,
    target,
    limits,
    dt,
    collision,
    geometry,
    offsets,
    *,
    roll_step_deg=45,
    extra_joint_seeds=0,
    next_grasp_joints=None,
    next_grasp_mask=None,
    config=ApproachGazeConfig(),
    lengths=None,
    max_requests=65536,
):
    """Screen sparse anchors, then fully validate four complete candidate paths."""
    batch, count = roots.shape[:2]
    if count <= 9 or extra_joint_seeds:
        return _plan_approach_gaze_dense(
            roots,
            initial_joints,
            target,
            limits,
            dt,
            collision,
            geometry,
            offsets,
            roll_step_deg=roll_step_deg,
            extra_joint_seeds=extra_joint_seeds,
            next_grasp_joints=next_grasp_joints,
            next_grasp_mask=next_grasp_mask,
            config=config,
            lengths=lengths,
            max_requests=max_requests,
        )
    if lengths is None:
        lengths = torch.full((batch,), count, device=roots.device, dtype=torch.long)
    torch._assert_async(((lengths > 0) & (lengths <= count)).all(), "Invalid gaze lengths")
    anchors = (torch.linspace(0, 1, 8, device=roots.device)[None] * (lengths - 1)[:, None]).long()
    blend_end = (lengths - 1).clamp_max(int(1.5 / dt))[:, None]
    anchors = torch.cat((anchors, blend_end), 1).sort(-1).values

    times = anchors.to(roots.dtype) * dt
    sparse_roots = roots.gather(1, anchors[..., None].expand(-1, -1, 13))
    p, q, template, body, shoulder, rolls = natural_gaze_candidates(
        sparse_roots,
        initial_joints,
        target,
        limits,
        dt,
        offsets,
        roll_step_deg,
        sample_times=times,
        joint_sampling=True,
    )
    candidates = len(shoulder)
    _, elbow, _ = gaze_candidate_offsets(roots, roll_step_deg, joint_sampling=True)
    p, q, template, body = (
        value.reshape(batch, candidates, anchors.shape[1], -1) for value in (p, q, template, body)
    )

    def evaluate(position, orientation, seed, robot, times, active):
        settled = active & (
            times[:, None]
            >= torch.minimum((lengths - 1).to(times.dtype) * dt, times.new_full((batch,), 1.5))[
                :, None, None
            ]
        )
        joints, _, error, angle, clearance, margin = _gaze_inverse_samples(
            position,
            orientation,
            seed,
            robot,
            times,
            active,
            limits,
            max_requests,
            collision,
            geometry,
            settled,
        )
        valid = (
            ((clearance > 0.02) | ~active).all(-1)
            & ((error < 0.02) | ~settled).all(-1)
            & ((angle < 0.15) | ~settled).all(-1)
            & ((margin >= math.radians(2)) | ~settled).all(-1)
            & (
                (position[..., 2] >= target[:, None, None, 2] + config.ee_object_height_clearance_m)
                | ~active
            ).all(-1)
        )
        ranking_error = error.masked_fill(margin < math.radians(2), torch.inf)
        return joints, clearance, ranking_error, angle, valid, settled, error, margin

    mask = torch.ones((batch, candidates, len(anchors[0])), device=roots.device, dtype=torch.bool)
    cj, cc, ce, ca, cv, _, _, _ = evaluate(p, q, template, body, times, mask)
    cost = (shoulder.abs() + elbow.abs())[None] + 0.01 * rolls.abs()[None]
    cost = cost + ce.mean(-1).nan_to_num(posinf=100) * 10
    if next_grasp_joints is not None:
        route = wrist_route_length(cj[:, :, -1], initial_joints, next_grasp_joints)
        cost += (
            route if next_grasp_mask is None else torch.where(next_grasp_mask[:, None], route, 0)
        )
    cost += (~cv).to(cost.dtype) * 1000
    choice = cost.topk(min(4, candidates), largest=False, sorted=True).indices

    dense = natural_gaze_candidates(
        roots,
        initial_joints,
        target,
        limits,
        dt,
        offsets,
        roll_step_deg,
        candidate_indices=choice,
        joint_sampling=True,
    )
    position, orientation, seed, robot = (
        value.reshape(batch, choice.shape[1], count, -1) for value in dense[:4]
    )
    times = (torch.arange(count, device=roots.device, dtype=roots.dtype) * dt)[None].expand(
        batch, -1
    )
    active = torch.arange(count, device=roots.device)[None] < lengths[:, None]
    active = active[:, None].expand(-1, choice.shape[1], -1)
    joints, clearance, error, angle, valid, settled, physical_error, margin = evaluate(
        position, orientation, seed, robot, times, active
    )
    rows = torch.arange(batch, device=roots.device)
    endpoint = joints[rows, :, lengths - 1]
    score = shoulder[choice].abs() + elbow[choice].abs() + 0.01 * rolls[choice].abs()
    score += error.masked_fill(~active, 0).sum(-1) / lengths[:, None] * 10
    route = None
    if next_grasp_joints is not None:
        route = wrist_route_length(endpoint, initial_joints, next_grasp_joints)
        score += (
            route if next_grasp_mask is None else torch.where(next_grasp_mask[:, None], route, 0)
        )
    selected = score.masked_fill(~valid, torch.inf).argmin(-1)
    original = choice[rows, selected]
    path, path_q = position[rows, selected], orientation[rows, selected]
    dense_valid = torch.zeros((batch, candidates), device=roots.device, dtype=torch.bool)
    dense_valid.scatter_(1, choice, valid)
    return (
        path,
        path_q,
        {
            "valid": valid.any(-1),
            "target_definition": "q2_q4_joint_candidates_q3_gaze_fk",
            "selected_q2_offset_deg": torch.rad2deg(shoulder[original]),
            "selected_q4_offset_deg": torch.rad2deg(elbow[original]),
            "selected_optical_roll_offset_deg": torch.rad2deg(rolls[original]),
            "predicted_gaze_end_joints_rad": endpoint[rows, selected],
            "wrist_route_length_rad": route,
            "candidate_q2_offset_deg": torch.rad2deg(shoulder),
            "candidate_optical_roll_offset_deg": torch.rad2deg(rolls),
            "candidate_valid": dense_valid,
            "samples": lengths,
            "coarse_anchor_indices": anchors,
            "fully_validated_candidate_indices": choice,
            "inverse_settled_validation_start_step": min(count - 1, int(1.5 / dt)),
            "minimum_clearance_m": clearance.masked_fill(~active, torch.inf).amin(-1),
            "collision_evaluated": torch.isfinite(clearance).any(-1),
            "collision_skipped_because_inverse_rejected": ~torch.isfinite(clearance).any(-1),
            "per_step_feasible_count_excludes_inverse_rejected_candidates": True,
            "maximum_inverse_position_error_m": physical_error.masked_fill(~active, 0).amax(-1),
            "maximum_inverse_orientation_error_rad": angle.masked_fill(~active, 0).amax(-1),
            "maximum_settled_position_error_m": physical_error.masked_fill(~settled, 0).amax(-1),
            "minimum_settled_joint_margin_rad": margin.masked_fill(~settled, torch.inf).amin(-1),
            "required_joint_margin_rad": math.radians(2),
            "walking_height_offsets_m": [-0.03, 0.04, 0.10],
            "per_step_best_position_error_m": physical_error.amin(1),
            "per_step_feasible_count": (
                (clearance > 0.02) & (error < 0.02) & (angle < 0.15) & active
            ).sum(1),
            "worst_position_error_step": physical_error.masked_fill(~active, -torch.inf).argmax(-1),
            "selected_path_start_m": path[:, 0],
            "selected_path_end_m": path[rows, lengths - 1],
            "base_path_start_m": roots[:, 0, :3],
            "base_path_end_m": roots[rows, lengths - 1, :3],
            "minimum_ee_height_above_object_m": (path[..., 2] - target[:, None, 2])
            .masked_fill(~active[:, 0], torch.inf)
            .amin(-1),
        },
    )


def plan_approach_gaze(*args, **kwargs):
    """Legacy scene-scoped diagnostic adapter; planning uses the shared batch implementation."""
    position, orientation, diagnostics = plan_approach_gaze_batch(*args, **kwargs)
    diagnostics = {
        key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
        for key, value in diagnostics.items()
    }
    diagnostics["valid"] = all(diagnostics["valid"])
    diagnostics["samples"] = position.shape[1]
    return position, orientation, diagnostics
