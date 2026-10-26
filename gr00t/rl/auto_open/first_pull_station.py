"""GPU station sampling and prospective first-insertion path validation."""

import math
from types import SimpleNamespace

import torch
from legged_gym.utils.ee_steady_inverse_gpu import (
    _quaternion_multiply_wxyz,
    batched_steady_ee_inverse_no_gravity,
)

from gr00t.rl.auto_open.first_pull_entry import plan_first_pull_entry
from gr00t.rl.auto_open.pull_entry_gpu import _matrix_quat
from gr00t.rl.auto_open.pull_station import robot_motion_envelope
from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian


def station_candidates(hinge, normal, tangent, frame_edge, wall_front, uniform, config):
    """Local origin is the free frame edge on the robot-facing wall surface."""
    lower = uniform.new_tensor(
        (config.first_x_range_m[0], config.first_y_range_m[0], config.first_yaw_range_deg[0])
    )
    upper = uniform.new_tensor(
        (config.first_x_range_m[1], config.first_y_range_m[1], config.first_yaw_range_deg[1])
    )
    local = lower + uniform * (upper - lower)
    origin = hinge[:, :2] + frame_edge[:, None] * tangent + wall_front[:, None] * normal[:, :2]
    xy = origin[:, None] + local[..., :1] * normal[:, None, :2] + local[..., 1:2] * tangent[:, None]
    angle = torch.deg2rad(local[..., 2])
    forward = (
        angle.cos()[..., None] * normal[:, None, :2] + angle.sin()[..., None] * tangent[:, None]
    )
    yaw = torch.atan2(forward[..., 1], forward[..., 0])
    return xy, yaw, local


def station_clearances(
    root, bodies, points, body_indices, xy, yaw, normal, tangent, hinge, frame_edge, wall_front
):
    envelope = robot_motion_envelope(root, bodies, points, body_indices).flatten(1, 2)
    c, s = yaw.cos()[..., None], yaw.sin()[..., None]
    x, y = envelope[:, None, :, 0], envelope[:, None, :, 1]
    offset = torch.stack((c * x - s * y, s * x + c * y), -1)
    relative = xy[:, :, None] + offset - hinge[:, None, None, :2]
    wall = (relative * normal[:, None, None, :2]).sum(-1).amin(-1) - wall_front[:, None]
    frame = (relative * tangent[:, None, None]).sum(-1).amin(-1) - frame_edge[:, None]
    return wall, frame


def first_insertion_residuals(
    cache,
    state,
    rows,
    xy,
    yaw,
    clearance_safe,
    pose_grid,
    arm_ids,
    joint_limits,
    frame_edge,
    wall_front,
    hardware_z,
    config,
    goal_reference=None,
):
    """Check all safe stations and 27 poses; chunk only to bound device memory."""
    shape = (*clearance_safe.shape, len(pose_grid))
    position_error = xy.new_full(shape, torch.inf)
    orientation_error = xy.new_full(shape, torch.inf)
    owner, station = torch.where(clearance_safe)
    for begin in range(0, len(owner), 32):
        group = owner[begin : begin + 32]
        candidate = station[begin : begin + 32]
        env_ids = rows[group]
        roots = state["root_states"][env_ids].clone()
        w, x, y, z = roots[:, 3:7].unbind(-1)
        old_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        half = 0.5 * (yaw[group, candidate] - old_yaw)
        zero = torch.zeros_like(half)
        turn = torch.stack((half.cos(), zero, zero, half.sin()), -1)
        roots[:, 3:7] = _quaternion_multiply_wxyz(turn, roots[:, 3:7])
        roots[:, :2] = xy[group, candidate]
        arms = state["dof_pos"][env_ids][:, arm_ids]
        ee, rotation, _ = b2z1_gripper_pose_and_jacobian(roots[:, :3], roots[:, [4, 5, 6, 3]], arms)
        geometry_cache = SimpleNamespace(max_samples=cache.max_samples, _sample=cache._sample)
        for name in (
            "hinge_world",
            "door_normal",
            "tangent",
            "panel_width",
            "panel_center",
            "panel_half_thickness",
            "hinge_axis_sign",
            "hinge_idx",
        ):
            setattr(geometry_cache, name, getattr(cache, name)[env_ids])
        hypothetical = dict(
            root_states=roots,
            ee_pos=ee,
            ee_orn=_matrix_quat(rotation),
            door_joint_position=state["door_joint_position"][env_ids],
        )
        samples = dict(
            height_m=pose_grid[None, :, 0].expand(len(group), -1),
            insertion_m=pose_grid[None, :, 1].expand(len(group), -1),
        )
        prospective_reference = None
        if goal_reference is not None:
            origin, reference, offset = (value[env_ids].clone() for value in goal_reference)
            origin[:, :2] += roots[:, :2] - state["root_states"][env_ids, :2]
            prospective_reference = (origin, _quaternion_multiply_wxyz(turn, reference), offset)
        geometry = plan_first_pull_entry(
            geometry_cache,
            hypothetical,
            torch.arange(len(group), device=xy.device),
            samples,
            frame_edge_m=frame_edge[group],
            wall_front_m=wall_front[group],
            hardware_height_bounds=hardware_z[group],
            config=config,
            goal_reference=prospective_reference,
        )
        station_owner, attempt = torch.where(geometry["feasible"])
        lengths = geometry["lengths"][station_owner, attempt]
        path_owner, sample = torch.where(
            torch.arange(cache.max_samples, device=xy.device)[None] < lengths[:, None]
        )
        if not len(path_owner):
            continue
        active = station_owner[path_owner]
        result = batched_steady_ee_inverse_no_gravity(
            geometry["positions"][active, attempt[path_owner], sample],
            geometry["orientations"][active, attempt[path_owner], sample],
            base_position_world=roots[active, :3],
            base_orientation_world_xyzw=roots[active][:, [4, 5, 6, 3]],
            initial_arm_joint_positions=arms[active],
            joint_limits=joint_limits,
        )
        error = torch.linalg.vector_norm(result.position_error, dim=-1)
        angle = 2 * torch.asin(
            torch.linalg.vector_norm(result.orientation_error, dim=-1).clamp(0.0, 1.0)
        )
        maximum = xy.new_zeros(len(station_owner)).scatter_reduce_(
            0, path_owner, torch.nan_to_num(error, nan=torch.inf), reduce="amax"
        )
        angular = xy.new_zeros(len(station_owner)).scatter_reduce_(
            0, path_owner, torch.nan_to_num(angle, nan=torch.inf), reduce="amax"
        )
        position_error[group[station_owner], candidate[station_owner], attempt] = maximum
        orientation_error[group[station_owner], candidate[station_owner], attempt] = angular
    feasible = (position_error <= config.position_tolerance_m) & (
        orientation_error <= math.radians(1.0)
    )
    return feasible, position_error, orientation_error
