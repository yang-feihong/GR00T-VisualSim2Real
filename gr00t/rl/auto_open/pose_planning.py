"""Standard direct-pose and candidate/path-bank planning on the input device.

Tasks generate geometry and feasibility predicates; they must not implement
another inverse or dispatch one inverse call per candidate or path point.
"""

import math
from dataclasses import fields

import torch


def solve_pose_targets(*args, **kwargs):
    from legged_gym.utils.ee_steady_inverse_gpu import batched_steady_ee_inverse_no_gravity

    return batched_steady_ee_inverse_no_gravity(*args, **kwargs)


def solve_pose_grid(
    positions,
    orientations,
    roots,
    initial_joints,
    limits,
    *,
    query_mask=None,
    max_requests=65536,
    solver=None,
):
    """Solve shared paths over a hypothesis grid, bounding GPU workspace by requests.

    Paths are [M, T, xyz], seeds are [..., M, joints], and roots either
    [..., M, rigid-state] or [..., M, T, rigid-state]. Inactive results are
    NaN/false. An entirely empty request set returns None without invoking IK.
    """
    if positions.ndim != 3 or positions.shape[-1] != 3 or max_requests < 1:
        raise ValueError("Pose grids require [paths, samples, xyz] and a positive request limit")
    count, length = positions.shape[:2]
    if orientations.shape != (count, length, 4) or initial_joints.shape[-2:] != (count, 6):
        raise ValueError("Pose grid path and joint-seed dimensions disagree")
    leading = initial_joints.shape[:-1]
    moving = roots.shape == (*leading, length, 13)
    if not moving and roots.shape != (*leading, 13):
        raise ValueError("Pose grid body poses must match the seed/path dimensions")
    if query_mask is not None and query_mask.shape != (count, length):
        raise ValueError("Pose grid mask must match the shared paths")
    if query_mask is not None and query_mask.dtype != torch.bool:
        raise ValueError("Pose grid mask must be boolean")
    if limits.ndim == 2:
        if limits.shape != (6, 2):
            raise ValueError("Shared pose grid joint limits must have shape [6, 2]")
        limit_mode = "shared"
    elif limits.shape == (*leading, 6, 2):
        limit_mode = "seed"
    elif limits.shape == (*leading, length, 6, 2):
        limit_mode = "point"
    else:
        raise ValueError("Pose grid joint limits must be shared, per seed or per point")
    total = math.prod(leading) * length
    flat_roots = roots.reshape(-1, 13)
    flat_joints = initial_joints.reshape(-1, 6)
    flat_limits = limits if limit_mode == "shared" else limits.reshape(-1, 6, 2)
    dense, result_type = {}, None
    solve = solve_pose_targets if solver is None else solver
    for start in range(0, total, max_requests):
        block = torch.arange(start, min(start + max_requests, total), device=positions.device)
        if query_mask is not None:
            block = block[query_mask[(block // length) % count, block % length]]
        if not len(block):
            continue
        owner, time = (block // length) % count, block % length
        body = flat_roots[block if moving else block // length]
        joint_limits = limits
        if limit_mode != "shared":
            joint_limits = flat_limits[block // length if limit_mode == "seed" else block]
        result = solve(
            positions[owner, time],
            orientations[owner, time],
            base_position_world=body[:, :3],
            base_orientation_world_xyzw=body[:, [4, 5, 6, 3]],
            initial_arm_joint_positions=flat_joints[block // length],
            joint_limits=joint_limits,
        )
        result_type = type(result)
        for field in fields(result):
            value = getattr(result, field.name)
            if field.name not in dense:
                dense[field.name] = torch.full(
                    (*leading, length, *value.shape[1:]),
                    False if value.dtype == torch.bool else torch.nan,
                    device=value.device,
                    dtype=value.dtype,
                )
            dense[field.name].reshape(-1, *value.shape[1:])[block] = value
    return None if result_type is None else result_type(**dense)


def first_feasible(feasible):
    """First valid entry of an already ordered candidate bank; no host reads."""
    return feasible.to(torch.int64).argmax(-1), feasible.any(-1)


def best_feasible(feasible, cost):
    """Minimum-cost valid entry, preserving the first index on ties."""
    return cost.masked_fill(~feasible, torch.inf).argmin(-1), feasible.any(-1)


class PosePlanningMixin:
    def _inverse_pose_bank(
        self,
        fsm,
        device_state,
        env_ids,
        desired_positions,
        desired_orientations,
        lengths,
        *,
        initial_arm_joint_positions=None,
    ):
        rows = torch.as_tensor(env_ids, dtype=torch.int64, device=fsm.device)
        arms = device_state["dof_pos"][rows][:, self.arm_joint_ids]
        roots = device_state["root_states"][rows]
        owner, sample = torch.where(
            torch.arange(desired_positions.shape[1], device=fsm.device)[None, :] < lengths[:, None]
        )
        result = solve_pose_targets(
            desired_positions[owner, sample],
            desired_orientations[owner, sample],
            base_position_world=roots[owner, :3],
            base_orientation_world_xyzw=roots[owner][:, [4, 5, 6, 3]],
            initial_arm_joint_positions=(
                arms[owner]
                if initial_arm_joint_positions is None
                else initial_arm_joint_positions[owner, sample]
            ),
            joint_limits=self.joint_limits,
        )
        return result, owner, sample
