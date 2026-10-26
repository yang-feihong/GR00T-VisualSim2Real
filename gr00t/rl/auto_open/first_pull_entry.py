"""Analytic first insertion: continuous alignment, then extend inward.

The closed-gripper dimensions are scalar asset configuration, not point clouds.
All candidates and environments are evaluated together on their input device.
"""

import ast
import math
from functools import lru_cache
from pathlib import Path

import torch
from legged_gym.utils.batch_invariant_math import fixed_norm

from gr00t.rl.auto_open.pull_entry_gpu import _yaw_rotate, quat_rotate


@lru_cache(maxsize=1)
def _offline_goal_center_xy():
    # Offline probes cannot import IsaacLab configuration without starting Kit.
    path = (
        Path(__file__).resolve().parents[1]
        / "third_party/visual_wholebody_lowlevel/legged_gym/envs/manip_loco/b2z1_config.py"
    )
    tree = ast.parse(path.read_text())
    center = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "sphere_center"
    )
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in center.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("x_offset", "y_offset")
    }
    return values["x_offset"], values["y_offset"]


def insertion_endpoint_radius(endpoint, roots, goal_reference=None):
    """Use the same goal-centered horizontal frame as the comfort limit."""
    if goal_reference is None:
        center_x, center_y = _offline_goal_center_xy()
        w, x, y, z = roots[:, 3:7].unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        relative = endpoint - roots[:, None, :3]
        local = _yaw_rotate(-yaw[:, None].expand_as(relative[..., 0]), relative)
        local[..., 0] -= center_x
        local[..., 1] -= center_y
    else:
        origin, reference, offset = (value.to(endpoint) for value in goal_reference)
        inverse = reference.clone()
        inverse[:, 1:] *= -1
        local = quat_rotate(inverse[:, None], endpoint - origin[:, None]) - offset[:, None]
    return fixed_norm(local[..., :2])


def plan_first_pull_entry(
    cache,
    state,
    rows,
    samples,
    *,
    frame_edge_m,
    wall_front_m,
    hardware_height_bounds,
    config,
    goal_reference=None,
    include_approach=False,
):
    dtype = state["ee_pos"].dtype
    batch, count = samples["height_m"].shape

    def repeat(value):
        return value[:, None].expand(-1, count, *value.shape[1:]).flatten(0, 1).double()

    ee, ee_q = repeat(state["ee_pos"][rows]), repeat(state["ee_orn"][rows])
    hinge = repeat(cache.hinge_world[rows])
    hinge_angle = repeat(state["door_joint_position"][rows, cache.hinge_idx[rows]])
    angle = repeat(cache.hinge_axis_sign[rows]) * hinge_angle
    closed_tangent = repeat(cache.tangent[rows])
    tangent = _yaw_rotate(angle, closed_tangent)
    normal = _yaw_rotate(angle, repeat(cache.door_normal[rows]))
    gap_vector = (
        closed_tangent * repeat(frame_edge_m)[:, None]
        + tangent * repeat(cache.panel_width[rows])[:, None]
    )
    gap_vector[:, 2] = 0.0
    outward = gap_vector / torch.linalg.vector_norm(gap_vector, dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )
    height = samples["height_m"].flatten().double()
    insertion = samples["insertion_m"].flatten().double()
    width = repeat(cache.panel_width[rows])
    inward = -outward
    yaw = torch.atan2(inward[:, 1], inward[:, 0])
    cosine, sine = torch.cos(yaw / 2.0), torch.sin(yaw / 2.0)
    half = math.sqrt(0.5)
    q0 = torch.stack((cosine * half, cosine * half, sine * half, sine * half), -1)
    q1 = torch.stack((cosine * half, -cosine * half, -sine * half, sine * half), -1)
    orientation = torch.where(
        ((q0 * ee_q).sum(-1).abs() >= (q1 * ee_q).sum(-1).abs())[:, None], q0, q1
    )

    tip_radius = width - insertion
    endpoint = hinge + (tip_radius + config.closed_tip_offset_m)[:, None] * outward
    endpoint[:, 2] = height
    alignment = (
        hinge
        + (width + config.first_entry_radial_clearance_m + config.closed_tip_offset_m)[:, None]
        * outward
    )
    alignment[:, 2] = height
    if include_approach:
        points = torch.stack((ee, alignment, endpoint), 1)
        orientations = torch.stack((ee_q, orientation, orientation), 1)
    else:
        # Station feasibility keeps its existing radial insertion probe.
        points = torch.stack((alignment, endpoint), 1)
        orientations = orientation[:, None].expand(-1, 2, -1)
    positions, path_q, lengths, overflow = cache._sample(
        points, orientations, torch.full_like(hinge_angle, points.shape[1], dtype=torch.int64)
    )

    bounds = repeat(hardware_height_bounds)
    z_bounds = torch.stack(
        (height - config.closed_half_height_m, height + config.closed_half_height_m), -1
    )
    hardware_safe = ~(
        (z_bounds[:, 0] < bounds[:, 1] + config.hardware_clearance_m)
        & (z_bounds[:, 1] > bounds[:, 0] - config.hardware_clearance_m)
    )
    tip = endpoint + config.closed_tip_offset_m * inward
    gap_safe = angle.abs() > 1.0e-6
    geometry_safe = hardware_safe & gap_safe & torch.isfinite(endpoint).all(-1)
    radius = insertion_endpoint_radius(
        endpoint.reshape(batch, count, 3), state["root_states"][rows].double(), goal_reference
    ).flatten()
    radius_safe = radius <= config.insertion_max_horizontal_radius_m
    geometry_safe &= radius_safe
    feasible = geometry_safe & ~overflow
    result = dict(
        positions=positions,
        orientations=path_q,
        lengths=lengths,
        feasible=feasible,
        geometry_safe=geometry_safe,
        route_safe=feasible,
        route_start_free=geometry_safe,
        route_end_free=geometry_safe,
        route_direct=geometry_safe,
        collision_safe=geometry_safe,
        overflow=overflow,
        endpoint=endpoint,
        orientation=orientation,
        contact_point=tip,
        endpoint_horizontal_radius_m=radius,
        endpoint_radius_safe=radius_safe,
        face_offset_local=torch.tensor(
            [config.closed_tip_offset_m, 0.0, 0.0], device=ee.device
        ).expand_as(ee),
        tip_radius_m=tip_radius,
        ee_tangent_radius_m=tip_radius + config.closed_tip_offset_m,
        gripper_height_bounds_m=z_bounds,
        entry_hinge=hinge_angle,
        normal=normal,
    )
    return {
        key: (value.to(dtype) if value.is_floating_point() else value).reshape(
            batch, count, *value.shape[1:]
        )
        for key, value in result.items()
    }
