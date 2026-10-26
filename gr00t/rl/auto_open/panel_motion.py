"""Shared hinge-speed feedback and geometry for panel manipulation."""

import math
from dataclasses import dataclass

import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_dot,
    fixed_norm,
    fixed_normalize,
)

PUSH_TIP_MAX_ANGULAR_LEAD_RAD = math.radians(4.0)


def panel_fingertip_reference(target, offsets, depth, normal, standoff):
    """Use the door-facing vertex in the distal 5% as the explicit tip marker."""
    projection = (offsets * depth[..., None, :]).sum(-1)
    distal = projection >= torch.quantile(projection, 0.95, dim=-1, keepdim=True)
    normal_projection = (offsets * normal[..., None, :]).sum(-1)
    index = normal_projection.masked_fill(~distal, torch.inf).argmin(-1)
    tip_offset = offsets.gather(-2, index[..., None, None].expand(*index.shape, 1, 3)).squeeze(-2)
    gap = standoff + (tip_offset * normal).sum(-1) - normal_projection.amin(-1)
    surface = target + tip_offset - gap[..., None] * normal
    return tip_offset, surface


def fingertip_arc_position(surface, tip_offset, hinge, delta):
    """Rotate the tip about the hinge, then recover EE at fixed world attitude."""
    relative = surface - hinge
    c, s = delta.cos(), delta.sin()
    rotated = torch.stack(
        (
            c * relative[..., 0] - s * relative[..., 1],
            s * relative[..., 0] + c * relative[..., 1],
            relative[..., 2].expand_as(c * relative[..., 0]),
        ),
        -1,
    )
    return hinge + rotated - tip_offset


def hinge_speed_target(
    hinge, velocity, integral, *, dt, target_speed, kp, ki, integral_limit, max_lead, goal, start
):
    """Identical PI/lead calculation for either side of a panel."""
    error = target_speed - velocity
    integral = (integral + error * dt).clamp(-integral_limit, integral_limit)
    lead = (kp * error + ki * integral).clamp(-max_lead, max_lead)
    target = torch.maximum(torch.minimum(hinge + lead, goal), start - math.radians(2.0))
    return target, integral


def outside_comfort_workspace(local_ee, radius):
    return fixed_norm(local_ee[:, :2]) > radius


@dataclass(frozen=True)
class PullPanelConfig:
    insertion_max_horizontal_radius_m: float = 0.60
    insertion_lengths_m: tuple[float, ...] = (0.02, 0.04, 0.06)
    height_nominal_m: float = 1.0
    height_offsets_m: tuple[float, ...] = (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20)
    station_angle_noise_rad: float = math.radians(5.0)
    opening_angle_rad: float = math.radians(89.0)
    contact_standoff_m: float = 0.002
    position_tolerance_m: float = 0.004
    hardware_clearance_m: float = 0.01
    first_entry_radial_clearance_m: float = 0.03
    closed_tip_offset_m: float = 0.101
    closed_half_height_m: float = 0.042

    @property
    def candidate_count(self) -> int:
        return len(self.height_offsets_m) * len(self.insertion_lengths_m)


def rotate_base_station(root, hinge, angle):
    """Rotate both position and heading by the same signed hinge angle."""
    relative = root[:, :2] - hinge[:, :2]
    c, s = torch.cos(angle), torch.sin(angle)
    xy = hinge[:, :2] + torch.stack(
        (c * relative[:, 0] - s * relative[:, 1], s * relative[:, 0] + c * relative[:, 1]), dim=-1
    )
    w, x, y, z = root[:, 3:7].unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return xy, yaw + angle


def follow_station_rotation(
    progress, noise, radius, *, max_noise_rad, inner_distance_m, inner_yaw_rad
):
    """Condition the approved noise range on a genuinely distinct next station."""
    minimum = torch.maximum(
        2.0 * torch.asin((inner_distance_m / radius.clamp_min(inner_distance_m)).clamp(max=1.0))
        + 1e-4,
        torch.full_like(progress, inner_yaw_rad + 1e-4),
    )
    if max_noise_rad <= 0.0:
        return progress.clamp_min(0.0), torch.zeros_like(noise), progress >= minimum
    lower = torch.maximum(torch.full_like(progress, -max_noise_rad), minimum - progress)
    feasible = lower <= max_noise_rad
    fraction = ((noise + max_noise_rad) / (2.0 * max_noise_rad)).clamp(0.0, 1.0)
    constrained = (
        lower.clamp(max=max_noise_rad) + (max_noise_rad - lower.clamp(max=max_noise_rad)) * fraction
    )
    effective = torch.where(lower > -max_noise_rad, constrained, noise)
    return (progress + effective).clamp_min(0.0), effective, feasible


def base_move_clearance(
    root,
    target_xy,
    target_yaw,
    body_points,
    *,
    hinge,
    normal,
    tangent,
    frame_edge,
    wall_front,
    panel_center,
    panel_normal,
    panel_tangent,
    panel_width,
    panel_half_thickness,
    return_details=False,
    body_margin_m=0.0,
):
    """Swept robot-envelope SAT check against the panel and both frame walls."""
    _, yaw = rotate_base_station(root, hinge, torch.zeros_like(target_yaw))
    turn = torch.atan2(torch.sin(target_yaw - yaw), torch.cos(target_yaw - yaw))
    fraction = torch.linspace(0.0, 1.0, 33, device=root.device)
    angle = yaw[:, None] + turn[:, None] * fraction
    center = root[:, None, :2] + (target_xy - root[:, :2])[:, None, :] * fraction[None, :, None]
    if body_points.ndim == 2:
        body_points = body_points[None].expand(len(root), -1, -1)
    if body_points.ndim == 3:
        body_points = body_points[:, None]
    c, s = torch.cos(angle)[..., None], torch.sin(angle)[..., None]
    forward = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    lateral = torch.stack((-forward[..., 1], forward[..., 0]), dim=-1)

    def separation(n, t, depth_low, depth_high, span_low, span_high):
        rectangle = torch.stack(
            [
                hinge[:, :2] + d[:, None] * n[:, :2] + s[:, None] * t[:, :2]
                for d in (depth_low, depth_high)
                for s in (span_low, span_high)
            ],
            dim=1,
        )
        gap = torch.full_like(angle, torch.inf)
        for part in body_points.unbind(dim=1):
            x, y = part[:, None, :, 0], part[:, None, :, 1]
            points = center[:, :, None, :] + torch.stack((c * x - s * y, s * x + c * y), dim=-1)
            edges = part[:, [4, 2, 1] if part.shape[1] == 8 else [1, 2, 3], :2] - part[:, :1, :2]
            edge_normal = fixed_normalize(
                torch.stack((-edges[..., 1], edges[..., 0]), dim=-1), eps=1e-8
            )
            ex, ey = edge_normal[:, None, :, 0], edge_normal[:, None, :, 1]
            rotated = torch.stack((c * ex - s * ey, s * ex + c * ey), dim=-1)
            axes = torch.cat(
                (
                    torch.stack(
                        (
                            n[:, None, :2].expand_as(forward),
                            t[:, None, :2].expand_as(forward),
                            forward,
                            lateral,
                        ),
                        dim=-2,
                    ),
                    rotated,
                ),
                dim=-2,
            )
            body = fixed_dot(points[:, :, :, None, :], axes[:, :, None, :, :])
            panel = fixed_dot(rectangle[:, None, :, None, :], axes[:, :, None, :, :])
            separation = torch.maximum(
                body.amin(dim=2) - panel.amax(dim=2), panel.amin(dim=2) - body.amax(dim=2)
            )
            separation = torch.where(fixed_norm(axes) > 0.5, separation - body_margin_m, -torch.inf)
            gap = torch.minimum(gap, separation.amax(dim=-1))
        return gap

    depth = fixed_dot(panel_center[:, :2] - hinge[:, :2], panel_normal[:, :2])
    panel_gap = separation(
        panel_normal,
        panel_tangent,
        depth - panel_half_thickness - 0.02,
        depth + panel_half_thickness + 0.02,
        torch.full_like(depth, -0.02),
        panel_width + 0.02,
    ).amin(dim=1)
    negative, positive, zero = (
        torch.full_like(depth, -5.0),
        torch.full_like(depth, 5.0),
        torch.zeros_like(depth),
    )
    wall_gap = torch.minimum(
        separation(normal, tangent, negative, wall_front + 0.02, negative, zero).amin(dim=1),
        separation(normal, tangent, negative, wall_front + 0.02, frame_edge, positive).amin(dim=1),
    )
    safe = (panel_gap >= 0.0) & (wall_gap >= 0.0)
    if return_details:
        return safe, {"wall_sweep_margin_m": wall_gap, "panel_sweep_margin_m": panel_gap}
    return safe


def base_station_waypoints(
    root, target_xy, target_yaw, body_points, *, body_margin_m=0.0, **geometry
):
    """Use the same pose tracker: direct, turn then translate, or translate then turn."""

    def check(source, xy, yaw, cloud=body_points, parameters=geometry):
        return base_move_clearance(
            source, xy, yaw, cloud, **parameters, return_details=True, body_margin_m=body_margin_m
        )

    direct, direct_margin = check(root, target_xy, target_yaw)
    _, current_yaw = rotate_base_station(root, geometry["hinge"], torch.zeros_like(target_yaw))
    turn_first, turn_margin = check(root, root[:, :2], target_yaw)
    turned = root.clone()
    zeros = torch.zeros_like(target_yaw)
    turned[:, 3:7] = torch.stack(
        (torch.cos(target_yaw / 2.0), zeros, zeros, torch.sin(target_yaw / 2.0)), dim=-1
    )
    after_turn, after_turn_margin = check(turned, target_xy, target_yaw)
    move_first, move_margin = check(root, target_xy, current_yaw)
    moved = root.clone()
    moved[:, :2] = target_xy
    after_move, after_move_margin = check(moved, target_xy, target_yaw)
    rotate = ~direct & turn_first & after_turn
    translate = ~direct & ~rotate & move_first & after_move
    final = torch.cat((target_xy, target_yaw[:, None]), dim=-1)
    path = final[:, None, :].repeat(1, 2, 1)
    path[rotate, 0, :2] = root[rotate, :2]
    path[translate, 0, 2] = current_yaw[translate]
    lengths = torch.where(rotate | translate, 2, 1).to(torch.int32)
    margins = {
        key: torch.where(
            rotate,
            torch.minimum(turn_margin[key], after_turn_margin[key]),
            torch.where(translate, torch.minimum(move_margin[key], after_move_margin[key]), value),
        )
        for key, value in direct_margin.items()
    }
    safe = direct | rotate | translate
    rejected = torch.where(~safe)[0]
    if len(rejected):
        # A corner waypoint can avoid a collision caused by moving and turning
        # near the free edge. Evaluate all candidates in one tensor batch.
        source = root[rejected]
        offsets = torch.linspace(-0.8, 0.8, 9, device=root.device, dtype=root.dtype)
        dx, dy = torch.meshgrid(offsets, offsets, indexing="ij")
        offset = torch.stack((dx.flatten(), dy.flatten()), dim=-1)
        n, t = geometry["normal"][rejected, :2], geometry["tangent"][rejected, :2]
        xy = (
            source[:, None, :2]
            + offset[None, :, :1] * n[:, None]
            + offset[None, :, 1:] * t[:, None]
        )
        xy = xy.repeat_interleave(3, dim=1)
        turn = torch.atan2(
            torch.sin(target_yaw[rejected] - current_yaw[rejected]),
            torch.cos(target_yaw[rejected] - current_yaw[rejected]),
        )
        headings = current_yaw[rejected, None] + turn[:, None] * torch.tensor(
            [0.0, 0.5, 1.0], device=root.device
        )
        headings = headings.repeat(1, len(offset))
        count = xy.shape[1]
        first = source[:, None].expand(-1, count, -1).reshape(-1, source.shape[-1])
        middle = first.clone()
        middle[:, :2] = xy.reshape(-1, 2)
        h = headings.flatten()
        middle[:, 3:7] = torch.stack(
            (torch.cos(h / 2.0), torch.zeros_like(h), torch.zeros_like(h), torch.sin(h / 2.0)),
            dim=-1,
        )
        cloud = (
            body_points[None].expand(len(root), -1, -1) if body_points.ndim == 2 else body_points
        )
        cloud = (
            cloud[rejected, None].expand(-1, count, *cloud.shape[1:]).reshape(-1, *cloud.shape[1:])
        )
        selected_geometry = {
            key: value[rejected, None]
            .expand(-1, count, *value.shape[1:])
            .reshape(-1, *value.shape[1:])
            for key, value in geometry.items()
        }
        first_safe, first_margin = check(first, xy.reshape(-1, 2), h, cloud, selected_geometry)
        final_xy = target_xy[rejected, None].expand(-1, count, -1).reshape(-1, 2)
        final_yaw = target_yaw[rejected, None].expand(-1, count).flatten()
        last_safe, last_margin = check(middle, final_xy, final_yaw, cloud, selected_geometry)
        feasible = (first_safe & last_safe).reshape(len(rejected), count)
        cost = fixed_norm(xy - source[:, None, :2]) + fixed_norm(xy - target_xy[rejected, None])
        cost = torch.where(feasible, cost, torch.inf)
        chosen = cost.argmin(dim=1)
        solved = feasible.any(dim=1)
        selected = rejected[solved]
        batch = torch.arange(len(rejected), device=root.device)[solved]
        best = chosen[solved]
        path[selected, 0, :2], path[selected, 0, 2] = xy[batch, best], headings[batch, best]
        lengths[selected], safe[selected] = 2, True
        for key in margins:
            combined = torch.minimum(first_margin[key], last_margin[key]).reshape(
                len(rejected), count
            )
            margins[key][selected] = combined[batch, best]
    return path, lengths, safe, margins


def base_station_candidate_paths(
    root, target_xy, target_yaw, body_points, *, body_margin_m=0.0, **geometry
):
    """Choose the first feasible pose candidate using the shared path planner."""
    batch, count = target_yaw.shape
    source = root[:, None].expand(-1, count, -1).reshape(-1, root.shape[-1])
    cloud = body_points[None].expand(batch, -1, -1) if body_points.ndim == 2 else body_points
    cloud = cloud[:, None].expand(-1, count, *cloud.shape[1:]).reshape(-1, *cloud.shape[1:])
    parameters = {
        key: value[:, None].expand(-1, count, *value.shape[1:]).reshape(-1, *value.shape[1:])
        for key, value in geometry.items()
    }
    path, lengths, safe, margins = base_station_waypoints(
        source,
        target_xy.reshape(-1, 2),
        target_yaw.flatten(),
        cloud,
        body_margin_m=body_margin_m,
        **parameters,
    )
    safe = safe.reshape(batch, count)
    selected = safe.to(torch.int64).argmax(dim=1)
    rows = torch.arange(batch, device=root.device)
    return (
        path.reshape(batch, count, 2, 3)[rows, selected],
        lengths.reshape(batch, count)[rows, selected],
        safe.any(dim=1),
        {key: value.reshape(batch, count)[rows, selected] for key, value in margins.items()},
        selected,
    )
