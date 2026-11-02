"""Bounded, seed-keyed joint station/posture/grasp proposals on the device."""

import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.auto_open.tensor_random import episode_uniform
from gr00t.rl.pick.grasp_quality import convex_grasp_quality
from gr00t.rl.pick.support_features import minimum_projection, uniform_surface_support
from gr00t.rl.pick.supported_asset_audit import gripper_collision_clouds


def sample_station_pockets(
    orientations, state, support, mask, collision, geometry, count=256, *, object_pose_offset
):
    """Screen shape-only poses, then verify a small bank using original geometry."""
    q = orientations
    features, feature_mask = uniform_surface_support(support, mask, count=1024)
    fixed, moving = gripper_collision_clouds(str(support.device))
    bottom = _quat_rotate(state["object_state"][:, None, 3:7], support)[..., 2]
    bottom = bottom.masked_fill(~mask, torch.inf).amin(-1)
    above_plane = (
        state["object_state"][:, 2] - state["support_z"] if "support_z" in state else -bottom
    )

    def plane_check(quaternions, *, floor_only=False):
        inverse = quaternions.clone()
        inverse[..., 1:] *= -1
        normal = _quat_rotate(inverse, support.new_tensor([0, 0, 1]))

        def clearance(center, aperture, feasible=None):
            if feasible is None:
                feasible = torch.ones_like(aperture, dtype=torch.bool)
            if floor_only and "on_floor" in state:
                feasible = feasible & state["on_floor"][:, None, None]
            owner, candidate, depth = torch.where(feasible)
            result = aperture.new_full(aperture.shape, 0.1)
            if not len(owner):
                return result
            n = normal[owner, candidate]
            c = center[owner, candidate, depth].clone()
            c[:, 2] -= 0.0005
            origin = above_plane[owner] - (n * c).sum(-1)
            gap = origin + minimum_projection(fixed, n)
            for weight in (0.0, 1 / 6, 2 / 6, 0.5, 4 / 6, 5 / 6, 1.0):
                angle = aperture[owner, candidate, depth]
                angle = angle + weight * (torch.pi / 3 - angle)
                nx, ny, nz = n.unbind(-1)
                direction = torch.stack(
                    (nx * angle.cos() + nz * angle.sin(), ny, -nx * angle.sin() + nz * angle.cos()),
                    -1,
                )
                gap = torch.minimum(gap, origin + minimum_projection(moving, direction))
            result[owner, candidate, depth] = gap
            return result

        return clearance

    parts = [
        convex_grasp_quality(
            block,
            state["object_state"][:, 3:7],
            features,
            support_mask=feature_mask,
            friction=100.0,
            depth_clearance=plane_check(block),
            masked_depth_clearance=False,
            return_depth_candidates=True,
            depth_fractions=torch.linspace(0, 1, 5, device=q.device),
        )
        for block in q.split(32, 1)
    ]
    coarse = {key: torch.cat([part[key] for part in parts], 1) for key in parts[0]}
    feasible = (
        (coarse["fixed_contact_normal_local"].norm(dim=-1) > 0.5)
        & (coarse["open_jaw_clearance_m"] >= -0.002)
        & (coarse["contact_depth_margin_m"] >= -0.002)
        & (coarse["center_depth_margin_m"] >= -0.001)
        & (coarse["width_clearance_m"] >= -0.002)
        & (coarse["antipodal_margin"] >= 0.15)
    )
    plane_penalty = 50 * (0.002 - coarse["environment_clearance_m"]).clamp(0, 0.05)
    depth_cost = plane_penalty - 0.1 * coarse["antipodal_margin"]
    depth = depth_cost.masked_fill(~feasible, torch.inf).argmin(-1)
    orientation_cost = 1 - (q * orientations[:, :1]).sum(-1).abs()
    score = orientation_cost + depth_cost.gather(-1, depth[..., None])[..., 0]
    unconstrained_score = orientation_cost - 0.1 * coarse["antipodal_margin"].masked_fill(
        ~feasible, -torch.inf
    ).amax(-1)
    equivalent = (q @ q.transpose(1, 2)).abs() > 1 - 1e-6
    earlier = torch.ones(q.shape[1], q.shape[1], device=q.device, dtype=torch.bool).tril(-1)
    unique = ~(equivalent & earlier[None]).any(-1)
    available = feasible.any(-1) & unique
    first = score.masked_fill(~available, torch.inf).argsort(dim=-1, stable=True)[:, : count // 2]
    available = available.clone()
    available.scatter_(1, first, False)
    second = unconstrained_score.masked_fill(~available, torch.inf).argsort(dim=-1, stable=True)[
        :, : count - count // 2
    ]
    choice = torch.cat((first, second), 1)
    q = q.gather(1, choice[..., None].expand(-1, -1, 4))
    fine_support, fine_mask = uniform_surface_support(support, mask, count=9984)

    def environment_check(center, aperture, feasible=None, *, quaternions):
        available = (
            torch.ones_like(aperture, dtype=torch.bool) if feasible is None else feasible.clone()
        )
        if "on_floor" in state:
            floor_gap = plane_check(quaternions, floor_only=True)(center, aperture, available)
            available &= ~state["on_floor"][:, None, None] | (floor_gap >= 0.002)
        result = aperture.new_full(aperture.shape, torch.nan)
        solved = torch.zeros_like(aperture[..., 0], dtype=torch.bool)
        for _ in range(aperture.shape[-1]):
            owner, candidate = torch.where(available.any(-1) & ~solved)
            if not len(owner):
                break
            depth = available.long().argmax(-1)[owner, candidate]
            c = center[owner, candidate, depth].clone()
            c[:, 2] -= 0.0005
            orientation = quaternions[owner, candidate]
            p = state["object_state"][owner, :3] - _quat_rotate(orientation, c)
            selected = geometry.select(owner)
            opened = collision.gripper_clearance(
                selected, p[:, None], orientation[:, None], maximum_distance_m=0.0021
            )[:, 0]
            closing = collision.gripper_closing_clearance(
                selected,
                p[:, None],
                orientation[:, None],
                aperture[owner, candidate, depth, None],
                maximum_distance_m=0.0021,
            )[:, 0]
            gap = torch.minimum(opened, closing)
            result[owner, candidate, depth] = gap
            available[owner, candidate, depth] = False
            solved[owner, candidate] = gap >= 0.002
        return result

    parts = []
    for block in q.split(128, 1):
        parts.append(
            convex_grasp_quality(
                block,
                state["object_state"][:, 3:7],
                fine_support,
                support_mask=fine_mask,
                depth_clearance=lambda center, aperture, feasible: environment_check(
                    center, aperture, feasible, quaternions=block
                ),
                masked_depth_clearance=True,
            )
        )
    quality = {key: torch.cat([part[key] for part in parts], 1) for key in parts[0]}
    center = quality["center_local"].clone()
    center[..., 2] -= 0.0005
    goals = state["object_state"][:, None, :3] - _quat_rotate(q, center)
    return q, goals, quality


def joint_station_requests(
    pocket, corridors, body_count, arrival_count, seeds, budget=1000, station_priors=None
):
    """Sample nominal joint proposals; robust arrival checks happen after ranking."""
    if budget < arrival_count or body_count < 1:
        raise ValueError("Joint station budget must cover one complete arrival stencil")
    batch, grasp_count = pocket.shape
    station_count = corridors.shape[1]
    if corridors.shape[0] != batch or seeds.shape != (batch,):
        raise ValueError("Joint proposals require scene-owned masks and seeds")
    count = budget
    draws = episode_uniform(seeds, 0x53544154, count, lanes=3)
    # Stable compaction samples valid marginal families without rejecting draws.
    stations = torch.argsort(~corridors, stable=True)
    grasps = torch.argsort(~pocket, stable=True)
    station_rank = (draws[..., 0] * corridors.sum(-1)[:, None]).long()
    grasp_rank = (draws[..., 1] * pocket.sum(-1)[:, None]).long()
    station = stations.gather(1, station_rank)
    grasp = grasps.gather(1, grasp_rank)
    body = (draws[..., 2] * body_count).long()
    slots = torch.arange(count, device=pocket.device)
    if station_priors is not None:
        structured = min(count // 2, 16 * body_count * station_priors.shape[1])
        ranks = (slots // (body_count * station_priors.shape[1]))[None]
        ranks = torch.minimum(ranks, (pocket.sum(-1) - 1).clamp_min(0)[:, None])
        preferred = grasps.gather(1, ranks.expand(batch, -1))
        prior = station_priors[:, slots % station_priors.shape[1]]
        station = torch.where(slots[None] < structured, prior, station)
        grasp = torch.where(slots[None] < structured, preferred, grasp)
        body = torch.where(
            slots[None] < structured, (slots // station_priors.shape[1])[None] % body_count, body
        )
    else:
        body = torch.where(slots[None] < body_count, slots[None] % body_count, body)
    own = torch.arange(batch, device=pocket.device)[:, None].expand(-1, count)
    code = ((own * station_count + station) * body_count + body) * grasp_count + grasp
    code = torch.unique(code[pocket.any(-1) & corridors.any(-1)])
    grasp = code % grasp_count
    code = code // grasp_count
    body = code % body_count
    code = code // body_count
    station = code % station_count
    own = code // station_count
    nominal = min(1, arrival_count - 1)
    posture = (station * body_count + body) * arrival_count + nominal
    return own, posture, grasp
