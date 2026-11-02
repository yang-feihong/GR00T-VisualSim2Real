"""GPU convex-support contact estimates for a hinged two-jaw gripper."""

import math
from functools import lru_cache
from pathlib import Path

import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.pocket import (
    GRASP_FRICTION,
    gripper_contact_profiles,
    gripper_pocket_geometry,
    jaw_distal_limits,
)


def grasp_support_from_hull(points, faces=None):
    """The same vertex/face support samples for baked assets and offline admission."""
    if faces is None:
        return points
    triangles = points[faces]
    weights = points.new_tensor(
        [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]]
    )
    surface = torch.einsum("sj,fjk->fsk", weights, triangles).reshape(-1, 3)
    return torch.cat((points, surface))


def load_grasp_support(usd_path, device):
    """Load topology once, then add face samples on the planning device."""
    import numpy as np

    directory = Path(usd_path).parent
    points = torch.as_tensor(np.load(directory / "collision_support.npy"), device=device)
    faces_path = directory / "collision_faces.npy"
    faces = (
        None
        if not faces_path.exists()
        else torch.as_tensor(np.load(faces_path), device=device, dtype=torch.long)
    )
    return grasp_support_from_hull(points, faces)


@lru_cache(maxsize=128)
def load_grasp_center(usd_path, device):
    """Uniform-density convex-hull centroid, reduced on the planning device."""
    import numpy as np

    directory = Path(usd_path).parent
    points = torch.as_tensor(
        np.load(directory / "collision_support.npy"), device=device, dtype=torch.float64
    )
    faces = torch.as_tensor(
        np.load(directory / "collision_faces.npy"), device=device, dtype=torch.long
    )
    a, b, c = points[faces].unbind(1)
    volume6 = (a * torch.cross(b, c, dim=-1)).sum(-1)
    total = volume6.sum()
    if not bool(torch.isfinite(total)) or abs(float(total)) < 1e-12:
        raise ValueError("Closed convex hull required for a grasp center of mass")
    return ((volume6[:, None] * (a + b + c)).sum(0) / (4 * total)).float()


def first_closing_contact(open_angle, gap):
    """Bracket the earliest sampled contact while closing a finite jaw."""
    sweep = torch.linspace(0, 1, 17, device=open_angle.device, dtype=open_angle.dtype)
    sweep_gap = gap(open_angle[..., None] * sweep)
    last_collision = torch.where(
        sweep_gap < 0, torch.arange(len(sweep), device=open_angle.device), 0
    ).amax(-1)
    lower = open_angle * sweep[last_collision]
    upper = open_angle * sweep[(last_collision + 1).clamp_max(len(sweep) - 1)]
    for _ in range(16):
        angle = (lower + upper) / 2
        distance = gap(angle)
        lower = torch.where(distance < 0, angle, lower)
        upper = torch.where(distance >= 0, angle, upper)
    return (lower + upper) / 2


def supported_pinch_alignment(orientations, fixed, moving, support_normal, max_tilt_deg=10.0):
    """Require a nearly level pinch, allowing either jaw to be the lower one.

    The support must not become the opposing finger while the hinged jaw closes.
    This geometric screen is not a substitute for measured bilateral contact.
    """
    direction = _quat_rotate(orientations, moving - fixed)
    length = direction.norm(dim=-1)
    normal = support_normal / support_normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sine = (direction * normal[:, None]).sum(-1) / length.clamp_min(1e-8)
    compatible = (length > 1e-6) & (sine.abs() <= math.sin(math.radians(max_tilt_deg)))
    return compatible, sine


def convex_grasp_quality(
    orientations,
    object_orientation,
    support_points,
    friction=GRASP_FRICTION,
    *,
    depth_clearance=None,
    depth_preference=0.0,
    minimum_environment_clearance_m=0.002,
    minimum_center_depth_m=0.0015,
    masked_depth_clearance=False,
    support_normal_world=None,
    return_depth_candidates=False,
    support_mask=None,
    depth_fractions=None,
    centers_only=False,
):
    if not math.isfinite(minimum_center_depth_m) or minimum_center_depth_m < 0.0015:
        raise ValueError("Minimum center depth must be finite and at least 0.0015 m")
    pocket = gripper_pocket_geometry(str(orientations.device), orientations.dtype)
    profiles = gripper_contact_profiles(str(orientations.device))
    if support_points.ndim == 3:
        if support_mask is None or support_mask.shape != support_points.shape[:2]:
            raise ValueError("Batched hull samples require a matching support mask")
        # Repeated valid support samples leave min/max and the fused contact
        # kernel unchanged; mask them out of contact centroids and normals.
        first = support_mask.long().argmax(-1)
        reference = support_points.gather(1, first[:, None, None].expand(-1, 1, 3))
        support_points = torch.where(support_mask[..., None], support_points, reference)
        world = _quat_rotate(object_orientation[:, None], support_points)
    else:
        if support_mask is not None:
            raise ValueError("Support masks require batched hull samples")
        world = _quat_rotate(object_orientation[:, None], support_points[None])
    inverse = orientations.clone()
    inverse[..., 1:] *= -1
    local = _quat_rotate(inverse[:, :, None], world[:, None])
    minimum = pocket["back_x"] + 0.0015 - local[..., 0].amin(-1)
    maximum = torch.ones_like(minimum) * (pocket["front_x"] - 0.0015)
    depth = (
        torch.linspace(0, 1, 17, device=local.device, dtype=local.dtype)
        if depth_fractions is None
        else depth_fractions
    )
    center = local.new_zeros(*local.shape[:2], depth.shape[-1], 3)
    center[..., 0] = minimum[..., None] + depth * (maximum - minimum).clamp_min(0)[..., None]
    fused = local.is_cuda and local.dtype == torch.float32
    if fused:
        from gr00t.rl.pick.contact_profile_kernel import fixed_profile_centers

        center, has_fixed = fixed_profile_centers(local, center, profiles.maps["fixed"])
    else:
        points = local[:, :, None] + center[..., None, :]
        fixed_height, fixed_normals, fixed_valid = profiles.sample(
            "fixed", points[..., :2], normals=not centers_only
        )
        if support_mask is not None:
            fixed_valid = fixed_valid & support_mask[:, None, None]
        required = (fixed_height - points[..., 2]).masked_fill(~fixed_valid, -torch.inf).amax(-1)
        has_fixed = torch.isfinite(required)
        center[..., 2] = torch.where(has_fixed, required, torch.zeros_like(required))
    if centers_only:
        return {
            "center_local": center,
            "pocket_fit": has_fixed & (maximum >= minimum)[..., None],
        }
    if not fused:
        points = local[:, :, None] + center[..., None, :]

    def moving_surface(angle, *, normals=False):
        extra_axes = angle.ndim - (points.ndim - 2)
        surface_points = points.reshape(
            *points.shape[:-2], *((1,) * extra_axes), *points.shape[-2:]
        )
        c, s = angle.cos()[..., None], angle.sin()[..., None]
        x = surface_points[..., 0] * c + surface_points[..., 2] * s
        z = -surface_points[..., 0] * s + surface_points[..., 2] * c
        height, normal, valid = profiles.sample(
            "moving", torch.stack((x, surface_points[..., 1].expand_as(x)), -1), normals=normals
        )
        clearance = (height - z).masked_fill(~valid, torch.inf)
        if support_mask is not None:
            mask = support_mask.reshape(len(support_mask), 1, 1, *((1,) * extra_axes), -1)
            clearance = clearance.masked_fill(~mask, torch.inf)
        if normals:
            nx, ny, nz = normal.unbind(-1)
            normal = torch.stack((c * nx - s * nz, ny, s * nx + c * nz), -1)
        return clearance, normal

    def moving_gap_minimum(angle):
        if local.device.type == "cuda" and local.dtype == torch.float32:
            from gr00t.rl.pick.contact_profile_kernel import moving_profile_minimum

            return moving_profile_minimum(local, center, angle, profiles.maps["moving"])
        return moving_surface(angle)[0].amin(-1)

    open_angle = torch.ones_like(center[..., 0]) * pocket["open_angle"]
    open_distance = moving_gap_minimum(open_angle)
    closed_distance = moving_gap_minimum(open_angle * 0)
    tip_clearance = pocket["front_x"] - (local[..., 0].amax(-1)[..., None] + center[..., 0])
    pocket_fit = has_fixed & (closed_distance <= 0) & (open_distance >= 0.0015)
    # A finite jaw can touch an overhanging object and later pass it. Bracket
    # the first collision from fully open before refining, not the last root.
    if local.device.type == "cuda" and local.dtype == torch.float32:
        from gr00t.rl.pick.contact_profile_kernel import first_profile_contact

        angle = first_profile_contact(local, center, open_angle, profiles.maps["moving"])
    else:
        angle = first_closing_contact(open_angle, moving_gap_minimum)
    if fused:
        from gr00t.rl.pick.contact_profile_kernel import profile_contact_statistics

        contacts, normals, _ = profile_contact_statistics(
            local, center, angle, profiles.maps["fixed"], profiles.maps["moving"], support_mask
        )
        fixed, moving = contacts.unbind(-2)
        fixed_normal, moving_normal = normals.unbind(-2)
    else:
        moving_gap, moving_normals = moving_surface(angle, normals=True)
        fixed_gap = (points[..., 2] - fixed_height).masked_fill(~fixed_valid, torch.inf)
        fixed_patch = fixed_gap <= fixed_gap.amin(-1, keepdim=True) + 0.0003
        moving_patch = moving_gap <= moving_gap.amin(-1, keepdim=True) + 0.0003
        fixed_patch &= fixed_valid
        moving_patch &= torch.isfinite(moving_gap)
        fixed = (
            (points * fixed_patch[..., None]).sum(-2, dtype=torch.float64)
            / fixed_patch.sum(-1, keepdim=True).clamp_min(1)
        ).to(points.dtype)
        moving = (
            (points * moving_patch[..., None]).sum(-2, dtype=torch.float64)
            / moving_patch.sum(-1, keepdim=True).clamp_min(1)
        ).to(points.dtype)
        fixed_normal = (fixed_normals * fixed_patch[..., None]).sum(-2, dtype=torch.float64)
        moving_normal = (moving_normals * moving_patch[..., None]).sum(-2, dtype=torch.float64)
        fixed_normal /= fixed_normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        moving_normal /= moving_normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        fixed_normal = fixed_normal.to(points.dtype)
        moving_normal = moving_normal.to(points.dtype)
    distal = jaw_distal_limits(angle)
    contact_depth = (distal - torch.stack((fixed[..., 0], moving[..., 0]), -1)).amin(-1)
    center_depth = distal.amin(-1) - center[..., 0]
    # Long objects may overhang the jaw sides; the contact patches must fit.
    width_clearance = pocket["half_width"] - torch.stack(
        (fixed[..., 1].abs(), moving[..., 1].abs()), -1
    ).amax(-1)
    pocket_fit &= (
        (contact_depth >= 0.0015)
        & (center_depth >= minimum_center_depth_m)
        & (width_clearance >= 0.0015)
    )
    direction = moving - fixed
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cosine_fixed = (direction * fixed_normal).sum(-1)
    cosine_moving = -(direction * moving_normal).sum(-1)
    margin = torch.minimum(cosine_fixed, cosine_moving) - 1 / math.sqrt(1 + friction * friction)
    feasible = pocket_fit & (margin >= 0) & (maximum >= minimum)[..., None]
    support_fit = support_sine = None
    if support_normal_world is not None:
        support_fit, support_sine = supported_pinch_alignment(
            orientations[:, :, None].expand(*fixed.shape[:-1], 4).flatten(1, 2),
            fixed.flatten(1, 2),
            moving.flatten(1, 2),
            support_normal_world,
        )
        support_fit = support_fit.reshape(feasible.shape)
        support_sine = support_sine.reshape(feasible.shape)
        feasible &= support_fit
    environment_clearance = None
    environment_evaluated = None
    if depth_clearance is not None:
        if masked_depth_clearance:
            environment_evaluated = feasible.clone()
            environment_clearance = depth_clearance(center, angle, environment_evaluated)
        else:
            environment_clearance = depth_clearance(center, angle)
        feasible = feasible & (environment_clearance >= minimum_environment_clearance_m)
    preference = torch.as_tensor(depth_preference, device=local.device, dtype=local.dtype)
    depth_cost = (depth - preference[..., None]).abs().expand_as(feasible)
    selected = torch.where(
        feasible.any(-1), depth_cost.masked_fill(~feasible, torch.inf).argmin(-1), margin.argmax(-1)
    )
    values = {
        "center_local": center,
        "depth_fraction": depth.expand_as(center[..., 0]),
        "aperture_angle_rad": angle,
        "fixed_contact_local": fixed,
        "moving_contact_local": moving,
        "fixed_contact_normal_local": fixed_normal,
        "moving_contact_normal_local": moving_normal,
        "antipodal_margin": margin,
        "pocket_fit": pocket_fit & (maximum >= minimum)[..., None],
        "open_jaw_clearance_m": open_distance,
        "tip_clearance_m": tip_clearance,
        "width_clearance_m": width_clearance,
        "contact_depth_margin_m": contact_depth,
        "center_depth_margin_m": center_depth,
    }
    if environment_clearance is not None:
        values["environment_clearance_m"] = environment_clearance
    if environment_evaluated is not None:
        values["environment_query_evaluated"] = environment_evaluated
    if support_fit is not None:
        values["support_pinch_compatible"] = support_fit
        values["support_contact_line_sine"] = support_sine
    values = {
        key: value.expand(feasible.shape[0], *value.shape[1:]) for key, value in values.items()
    }
    if return_depth_candidates:
        return values
    return {
        key: value.gather(
            2,
            selected[..., None, None].expand(-1, -1, 1, 3)
            if value.ndim == 4
            else selected[..., None],
        ).squeeze(2)
        for key, value in values.items()
    }


def main():
    import argparse
    import json
    from pathlib import Path

    import numpy as np

    from gr00t.rl.pick.planning import grasp_approach_origin, grasp_pose_bank

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episode = json.loads((args.run / "episode.json").read_text())
    state = torch.load(args.run / "trajectory.pt", weights_only=False)[args.frame]
    target = state["object_state"][:, :3].cuda()
    root = state["root_states"].cuda()
    heading = torch.atan2(target[:, 1] - root[:, 1], target[:, 0] - root[:, 0])
    _, orientations, _ = grasp_pose_bank(
        target, heading, episode["asset"]["size_m"], grasp_approach_origin(root)
    )
    path = Path(episode["asset"]["usd_path"]).with_name("collision_support.npy")
    points = torch.tensor(np.load(path), device=target.device)
    quality = convex_grasp_quality(orientations, state["object_state"][:, 3:7].cuda(), points)
    report = {key: value.cpu().tolist() for key, value in quality.items()}
    report["source_run"] = str(args.run)
    report["source_frame"] = args.frame
    report["support_vertex_count"] = len(points)
    feasible = quality["pocket_fit"] & (quality["antipodal_margin"] >= 0)
    report["feasible"] = feasible.cpu().tolist()
    report["pocket_geometry"] = {
        key: float(value) for key, value in gripper_pocket_geometry(str(target.device)).items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "feasible_count": int(feasible.sum())}))


if __name__ == "__main__":
    main()
