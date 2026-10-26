"""Pull-door stations clearing the frame along its width direction."""

from dataclasses import dataclass

import numpy as np
import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_dot,
)


@dataclass(frozen=True)
class PullStationConfig:
    panel_side_clearance_m: float = 0.20
    # Hinge on the left/right, viewed from the robot's approach side.
    shoulder_hinge_radius_m: tuple[float, float] = (0.40, 0.60)
    clearance_m: float = 0.0
    wall_clearance_m: float = 0.02
    planning_margin_m: float = 0.05
    first_candidate_count: int = 32
    first_max_station_attempts: int = 4
    first_x_range_m: tuple[float, float] = (0.535, 0.555)
    first_y_range_m: tuple[float, float] = (0.630, 0.650)
    first_yaw_range_deg: tuple[float, float] = (-132., -126.)
    panel_contact_tangent_search_radius_m: float = 0.40
    panel_contact_forward_offset_m: float = 0.20
    panel_contact_baseline: str = "trunk_front_minus10"

    def __post_init__(self):
        radii = np.asarray(self.shoulder_hinge_radius_m)
        if radii.shape != (2,) or not np.isfinite(radii).all() or (radii <= 0.).any():
            raise ValueError("shoulder_hinge_radius_m must contain positive finite left/right radii")
        if self.panel_contact_baseline not in ("base_forward", "trunk_front", "trunk_front_minus10"):
            raise ValueError("Unknown pull-door panel-contact baseline")


def angle_to_wall(yaw, wall_tangent, wall_normal):
    forward = torch.stack((yaw.cos(), yaw.sin()), -1)
    return torch.atan2(fixed_dot(forward, wall_normal).abs(),
                       fixed_dot(forward, wall_tangent).abs())


def shoulder_wall_curve(anchor, shoulder_local, hinge, wall_tangent, wall_normal,
                        entry_yaw, body_width, clearance_m=0.20, samples=257,
                        shoulder_hinge_radius_m=0.50):
    """Precompute the user's arcsin geometry, without a door-angle input."""
    along = fixed_dot(anchor - hinge, wall_tangent).clamp_min(0.)
    fraction = torch.linspace(0., 1., samples, device=anchor.device, dtype=anchor.dtype)
    shoulder = anchor[:, None] - along[:, None, None] * fraction[None, :, None] * wall_tangent[:, None]
    normal_distance = fixed_dot(anchor - hinge, wall_normal)
    initial_radius = torch.sqrt(along.square() + normal_distance.square())
    radius = initial_radius.clamp(max=shoulder_hinge_radius_m)
    turn_along = torch.sqrt((radius.square() - normal_distance.square()).clamp_min(0.))
    straight_length = (along - turn_along).clamp_min(0.)
    start_angle = torch.atan2(normal_distance.abs(), turn_along)
    arc_angle = .5 * torch.pi - start_angle
    progress = fraction[None] * (straight_length + radius * arc_angle)[:, None]
    angle = start_angle[:, None] + (progress - straight_length[:, None]).clamp_min(0.) / radius[:, None].clamp_min(1.e-8)
    normal_sign = torch.where(normal_distance >= 0., 1., -1.)
    arc = (hinge[:, None] + (radius[:, None] * angle.cos())[..., None] * wall_tangent[:, None]
           + (normal_sign[:, None] * radius[:, None] * angle.sin())[..., None] * wall_normal[:, None])
    line = anchor[:, None] - progress[..., None] * wall_tangent[:, None]
    curved = torch.where((progress <= straight_length[:, None])[..., None], line, arc)
    # If the wall-parallel line never reaches the radius, retain that line.
    reaches_radius = normal_distance.abs() < radius
    shoulder = torch.where(reaches_radius[:, None, None], curved, shoulder)
    ab = hinge[:, None] - shoulder
    a = torch.sqrt(fixed_dot(ab, ab))
    alpha = torch.atan2(fixed_dot(ab, wall_normal[:, None]).abs(),
                        fixed_dot(ab, wall_tangent[:, None]).abs())
    lower = angle_to_wall(entry_yaw, wall_tangent, wall_normal)
    b = body_width + clearance_m
    beta = torch.asin((b[:, None] / a.clamp_min(1.e-8)).clamp(max=1.)) + alpha
    beta = torch.maximum(beta, lower[:, None]).clamp(max=0.5 * torch.pi)
    # Resolve the initial yaw change about the same shoulder with speed-bounded
    # samples, rather than jumping to the first geometric heading.
    initial_fraction = torch.linspace(0., 1., 33, device=anchor.device, dtype=anchor.dtype)
    initial_beta = lower[:, None] + initial_fraction[None] * (beta[:, :1] - lower[:, None])
    beta = torch.cat((initial_beta[:, :-1], beta), dim=1)
    shoulder = torch.cat((anchor[:, None].expand(-1, 32, -1), shoulder), dim=1)
    forward = (-beta.cos()[..., None] * wall_tangent[:, None]
               - beta.sin()[..., None] * wall_normal[:, None])
    yaw = torch.atan2(forward[..., 1], forward[..., 0])
    c, s = yaw.cos(), yaw.sin()
    offset = torch.stack((c * shoulder_local[:, :1] - s * shoulder_local[:, 1:2],
                          s * shoulder_local[:, :1] + c * shoulder_local[:, 1:2]), -1)
    return torch.cat((shoulder - offset, yaw[..., None]), dim=-1)


def base_collider_points(controller):
    """Read enabled trunk colliders in the base_link frame."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from gr00t.rl.auto_open.geometry import _stage_robot_root

    stage = omni.usd.get_context().get_stage()
    root = _stage_robot_root(stage, controller.env_id)
    if root is None:
        raise RuntimeError("Robot root is unavailable for pull-station geometry")
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    body = next((prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                 if prim.GetName() == "base_link" and prim.HasAPI(UsdPhysics.RigidBodyAPI)), None)
    if body is None:
        raise RuntimeError("base_link is unavailable for pull-station geometry")
    inverse = transforms.GetLocalToWorldTransform(body).GetInverse()
    clouds = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.GetAttribute("physics:collisionEnabled").Get() is False:
            continue
        transform = transforms.GetLocalToWorldTransform(prim) * inverse
        clouds.extend(transform.Transform(Gf.Vec3d(*point))
                      for point in collider_local_corners(prim).tolist())
    if not clouds:
        raise RuntimeError("No base collision geometry found for pull station")
    return np.asarray(clouds, dtype=np.float32)


def robot_collider_cache(controller):
    """Cache enabled robot collision corners in their owning rigid-body frames."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from gr00t.rl.auto_open.geometry import _stage_robot_root

    stage = omni.usd.get_context().get_stage()
    root = _stage_robot_root(stage, controller.env_id)
    indices = {name: i for i, name in enumerate(controller.env.body_names)}
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    points, bodies = [], []
    for body in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not body.HasAPI(UsdPhysics.RigidBodyAPI) or body.GetName() not in indices:
            continue
        inverse = transforms.GetLocalToWorldTransform(body).GetInverse()
        body_points = []
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:collisionEnabled').Get() is False:
                continue
            transform = transforms.GetLocalToWorldTransform(prim) * inverse
            corners = [transform.Transform(Gf.Vec3d(*point)) for point in collider_local_corners(prim).tolist()]
            body_points.extend(corners)
        if body_points:
            values = np.asarray(body_points)
            lower, upper = values.min(axis=0), values.max(axis=0)
            points.append([[x, y, z] for x in (lower[0], upper[0])
                           for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
            bodies.append(indices[body.GetName()])
    if not points:
        raise RuntimeError('No robot collision geometry available for base motion')
    return np.asarray(points, dtype=np.float32), np.asarray(bodies, dtype=np.int64)


def robot_motion_envelope(root, body_states, points_local, body_indices):
    """Keep separate rigid-body envelopes in a level root-yaw reference frame."""
    poses = body_states[:, body_indices, :7]
    q = poses[..., None, 3:7]
    local = points_local[None].expand(len(root), -1, -1, -1)
    cross = torch.cross(q[..., 1:], local, dim=-1)
    world = local + 2. * (q[..., :1] * cross + torch.cross(q[..., 1:], cross, dim=-1)) + poses[..., None, :3]
    relative = world - root[:, None, None, :3]
    w, x, y, z = root[:, 3:7].unbind(dim=-1)
    yaw = torch.atan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))
    c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    return torch.stack((c * relative[..., 0] + s * relative[..., 1],
                        -s * relative[..., 0] + c * relative[..., 1], relative[..., 2]), dim=-1)


def frame_boundaries(controller):
    """Read the opening edge and robot-facing wall surface from USD."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    door = stage.GetPrimAtPath(controller._door_prim_path())
    hinges = [prim for prim in Usd.PrimRange(door, Usd.TraverseInstanceProxies())
              if prim.GetName() == "hinge_joint"]
    if len(hinges) != 1:
        raise RuntimeError("Expected one door hinge for static frame lookup")
    roots = hinges[0].GetRelationship("physics:body0").GetTargets()
    if len(roots) != 1:
        raise RuntimeError("Door hinge does not identify a unique static frame root")
    clouds = {}
    for name in ("top_frame", "left_frame", "right_frame"):
        prim = stage.GetPrimAtPath(roots[0].AppendChild(name))
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Door frame collider is unavailable: {name}")
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        clouds[name] = np.asarray([transform.Transform(Gf.Vec3d(*point))
                                  for point in collider_local_corners(prim).tolist()])
    free_edge = float(((clouds["top_frame"] - controller.hinge_world)
                       @ controller.panel_hinge_to_free_direction).max())
    wall_points = np.concatenate((clouds["left_frame"], clouds["right_frame"]))
    wall_front = float(((wall_points - controller.hinge_world) @ controller.door_normal).max())
    return free_edge, wall_front


def door_hardware_height_bounds(controller):
    """Vertical extent of the actual handle/latch collision shapes."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(controller._door_prim_path())
    points = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not any(token in path for token in ('/door_handle/', '/latch_link/')):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:collisionEnabled').Get() is False:
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        points.extend(transform.Transform(Gf.Vec3d(*point)) for point in collider_local_corners(prim).tolist())
    if not points:
        raise RuntimeError('Door handle/latch collision geometry is unavailable')
    z = np.asarray(points)[:, 2]
    return float(z.min()), float(z.max())


def collider_local_corners(prim):
    """Use physical shape dimensions, not visibility/purpose or authored extent.

    The robot's box extents already contain dimensions that are also present
    in Xform scale. BBoxCache therefore double-scales them; PhysX uses size.
    """
    from pxr import Gf, Usd, UsdGeom

    if prim.IsA(UsdGeom.Xform):
        transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        inverse = transforms.GetLocalToWorldTransform(prim).GetInverse()
        points = []
        for child in prim.GetChildren():
            if not child.IsA(UsdGeom.Xformable):
                continue
            transform = transforms.GetLocalToWorldTransform(child) * inverse
            points.extend(transform.Transform(Gf.Vec3d(*point))
                          for point in collider_local_corners(child).tolist())
        if not points:
            raise RuntimeError(f"Empty collision hierarchy: {prim.GetPath()}")
        return np.asarray(points)

    if prim.IsA(UsdGeom.Mesh):
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        if not points.size:
            raise RuntimeError(f"Empty collision mesh: {prim.GetPath()}")
        lower, upper = points.min(axis=0), points.max(axis=0)
    else:
        if prim.IsA(UsdGeom.Cube):
            half = np.full(3, float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2.)
        elif prim.IsA(UsdGeom.Sphere):
            half = np.full(3, float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
        else:
            shape_type = next((cls for cls in (UsdGeom.Capsule, UsdGeom.Cylinder, UsdGeom.Cone)
                               if prim.IsA(cls)), None)
            if shape_type is None:
                raise RuntimeError(f"Unsupported robot collider: {prim.GetPath()} ({prim.GetTypeName()})")
            shape = shape_type(prim)
            radius = float(shape.GetRadiusAttr().Get())
            half = np.full(3, radius)
            axis = {'X': 0, 'Y': 1, 'Z': 2}[str(shape.GetAxisAttr().Get()).upper()]
            half[axis] = float(shape.GetHeightAttr().Get()) / 2.
            if shape_type is UsdGeom.Capsule:
                half[axis] += radius
        lower, upper = -half, half
    return np.asarray([[x, y, z] for x in (lower[0], upper[0])
                       for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])


def base_projection_clearances(origin_xy, axes_xy, boundaries_m, base_points_world):
    """Minimum trunk projection minus boundary on each of the two door axes.

    Supports actual poses [env, point, xy] and candidate poses
    [env, candidate, point, xy] through the same calculation.
    """
    batch = len(origin_xy)
    extra_dims = base_points_world.ndim - 2
    origin = origin_xy.reshape(batch, *([1] * extra_dims), 2)
    axes = axes_xy.reshape(batch, *([1] * extra_dims), 2, 2)
    projection = fixed_dot((base_points_world - origin).unsqueeze(-2), axes)
    boundaries = boundaries_m.reshape(batch, *([1] * (extra_dims - 1)), 2)
    return projection.amin(dim=-2) - boundaries


def pull_station_parameter_grid(hinge_xy, normal_xy, free_direction_xy, door_width_m, *,
                               position_angles_deg=(15., 20., 25.), heading_offsets_deg=(0., 5., 10.)):
    """Fixed station grid: radius W+.77, wall angle x, heading angle x+d."""
    x, delta = torch.meshgrid(
        torch.tensor(position_angles_deg, dtype=hinge_xy.dtype, device=hinge_xy.device),
        torch.tensor(heading_offsets_deg, dtype=hinge_xy.dtype, device=hinge_xy.device),
        indexing='ij')
    x, delta = x.flatten(), delta.flatten()
    angle = torch.deg2rad(x)
    radius = door_width_m + .77
    direction = angle.sin()[None, :, None] * normal_xy[:, None] + angle.cos()[None, :, None] * free_direction_xy[:, None]
    positions = hinge_xy[:, None] + radius[:, None, None] * direction
    heading_angle = torch.deg2rad(x + delta)
    forward = -(heading_angle.sin()[None, :, None] * normal_xy[:, None]
                + heading_angle.cos()[None, :, None] * free_direction_xy[:, None])
    yaws = torch.atan2(forward[..., 1], forward[..., 0])
    return positions, yaws, torch.stack((x, delta), -1)

