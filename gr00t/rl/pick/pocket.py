"""Asset-derived pocket bounds, using the same inner jaw colliders as AutoOpen."""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
import warp as wp

from gr00t.rl.auto_open.geometry import ROBOT_COLLIDER_SPECS, _read_collider_points

GRIP_STATIC_FRICTION = 1.1
GRIP_DYNAMIC_FRICTION = 1.0
GRASP_FRICTION = 0.75
PICK_OBJECT_CONTACT_OFFSET_M = 0.002
BASKET_MINIMUM_CENTER_DEPTH_M = 0.005


@wp.kernel
def _distal_support_kernel(
    angles: wp.array(dtype=float),
    profile: wp.array2d(dtype=float),
    count: int,
    result: wp.array(dtype=float),
):
    i = wp.tid()
    c = wp.cos(angles[i])
    s = wp.sin(angles[i])
    maximum = float(-1.0e30)
    for j in range(count):
        maximum = wp.max(maximum, c * profile[j, 0] - s * profile[j, 1])
    result[i] = maximum


@lru_cache(maxsize=1)
def _contact_colliders():
    from pxr import Usd

    from gr00t.rl.utils.b2z1_asset import ROBOT_USD

    stage = Usd.Stage.Open(str(ROBOT_USD))
    result = {}
    for body_name, name, tokens in ROBOT_COLLIDER_SPECS:
        if name not in ("fixed_contact_surfaces", "moving_contact_surfaces"):
            continue
        body = next(prim for prim in stage.Traverse() if prim.GetName() == body_name)
        result[name] = _read_collider_points(stage, body, tokens).triangles_body
    return result


class ContactProfiles:
    """GPU ray-derived inner surfaces, preserving the actual pocket curvature."""

    def __init__(self, device):
        from gr00t.rl.pick.scene_geometry import SceneGeometry

        self.maps = {}
        self.meshes = []
        x = torch.linspace(-0.1, 0.12, 441, device=device)
        y = torch.linspace(-0.05, 0.05, 201, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        for name, triangles in _contact_colliders().items():
            points = torch.as_tensor(triangles, device=device).reshape(-1, 3)
            fixed = name == "fixed_contact_surfaces"
            if fixed:
                points = (points - points.new_tensor([0.1, 0, 0])) @ points.new_tensor(
                    [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
                )
            mesh = SceneGeometry(
                points, torch.arange(len(points), device=device).reshape(-1, 3), device
            )
            self.meshes.append(mesh)
            origins = torch.stack((xx, yy, torch.full_like(xx, 0.2 if fixed else -0.2)), -1)
            targets = origins.clone()
            targets[..., 2] *= -1
            hit, normals, _ = mesh.rays(origins, targets)
            # A missed ray returns its target, whereas every jaw surface lies within +/-0.1 m.
            valid = (hit[..., 2].abs() < 0.1).to(xx.dtype)
            values = torch.cat((hit[..., 2:3], normals, valid[..., None]), -1)
            self.maps["fixed" if fixed else "moving"] = values.permute(2, 0, 1)[None].contiguous()

    def sample(self, name, xy, *, normals=True):
        shape = xy.shape[:-1]
        grid = torch.stack(((xy[..., 0] - 0.01) / 0.11, xy[..., 1] / 0.05), -1)
        source = self.maps[name] if normals else self.maps[name][:, [0, 4]]
        values = F.grid_sample(
            source, grid.reshape(1, 1, -1, 2), align_corners=True, padding_mode="zeros"
        )[0, :, 0].T.reshape(*shape, source.shape[1])
        if not normals:
            return values[..., 0], None, values[..., 1] > 0.99
        normal = values[..., 1:4]
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return values[..., 0], normal, values[..., 4] > 0.99


@lru_cache(maxsize=8)
def gripper_contact_profiles(device):
    return ContactProfiles(device)


@lru_cache(maxsize=8)
def _moving_vertices(device, dtype):
    return (
        torch.as_tensor(_contact_colliders()["moving_contact_surfaces"], device=device, dtype=dtype)
        .reshape(-1, 3)
        .unique(dim=0)
    )


@lru_cache(maxsize=8)
def _moving_distal_profile(device, dtype):
    return _moving_vertices(device, dtype)[:, [0, 2]].unique(dim=0)


def jaw_distal_limits(aperture):
    """Each jaw's actual distal edge in EE coordinates at this opening angle."""
    points = _moving_distal_profile(str(aperture.device), aperture.dtype)
    if aperture.is_cuda and aperture.dtype == torch.float32:
        wp.init()
        angles = aperture.contiguous().flatten()
        moving = torch.empty_like(angles)
        wp.launch(
            _distal_support_kernel,
            dim=angles.numel(),
            inputs=[wp.from_torch(angles), wp.from_torch(points), len(points)],
            outputs=[wp.from_torch(moving)],
            stream=wp.stream_from_torch(torch.cuda.current_stream(aperture.device)),
        )
        moving = moving.reshape(aperture.shape)
    else:
        moving = (
            aperture.cos()[..., None] * points[:, 0] - aperture.sin()[..., None] * points[:, 1]
        ).amax(-1)
    fixed = gripper_pocket_geometry(str(aperture.device), aperture.dtype)["front_x"]
    return torch.stack((torch.ones_like(moving) * fixed, moving), -1)


@lru_cache(maxsize=8)
def gripper_pocket_geometry(device="cpu", dtype=torch.float32):
    from pxr import Usd

    from gr00t.rl.utils.b2z1_asset import ROBOT_USD

    stage = Usd.Stage.Open(str(ROBOT_USD))
    specs = {name: (body, tokens) for body, name, tokens in ROBOT_COLLIDER_SPECS}
    clouds = {}
    for name in ("fixed_finger", "moving_finger"):
        body_name, tokens = specs[name]
        body = next(prim for prim in stage.Traverse() if prim.GetName() == body_name)
        geometry = _read_collider_points(stage, body, tokens)
        clouds[name] = torch.as_tensor(geometry.points_body, device=device, dtype=dtype)
    fixed = clouds["fixed_finger"]
    # Same link06 -> TCP transform used by ArmCollisionSamples.gripper_clearance.
    fixed = (fixed - fixed.new_tensor([0.100, 0, 0])) @ fixed.new_tensor(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    )
    moving = clouds["moving_finger"]
    angle = fixed.new_tensor(math.pi / 3)
    c, s = angle.cos(), angle.sin()
    rotation = torch.stack(
        (torch.stack((c, c * 0, -s)), fixed.new_tensor([0, 1, 0]), torch.stack((s, s * 0, c)))
    )
    opened = moving @ rotation.T
    fixed_tip = fixed[fixed[:, 0] >= torch.quantile(fixed[:, 0], 0.95), 0].mean()
    moving_tip = opened[opened[:, 0] >= torch.quantile(opened[:, 0], 0.95), 0].mean()
    # AutoOpen seats the captured target 10 mm past the proximal jaw envelope.
    back = torch.maximum(fixed[:, 0].min(), opened[:, 0].min()) - 0.010
    return {
        "back_x": back,
        "front_x": fixed[:, 0].max(),
        "open_tip_midpoint_x": (fixed_tip + moving_tip) / 2,
        "half_width": torch.minimum(fixed[:, 1].abs().max(), moving[:, 1].abs().max()),
        "fixed_plane_z": fixed[:, 2].max(),
        "moving_plane_offset": -moving[:, 2].min(),
        "open_angle": angle,
    }
