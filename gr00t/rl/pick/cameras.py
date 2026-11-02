"""Pick-only rendering setup and deterministic diagnostic viewpoints."""

import torch

DIAGNOSTIC_VIEWS = (
    "robot_top_diagnostic",
    "object_side_diagnostic",
    "object_top_diagnostic",
    "robot_rear_diagnostic",
)


def hide_auxiliary_ground_mesh(stage, terrain_prim_path):
    from pxr import UsdGeom

    path = f"{terrain_prim_path}/terrain/Environment/Geometry"
    mesh = stage.GetPrimAtPath(path)
    if not mesh.IsValid() or not mesh.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"Expected auxiliary ground visual mesh at {path}")
    # Only the render mesh is hidden; the sibling GroundPlane collider stays intact.
    UsdGeom.Imageable(mesh).MakeInvisible()


def effective_camera_config(config, seed):
    from gr00t.rl.data_collection.camera_mount_randomization import randomize_camera_mounts

    # Reuse the door pipeline's float32 lens calibration and unsupported-skew handling.
    # Zero noise preserves the nominal onboard mounts; diagnostics never randomize.
    return randomize_camera_mounts(config, seed=seed, position_radius_m=0.0, rotation_angle_deg=0.0)


def diagnostic_pose(name, root, item):
    """Return camera position and world-convention quaternion (forward +X, up +Z)."""
    w, x, y, z = root[3:7]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    forward = torch.stack((yaw.cos(), yaw.sin(), yaw.new_zeros(())))
    up = root.new_tensor([0, 0, 1])
    if name == "robot_top_diagnostic":
        target = root[:3]
        eye = target + 1.8 * up
    elif name == "object_top_diagnostic":
        target = item[:3]
        eye = target + 1.25 * up
    elif name == "object_side_diagnostic":
        target = item[:3]
        toward_robot = root[:3] - target
        toward_robot = toward_robot.clone()
        toward_robot[2] = 0
        toward_robot = toward_robot / toward_robot.norm().clamp_min(1e-6)
        lateral = torch.cross(up, toward_robot, dim=0)
        eye = target + 0.65 * toward_robot + 0.45 * lateral + 0.22 * up
    elif name == "robot_rear_diagnostic":
        target = root[:3] + 0.4 * forward
        eye = root[:3] - 1.6 * forward + 0.8 * up
    else:
        raise ValueError(f"Unknown pick diagnostic camera: {name}")
    direction = target - eye
    horizontal = direction[:2].norm()
    heading = torch.where(horizontal > 1e-6, torch.atan2(direction[1], direction[0]), yaw)
    pitch = torch.atan2(-direction[2], horizontal)
    cy, sy = (heading / 2).cos(), (heading / 2).sin()
    cp, sp = (pitch / 2).cos(), (pitch / 2).sin()
    orientation = torch.stack((cy * cp, -sy * sp, cy * sp, sy * cp))
    return eye, orientation
