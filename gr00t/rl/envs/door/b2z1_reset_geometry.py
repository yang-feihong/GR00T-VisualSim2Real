"""B2Z1 and closed-door geometry used to validate randomized reset states."""

from __future__ import annotations

import math

import torch


_ARM_JOINT_ORIGINS = (
    (0.0, 0.0, 0.0585),
    (0.0, 0.0, 0.0450),
    (-0.3500, 0.0, 0.0),
    (0.2180, 0.0, 0.0570),
    (0.0700, 0.0, 0.0),
    (0.0492, 0.0, 0.0),
)
_ARM_JOINT_AXES = (2, 1, 1, 1, 2, 0)

# The union of the refined fixed and moving gripper collision hulls at the
# closed reset position, expressed in gripper_link coordinates. These bounds
# are derived from the OBJ vertices referenced by the production URDF.
_GRIPPER_COLLISION_MIN = (-0.05635747, -0.04135744, -0.03677895)
_GRIPPER_COLLISION_MAX = (0.10042901, 0.04135744, 0.03285408)


def quaternion_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized-or-not xyzw quaternions to rotation matrices."""
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp(min=1.0e-12)
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _axis_rotation(
    angle: torch.Tensor, axis: int, dtype: torch.dtype
) -> torch.Tensor:
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    if axis == 0:
        values = (one, zero, zero, zero, cosine, -sine, zero, sine, cosine)
    elif axis == 1:
        values = (cosine, zero, sine, zero, one, zero, -sine, zero, cosine)
    elif axis == 2:
        values = (cosine, -sine, zero, sine, cosine, zero, zero, zero, one)
    else:
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    return torch.stack(values, dim=-1).reshape(angle.shape + (3, 3)).to(dtype=dtype)


def b2z1_gripper_pose_and_jacobian(
    root_position: torch.Tensor,
    root_quaternion_xyzw: torch.Tensor,
    arm_joint_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gripper-link pose and geometric arm Jacobian in world axes."""
    if root_position.shape[-1] != 3 or root_quaternion_xyzw.shape[-1] != 4:
        raise ValueError("root position/quaternion must end in 3/4 values")
    if arm_joint_position.shape[-1] != 6:
        raise ValueError("arm_joint_position must contain joint1 through joint6")
    if root_position.shape[:-1] != arm_joint_position.shape[:-1]:
        raise ValueError("root and arm batches must match")

    rotation = quaternion_xyzw_to_matrix(root_quaternion_xyzw)
    position = root_position
    dtype = root_position.dtype
    device = root_position.device

    mount_offset = torch.tensor((0.22, 0.0, 0.17), dtype=dtype, device=device)
    position = position + torch.einsum("...ij,j->...i", rotation, mount_offset)
    mount_yaw = torch.full(
        arm_joint_position.shape[:-1], math.pi / 2.0, dtype=dtype, device=device
    )
    rotation = rotation @ _axis_rotation(mount_yaw, 2, dtype)

    joint_origins = []
    joint_axes = []

    for joint_index, (origin, axis) in enumerate(
        zip(_ARM_JOINT_ORIGINS, _ARM_JOINT_AXES)
    ):
        offset = torch.tensor(origin, dtype=dtype, device=device)
        position = position + torch.einsum("...ij,j->...i", rotation, offset)
        joint_origins.append(position)
        joint_axes.append(rotation[..., :, axis])
        rotation = rotation @ _axis_rotation(
            arm_joint_position[..., joint_index], axis, dtype
        )

    stator_offset = torch.tensor((0.051, 0.0, 0.0), dtype=dtype, device=device)
    position = position + torch.einsum("...ij,j->...i", rotation, stator_offset)
    stator_roll = torch.full(
        arm_joint_position.shape[:-1], -math.pi / 2.0, dtype=dtype, device=device
    )
    rotation = rotation @ _axis_rotation(stator_roll, 0, dtype)
    ee_offset = torch.tensor((0.049, 0.0, 0.0), dtype=dtype, device=device)
    position = position + torch.einsum("...ij,j->...i", rotation, ee_offset)

    origins = torch.stack(joint_origins, dim=-2)
    axes = torch.stack(joint_axes, dim=-1)
    linear_jacobian = torch.cross(
        axes, position.unsqueeze(-1) - origins.transpose(-1, -2), dim=-2
    )
    jacobian = torch.cat((linear_jacobian, axes), dim=-2)
    return position, rotation, jacobian


def solve_b2z1_arm_ik(
    root_position: torch.Tensor,
    root_quaternion_xyzw: torch.Tensor,
    target_position: torch.Tensor,
    target_rotation: torch.Tensor,
    initial_joint_position: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    *,
    iterations: int = 80,
    damping: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve batched six-DoF gripper poses with damped least-squares IK."""
    joint_position = initial_joint_position.clone()
    identity = torch.eye(
        6, dtype=joint_position.dtype, device=joint_position.device
    ).expand(joint_position.shape[0], -1, -1)
    for _ in range(iterations):
        position, rotation, jacobian = b2z1_gripper_pose_and_jacobian(
            root_position, root_quaternion_xyzw, joint_position
        )
        position_error = target_position - position
        orientation_error = 0.5 * (
            torch.cross(rotation[..., :, 0], target_rotation[..., :, 0], dim=-1)
            + torch.cross(rotation[..., :, 1], target_rotation[..., :, 1], dim=-1)
            + torch.cross(rotation[..., :, 2], target_rotation[..., :, 2], dim=-1)
        )
        error = torch.cat((position_error, orientation_error), dim=-1)
        delta = torch.linalg.solve(
            jacobian.transpose(-1, -2) @ jacobian
            + (damping * damping) * identity,
            (jacobian.transpose(-1, -2) @ error.unsqueeze(-1)),
        ).squeeze(-1)
        joint_position += delta.clamp(-0.15, 0.15)
        joint_position.clamp_(joint_lower, joint_upper)

    position, rotation, _ = b2z1_gripper_pose_and_jacobian(
        root_position, root_quaternion_xyzw, joint_position
    )
    position_error = torch.linalg.vector_norm(target_position - position, dim=-1)
    trace = torch.einsum("...ij,...ij->...", target_rotation, rotation)
    orientation_error = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    return joint_position, position_error, orientation_error


def b2z1_gripper_collision_obb(
    root_position: torch.Tensor,
    root_quaternion_xyzw: torch.Tensor,
    arm_joint_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the closed-gripper collision OBB for candidate reset states."""
    position, rotation, _ = b2z1_gripper_pose_and_jacobian(
        root_position, root_quaternion_xyzw, arm_joint_position
    )
    dtype = root_position.dtype
    device = root_position.device

    collision_min = torch.tensor(
        _GRIPPER_COLLISION_MIN, dtype=dtype, device=device
    )
    collision_max = torch.tensor(
        _GRIPPER_COLLISION_MAX, dtype=dtype, device=device
    )
    local_center = 0.5 * (collision_min + collision_max)
    half_extent = 0.5 * (collision_max - collision_min)
    center = position + torch.einsum("...ij,j->...i", rotation, local_center)
    return center, rotation, half_extent.expand(center.shape)


def oriented_boxes_overlap(
    center_a: torch.Tensor,
    axes_a: torch.Tensor,
    half_extent_a: torch.Tensor,
    center_b: torch.Tensor,
    axes_b: torch.Tensor,
    half_extent_b: torch.Tensor,
    *,
    separation_margin: float = 0.0,
) -> torch.Tensor:
    """Batched 3-D OBB overlap using all 15 separating axes."""
    if separation_margin < 0.0:
        raise ValueError("separation_margin must be non-negative")
    relative_rotation = axes_a.transpose(-1, -2) @ axes_b
    absolute_rotation = torch.abs(relative_rotation) + 1.0e-7
    translation = torch.einsum(
        "...ij,...j->...i", axes_a.transpose(-1, -2), center_b - center_a
    )
    margin = 0.5 * float(separation_margin)
    extent_a = half_extent_a + margin
    extent_b = half_extent_b + margin

    separated_a = torch.abs(translation) > (
        extent_a
        + torch.einsum("...ij,...j->...i", absolute_rotation, extent_b)
    )
    translation_b = torch.einsum("...ji,...j->...i", relative_rotation, translation)
    separated_b = torch.abs(translation_b) > (
        extent_b
        + torch.einsum(
            "...ji,...j->...i", absolute_rotation, extent_a
        )
    )

    cross_separated = torch.zeros_like(separated_a[..., 0])
    for axis_a in range(3):
        next_a = (axis_a + 1) % 3
        last_a = (axis_a + 2) % 3
        for axis_b in range(3):
            next_b = (axis_b + 1) % 3
            last_b = (axis_b + 2) % 3
            projected_translation = torch.abs(
                translation[..., last_a] * relative_rotation[..., next_a, axis_b]
                - translation[..., next_a] * relative_rotation[..., last_a, axis_b]
            )
            radius_a = (
                extent_a[..., next_a] * absolute_rotation[..., last_a, axis_b]
                + extent_a[..., last_a] * absolute_rotation[..., next_a, axis_b]
            )
            radius_b = (
                extent_b[..., next_b] * absolute_rotation[..., axis_a, last_b]
                + extent_b[..., last_b] * absolute_rotation[..., axis_a, next_b]
            )
            cross_separated |= projected_translation > (radius_a + radius_b)

    separated = separated_a.any(dim=-1) | separated_b.any(dim=-1) | cross_separated
    return ~separated


def closed_door_obstacle_obbs(
    door_root_position: torch.Tensor,
    door_root_quaternion_wxyz: torch.Tensor,
    door_width: torch.Tensor,
    door_height: torch.Tensor,
    door_handle_height: torch.Tensor,
    door_handle_width: torch.Tensor,
    door_open_lr: torch.Tensor,
    door_axle_length: torch.Tensor,
    door_handle_length: torch.Tensor,
    door_handle_radius: torch.Tensor,
    total_wall_height: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return panel, handle, latch, and frame OBBs for a closed door."""
    door_axes = quaternion_xyzw_to_matrix(
        door_root_quaternion_wxyz[..., (1, 2, 3, 0)]
    )
    dtype = door_root_position.dtype
    device = door_root_position.device
    batch_shape = door_root_position.shape[:-1]
    zero = torch.zeros(batch_shape, dtype=dtype, device=device)

    def world_center(local_center: torch.Tensor) -> torch.Tensor:
        return door_root_position.unsqueeze(-2) + torch.matmul(
            local_center, door_axes.transpose(-1, -2)
        )

    half_width = 0.5 * door_width
    panel_center_local = torch.stack((zero, zero, 0.5 * door_height), dim=-1)
    panel_half = torch.stack(
        (torch.full_like(zero, 0.02), half_width - 0.002, 0.5 * door_height - 0.002),
        dim=-1,
    )

    handle_axis_y = (half_width - door_handle_width) * door_open_lr
    handle_center_y = handle_axis_y - 0.5 * door_handle_length * door_open_lr
    handle_center_local = torch.stack(
        (zero, handle_center_y, door_handle_height), dim=-1
    )
    handle_half = torch.stack(
        (
            0.5 * door_axle_length + door_handle_radius,
            0.5 * door_handle_length + door_handle_radius,
            door_handle_radius,
        ),
        dim=-1,
    )

    latch_center_local = torch.stack(
        (
            torch.full_like(zero, 0.043),
            (half_width - 0.0028) * door_open_lr,
            door_handle_height,
        ),
        dim=-1,
    )
    latch_half = torch.full_like(latch_center_local, 0.026)

    total_half_width = torch.full_like(half_width, 10.0)
    side_half_width = 0.5 * (total_half_width - half_width)
    side_center_offset = half_width + side_half_width
    side_half = torch.stack(
        (
            torch.full_like(zero, 0.05),
            side_half_width,
            0.5 * total_wall_height,
        ),
        dim=-1,
    )
    left_center_local = torch.stack(
        (
            torch.full_like(zero, -0.02),
            side_center_offset,
            0.5 * total_wall_height,
        ),
        dim=-1,
    )
    right_center_local = left_center_local.clone()
    right_center_local[..., 1] *= -1.0

    top_center_local = torch.stack(
        (
            torch.full_like(zero, -0.02),
            zero,
            door_height + 0.5 * (total_wall_height - door_height),
        ),
        dim=-1,
    )
    top_half = torch.stack(
        (
            torch.full_like(zero, 0.05),
            half_width,
            0.5 * (total_wall_height - door_height),
        ),
        dim=-1,
    )

    local_centers = torch.stack(
        (
            panel_center_local,
            handle_center_local,
            latch_center_local,
            left_center_local,
            right_center_local,
            top_center_local,
        ),
        dim=-2,
    )
    half_extents = torch.stack(
        (panel_half, handle_half, latch_half, side_half, side_half, top_half), dim=-2
    )
    centers = world_center(local_centers)
    axes = door_axes.unsqueeze(-3).expand(batch_shape + (6, 3, 3))
    return centers, axes, half_extents


def initial_gripper_door_collision(
    root_position: torch.Tensor,
    root_quaternion_xyzw: torch.Tensor,
    arm_joint_position: torch.Tensor,
    door_root_position: torch.Tensor,
    door_root_quaternion_wxyz: torch.Tensor,
    door_width: torch.Tensor,
    door_height: torch.Tensor,
    door_handle_height: torch.Tensor,
    door_handle_width: torch.Tensor,
    door_open_lr: torch.Tensor,
    door_axle_length: torch.Tensor,
    door_handle_length: torch.Tensor,
    door_handle_radius: torch.Tensor,
    total_wall_height: torch.Tensor,
    *,
    separation_margin: float,
) -> torch.Tensor:
    """Return whether the closed gripper overlaps any closed-door collider proxy."""
    gripper_center, gripper_axes, gripper_half = b2z1_gripper_collision_obb(
        root_position, root_quaternion_xyzw, arm_joint_position
    )
    obstacle_center, obstacle_axes, obstacle_half = closed_door_obstacle_obbs(
        door_root_position,
        door_root_quaternion_wxyz,
        door_width,
        door_height,
        door_handle_height,
        door_handle_width,
        door_open_lr,
        door_axle_length,
        door_handle_length,
        door_handle_radius,
        total_wall_height,
    )
    return oriented_boxes_overlap(
        gripper_center.unsqueeze(-2),
        gripper_axes.unsqueeze(-3),
        gripper_half.unsqueeze(-2),
        obstacle_center,
        obstacle_axes,
        obstacle_half,
        separation_margin=separation_margin,
    ).any(dim=-1)
