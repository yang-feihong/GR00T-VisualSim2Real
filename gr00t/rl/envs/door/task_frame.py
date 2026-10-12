import torch
import torch.nn.functional as F


def build_door_task_basis(frame_normal, frame_tangent, door_open_lr):
    """Build the mirrored door-task basis and its handedness.

    The task axes point through the door, toward the handle free end, and up.
    The right-hinged variant is a reflection, so axial vectors must include the
    returned determinant sign while polar vectors must not.
    """
    normal = F.normalize(frame_normal, dim=-1)
    tangent = F.normalize(
        frame_tangent
        - normal * torch.sum(frame_tangent * normal, dim=-1, keepdim=True),
        dim=-1,
    )
    up = F.normalize(torch.cross(normal, tangent, dim=-1), dim=-1)
    through = -normal
    free_end = tangent * door_open_lr.unsqueeze(-1)
    basis = torch.stack((through, free_end, up), dim=-1)
    handedness = torch.linalg.det(basis)
    return basis, handedness


def task_polar_to_world(vector, basis):
    """Transform a polar vector from mirrored task coordinates to world."""
    return torch.matmul(basis, vector.unsqueeze(-1)).squeeze(-1)


def world_polar_to_task(vector, basis):
    """Transform a world polar vector into mirrored task coordinates."""
    return torch.matmul(basis.transpose(-1, -2), vector.unsqueeze(-1)).squeeze(-1)


def task_axial_to_world(vector, basis, handedness):
    """Transform an axial vector, including the sign of the reflection."""
    return handedness.unsqueeze(-1) * task_polar_to_world(vector, basis)


def world_axial_to_task(vector, basis, handedness):
    """Transform a world axial vector into mirrored task coordinates."""
    return handedness.unsqueeze(-1) * world_polar_to_task(vector, basis)


def task_polar_to_frame(vector, task_basis, frame_basis):
    """Transform a task polar vector into another orthonormal frame."""
    world = task_polar_to_world(vector, task_basis)
    return world_polar_to_task(world, frame_basis)


def frame_polar_to_task(vector, frame_basis, task_basis):
    """Transform a polar vector from another orthonormal frame into task space."""
    world = task_polar_to_world(vector, frame_basis)
    return world_polar_to_task(world, task_basis)


def task_axial_to_frame(vector, task_basis, task_handedness, frame_basis):
    """Transform a task axial vector into a right-handed physical frame."""
    world = task_axial_to_world(vector, task_basis, task_handedness)
    return world_polar_to_task(world, frame_basis)


def frame_axial_to_task(vector, frame_basis, task_basis, task_handedness):
    """Transform an axial vector from a right-handed frame into task space."""
    world = task_polar_to_world(vector, frame_basis)
    return world_axial_to_task(world, task_basis, task_handedness)


def canonicalize_jaw_order(values, door_open_lr):
    """Order two physical jaw slots consistently across mirrored grasp poses."""
    if values.ndim < 2 or values.shape[1] != 2:
        raise ValueError("jaw values must have shape [N, 2, ...]")
    swap = door_open_lr < 0.0
    swap = swap.reshape(-1, *([1] * (values.ndim - 1)))
    return torch.where(swap, values.flip(1), values)


def canonicalize_jaw_axial(values, door_open_lr):
    """Express jaw contact positions along the handle's canonical free-end axis."""
    side = door_open_lr.reshape(-1, *([1] * (values.ndim - 1)))
    return canonicalize_jaw_order(values * side, door_open_lr)
