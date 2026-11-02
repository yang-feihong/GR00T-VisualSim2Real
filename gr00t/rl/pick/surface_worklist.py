"""Conservative BVH rejection of empty point-cloud blocks, entirely on the GPU."""

from collections import OrderedDict

import torch
import torch.nn.functional as functional
import warp as wp

SURFACE_BLOCK_SIZE = 256
_BOUNDS_CACHE = OrderedDict()


def _bounds(points, point_active, local_stride, point_mask):
    samples = local_stride or len(points)
    key = None
    if (
        not local_stride
        and samples <= 65536
        and not points.is_inference()
        and (point_mask is None or not point_mask.is_inference())
    ):
        mask_key = None if point_mask is None else (point_mask.data_ptr(), point_mask._version)
        key = (points.device, points.data_ptr(), samples, points._version, mask_key)
        if key in _BOUNDS_CACHE:
            _BOUNDS_CACHE.move_to_end(key)
            return _BOUNDS_CACHE[key][3:]
    rows = 1 if not local_stride else len(points) // local_stride
    blocks = (samples + SURFACE_BLOCK_SIZE - 1) // SURFACE_BLOCK_SIZE
    padding = blocks * SURFACE_BLOCK_SIZE - samples
    cloud = functional.pad(points.reshape(rows, samples, 3), (0, 0, 0, padding))
    valid = functional.pad(point_active.reshape(rows, samples).bool(), (0, padding))
    valid &= torch.isfinite(cloud).all(-1)
    cloud = cloud.reshape(rows, blocks, SURFACE_BLOCK_SIZE, 3)
    valid = valid.reshape(rows, blocks, SURFACE_BLOCK_SIZE, 1)
    low = cloud.masked_fill(~valid, torch.inf).amin(-2)
    high = cloud.masked_fill(~valid, -torch.inf).amax(-2)
    whole_low, whole_high = low.amin(1), high.amax(1)
    result = low.contiguous(), high.contiguous(), whole_low, whole_high
    if key is not None:
        _BOUNDS_CACHE[key] = (points, point_active, point_mask, *result)
        while len(_BOUNDS_CACHE) > 32:
            _BOUNDS_CACHE.popitem(last=False)
    return result


@wp.kernel
def _overlapping_blocks(
    meshes: wp.array(dtype=wp.uint64),
    offsets: wp.array(dtype=wp.vec3),
    frames: wp.array(dtype=wp.vec4),
    stride: int,
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.vec4),
    poses: wp.array(dtype=wp.int64),
    low: wp.array(dtype=wp.vec3),
    high: wp.array(dtype=wp.vec3),
    blocks: int,
    per_scene: int,
    padding: float,
    check_rays: int,
    flags: wp.array2d(dtype=wp.int32),
):
    work, block = wp.tid()
    pose = poses[work]
    row = int(pose) // stride
    index = row * blocks * per_scene + block
    lower, upper = wp.vec3(low[index]), wp.vec3(high[index])
    flags[work, block] = 0
    if lower[0] > upper[0] or lower[1] > upper[1] or lower[2] > upper[2]:
        return
    if check_rays != 0:
        lower = wp.vec3(wp.min(lower[0], 0.0), wp.min(lower[1], 0.0), wp.min(lower[2], 0.0))
        upper = wp.vec3(wp.max(upper[0], 0.0), wp.max(upper[1], 0.0), wp.max(upper[2], 0.0))
    frame = frames[row]
    rotation = wp.quat(frame[1], frame[2], frame[3], frame[0])
    origin = wp.quat_rotate_inv(rotation, positions[pose] - offsets[row])
    q = orientations[pose]
    vector = wp.vec3(q[1], q[2], q[3])
    box_low, box_high = wp.vec3(1.0e30), wp.vec3(-1.0e30)
    for corner in range(8):
        local = wp.vec3(
            lower[0] if (corner & 1) == 0 else upper[0],
            lower[1] if (corner & 2) == 0 else upper[1],
            lower[2] if (corner & 4) == 0 else upper[2],
        )
        cross = wp.cross(vector, local)
        rotated = local + 2.0 * (q[0] * cross + wp.cross(vector, cross))
        point = origin + wp.quat_rotate_inv(rotation, rotated)
        box_low = wp.vec3(
            wp.min(box_low[0], point[0]), wp.min(box_low[1], point[1]), wp.min(box_low[2], point[2])
        )
        box_high = wp.vec3(
            wp.max(box_high[0], point[0]),
            wp.max(box_high[1], point[1]),
            wp.max(box_high[2], point[2]),
        )
    extra = wp.vec3(padding)
    query = wp.mesh_query_aabb(meshes[row], box_low - extra, box_high + extra)
    face = int(0)
    if wp.mesh_query_aabb_next(query, face):
        flags[work, block] = 1


def surface_worklist(
    meshes,
    offsets,
    frames,
    stride,
    positions,
    orientations,
    points,
    local_stride,
    point_active,
    active,
    maximum,
    check_rays,
    *,
    point_mask=None,
):
    """Every omitted block is farther than the original capped query and its rays."""
    low, high, whole_low, whole_high = _bounds(points, point_active, local_stride, point_mask)
    blocks = low.shape[1]
    poses = torch.where(active != 0)[0]
    if meshes.dtype == torch.int32:
        return (poses[:, None] * blocks + torch.arange(blocks, device=poses.device)).reshape(
            -1
        ), blocks

    def query(poses, lower, upper, block_count):
        flags = torch.empty((len(poses), block_count), device=positions.device, dtype=torch.int32)
        if not len(poses):
            return flags
        wp.launch(
            _overlapping_blocks,
            dim=tuple(flags.shape),
            inputs=[
                wp.from_torch(meshes, dtype=wp.uint64),
                wp.from_torch(offsets, dtype=wp.vec3),
                wp.from_torch(frames, dtype=wp.vec4),
                stride,
                wp.from_torch(positions, dtype=wp.vec3),
                wp.from_torch(orientations, dtype=wp.vec4),
                wp.from_torch(poses, dtype=wp.int64),
                wp.from_torch(lower.reshape(-1, 3), dtype=wp.vec3),
                wp.from_torch(upper.reshape(-1, 3), dtype=wp.vec3),
                block_count,
                int(bool(local_stride)),
                maximum + 0.001,
                int(check_rays),
                wp.from_torch(flags, dtype=wp.int32),
            ],
            device=str(positions.device),
            stream=wp.stream_from_torch(torch.cuda.current_stream(positions.device)),
        )
        return flags

    if blocks > 1:
        broad = query(poses, whole_low, whole_high, 1)
        poses = poses[torch.where(broad[:, 0] != 0)[0]]
    flags = query(poses, low, high, blocks)
    owner, block = torch.where(flags != 0)
    return poses[owner] * blocks + block, blocks
