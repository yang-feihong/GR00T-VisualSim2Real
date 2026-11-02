"""Per-environment scene BVHs queried in one GPU launch, without row loops."""

import math

import torch
import warp as wp

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_normalize,
)


@wp.kernel
def _batched_ray(
    meshes: wp.array(dtype=wp.uint64),
    offsets: wp.array(dtype=wp.vec3),
    frames: wp.array(dtype=wp.vec4),
    stride: int,
    origins: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    hits: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
    label_offsets: wp.array(dtype=wp.int64),
    labels: wp.array(dtype=wp.int32),
    hit_types: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    row = i // stride
    frame = frames[row]
    rotation = wp.quat(frame[1], frame[2], frame[3], frame[0])
    origin = wp.quat_rotate_inv(rotation, origins[i] - offsets[row])
    world_delta = targets[i] - origins[i]
    delta = wp.quat_rotate_inv(rotation, world_delta)
    length = wp.length(delta)
    query = wp.mesh_query_ray(meshes[row], origin, wp.normalize(delta), length)
    hits[i] = targets[i]
    normals[i] = wp.vec3(0.0, 0.0, 0.0)
    distances[i] = length
    hit_types[i] = wp.int64(5)
    if query.result:
        hits[i] = origins[i] + query.t * wp.normalize(world_delta)
        normals[i] = wp.quat_rotate(rotation, query.normal)
        distances[i] = query.t
        if label_offsets[row] >= 0:
            hit_types[i] = wp.int64(labels[label_offsets[row] + wp.int64(query.face)])


@wp.kernel
def _batched_distance(
    meshes: wp.array(dtype=wp.uint64),
    offsets: wp.array(dtype=wp.vec3),
    frames: wp.array(dtype=wp.vec4),
    stride: int,
    points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
):
    i = wp.tid()
    row = i // stride
    frame = frames[row]
    rotation = wp.quat(frame[1], frame[2], frame[3], frame[0])
    point = wp.quat_rotate_inv(rotation, points[i] - offsets[row])
    nearest = wp.mesh_query_point(meshes[row], point, 2.0)
    distances[i] = 2.0
    if nearest.result:
        closest = wp.mesh_eval_position(meshes[row], nearest.face, nearest.u, nearest.v)
        distances[i] = wp.length(point - closest)


@wp.kernel
def _batched_surface(
    meshes: wp.array(dtype=wp.uint64),
    offsets: wp.array(dtype=wp.vec3),
    frames: wp.array(dtype=wp.vec4),
    stride: int,
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.vec4),
    local_points: wp.array(dtype=wp.vec3),
    local_stride: int,
    point_active: wp.array(dtype=wp.int32),
    tolerance: float,
    check_rays: int,
    signed: int,
    maximum: float,
    active: wp.array(dtype=wp.int32),
    pairs: wp.array(dtype=wp.int64),
    blocks: int,
    block_size: int,
    samples: int,
    minima: wp.array(dtype=float),
):
    work, offset = wp.tid()
    pair = pairs[work]
    pose = int(pair // wp.int64(blocks))
    sample = int(pair % wp.int64(blocks)) * block_size + offset
    if sample >= samples:
        return
    if active[pose] == 0:
        return
    if signed == 0 and minima[pose] < 0.0:
        return
    row = pose // stride
    local_index = row * local_stride + sample
    if point_active[local_index] == 0:
        return
    q = orientations[pose]
    vector = wp.vec3(q[1], q[2], q[3])
    local = local_points[local_index]
    cross = wp.cross(vector, local)
    rotated = local + 2.0 * (q[0] * cross + wp.cross(vector, cross))
    frame = frames[row]
    rotation = wp.quat(frame[1], frame[2], frame[3], frame[0])
    origin = wp.quat_rotate_inv(rotation, positions[pose] - offsets[row])
    rotated = wp.quat_rotate_inv(rotation, rotated)
    point = origin + rotated
    distance = maximum
    if signed != 0:
        nearest = wp.mesh_query_point(meshes[row], point, maximum)
    else:
        nearest = wp.mesh_query_point_no_sign(meshes[row], point, maximum)
    if nearest.result:
        closest = wp.mesh_eval_position(meshes[row], nearest.face, nearest.u, nearest.v)
        distance = wp.length(point - closest)
        if signed != 0:
            distance = distance * nearest.sign
    length = wp.length(rotated)
    if check_rays != 0 and length > 1e-8:
        ray = wp.mesh_query_ray(meshes[row], origin, rotated / length, length)
        if ray.result and ray.t < length - tolerance:
            distance = -1.0
    wp.atomic_min(minima, pose, distance)


class BatchedSceneGeometry:
    """Share scene BVH allocations; keep every query assigned to its own scene.

    Queries have leading environment dimension, followed by arbitrary candidate
    dimensions. Selection and all spatial computations remain on the GPU.
    """

    supports_query_mask = True

    def __new__(cls, geometries, *args, **kwargs):
        if cls is BatchedSceneGeometry and any(
            getattr(geometry, "uses_packet_bvh", False) for geometry in geometries
        ):
            from gr00t.rl.pick.packet_batch import BatchedPacketGeometry

            return object.__new__(BatchedPacketGeometry)
        return object.__new__(cls)

    @classmethod
    def from_views(cls, geometries):
        """Reuse existing room/distractor BVHs; never rebuild a room per batch row."""

        def components(geometry, offset):
            if hasattr(geometry, "components"):
                return [part for child in geometry.components for part in components(child, offset)]
            if hasattr(geometry, "source"):
                return components(geometry.source, offset + geometry.origin)
            return [(geometry, offset)]

        device = geometries[0].device
        parts = [components(geometry, torch.zeros(3, device=device)) for geometry in geometries]
        layers = []
        for index in range(max(map(len, parts))):
            # Repeating a row's room for an absent obstacle layer leaves min-distance unchanged.
            selected = [row[index] if index < len(row) else row[0] for row in parts]
            layers.append(
                cls(
                    [geometry for geometry, _ in selected],
                    torch.arange(len(parts), device=device),
                    torch.stack([offset for _, offset in selected]),
                )
            )
        return layers[0] if len(layers) == 1 else BatchedCompositeGeometry(layers)

    def __init__(self, geometries, scene_indices, origins, orientations=None):
        if not geometries or origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("Scene batch requires BVHs and (environments, 3) origins")
        self.geometries = tuple(geometries)
        self.device = origins.device
        ids = torch.tensor([g.mesh.id for g in geometries], device=self.device, dtype=torch.int64)
        self.mesh_ids = ids[scene_indices.long()].contiguous()
        label_parts, label_offsets, unique_offsets = [], [], {}
        label_count = 0
        for geometry in geometries:
            identity = geometry.mesh.id
            if identity not in unique_offsets:
                labels = getattr(geometry, "labels", None)
                offset = -1
                if labels is not None:
                    if labels.numel() != geometry.faces.numel() // 3:
                        raise ValueError("Scene surface labels must match the mesh faces")
                    labels = labels.to(device=self.device, dtype=torch.int32).flatten()
                    offset = label_count
                    label_parts.append(labels)
                    label_count += len(labels)
                unique_offsets[identity] = offset
            label_offsets.append(unique_offsets[identity])
        self.face_labels = (
            torch.cat(label_parts)
            if label_parts
            else torch.full((1,), 5, device=self.device, dtype=torch.int32)
        )
        self.label_offsets = torch.tensor(label_offsets, device=self.device, dtype=torch.long)[
            scene_indices.long()
        ].contiguous()
        self.origins = origins.float().contiguous()
        if orientations is None:
            self.orientations = self.origins.new_zeros(len(origins), 4)
            self.orientations[:, 0] = 1
        else:
            if orientations.shape != (len(origins), 4):
                raise ValueError("Rigid scene frames require one wxyz quaternion per environment")
            self.orientations = fixed_normalize(orientations, eps=1e-8).float().contiguous()
        if len(self.mesh_ids) != len(origins):
            raise ValueError("Every environment requires exactly one scene BVH")

    def select(self, rows):
        selected = object.__new__(type(self))
        selected.geometries, selected.device = self.geometries, self.device
        selected.mesh_ids = self.mesh_ids[rows].contiguous()
        selected.face_labels = self.face_labels
        selected.label_offsets = self.label_offsets[rows].contiguous()
        selected.origins = self.origins[rows].contiguous()
        selected.orientations = self.orientations[rows].contiguous()
        return selected

    def _inputs(self, shape):
        if not shape or shape[0] != len(self.mesh_ids):
            raise ValueError("Scene queries require a leading environment dimension")
        return [
            wp.from_torch(self.mesh_ids, dtype=wp.uint64),
            wp.from_torch(self.origins, dtype=wp.vec3),
            wp.from_torch(self.orientations, dtype=wp.vec4),
            math.prod(shape[1:]),
        ]

    def _launch(self, kernel, dim, inputs):
        wp.launch(
            kernel,
            dim=dim,
            inputs=inputs,
            device=str(self.device),
            stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)),
        )

    def rays(self, origins, targets):
        if origins.shape != targets.shape:
            raise ValueError("Ray origins and targets must have matching shapes")
        shape = origins.shape[:-1]
        points = origins.reshape(-1, 3).float().contiguous()
        targets = targets.reshape(-1, 3).float().contiguous()
        hits, normals = torch.empty_like(points), torch.empty_like(points)
        distances = torch.empty(len(points), device=self.device)
        hit_types = torch.empty(len(points), device=self.device, dtype=torch.long)
        inputs = self._inputs(shape)
        if not len(points):
            self.last_hit_types = hit_types.reshape(shape)
            return hits.reshape(*shape, 3), normals.reshape(*shape, 3), distances.reshape(shape)
        self._launch(
            _batched_ray,
            len(points),
            inputs
            + [
                wp.from_torch(points, dtype=wp.vec3),
                wp.from_torch(targets, dtype=wp.vec3),
                wp.from_torch(hits, dtype=wp.vec3),
                wp.from_torch(normals, dtype=wp.vec3),
                wp.from_torch(distances),
                wp.from_torch(self.label_offsets),
                wp.from_torch(self.face_labels),
                wp.from_torch(hit_types),
            ],
        )
        self.last_hit_types = hit_types.reshape(shape)
        return hits.reshape(*shape, 3), normals.reshape(*shape, 3), distances.reshape(shape)

    def visible(self, origins, targets, tolerance_m=0.01):
        return self.rays(origins, targets)[2] >= (targets - origins).norm(dim=-1) - tolerance_m

    def distances(self, points):
        shape = points.shape[:-1]
        packed = points.reshape(-1, 3).float().contiguous()
        output = torch.empty(len(packed), device=self.device)
        inputs = self._inputs(shape)
        if not len(packed):
            return output.reshape(shape)
        self._launch(
            _batched_distance,
            len(packed),
            inputs
            + [
                wp.from_torch(packed, dtype=wp.vec3),
                wp.from_torch(output),
            ],
        )
        return output.reshape(shape)

    def rigid_surface_clearance(
        self,
        positions,
        orientations,
        local_points,
        tolerance_m=0.0005,
        *,
        check_rays=True,
        signed_distance=False,
        maximum_distance_m=2.0,
        query_mask=None,
        local_point_mask=None,
    ):
        shape = positions.shape[:-1]
        packed = positions.reshape(-1, 3).float().contiguous()
        rotations = orientations.reshape(-1, 4).float().contiguous()
        if local_points.ndim not in (2, 3) or local_points.shape[-1] != 3:
            raise ValueError(
                "Local surface points require (points, 3) or (environments, points, 3)"
            )
        if local_points.ndim == 3 and local_points.shape[0] != len(self.mesh_ids):
            raise ValueError("Every environment requires its own local point cloud")
        if local_point_mask is not None and local_point_mask.shape != local_points.shape[:-1]:
            raise ValueError("Local surface point mask must match the point cloud")
        local_stride = 0 if local_points.ndim == 2 else local_points.shape[1]
        samples = local_points.shape[-2]
        local = local_points.reshape(-1, 3).float().contiguous()
        point_active = (
            torch.ones(len(local), device=self.device, dtype=torch.int32)
            if local_point_mask is None
            else local_point_mask.reshape(-1).to(dtype=torch.int32).contiguous()
        )
        if query_mask is not None and query_mask.shape != shape:
            raise ValueError("Surface query mask must match the pose dimensions")
        active = (
            torch.ones(len(packed), device=self.device, dtype=torch.int32)
            if query_mask is None
            else query_mask.reshape(-1).to(dtype=torch.int32).contiguous()
        )
        output = torch.full((len(packed),), maximum_distance_m, device=self.device)
        inputs = self._inputs(shape)
        if not len(packed) or not samples:
            return output.reshape(shape)
        from gr00t.rl.pick.surface_worklist import SURFACE_BLOCK_SIZE, surface_worklist

        pairs, blocks = surface_worklist(
            self.mesh_ids,
            self.origins,
            self.orientations,
            math.prod(shape[1:]),
            packed,
            rotations,
            local,
            local_stride,
            point_active,
            active,
            maximum_distance_m,
            check_rays,
            point_mask=local_point_mask,
        )
        if not len(pairs):
            return output.reshape(shape)
        self._launch(
            _batched_surface,
            (len(pairs), SURFACE_BLOCK_SIZE),
            inputs
            + [
                wp.from_torch(packed, dtype=wp.vec3),
                wp.from_torch(rotations, dtype=wp.vec4),
                wp.from_torch(local, dtype=wp.vec3),
                local_stride,
                wp.from_torch(point_active),
                tolerance_m,
                int(check_rays),
                int(signed_distance),
                maximum_distance_m,
                wp.from_torch(active),
                wp.from_torch(pairs, dtype=wp.int64),
                blocks,
                SURFACE_BLOCK_SIZE,
                samples,
                wp.from_torch(output),
            ],
        )
        return output.reshape(shape)


class BatchedCompositeGeometry:
    """A fixed set of batch-wide BVH layers, with independent selection per row."""

    supports_query_mask = True

    def __init__(self, components):
        self.components = tuple(components)
        self.device = self.components[0].device

    def select(self, rows):
        return type(self)([part.select(rows) for part in self.components])

    def rays(self, origins, targets):
        hit, normal, distance = self.components[0].rays(origins, targets)
        labels = self.components[0].last_hit_types
        for part in self.components[1:]:
            other_hit, other_normal, other_distance = part.rays(origins, targets)
            closer = other_distance < distance
            hit = torch.where(closer[..., None], other_hit, hit)
            normal = torch.where(closer[..., None], other_normal, normal)
            distance = torch.minimum(distance, other_distance)
            labels = torch.where(closer, part.last_hit_types, labels)
        self.last_hit_types = labels
        return hit, normal, distance

    def visible(self, origins, targets, tolerance_m=0.01):
        return self.rays(origins, targets)[2] >= (targets - origins).norm(dim=-1) - tolerance_m

    def distances(self, points):
        return torch.stack([part.distances(points) for part in self.components]).amin(0)

    def rigid_surface_clearance(self, positions, orientations, points, **kwargs):
        return torch.stack(
            [
                part.rigid_surface_clearance(positions, orientations, points, **kwargs)
                for part in self.components
            ]
        ).amin(0)
