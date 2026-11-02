"""Mixed native/packet implementation of the standard batched scene queries."""

import math

import torch
import warp as wp

from gr00t.rl.pick import batched_geometry as native
from gr00t.rl.pick.packet_geometry import packet_distance, packet_ray, packet_signed_distance


@wp.struct
class Scene:
    packet: int
    tree: wp.uint64
    points: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=int)
    priorities: wp.array(dtype=int)
    labels: wp.array(dtype=int)
    has_labels: int
    packet_size: int
    triangles: int
    upper: wp.array(dtype=wp.vec3)


@wp.func
def scene_distance(scene: Scene, point: wp.vec3, maximum: float, signed: int):
    result = maximum
    if scene.packet != 0:
        if signed != 0:
            result = packet_signed_distance(
                scene.tree,
                scene.points,
                scene.indices,
                scene.packet_size,
                scene.triangles,
                scene.priorities,
                point,
                maximum,
                scene.upper[0],
                scene.priorities,
            )
        else:
            result = packet_distance(
                scene.tree,
                scene.points,
                scene.indices,
                scene.packet_size,
                scene.triangles,
                point,
                maximum,
            )
    else:
        if signed != 0:
            query = wp.mesh_query_point(scene.tree, point, maximum)
        else:
            query = wp.mesh_query_point_no_sign(scene.tree, point, maximum)
        if query.result:
            closest = wp.mesh_eval_position(scene.tree, query.face, query.u, query.v)
            result = wp.length(point - closest)
            if signed != 0:
                result = result * query.sign
    return result


@wp.func
def scene_ray(
    scene: Scene,
    origin: wp.vec3,
    direction: wp.vec3,
    length: float,
    face_output: wp.array(dtype=int),
    output_index: int,
):
    result = wp.vec4(length, 0.0, 0.0, 0.0)
    if scene.packet != 0:
        result = packet_ray(
            scene.tree,
            scene.points,
            scene.indices,
            scene.packet_size,
            scene.triangles,
            scene.priorities,
            origin,
            direction,
            length,
            face_output,
            output_index,
        )
    else:
        query = wp.mesh_query_ray(scene.tree, origin, direction, length)
        if output_index >= 0:
            face_output[output_index] = -1
        if query.result:
            if output_index >= 0:
                face_output[output_index] = query.face
            result = wp.vec4(query.t, query.normal[0], query.normal[1], query.normal[2])
    return result


@wp.kernel
def batched_ray(
    ids: wp.array(dtype=int),
    scenes: wp.array(dtype=Scene),
    offsets: wp.array(dtype=wp.vec3),
    frames: wp.array(dtype=wp.vec4),
    stride: int,
    origins: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    hits: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
    unused_offsets: wp.array(dtype=wp.int64),
    faces: wp.array(dtype=int),
    hit_types: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    row = i // stride
    scene = scenes[ids[row]]
    frame = frames[row]
    rotation = wp.quat(frame[1], frame[2], frame[3], frame[0])
    origin = wp.quat_rotate_inv(rotation, origins[i] - offsets[row])
    world_delta = targets[i] - origins[i]
    delta = wp.quat_rotate_inv(rotation, world_delta)
    length = wp.length(delta)
    ray = scene_ray(scene, origin, wp.normalize(delta), length, faces, i)
    distances[i] = ray[0]
    hits[i] = origins[i] + ray[0] * wp.normalize(world_delta)
    normals[i] = wp.quat_rotate(rotation, wp.vec3(ray[1], ray[2], ray[3]))
    hit_types[i] = wp.int64(5)
    if faces[i] < 0:
        hits[i] = targets[i]
    elif scene.has_labels != 0:
        hit_types[i] = wp.int64(scene.labels[faces[i]])


@wp.kernel
def batched_distance(
    ids: wp.array(dtype=int),
    scenes: wp.array(dtype=Scene),
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
    local = wp.quat_rotate_inv(rotation, points[i] - offsets[row])
    distances[i] = scene_distance(scenes[ids[row]], local, 2.0, 0)


@wp.kernel
def batched_surface(
    ids: wp.array(dtype=int),
    scenes: wp.array(dtype=Scene),
    offsets: wp.array(dtype=wp.vec3),
    frames: wp.array(dtype=wp.vec4),
    stride: int,
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.vec4),
    local_points: wp.array(dtype=wp.vec3),
    local_stride: int,
    point_active: wp.array(dtype=int),
    tolerance: float,
    check_rays: int,
    signed: int,
    maximum: float,
    active: wp.array(dtype=int),
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
    if active[pose] == 0 or (signed == 0 and minima[pose] < 0.0):
        return
    row = pose // stride
    local_index = row * local_stride + sample
    if point_active[local_index] == 0:
        return
    scene = scenes[ids[row]]
    q = orientations[pose]
    vector = wp.vec3(q[1], q[2], q[3])
    local = local_points[local_index]
    cross = wp.cross(vector, local)
    rotated = local + 2.0 * (q[0] * cross + wp.cross(vector, cross))
    frame = frames[row]
    rotation = wp.quat(frame[1], frame[2], frame[3], frame[0])
    origin = wp.quat_rotate_inv(rotation, positions[pose] - offsets[row])
    delta = wp.quat_rotate_inv(rotation, rotated)
    distance = scene_distance(scene, origin + delta, maximum, signed)
    length = wp.length(delta)
    if check_rays != 0 and length > 1.0e-8:
        hit = scene_ray(scene, origin, delta / length, length, scene.priorities, -1)
        if hit[0] < length - tolerance:
            distance = -1.0
    wp.atomic_min(minima, pose, distance)


class BatchedPacketGeometry(native.BatchedSceneGeometry):
    def __init__(self, geometries, scene_indices, origins, orientations=None):
        # Keep public query, mask, shape, frame-normalization and allocation semantics.
        # Native initialization is deliberately not used: it duplicates all face labels.
        if not geometries or origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("Scene batch requires BVHs and (environments, 3) origins")
        self.geometries = tuple(geometries)
        self.device = origins.device
        self.scene_indices = torch.arange(len(geometries), device=self.device, dtype=torch.int32)[
            scene_indices.long()
        ].contiguous()
        if len(self.scene_indices) != len(origins):
            raise ValueError("Every environment requires exactly one scene BVH")
        self.mesh_ids = self.scene_indices
        self.origins = origins.float().contiguous()
        if orientations is None:
            self.orientations = self.origins.new_zeros(len(origins), 4)
            self.orientations[:, 0] = 1
        else:
            if orientations.shape != (len(origins), 4):
                raise ValueError("Rigid scene frames require one wxyz quaternion per environment")
            self.orientations = native.fixed_normalize(orientations, eps=1e-8).float().contiguous()
        self.label_offsets = torch.zeros(len(origins), device=self.device, dtype=torch.long)
        self.face_labels = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.dummy_indices = self.face_labels
        self.label_storage, descriptors = [], []
        label_cache = {}
        dummy_points = torch.zeros(1, 3, device=self.device)
        self.dummy_points = dummy_points
        for geometry in geometries:
            descriptor = Scene()
            descriptor.packet = int(hasattr(geometry, "bvh"))
            descriptor.tree = geometry.bvh.id if descriptor.packet else geometry.mesh.id
            descriptor.points = wp.from_torch(geometry.points, dtype=wp.vec3)
            descriptor.indices = wp.from_torch(geometry.faces, dtype=wp.int32)
            descriptor.priorities = wp.from_torch(
                geometry.priorities if descriptor.packet else self.face_labels
            )
            labels = getattr(geometry, "labels", None)
            if labels is not None and labels.numel() != geometry.faces.numel() // 3:
                raise ValueError("Scene surface labels must match the mesh faces")
            descriptor.has_labels = int(labels is not None)
            key = id(geometry)
            if key not in label_cache:
                label_cache[key] = (
                    labels.to(dtype=torch.int32).contiguous()
                    if labels is not None
                    else self.face_labels
                )
            label = label_cache[key]
            self.label_storage.append(label)
            descriptor.labels = wp.from_torch(label)
            descriptor.packet_size = geometry.packet_size if descriptor.packet else 0
            descriptor.triangles = geometry.faces.numel() // 3
            descriptor.upper = wp.from_torch(
                geometry.scene_upper if descriptor.packet else dummy_points, dtype=wp.vec3
            )
            descriptors.append(descriptor)
        self.descriptors = wp.array(descriptors, dtype=Scene, device=str(self.device))

    def select(self, rows):
        selected = object.__new__(type(self))
        selected.__dict__.update(self.__dict__)
        selected.scene_indices = self.scene_indices[rows].contiguous()
        selected.mesh_ids = selected.scene_indices
        selected.origins = self.origins[rows].contiguous()
        selected.orientations = self.orientations[rows].contiguous()
        selected.label_offsets = self.label_offsets[rows].contiguous()
        return selected

    def _inputs(self, shape):
        if not shape or shape[0] != len(self.mesh_ids) or not len(self.mesh_ids):
            raise ValueError("Scene queries require a leading environment dimension")
        return [
            wp.from_torch(self.scene_indices),
            self.descriptors,
            wp.from_torch(self.origins, dtype=wp.vec3),
            wp.from_torch(self.orientations, dtype=wp.vec4),
            math.prod(shape[1:]),
        ]

    def _launch(self, kernel, dim, inputs):
        kernels = {
            native._batched_ray: batched_ray,
            native._batched_distance: batched_distance,
            native._batched_surface: batched_surface,
        }
        if kernel is native._batched_ray:
            # Native API passes the concatenated label array in this slot.
            self.face_labels = torch.empty(dim, device=self.device, dtype=torch.int32)
            inputs[-2] = wp.from_torch(self.face_labels)
        super()._launch(kernels[kernel], dim, inputs)
