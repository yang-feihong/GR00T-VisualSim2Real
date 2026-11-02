"""Static USD preprocessing and GPU BVH queries for grasp randomization/planning."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import tempfile
import zipfile
import zlib
from pathlib import Path

import numpy as np
import torch
import warp as wp


@wp.kernel
def _ray_kernel(
    mesh: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
    faces: wp.array(dtype=int),
):
    i = wp.tid()
    delta = targets[i] - origins[i]
    length = wp.length(delta)
    query = wp.mesh_query_ray(mesh, origins[i], wp.normalize(delta), length)
    distances[i] = length
    points[i] = targets[i]
    normals[i] = wp.vec3(0.0, 0.0, 0.0)
    faces[i] = -1
    if query.result:
        points[i] = origins[i] + query.t * wp.normalize(delta)
        normals[i] = query.normal
        distances[i] = query.t
        faces[i] = query.face


@wp.kernel
def _distance_kernel(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
):
    i = wp.tid()
    query = wp.mesh_query_point(mesh, points[i], 2.0)
    distances[i] = 2.0
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        distances[i] = wp.length(points[i] - closest)


SURFACE_NAMES = ("floor", "table", "chair", "sofa", "bed", "other_furniture")


@wp.kernel
def _rigid_surface_clearance_kernel(
    mesh: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.vec4),
    local_points: wp.array(dtype=wp.vec3),
    ray_tolerance: float,
    check_rays: int,
    signed_distance: int,
    maximum_distance: float,
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
    if active[pose] == 0:
        return
    if signed_distance == 0 and minima[pose] < 0.0:
        return
    q = orientations[pose]
    vector = wp.vec3(q[1], q[2], q[3])
    local = local_points[sample]
    cross = wp.cross(vector, local)
    rotated = local + 2.0 * (q[0] * cross + wp.cross(vector, cross))
    world = rotated + positions[pose]
    distance = maximum_distance
    if signed_distance != 0:
        nearest = wp.mesh_query_point(mesh, world, maximum_distance)
    else:
        nearest = wp.mesh_query_point_no_sign(mesh, world, maximum_distance)
    if nearest.result:
        closest = wp.mesh_eval_position(mesh, nearest.face, nearest.u, nearest.v)
        distance = wp.length(world - closest)
        if signed_distance != 0:
            distance = distance * nearest.sign
    delta = world - positions[pose]
    length = wp.length(delta)
    if check_rays != 0 and length > 1e-8:
        ray = wp.mesh_query_ray(mesh, positions[pose], delta / length, length)
        if ray.result and ray.t < length - ray_tolerance:
            distance = -1.0
    if distance < minima[pose]:
        wp.atomic_min(minima, pose, distance)


def surface_type(path):
    import re

    path = str(path).lower()
    for index, names in enumerate(
        (
            ("floor",),
            ("table", "desk"),
            ("chair", "stool"),
            ("sofa", "couch"),
            ("bed",),
        )
    ):
        if any(re.search(rf"/{name}(?:_\d+)?(?:/|$)", path) for name in names):
            return index
    return 5


def _read_scene_cache(cache):
    try:
        with np.load(cache) as data:
            result = data["points"], data["faces"], data["labels"]
            if (
                result[0].ndim != 2
                or result[0].shape[1] != 3
                or result[1].ndim != 2
                or result[1].shape[1] != 3
                or result[2].shape != (len(result[1]),)
            ):
                raise ValueError("Invalid scene cache array shapes")
            return result
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile, zlib.error):
        logging.getLogger(__name__).warning("Invalid geometry cache %s; rebuilding", cache)
        return None


def _write_scene_cache(cache, points, faces, labels):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache.parent, prefix=f".{cache.name}.", suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            # Deflate is CPU-bound; preserve the exact arrays without compression.
            np.savez(output, points=points, faces=faces, labels=labels)
        os.replace(temporary, cache)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_scene_triangles(usd_path: Path, cache_root: Path, *, return_labels=False):
    """Publish complete caches once across ranks; runtime queries stay on GPU."""

    identity = f"{usd_path}:{usd_path.stat().st_mtime_ns}:collision-v4-labels"
    cache = cache_root / (hashlib.sha256(identity.encode()).hexdigest()[:20] + ".npz")
    result = _read_scene_cache(cache)
    if result is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = _read_scene_cache(cache)
            if result is None:
                result = _flatten_scene_triangles(usd_path)
                _write_scene_cache(cache, *result)
    return result if return_labels else result[:2]


def _flatten_scene_triangles(usd_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    transforms = UsdGeom.XformCache()
    vertices, triangles, labels = [], [], []
    offset = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        collider = prim
        while collider and not collider.HasAPI(UsdPhysics.CollisionAPI):
            collider = collider.GetParent()
        if not collider:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        if not len(points):
            continue
        matrix = np.asarray(transforms.GetLocalToWorldTransform(prim))
        points = points @ matrix[:3, :3] + matrix[3, :3]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        starts = np.cumsum(counts) - counts
        faces = []
        for size in np.unique(counts):
            selected = starts[counts == size]
            for corner in range(1, int(size) - 1):
                faces.append(indices[selected[:, None] + np.array([0, corner, corner + 1])])
        if faces:
            vertices.append(points.astype(np.float32))
            part_faces = np.concatenate(faces) + offset
            triangles.append(part_faces)
            labels.append(np.full(len(part_faces), surface_type(prim.GetPath()), dtype=np.int8))
            offset += len(points)
    if not vertices:
        raise RuntimeError(f"Scene has no static collision triangles: {usd_path}")
    points, faces = np.concatenate(vertices), np.concatenate(triangles)
    labels = np.concatenate(labels)
    return points, faces, labels


def read_support_regions(usd_path):
    """Furniture AABBs only propose XY candidates; ray hits decide actual support."""
    import re

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    boxes = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    regions = []
    for prim in stage.Traverse():
        if not re.fullmatch(r"(?:table|desk|chair|stool|sofa|couch|bed)_\d+", prim.GetName()):
            continue
        bounds = boxes.ComputeWorldBound(prim).ComputeAlignedRange()
        lower, upper = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        if (upper[:2] - lower[:2]).min() > 0.12:
            regions.append(np.r_[lower[:2], upper[:2]])
    return np.array(regions, dtype=np.float32).reshape(-1, 4)


def static_scene_geometry(
    points, faces, device="cuda:0", *, labels=None, backend="native", ground_plane_z_m=None
):
    """Keep static-room storage independent of the shared query/control interface."""
    if ground_plane_z_m is not None:
        points = torch.as_tensor(points, device=device, dtype=torch.float32)
        faces = torch.as_tensor(faces, device=device, dtype=torch.int32).reshape(-1, 3)
        bounds = torch.stack((points[:, :2].amin(0) - 1.0, points[:, :2].amax(0) + 1.0))
        # The simulator always installs /World/ground at z=0, even when a
        # room's original floor is lower. Keep every original collider intact.
        quad = torch.stack(
            (
                bounds[[0, 1, 1, 0], 0],
                bounds[[0, 0, 1, 1], 1],
                points.new_full((4,), ground_plane_z_m),
            ),
            -1,
        )
        extra = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device, dtype=torch.int32) + len(points)
        points = torch.cat((points, quad))
        faces = torch.cat((faces, extra))
        if labels is not None:
            labels = torch.as_tensor(labels, device=device)
            labels = torch.cat((labels, labels.new_zeros(2)))
    if backend == "native":
        return SceneGeometry(points, faces, device, labels=labels)
    if backend == "packet":
        from gr00t.rl.pick.packet_geometry import PacketGeometry

        return PacketGeometry(points, faces, device, packet_size=64, labels=labels)
    raise ValueError(f"Unknown static scene geometry backend: {backend}")


class SceneGeometry:
    supports_query_mask = True

    def __init__(self, points, faces, device="cuda:0", labels=None):
        wp.init()
        self.device = torch.device(device)
        self.points = torch.as_tensor(points, device=self.device, dtype=torch.float32)
        self.faces = torch.as_tensor(faces, device=self.device, dtype=torch.int32).reshape(-1)
        self.labels = None if labels is None else torch.as_tensor(labels, device=self.device)
        self.mesh = wp.Mesh(
            points=wp.from_torch(self.points, dtype=wp.vec3),
            indices=wp.from_torch(self.faces, dtype=wp.int32),
        )

    def rays(self, origins, targets):
        shape = origins.shape[:-1]
        origins = origins.reshape(-1, 3).float().contiguous()
        targets = targets.reshape(-1, 3).float().contiguous()
        points, normals = torch.empty_like(origins), torch.empty_like(origins)
        distances = torch.empty(len(origins), device=origins.device)
        faces = torch.empty(len(origins), device=origins.device, dtype=torch.int32)
        wp.launch(
            _ray_kernel,
            dim=len(origins),
            inputs=[
                self.mesh.id,
                wp.from_torch(origins, dtype=wp.vec3),
                wp.from_torch(targets, dtype=wp.vec3),
                wp.from_torch(points, dtype=wp.vec3),
                wp.from_torch(normals, dtype=wp.vec3),
                wp.from_torch(distances),
                wp.from_torch(faces),
            ],
            device=str(self.device),
            stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)),
        )
        self.last_hit_types = torch.full_like(faces, 5)
        if self.labels is not None:
            self.last_hit_types = torch.where(faces >= 0, self.labels[faces.clamp_min(0)].long(), 5)
        self.last_hit_types = self.last_hit_types.reshape(shape)
        return points.reshape(*shape, 3), normals.reshape(*shape, 3), distances.reshape(shape)

    def visible(self, origins, targets, tolerance_m=0.01):
        _, _, hit = self.rays(origins, targets)
        return hit >= torch.linalg.vector_norm(targets - origins, dim=-1) - tolerance_m

    def distances(self, points):
        shape = points.shape[:-1]
        points = points.reshape(-1, 3).float().contiguous()
        distances = torch.empty(len(points), device=points.device)
        wp.launch(
            _distance_kernel,
            dim=len(points),
            inputs=[self.mesh.id, wp.from_torch(points, dtype=wp.vec3), wp.from_torch(distances)],
            device=str(self.device),
            stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)),
        )
        return distances.reshape(shape)

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
    ):
        shape = positions.shape[:-1]
        if query_mask is not None and query_mask.shape != shape:
            raise ValueError("Surface query mask must match the pose dimensions")
        positions = positions.reshape(-1, 3).float().contiguous()
        orientations = orientations.reshape(-1, 4).float().contiguous()
        local_points = local_points.float().contiguous()
        minima = torch.full((len(positions),), maximum_distance_m, device=self.device)
        active = (
            torch.ones(len(positions), device=self.device, dtype=torch.int32)
            if query_mask is None
            else query_mask.reshape(-1).to(device=self.device, dtype=torch.int32).contiguous()
        )
        if not len(positions) or not len(local_points):
            return minima.reshape(shape)
        from gr00t.rl.pick.surface_worklist import SURFACE_BLOCK_SIZE, surface_worklist

        if not hasattr(self, "_surface_scene_inputs"):
            self._surface_scene_inputs = (
                torch.tensor([int(self.mesh.id)], device=self.device, dtype=torch.int64),
                torch.zeros(1, 3, device=self.device),
                torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32),
            )
        pairs, blocks = surface_worklist(
            *self._surface_scene_inputs,
            len(positions),
            positions,
            orientations,
            local_points,
            0,
            torch.ones(len(local_points), device=self.device, dtype=torch.int32),
            active,
            maximum_distance_m,
            check_rays,
        )
        if not len(pairs):
            return minima.reshape(shape)
        wp.launch(
            _rigid_surface_clearance_kernel,
            dim=(len(pairs), SURFACE_BLOCK_SIZE),
            inputs=[
                self.mesh.id,
                wp.from_torch(positions, dtype=wp.vec3),
                wp.from_torch(orientations, dtype=wp.vec4),
                wp.from_torch(local_points, dtype=wp.vec3),
                tolerance_m,
                int(check_rays),
                int(signed_distance),
                maximum_distance_m,
                wp.from_torch(active),
                wp.from_torch(pairs, dtype=wp.int64),
                blocks,
                SURFACE_BLOCK_SIZE,
                len(local_points),
                wp.from_torch(minima),
            ],
            device=str(self.device),
            stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)),
        )
        return minima.reshape(shape)
