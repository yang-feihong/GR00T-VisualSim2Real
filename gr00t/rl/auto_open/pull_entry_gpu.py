"""Device-only closed-jaw pull entry geometry and collision-aware routing.

``PullEntryGPU.from_controllers`` is an initialization boundary: it reads static
body-local collider vertices. ``plan`` accepts device state, device row indices,
and externally sampled candidate tensors; it never samples RNG or reads device
values on the host. Quaternions are wxyz throughout. No inverse solve is done.

Routes use a continuous visibility graph of exact convex configuration-space
obstacles rather than the legacy 5 mm raster. The same 1 mm panel/frame routing
inflation, 35 cm search window, 2 mm/0.025 rad sampling, and final uninflated
panel / 2 mm inflated wall checks apply. Static loops over collider parts, graph
iterations, and memory-bounded candidate/query chunks are not environment loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_rotate(q, v):
    xyz, v = torch.broadcast_tensors(q[..., 1:], v)
    uv = torch.cross(xyz, v, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + torch.cross(xyz, uv, dim=-1))


def _yaw_rotate(angle, v):
    c, s = angle.cos(), angle.sin()
    x, y, z = v.unbind(-1)
    return torch.stack((c * x - s * y, s * x + c * y, z.expand_as(c)), -1)


def _matrix_quat(m):
    a, b, c = m[..., 0, :].unbind(-1)
    d, e, f = m[..., 1, :].unbind(-1)
    g, h, i = m[..., 2, :].unbind(-1)
    terms = torch.stack((1 + a + e + i, 1 + a - e - i, 1 - a + e - i, 1 - a - e + i), -1)
    magnitude = terms.clamp_min(0.0).sqrt()
    rows = torch.stack(
        (
            torch.stack((terms[..., 0], h - f, c - g, d - b), -1),
            torch.stack((h - f, terms[..., 1], b + d, c + g), -1),
            torch.stack((c - g, b + d, terms[..., 2], f + h), -1),
            torch.stack((d - b, c + g, f + h, terms[..., 3]), -1),
        ),
        -2,
    )
    rows = rows / (2.0 * magnitude.clamp_min(1e-8)[..., :, None])
    index = magnitude.argmax(-1)
    q = rows.gather(-2, index[..., None, None].expand(*index.shape, 1, 4)).squeeze(-2)
    return F.normalize(q, dim=-1)


def _nlerp(a, b, t):
    b = torch.where((a * b).sum(-1, keepdim=True) < 0.0, -b, b)
    return F.normalize(a + t[..., None] * (b - a), dim=-1)


def _halfspaces(offset, rectangle):
    return _polygon_halfspaces(_minkowski_polygon(_ordered(offset), _ordered(rectangle)))


def _minkowski_polygon(offset, rectangle):
    """Exact convex polygon sum by merging cyclic edge directions, O(H log H)."""
    offset, rectangle = offset.double(), rectangle.double()
    polygons = []
    for polygon in (-offset, rectangle):
        low_y = polygon[..., 1].amin(1, keepdim=True)
        x = torch.where(polygon[..., 1] == low_y, polygon[..., 0], torch.inf)
        first = x.argmin(1)
        index = (
            first[:, None] + torch.arange(polygon.shape[1], device=polygon.device)
        ) % polygon.shape[1]
        polygons.append(polygon.gather(1, index[..., None].expand(-1, -1, 2)))
    edges = torch.cat([p.roll(-1, dims=1) - p for p in polygons], 1)
    angle = torch.atan2(edges[..., 1], edges[..., 0]).remainder(2.0 * math.pi)
    order = angle.argsort(dim=1, stable=True)
    edges = edges.gather(1, order[..., None].expand(-1, -1, 2))
    origin = polygons[0][:, :1] + polygons[1][:, :1]
    return origin + F.pad(edges.cumsum(1)[:, :-1], (0, 0, 1, 0))


def _polygon_halfspaces(polygon):
    polygon = polygon.double()
    delta = polygon.roll(-1, dims=1) - polygon
    axes = F.normalize(torch.stack((delta[..., 1], -delta[..., 0]), -1), dim=-1, eps=1e-12)
    return axes, (polygon * axes).sum(-1), delta.square().sum(-1) > 1e-20


@dataclass
class _Obstacles:
    axes: torch.Tensor
    support: torch.Tensor
    valid: torch.Tensor

    @property
    def plane_count(self):
        return self.axes.shape[1] * self.axes.shape[2]

    def select(self, rows):
        return _Obstacles(self.axes[rows], self.support[rows], self.valid[rows])


def _pack_obstacles(obstacles):
    if isinstance(obstacles, _Obstacles):
        return obstacles
    capacity = max(a.shape[1] for a, _, _ in obstacles)
    axes = torch.stack([F.pad(a, (0, 0, 0, capacity - a.shape[1])) for a, _, _ in obstacles], 1)
    support = torch.stack([F.pad(b, (0, capacity - b.shape[1])) for _, b, _ in obstacles], 1)
    valid = torch.stack(
        [F.pad(c, (0, capacity - c.shape[1]), value=False) for _, _, c in obstacles], 1
    )
    return _Obstacles(axes, support, valid)


def _plane_projection(points, obstacles):
    return (
        points[..., 0, None, None] * obstacles.axes[:, None, ..., 0]
        + points[..., 1, None, None] * obstacles.axes[:, None, ..., 1]
    )


def _visible(start, end, obstacles, *, epsilon=1e-8):
    """Batch all obstacle slabs in one tensor operation, not 153 dispatch loops."""
    obstacles = _pack_obstacles(obstacles)
    if start.is_cuda:
        from gr00t.rl.auto_open.pull_entry_kernels import visible

        return visible(start.double(), end.double(), obstacles, epsilon=epsilon)
    valid = obstacles.valid[:, None]
    d = _plane_projection(start, obstacles) - obstacles.support[:, None]
    slope = _plane_projection(end - start, obstacles)
    parallel = slope.abs() < 1e-12
    outside = (valid & parallel & (d >= -epsilon)).any(-1)
    crossing = (-epsilon - d) / torch.where(parallel, torch.ones_like(slope), slope)
    lower = torch.where(valid & (slope < -1e-12), crossing, 0.0).amax(-1).clamp_min(0.0)
    upper = torch.where(valid & (slope > 1e-12), crossing, 1.0).amin(-1).clamp_max(1.0)
    return (outside | (lower >= upper - 1e-8)).all(-1)


def _free(points, obstacles):
    obstacles = _pack_obstacles(obstacles)
    if points.is_cuda:
        from gr00t.rl.auto_open.pull_entry_kernels import free

        return free(points.double(), obstacles)
    free = torch.empty(points.shape[:-1], dtype=torch.bool, device=points.device)
    chunk = max(1, 4_000_000 // (points.shape[0] * obstacles.plane_count))
    for begin in range(0, points.shape[1], chunk):
        d = (
            _plane_projection(points[:, begin : begin + chunk], obstacles)
            - obstacles.support[:, None]
        )
        free[:, begin : begin + chunk] = ~((d < -1e-8) | ~obstacles.valid[:, None]).all(-1).any(-1)
    return free


def _route_graph(start, end, offsets, rectangles, *, max_nodes=16, query_chunk=1024):
    """Only exposed union-boundary vertices enter the visibility graph."""
    obstacles, nodes, masks = [], [start[:, None], end[:, None]], []
    for offset in offsets:
        for rectangle in rectangles:
            polygon = _minkowski_polygon(offset[..., :2], rectangle)
            obstacles.append(_polygon_halfspaces(polygon))
            nodes.append(polygon)
            masks.append((polygon != polygon.roll(1, dims=1)).any(-1))
    points = torch.cat(nodes, 1)
    obstacles = _pack_obstacles(obstacles)
    low, high = torch.minimum(start, end) - 0.35, torch.maximum(start, end) + 0.35
    available = _free(points, obstacles) & (
        (points >= low[:, None]) & (points <= high[:, None])
    ).all(-1)
    available &= torch.cat((torch.ones_like(available[:, :2]), *masks), 1)
    population = available[:, 2:].sum(1) + 2
    route = end[:, None].expand(-1, max_nodes, -1).clone()
    route[:, 0] = start
    length = torch.full((len(start),), 2, device=start.device, dtype=torch.long)
    safe = torch.zeros(len(start), device=start.device, dtype=torch.bool)
    previous, capacity = 0, 8
    while previous < points.shape[1]:
        ids = torch.where((population > previous) & (population <= capacity))[0]
        if len(ids):
            size = min(capacity - 2, points.shape[1] - 2)
            chosen = available[ids, 2:].to(torch.int32).topk(size, dim=1).indices + 2
            chosen = torch.cat(
                (torch.arange(2, device=start.device)[None].expand(len(ids), -1), chosen), 1
            )
            selected = points[ids].gather(1, chosen[..., None].expand(-1, -1, 2))
            valid = available[ids].gather(1, chosen)
            selected_obstacles = obstacles.select(ids)
            candidate_route, candidate_length, candidate_safe = _solve_visibility(
                selected, valid, selected_obstacles, max_nodes, query_chunk
            )
            route[ids], length[ids], safe[ids] = candidate_route, candidate_length, candidate_safe
        previous, capacity = capacity, capacity * 2
    return route, length, safe


def _solve_visibility(points, available, obstacles, max_nodes, query_chunk):
    batch, count, _ = points.shape
    obstacles = _pack_obstacles(obstacles)
    if points.is_cuda:
        # Fused slab kernels store one boolean per obstacle, not every plane.
        query_chunk = min(
            max(query_chunk, 16384), max(1, 4_000_000 // (batch * obstacles.support.shape[1]))
        )
    else:
        query_chunk = min(query_chunk, max(1, 4_000_000 // (batch * obstacles.plane_count)))
    edges = torch.empty((batch, count * count), device=points.device, dtype=points.dtype)
    for begin in range(0, count * count, query_chunk):
        ids = torch.arange(begin, min(begin + query_chunk, count * count), device=points.device)
        left, right = ids // count, ids % count
        a, b = points[:, left], points[:, right]
        safe = _visible(a, b, obstacles) & available[:, left] & available[:, right]
        edges[:, begin : begin + len(ids)] = torch.where(
            safe, torch.linalg.vector_norm(b - a, dim=-1), torch.inf
        )
    edges = edges.reshape(batch, count, count)
    distance = torch.full((batch, count), torch.inf, device=points.device, dtype=points.dtype)
    distance[:, 0] = 0.0
    visited = torch.zeros_like(distance, dtype=torch.bool)
    previous = torch.zeros_like(distance, dtype=torch.long)
    rows = torch.arange(batch, device=points.device)
    for _ in range(count):
        chosen = torch.where(visited, torch.inf, distance).argmin(-1)
        visited[rows, chosen] = True
        alternative = distance[rows, chosen, None] + edges[rows, chosen]
        better = alternative < distance
        previous = torch.where(better, chosen[:, None], previous)
        distance = torch.minimum(distance, alternative)
    reachable = available[:, 0] & available[:, 1] & torch.isfinite(distance[:, 1])
    node = torch.ones(batch, dtype=torch.long, device=points.device)
    reverse = torch.zeros((batch, max_nodes), dtype=torch.long, device=points.device)
    length = torch.zeros(batch, dtype=torch.long, device=points.device)
    for index in range(min(max_nodes, count)):
        active = node != 0
        reverse[:, index] = node
        length += active.to(torch.long)
        node = torch.where(active, previous[rows, node], node)
    overflow = (node != 0) | (length + 1 > max_nodes)
    length += 1
    index = (length[:, None] - 1 - torch.arange(max_nodes, device=points.device)).clamp(
        0, max_nodes - 1
    )
    ordered = reverse.gather(1, index)
    route = points.gather(1, ordered[..., None].expand(-1, -1, 2))
    return route, length.clamp_max(max_nodes), reachable & ~overflow


def _route_polygons(
    start, end, offsets, rectangles, *, max_nodes=16, candidate_chunk=128, diagnostics=None
):
    obstacles = [
        _halfspaces(offset[..., :2], rectangle) for offset in offsets for rectangle in rectangles
    ]
    obstacles = _pack_obstacles(obstacles)
    endpoint_free = _free(torch.stack((start, end), 1), obstacles)
    endpoints_safe = endpoint_free.all(1)
    direct = _visible(start[:, None], end[:, None], obstacles).squeeze(1)
    if diagnostics is not None:
        diagnostics.update(
            start_free=endpoint_free[:, 0], end_free=endpoint_free[:, 1], direct=direct
        )
    route = end[:, None].expand(-1, max_nodes, -1).clone()
    route[:, 0] = start
    length = torch.full((len(start),), 2, device=start.device, dtype=torch.long)
    safe = direct & endpoints_safe
    rejected = torch.where(~direct & endpoints_safe)[0]
    graph_nodes = 2 + sum(x.shape[1] + 4 for x in offsets) * len(rectangles)
    max_axes = max(x.shape[1] + 4 for x in offsets)
    chunk = min(candidate_chunk, max(1, 8_000_000 // (graph_nodes * max_axes)))
    for begin in range(0, len(rejected), chunk):
        ids = rejected[begin : begin + chunk]
        candidate_route, candidate_length, candidate_safe = _route_graph(
            start[ids],
            end[ids],
            [x[ids] for x in offsets],
            [x[ids] for x in rectangles],
            max_nodes=max_nodes,
        )
        route[ids], length[ids], safe[ids] = candidate_route, candidate_length, candidate_safe
    return route, length, safe


def _silhouette(points, topology):
    """Exact projected hull boundary from oriented 3D convex-hull adjacency."""
    triangles = topology.triangles
    a, b, c = (points[:, triangles[:, i], :2] for i in range(3))
    ab, ac = b - a, c - a
    front = ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0] > 0.0
    adjacent = topology.edge_faces
    left, right = front[:, adjacent[:, 0]], front[:, adjacent[:, 1]]
    start = torch.where(left, topology.edges[None, :, 0], topology.edges[None, :, 1])
    end = torch.where(left, topology.edges[None, :, 1], topology.edges[None, :, 0])
    first = points.gather(1, start[..., None].expand(-1, -1, points.shape[-1]))
    last = points.gather(1, end[..., None].expand(-1, -1, points.shape[-1]))
    mask = (left != right) & ((last[..., :2] - first[..., :2]).square().sum(-1) > 0.0)
    return first, last, mask


def _ordered(points):
    center = points.mean(1, keepdim=True)
    delta = points - center
    order = torch.atan2(delta[..., 1], delta[..., 0]).argsort(1)
    return points.gather(1, order[..., None].expand(-1, -1, points.shape[-1]))


def _route(
    start,
    end,
    offsets,
    rectangles,
    *,
    max_nodes=16,
    candidate_chunk=128,
    topologies=None,
    eligible=None,
    diagnostics=None,
):
    if diagnostics is not None:
        diagnostics.update(
            {
                key: torch.zeros(len(start), device=start.device, dtype=torch.bool)
                for key in ("start_free", "end_free", "direct")
            }
        )
    if eligible is not None:
        ids = torch.where(eligible)[0]
        route = end[:, None].expand(-1, max_nodes, -1).clone()
        route[:, 0] = start
        length = torch.full((len(start),), 2, device=start.device, dtype=torch.long)
        safe = torch.zeros(len(start), device=start.device, dtype=torch.bool)
        if len(ids):
            selected_diagnostics = {} if diagnostics is not None else None
            candidate_route, candidate_length, candidate_safe = _route(
                start[ids],
                end[ids],
                [x[ids] for x in offsets],
                [x[ids] for x in rectangles],
                max_nodes=max_nodes,
                candidate_chunk=candidate_chunk,
                topologies=topologies,
                diagnostics=selected_diagnostics,
            )
            route[ids], length[ids], safe[ids] = candidate_route, candidate_length, candidate_safe
            if diagnostics is not None:
                for key in diagnostics:
                    diagnostics[key][ids] = selected_diagnostics[key]
        return route, length, safe
    dtype = start.dtype
    start, end = start.double(), end.double()
    offsets, rectangles = [x.double() for x in offsets], [x.double() for x in rectangles]
    rectangles = [_ordered(x) for x in rectangles]
    if topologies is None:
        route, length, safe = _route_polygons(
            start,
            end,
            [_ordered(x[..., :2]) for x in offsets],
            rectangles,
            max_nodes=max_nodes,
            candidate_chunk=candidate_chunk,
            diagnostics=diagnostics,
        )
        return route.to(dtype), length, safe
    projected = [_silhouette(x, topology) for x, topology in zip(offsets, topologies)]
    counts = torch.stack([mask.sum(1) for _, _, mask in projected], 1).amax(1)
    route = end[:, None].expand(-1, max_nodes, -1).clone()
    route[:, 0] = start
    length = torch.full((len(start),), 2, device=start.device, dtype=torch.long)
    safe = torch.zeros(len(start), device=start.device, dtype=torch.bool)
    # Power-of-two silhouette buckets avoid padding every projection to the
    # asset's 3D vertex count. No vertex cap or collision approximation is used.
    maximum = max(x.edges.shape[0] for x in topologies)
    previous, capacity = 0, 8
    while previous < maximum:
        ids = torch.where((counts > previous) & (counts <= capacity))[0]
        if len(ids):
            polygons = []
            for first, _, mask in projected:
                size = min(capacity, mask.shape[1])
                chosen = mask[ids].to(torch.int32).topk(size, dim=1).indices
                selected = first[ids].gather(1, chosen[..., None].expand(-1, -1, first.shape[-1]))[
                    ..., :2
                ]
                valid = mask[ids].gather(1, chosen)
                selected = torch.where(valid[..., None], selected, selected[:, :1])
                polygons.append(_ordered(selected))
            selected_diagnostics = {} if diagnostics is not None else None
            candidate_route, candidate_length, candidate_safe = _route_polygons(
                start[ids],
                end[ids],
                polygons,
                [x[ids] for x in rectangles],
                max_nodes=max_nodes,
                candidate_chunk=candidate_chunk,
                diagnostics=selected_diagnostics,
            )
            route[ids], length[ids], safe[ids] = candidate_route, candidate_length, candidate_safe
            if diagnostics is not None:
                for key in diagnostics:
                    diagnostics[key][ids] = selected_diagnostics[key]
        previous, capacity = capacity, capacity * 2
    return route.to(dtype), length, safe


@dataclass
class ColliderPart:
    points_body: torch.Tensor  # [environments, vertices, 3]
    body_ids: torch.Tensor  # [environments]
    edges: torch.Tensor | None = None  # oriented [E,2], initialization only
    triangles: torch.Tensor | None = None  # outward-oriented [F,3]
    edge_faces: torch.Tensor | None = None  # adjacent face indices [E,2]


def _static_convex_topology(points):
    """Asset-import-only exact hull reduction; never called by online planning."""
    import numpy as np
    from scipy.spatial import ConvexHull

    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    hull = ConvexHull(points)
    keep = hull.vertices
    inverse = np.full(len(points), -1, dtype=np.int64)
    inverse[keep] = np.arange(len(keep))
    triangles = inverse[hull.simplices].copy()
    vertices = points[keep]
    for index, face in enumerate(triangles):
        a, b, c = vertices[face]
        if np.dot(np.cross(b - a, c - a), hull.equations[index, :3]) < 0.0:
            triangles[index] = face[[0, 2, 1]]
    adjacency = {}
    for index, face in enumerate(triangles):
        for a, b in zip(face, np.roll(face, -1)):
            adjacency.setdefault(tuple(sorted((int(a), int(b)))), []).append(
                (index, int(a), int(b))
            )
    edges, faces = [], []
    for entries in adjacency.values():
        if len(entries) != 2:
            raise ValueError("Convex collider hull must be a closed two-manifold")
        left, right = entries
        edges.append((left[1], left[2]))
        faces.append((left[0], right[0]))
    return vertices.astype(np.float32), np.asarray(edges), triangles, np.asarray(faces)


class PullEntryGPU:
    """Static geometry cache; all outputs of ``plan`` retain candidate axes.

    ``samples`` maps height_m/insertion_m/yaw_rad to [B,C] device tensors.
    frame_edge_m/wall_front_m are [B], hardware_height_bounds is optional [B,2].
    Output positions/orientations are [B,C,max_samples,3/4], lengths [B,C].
    ``feasible`` includes geometry, route, full interpolated collision checks,
    and buffer overflow rejection. Invalid candidates must not be sent to IK.
    ``normal``, endpoint/contact_point/face_offset_local are [B,C,3].
    """

    def __init__(
        self,
        *,
        hinge_world,
        door_normal,
        tangent,
        panel_center,
        panel_width,
        panel_half_thickness,
        hinge_axis_sign,
        hinge_idx,
        clouds,
        parts,
        max_samples=2048,
        max_route_nodes=None,
        candidate_chunk=128,
        contact_standoff_m=0.002,
        hardware_clearance_m=0.01,
    ):
        self.hinge_world, self.door_normal, self.tangent = hinge_world, door_normal, tangent
        self.panel_center, self.panel_width = panel_center, panel_width
        self.panel_half_thickness = panel_half_thickness
        self.hinge_axis_sign, self.hinge_idx = hinge_axis_sign, hinge_idx
        self.clouds, self.parts = clouds, parts
        self.max_samples = max_samples
        self.max_route_nodes = max_samples if max_route_nodes is None else max_route_nodes
        self.candidate_chunk = candidate_chunk
        self.contact_standoff_m = contact_standoff_m
        self.hardware_clearance_m = hardware_clearance_m

    @classmethod
    def from_controllers(cls, controllers, *, device, dtype=torch.float32, config=None, **kwargs):
        """Host initialization only. Topology must be shared; vertices may differ."""
        import numpy as np

        def tensor(values, integer=False):
            return torch.as_tensor(values, device=device, dtype=torch.long if integer else dtype)

        def field(name):
            # .tolist here applies only to host geometry arrays at initialization.
            return tensor(
                [
                    getattr(c, name).tolist()
                    if hasattr(getattr(c, name), "tolist")
                    else getattr(c, name)
                    for c in controllers
                ]
            )

        clouds = {}
        parts = []
        cloud_sources, metadata, topology_cache = {}, [], {}

        def pack(sources, body_ids, name):
            packed = []
            for source in sources:
                key = id(source)
                if key not in topology_cache:
                    topology_cache[key] = _static_convex_topology(source)
                packed.append(topology_cache[key])
            first = packed[0]
            for other in packed[1:]:
                if other[0].shape != first[0].shape or any(
                    not np.array_equal(a, b) for a, b in zip(first[1:], other[1:])
                ):
                    raise ValueError(
                        "Convex collider topology differs across environments: " + name
                    )
            metadata.append(
                dict(
                    name=name,
                    original_vertices=len(sources[0]),
                    hull_vertices=len(first[0]),
                    edges=len(first[1]),
                    triangles=len(first[2]),
                )
            )
            return ColliderPart(
                tensor([x[0].tolist() for x in packed]),
                body_ids,
                tensor(first[1].tolist(), True),
                tensor(first[2].tolist(), True),
                tensor(first[3].tolist(), True),
            )

        for name in (
            "fixed_finger",
            "moving_finger",
            "fixed_contact_surfaces",
            "moving_contact_surfaces",
            "wrist",
        ):
            present = [name in c.center_geometry for c in controllers]
            if not any(present):
                continue
            if not all(present):
                raise ValueError(
                    "Pull collider topology must be shared across environments: " + name
                )
            geometries = [c.center_geometry[name] for c in controllers]
            body_ids = tensor(
                [
                    c.moving_finger_body_idx
                    if name.startswith("moving")
                    else c.fixed_finger_body_idx
                    for c in controllers
                ],
                integer=True,
            )
            points = tensor([g.points_body.tolist() for g in geometries])
            clouds[name] = ColliderPart(points, body_ids)
            cloud_sources[name] = [g.points_body for g in geometries]
            if name in ("fixed_contact_surfaces", "moving_contact_surfaces", "wrist"):
                counts = [len(g.convex_parts_body) for g in geometries]
                if len(set(counts)) != 1:
                    raise ValueError("Pull convex-part topology differs: " + name)
                for index in range(counts[0]):
                    parts.append(
                        pack(
                            [g.convex_parts_body[index] for g in geometries],
                            body_ids,
                            f"{name}:{index}",
                        )
                    )
        if "fixed_finger" not in clouds or "moving_finger" not in clouds:
            raise ValueError("Both closed finger point clouds are required")
        if config is not None:
            kwargs.setdefault("contact_standoff_m", config.contact_standoff_m)
            kwargs.setdefault("hardware_clearance_m", config.hardware_clearance_m)
        if not parts:
            parts = [
                pack(cloud_sources[name], cloud.body_ids, name) for name, cloud in clouds.items()
            ]
        result = cls(
            hinge_world=field("hinge_world"),
            door_normal=field("door_normal"),
            tangent=field("panel_hinge_to_free_direction"),
            panel_center=field("panel_center_world"),
            panel_width=field("panel_hinge_to_free_edge"),
            panel_half_thickness=field("panel_half_thickness"),
            hinge_axis_sign=field("hinge_axis_sign"),
            hinge_idx=tensor([c.hinge_idx for c in controllers], True),
            clouds=clouds,
            parts=parts,
            **kwargs,
        )
        result.collider_metadata = metadata
        print("[pull_entry_gpu] exact collider topology: " + str(metadata), flush=True)
        return result

    def _local(self, part, state, rows):
        ee, q = state["ee_pos"][rows], state["ee_orn"][rows]
        body = state["robot_body_state_w"][rows, part.body_ids[rows], :7]
        world = (
            quat_rotate(body[:, None, 3:7], part.points_body[rows].to(body.dtype))
            + body[:, None, :3]
        )
        return quat_rotate(quat_conjugate(q)[:, None], world - ee[:, None])

    def plan(
        self, state, rows, samples, *, frame_edge_m, wall_front_m, hardware_height_bounds=None
    ):
        """Plan every externally drawn candidate with no device-to-host reads."""
        output_dtype = state["ee_pos"].dtype
        # Rounding a rigidly transformed dense hull to float32 can destroy its
        # convexity before silhouette extraction. Promoting only the finished
        # offsets is too late at the legacy 1e-8 strict-interior threshold.
        state = dict(state)
        for key in ("ee_pos", "ee_orn", "root_states", "robot_body_state_w", "door_joint_position"):
            state[key] = state[key].to(torch.float64)
        samples = {key: value.to(torch.float64) for key, value in samples.items()}
        frame_edge_m, wall_front_m = frame_edge_m.double(), wall_front_m.double()
        if hardware_height_bounds is not None:
            hardware_height_bounds = hardware_height_bounds.double()
        batch, count = samples["height_m"].shape
        if batch != rows.shape[0]:
            raise ValueError("samples first dimension must match rows")
        if batch == 0 or count == 0:
            reference = state["ee_pos"].to(output_dtype)
            shapes = dict(
                positions=(self.max_samples, 3),
                orientations=(self.max_samples, 4),
                endpoint=(3,),
                orientation=(4,),
                contact_point=(3,),
                face_offset_local=(3,),
                normal=(3,),
                gripper_height_bounds_m=(2,),
                tip_radius_m=(),
                ee_tangent_radius_m=(),
                entry_hinge=(),
            )
            result = {
                key: reference.new_empty((batch, count, *shape)) for key, shape in shapes.items()
            }
            result["lengths"] = torch.zeros(
                (batch, count), device=reference.device, dtype=torch.long
            )
            for key in (
                "feasible",
                "geometry_safe",
                "route_safe",
                "collision_safe",
                "overflow",
                "route_start_free",
                "route_end_free",
                "route_direct",
            ):
                result[key] = torch.zeros((batch, count), device=reference.device, dtype=torch.bool)
            return result

        def repeat(x):
            return x[:, None].expand(-1, count, *x.shape[1:]).flatten(0, 1)

        ee, ee_q = repeat(state["ee_pos"][rows]), repeat(state["ee_orn"][rows])
        local = {name: repeat(self._local(part, state, rows)) for name, part in self.clouds.items()}
        parts = [repeat(self._local(part, state, rows)) for part in self.parts]
        hinge = repeat(self.hinge_world[rows])
        base_normal, base_tangent = repeat(self.door_normal[rows]), repeat(self.tangent[rows])
        entry_hinge = repeat(state["door_joint_position"][rows, self.hinge_idx[rows]])
        angle = repeat(self.hinge_axis_sign[rows]) * entry_hinge
        normal, tangent = _yaw_rotate(angle, base_normal), _yaw_rotate(angle, base_tangent)
        midplane = hinge + _yaw_rotate(angle, repeat(self.panel_center[rows]) - hinge)
        width, half = repeat(self.panel_width[rows]), repeat(self.panel_half_thickness[rows])
        height, insertion, yaw_noise = (
            samples[key].reshape(-1) for key in ("height_m", "insertion_m", "yaw_rad")
        )
        depth = torch.zeros_like(ee)
        depth[:, 0] = 1.0
        side = local["fixed_finger"].mean(1) - local["moving_finger"].mean(1)
        side = side - depth * (side * depth).sum(-1, keepdim=True)
        side_norm = torch.linalg.vector_norm(side, dim=-1)
        side = F.normalize(side, dim=-1)
        forward = quat_rotate(repeat(state["root_states"][rows, 3:7]), depth)
        yaw = torch.atan2(forward[:, 1], forward[:, 0]) + yaw_noise
        shaft = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1)
        outside = normal - shaft * (normal * shaft).sum(-1, keepdim=True)
        outside_norm = torch.linalg.vector_norm(outside, dim=-1)
        outside = F.normalize(outside, dim=-1)
        basis = torch.stack((depth, side, torch.cross(depth, side, dim=-1)), -1)
        orientations = []
        for sign in (1.0, -1.0):
            world_basis = torch.stack(
                (shaft, sign * outside, torch.cross(shaft, sign * outside, dim=-1)), -1
            )
            orientations.append(_matrix_quat(world_basis @ basis.transpose(-1, -2)))
        q0, q1 = orientations
        orientation = torch.where(
            ((q0 * ee_q).sum(-1).abs() >= (q1 * ee_q).sum(-1).abs())[:, None], q0, q1
        )
        rotated = [quat_rotate(orientation[:, None], part) for part in parts]
        all_offsets = quat_rotate(orientation[:, None], torch.cat(list(local.values()), 1))
        z_bounds = height[:, None] + torch.stack(
            (all_offsets[..., 2].amin(1), all_offsets[..., 2].amax(1)), -1
        )
        hardware_safe = torch.ones_like(height, dtype=torch.bool)
        if hardware_height_bounds is not None:
            bounds = repeat(hardware_height_bounds)
            hardware_safe = ~(
                (z_bounds[:, 0] < bounds[:, 1] + self.hardware_clearance_m)
                & (z_bounds[:, 1] > bounds[:, 0] - self.hardware_clearance_m)
            )
        tips = torch.cat(
            (
                local.get("fixed_contact_surfaces", local["fixed_finger"]),
                local.get("moving_contact_surfaces", local["moving_finger"]),
            ),
            1,
        )
        tip_offset = (quat_rotate(orientation[:, None], tips) * tangent[:, None]).sum(-1).amin(1)
        tip_radius = width - insertion
        radius = tip_radius - tip_offset
        # A clipped convex support occurs at a hull vertex or a true 3D hull
        # edge/boundary crossing. Interior mesh vertices are removed at import.
        support_points, support_scores = [], []
        for offset, topology in zip(rotated, self.parts):
            span = radius[:, None] + (offset * tangent[:, None]).sum(-1)
            a, b = topology.edges[:, 0], topology.edges[:, 1]
            possible = [offset]
            masks = [(span >= 0.0) & (span <= width[:, None])]
            for boundary in (torch.zeros_like(width), width):
                left, right = span[:, a] - boundary[:, None], span[:, b] - boundary[:, None]
                denominator = left - right
                fraction = left / torch.where(
                    denominator.abs() > 1e-12, denominator, torch.ones_like(denominator)
                )
                possible.append(offset[:, a] + fraction[..., None] * (offset[:, b] - offset[:, a]))
                masks.append(left * right < 0.0)
            points, mask = torch.cat(possible, 1), torch.cat(masks, 1)
            score = torch.where(mask, (points * normal[:, None]).sum(-1), -torch.inf)
            selected = score.argmax(1)
            support_points.append(
                points.gather(1, selected[:, None, None].expand(-1, 1, 3)).squeeze(1)
            )
            support_scores.append(score.amax(1))
        scores = torch.stack(support_scores, 1)
        face_offset = (
            torch.stack(support_points, 1)
            .gather(1, scores.argmax(1)[:, None, None].expand(-1, 1, 3))
            .squeeze(1)
        )
        endpoint = hinge + radius[:, None] * tangent
        endpoint = torch.cat((endpoint[:, :2], height[:, None]), -1)
        endpoint += (
            ((midplane - endpoint - face_offset) * normal).sum(-1) - half - self.contact_standoff_m
        )[:, None] * normal
        contact = endpoint + face_offset + self.contact_standoff_m * normal
        geometry_safe = (
            (side_norm >= 1e-6)
            & (outside_norm >= 0.1)
            & hardware_safe
            & (-(shaft * tangent).sum(-1) > 0.1)
            & torch.isfinite(scores.amax(1))
        )
        edge, front = repeat(frame_edge_m), repeat(wall_front_m)
        rectangles = self._rectangles(
            hinge, normal, tangent, base_normal, base_tangent, midplane, width, half, edge, front
        )
        route_diagnostics = {}
        route, route_length, route_safe = _route(
            ee[:, :2],
            endpoint[:, :2],
            rotated,
            rectangles,
            max_nodes=self.max_route_nodes,
            candidate_chunk=self.candidate_chunk,
            topologies=self.parts,
            eligible=geometry_safe,
            diagnostics=route_diagnostics,
        )
        route3 = torch.cat((route, height[:, None, None].expand(-1, self.max_route_nodes, 1)), -1)
        positioned = torch.cat((ee[:, :2], height[:, None]), -1)
        waypoints = torch.cat((ee[:, None], positioned[:, None], route3), 1)
        quats = torch.cat(
            (
                ee_q[:, None].expand(-1, 2, -1),
                orientation[:, None].expand(-1, self.max_route_nodes, -1),
            ),
            1,
        )
        positions, path_q, lengths, overflow = self._sample(waypoints, quats, route_length + 2)
        validation_lengths = torch.where(geometry_safe & route_safe & ~overflow, lengths, 0)
        collision_safe = self._validate(
            positions,
            path_q,
            validation_lengths,
            parts,
            hinge,
            normal,
            tangent,
            midplane,
            width,
            half,
            base_normal,
            base_tangent,
            edge,
            front,
        )
        finite = torch.isfinite(endpoint).all(-1) & torch.isfinite(orientation).all(-1)
        feasible = geometry_safe & route_safe & ~overflow & collision_safe & finite
        result = dict(
            positions=positions,
            orientations=path_q,
            lengths=lengths,
            feasible=feasible,
            geometry_safe=geometry_safe,
            route_safe=route_safe,
            route_start_free=route_diagnostics["start_free"],
            route_end_free=route_diagnostics["end_free"],
            route_direct=route_diagnostics["direct"],
            collision_safe=collision_safe,
            overflow=overflow,
            endpoint=endpoint,
            orientation=orientation,
            contact_point=contact,
            face_offset_local=quat_rotate(quat_conjugate(orientation), face_offset),
            tip_radius_m=tip_radius,
            ee_tangent_radius_m=radius,
            gripper_height_bounds_m=z_bounds,
            entry_hinge=entry_hinge,
            normal=normal,
        )
        return {
            key: (value.to(output_dtype) if value.is_floating_point() else value).reshape(
                batch, count, *value.shape[1:]
            )
            for key, value in result.items()
        }

    @staticmethod
    def _rectangles(
        hinge, normal, tangent, base_normal, base_tangent, midplane, width, half, edge, front
    ):
        depth = ((midplane - hinge) * normal).sum(-1)

        def rectangle(n, t, d0, d1, s0, s1):
            return torch.stack(
                [
                    hinge[:, :2] + d[:, None] * n[:, :2] + s[:, None] * t[:, :2]
                    for d in (d0, d1)
                    for s in (s0, s1)
                ],
                1,
            )

        neg, pos = torch.full_like(width, -5.0), torch.full_like(width, 5.0)
        return [
            rectangle(
                normal,
                tangent,
                depth - half - 0.001,
                depth + half + 0.001,
                torch.full_like(width, -0.001),
                width + 0.001,
            ),
            rectangle(
                base_normal, base_tangent, neg, front + 0.003, neg, torch.full_like(width, -0.001)
            ),
            rectangle(base_normal, base_tangent, neg, front + 0.003, edge + 0.001, pos),
        ]

    def _sample(self, points, quats, waypoint_lengths):
        delta = torch.linalg.vector_norm(points[:, 1:] - points[:, :-1], dim=-1)
        angle = 2.0 * torch.acos((quats[:, 1:] * quats[:, :-1]).sum(-1).abs().clamp(0.0, 1.0))
        steps = (
            torch.maximum(torch.ceil(delta / 0.002), torch.ceil(angle / 0.025)).clamp_min(1).long()
        )
        active = (
            torch.arange(steps.shape[1], device=points.device)[None] < waypoint_lengths[:, None] - 1
        )
        steps = torch.where(active, steps, 0)
        ends = steps.cumsum(1)
        lengths = 1 + steps.sum(1)
        overflow = lengths > self.max_samples
        time = torch.arange(self.max_samples, device=points.device)[None].expand(
            points.shape[0], -1
        )
        time = torch.minimum(time, (lengths - 1)[:, None])
        segment = torch.searchsorted(ends.contiguous(), time.contiguous(), right=False).clamp_max(
            steps.shape[1] - 1
        )
        before = F.pad(ends[:, :-1], (1, 0)).gather(1, segment)
        fraction = (time - before).to(points.dtype) / steps.gather(1, segment).clamp_min(1)

        def gather(x, index):
            return x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))

        start, end = gather(points, segment), gather(points, segment + 1)
        q0, q1 = gather(quats, segment), gather(quats, segment + 1)
        positions = (1.0 - fraction[..., None]) * start + fraction[..., None] * end
        orientations = _nlerp(q0, q1, fraction)
        positions[:, 0], orientations[:, 0] = points[:, 0], quats[:, 0]
        return positions, orientations, lengths.clamp_max(self.max_samples), overflow

    def _validate(
        self,
        positions,
        quats,
        lengths,
        parts,
        hinge,
        normal,
        tangent,
        midplane,
        width,
        half,
        base_normal,
        base_tangent,
        edge,
        front,
    ):
        owner, sample = torch.where(
            torch.arange(positions.shape[1], device=positions.device)[None] < lengths[:, None]
        )
        bad = torch.zeros(positions.shape[0], dtype=torch.int32, device=positions.device)
        sample_chunk = min(4096, max(1, 1_000_000 // max(x.edges.shape[0] for x in self.parts)))
        for begin in range(0, len(owner), sample_chunk):
            ids, times = owner[begin : begin + sample_chunk], sample[begin : begin + sample_chunk]
            p, q = positions[ids, times], quats[ids, times]
            hit = torch.zeros(len(ids), dtype=torch.bool, device=p.device)
            zero = torch.zeros_like(width[ids])
            neg, pos = torch.full_like(zero, -100.0), torch.full_like(zero, 100.0)
            for part, topology in zip(parts, self.parts):
                world = p[:, None] + quat_rotate(q[:, None], part[ids])
                relative = world - hinge[ids, None]
                panel = torch.stack(
                    (
                        ((world - midplane[ids, None]) * normal[ids, None]).sum(-1),
                        (relative * tangent[ids, None]).sum(-1),
                    ),
                    -1,
                )
                wall = torch.stack(
                    (
                        (relative * base_normal[ids, None]).sum(-1),
                        (relative * base_tangent[ids, None]).sum(-1),
                    ),
                    -1,
                )
                for points, low, high in (
                    (
                        panel,
                        torch.stack((-half[ids], zero), -1),
                        torch.stack((half[ids], width[ids]), -1),
                    ),
                    (
                        wall,
                        torch.stack((neg, neg), -1),
                        torch.stack((front[ids] + 0.002, zero), -1),
                    ),
                    (
                        wall,
                        torch.stack((neg, edge[ids]), -1),
                        torch.stack((front[ids] + 0.002, pos), -1),
                    ),
                ):
                    first, last, boundary = _silhouette(points, topology)
                    delta = last - first
                    axes = F.normalize(
                        torch.stack((delta[..., 1], -delta[..., 0]), -1), dim=-1, eps=1e-12
                    )
                    center, extent = (low + high) / 2.0, (high - low) / 2.0
                    rectangle_min = (center[:, None] * axes).sum(-1) - (
                        extent[:, None] * axes.abs()
                    ).sum(-1)
                    support = (first * axes).sum(-1)
                    edge_separated = ((rectangle_min >= support) & boundary).any(-1)
                    box_separated = ((points.amax(1) <= low) | (points.amin(1) >= high)).any(-1)
                    hit |= ~(edge_separated | box_separated)
            bad.scatter_reduce_(0, ids, hit.to(torch.int32), reduce="amax")
        return bad == 0
