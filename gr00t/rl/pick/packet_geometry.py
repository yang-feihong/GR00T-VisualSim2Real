"""Exact GPU triangle-packet BVH using Warp geometry primitives."""

from string import Template

import torch
import warp as wp

from gr00t.rl.pick.scene_geometry import SceneGeometry

_MODERN_BVH = tuple(int(part) for part in wp.__version__.split(".")[:2]) >= (1, 11)


@wp.func_native(
    snippet=r"""
wp::Mesh mesh = wp::mesh_get(mesh_id);
if (mesh.bvh.num_leaf_nodes != count || mesh.bvh.max_nodes != 2 * count - 1) {
    output[position] = -1;
    return;
}
int node = position;
int root = *mesh.bvh.root;
while (node != root) {
    int parent = mesh.bvh.node_parents[node];
    if (parent >= mesh.bvh.max_nodes) {
        output[position] = -2;
        return;
    }
    if (parent < 0 || !mesh.bvh.node_lowers[parent].b)
        break;
    // The completed library builder guarantees an acyclic parent graph.
    // Muted descendants are not bounded by the active query-stack depth.
    node = parent;
}
int start = mesh.bvh.node_lowers[node].i;
int end = mesh.bvh.node_uppers[node].i;
if (start < 0 || end > count || position < start || position >= end) {
    output[position] = -2;
    return;
}
output[mesh.bvh.primitive_indices[position]] = count - end + position - start;
"""
)
def reference_priority(mesh_id: wp.uint64, position: int, count: int, output: wp.array(dtype=int)):
    pass


@wp.kernel
def priority_kernel(mesh_id: wp.uint64, count: int, output: wp.array(dtype=int)):
    reference_priority(mesh_id, wp.tid(), count, output)


@wp.func_native(
    snippet=Template(r"""
float best_sq = maximum * maximum;
wp::BVH bvh = wp::bvh_get(tree);
int stack[BVH_QUERY_STACK_SIZE];
stack[0] = *bvh.root;
int pending = 1;
while (pending) {
    int node = stack[--pending];
    wp::BVHPackedNodeHalf lo = wp::bvh_load_node(bvh.node_lowers, node);
    wp::BVHPackedNodeHalf hi = wp::bvh_load_node(bvh.node_uppers, node);
    if (wp::distance_to_aabb_sq(position, wp::vec3(lo.x, lo.y, lo.z),
                                wp::vec3(hi.x, hi.y, hi.z)) > best_sq)
        continue;
    if (lo.b) {
        for (int index = lo.i; index < hi.i; ++index) {
            int packet = bvh.primitive_indices[index];
            if (wp::distance_to_aabb_sq(position, bvh.item_lowers[packet],
                                       bvh.item_uppers[packet]) > best_sq)
                continue;
            int end = wp::min((packet + 1) * packet_size, triangle_count);
            for (int face = packet * packet_size; face < end; ++face) {
                wp::vec3 a = points[indices[face * 3]];
                wp::vec3 b = points[indices[face * 3 + 1]];
                wp::vec3 c = points[indices[face * 3 + 2]];
                wp::vec3 e0 = b - a, e1 = c - a, e2 = c - b;
                if (wp::length(wp::cross(e0, e1)) /
                    (wp::dot(e0, e0) + wp::dot(e1, e1) + wp::dot(e2, e2)) < 1.e-6f)
                    continue;
                wp::vec2 bary = wp::closest_point_to_triangle(a, b, c, position);
                wp::vec3 closest = bary[0] * a + bary[1] * b + (1.f - bary[0] - bary[1]) * c;
                float distance_sq = wp::length_sq(closest - position);
                if (distance_sq < best_sq)
                    best_sq = distance_sq;
            }
        }
    } else {
        wp::BVHPackedNodeHalf ll = wp::bvh_load_node(bvh.node_lowers, lo.i);
        wp::BVHPackedNodeHalf lu = wp::bvh_load_node(bvh.node_uppers, lo.i);
        wp::BVHPackedNodeHalf rl = wp::bvh_load_node(bvh.node_lowers, hi.i);
        wp::BVHPackedNodeHalf ru = wp::bvh_load_node(bvh.node_uppers, hi.i);
        float dl = wp::distance_to_aabb_sq(position, wp::vec3(ll.x, ll.y, ll.z),
                                         wp::vec3(lu.x, lu.y, lu.z));
        float dr = wp::distance_to_aabb_sq(position, wp::vec3(rl.x, rl.y, rl.z),
                                         wp::vec3(ru.x, ru.y, ru.z));
        int farther = dl < dr ? hi.i : lo.i;
        int nearer = dl < dr ? lo.i : hi.i;
        if (wp::max(dl, dr) < best_sq)
            stack[pending++] = farther;
        if (wp::min(dl, dr) < best_sq)
            stack[pending++] = nearer;
        }
}
return wp::sqrt(best_sq);
""").substitute(root=", -1" if _MODERN_BVH else "", maximum=", maximum" if _MODERN_BVH else "")
)
def packet_distance(
    tree: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    packet_size: int,
    triangle_count: int,
    position: wp.vec3,
    maximum: float,
) -> float:
    pass


@wp.func_native(
    snippet=Template(r"""
float best = maximum;
int best_face = -1;
wp::vec3 best_normal(0.f, 0.f, 0.f);
if (maximum > 0.f) {
    wp::vec3 end = origin + maximum * direction;
    wp::vec3 epsilon(0.001f, 0.001f, 0.001f);
    wp::bvh_query_t query = wp::bvh_query_aabb(tree, wp::min(origin, end) - epsilon,
                                              wp::max(origin, end) + epsilon${root});
    int packet;
    while (wp::bvh_query_next(query, packet${maximum})) {
        int end = wp::min((packet + 1) * packet_size, triangle_count);
        for (int face = packet * packet_size; face < end; ++face) {
            wp::vec3 a = points[indices[face * 3]];
            wp::vec3 b = points[indices[face * 3 + 1]];
            wp::vec3 c = points[indices[face * 3 + 2]];
            float t, u, v, sign;
            wp::vec3 normal;
            if (wp::intersect_ray_tri_woop(origin, direction, a, b, c, t, u, v, sign, &normal)
                && t >= 0.f && (t < best || (best_face >= 0 && t == best
                                               && priorities[face] < priorities[best_face]))) {
                best = t;
                best_face = face;
                best_normal = normal;
                wp::vec3 end = origin + best * direction;
                query.input_lower = wp::min(origin, end) - epsilon;
                query.input_upper = wp::max(origin, end) + epsilon;
            }
        }
    }
}
if (output_index >= 0)
    face_output[output_index] = best_face;
wp::vec3 normal = wp::normalize(best_normal);
return wp::vec4(best, normal[0], normal[1], normal[2]);
""").substitute(root=", -1" if _MODERN_BVH else "", maximum=", best" if _MODERN_BVH else "")
)
def packet_ray(
    tree: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    packet_size: int,
    triangle_count: int,
    priorities: wp.array(dtype=int),
    origin: wp.vec3,
    direction: wp.vec3,
    maximum: float,
    face_output: wp.array(dtype=int),
    output_index: int,
) -> wp.vec4:
    pass


@wp.kernel
def packet_bounds(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    packet_size: int,
    triangle_count: int,
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
):
    packet = wp.tid()
    lo, hi = wp.vec3(wp.inf), wp.vec3(-wp.inf)
    for face in range(packet * packet_size, wp.min((packet + 1) * packet_size, triangle_count)):
        for corner in range(3):
            point = points[indices[face * 3 + corner]]
            lo = wp.min(lo, point)
            hi = wp.max(hi, point)
    lower[packet] = lo - wp.vec3(0.001)
    upper[packet] = hi + wp.vec3(0.001)


@wp.kernel
def distance_queries(
    tree: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    packet_size: int,
    triangle_count: int,
    queries: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=float),
):
    index = wp.tid()
    output[index] = packet_distance(
        tree, points, indices, packet_size, triangle_count, queries[index], 2.0
    )


@wp.kernel
def ray_queries(
    tree: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    packet_size: int,
    triangle_count: int,
    priorities: wp.array(dtype=int),
    origins: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
    faces: wp.array(dtype=int),
):
    index = wp.tid()
    delta = targets[index] - origins[index]
    result = packet_ray(
        tree,
        points,
        indices,
        packet_size,
        triangle_count,
        priorities,
        origins[index],
        wp.normalize(delta),
        wp.length(delta),
        faces,
        index,
    )
    distances[index] = result[0]
    normals[index] = wp.vec3(result[1], result[2], result[3])


@wp.func
def packet_signed_distance(
    tree: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    packet_size: int,
    triangle_count: int,
    priorities: wp.array(dtype=int),
    position: wp.vec3,
    maximum: float,
    scene_upper: wp.vec3,
    scratch: wp.array(dtype=int),
):
    distance = packet_distance(
        tree, points, indices, packet_size, triangle_count, position, maximum
    )
    if distance < maximum:
        vote = int(0)
        for axis in range(3):
            direction = wp.vec3(0.0)
            direction[axis] = 1.0
            length = wp.max(1.0, scene_upper[axis] - position[axis] + 1.0)
            hit = packet_ray(
                tree,
                points,
                indices,
                packet_size,
                triangle_count,
                priorities,
                position,
                direction,
                length,
                scratch,
                -1,
            )
            # The native Woop determinant is negative for an outward-facing hit.
            if hit[axis + 1] > 0.0:
                vote += 1
        if vote >= 2:
            distance = -distance
    return distance


@wp.kernel
def surface_queries(
    tree: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    packet_size: int,
    triangle_count: int,
    priorities: wp.array(dtype=int),
    scene_upper: wp.array(dtype=wp.vec3),
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.vec4),
    local_points: wp.array(dtype=wp.vec3),
    tolerance: float,
    check_rays: int,
    signed: int,
    maximum: float,
    minima: wp.array(dtype=float),
    scratch: wp.array(dtype=int),
):
    pose, sample = wp.tid()
    if signed == 0 and minima[pose] < 0.0:
        return
    q = orientations[pose]
    vector = wp.vec3(q[1], q[2], q[3])
    local = local_points[sample]
    cross = wp.cross(vector, local)
    delta = local + 2.0 * (q[0] * cross + wp.cross(vector, cross))
    world = delta + positions[pose]
    distance = float(0.0)
    if signed != 0:
        distance = packet_signed_distance(
            tree,
            points,
            indices,
            packet_size,
            triangle_count,
            priorities,
            world,
            maximum,
            scene_upper[0],
            scratch,
        )
    else:
        distance = packet_distance(
            tree, points, indices, packet_size, triangle_count, world, maximum
        )
    length = wp.length(delta)
    if check_rays != 0 and length > 1.0e-8:
        hit = packet_ray(
            tree,
            points,
            indices,
            packet_size,
            triangle_count,
            priorities,
            positions[pose],
            delta / length,
            length,
            scratch,
            -1,
        )
        if hit[0] < length - tolerance:
            distance = -1.0
    wp.atomic_min(minima, pose, distance)


class PacketGeometry(SceneGeometry):
    uses_packet_bvh = True

    def __init__(
        self, points, faces, device="cuda:0", packet_size=32, labels=None, reference_mesh=None
    ):
        wp.init()
        self.device = torch.device(device)
        self.points = torch.as_tensor(points, device=self.device, dtype=torch.float32)
        source_faces = torch.as_tensor(faces, device=self.device, dtype=torch.int32).reshape(-1, 3)
        original = reference_mesh
        if original is None:
            original = wp.Mesh(
                points=wp.from_torch(self.points, dtype=wp.vec3),
                indices=wp.from_torch(source_faces.flatten()),
            )
        priorities = torch.full((len(source_faces),), -1, device=self.device, dtype=torch.int32)
        self.launch(
            priority_kernel,
            len(source_faces),
            [original.id, len(source_faces), wp.from_torch(priorities)],
        )
        if bool((priorities < 0).any()):
            invalid = int((priorities == -1).sum())
            invalid_links = int((priorities == -2).sum())
            raise RuntimeError(
                f"Reference LBVH audit rejected {invalid} layout entries and "
                f"{invalid_links} invalid links/ranges"
            )
        centers = self.points[source_faces.long()].mean(1)
        lower, upper = centers.amin(0), centers.amax(0)
        coordinates = (
            (((centers - lower) / (upper - lower).clamp_min(1e-8)) * 1023).long().clamp(0, 1023)
        )
        morton = torch.zeros(len(centers), device=self.device, dtype=torch.long)
        for bit in range(10):
            for axis in range(3):
                morton |= ((coordinates[:, axis] >> bit) & 1) << (bit * 3 + axis)
        order = torch.argsort(morton, stable=True)
        self.faces = source_faces[order].flatten().contiguous()
        self.priorities = priorities[order].contiguous()
        self.labels = None if labels is None else torch.as_tensor(labels, device=self.device)[order]
        self.packet_size, self.triangle_count = packet_size, len(source_faces)
        count = (self.triangle_count + packet_size - 1) // packet_size
        self.lower = self.points.new_empty(count, 3)
        self.upper = torch.empty_like(self.lower)
        self.scene_upper = self.points.amax(0, keepdim=True)
        self.launch(
            packet_bounds,
            count,
            [
                *self.inputs(),
                wp.from_torch(self.lower, dtype=wp.vec3),
                wp.from_torch(self.upper, dtype=wp.vec3),
            ],
        )
        self.bvh = wp.Bvh(
            wp.from_torch(self.lower, dtype=wp.vec3),
            wp.from_torch(self.upper, dtype=wp.vec3),
            constructor="lbvh",
        )
        torch.cuda.synchronize(self.device)

    def inputs(self):
        return [
            wp.from_torch(self.points, dtype=wp.vec3),
            wp.from_torch(self.faces),
            self.packet_size,
            self.triangle_count,
        ]

    def launch(self, kernel, count, inputs):
        wp.launch(
            kernel,
            dim=count,
            inputs=inputs,
            device=str(self.device),
            stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)),
        )

    def distances(self, points):
        shape = points.shape[:-1]
        points = points.float().reshape(-1, 3).contiguous()
        output = points.new_empty(len(points))
        self.launch(
            distance_queries,
            len(points),
            [
                self.bvh.id,
                *self.inputs(),
                wp.from_torch(points, dtype=wp.vec3),
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
    ):
        shape = positions.shape[:-1]
        positions = positions.float().reshape(-1, 3).contiguous()
        orientations = orientations.float().reshape(-1, 4).contiguous()
        local_points = local_points.float().contiguous()
        minima = torch.full((len(positions),), maximum_distance_m, device=self.device)
        scratch = torch.empty(1, device=self.device, dtype=torch.int32)
        self.launch(
            surface_queries,
            (len(positions), len(local_points)),
            [
                self.bvh.id,
                *self.inputs(),
                wp.from_torch(self.priorities),
                wp.from_torch(self.scene_upper, dtype=wp.vec3),
                wp.from_torch(positions, dtype=wp.vec3),
                wp.from_torch(orientations, dtype=wp.vec4),
                wp.from_torch(local_points, dtype=wp.vec3),
                tolerance_m,
                int(check_rays),
                int(signed_distance),
                maximum_distance_m,
                wp.from_torch(minima),
                wp.from_torch(scratch),
            ],
        )
        return minima.reshape(shape)

    def rays(self, origins, targets):
        shape = origins.shape[:-1]
        origins, targets = (
            origins.reshape(-1, 3).float().contiguous(),
            targets.reshape(-1, 3).float().contiguous(),
        )
        distances, normals = origins.new_empty(len(origins)), torch.empty_like(origins)
        faces = torch.empty(len(origins), device=self.device, dtype=torch.int32)
        self.launch(
            ray_queries,
            len(origins),
            [
                self.bvh.id,
                *self.inputs(),
                wp.from_torch(self.priorities),
                wp.from_torch(origins, dtype=wp.vec3),
                wp.from_torch(targets, dtype=wp.vec3),
                wp.from_torch(distances),
                wp.from_torch(normals, dtype=wp.vec3),
                wp.from_torch(faces),
            ],
        )
        self.last_hit_types = torch.full_like(faces, 5, dtype=torch.long)
        if self.labels is not None:
            self.last_hit_types = torch.where(faces >= 0, self.labels[faces.clamp_min(0)].long(), 5)
        self.last_hit_types = self.last_hit_types.reshape(shape)
        hit = origins + distances[:, None] * torch.nn.functional.normalize(
            targets - origins, dim=-1
        )
        return hit.reshape(*shape, 3), normals.reshape(*shape, 3), distances.reshape(shape)
