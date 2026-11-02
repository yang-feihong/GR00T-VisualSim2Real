"""GPU shape features and exact linear support reductions, without dense contact grids."""

import torch
import warp as wp


def asset_support_features(assets, geometry, *, contact_band_m=0.002):
    """Load immutable hulls once, then reduce all missing shapes in one GPU batch."""
    from pathlib import Path

    import numpy as np

    if not hasattr(geometry, "pick_asset_support_cache"):
        geometry.pick_asset_support_cache = {}
    cache = geometry.pick_asset_support_cache
    keys = [(asset.get("usd_path", ("box", *asset["size_m"])), contact_band_m) for asset in assets]
    missing = dict(zip(keys, assets))
    missing = {key: value for key, value in missing.items() if key not in cache}
    if missing:
        vertices, faces, counts, face_counts, genuine = [], [], [], [], []
        offset = 0
        for asset in missing.values():
            if "usd_path" in asset:
                directory = Path(asset["usd_path"]).parent
                points = np.load(directory / "collision_support.npy")
                triangles = np.load(directory / "collision_faces.npy")
                genuine.append(True)
            else:
                points = np.asarray(
                    [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
                    dtype=np.float32,
                ) * np.asarray(asset["size_m"], dtype=np.float32)
                triangles = np.empty((0, 3), dtype=np.int64)
                genuine.append(False)
            vertices.append(points)
            faces.append(triangles + offset)
            counts.append(len(points))
            face_counts.append(len(triangles))
            offset += len(points)
        device = geometry.device
        points = torch.as_tensor(np.concatenate(vertices), device=device, dtype=torch.float32)
        triangles = torch.as_tensor(np.concatenate(faces), device=device, dtype=torch.long)
        rows = torch.arange(len(missing), device=device)
        owner = rows.repeat_interleave(torch.tensor(counts, device=device))
        bottom = points.new_full((len(missing),), torch.inf)
        bottom.scatter_reduce_(0, owner, points[:, 2], reduce="amin")
        directions = points.new_tensor(
            [
                [1, 0],
                [1, 1],
                [0, 1],
                [-1, 1],
                [-1, 0],
                [-1, -1],
                [0, -1],
                [1, -1],
            ]
        )
        score = points[:, :2] @ directions.T
        score = score.masked_fill(
            (points[:, 2] > bottom[owner] + contact_band_m)[:, None], -torch.inf
        )
        maxima = points.new_full((len(missing), 8), -torch.inf)
        owners = owner[:, None].expand(-1, 8)
        maxima.scatter_reduce_(0, owners, score, reduce="amax")
        indices = torch.arange(len(points), device=device)[:, None].expand(-1, 8)
        indices = indices.masked_fill(score != maxima[owner], len(points))
        selected = torch.full((len(missing), 8), len(points), device=device, dtype=torch.long)
        selected.scatter_reduce_(0, owners, indices, reduce="amin")
        contacts = points[selected]
        centers = points.new_zeros(len(missing), 3)
        if len(triangles):
            corners = points[triangles].double()
            a, b, c = corners.unbind(1)
            volume = (a * torch.cross(b, c, dim=-1)).sum(-1)
            triangle_owner = rows.repeat_interleave(torch.tensor(face_counts, device=device))
            total = volume.new_zeros(len(missing)).scatter_add_(0, triangle_owner, volume)
            numerator = volume.new_zeros(len(missing), 3)
            numerator.scatter_add_(
                0, triangle_owner[:, None].expand(-1, 3), volume[:, None] * (a + b + c)
            )
            real = torch.tensor(genuine, device=device)
            torch._assert_async(((total.abs() > 1e-12) | ~real).all(), "Closed hull needed for COM")
            denominator = torch.where(real, total * 4, torch.ones_like(total))
            centers = (numerator / denominator[:, None]).float()
        for index, key in enumerate(missing):
            cache[key] = (contacts[index], centers[index])
    return torch.stack([cache[key][0] for key in keys]), torch.stack(
        [cache[key][1] for key in keys]
    )


def bottom_contact_extrema(points, mask, *, contact_band_m):
    """Eight actual contact-band points, retaining shape rather than inventing AABB corners."""
    bottom = points[..., 2].masked_fill(~mask, torch.inf).amin(-1)
    contact = mask & (points[..., 2] <= bottom[:, None] + contact_band_m)
    directions = points.new_tensor(
        [
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [0, -1],
            [1, -1],
        ]
    )
    score = (points[..., :2, None] * directions.T).sum(-2)
    choice = score.masked_fill(~contact[..., None], -torch.inf).argmax(1)
    return points.gather(1, choice[..., None].expand(-1, -1, 3))


def supported_contact_margin(points, supported, center):
    """Minimum COM distance to the convex support polygon, in metres on device."""
    start = points[:, :, None, :2]
    edge = points[:, None, :, :2] - start
    delta = points[:, None, None, :, :2] - start[..., None, :]
    cross = edge[..., 0, None] * delta[..., 1] - edge[..., 1, None] * delta[..., 0]
    length = edge.norm(dim=-1)
    boundary = ((cross >= -1e-9) | ~supported[:, None, None]).all(-1)
    boundary &= supported[:, :, None] & supported[:, None, :] & (length > 1e-7)
    to_center = center[:, None, None, :2] - start
    distance = edge[..., 0] * to_center[..., 1] - edge[..., 1] * to_center[..., 0]
    distance /= length.clamp_min(1e-7)
    margin = distance.masked_fill(~boundary, torch.inf).amin((-2, -1))
    return torch.where(boundary.any((-2, -1)), margin, torch.full_like(margin, -torch.inf))


@wp.kernel
def _minimum_projection(
    points: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    result: wp.array(dtype=float),
):
    row, chunk, lane = wp.tid()
    index = chunk * 128 + lane
    value = float(wp.inf)
    if index < points.shape[0]:
        value = wp.dot(points[index], directions[row])
    minimum = wp.tile_min(wp.tile(value))[0]
    if lane == 0:
        wp.atomic_min(result, row, minimum)


def minimum_projection(points, directions):
    if not points.is_cuda:
        return (directions @ points.T).amin(-1)
    wp.init()
    output = directions.new_full(directions.shape[:-1], torch.inf)
    wp.launch_tiled(
        _minimum_projection,
        dim=(output.numel(), (len(points) + 127) // 128),
        inputs=[
            wp.from_torch(points.contiguous(), dtype=wp.vec3),
            wp.from_torch(directions.reshape(-1, 3).contiguous(), dtype=wp.vec3),
            wp.from_torch(output.flatten()),
        ],
        block_dim=128,
        stream=wp.stream_from_torch(torch.cuda.current_stream(points.device)),
    )
    return output


def uniform_surface_support(points, mask, count=2048):
    lengths = mask.sum(-1)
    fraction = torch.linspace(0, 1, count, device=points.device)
    indices = ((lengths - 1)[:, None] * fraction[None]).long()
    valid = torch.cat(
        (torch.ones_like(indices[:, :1], dtype=torch.bool), indices[:, 1:] != indices[:, :-1]), 1
    )
    return points.gather(1, indices[..., None].expand(-1, -1, 3)), valid
