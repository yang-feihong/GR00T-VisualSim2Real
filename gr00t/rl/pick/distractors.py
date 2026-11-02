"""Nearby supported object proposals with unambiguous category/support labels."""

import math

import torch

from gr00t.rl.pick.scene_geometry import SURFACE_NAMES


def object_category(asset):
    return asset.get("category", asset["name"].split("_", 1)[-1])


def target_semantics(episode):
    asset = episode["asset"]
    result = {
        "instance_id": "target",
        "object_category": object_category(asset),
        "support_category": episode["support_type"],
    }
    if "variant_id" in asset:
        result.update(
            source_instance_id=asset["source_instance_id"],
            variant_id=asset["variant_id"],
            randomization=asset["randomization"],
            size_m=asset["size_m"],
            mass_kg=asset["mass_kg"],
        )
    return result


def sample_nearby_objects(
    episode, assets, geometry, *, count=None, proposals=512, fixed_sequence=False
):
    """Single-episode tools use the same spatial batch implementation."""
    return sample_nearby_objects_batch(
        [episode],
        [assets],
        geometry,
        counts=[count],
        proposals=proposals,
        fixed_sequence=fixed_sequence,
    )[0]


def sample_nearby_objects_batch(
    episodes, asset_rows, geometry, *, counts=None, proposals=512, fixed_sequence=False
):
    """Query all scene-local environments together; only dependent object slots iterate."""
    from gr00t.rl.auto_open.tensor_metadata import to_host
    from gr00t.rl.auto_open.tensor_random import episode_uniform

    if not episodes or len(episodes) != len(asset_rows) or proposals < 1:
        raise ValueError("Object sampling requires matching nonempty rows and proposals")
    counts = [None] * len(episodes) if counts is None else counts
    if len(counts) != len(episodes):
        raise ValueError("Nearby object counts must match episode rows")
    if all(count == 0 for count in counts):
        return [([], {"requested_count": 0, "placed_count": 0}) for _ in episodes]
    for assets, count in zip(asset_rows, counts, strict=True):
        if count is not None and not 0 <= count <= 4:
            raise ValueError("Nearby object count must be between zero and four")
        if not assets and count != 0:
            raise ValueError("Object sampling requires assets")
        if fixed_sequence and count != len(assets):
            raise ValueError("A fixed asset sequence requires one placement per installed asset")
    device = geometry.device
    effective_assets = [
        assets or [episode["asset"]] for episode, assets in zip(episodes, asset_rows, strict=True)
    ]
    seeds = torch.tensor([e["seed"] for e in episodes], device=device, dtype=torch.int64)
    draws = episode_uniform(seeds, 0x44495354, proposals, 7)
    random = draws[..., :5]
    asset_counts = torch.tensor(list(map(len, effective_assets)), device=device)
    asset_ids = (draws[..., 5] * asset_counts[:, None]).long()
    requested = asset_counts.new_tensor([-1 if count is None else count for count in counts])
    requested = torch.where(requested < 0, (draws[:, 0, 6] * 5).long(), requested)
    batch, width = len(episodes), max(map(len, effective_assets))
    size_table = torch.tensor(
        [
            [a["size_m"] for a in assets] + [[0, 0, 0]] * (width - len(assets))
            for assets in effective_assets
        ],
        device=device,
        dtype=torch.float32,
    )
    categories = [[object_category(asset) for asset in assets] for assets in effective_assets]
    labels = sorted(
        {label for row in categories for label in row}
        | {object_category(e["asset"]) for e in episodes}
    )
    category_table = torch.tensor(
        [[labels.index(label) for label in row] + [0] * (width - len(row)) for row in categories],
        device=device,
        dtype=torch.long,
    )
    sizes = size_table.gather(1, asset_ids[..., None].expand(-1, -1, 3))
    target = sizes.new_tensor([e["object_position"] for e in episodes])
    angle, radius = random[..., 0] * (2 * math.pi), 0.18 + random[..., 1] * 0.82
    xy = target[:, None, :2] + radius[..., None] * torch.stack((angle.cos(), angle.sin()), -1)
    origin = torch.cat((xy, torch.full_like(xy[..., :1], 1.35)), -1)
    end = origin.clone()
    end[..., 2] = -0.20
    surface, normals, _ = geometry.rays(origin, end)
    support = geometry.last_hit_types.clone()
    support = torch.where((support == 5) & (surface[..., 2] < 0.12), 0, support)
    margin = sizes[..., :2].amax(-1) / 2 + 0.008
    offsets = sizes.new_tensor([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])
    corners = origin[..., None, :].expand(-1, -1, len(offsets), -1).clone()
    corners[..., :2] += margin[..., None, None] * offsets
    corner_ends = corners.clone()
    corner_ends[..., 2] = -0.20
    hits, corner_normals, _ = geometry.rays(corners, corner_ends)
    valid = ((hits[..., 2] - surface[..., 2, None]).abs() < 0.006).all(-1)
    valid &= (corner_normals[..., 2].abs() > 0.97).all(-1)
    valid &= (normals[..., 2].abs() > 0.97) & (surface[..., 2] >= -0.02)
    valid &= surface[..., 2] <= 1.05
    positions = surface.clone()
    positions[..., 2] += sizes[..., 2] / 2 + 0.003
    bounding_radius = sizes.norm(dim=-1) / 2
    target_radius = sizes.new_tensor([e["asset"]["size_m"] for e in episodes]).norm(dim=-1) / 2
    valid &= (positions - target[:, None]).norm(dim=-1) > (
        bounding_radius + target_radius[:, None] + 0.08
    )
    start = sizes.new_tensor([e["base_xy"] for e in episodes])
    station = sizes.new_tensor([e["station_xy"] for e in episodes])
    segment = station - start
    fraction = ((xy - start[:, None]) * segment[:, None]).sum(-1)
    fraction /= segment.square().sum(-1).clamp_min(1e-8)[:, None]
    closest = start[:, None] + fraction.clamp(0, 1)[..., None] * segment[:, None]
    valid &= (xy - closest).norm(dim=-1) > 0.75 + bounding_radius
    overhead = positions.clone()
    overhead[..., 2] += sizes[..., 2] / 2 + 0.01
    overhead_end = overhead.clone()
    overhead_end[..., 2] += 0.05
    valid &= geometry.visible(overhead, overhead_end)
    ring = torch.arange(16, device=device) * (2 * math.pi / 16)
    levels = sizes.new_tensor([0.0, 0.5, 1.0])
    centers = positions[..., None, None, :].expand(-1, -1, 3, 16, -1).clone()
    centers[..., 2] = (
        surface[..., 2, None, None]
        + 0.003
        + sizes[..., 2, None, None] * levels[None, None, :, None]
    )
    ring_ends = centers.clone()
    xy_radius = sizes[..., :2].norm(dim=-1) / 2 + 0.003
    ring_ends[..., :2] += xy_radius[..., None, None, None] * torch.stack(
        (ring.cos(), ring.sin()), -1
    )
    valid &= geometry.visible(centers, ring_ends, tolerance_m=0.0005).all(dim=(-1, -2))
    heading = sizes.new_tensor([e["base_yaw"] for e in episodes])
    camera = torch.zeros(batch, 3, device=device)
    camera[:, :2] = start + 0.46 * torch.stack((heading.cos(), heading.sin()), -1)
    camera[:, 2] = 0.546
    sight = target - camera
    fraction = ((positions - camera[:, None]) * sight[:, None]).sum(-1)
    fraction /= sight.square().sum(-1).clamp_min(1e-8)[:, None]
    closest = camera[:, None] + fraction.clamp(0, 1)[..., None] * sight[:, None]
    valid &= (positions - closest).norm(dim=-1) > bounding_radius + 0.01
    pair_ids = category_table.gather(1, asset_ids) * len(SURFACE_NAMES) + support
    target_pair = category_table.new_tensor(
        [
            labels.index(object_category(e["asset"])) * len(SURFACE_NAMES)
            + SURFACE_NAMES.index(e["support_type"])
            for e in episodes
        ]
    )
    valid &= pair_ids != target_pair[:, None]
    alive = torch.ones(batch, device=device, dtype=torch.bool)
    selected, active_slots = [], []
    owner = torch.arange(batch, device=device)
    for slot in range(4):
        allowed = valid & (asset_ids == slot) if fixed_sequence else valid
        index = random[..., 4].masked_fill(~allowed, torch.inf).argmin(-1)
        active = alive & (slot < requested) & allowed[owner, index]
        selected.append(index)
        active_slots.append(active)
        alive &= active
        separation = (positions - positions[owner, index, None]).norm(dim=-1)
        valid &= separation > bounding_radius + bounding_radius[owner, index, None] + 0.03
        valid &= pair_ids != pair_ids[owner, index, None]
    indices = torch.stack(selected, -1)
    owner = owner[:, None]
    host = to_host(
        {
            "requested": requested,
            "active": torch.stack(active_slots, -1),
            "asset": asset_ids[owner, indices],
            "position": positions[owner, indices],
            "yaw": random[owner, indices, 2] * (2 * math.pi),
            "support_z": surface[owner, indices, 2],
            "support": support[owner, indices],
        }
    )
    results = []
    for row, assets in enumerate(asset_rows):
        objects = []
        for slot, active in enumerate(host["active"][row]):
            if active:
                asset = assets[host["asset"][row][slot]]
                objects.append(
                    {
                        "instance_id": f"distractor_{slot}",
                        "asset": asset,
                        "object_category": object_category(asset),
                        "object_position": host["position"][row][slot],
                        "object_yaw": host["yaw"][row][slot],
                        "support_z": host["support_z"][row][slot],
                        "support_type": SURFACE_NAMES[host["support"][row][slot]],
                    }
                )
        results.append(
            (objects, {"requested_count": host["requested"][row], "placed_count": len(objects)})
        )
    return results


def build_distractor_mesh(descriptors, states):
    """Use cached convex collider topology and measured poses, entirely on GPU."""
    from pathlib import Path

    import numpy as np

    from gr00t.rl.auto_open.planning_geometry import _quat_rotate

    if states.shape[:2] != (1, len(descriptors)):
        raise ValueError("Distractor geometry requires one scene and matching rigid-object states")
    points, faces = [], []
    offset = 0
    for i, descriptor in enumerate(descriptors):
        directory = Path(descriptor["asset"]["usd_path"]).parent
        vertices = torch.as_tensor(
            np.load(directory / "collision_support.npy"), device=states.device
        )
        indices = torch.as_tensor(np.load(directory / "collision_faces.npy"), device=states.device)
        pose = states[0, i]
        points.append(_quat_rotate(pose[None, 3:7].expand(len(vertices), -1), vertices) + pose[:3])
        faces.append(indices + offset)
        offset += len(vertices)
    return {
        "points": torch.cat(points) if points else states.new_empty(0, 3),
        "faces": torch.cat(faces)
        if faces
        else torch.empty(0, 3, dtype=torch.int32, device=states.device),
    }


def augment_scene_geometry(geometry, obstacles):
    from gr00t.rl.pick.scene_geometry import SceneGeometry

    if not len(obstacles["points"]):
        return geometry
    if hasattr(geometry, "with_obstacles"):
        return geometry.with_obstacles(obstacles)
    points = torch.cat((geometry.points, obstacles["points"].to(geometry.device)))
    faces = torch.cat(
        (
            geometry.faces.reshape(-1, 3),
            obstacles["faces"].to(geometry.device) + len(geometry.points),
        )
    )
    labels = None
    if geometry.labels is not None:
        labels = torch.cat(
            (geometry.labels, geometry.labels.new_full((len(obstacles["faces"]),), 5))
        )
    return SceneGeometry(points, faces, device=geometry.device, labels=labels)


def main():
    """Write offline proposals, or an episode that still requires physical admission."""
    import argparse
    import json
    from pathlib import Path

    from gr00t.rl.pick.asset_library import episode_asset_pool, materialize_variant
    from gr00t.rl.pick.scene_geometry import SceneGeometry, read_scene_triangles

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=range(5))
    parser.add_argument("--write-episode", action="store_true")
    args = parser.parse_args()
    episode = json.loads(args.episode.read_text())
    points, faces, labels = read_scene_triangles(
        Path(episode["scene_usd_path"]), Path(episode["geometry_cache_root"]), return_labels=True
    )
    geometry = SceneGeometry(points, faces, labels=labels)
    assets = episode_asset_pool(episode["seed"])
    objects, diagnostics = sample_nearby_objects(episode, assets, geometry, count=args.count)
    for item in objects:
        item["asset"] = materialize_variant(item["asset"], episode["seed"])
    report = {
        "source_episode": str(args.episode),
        "target_semantics": target_semantics(episode),
        "distractors": objects,
        "sampling": diagnostics,
        "note": "Offline placement proposals, not a physically validated or runnable episode",
    }
    if args.write_episode:
        report = {
            **episode,
            "target_semantics": target_semantics(episode),
            "distractors": objects,
            "distractor_sampling": diagnostics,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[pick:distractors] {args.output} {diagnostics}", flush=True)


if __name__ == "__main__":
    main()
