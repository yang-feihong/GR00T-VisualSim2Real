"""Reject invalid support, approach, and visibility combinations before spawning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.scene_geometry import SURFACE_NAMES, read_support_regions
from gr00t.rl.pick.support_features import asset_support_features, supported_contact_margin
from gr00t.rl.simulator.isaacsim.vlnverse_scene import _clearance_mask


@dataclass(frozen=True)
class PickRandomizationConfig:
    camera_distance_min_m: float = 1.0
    camera_distance_max_m: float = 3.0
    max_heading_error_deg: float = 60.0
    spawn_clearance_m: float = 0.55
    station_clearance_m: float = 0.48
    edge_setback_max_m: float = 0.16
    support_height_max_m: float = 1.05
    support_flatness_m: float = 0.006
    arm_joint_noise_rad: float = 0.05
    candidates: int = 1000


class NoSafeRobotSpawn(ValueError):
    """A scene has no occupancy cell meeting the requested body clearance."""


class Occupancy:
    def __init__(self, record, device, config):
        metadata = json.loads(record.occupancy_metadata_path.read_text())
        self.scale = float(metadata["scale"])
        self.upper_x = float(metadata["upper"][0])
        self.lower_y = float(metadata["lower"][1])
        free = np.asarray(Image.open(record.occupancy_path).convert("L")) == 255
        self.spawn = torch.as_tensor(
            _clearance_mask(free, math.ceil(config.spawn_clearance_m / self.scale)), device=device
        )
        self.station = torch.as_tensor(
            _clearance_mask(free, math.ceil(config.station_clearance_m / self.scale)), device=device
        )
        row, col = torch.where(self.spawn)
        self.positions = torch.stack(
            (self.upper_x - (col + 0.5) * self.scale, self.lower_y + (row + 0.5) * self.scale), -1
        )
        if not len(self.positions):
            raise NoSafeRobotSpawn(f"No safe robot spawn: {record.scene_id}")
        row, col = torch.where(self.station)
        self.station_positions = torch.stack(
            (self.upper_x - (col + 0.5) * self.scale, self.lower_y + (row + 0.5) * self.scale), -1
        )

    def free(self, xy, *, station=False):
        mask = self.station if station else self.spawn
        col = torch.floor((self.upper_x - xy[..., 0]) / self.scale).long()
        row = torch.floor((xy[..., 1] - self.lower_y) / self.scale).long()
        valid = (row >= 0) & (col >= 0) & (row < mask.shape[0]) & (col < mask.shape[1])
        return valid & mask[row.clamp(0, mask.shape[0] - 1), col.clamp(0, mask.shape[1] - 1)]


def _proposal_inputs(record, geometry, config):
    if not hasattr(geometry, "pick_proposal_inputs"):
        geometry.pick_proposal_inputs = {}
    key = (str(record.usd_path), config)
    if key not in geometry.pick_proposal_inputs:
        geometry.pick_proposal_inputs[key] = (
            Occupancy(record, geometry.device, config),
            torch.as_tensor(read_support_regions(record.usd_path), device=geometry.device),
        )
    return geometry.pick_proposal_inputs[key]


def sample_candidates(record, asset, geometry, seed, config=PickRandomizationConfig()):
    """Single-seed tools use the same GPU batch implementation."""
    return sample_candidates_batch(record, [asset], geometry, [seed], config)[0]


def sample_candidates_batch(record, assets, geometry, seeds, config=PickRandomizationConfig()):
    """One proposal and geometry batch per scene; no per-environment device dispatch."""
    from gr00t.rl.auto_open.tensor_random import episode_uniform

    if len(assets) != len(seeds) or not seeds:
        raise ValueError("A candidate batch requires one asset for each seed")
    occupancy, regions = _proposal_inputs(record, geometry, config)
    seed_tensor = torch.as_tensor(seeds, device=geometry.device, dtype=torch.int64)
    random = episode_uniform(seed_tensor, 0x504C4143, config.candidates, 12)
    sizes = random.new_tensor([asset["size_m"] for asset in assets])[:, None]
    contacts, centers = asset_support_features(assets, geometry)
    station = occupancy.station_positions[
        (random[..., 8] * len(occupancy.station_positions)).long()
    ]
    heading = random[..., 0] * (2 * math.pi)
    forward = torch.stack((heading.cos(), heading.sin()), -1)
    radius = 0.60 + random[..., 1] * 0.18
    target_xy = station + radius[..., None] * forward
    if len(regions):
        number = config.candidates // 2
        chosen = regions[(random[:, :number, 9] * len(regions)).long()]
        side = (random[:, :number, 10] * 4).long()
        inward = random.new_tensor([[1, 0], [-1, 0], [0, 1], [0, -1]])[side]
        yaw = random[:, :number, 4] * (2 * math.pi)
        direction = torch.stack(
            (
                yaw.cos() * inward[..., 0] + yaw.sin() * inward[..., 1],
                -yaw.sin() * inward[..., 0] + yaw.cos() * inward[..., 1],
            ),
            -1,
        )
        closest = (contacts[:, None, :, :2] * direction[:, :, None]).sum(-1).amin(-1)
        margin = (-closest + 0.003 + 0.02 * random[:, :number, 6]).clamp_min(0.003)
        span = (chosen[..., 2:] - chosen[..., :2] - 2 * margin[..., None]).clamp_min(0)
        xy = chosen[..., :2] + margin[..., None] + span * random[:, :number, 5:7]
        xy[..., 0] = torch.where(side == 0, chosen[..., 0] + margin, xy[..., 0])
        xy[..., 0] = torch.where(side == 1, chosen[..., 2] - margin, xy[..., 0])
        xy[..., 1] = torch.where(side == 2, chosen[..., 1] + margin, xy[..., 1])
        xy[..., 1] = torch.where(side == 3, chosen[..., 3] - margin, xy[..., 1])
        target_xy[:, :number] = xy
        station[:, :number] = xy - radius[:, :number, None] * forward[:, :number]
    proposals = (
        random[..., :8].flatten(0, 1),
        station.flatten(0, 1),
        heading.flatten(),
        forward.flatten(0, 1),
        target_xy.flatten(0, 1),
        sizes.expand(-1, config.candidates, -1).flatten(0, 1),
    )
    result = _evaluate_candidates(
        geometry,
        occupancy,
        proposals,
        config,
        support_contacts=contacts[:, None].expand(-1, config.candidates, -1, -1).flatten(0, 1),
        support_centers=centers[:, None].expand(-1, config.candidates, -1).flatten(0, 1),
    )
    rows = []
    for index in range(len(seeds)):
        sl = slice(index * config.candidates, (index + 1) * config.candidates)
        rows.append(
            {
                key: (
                    {name: mask[sl] for name, mask in value.items()}
                    if key == "checks"
                    else value[sl]
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in result.items()
            }
        )
    return rows


def _evaluate_candidates(
    geometry, occupancy, proposals, config, *, support_contacts=None, support_centers=None
):
    random, station, heading, forward, target_xy, sizes = proposals
    count, device = len(random), geometry.device
    down_from = torch.cat((target_xy, torch.full((count, 1), 1.35, device=device)), -1)
    down_to = down_from.clone()
    down_to[:, 2] = -0.2
    surface, normal, _ = geometry.rays(down_from, down_to)
    support_z = surface[:, 2]
    support_type = geometry.last_hit_types.clone()
    support_type = torch.where((support_type == 5) & (support_z < 0.12), 0, support_type)
    half_height = sizes[:, 2] / 2
    target = surface.clone()
    target[:, 2] += half_height + 0.003
    if support_contacts is None:
        half_width = sizes[:, :2].amax(-1) / 2 + 0.008
        offsets = target.new_tensor([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])
        offsets = offsets[None] * half_width[:, None, None]
    else:
        yaw = random[:, 4] * (2 * math.pi)
        rotation = torch.stack(((yaw / 2).cos(), yaw * 0, yaw * 0, (yaw / 2).sin()), -1)
        stencil = torch.cat((support_contacts, support_centers[:, None]), 1)
        offsets = _quat_rotate(rotation[:, None], stencil)[..., :2]
    corners = down_from[:, None, :].expand(-1, offsets.shape[1], -1).clone()
    corners[..., :2] += offsets
    ends = corners.clone()
    ends[..., 2] = -0.2
    corner_hits, corner_normals, _ = geometry.rays(corners, ends)
    hit_supported = (corner_hits[..., 2] - support_z[:, None]).abs() < config.support_flatness_m
    hit_supported &= corner_normals[..., 2].abs() > 0.97
    supported = (
        hit_supported.all(-1)
        if support_contacts is None
        else (
            supported_contact_margin(corners[:, :-1], hit_supported[:, :-1], corners[:, -1])
            >= 0.001
        )
        & hit_supported[:, -1]
    )
    support = (
        (normal[:, 2].abs() > 0.97)
        & (support_z >= -0.02)
        & (support_z <= config.support_height_max_m)
        & supported
    )
    edge_probe = down_from.clone()
    edge_probe[:, :2] -= forward * config.edge_setback_max_m
    edge_end = edge_probe.clone()
    edge_end[:, 2] = -0.2
    edge_hit, _, _ = geometry.rays(edge_probe, edge_end)
    floor = support_z < 0.12
    near_edge = floor | (edge_hit[:, 2] < support_z - 0.15)
    offsets = target.new_tensor(
        [
            [0, 0],
            [0.10, 0],
            [-0.10, 0],
            [0, 0.10],
            [0, -0.10],
            [0.10, 0.10],
            [-0.10, 0.10],
            [0.10, -0.10],
            [-0.10, -0.10],
        ]
    )
    overhead_from = surface[:, None].expand(-1, len(offsets), -1).clone()
    overhead_from[..., :2] += offsets
    overhead_from[..., 2] += 0.07
    overhead_to = overhead_from.clone()
    overhead_to[..., 2] += 0.25
    overhead_clear = geometry.visible(overhead_from, overhead_to).all(-1)
    start_distance = 1.5 + random[:, 2] * 1.7
    start = target_xy - start_distance[:, None] * forward
    start_yaw = heading + (random[:, 3] - 0.5) * math.radians(36)
    camera = target.new_zeros(count, 3)
    camera[:, :2] = start + 0.46 * torch.stack((start_yaw.cos(), start_yaw.sin()), -1)
    camera[:, 2] = 0.546
    camera_delta = target - camera
    camera_distance = torch.linalg.vector_norm(camera_delta, dim=-1)
    horizontal = torch.linalg.vector_norm(camera_delta[:, :2], dim=-1)
    camera_yaw = torch.atan2(camera_delta[:, 1], camera_delta[:, 0])
    heading_error = torch.atan2((camera_yaw - start_yaw).sin(), (camera_yaw - start_yaw).cos())
    # Narrow calibration's usable image requires more than the user's 60-degree heading bound.
    in_fov = (heading_error.abs() < math.radians(30)) & (
        torch.atan2(camera_delta[:, 2].abs(), horizontal) < math.radians(32)
    )
    fraction = torch.linspace(0, 1, 40, device=device)
    corridor = start[:, None, :] + fraction[None, :, None] * (station - start)[:, None, :]
    safe_corridor = occupancy.free(corridor, station=True).all(-1)
    robot_floor_from = torch.cat((start, torch.full((count, 1), 0.2, device=device)), -1)
    robot_floor_to = robot_floor_from.clone()
    robot_floor_to[:, 2] = -0.2
    ground, ground_normal, _ = geometry.rays(robot_floor_from, robot_floor_to)
    ground_valid = (ground[:, 2].abs() < 0.025) & (ground_normal[:, 2].abs() > 0.97)
    visible = geometry.visible(camera, target)
    valid = (
        support
        & near_edge
        & overhead_clear
        & occupancy.free(start)
        & occupancy.free(station, station=True)
        & safe_corridor
        & ground_valid
        & visible
        & in_fov
        & (heading_error.abs() <= math.radians(config.max_heading_error_deg))
        & (camera_distance >= config.camera_distance_min_m)
        & (camera_distance <= config.camera_distance_max_m)
    )
    checks = {
        "furniture_surface": support_z >= 0.12,
        "furniture_support": support & (support_z >= 0.12),
        "accessible_furniture_edge": support & near_edge & (support_z >= 0.12),
        "flat_supported": support,
        "near_accessible_edge_or_floor": near_edge,
        "gripper_overhead_volume_clear": overhead_clear,
        "spawn_footprint_clear": occupancy.free(start),
        "approach_corridor_clear": safe_corridor,
        "ground_support": ground_valid,
        "unoccluded_body_camera": visible,
        "body_camera_fov": in_fov,
        "valid": valid,
    }
    return {
        "object_position": target,
        "object_yaw": random[:, 4] * 2 * math.pi,
        "support_z": support_z,
        "support_type": support_type,
        "base_xy": start,
        "base_yaw": start_yaw,
        "station_xy": station,
        "station_yaw": heading,
        "camera_distance": camera_distance,
        "heading_error": heading_error,
        "checks": checks,
        "occupancy": occupancy,
        "selection_priority": random[:, 7],
    }


def select_and_serialize_batch(candidates, record, assets, seeds, supports, config):
    """GPU support/point selection; transfer only final descriptors and diagnostics."""
    from gr00t.rl.auto_open.tensor_metadata import to_host
    from gr00t.rl.auto_open.tensor_random import episode_uniform

    valid = torch.stack([row["checks"]["valid"] for row in candidates])
    kinds = torch.stack([row["support_type"] for row in candidates])
    heights = torch.stack([row["support_z"] for row in candidates])
    requested = kinds.new_tensor(
        [
            -1
            if support == "any"
            else -2
            if support == "furniture"
            else -3
            if support == "floor"
            else SURFACE_NAMES.index(support)
            for support in supports
        ]
    )
    present = kinds[..., None] == torch.arange(len(SURFACE_NAMES), device=kinds.device)
    present = (present & valid[..., None]).any(1)
    seed_tensor = torch.as_tensor(seeds, device=valid.device, dtype=torch.int64)
    uniforms = episode_uniform(seed_tensor, 0x53555050, 1, 1)[:, 0, 0]
    rank = (uniforms * present.sum(-1)).long()
    sampled_kind = (present.cumsum(-1) > rank[:, None]).long().argmax(-1)
    requested = torch.where(requested == -1, sampled_kind, requested)
    valid &= torch.where(
        requested[:, None] == -2,
        heights >= 0.12,
        torch.where(requested[:, None] == -3, heights < 0.12, kinds == requested[:, None]),
    )
    priority = torch.stack([row["selection_priority"] for row in candidates])
    selected = priority.masked_fill(~valid, torch.inf).argmin(-1)
    rows = torch.arange(len(candidates), device=valid.device)
    checks = {
        key: torch.stack([row["checks"][key] for row in candidates])
        for key in candidates[0]["checks"]
    }
    checks["supported_grasp_surface"] = torch.ones_like(valid)
    checks["requested_support"] = valid
    fields = (
        "object_position",
        "object_yaw",
        "support_z",
        "support_type",
        "base_xy",
        "base_yaw",
        "station_xy",
        "station_yaw",
        "camera_distance",
        "heading_error",
    )
    host = to_host(
        {
            "valid": valid.any(-1),
            "index": selected,
            "fields": {
                key: torch.stack([row[key] for row in candidates])[rows, selected] for key in fields
            },
            "checks": {key: value[rows, selected] for key, value in checks.items()},
            "counts": {key: value.sum(-1) for key, value in checks.items()},
            "initial_arm_joints": valid.new_tensor(
                [-1.57, 2.6, -1.5, 0.4, 0.0, 0.0], dtype=torch.float32
            )
            + (episode_uniform(seed_tensor, 0x41524D, 1, 6)[:, 0] * 2 - 1)
            * config.arm_joint_noise_rad,
        }
    )
    episodes = []
    for row, (asset, seed) in enumerate(zip(assets, seeds, strict=True)):
        if not host["valid"][row]:
            episodes.append(None)
            continue
        values = {key: value[row] for key, value in host["fields"].items()}
        episodes.append(
            {
                "seed": seed,
                "scene_id": record.scene_id,
                "scene_usd_path": str(record.usd_path),
                "asset": asset,
                "object_position": values["object_position"],
                "object_yaw": values["object_yaw"],
                "support_z": values["support_z"],
                "support_kind": "floor" if values["support_z"] < 0.12 else "furniture",
                "support_type": SURFACE_NAMES[values["support_type"]],
                "base_xy": values["base_xy"],
                "base_yaw": values["base_yaw"],
                "station_xy": values["station_xy"],
                "station_yaw": values["station_yaw"],
                "camera_distance_m": values["camera_distance"],
                "heading_error_rad": values["heading_error"],
                "checks": {key: value[row] for key, value in host["checks"].items()},
                "candidate_counts": {key: value[row] for key, value in host["counts"].items()},
                "candidate_index": host["index"][row],
                "arm_preset": [-1.57, 2.6, -1.5, 0.4, 0.0, 0.0],
                "arm_joint_noise_rad": config.arm_joint_noise_rad,
                "initial_arm_joints": host["initial_arm_joints"][row],
            }
        )
    return episodes, host["counts"]
