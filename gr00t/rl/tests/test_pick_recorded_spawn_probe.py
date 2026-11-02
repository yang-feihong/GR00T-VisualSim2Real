"""Offline scene-backed rejection counts for stationary robot spawn proposals."""

import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not os.environ.get("PICK_SPAWN_PROBE_EPISODE"), reason="Scene probe requested")
def test_stationary_spawn_condition_counts_in_recorded_scene(tmp_path):
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.auto_open.tensor_random import episode_uniform
    from gr00t.rl.pick.randomization import NoSafeRobotSpawn, Occupancy, PickRandomizationConfig
    from gr00t.rl.pick.scene_selection import PreparationSceneCache
    from gr00t.rl.simulator.isaacsim.vlnverse_scene import discover_vlnverse_scenes

    episode = json.loads(Path(os.environ["PICK_SPAWN_PROBE_EPISODE"]).read_text())
    record = discover_vlnverse_scenes(Path(episode["scene_usd_path"]).parents[1],
                                    scene_ids=[episode["scene_id"]])[0]
    geometry = PreparationSceneCache().get(record, Path(episode["geometry_cache_root"]))
    seeds = torch.tensor([1700, 1701, 1702, 1703], device="cuda")
    draws = episode_uniform(seeds, 0x4241534B, 4096, 3)
    report = {"scene_id": record.scene_id, "physics_simulated": False, "cases": {}}
    for radius in (0.75, 0.55, 0.40):
        try:
            occupancy = Occupancy(record, geometry.device, PickRandomizationConfig(spawn_clearance_m=radius))
        except NoSafeRobotSpawn:
            report["cases"][str(radius)] = {"occupancy_cells": 0}
            continue
        xy = occupancy.positions[(draws[..., 0] * len(occupancy.positions)).long()]
        yaw = draws[..., 1] * (2 * torch.pi)
        roots = xy.new_zeros(*xy.shape[:2], 13)
        roots[..., :2], roots[..., 2] = xy, 0.55
        roots[..., 3], roots[..., 6] = (yaw / 2).cos(), (yaw / 2).sin()
        local = xy.new_tensor([[0, 0, 0.8], [-0.3, -0.2, 0.8], [-0.3, 0.2, 0.8],
                              [0.4, -0.2, 0.8], [0.4, 0.2, 0.8]])
        overhead = roots[..., None, :3] + _quat_rotate(roots[..., None, 3:7], local)
        ceiling = overhead.clone()
        ceiling[..., 2] += 0.55
        upper = geometry.visible(overhead, ceiling).all(-1)
        origins = torch.cat((xy, xy.new_full((*xy.shape[:2], 1), 0.2)), -1)
        below = origins.clone()
        below[..., 2] = -0.2
        hit, normal, _ = geometry.rays(origins, below)
        ground = (hit[..., 2].abs() < 0.025) & (normal[..., 2].abs() > 0.97)
        report["cases"][str(radius)] = {
            "occupancy_cells": len(occupancy.positions),
            "ground_counts": ground.sum(-1).tolist(), "overhead_counts": upper.sum(-1).tolist(),
            "valid_counts": (ground & upper).sum(-1).tolist(),
            "ground_z_range": [hit[..., 2].min().item(), hit[..., 2].max().item()],
        }
    (tmp_path / "stationary_spawn_conditions.json").write_text(json.dumps(report, indent=2) + "\n")
