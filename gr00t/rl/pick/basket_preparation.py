"""Scene-grouped basket layout and robot spawn preparation on GPU."""

from pathlib import Path

import torch

from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.auto_open.tensor_metadata import to_host
from gr00t.rl.pick.asset_library import episode_asset_pool, materialize_variant
from gr00t.rl.pick.basket import BASKET_GRASP_CLEARANCE_M, BasketConfig, BasketGeometry
from gr00t.rl.pick.basket_layout import sample_layouts
from gr00t.rl.pick.basket_layout_reachability import prefer_materialized_layouts
from gr00t.rl.pick.collision import ArmCollisionSamples
from gr00t.rl.pick.distractors import target_semantics
from gr00t.rl.pick.prepare import NoValidInitialScene
from gr00t.rl.pick.randomization import NoSafeRobotSpawn, Occupancy, PickRandomizationConfig
from gr00t.rl.pick.scene_selection import scene_selection_metadata, select_pick_scene
from gr00t.rl.simulator.isaacsim.vlnverse_scene import discover_vlnverse_scenes
from gr00t.rl.utils.b2z1_asset import ROBOT_URDF


def prepare_basket_batch(
    seeds,
    scenes_root,
    scene_cache,
    *,
    scene_id=None,
    asset_name=None,
    config=BasketConfig(),
    count=None,
):
    if not seeds:
        return {}
    basket = BasketGeometry.from_robot_usd()
    assets = {
        seed: episode_asset_pool(
            seed, names=[asset_name] if asset_name else None, placement_profile="basket"
        )
        for seed in seeds
    }
    pending, counts = list(seeds), dict.fromkeys(seeds, count)
    results, layouts, rejections = {}, {}, {seed: [] for seed in seeds}
    # Retry attempts are dependent; each attempt evaluates all remaining seeds together.
    for attempt in range(16):
        if not pending:
            break
        layout_seeds = [(seed + attempt * 0x4C41594F) % (2**63) for seed in pending]
        proposed = sample_layouts(
            [assets[seed] for seed in pending],
            basket,
            layout_seeds,
            config,
            counts=[counts[seed] for seed in pending],
        )
        retry = []
        for seed, layout_seed, proposal in zip(pending, layout_seeds, proposed, strict=True):
            if isinstance(proposal, NoValidInitialScene):
                counts[seed] = proposal.details["object_count"]
                rejections[seed].append(
                    {"layout_seed": layout_seed, "reason": str(proposal), **proposal.details}
                )
                retry.append(seed)
            else:
                layout, selected, diagnostics = proposal
                diagnostics.update(
                    layout_seed=layout_seed,
                    combination_rejections=rejections[seed],
                    asset_placement_profile="basket",
                    eligible_asset_category_count=len({a["category"] for a in assets[seed]}),
                    required_gripper_clearance_m=BASKET_GRASP_CLEARANCE_M,
                )
                for slot, item in enumerate(layout):
                    item["asset"] = materialize_variant(item["asset"], seed, slot)
                layouts[seed] = layout, selected, diagnostics
        pending = retry
    for seed in pending:
        results[seed] = NoValidInitialScene(
            "basket_asset_combinations_have_no_nonpenetrating_layout",
            {"object_count": counts[seed], "combination_rejections": rejections[seed]},
        )
    records = discover_vlnverse_scenes(scenes_root, scene_ids=[scene_id] if scene_id else None)
    groups = {}
    for seed in seeds:
        if seed in results:
            continue
        if not records:
            results[seed] = NoValidInitialScene("no_clear_stationary_basket_spawn")
            continue
        record = select_pick_scene(records, seed)
        groups.setdefault(record.scene_id, (record, []))[1].append(seed)
    cache_root = Path(__file__).resolve().parents[3] / "datasets/pick/.cache/geometry"
    device = basket.boxes.device
    preset = basket.boxes.new_tensor([1.57, 1.57, -1.3, 1.1, 0, 0])
    limits = basket.boxes.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    from gr00t.rl.auto_open.tensor_random import episode_uniform

    layout_keys = list(layouts)
    seed_tensor = torch.tensor(layout_keys, device=device, dtype=torch.int64)
    initial_joints = (
        preset
        + (episode_uniform(seed_tensor, 0x41524D, 1, 6)[:, 0] * 2 - 1) * config.arm_joint_noise_rad
    ).clamp(limits[:, 0] + 0.001, limits[:, 1] - 0.001)
    if layout_keys:
        improved = prefer_materialized_layouts(
            [layouts[seed] for seed in layout_keys],
            basket,
            [layouts[seed][2]["layout_seed"] for seed in layout_keys],
            initial_joints,
            limits,
            ArmCollisionSamples.from_robot_usd(device=device),
            config,
        )
        layouts.update(zip(layout_keys, improved, strict=True))
    for record, group in groups.values():
        geometry = scene_cache.get(record, cache_root)
        try:
            occupancy = Occupancy(
                record, device, PickRandomizationConfig(spawn_clearance_m=config.spawn_gap_m)
            )
        except NoSafeRobotSpawn as error:
            for seed in group:
                results[seed] = NoValidInitialScene(
                    "no_clear_stationary_basket_spawn",
                    {"scene_rejections": [record.scene_id], "reason": str(error)},
                )
            continue
        from gr00t.rl.auto_open.tensor_random import episode_uniform

        seed_tensor = torch.tensor(group, device=device, dtype=torch.int64)
        draws = episode_uniform(seed_tensor, 0x4241534B, config.base_candidates, 3)
        xy = occupancy.positions[(draws[..., 0] * len(occupancy.positions)).long()]
        yaw, priority = draws[..., 1] * (2 * torch.pi), draws[..., 2]
        joints = (
            preset
            + (episode_uniform(seed_tensor, 0x41524D, 1, 6)[:, 0] * 2 - 1)
            * config.arm_joint_noise_rad
        ).clamp(limits[:, 0] + 0.001, limits[:, 1] - 0.001)
        roots = xy.new_zeros(*xy.shape[:2], 13)
        roots[..., :2], roots[..., 2] = xy, 0.55
        roots[..., 3], roots[..., 6] = (yaw / 2).cos(), (yaw / 2).sin()
        local = xy.new_tensor(
            [[0, 0, 0.8], [-0.3, -0.2, 0.8], [-0.3, 0.2, 0.8], [0.4, -0.2, 0.8], [0.4, 0.2, 0.8]]
        )
        overhead = roots[..., None, :3] + _quat_rotate(roots[..., None, 3:7], local)
        ceiling = overhead.clone()
        ceiling[..., 2] += 0.55
        floor = torch.cat((xy, xy.new_full((*xy.shape[:2], 1), 0.2)), -1)
        below = floor.clone()
        below[..., 2] = -0.2
        hit, normal, _ = geometry.rays(floor, below)
        valid = geometry.visible(overhead, ceiling).all(-1)
        valid &= (hit[..., 2].abs() < 0.025) & (normal[..., 2].abs() > 0.97)
        selected = priority.masked_fill(~valid, torch.inf).argmin(-1)
        owner = torch.arange(len(group), device=device)
        root = roots[owner, selected]
        payloads = [layouts[seed][0] for seed in group]
        local_positions = root.new_tensor(
            [
                [d["basket_local_position"] for d in row] + [[0, 0, 0]] * (4 - len(row))
                for row in payloads
            ]
        )
        local_yaw = root.new_tensor(
            [[d["basket_local_yaw"] for d in row] + [0] * (4 - len(row)) for row in payloads]
        )
        local_support = root.new_tensor(
            [[d["support_z_local"] for d in row] + [0] * (4 - len(row)) for row in payloads]
        )
        host = to_host(
            {
                "valid": valid.any(-1),
                "xy": root[:, :2],
                "yaw": yaw[owner, selected],
                "position": _quat_rotate(root[:, None, 3:7], local_positions) + root[:, None, :3],
                "object_yaw": local_yaw + yaw[owner, selected, None],
                "support_z": local_support + root[:, 2, None],
                "joints": joints,
                "preset": preset,
            }
        )
        for row, seed in enumerate(group):
            if not host["valid"][row]:
                results[seed] = NoValidInitialScene(
                    "no_clear_stationary_basket_spawn", {"scene_rejections": [record.scene_id]}
                )
                continue
            layout, selected_target, diagnostics = layouts[seed]
            for slot, descriptor in enumerate(layout):
                descriptor.update(
                    object_position=host["position"][row][slot],
                    object_yaw=host["object_yaw"][row][slot],
                    support_z=host["support_z"][row][slot],
                )
            episode = {
                **layout[selected_target],
                "task_type": "pick_basket",
                "seed": seed,
                "scene_id": record.scene_id,
                "scene_usd_path": str(record.usd_path),
                "geometry_cache_root": str(cache_root),
                "base_xy": host["xy"][row],
                "base_yaw": host["yaw"][row],
                "station_xy": host["xy"][row],
                "station_yaw": host["yaw"][row],
                "arm_preset": host["preset"],
                "initial_arm_joints": host["joints"][row],
                "arm_joint_noise_rad": config.arm_joint_noise_rad,
                "distractors": [d for slot, d in enumerate(layout) if slot != selected_target],
                "basket_sampling": diagnostics,
                "scene_rejections": [],
                "scene_selection": scene_selection_metadata(seed, fixed=scene_id is not None),
                "lift_above_rim_m": config.lift_above_rim_m,
                "initial_admission_required": True,
            }
            episode["target_semantics"] = target_semantics(episode)
            results[seed] = episode
    return results
