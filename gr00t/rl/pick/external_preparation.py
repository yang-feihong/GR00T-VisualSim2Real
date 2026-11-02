"""Prepare a rank's external tasks with scene-grouped GPU placement queries."""

from pathlib import Path

import numpy as np

from gr00t.rl.pick.asset_library import episode_asset_pool, installed_assets, materialize_variant
from gr00t.rl.pick.distractors import sample_nearby_objects_batch, target_semantics
from gr00t.rl.pick.prepare import NoValidInitialScene
from gr00t.rl.pick.randomization import (
    NoSafeRobotSpawn,
    PickRandomizationConfig,
    sample_candidates_batch,
    select_and_serialize_batch,
)
from gr00t.rl.pick.scene_selection import scene_selection_metadata, select_pick_scene
from gr00t.rl.simulator.isaacsim.vlnverse_scene import discover_vlnverse_scenes


def prepare_external_batch(
    seeds,
    scenes_root,
    scene_cache,
    *,
    scene_id=None,
    support="any",
    asset_name=None,
    fixed_asset=None,
    distractor_assets=None,
):
    config = PickRandomizationConfig()
    records = discover_vlnverse_scenes(scenes_root, scene_ids=[scene_id] if scene_id else None)
    groups, results = {}, {}
    cache_root = Path(__file__).resolve().parents[3] / "datasets/pick/.cache/geometry"
    fixed = None
    if asset_name:
        fixed = next((a for a in installed_assets() if a["name"] == asset_name), None)
        if fixed is None:
            raise ValueError(f"Pick asset is not installed: {asset_name}")
    for seed in seeds:
        pool = (
            [fixed_asset, *distractor_assets]
            if fixed_asset is not None and distractor_assets is not None
            else episode_asset_pool(seed)
        )
        eligible = [a for a in pool if a.get("support_plane_admitted") is not False]
        if not eligible:
            results[seed] = NoValidInitialScene(
                "No assets with admitted support-plane grasps are installed"
            )
            continue
        random = np.random.default_rng(np.random.SeedSequence([seed, 0x5049434B]))
        if fixed_asset is not None:
            asset = dict(fixed_asset)
        else:
            source = fixed if fixed is not None else eligible[int(random.integers(len(eligible)))]
            asset = materialize_variant(source, seed)
        if asset.get("support_plane_admitted") is False:
            results[seed] = NoValidInitialScene(
                f"Asset has no admitted support-plane grasp: {asset['name']}"
            )
            continue
        if not records:
            results[seed] = NoValidInitialScene(f"No valid initial scene for seed={seed}")
            continue
        record = select_pick_scene(records, seed)
        groups.setdefault(record.scene_id, (record, []))[1].append(
            (seed, asset, pool, eligible, random)
        )
    for record, group in groups.values():
        geometry = scene_cache.get(record, cache_root)
        group_seeds = [seed for seed, *_ in group]
        assets = [asset for _, asset, *_ in group]
        try:
            candidates = sample_candidates_batch(record, assets, geometry, group_seeds, config)
            episodes, counts = select_and_serialize_batch(
                candidates, record, assets, group_seeds, [support] * len(group), config
            )
        except NoSafeRobotSpawn as error:
            for seed in group_seeds:
                results[seed] = NoValidInitialScene(f"No valid initial scene: {error}")
            continue
        accepted = []
        for index, (episode, context) in enumerate(zip(episodes, group, strict=True)):
            seed, _, pool, eligible, _ = context
            if episode is None:
                results[seed] = NoValidInitialScene(
                    "No valid initial scene in GPU placement batch",
                    {"scene": record.scene_id, "counts": {k: v[index] for k, v in counts.items()}},
                )
                continue
            episode.update(
                target_asset_sampling={
                    "policy": "support_plane_admitted_for_fully_supported_placements",
                    "eligible_instances": len(eligible),
                    "eligible_categories": len({a.get("category", a["name"]) for a in eligible}),
                    "non_target_assets_retained": True,
                },
                scene_rejections=[],
                geometry_cache_root=str(cache_root),
                scene_selection=scene_selection_metadata(seed, fixed=scene_id is not None),
                candidate_selection_policy="gpu_uniform_priority_v1",
            )
            accepted.append((episode, pool if distractor_assets is None else distractor_assets))
        if not accepted:
            continue
        objects = sample_nearby_objects_batch(
            [episode for episode, _ in accepted],
            [pool for _, pool in accepted],
            geometry,
            counts=None if distractor_assets is None else [len(distractor_assets)] * len(accepted),
            fixed_sequence=distractor_assets is not None,
        )
        for (episode, _), (distractors, sampling) in zip(accepted, objects, strict=True):
            seed = episode["seed"]
            if distractor_assets is not None and len(distractors) != len(distractor_assets):
                results[seed] = NoValidInitialScene(
                    "Installed distractor assets have no complete supported placement", sampling
                )
                continue
            for item in distractors:
                if "_variant_source" in item["asset"]:
                    item["asset"] = materialize_variant(item["asset"], seed)
            episode.update(
                target_semantics=target_semantics(episode),
                distractors=distractors,
                distractor_sampling=sampling,
            )
            results[seed] = episode
    return results
