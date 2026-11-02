"""Spatial preparation stays batched, scene-local, and independent of partner seeds."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.pick import randomization
from gr00t.rl.pick.basket import BasketConfig, BasketGeometry
from gr00t.rl.pick.basket_initialization import resample_basket_payloads
from gr00t.rl.pick.basket_layout import sample_layouts
from gr00t.rl.pick.distractors import sample_nearby_objects_batch
from gr00t.rl.tests.test_pick_distractors import Surfaces, episode


def test_preparation_subprocess_bootstraps_lowlevel_without_test_paths():
    root = Path(__file__).resolve().parents[3]
    dependencies = [
        path
        for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if path and "visual_wholebody_lowlevel" not in path
    ]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), *dependencies]))
    environment.pop("AUTO_OPEN_MANAGED_RANK_PARENT_PID", None)
    result = subprocess.run(
        [sys.executable, "-m", "gr00t.rl.pick.prepare_batch", "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--seeds" in result.stdout


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("Production GPU spatial preparation")
    return torch.device("cuda:0")


@pytest.mark.skipif(not os.environ.get("PICK_REAL_PREPARE"), reason="Real assets required")
def test_real_external_preparation_batch(tmp_path, device):
    from gr00t.rl.pick.external_preparation import prepare_external_batch
    from gr00t.rl.pick.prepare import NoValidInitialScene
    from gr00t.rl.pick.scene_selection import PreparationSceneCache

    start = int(os.environ.get("PICK_PREPARE_START", "0"))
    count = int(os.environ.get("PICK_PREPARE_COUNT", "100"))
    started = time.perf_counter()
    results = prepare_external_batch(
        list(range(start, start + count)),
        Path("/datasets/vlnverse_scenes"),
        PreparationSceneCache(),
    )
    torch.cuda.synchronize()
    accepted = {
        seed: value for seed, value in results.items() if not isinstance(value, NoValidInitialScene)
    }
    report = {
        "elapsed_s": time.perf_counter() - started,
        "requested": count,
        "accepted": len(accepted),
        "rejections": {
            seed: {"reason": str(value), "details": value.details}
            for seed, value in results.items()
            if seed not in accepted
        },
    }
    (tmp_path / "preparation_report.json").write_text(json.dumps(report, indent=2))
    (tmp_path / "prepared_episodes.json").write_text(json.dumps(accepted, indent=2))
    print(report)
    assert len(results) == count
    assert len(accepted) == count, report


def proposal_fixture(monkeypatch, device):
    class Occupancy:
        def __init__(self, record, device, config):
            self.station_positions = torch.zeros(1, 2, device=device)

        def free(self, xy, **kwargs):
            return torch.ones(xy.shape[:-1], device=xy.device, dtype=torch.bool)

    class Geometry(Surfaces):
        def __init__(self):
            self.device = device
            self.ray_queries = self.visibility_queries = 0

        def rays(self, origins, ends):
            self.ray_queries += 1
            hit = origins.clone()
            hit[..., 2] = 0
            normal = torch.zeros_like(hit)
            normal[..., 2] = 1
            self.last_hit_types = torch.zeros(hit.shape[:-1], device=device, dtype=torch.long)
            return hit, normal, (hit - origins).norm(dim=-1)

        def visible(self, origins, ends, **kwargs):
            self.visibility_queries += 1
            return torch.ones(origins.shape[:-1], device=device, dtype=torch.bool)

    monkeypatch.setattr(randomization, "Occupancy", Occupancy)
    monkeypatch.setattr(
        randomization, "read_support_regions", lambda path: np.zeros((0, 4), np.float32)
    )
    return SimpleNamespace(scene_id="room", usd_path="room.usd"), Geometry()


def test_scene_proposals_merge_queries_and_preserve_every_seed(monkeypatch, device):
    record, geometry = proposal_fixture(monkeypatch, device)
    seeds = [1700, 1701, 1702]
    assets = [
        {"name": str(seed), "size_m": [0.04 + index * 0.01] * 3} for index, seed in enumerate(seeds)
    ]
    config = randomization.PickRandomizationConfig(candidates=256)
    combined = randomization.sample_candidates_batch(record, assets, geometry, seeds, config)
    assert geometry.ray_queries == 4 and geometry.visibility_queries == 2
    for row, (asset, seed) in enumerate(zip(assets, seeds, strict=True)):
        single = randomization.sample_candidates(record, asset, geometry, seed, config)
        for key, value in single.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, combined[row][key], rtol=0, atol=0)
            elif key == "checks":
                for name, mask in value.items():
                    torch.testing.assert_close(mask, combined[row][key][name], rtol=0, atol=0)
    original = randomization.select_and_serialize_batch(
        combined, record, assets, seeds, ["floor"] * len(seeds), config
    )[0]
    assert all(row is not None for row in original)
    order = [2, 0, 1]
    reordered = randomization.select_and_serialize_batch(
        [combined[i] for i in order],
        record,
        [assets[i] for i in order],
        [seeds[i] for i in order],
        ["floor"] * len(seeds),
        config,
    )[0]
    assert reordered == [original[i] for i in order]
    for row in range(len(seeds)):
        single = randomization.select_and_serialize_batch(
            [combined[row]], record, [assets[row]], [seeds[row]], ["floor"], config
        )[0]
        assert single[0] == original[row]


def test_distractor_batch_mixed_counts_assets_and_permutations(device):
    geometry = Surfaces()
    geometry.device = device
    episodes = [episode(seed) for seed in [1700, 1701, 1702]]
    assets = [
        {"name": name, "size_m": [0.06] * 3}
        for name in ["014_lemon", "015_peach", "018_plum", "012_strawberry"]
    ]
    asset_rows = [[], assets[:2], assets]
    counts = [0, 2, 4]
    combined = sample_nearby_objects_batch(episodes, asset_rows, geometry, counts=counts)
    for row in range(3):
        single = sample_nearby_objects_batch(
            episodes[row : row + 1],
            asset_rows[row : row + 1],
            geometry,
            counts=counts[row : row + 1],
        )
        assert single[0] == combined[row]
    order = [2, 0, 1]
    reordered = sample_nearby_objects_batch(
        [episodes[i] for i in order],
        [asset_rows[i] for i in order],
        geometry,
        counts=[counts[i] for i in order],
    )
    assert reordered == [combined[i] for i in order]


def test_basket_layout_mixed_counts_preserve_seed_results(device):
    basket = BasketGeometry.from_robot_usd(device=str(device))
    assets = [{"name": f"object_{i}", "size_m": [0.035] * 3} for i in range(4)]
    asset_rows = [assets[:1], assets[:2], assets]
    seeds, counts = [1700, 1701, 1702], [1, 2, 4]
    config = BasketConfig()
    combined = sample_layouts(asset_rows, basket, seeds, config, counts=counts)
    assert all(not isinstance(result, Exception) for result in combined)
    for row in range(3):
        single = sample_layouts(
            asset_rows[row : row + 1],
            basket,
            seeds[row : row + 1],
            config,
            counts=counts[row : row + 1],
        )
        assert single[0] == combined[row]
    order = [2, 0, 1]
    reordered = sample_layouts(
        [asset_rows[i] for i in order],
        basket,
        [seeds[i] for i in order],
        config,
        counts=[counts[i] for i in order],
    )
    assert reordered == [combined[i] for i in order]


def test_basket_resampling_preserves_slots_and_local_origins(device):
    basket = BasketGeometry.from_robot_usd(device=str(device))
    assets = [{"name": f"object_{i}", "size_m": [0.035] * 3} for i in range(4)]
    episodes = []
    for count, seed in [(1, 1700), (4, 1701)]:
        layout, target, diagnostics = sample_layouts(
            [assets[:count]], basket, [seed], BasketConfig(), counts=[count]
        )[0]
        for item in layout:
            item["support_z"] = 0.55 + item["support_z_local"]
        episodes.append(
            {
                **layout[target],
                "seed": seed,
                "base_xy": [10, -20],
                "base_yaw": 0.4,
                "distractors": [d for i, d in enumerate(layout) if i != target],
                "basket_sampling": diagnostics,
            }
        )
    combined = resample_basket_payloads(episodes, [0, 1], basket, 1)
    for row in range(2):
        single = resample_basket_payloads(episodes, [row], basket, 1)
        assert single[row] == combined[row]
        assert combined[row]["asset"] == episodes[row]["asset"]
        assert [d["asset"] for d in combined[row]["distractors"]] == [
            d["asset"] for d in episodes[row]["distractors"]
        ]
    assert resample_basket_payloads(episodes, [1, 0], basket, 1) == combined


@pytest.mark.skipif(
    not os.environ.get("PICK_GPU_AUDIT_MANIFEST"), reason="Saved real-scene input required"
)
@pytest.mark.parametrize("backend", ["native", "packet"])
def test_real_saved_twenty_seed_resampling_batch_and_split(device, tmp_path, backend):
    from gr00t.rl.pick.prepare import NoValidInitialScene
    from gr00t.rl.pick.scene_geometry import read_scene_triangles, static_scene_geometry
    from gr00t.rl.pick.scene_selection import AdmissionSceneCache
    from gr00t.rl.pick.task_initialization import _external_resample_proposals

    manifest = Path(os.environ["PICK_GPU_AUDIT_MANIFEST"])
    records = json.loads(manifest.read_text())
    first = next(row["episode"] for row in records if "episode" in row)
    episodes = [
        row["episode"]
        for row in records
        if "episode" in row and row["episode"]["scene_usd_path"] == first["scene_usd_path"]
    ][:20]
    assert len(episodes) == 20
    started = time.perf_counter()
    points, faces, labels = read_scene_triangles(
        Path(first["scene_usd_path"]), Path(first["geometry_cache_root"]), return_labels=True
    )
    geometry = static_scene_geometry(points, faces, str(device), labels=labels, backend=backend)
    cache = AdmissionSceneCache(episodes, [SimpleNamespace(source=geometry)] * len(episodes))
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    # Warm-up is not included in the GPU search timing. No simulator is started.
    _external_resample_proposals(episodes, [0], cache, 1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    combined = _external_resample_proposals(episodes, list(range(20)), cache, 1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    for row in (0, 7, 19):
        single = _external_resample_proposals(episodes, [row], cache, 1)[row]
        if isinstance(single, NoValidInitialScene):
            assert isinstance(combined[row], NoValidInitialScene)
            assert str(single) == str(combined[row])
            assert single.details == combined[row].details
        else:
            assert single == combined[row]
    report = {
        "manifest": str(manifest),
        "scene_id": first["scene_id"],
        "geometry_backend": backend,
        "episodes": 20,
        "scene_load_s": load_s,
        "warm_twenty_seed_resampling_s": elapsed,
        "proposals_accepted": sum(
            not isinstance(value, NoValidInitialScene) for value in combined.values()
        ),
        "split_rows_checked": [0, 7, 19],
        "simulator_started": False,
        "not_end_to_end_initialization_timing": True,
        "rejections": {
            str(episodes[row]["seed"]): {"reason": str(value), "details": value.details}
            for row, value in combined.items()
            if isinstance(value, NoValidInitialScene)
        },
    }
    (tmp_path / "real_twenty_seed_resampling.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


def test_initial_external_preparation_uses_one_spatial_batch_for_one_hundred_seeds(
    monkeypatch, device
):
    from gr00t.rl.pick import external_preparation

    record, geometry = proposal_fixture(monkeypatch, device)
    asset = {
        "name": "fruit",
        "category": "fruit",
        "size_m": [0.04] * 3,
        "usd_path": "fruit.usd",
        "support_plane_admitted": True,
    }
    real_features = randomization.asset_support_features
    monkeypatch.setattr(
        randomization,
        "asset_support_features",
        lambda assets, geometry: real_features(
            [{"size_m": item["size_m"]} for item in assets], geometry
        ),
    )
    monkeypatch.setattr(external_preparation, "discover_vlnverse_scenes", lambda *a, **kw: [record])
    monkeypatch.setattr(external_preparation, "episode_asset_pool", lambda seed: [asset])
    monkeypatch.setattr(
        external_preparation, "materialize_variant", lambda asset, seed: dict(asset)
    )
    monkeypatch.setattr(
        external_preparation,
        "PickRandomizationConfig",
        lambda: randomization.PickRandomizationConfig(candidates=128),
    )
    cache = SimpleNamespace(get=lambda *args: geometry)
    combined = external_preparation.prepare_external_batch(list(range(100)), None, cache)
    assert len(combined) == 100
    assert all(isinstance(value, dict) for value in combined.values())
    assert geometry.ray_queries == 6
    for row in (0, 47, 99):
        single = external_preparation.prepare_external_batch([row], None, cache)
        assert single[row] == combined[row]


def test_initial_basket_preparation_is_batch_independent(monkeypatch, device):
    from gr00t.rl.pick import basket_preparation

    record, geometry = proposal_fixture(monkeypatch, device)
    basket = BasketGeometry.from_robot_usd(device=str(device))
    assets = [
        {"name": f"fruit_{i}", "category": f"fruit_{i}", "size_m": [0.035] * 3} for i in range(4)
    ]
    monkeypatch.setattr(basket_preparation.BasketGeometry, "from_robot_usd", lambda: basket)
    monkeypatch.setattr(basket_preparation, "episode_asset_pool", lambda *a, **kw: assets)
    monkeypatch.setattr(basket_preparation, "materialize_variant", lambda asset, *a: dict(asset))
    monkeypatch.setattr(basket_preparation, "discover_vlnverse_scenes", lambda *a, **kw: [record])
    monkeypatch.setattr(
        basket_preparation,
        "Occupancy",
        lambda *args: SimpleNamespace(
            positions=torch.tensor([[0.0, 0.0], [1.0, 2.0]], device=device)
        ),
    )
    cache = SimpleNamespace(get=lambda *args: geometry)
    seeds = [1700, 1701, 1702, 1703]
    combined = basket_preparation.prepare_basket_batch(seeds, None, cache)
    assert all(isinstance(value, dict) for value in combined.values())
    assert geometry.ray_queries == 1 and geometry.visibility_queries == 1
    for seed in seeds:
        assert basket_preparation.prepare_basket_batch([seed], None, cache)[seed] == combined[seed]
