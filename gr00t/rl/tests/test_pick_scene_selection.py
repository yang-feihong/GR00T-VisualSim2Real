"""Twenty independently randomized episodes share one deterministic static scene."""

from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from gr00t.rl.pick.scene_selection import (
    AdmissionSceneCache,
    PreparationSceneCache,
    scene_selection_metadata,
    select_pick_scene,
)


def records(count=12):
    return [
        SimpleNamespace(scene_id=f"room_{index:03d}", usd_path=f"room_{index:03d}.usd")
        for index in range(count)
    ]


@pytest.mark.parametrize("start", [0, 1700, 1000000, 2000000])
def test_consecutive_hundred_seeds_use_five_group_scenes(start):
    catalog = records()
    selected = [select_pick_scene(catalog, seed).scene_id for seed in range(start, start + 100)]
    for group in range(5):
        assert len(set(selected[group * 20 : (group + 1) * 20])) == 1
    assert len(set(selected)) == 5


def test_selection_is_independent_of_rank_order_catalog_order_and_task_rng():
    catalog = records()
    expected = {seed: select_pick_scene(catalog, seed).scene_id for seed in range(100)}
    for rank in range(4):
        for seed in reversed(range(rank, 100, 4)):
            assert select_pick_scene(list(reversed(catalog)), seed).scene_id == expected[seed]
    random = np.random.default_rng(123)
    reference = np.random.default_rng(123)
    select_pick_scene(catalog, 45)
    assert random.random() == reference.random()
    assert scene_selection_metadata(39) == {
        "policy": "seed_group",
        "scene_seed": 1,
        "episodes_per_scene": 20,
    }
    assert scene_selection_metadata(40)["scene_seed"] == 2
    assert select_pick_scene(catalog[:1], 123456).scene_id == catalog[0].scene_id


def test_preparation_reuses_gpu_geometry_without_unbounded_room_storage(monkeypatch):
    from gr00t.rl.pick import scene_geometry

    reads, builds = [], []

    def read(path, cache_root, *, return_labels):
        assert return_labels
        reads.append(path)
        return path, "faces", "labels"

    def build(points, faces, *, labels, ground_plane_z_m):
        assert ground_plane_z_m == 0.0
        result = SimpleNamespace(path=points)
        builds.append(result)
        return result

    monkeypatch.setattr(scene_geometry, "read_scene_triangles", read)
    monkeypatch.setattr(scene_geometry, "static_scene_geometry", build)
    cache = PreparationSceneCache()
    for seed in range(100):
        record = select_pick_scene(records(), seed)
        geometry = cache.get(record, "unused")
        assert geometry.path == record.usd_path
        assert cache.get(record, "unused") is geometry
    assert len(reads) == len(builds) == 5
    assert cache.geometry is builds[-1]


def test_full_scene_preserves_one_asset_path_across_different_layouts(tmp_path):
    source = tmp_path / "scene.usdc"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Room").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Cube.Define(stage, "/Room/Table")
    stage.GetRootLayer().Save()
    assembled = Usd.Stage.CreateInMemory()
    for seed in range(20):
        episode = {"seed": seed, "scene_usd_path": str(source), "base_xy": [seed, seed]}
        actor = UsdGeom.Xform.Define(assembled, f"/World/env_{seed}/Scene")
        actor.GetPrim().GetReferences().AddReference(episode["scene_usd_path"])
        actor.AddTranslateOp().Set(Gf.Vec3d(seed * 32, 0, 0))
        UsdGeom.Cube.Define(assembled, f"/World/env_{seed}/Object").AddTranslateOp().Set(
            Gf.Vec3d(seed * 32, seed / 100, 1)
        )
    assert sum(layer.realPath == str(source) for layer in assembled.GetUsedLayers()) == 1
    assert len(assembled.GetPrimAtPath("/World").GetChildren()) == 20
    for seed in range(20):
        assert assembled.GetPrimAtPath(f"/World/env_{seed}/Object")


def test_legacy_cropped_descriptor_cannot_override_full_scene():
    from gr00t.rl.pick.prepare_batch import local_episode_descriptor

    episode = {
        "seed": 20,
        "scene_usd_path": "original.usd",
        "physics_scene_usd_path": "cropped.usd",
        "planning_scene_usd_path": "cropped.usd",
        "scene_scope": {"mode": "rooms"},
    }
    assert local_episode_descriptor(episode) == {
        "seed": 20,
        "scene_usd_path": "original.usd",
    }
    assert episode["physics_scene_usd_path"] == "cropped.usd"


def test_admission_borrows_scene_local_geometry_without_rebuild():
    source = object()
    cache = AdmissionSceneCache(
        [{"scene_usd_path": "room.usd"}] * 2,
        [SimpleNamespace(source=source), SimpleNamespace(source=source)],
    )
    record = SimpleNamespace(usd_path="room.usd")
    assert cache.get(record, "unused") is source
    assert cache.get(record, "unused") is source
    with pytest.raises(KeyError):
        cache.get(SimpleNamespace(usd_path="other.usd"), "unused")


def test_proposal_inputs_are_cached_per_scene_and_config(monkeypatch):
    import torch

    from gr00t.rl.pick import randomization

    calls = []

    class Occupancy:
        def __init__(self, record, device, config):
            calls.append("occupancy")
            self.station_positions = torch.zeros(1, 2)

        def free(self, xy, **kwargs):
            return torch.ones(xy.shape[:-1], dtype=torch.bool)

    class Geometry:
        device = torch.device("cpu")

        def rays(self, start, end):
            hit = start.clone()
            hit[..., 2] = 0
            normal = torch.zeros_like(hit)
            normal[..., 2] = 1
            self.last_hit_types = torch.zeros(hit.shape[:-1], dtype=torch.long)
            return hit, normal, (hit - start).norm(dim=-1)

        def visible(self, start, end):
            return torch.ones(start.shape[:-1], dtype=torch.bool)

    def regions(path):
        calls.append("regions")
        return np.zeros((0, 4), dtype=np.float32)

    monkeypatch.setattr(randomization, "Occupancy", Occupancy)
    monkeypatch.setattr(randomization, "read_support_regions", regions)
    record, geometry = SimpleNamespace(usd_path="room.usd"), Geometry()
    config = randomization.PickRandomizationConfig(candidates=16)
    asset = {"size_m": [0.04, 0.04, 0.04]}
    first = randomization.sample_candidates(record, asset, geometry, 19, config)
    again = randomization.sample_candidates(record, asset, geometry, 19, config)
    assert calls == ["occupancy", "regions"]
    for name, value in first.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, again[name], rtol=0, atol=0)


def test_embedded_physics_scan_is_once_per_unique_asset_not_per_environment(monkeypatch):
    from pxr import UsdPhysics

    from gr00t.rl.pick.scene_selection import disable_embedded_physics_scenes

    stage = Usd.Stage.CreateInMemory()
    roots = []
    for row in range(40):
        root = f"/World/env_{row}/Room"
        UsdGeom.Xform.Define(stage, root)
        UsdPhysics.Scene.Define(stage, f"{root}/Physics")
        UsdGeom.Cube.Define(stage, f"{root}/Wall")
        roots.append((root, f"asset_{row // 20}"))
    original, calls = Usd.PrimRange, []

    def tracked(root):
        calls.append(root.GetPath())
        return original(root)

    monkeypatch.setattr(Usd, "PrimRange", tracked)
    disable_embedded_physics_scenes(stage, roots)
    assert len(calls) == 2
    for root, _ in roots:
        assert not stage.GetPrimAtPath(f"{root}/Physics").IsActive()
        assert stage.GetPrimAtPath(f"{root}/Wall").IsActive()
