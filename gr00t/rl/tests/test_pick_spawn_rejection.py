"""A room without a safe base cell is a scene rejection, not a batch crash."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick import basket, basket_preparation, external_preparation, prepare
from gr00t.rl.pick.randomization import NoSafeRobotSpawn
from gr00t.rl.pick.scene_selection import select_pick_scene


def test_external_rejects_selected_group_scene_without_switching_rooms(monkeypatch, tmp_path):
    records = [SimpleNamespace(scene_id=name, usd_path=tmp_path / name) for name in ("a", "b")]
    monkeypatch.setattr(external_preparation, "discover_vlnverse_scenes", lambda *a, **kw: records)
    monkeypatch.setattr(
        external_preparation, "episode_asset_pool", lambda *a, **kw: [{"name": "fruit"}]
    )
    monkeypatch.setattr(external_preparation, "materialize_variant", lambda asset, seed: asset)
    visited = []

    def sample(record, *args):
        visited.append(record.scene_id)
        raise NoSafeRobotSpawn(f"No safe robot spawn: {record.scene_id}")

    monkeypatch.setattr(external_preparation, "sample_candidates_batch", sample)
    with pytest.raises(prepare.NoValidInitialScene, match="No valid initial scene"):
        prepare.prepare_episode(
            56,
            tmp_path,
            tmp_path / "episode.json",
            scene_cache=SimpleNamespace(get=lambda *a: object()),
        )
    assert visited == [select_pick_scene(records, 56).scene_id]


def test_basket_rejects_selected_group_scene_without_switching_rooms(monkeypatch, tmp_path):
    records = [SimpleNamespace(scene_id=name, usd_path=tmp_path / name) for name in ("a", "b")]
    geometry = SimpleNamespace(boxes=torch.zeros(1))
    monkeypatch.setattr(basket.BasketGeometry, "from_robot_usd", lambda: geometry)
    monkeypatch.setattr(
        basket_preparation, "episode_asset_pool", lambda *a, **kw: [{"category": "fruit"}]
    )
    monkeypatch.setattr(basket_preparation, "sample_layouts", lambda *a, **kw: [([], 0, {})])
    monkeypatch.setattr(basket_preparation, "discover_vlnverse_scenes", lambda *a, **kw: records)
    visited = []

    def occupancy(record, *args):
        visited.append(record.scene_id)
        raise NoSafeRobotSpawn(f"No safe robot spawn: {record.scene_id}")

    monkeypatch.setattr(basket_preparation, "Occupancy", occupancy)
    with pytest.raises(
        prepare.NoValidInitialScene, match="no_clear_stationary_basket_spawn"
    ) as exc:
        basket.prepare_basket_episode(
            56,
            tmp_path,
            tmp_path / "episode.json",
            scene_cache=SimpleNamespace(get=lambda *a: object()),
        )
    assert visited == [select_pick_scene(records, 56).scene_id]
    assert exc.value.details["scene_rejections"] == visited
