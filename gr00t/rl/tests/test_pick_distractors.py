"""Nearby object proposals preserve label uniqueness and admitted walking geometry."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.pick.distractors import (
    augment_scene_geometry,
    build_distractor_mesh,
    sample_nearby_objects,
    target_semantics,
)
from gr00t.rl.pick.external_initialization import ExternalAdmissionInputs, scene_object_checks_batch


class Surfaces:
    device = torch.device("cpu")

    def rays(self, origins, targets):
        table = origins[..., 0] > 0.45
        points = origins.clone()
        points[..., 2] = torch.where(table, 0.6, 0.0)
        normals = torch.zeros_like(points)
        normals[..., 2] = 1
        self.last_hit_types = table.long()
        return points, normals, (points - origins).norm(dim=-1)

    def visible(self, origins, targets, **kwargs):
        return torch.ones(origins.shape[:-1], dtype=torch.bool, device=origins.device)


def episode(seed=1700):
    return {
        "seed": seed,
        "asset": {"name": "014_lemon", "size_m": [0.06, 0.06, 0.05]},
        "object_position": [0.0, 0.0, 0.028],
        "support_type": "floor",
        "base_xy": [-2.0, 0.0],
        "base_yaw": 0.0,
        "station_xy": [-0.7, 0.0],
    }


def test_distractors_are_supported_separated_and_semantically_unambiguous():
    scene = episode()
    assets = [
        {"name": name, "size_m": [0.06, 0.06, 0.05]}
        for name in ("014_lemon", "015_peach", "018_plum", "012_strawberry")
    ]
    items, diagnostics = sample_nearby_objects(scene, assets, Surfaces(), count=4)
    assert diagnostics == {"requested_count": 4, "placed_count": 4}
    pairs = [(item["object_category"], item["support_type"]) for item in items]
    assert len(pairs) == len(set(pairs))
    assert ("lemon", "floor") not in pairs
    assert {item["support_type"] for item in items} == {"floor", "table"}
    positions = torch.tensor([item["object_position"] for item in items])
    distances = torch.cdist(positions, positions) + torch.eye(len(items)) * 10
    assert distances.min() > 0.12
    for item in items:
        assert item["object_position"][2] == pytest.approx(item["support_z"] + 0.028)
        assert item["object_position"][0] > -0.3
    assert target_semantics(scene) == {
        "instance_id": "target",
        "object_category": "lemon",
        "support_category": "floor",
    }


def test_same_object_category_can_only_reappear_on_a_different_support():
    scene = episode()
    items, diagnostics = sample_nearby_objects(scene, [scene["asset"]], Surfaces(), count=4)
    assert diagnostics["placed_count"] == 1
    assert items[0]["object_category"] == "lemon"
    assert items[0]["support_type"] == "table"


def test_distractor_sampling_is_seed_local_and_empty_count_is_valid():
    scene = episode()
    assets = [scene["asset"], {"name": "015_peach", "size_m": [0.06] * 3}]
    first = sample_nearby_objects(scene, assets, Surfaces(), count=3)
    torch.manual_seed(999)
    torch.rand(100)
    assert sample_nearby_objects(scene, assets, Surfaces(), count=3) == first
    assert sample_nearby_objects(scene, assets, SimpleNamespace(device="cpu"), count=0) == (
        [],
        {"requested_count": 0, "placed_count": 0},
    )
    with pytest.raises(ValueError):
        sample_nearby_objects(scene, assets, Surfaces(), count=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA planning path")
def test_distractor_queries_and_seed_local_selection_run_on_gpu():
    geometry = Surfaces()
    geometry.device = torch.device("cuda:0")
    scene = episode()
    assets = [scene["asset"], {"name": "015_peach", "size_m": [0.06] * 3}]
    first = sample_nearby_objects(scene, assets, geometry, count=3)
    assert first[1]["placed_count"] == 3
    torch.rand(128, device="cuda:0")
    assert sample_nearby_objects(scene, assets, geometry, count=3) == first
    assert geometry.last_hit_types.is_cuda


def test_resampling_preserves_installed_asset_slots_and_rechecks_actual_support():
    scene = episode()
    assets = [
        scene["asset"],
        {"name": "015_peach", "size_m": [0.06] * 3},
        {"name": "018_plum", "size_m": [0.06] * 3},
    ]
    items, diagnostics = sample_nearby_objects(
        scene, assets, Surfaces(), count=3, fixed_sequence=True
    )
    assert diagnostics["placed_count"] == 3
    assert [item["asset"]["name"] for item in items] == [asset["name"] for asset in assets]
    scene["distractors"] = items
    states = torch.zeros(1, len(items) + 1, 13)
    states[..., 3] = 1
    states[0, :, :3] = torch.tensor(
        [scene["object_position"], *[item["object_position"] for item in items]]
    )
    state = {
        "object_state": states[:, 0],
        "distractor_states": states[:, 1:],
        "root_states": torch.zeros(1, 13),
    }
    metadata = ExternalAdmissionInputs([scene], torch.zeros(1, 3), len(items) + 1)
    checks, _ = scene_object_checks_batch(state, Surfaces(), metadata)
    assert all(bool(value.all()) for value in checks.values())
    state["distractor_states"][0, 0, 0] = 0.1
    checks, _ = scene_object_checks_batch(state, Surfaces(), metadata)
    assert not bool(checks["object_support_labels_match"].all())
    assert not bool(checks["object_category_support_pairs_unique"].all())
    with pytest.raises(ValueError, match="fixed asset sequence"):
        sample_nearby_objects(scene, assets, Surfaces(), count=2, fixed_sequence=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU obstacle geometry")
def test_measured_distractors_are_real_gpu_planning_obstacles(tmp_path):
    from gr00t.rl.pick.scene_geometry import SceneGeometry

    vertices = np.array(
        [
            [-0.1, -0.2, -0.1],
            [-0.1, -0.2, 0.1],
            [-0.1, 0.2, -0.1],
            [-0.1, 0.2, 0.1],
            [0.1, -0.2, -0.1],
            [0.1, -0.2, 0.1],
            [0.1, 0.2, -0.1],
            [0.1, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int32,
    )
    np.save(tmp_path / "collision_support.npy", vertices)
    np.save(tmp_path / "collision_faces.npy", faces)
    states = torch.tensor(
        [[[0.5, 0, 0.2, math.sqrt(0.5), 0, 0, math.sqrt(0.5), 0, 0, 0, 0, 0, 0]]], device="cuda:0"
    )
    mesh = build_distractor_mesh([{"asset": {"usd_path": str(tmp_path / "object.usda")}}], states)
    torch.testing.assert_close(mesh["points"][0], states.new_tensor([0.7, -0.1, 0.1]))
    ground = SceneGeometry(
        np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], dtype=np.float32),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        labels=np.zeros(2, dtype=np.int32),
    )
    start, end = states.new_tensor([[0, 0, 0.2]]), states.new_tensor([[1, 0, 0.2]])
    assert bool(ground.visible(start, end).all())
    augmented = augment_scene_geometry(ground, mesh)
    assert augmented.points.is_cuda
    assert not bool(augmented.visible(start, end).all())
    assert augmented.last_hit_types.item() == 5
