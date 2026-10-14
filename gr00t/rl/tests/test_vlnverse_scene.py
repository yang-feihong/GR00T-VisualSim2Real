import json

import numpy as np
from PIL import Image

from gr00t.rl.simulator.isaacsim.vlnverse_scene import (
    discover_vlnverse_scenes,
    sample_scene_spawn_candidates,
    select_vlnverse_scenes,
)


def _make_scene(root, scene_id, *, with_navigation=True):
    scene_dir = root / scene_id
    scene_dir.mkdir()
    (scene_dir / "start_result_navigation.usd").write_text(
        '#usda 1.0\n(defaultPrim = "Root" metersPerUnit = 1 upAxis = "Z")\ndef Xform "Root" {}\n',
        encoding="utf-8",
    )
    if with_navigation:
        occupancy = np.full((21, 21), 255, dtype=np.uint8)
        occupancy[[0, -1], :] = 0
        occupancy[:, [0, -1]] = 0
        Image.fromarray(occupancy).save(scene_dir / "occupancy.png")
        (scene_dir / "occupancy.json").write_text(
            json.dumps(
                {
                    "scale": 0.1,
                    "lower": [-1.05, -1.05, 0.0],
                    "upper": [1.05, 1.05, 1.0],
                }
            ),
            encoding="utf-8",
        )
        (scene_dir / "rooms.json").write_text("[]", encoding="utf-8")
    return scene_dir


def test_discovery_filters_incomplete_scenes(tmp_path):
    _make_scene(tmp_path, "kujiale_0002", with_navigation=False)
    _make_scene(tmp_path, "kujiale_0001")
    records = discover_vlnverse_scenes(tmp_path)
    assert [record.scene_id for record in records] == ["kujiale_0001"]
    assert records[0].has_navigation_metadata


def test_scene_selection_is_deterministic_and_batch_prefix_stable(tmp_path):
    for index in range(5):
        _make_scene(tmp_path, f"kujiale_{index:04d}")
    records = discover_vlnverse_scenes(tmp_path)
    short = select_vlnverse_scenes(records, 3, seed=73)
    long = select_vlnverse_scenes(records, 12, seed=73)
    assert [record.scene_id for record in short] == [
        record.scene_id for record in long[:3]
    ]
    assert len({record.scene_id for record in long[:5]}) == 5
    assert [record.scene_id for record in long] == [
        record.scene_id for record in select_vlnverse_scenes(records, 12, seed=73)
    ]


def test_spawn_candidates_are_clear_and_deterministic(tmp_path):
    _make_scene(tmp_path, "kujiale_0001")
    record = discover_vlnverse_scenes(tmp_path)[0]
    first = sample_scene_spawn_candidates(
        record,
        count=64,
        seed=19,
        clearance_radius_m=0.2,
    )
    second = sample_scene_spawn_candidates(
        record,
        count=64,
        seed=19,
        clearance_radius_m=0.2,
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (64, 3)
    assert np.all(np.abs(first[:, :2]) <= 0.75)
    assert np.all(np.abs(first[:, 2]) <= np.pi)


def test_fixed_scene_selection_requires_known_id(tmp_path):
    _make_scene(tmp_path, "kujiale_0001")
    records = discover_vlnverse_scenes(tmp_path)
    selected = select_vlnverse_scenes(
        records,
        4,
        seed=0,
        mode="fixed",
        fixed_scene_id="kujiale_0001",
    )
    assert [record.scene_id for record in selected] == ["kujiale_0001"] * 4
