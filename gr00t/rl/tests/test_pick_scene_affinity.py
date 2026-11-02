"""Scene grouping reads asset metadata only and never excludes failed seed rows."""

import hashlib
import json

import numpy as np
import pytest

from gr00t.rl.pick.scene_affinity import build_manifests


def test_manifest_preserves_full_descriptors_rejections_and_snapshot_paths(tmp_path):
    source = tmp_path / "room.usdc"
    source.write_text("test scene identity; geometry read from cached NPY header")
    cache = tmp_path / "geometry"
    cache.mkdir()
    identity = f"{source}:{source.stat().st_mtime_ns}:collision-v4-labels"
    np.savez(
        cache / (hashlib.sha256(identity.encode()).hexdigest()[:20] + ".npz"),
        faces=np.zeros((123, 3), dtype=np.int32),
    )
    episodes = [
        {
            "seed": seed,
            "episode": {
                "seed": seed,
                "scene_usd_path": str(source),
                "geometry_cache_root": str(cache),
                "base_xy": [seed, -seed],
                "object_position": [0, 1, 2],
            },
        }
        for seed in (0, 1)
    ]
    episodes.append({"seed": 2, "rejection": {"reason": "no_valid_initial_scene"}})
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps(episodes))
    output = tmp_path / "snapshot"
    result = build_manifests([prepared], [0, 1, 2], output)
    assert result["0"]["weight"] == result["1"]["weight"] == 123
    assert result["2"]["weight"] == 0
    paths = json.loads((output / "episodes.json").read_text())
    for row in episodes:
        assert json.loads((output / "episodes" / f"seed_{row['seed']}.json").read_text()) == row
        assert paths[str(row["seed"])].endswith(f"episodes/seed_{row['seed']}.json")
    with pytest.raises(FileExistsError):
        build_manifests([prepared], [0, 1], output)
    with pytest.raises(ValueError, match="Missing"):
        build_manifests([prepared], [5], tmp_path / "missing")


def test_shared_preparer_accepts_snapshot_rows_without_resampling_rejections(tmp_path, monkeypatch):
    from gr00t.rl.pick import prepare_batch

    episode = {"seed": 0, "scene_usd_path": "unused.usdc"}
    descriptors = [
        {"seed": 0, "episode": episode},
        {"seed": 1, "rejection": {"reason": "initial_sampling_failed", "details": {"count": 0}}},
    ]
    paths = {}
    for row in descriptors:
        path = tmp_path / f"source_{row['seed']}.json"
        path.write_text(json.dumps(row))
        paths[str(row["seed"])] = str(path)
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps(paths))
    output = tmp_path / "prepared.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit episodes and rejection records must not be resampled")

    monkeypatch.setattr(prepare_batch, "prepare_external_batch", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_batch",
            "--seeds",
            "0",
            "1",
            "--task",
            "external",
            "--output",
            str(output),
            "--episodes-manifest",
            str(manifest),
        ],
    )
    prepare_batch.main()
    actual = json.loads(output.read_text())
    assert actual[0]["episode"] == episode
    assert actual[1] == descriptors[1]
