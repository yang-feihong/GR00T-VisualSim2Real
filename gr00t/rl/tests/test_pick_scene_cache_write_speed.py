"""Measure lossless static-cache serialization on the actual initialization rooms."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get("PICK_CACHE_BENCH_SCENES"), reason="Recorded rooms required")
def test_recorded_room_cache_compression_cost(tmp_path):
    from gr00t.rl.pick.scene_geometry import _read_scene_cache, read_scene_triangles

    prepared = json.loads(Path(os.environ["PICK_CACHE_BENCH_SCENES"]).read_text())
    episodes = [row["episode"] for row in prepared if "episode" in row]
    unique = {episode["scene_usd_path"]: episode for episode in episodes}
    report = {"physics_simulated": False, "rooms": [], "totals_s": {}}
    for index, (path, episode) in enumerate(unique.items()):
        arrays = read_scene_triangles(
            Path(path), Path(episode["geometry_cache_root"]), return_labels=True
        )
        detail = {"scene_id": episode["scene_id"], "array_bytes": sum(a.nbytes for a in arrays)}
        for name, writer in (("deflate", np.savez_compressed), ("stored", np.savez)):
            output = tmp_path / f"room_{index}_{name}.npz"
            started = time.perf_counter()
            writer(output, points=arrays[0], faces=arrays[1], labels=arrays[2])
            detail[f"{name}_write_s"] = time.perf_counter() - started
            detail[f"{name}_file_bytes"] = output.stat().st_size
            started = time.perf_counter()
            recovered = _read_scene_cache(output)
            detail[f"{name}_read_s"] = time.perf_counter() - started
            for actual, expected in zip(recovered, arrays, strict=True):
                assert actual.dtype == expected.dtype
                np.testing.assert_array_equal(actual, expected)
            del recovered
        report["rooms"].append(detail)
    for name in ("deflate_write_s", "stored_write_s", "deflate_read_s", "stored_read_s"):
        report["totals_s"][name] = sum(room[name] for room in report["rooms"])
    (tmp_path / "recorded_scene_cache_io.json").write_text(json.dumps(report, indent=2) + "\n")
