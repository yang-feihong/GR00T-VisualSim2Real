"""Opt-in recorded-input benchmark for batching, without stepping simulation."""

import json
import os
import time
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


@pytest.mark.skipif(not os.environ.get("PICK_STATION_TIMING_ROOT"), reason="Records required")
def test_recorded_station_throughput(tmp_path):
    from gr00t.rl.pick.station_batch import select_initial_pick_station_batch

    state, destination, sizes, support, mask, geometry, collision, limits = _recorded_batch(
        Path(os.environ["PICK_STATION_TIMING_ROOT"]), count=4
    )
    rows = torch.arange(100, device="cuda").remainder(4)
    results, report = [], []
    for chunk in (32, 100):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            result = select_initial_pick_station_batch(
                {key: value[rows] for key, value in state.items()},
                destination[rows],
                sizes[rows],
                support[rows],
                limits,
                collision,
                geometry.select(rows),
                support_mask=mask[rows],
                max_scenes=chunk,
            )
        torch.cuda.synchronize()
        report.append(
            {
                "chunk": chunk,
                "elapsed_s": time.perf_counter() - started,
                "peak_bytes": torch.cuda.max_memory_allocated(),
                "valid": int(result[3].sum()),
            }
        )
        (tmp_path / "station_throughput.json").write_text(json.dumps(report, indent=2))
        print(report[-1], flush=True)
        results.append(result)
    for value, reference in zip(results[1][:4], results[0][:4], strict=True):
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=0, equal_nan=True)
    for key in (
        "selected",
        "selected_candidate",
        "selected_arrival_coverage",
        "selected_endpoint_joints_rad",
    ):
        torch.testing.assert_close(
            results[1][4][key], results[0][4][key], atol=1e-6, rtol=0, equal_nan=True
        )
    (tmp_path / "station_throughput.json").write_text(json.dumps(report, indent=2))
    print(report)
