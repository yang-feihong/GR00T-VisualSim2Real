"""Bounded preferred-pose contact fitting with real shapes and full room geometry."""

import json
import os
import time
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


@pytest.mark.skipif(not os.environ.get("PICK_PREFERRED_BANK_ROOT"), reason="Records required")
def test_preferred_bank_hundred_rows_matches_singletons(tmp_path):
    from gr00t.rl.pick.planning import grasp_candidate_bank

    root = Path(os.environ["PICK_PREFERRED_BANK_ROOT"])
    state, _, sizes, support, mask, geometry, collision, _ = _recorded_batch(root, count=4)
    paths = sorted(root.glob("logs/rank*-control/seed_*/summary.json"))
    preferred = []
    for path in paths:
        summary = json.loads(path.read_text())
        attempt = summary["planning"].get("grasp_attempts", [summary["planning"]["grasp"]])[0]
        preferred.append(attempt["candidate_orientations_world_wxyz"][0][0])
    preferred = support.new_tensor(preferred)

    def solve(rows):
        local = {key: value[rows] for key, value in state.items()}
        return grasp_candidate_bank(
            local,
            local["object_state"][:, :3],
            sizes[rows],
            collision,
            geometry.select(rows),
            support[rows],
            support_mask=mask[rows],
            candidate_orientations=preferred[rows, None],
        )

    rows = torch.arange(100, device="cuda") % 4
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        bank = solve(rows)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    for row in range(4):
        reference = solve(rows.new_tensor([row]))
        torch.testing.assert_close(bank[1][row : row + 1], reference[1], atol=1e-6, rtol=0)
        for key in ("pocket_fit", "aperture_angle_rad", "antipodal_margin"):
            torch.testing.assert_close(
                bank[2][key][row : row + 1], reference[2][key], atol=1e-6, rtol=0, equal_nan=True
            )
    report = {
        "physics_simulated": False,
        "cloned_recorded_rows": 100,
        "contact_bank_s": elapsed,
        "maximum_contact_points": 9984,
        "singleton_goal_atol_m": 1e-6,
    }
    (tmp_path / "preferred_bank_timing.json").write_text(json.dumps(report, indent=2))
    print(report)
