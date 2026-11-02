"""Counterfactual grasp refinement on real observations through production GPU helpers."""

import json
import os
import time
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


@pytest.mark.skipif(
    not os.environ.get("PICK_GRASP_REFINEMENT_ROOT"), reason="Recorded probe required"
)
def test_recorded_grasp_refinement_realized_body_candidates(tmp_path):
    from gr00t.rl.pick.grasp_inverse import solve_default_grasp_inverse
    from gr00t.rl.pick.grasp_paths import score_default_grasp_paths
    from gr00t.rl.pick.planning import grasp_candidate_bank

    state, _, sizes, support, mask, geometry, collision, limits = _recorded_batch(
        Path(os.environ["PICK_GRASP_REFINEMENT_ROOT"]), count=4, frame=-1
    )
    report = {
        "seeds": state["episode_seed"].tolist(),
        "physics_simulated": False,
        "reference": "recorded terminal EE orientation, no candidate exclusions",
        "cases": {},
    }
    for refined in (False, True):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            bank = grasp_candidate_bank(
                state,
                state["object_state"][:, :3],
                sizes,
                collision,
                geometry,
                support,
                support_mask=mask,
                preferred_orientation=state["ee_orn"],
                refine_orientations=refined,
            )
            inverse = solve_default_grasp_inverse(
                state, bank, limits, collision.arm_joint_ids, state["object_state"][:, 2], 0.02
            )
            path, valid, detail = score_default_grasp_paths(
                state,
                bank,
                inverse,
                limits,
                collision,
                geometry,
                0.02,
                prefer_realized_body=torch.ones(4, device="cuda", dtype=torch.bool),
            )
        torch.cuda.synchronize()
        actual = detail.get(
            "realized_body_grasp_candidate_mask", torch.zeros_like(detail["feasible"])
        )
        report["cases"][str(refined)] = {
            "planning_s": time.perf_counter() - started,
            "candidate_count": bank[0].shape[1],
            "valid": valid.tolist(),
            "selected": detail["selected"].tolist(),
            "safe_candidates": detail["feasible"].sum(-1).tolist(),
            "realized_body_safe_candidates": (actual & detail["feasible"]).sum(-1).tolist(),
            "selected_realized_body": actual.gather(1, detail["selected"].clamp_min(0)[:, None])[
                :, 0
            ].tolist(),
            "selected_path_end_world_m": path[0][torch.arange(4), (path[2] - 1).long()].tolist(),
        }
    (tmp_path / "recorded_grasp_refinement.json").write_text(json.dumps(report, indent=2) + "\n")
