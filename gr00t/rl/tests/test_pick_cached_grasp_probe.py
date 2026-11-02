"""Offline prototype for revalidating a previous grasp before a full search."""

import json
import os
import time

import pytest
import torch

from gr00t.rl.tests.test_pick_basket_planning_throughput import _basket_inputs


@pytest.mark.skipif(not os.environ.get("PICK_BASKET_TIMING_ROOT"), reason="Records required")
def test_real_cached_admission_api_keeps_robust_checks():
    from gr00t.rl.pick.basket_grasp import plan_basket_grasp_batch
    from gr00t.rl.pick.basket_initialization import basket_grasp_access_batch

    state, sizes, support, mask, geometry, collision, limits, minimum = _basket_inputs()
    rows = torch.arange(4, device=sizes.device)
    state = {key: value[:4] for key, value in state.items()}
    geometry = geometry.select(rows)
    with torch.inference_mode():
        _, baseline_valid, baseline = plan_basket_grasp_batch(
            state,
            sizes[:4],
            limits,
            collision,
            geometry,
            0.02,
            support_points=support[:4],
            support_mask=mask[:4],
            initial_height=state["object_state"][:, 2],
            minimum_height=minimum[:4],
            prefer_realized_body=torch.zeros(4, device=sizes.device, dtype=torch.bool),
            realized_body=torch.ones(4, device=sizes.device, dtype=torch.bool),
            minimum_realized_grasp_margin_rad=0.03490658503988659,
        )
        assert baseline_valid.all()
        orientation = baseline["entry_orientation_world_wxyz"].clone()
        slots = state["distractor_states"].shape[1] + 1

        def run():
            return basket_grasp_access_batch(
                state,
                sizes[:4],
                sizes[:4, None].expand(-1, slots, -1),
                torch.ones(4, slots, device=sizes.device, dtype=torch.bool),
                geometry.basket_model,
                collision,
                geometry,
                geometry,
                geometry,
                support[:4],
                mask[:4],
                sizes.new_zeros(4, 3),
                limits,
                0.02,
                lift_margin=sizes.new_full((4,), 0.1),
                candidate_orientations=orientation[:, None],
                direct_only=True,
            )

        accepted, report, _ = run()
        assert accepted.all() and report["direct_path_available"].all()
        assert (report["direct_reachable_grasp_candidates"] > 0).all()
        orientation[0] = orientation.new_tensor([1, 0, 0, 0])
        rejected, _, _ = run()
        assert not rejected[0] and rejected[1:].all()


@pytest.mark.skipif(not os.environ.get("PICK_BASKET_TIMING_ROOT"), reason="Records required")
def test_cached_grasp_revalidation_under_small_state_changes(tmp_path):
    from gr00t.rl.pick import basket_grasp
    from gr00t.rl.pick.basket_access import shifted_grasp_counts_batch

    state, sizes, support, mask, geometry, collision, limits, minimum = _basket_inputs()
    options = {}

    def run(count):
        rows = torch.arange(count, device=sizes.device)
        return basket_grasp.plan_basket_grasp_batch(
            {key: value[:count] for key, value in state.items()},
            sizes[:count],
            limits,
            collision,
            geometry.select(rows),
            0.02,
            support_points=support[:count],
            support_mask=mask[:count],
            initial_height=state["object_state"][:count, 2],
            minimum_height=minimum[:count],
            prefer_realized_body=torch.zeros(count, device=sizes.device, dtype=torch.bool),
            realized_body=torch.ones(count, device=sizes.device, dtype=torch.bool),
            minimum_realized_grasp_margin_rad=0.03490658503988659,
            **options,
        )

    with torch.inference_mode():
        baseline = run(4)
        assert baseline[1].all()
        orientations = baseline[2]["entry_orientation_world_wxyz"][
            torch.arange(100, device="cuda") % 4
        ]
        offset = torch.linspace(-0.005, 0.005, 100, device="cuda")
        state["root_states"][:, 2] += offset
        state["root_states"][:, 0] += offset * 0.4
        state["ee_pos"][:, 2] += offset
        state["ee_pos"][:, 0] += offset * 0.4
        state["object_state"][:, 0] += offset * 0.2
        geometry.basket_world.origins.copy_(state["root_states"][:, :3])
        geometry.basket_world.orientations.copy_(state["root_states"][:, 3:7])
        options["candidate_orientations"] = orientations[:, None]
        torch.cuda.synchronize()
        started = time.perf_counter()
        counts, _ = shifted_grasp_counts_batch(
            state,
            sizes,
            state["object_state"][:, None, :3],
            collision,
            geometry,
            support,
            limits,
            support_mask=mask,
            candidate_orientations=orientations[:, None],
        )
        result = run(100)
        torch.cuda.synchronize()
        valid = result[1] & (counts[:, 0] > 0)
        report = {
            "rows": 100,
            "cached_revalidation_s": time.perf_counter() - started,
            "valid": int(valid.sum()),
            "full_search_required": int((~valid).sum()),
        }
        orientations[0] = orientations.new_tensor([1, 0, 0, 0])
        bad_counts, _ = shifted_grasp_counts_batch(
            state,
            sizes,
            state["object_state"][:, None, :3],
            collision,
            geometry,
            support,
            limits,
            support_mask=mask,
            candidate_orientations=orientations[:, None],
        )
        bad_result = run(100)
        assert not bad_result[1][0] and bad_counts[0, 0] == 0
    (tmp_path / "cached_grasp_timing.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    assert report["cached_revalidation_s"] < 60
