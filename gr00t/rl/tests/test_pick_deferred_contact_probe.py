"""Offline-only prototype: query the scene only for the active candidate cohort."""

import json
import os
import time

import pytest
import torch

from gr00t.rl.tests.test_pick_basket_planning_throughput import _basket_inputs


@pytest.mark.skipif(not os.environ.get("PICK_BASKET_TIMING_ROOT"), reason="Records required")
def test_deferred_contact_probe(monkeypatch, tmp_path):
    from gr00t.rl.pick import basket_grasp

    state, sizes, support, mask, geometry, collision, limits, minimum = _basket_inputs()
    active_geometry = [geometry]

    def run(count):
        rows = torch.arange(count, device=sizes.device)
        active_geometry[0] = geometry.select(rows)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = basket_grasp.plan_basket_grasp_batch(
            {key: value[:count] for key, value in state.items()},
            sizes[:count],
            limits,
            collision,
            active_geometry[0],
            0.02,
            support_points=support[:count],
            support_mask=mask[:count],
            initial_height=state["object_state"][:count, 2],
            minimum_height=minimum[:count],
            prefer_realized_body=torch.zeros(count, device=sizes.device, dtype=torch.bool),
            realized_body=torch.ones(count, device=sizes.device, dtype=torch.bool),
            minimum_realized_grasp_margin_rad=0.03490658503988659,
        )
        torch.cuda.synchronize()
        return result, time.perf_counter() - started

    class DeferredCollision:
        def __getattr__(self, name):
            return getattr(collision, name)

        def gripper_clearance(self, geometry, positions, orientations, **kwargs):
            return positions.new_full(positions.shape[:-1], kwargs["maximum_distance_m"])

        def gripper_closing_clearance(self, geometry, positions, orientations, *args, **kwargs):
            return positions.new_full(positions.shape[:-1], kwargs["maximum_distance_m"])

    original_cone = basket_grasp.cone_grasp_candidate_bank
    original_search = basket_grasp.search_grasp_batches

    def deferred_cone(*args, **kwargs):
        args = list(args)
        args[2] = DeferredCollision()
        return original_cone(*args, **kwargs)

    def deferred_search(local_state, bank, evaluate, **kwargs):
        def checked(rows, local_bank):
            q, goals, quality, _, _ = local_bank
            owner, candidate = torch.where(quality["pocket_fit"])
            if len(owner):
                g = active_geometry[0].select(rows).select(owner)
                p, orientation = goals[owner, candidate, None], q[owner, candidate, None]
                aperture = quality["aperture_angle_rad"][owner, candidate, None]
                opened = collision.gripper_clearance(g, p, orientation, maximum_distance_m=0.0061)
                closing = collision.gripper_closing_clearance(
                    g, p, orientation, aperture, maximum_distance_m=0.0061
                )
                gap = torch.minimum(opened, closing)[:, 0]
                quality["environment_clearance_m"][owner, candidate] = gap
                quality["pocket_fit"][owner, candidate] &= gap >= 0.006
            return evaluate(rows, local_bank)

        return original_search(local_state, bank, checked, **kwargs)

    with torch.inference_mode():
        reference, reference_s = run(4)
        monkeypatch.setattr(basket_grasp, "cone_grasp_candidate_bank", deferred_cone)
        monkeypatch.setattr(basket_grasp, "search_grasp_batches", deferred_search)
        actual, elapsed = run(100)
    torch.testing.assert_close(actual[1][:4], reference[1], rtol=0, atol=0)
    report = {
        "rows": 100,
        "reference_rows": 4,
        "reference_s": reference_s,
        "deferred_s": elapsed,
        "valid": int(actual[1].sum()),
        "endpoint_delta_first_four_m": (actual[0][0][:4, -1] - reference[0][0][:, -1])
        .norm(dim=-1)
        .tolist(),
    }
    (tmp_path / "deferred_contact_timing.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    assert elapsed < 60
