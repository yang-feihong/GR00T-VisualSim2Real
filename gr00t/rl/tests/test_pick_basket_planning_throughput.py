"""Opt-in full basket planning benchmark, using recorded independent row inputs."""

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


@pytest.mark.skipif(not os.environ.get("PICK_BASKET_TIMING_ROOT"), reason="Records required")
def test_capture_precheck_preserves_hundred_row_paths(monkeypatch, tmp_path):
    from gr00t.rl.pick import basket_grasp

    state, sizes, support, mask, geometry, collision, limits, minimum = _basket_inputs()

    def run():
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = basket_grasp.plan_basket_grasp_batch(
            state,
            sizes,
            limits,
            collision,
            geometry,
            0.02,
            support_points=support,
            support_mask=mask,
            initial_height=state["object_state"][:, 2],
            minimum_height=minimum,
            prefer_realized_body=torch.zeros(len(sizes), device=sizes.device, dtype=torch.bool),
            realized_body=torch.ones(len(sizes), device=sizes.device, dtype=torch.bool),
            minimum_realized_grasp_margin_rad=math.radians(2),
        )
        torch.cuda.synchronize()
        return result, time.perf_counter() - started

    with torch.inference_mode():
        with monkeypatch.context() as scope:
            scope.setattr(
                basket_grasp,
                "realized_grasp_candidate_mask",
                lambda state, bank, *args: bank[2]["pocket_fit"],
            )
            old, old_s = run()
        new, new_s = run()
    torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)
    assert old[1].any(), "Recorded comparison needs positive grasp plans"
    for left, right in zip(old[0], new[0], strict=True):
        torch.testing.assert_close(left[old[1]], right[old[1]], rtol=0, atol=0)
    torch.testing.assert_close(old[2]["selected"], new[2]["selected"], rtol=0, atol=0)
    report = {
        "physics_simulated": False,
        "cloned_recorded_rows": len(sizes),
        "valid": int(new[1].sum()),
        "baseline_s": old_s,
        "capture_precheck_s": new_s,
        "valid_paths_bitwise_equal": True,
    }
    (tmp_path / "capture_precheck_timing.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    assert new_s < 60


@pytest.mark.skipif(not os.environ.get("PICK_BASKET_TIMING_ROOT"), reason="Records required")
def test_recorded_compaction_preserves_real_inverse_paths(monkeypatch, tmp_path):
    from gr00t.rl.pick import basket_grasp

    local, sizes, support, mask, geometry, collision, limits, minimum = _basket_inputs()
    rows = torch.arange(4, device=sizes.device)

    def run():
        return basket_grasp.plan_basket_grasp_batch(
            {key: value[:4] for key, value in local.items()},
            sizes[:4],
            limits,
            collision,
            geometry.select(rows),
            0.02,
            support_points=support[:4],
            support_mask=mask[:4],
            initial_height=local["object_state"][:4, 2],
            minimum_height=minimum[:4],
            prefer_realized_body=torch.zeros(4, device=sizes.device, dtype=torch.bool),
            realized_body=torch.ones(4, device=sizes.device, dtype=torch.bool),
            minimum_realized_grasp_margin_rad=math.radians(2),
        )

    with torch.inference_mode():
        compact = run()
        monkeypatch.setattr(basket_grasp, "compact_grasp_bank", lambda bank: (bank, {}))
        original = run()
    torch.testing.assert_close(compact[1], original[1], rtol=0, atol=0)
    for left, right in zip(compact[0], original[0]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    selected = compact[2]["selected"]
    passed = compact[1]
    source = compact[2]["candidate_source_indices"].gather(1, selected.clamp_min(0)[:, None])[:, 0]
    torch.testing.assert_close(source[passed], original[2]["selected"][passed])
    report = {
        "physics_simulated": False,
        "independent_recorded_rows": 4,
        "valid": int(passed.sum()),
        "original_width": original[2]["candidate_cost"].shape[1],
        "compact_width": compact[2]["candidate_cost"].shape[1],
        "paths_bitwise_equal": True,
    }
    (tmp_path / "compaction_equivalence.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    assert passed.any()


def _basket_inputs():
    from gr00t.rl.pick.basket import BasketGeometry
    from gr00t.rl.pick.basket_controller import BasketArmCollisionSamples, basket_required_height
    from gr00t.rl.pick.basket_geometry import BatchedBasketPlanningGeometry
    from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry

    state, _, sizes, support, mask, nonrobot, arm, limits = _recorded_batch(
        Path(os.environ["PICK_BASKET_TIMING_ROOT"]), count=4
    )
    model = BasketGeometry.from_robot_usd(device=str(support.device))
    root = state["root_states"]
    basket = BatchedSceneGeometry(
        [model.geometry],
        torch.zeros(4, device=root.device, dtype=torch.long),
        root[:, :3],
        root[:, 3:7],
    )
    geometry = BatchedBasketPlanningGeometry(nonrobot, basket, model)
    collision = BasketArmCollisionSamples(arm)
    minimum = basket_required_height(
        model, state, support, root.new_full((4,), 0.10), support_mask=mask
    )
    rows = torch.arange(100, device=root.device) % 4
    local = {key: value[rows] for key, value in state.items()}
    return (
        local,
        sizes[rows],
        support[rows],
        mask[rows],
        geometry.select(rows),
        collision,
        limits,
        minimum[rows],
    )


@pytest.mark.skipif(not os.environ.get("PICK_BASKET_TIMING_ROOT"), reason="Records required")
def test_basket_hundred_row_direct_access(tmp_path):
    from gr00t.rl.pick.basket_access import shifted_grasp_counts_batch

    local, sizes, support, mask, geometry, collision, limits, _ = _basket_inputs()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        counts, _ = shifted_grasp_counts_batch(
            local,
            sizes,
            local["object_state"][:, None, :3],
            collision,
            geometry,
            support,
            limits,
            support_mask=mask,
        )
    torch.cuda.synchronize()
    report = {
        "physics_simulated": False,
        "cloned_recorded_rows": 100,
        "direct_access_s": time.perf_counter() - started,
        "valid": int((counts[:, 0] > 0).sum()),
    }
    (tmp_path / "basket_direct_access_timing.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    assert report["direct_access_s"] < 60, report


@pytest.mark.skipif(not os.environ.get("PICK_BASKET_TIMING_ROOT"), reason="Records required")
def test_basket_hundred_row_full_grasp_planning(tmp_path, monkeypatch):
    from gr00t.rl.pick import basket_grasp
    from gr00t.rl.pick.basket_grasp import plan_basket_grasp_batch

    local, sizes, support, mask, geometry, collision, limits, minimum = _basket_inputs()
    root = local["root_states"]
    phases = {}

    def timed(name, function):
        def call(*args, **kwargs):
            torch.cuda.synchronize()
            before = time.perf_counter()
            result = function(*args, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - before
            phases[name] = phases.get(name, 0.0) + elapsed
            print({"phase": name, "elapsed_s": elapsed}, flush=True)
            return result

        return call

    for name in (
        "cone_grasp_candidate_bank",
        "solve_default_grasp_inverse",
        "score_default_grasp_paths",
    ):
        monkeypatch.setattr(basket_grasp, name, timed(name, getattr(basket_grasp, name)))
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        path, valid, _ = plan_basket_grasp_batch(
            local,
            sizes,
            limits,
            collision,
            geometry,
            0.02,
            support_points=support,
            support_mask=mask,
            initial_height=local["object_state"][:, 2],
            minimum_height=minimum,
            prefer_realized_body=torch.zeros(100, device=root.device, dtype=torch.bool),
            realized_body=torch.ones(100, device=root.device, dtype=torch.bool),
            minimum_realized_grasp_margin_rad=math.radians(2),
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "physics_simulated": False,
        "cloned_recorded_rows": 100,
        "full_grasp_planning_s": elapsed,
        "valid": int(valid.sum()),
        "maximum_path_length": int(path[2].max()),
        "phases": phases,
    }
    (tmp_path / "basket_planning_timing.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    assert elapsed < 60, report
