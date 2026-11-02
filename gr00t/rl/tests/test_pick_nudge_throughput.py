"""Force the real closed-jaw planning branch on 100 recorded basket rows."""

import json
import os
import time
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_basket_planning_throughput import _basket_inputs


@pytest.mark.skipif(not os.environ.get("PICK_BASKET_TIMING_ROOT"), reason="Records required")
def test_hundred_row_clear_query_cap_equivalence(tmp_path, monkeypatch):
    from gr00t.rl.pick import basket_clear
    from gr00t.rl.pick.basket_geometry import BatchedBasketPlanningGeometry
    from gr00t.rl.pick.batched_geometry import BatchedCompositeGeometry, BatchedSceneGeometry
    from gr00t.rl.pick.scene_object_geometry import load_collider_mesh

    root = Path(os.environ["PICK_BASKET_TIMING_ROOT"])
    files = sorted(
        root.glob("logs/rank*-control/seed_*/episode.json"), key=lambda p: int(p.parent.name[5:])
    )
    templates = [json.loads(path.read_text()) for path in files]
    episodes = [templates[index % 4] for index in range(100)]
    state, _, _, _, geometry, collision, limits, _ = _basket_inputs()
    assets = list(dict.fromkeys(item["asset"]["usd_path"] for item in episodes))
    target = BatchedSceneGeometry(
        [load_collider_mesh(asset, "cuda") for asset in assets],
        torch.tensor([assets.index(item["asset"]["usd_path"]) for item in episodes], device="cuda"),
        state["object_state"][:, :3],
        state["object_state"][:, 3:7],
    )
    combined = BatchedBasketPlanningGeometry(
        BatchedCompositeGeometry((geometry.nonrobot_geometry, target)),
        geometry.basket_world,
        geometry.basket_model,
    )

    def run(count):
        torch.cuda.synchronize()
        started = time.perf_counter()
        rows = torch.arange(count, device="cuda")
        result = basket_clear.plan_clear_for_nudge_batch(
            {key: value[:count] for key, value in state.items()},
            geometry.basket_model,
            limits,
            collision,
            geometry.select(rows),
            combined.select(rows),
            target.select(rows),
            0.02,
        )
        torch.cuda.synchronize()
        return result, time.perf_counter() - started

    with torch.inference_mode():
        original_gripper = collision.gripper_clearance
        original_inverse = basket_clear.safe_inverse_paths_batch

        def uncapped_gripper(*args, **kwargs):
            kwargs["maximum_distance_m"] = 2.0
            return original_gripper(*args, **kwargs)

        def uncapped_inverse(*args, **kwargs):
            kwargs["maximum_arm_query_distance_m"] = None
            return original_inverse(*args, **kwargs)

        monkeypatch.setattr(collision, "gripper_clearance", uncapped_gripper)
        monkeypatch.setattr(basket_clear, "safe_inverse_paths_batch", uncapped_inverse)
        original, original_s = run(4)

        def gripper(*args, **kwargs):
            kwargs["maximum_distance_m"] = 0.0021
            return original_gripper(*args, **kwargs)

        def inverse(*args, **kwargs):
            kwargs["maximum_arm_query_distance_m"] = 0.0021
            return original_inverse(*args, **kwargs)

        monkeypatch.setattr(collision, "gripper_clearance", gripper)
        monkeypatch.setattr(basket_clear, "safe_inverse_paths_batch", inverse)
        actual, capped_s = run(100)
    torch.testing.assert_close(actual[1][:4], original[1], rtol=0, atol=0)
    for left, right in zip(actual[0], original[0]):
        torch.testing.assert_close(left[:4], right, rtol=0, atol=0)
    report = {
        "rows": 100,
        "original_rows": 4,
        "original_s": original_s,
        "capped_s": capped_s,
        "valid": int(actual[1].sum()),
        "paths_bitwise_equal": True,
    }
    (tmp_path / "clear_cap_timing.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    assert capped_s < 60


@pytest.mark.skipif(not os.environ.get("PICK_BASKET_TIMING_ROOT"), reason="Records required")
def test_hundred_row_closed_nudge_planning(tmp_path, monkeypatch):
    from gr00t.rl.pick import basket_nudge_geometry, basket_push
    from gr00t.rl.pick.basket_initialization import BasketAdmissionInputs
    from gr00t.rl.pick.batched_geometry import BatchedCompositeGeometry, BatchedSceneGeometry
    from gr00t.rl.pick.scene_object_geometry import load_collider_mesh

    root = Path(os.environ["PICK_BASKET_TIMING_ROOT"])
    paths = sorted(
        root.glob("logs/rank*-control/seed_*/control_record.pt"),
        key=lambda p: int(p.parent.name.removeprefix("seed_")),
    )
    selected = [paths[index * (len(paths) - 1) // 3] for index in range(4)]
    templates = [json.loads(path.with_name("episode.json").read_text()) for path in selected]
    episodes = [templates[index % 4] for index in range(100)]
    state, _, support, support_mask, geometry, collision, limits, _ = _basket_inputs()
    metadata = BasketAdmissionInputs(episodes, state["distractor_states"].shape[1] + 1, "cuda")
    assets = list(dict.fromkeys(item["asset"]["usd_path"] for item in episodes))
    target = BatchedSceneGeometry(
        [load_collider_mesh(asset, "cuda") for asset in assets],
        torch.tensor([assets.index(item["asset"]["usd_path"]) for item in episodes], device="cuda"),
        state["object_state"][:, :3],
        state["object_state"][:, 3:7],
    )
    target_geometry = BatchedCompositeGeometry((geometry, target))
    phases = {}

    def timed(name, function):
        def call(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = function(*args, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            phases[name] = phases.get(name, 0) + elapsed
            print({"phase": name, "elapsed_s": elapsed}, flush=True)
            return result

        return call

    for module, name in (
        (basket_push, "shifted_grasp_counts_batch"),
        (basket_nudge_geometry, "slide_stability_batch"),
        (basket_push, "safe_inverse_paths_batch"),
    ):
        monkeypatch.setattr(module, name, timed(name, getattr(module, name)))
    monkeypatch.setattr(
        collision,
        "gripper_pose_clearance",
        timed("gripper_pose_clearance", collision.gripper_pose_clearance),
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        _, valid, diagnostic = basket_push.plan_closed_nudge_batch(
            state,
            metadata.payload_sizes,
            metadata.sizes,
            metadata.hulls[2],
            geometry.basket_model,
            limits,
            collision,
            geometry,
            target_geometry,
            target,
            0.02,
            support_points=support,
            support_mask=support_mask,
            current_path_unavailable=torch.ones(100, device="cuda", dtype=torch.bool),
            require_sliding_stability=True,
            object_centers_local=metadata.centers,
        )
    torch.cuda.synchronize()
    report = {
        "physics_simulated": False,
        "rows": 100,
        "seconds": time.perf_counter() - started,
        "valid": int(valid.sum()),
        "candidate_count": diagnostic.get("candidate_count", torch.zeros(1)).tolist(),
        "unique_contacts": diagnostic.get("sliding_stability_unique_contacts"),
        "expanded_contacts": diagnostic.get("sliding_stability_expanded_candidates"),
        "peak_pytorch_bytes": torch.cuda.max_memory_allocated(),
        "phases": phases,
    }
    (tmp_path / "nudge_timing.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    assert report["seconds"] < 60, report
