"""Opt-in actual mixed-layout admission profile, without physical simulation."""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from gr00t.rl.tests.test_pick_layout_admission_comparison import evaluate


@pytest.mark.skipif(not os.environ.get("PICK_LAYOUT_MANIFEST"), reason="Real episodes required")
def test_profile_mixed_layout_admission(monkeypatch, tmp_path):
    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.pick import basket_initialization as initial
    from gr00t.rl.pick.basket import BasketConfig, BasketGeometry
    from gr00t.rl.pick.basket_layout import sample_layouts
    from gr00t.rl.pick.collision import ArmCollisionSamples
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    records = json.loads(Path(os.environ["PICK_LAYOUT_MANIFEST"]).read_text())
    episodes = [r["episode"] for r in records if "episode" in r]
    assets = [[e["asset"], *[d["asset"] for d in e.get("distractors", [])]] for e in episodes]
    basket = BasketGeometry.from_robot_usd(device="cuda")
    arm = ArmCollisionSamples.from_robot_usd(device="cuda")
    limits = basket.boxes.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    layouts = sample_layouts(
        assets, basket, [e["seed"] for e in episodes], BasketConfig(), counts=list(map(len, assets))
    )
    valid = [i for i, layout in enumerate(layouts) if isinstance(layout, tuple)]
    indices = [valid[i * (len(valid) - 1) // 19] for i in range(20)]
    timings = {}

    def timed(name, method):
        def call(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = method(*args, **kwargs)
            torch.cuda.synchronize()
            timings.setdefault(name, []).append(time.perf_counter() - started)
            return result

        return call

    for name in (
        "shifted_grasp_counts_batch",
        "plan_basket_grasp_batch",
        "plan_clear_for_nudge_batch",
        "plan_closed_nudge_batch",
    ):
        monkeypatch.setattr(initial, name, timed(name, getattr(initial, name)))
    with torch.inference_mode():
        accepted, report, elapsed = evaluate(
            [layouts[i] for i in indices], [episodes[i] for i in indices], basket, arm, limits
        )
    result = {
        "physics_simulated": False,
        "room_obstacles_included": False,
        "rows": len(indices),
        "accepted": int(accepted.sum()),
        "direct_paths": int(report["direct_path_available"].sum()),
        "elapsed_s": elapsed,
        "phase_s": timings,
    }
    (tmp_path / "admission_phase_profile.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)
