"""Compare bounded carry early exit against exhaustive recorded-state planning."""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not os.environ.get("PICK_CARRY_RECORD_ROOT"), reason="Recorded batch required")
def test_recorded_carry_cost_bound_preserves_original_paths(tmp_path):
    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.pick.basket import BasketGeometry
    from gr00t.rl.pick.basket_carry import plan_basket_carry_batch
    from gr00t.rl.pick.basket_controller import BasketArmCollisionSamples, basket_required_height
    from gr00t.rl.pick.basket_geometry import BasketGeometryBatch
    from gr00t.rl.pick.basket_initialization import BasketAdmissionInputs
    from gr00t.rl.pick.collision import ArmCollisionSamples
    from gr00t.rl.pick.scene_geometry import SceneGeometry
    from gr00t.rl.pick.tensor_rows import pad_device_rows
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    root = Path(os.environ["PICK_CARRY_RECORD_ROOT"])
    available = []
    for path in root.glob("logs/rank*-control/seed_*/summary.json"):
        summary = json.loads(path.read_text())
        if "lift" in summary.get("planning", {}):
            available.append(path)
    available.sort(key=lambda path: int(path.parent.name.removeprefix("seed_")))
    count = int(os.environ.get("PICK_CARRY_RECORD_COUNT", "8"))
    assert len(available) >= count >= 2
    paths = [available[i * (len(available) - 1) // (count - 1)] for i in range(count)]
    episodes, snapshots, apertures = [], [], []
    for path in paths:
        episodes.append(json.loads(path.with_name("episode.json").read_text()))
        record = torch.load(path.with_name("control_record.pt"), weights_only=False)
        frame = (
            record["stages"].index("LIFT_OBJECT")
            if "LIFT_OBJECT" in record["stages"]
            else record["stages"].index("FAILURE")
        )
        snapshots.append(
            {k: v[frame].cuda() for k, v in record["pre"].items() if torch.is_tensor(v)}
        )
        apertures.append(-snapshots[-1]["dof_pos"][record["gripper_joint_id"]])
    state = {
        k: torch.stack([s[k] for s in snapshots])
        for k in ("root_states", "dof_pos", "ee_pos", "ee_orn", "object_state")
    }
    state["distractor_states"], _ = pad_device_rows([s["distractor_states"] for s in snapshots])
    saved = torch.load(paths[0].with_name("collision_model.pt"), weights_only=False)
    arm = ArmCollisionSamples.__new__(ArmCollisionSamples)
    arm.parts = [(i, p.cuda()) for i, p in saved["parts"]]
    arm.gripper_surface = [p.cuda() for p in saved["gripper_surface"]]
    arm.arm_joint_ids = saved["arm_joint_ids"].cuda()
    collision = BasketArmCollisionSamples(arm)
    basket = BasketGeometry.from_robot_usd(device="cuda")
    p = state["ee_pos"]
    ground = SceneGeometry(
        p.new_tensor([[-4000, -4000, 0], [4000, -4000, 0], [4000, 4000, 0], [-4000, 4000, 0]]),
        torch.tensor([[0, 1, 2], [0, 2, 3]], device=p.device),
    )
    geometry = BasketGeometryBatch(
        [SimpleNamespace(static_scene=ground, basket=basket, episode=e) for e in episodes]
    ).update(state)
    inputs = BasketAdmissionInputs(episodes, 4, p.device)
    minimum = basket_required_height(
        basket, state, inputs.support, inputs.lift_margin, support_mask=inputs.support_mask
    )
    limits = p.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    results, seconds = [], []
    with torch.inference_mode():
        for early_exit in (False, True):
            torch.cuda.synchronize()
            started = time.perf_counter()
            results.append(
                plan_basket_carry_batch(
                    state,
                    state["object_state"][:, 2],
                    torch.stack(apertures),
                    limits,
                    collision,
                    geometry,
                    0.02,
                    basket=basket,
                    support_points=inputs.support,
                    support_mask=inputs.support_mask,
                    minimum_object_height=minimum,
                    early_exit=early_exit,
                )
            )
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
    original, optimized = results
    torch.save({"original": original, "optimized": optimized}, tmp_path / "carry_comparison.pt")
    torch.testing.assert_close(original[1], optimized[1], rtol=0, atol=0)
    for a, b in zip(original[0], optimized[0], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    for key in ("selected", "planning_failure_code", "dense_geometry_safe"):
        torch.testing.assert_close(original[2][key], optimized[2][key], rtol=0, atol=0)
    assert optimized[2]["cost_bound_early_exit"].any(), "Comparison must exercise the fast path"
    report = {
        "physics_simulated": False,
        "room_geometry_included": False,
        "original_basket_object_arm_collision": True,
        "seeds": [e["seed"] for e in episodes],
        "rows": count,
        "valid": int(optimized[1].sum()),
        "cost_bound_resolved": int(optimized[2]["cost_bound_early_exit"].sum()),
        "exhaustive_s": seconds[0],
        "early_exit_s": seconds[1],
        "exhaustive_candidates": int(original[2]["candidate_evaluated_mask"].sum()),
        "early_exit_candidates": int(optimized[2]["candidate_evaluated_mask"].sum()),
        "paths_bitwise_equal": True,
    }
    (tmp_path / "carry_cost_bound_comparison.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
