"""Same real layouts, full admission standards, cached contacts followed by legacy fallback."""

import json
import os
import time
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not os.environ.get("PICK_LAYOUT_MANIFEST"), reason="Real assets required")
def test_cached_direct_admission_keeps_successes_and_full_checks(tmp_path):
    from gr00t.rl.pick.basket_intrinsic_proposals import IntrinsicGraspProposals
    from gr00t.rl.tests.test_pick_layout_admission_comparison import evaluate

    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.auto_open.tensor_metadata import to_host
    from gr00t.rl.pick.basket import BasketConfig, BasketGeometry
    from gr00t.rl.pick.basket_initialization import BasketAdmissionInputs
    from gr00t.rl.pick.basket_layout import sample_layouts
    from gr00t.rl.pick.collision import ArmCollisionSamples
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    records = json.loads(Path(os.environ["PICK_LAYOUT_MANIFEST"]).read_text())
    episodes = [record["episode"] for record in records if "episode" in record]
    exclusion = os.environ.get("PICK_LAYOUT_EXCLUDE_REPORT")
    if exclusion:
        used = set(json.loads(Path(exclusion).read_text())["seeds"])
        episodes = [episode for episode in episodes if episode["seed"] not in used]
    episodes = [episodes[index * (len(episodes) - 1) // 23] for index in range(24)]
    assets = [[e["asset"], *[d["asset"] for d in e.get("distractors", [])]] for e in episodes]
    basket = BasketGeometry.from_robot_usd(device="cuda")
    arm = ArmCollisionSamples.from_robot_usd(device="cuda")
    limits = basket.boxes.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    layouts = sample_layouts(
        assets,
        basket,
        [e["seed"] for e in episodes],
        BasketConfig(),
        counts=list(map(len, assets)),
        target_asset_indices=[0] * len(assets),
    )
    selected = [index for index, layout in enumerate(layouts) if isinstance(layout, tuple)][:20]
    assert len(selected) == 20
    layouts, episodes = [layouts[i] for i in selected], [episodes[i] for i in selected]
    metadata = BasketAdmissionInputs(episodes, 4, "cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    cache = IntrinsicGraspProposals(
        metadata.support,
        metadata.support_mask,
        metadata.sizes,
        torch.tensor([e["seed"] for e in episodes], device="cuda"),
    )
    torch.cuda.synchronize()
    build_s = time.perf_counter() - started
    with torch.inference_mode():
        before, _, baseline_s = evaluate(layouts, episodes, basket, arm, limits)
        after, detail, candidate_s = evaluate(
            layouts, episodes, basket, arm, limits, contact_proposals=cache
        )
    report = {
        "physics_simulated": False,
        "room_obstacles_included": False,
        "seeds": [e["seed"] for e in episodes],
        "rows": len(episodes),
        "baseline": int(before.sum()),
        "candidate": int(after.sum()),
        "baseline_s": baseline_s,
        "candidate_s": candidate_s,
        "cache_build_s": build_s,
        "regressed_seeds": [episodes[i]["seed"] for i in torch.where(before & ~after)[0].tolist()],
    }
    (tmp_path / "cached_direct_admission.json").write_text(json.dumps(report, indent=2))
    (tmp_path / "candidate_diagnostics.json").write_text(json.dumps(to_host(detail)))
    assert not (before & ~after).any(), report
