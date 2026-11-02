"""Compare early neighbor pruning against the original late rejection on real layouts."""

import ast
import copy
import inspect
import json
import os
import time
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not os.environ.get("PICK_LAYOUT_MANIFEST"), reason="Real assets required")
def test_recorded_layout_inverse_pruning_preserves_selected_candidates(tmp_path, monkeypatch):
    from gr00t.rl.pick.basket_intrinsic_proposals import IntrinsicGraspProposals

    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.pick import basket_layout_reachability as module
    from gr00t.rl.pick.basket import BasketConfig, BasketGeometry
    from gr00t.rl.pick.basket_initialization import BasketAdmissionInputs
    from gr00t.rl.pick.basket_layout import sample_layouts
    from gr00t.rl.pick.collision import ArmCollisionSamples
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    records = json.loads(Path(os.environ["PICK_LAYOUT_MANIFEST"]).read_text())
    episodes = [item["episode"] for item in records if "episode" in item]
    assets = [[e["asset"], *[d["asset"] for d in e.get("distractors", [])]] for e in episodes]
    seeds = [e["seed"] for e in episodes]
    basket = BasketGeometry.from_robot_usd(device="cuda")
    arm = ArmCollisionSamples.from_robot_usd(device="cuda")
    metadata = BasketAdmissionInputs(episodes, max(map(len, assets)), "cuda")
    joints = basket.boxes.new_tensor([e["initial_arm_joints"] for e in episodes])
    limits = basket.boxes.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    seed_tensor = torch.tensor(seeds, device="cuda", dtype=torch.int64)
    bank = IntrinsicGraspProposals(
        metadata.support, metadata.support_mask, metadata.sizes, seed_tensor
    )

    source = ast.parse(inspect.getsource(module.screen_layouts))
    original = copy.deepcopy(source)
    function = original.body[0]
    removed = [
        node
        for node in function.body
        if isinstance(node, ast.If) and "preinverse_neighbor_pass_count" in ast.unparse(node)
    ]
    assert len(removed) == 1
    function.body.remove(removed[0])
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(original), "late_neighbor_oracle", "exec"), namespace)
    oracle = namespace["screen_layouts"]
    native_solve = module.solve_pose_grid
    calls = []

    def counted(*args, **kwargs):
        calls.append(len(args[0]))
        return native_solve(*args, **kwargs)

    monkeypatch.setattr(module, "solve_pose_grid", counted)
    namespace["solve_pose_grid"] = counted
    reports = []

    def paired(data):
        kwargs = {"collision": arm, "basket": basket, "intrinsic_proposals": bank}
        results, timings, queries = [], [], []
        for implementation in (oracle, module.screen_layouts):
            calls.clear()
            torch.cuda.synchronize()
            started = time.perf_counter()
            results.append(implementation(data, joints, limits, seed_tensor, **kwargs))
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            queries.append(sum(calls))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        torch.testing.assert_close(
            results[0][1]["orientation_local_wxyz"],
            results[1][1]["orientation_local_wxyz"],
            rtol=0,
            atol=0,
        )
        reports.append(
            {
                "baseline_s": timings[0],
                "candidate_s": timings[1],
                "baseline_inverse_candidates": queries[0],
                "candidate_inverse_candidates": queries[1],
                "passed_layouts": int(results[1][0].sum()),
            }
        )
        assert queries[1] <= queries[0]
        reverse = torch.arange(len(episodes) - 1, -1, -1, device="cuda")
        reordered = module.screen_layouts(
            {
                key: value[reverse] if torch.is_tensor(value) and value.ndim else value
                for key, value in data.items()
            },
            joints[reverse],
            limits,
            seed_tensor[reverse],
            collision=arm,
            basket=basket,
            intrinsic_proposals=bank.select(reverse),
        )
        torch.testing.assert_close(results[1][0][reverse], reordered[0], rtol=0, atol=0)
        torch.testing.assert_close(
            results[1][1]["orientation_local_wxyz"][reverse],
            reordered[1]["orientation_local_wxyz"],
            rtol=0,
            atol=0,
        )
        return results[1]

    with torch.inference_mode():
        sample_layouts(
            assets,
            basket,
            seeds,
            BasketConfig(),
            counts=list(map(len, assets)),
            target_asset_indices=[0] * len(assets),
            proposal_filter=paired,
        )
    report = {
        "physics_simulated": False,
        "rows": len(episodes),
        "selected_candidates_bitwise_equal": True,
        "measurements": reports,
    }
    (tmp_path / "inverse_pruning.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
