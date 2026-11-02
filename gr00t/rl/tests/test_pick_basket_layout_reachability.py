"""Bounded layout ranking preserves geometric fallback and seed ownership."""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick import basket_layout_reachability as reach
from gr00t.rl.pick.basket import BasketConfig, BasketGeometry
from gr00t.rl.pick.basket_layout import sample_layouts


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gripper_axis_projection_rejects_real_overlap_not_empty_aabb_volume(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    p = torch.zeros(1, 1, 3, device=device)
    angle = p.new_tensor(torch.pi / 4)
    q = torch.stack(((angle / 2).cos(), angle * 0, angle * 0, (angle / 2).sin()))[None, None]
    local_low = p.new_tensor([-1, -0.05, -0.05])
    local_high = -local_low
    signs = torch.cartesian_prod(*[p.new_tensor([-1, 1])] * 3)
    world = reach._quat_rotate(q[..., None, :], signs * local_high)
    low, high = world.amin(-2), world.amax(-2)
    centers = p.new_tensor([[0.7, -0.7, 0], [0.7, 0.7, 0]])
    obstacle_low, obstacle_high = centers - 0.025, centers + 0.025
    assert not reach._boxes_clear(low, high, obstacle_low, obstacle_high).any()
    for index, expected in enumerate((True, False)):
        clear = reach._gripper_boxes_clear(
            low,
            high,
            p,
            q,
            local_low,
            local_high,
            obstacle_low[index : index + 1],
            obstacle_high[index : index + 1],
        )
        assert clear.eq(expected).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_joint_proposals_are_bounded_and_batch_independent(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    n, width = 3, 12
    xy = torch.zeros(n, width, 1, 2)
    xy[..., 0] = -0.1
    data = {
        "valid": torch.ones(n, width, dtype=torch.bool),
        "xy": xy,
        "target": torch.zeros(n, dtype=torch.long),
        "height": torch.full((n, width, 1), 0.13),
        "sizes": torch.full((n, 1, 3), 0.04),
        "rim_z": 0.21,
    }
    data = {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()
    }
    joints = torch.tensor([[1.57, 1.57, -1.3, 1.1, 0, 0]], device=device).expand(n, -1)
    limits = torch.tensor([[-3, 3]] * 6, dtype=torch.float32, device=device)
    calls = []

    def solve(p, q, root, seed, limits):
        calls.append(len(p))
        shape = (*seed.shape[:-1], p.shape[1])
        return SimpleNamespace(
            exact=torch.ones(shape, dtype=torch.bool, device=p.device),
            position_error=p.new_zeros(*shape, 3),
            arm_joint_positions=p.new_zeros(*shape, 6),
        )

    monkeypatch.setattr(reach, "solve_pose_grid", solve)
    monkeypatch.setattr(
        reach,
        "gripper_pocket_geometry",
        lambda *args: {"back_x": 0.02, "front_x": 0.10, "fixed_plane_z": -0.02},
    )
    seeds = torch.tensor([1700, 1701, 1702], device=device)

    def run(rows):
        return reach.screen_layouts(
            {k: v[rows] if torch.is_tensor(v) else v for k, v in data.items()},
            joints[rows],
            limits,
            seeds[rows],
            maximum_candidates=4,
        )

    combined, report = run(torch.arange(3, device=device))
    assert calls == [12]
    assert (combined.sum(1) <= 4).all() and (combined.sum(1) > 0).all()
    for row in range(3):
        single, local = run(torch.tensor([row], device=device))
        torch.testing.assert_close(combined[row : row + 1], single, rtol=0, atol=0)
        torch.testing.assert_close(
            report["orientation_local_wxyz"][row : row + 1],
            local["orientation_local_wxyz"],
            rtol=0,
            atol=0,
        )
    data["valid"].zero_()
    empty, _ = run(torch.arange(3, device=device))
    assert not empty.any()
    assert len(calls) == 4


def test_failed_reachability_keeps_original_geometric_sample():
    class Basket:
        boxes = torch.zeros(1, 8, 3)
        floor_lower, floor_upper = torch.tensor([-0.3, -0.2, 0.1]), torch.tensor([0.3, 0.2, 0.1])
        rim_z, source = 0.3, "test"

        def support(self, xy):
            return torch.full(xy.shape[:-1], 0.1), torch.ones(xy.shape[:-1])

    assets = [[{"name": "test", "size_m": [0.04] * 3}]]
    config = BasketConfig(min_objects=1, max_objects=1, placement_candidates=16)
    expected = sample_layouts(assets, Basket(), [1700], config, counts=[1])

    def reject(data):
        return torch.zeros_like(data["valid"]), {
            "evaluated": data["valid"],
            "orientation_local_wxyz": torch.zeros(1, 16, 4),
        }

    actual = sample_layouts(assets, Basket(), [1700], config, counts=[1], proposal_filter=reject)
    assert actual[0][0] == expected[0][0]
    assert actual[0][1] == expected[0][1]
    assert actual[0][2]["reachability_fallback"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("box_overlap_fallback", [False, True])
def test_sparse_rejection_never_runs_inverse_or_claims_acceptance(
    monkeypatch, device, box_overlap_fallback
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    sizes = torch.full((2, 1, 3), 0.04, device=device)
    data = {
        "valid": torch.ones(2, 4, dtype=torch.bool, device=device),
        "xy": sizes.new_zeros(2, 4, 1, 2),
        "target": torch.zeros(2, dtype=torch.long, device=device),
        "height": sizes.new_full((2, 4, 1), 0.13),
        "sizes": sizes,
        "yaw": sizes.new_zeros(2, 4, 1),
        "mask": torch.ones(2, 1, dtype=torch.bool, device=device),
        "rim_z": 0.21,
    }
    calls = []

    def sparse(positions, orientations, points, **kwargs):
        calls.append(len(points))
        assert len(points) == 6 and kwargs["check_rays"] is False
        return positions.new_zeros(positions.shape[:-1])

    def inverse(*args, **kwargs):
        pytest.fail("Rejected surface samples must not reach IK")

    monkeypatch.setattr(reach, "solve_pose_grid", inverse)
    collision = SimpleNamespace(_gripper_local_surfaces=lambda p: (p.new_zeros(8, 3),))
    bank = SimpleNamespace(
        work={},
        counts=torch.ones(2, device=device, dtype=torch.long),
        sample=lambda owners, draw, yaw: (
            sizes.new_tensor([0.70710678, 0, 0.70710678, 0]).expand(len(owners), -1),
            sizes.new_zeros(len(owners), 3),
            torch.ones_like(owners, dtype=torch.bool),
        ),
    )
    passed, report = reach.screen_layouts(
        data,
        sizes.new_zeros(2, 6),
        sizes.new_tensor([[-3, 3]] * 6),
        torch.tensor([12, 34], device=device),
        maximum_candidates=4,
        collision=collision,
        basket=SimpleNamespace(geometry=SimpleNamespace(rigid_surface_clearance=sparse)),
        intrinsic_proposals=bank,
        box_overlap_fallback=box_overlap_fallback,
    )
    assert calls == [6] and not passed.any()
    assert not report["preinverse_sparse_basket_pass_count"].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU ray queries required")
def test_fixed_target_identity_survives_layout_permutation():
    basket = BasketGeometry.from_robot_usd(device="cuda")
    assets = [{"name": "one", "size_m": [0.035] * 3}, {"name": "target", "size_m": [0.035] * 3}]
    rows = sample_layouts(
        [assets] * 4,
        basket,
        [100, 101, 102, 103],
        BasketConfig(),
        counts=[2] * 4,
        target_asset_indices=[1] * 4,
    )
    for items, target, _ in rows:
        assert items[target]["asset"]["name"] == "target"
    with pytest.raises(ValueError, match="all supplied assets"):
        sample_layouts(
            [assets], basket, [100], BasketConfig(), counts=[1], target_asset_indices=[1]
        )


@pytest.mark.skipif(
    not os.environ.get("PICK_LAYOUT_MANIFEST"), reason="Real prepared episodes required"
)
def test_hundred_real_layouts_screening_timing(tmp_path):
    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    records = json.loads(Path(os.environ["PICK_LAYOUT_MANIFEST"]).read_text())
    episodes = [r["episode"] for r in records if "episode" in r]
    assert len(episodes) == 100
    assets = [[e["asset"], *[d["asset"] for d in e.get("distractors", [])]] for e in episodes]
    seeds = [e["seed"] for e in episodes]
    basket = BasketGeometry.from_robot_usd(device="cuda")
    joints = basket.boxes.new_tensor([e["initial_arm_joints"] for e in episodes])
    limits = basket.boxes.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    seed_tensor = torch.tensor(seeds, device="cuda", dtype=torch.int64)
    timings = []
    summaries = []
    from gr00t.rl.pick.collision import ArmCollisionSamples

    arm = ArmCollisionSamples.from_robot_usd(device="cuda")

    def screen(data):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = reach.screen_layouts(
            data, joints, limits, seed_tensor, collision=arm, basket=basket
        )
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        summaries.append(
            {name: value for name, value in result[1].items() if name.endswith("pass_count")}
        )
        return result

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        selected = sample_layouts(
            assets,
            basket,
            seeds,
            BasketConfig(),
            counts=list(map(len, assets)),
            proposal_filter=screen,
        )
    torch.cuda.synchronize()
    report = {
        "physics_simulated": False,
        "rows": len(episodes),
        "total_s": time.perf_counter() - started,
        "screening_s": timings,
        "preferred": sum(
            isinstance(x, tuple) and not x[2]["reachability_fallback"] for x in selected
        ),
        "geometric_failures": sum(not isinstance(x, tuple) for x in selected),
        "max_joint_queries_per_row": 1000,
        "additional_contact_points": 0,
    }
    (tmp_path / "layout_prefilter_timing.json").write_text(json.dumps(report, indent=2))
    from gr00t.rl.auto_open.tensor_metadata import to_host

    (tmp_path / "layout_prefilter_predicates.json").write_text(
        json.dumps(to_host(summaries), separators=(",", ":"))
    )
    print(report, flush=True)
    assert report["total_s"] < 60
