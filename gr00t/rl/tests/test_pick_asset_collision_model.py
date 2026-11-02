"""Asset-only planning uses the same deployed colliders as the scene adapter."""

from types import SimpleNamespace
import json

import pytest
import torch

from gr00t.rl.pick.collision import ArmCollisionSamples


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_asset_collision_model_matches_referenced_scene(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    from pxr import Usd

    from gr00t.rl.utils.b2z1_asset import ROBOT_USD

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
    root.GetReferences().AddReference(str(ROBOT_USD))
    env = SimpleNamespace(
        device=device,
        arm_joint_ids=torch.arange(6, device=device),
        sim=SimpleNamespace(stage=stage),
    )
    actual = ArmCollisionSamples.from_robot_usd(device=device)
    expected = ArmCollisionSamples(env)
    torch.testing.assert_close(actual.arm_joint_ids, expected.arm_joint_ids)
    assert len(actual.parts) == len(expected.parts)
    for (i, left), (j, right) in zip(actual.parts, expected.parts, strict=True):
        assert i == j
        torch.testing.assert_close(left, right, rtol=0, atol=1e-7)
    for left, right in zip(actual.gripper_surface, expected.gripper_surface, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU source-point compaction")
def test_cached_gripper_surfaces_preserve_the_exact_query_point_sets(tmp_path):
    from gr00t.rl.envs.door.b2z1_reset_geometry import _arm_offset_tensors

    arm = ArmCollisionSamples.from_robot_usd(device="cuda")
    p = torch.zeros(1, 1, 3, device="cuda")
    q = p.new_tensor([1, 0, 0, 0]).reshape(1, 1, 4)
    raw = arm.gripper_surface
    offsets = _arm_offset_tensors(raw[0].dtype, raw[0].device)
    rotation = p.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    original = tuple((points - (offsets[7] + offsets[8])) @ rotation for points in raw)
    compact = arm._gripper_local_surfaces(p)
    for before, after in zip(original, compact, strict=True):
        torch.testing.assert_close(torch.unique(before, dim=0), after, rtol=0, atol=0)
    assert compact is arm._gripper_local_surfaces(p)
    geometry = SimpleNamespace(
        rigid_surface_clearance=lambda p, *a, **kw: p.new_ones(p.shape[:-1]),
        visible=lambda p, *a, **kw: torch.ones(p.shape[:-1], device=p.device, dtype=torch.bool),
    )
    arm.gripper_closing_clearance(geometry, p, q, p.new_full((1, 1), 0.2))
    original_closing = (raw[1] - offsets[7] - offsets[8]) @ rotation
    torch.testing.assert_close(
        torch.unique(original_closing, dim=0), arm._closing_gripper_surface, rtol=0, atol=0
    )
    report = {
        "raw_open_points": [len(points) for points in original],
        "unique_open_points": [len(points) for points in compact],
        "unique_combined_points": len(arm._combined_gripper_surface),
        "unique_closing_points": len(arm._closing_gripper_surface),
        "point_sets_unchanged": True,
    }
    (tmp_path / "gripper_point_compaction.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
