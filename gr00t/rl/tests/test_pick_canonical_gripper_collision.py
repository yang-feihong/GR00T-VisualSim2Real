"""Shared jaw geometry uses asset joints rather than an arbitrary batch member's state."""

import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from gr00t.rl.auto_open.geometry import _read_collider_points
from gr00t.rl.pick.collision import ArmCollisionSamples, moving_jaw_in_wrist
from gr00t.rl.utils.b2z1_asset import ROBOT_USD


def asset_points_and_transform(aperture):
    stage = Usd.Stage.Open(str(ROBOT_USD))
    mover = next(p for p in stage.Traverse() if p.GetName() == "gripperMover")
    points = _read_collider_points(stage, mover, ("/collisions/",)).points_body
    prim = next(p for p in stage.Traverse() if p.GetName() == "jointGripper")
    joint = UsdPhysics.RevoluteJoint(prim)
    assert joint.GetBody0Rel().GetTargets()[0].name == "link06"
    assert joint.GetBody1Rel().GetTargets()[0].name == "gripperMover"
    frames = []
    for side in (0, 1):
        frame = Gf.Matrix4d(1)
        frame.SetRotate(Gf.Quatd(prim.GetAttribute(f"physics:localRot{side}").Get()))
        frame.SetTranslateOnly(Gf.Vec3d(prim.GetAttribute(f"physics:localPos{side}").Get()))
        frames.append(frame)
    axis = {"X": Gf.Vec3d.XAxis(), "Y": Gf.Vec3d.YAxis(), "Z": Gf.Vec3d.ZAxis()}[
        joint.GetAxisAttr().Get()
    ]
    rotation = Gf.Matrix4d(1).SetRotate(Gf.Rotation(axis, -math.degrees(aperture)))
    transform = frames[1].GetInverse() * rotation * frames[0]
    expected = np.array([transform.Transform(Gf.Vec3d(*map(float, point))) for point in points])
    return points, expected


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("aperture", [0.0, 0.35, math.pi / 3])
def test_canonical_mover_matches_deployed_usd_joint_frames(device, aperture):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU jaw transform")
    points, expected = asset_points_and_transform(aperture)
    cloud = torch.as_tensor(points, device=device, dtype=torch.float32)
    actual = moving_jaw_in_wrist(cloud, aperture)
    torch.testing.assert_close(actual, cloud.new_tensor(expected), atol=1e-6, rtol=0)
    if aperture == math.pi / 3:
        perturbed = moving_jaw_in_wrist(cloud, aperture - 0.01)
        assert (perturbed - actual).norm(dim=-1).max() > 0.0001


def test_collision_constructor_requires_no_robot_state_and_reuses_canonical_open_surface(
    monkeypatch,
):
    import omni

    source = Usd.Stage.Open(str(ROBOT_USD))
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
    root = stage.DefinePrim("/World/envs/env_0/Robot")
    root.GetReferences().AddReference(str(ROBOT_USD))
    module = ModuleType("omni.usd")
    def wrong_context():
        raise AssertionError("In-memory scenes must not read Kit's unrelated stage")
    module.get_context = wrong_context
    monkeypatch.setitem(sys.modules, "omni.usd", module)
    monkeypatch.setattr(omni, "usd", module, raising=False)
    # Deliberately no robot/data/body-state interface: none is needed to load colliders.
    env = SimpleNamespace(
        device="cpu", arm_joint_ids=torch.arange(6), sim=SimpleNamespace(stage=stage)
    )
    collision = ArmCollisionSamples(env)
    points, expected = asset_points_and_transform(math.pi / 3)
    torch.testing.assert_close(
        collision.gripper_surface[1], torch.tensor(expected, dtype=torch.float32), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(collision.parts[-1][1], collision.gripper_surface[1])
    assert collision.gripper_surface[1].shape == points.shape
    from gr00t.rl.pick import initialization

    monkeypatch.setattr(initialization, "SceneGeometry", lambda points, faces, device:
                        SimpleNamespace(points=points, faces=faces))
    env.body_names = [f"link{index:02d}" for index in range(7)]
    visibility_meshes = initialization.robot_collision_meshes(env)
    assert visibility_meshes
    assert all(mesh.points.numel() for _, mesh in visibility_meshes)
