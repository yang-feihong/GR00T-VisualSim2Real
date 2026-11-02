"""Embedded bolt and strike checks, with no Isaac Sim initialization."""

import ast
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

from gr00t.rl.isaac_utils.playground.env_rand.latch_geometry import (
    build_latch_bolt,
    build_strike_pocket,
)


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("opening", [-1, 1])
def test_bolt_stays_inside_panel_thickness_and_withdraws_from_strike(side, opening):
    stage = Usd.Stage.CreateInMemory()
    build_latch_bolt(stage, "/Latch/bolt", side, opening)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Latch/bolt"))
    points = np.asarray(mesh.GetPointsAttr().Get())
    np.testing.assert_allclose(np.abs(points[:, [0, 2]]).max(0), [0.008, 0.008])
    assert (points[:, 1] * side).min() == pytest.approx(-0.030)
    assert (points[:, 1] * side).max() == pytest.approx(0.020)
    assert (points[:, 1] * side - 0.025).max() == pytest.approx(-0.005)
    assert mesh.GetPurposeAttr().Get() == "default"
    assert UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get()
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 4)
    vertices = points[faces]
    normal = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    assert np.all((normal * (vertices.mean(1) - points.mean(0))).sum(1) > 0)
    frame = UsdGeom.Cube.Define(stage, "/Frame")
    UsdPhysics.CollisionAPI.Apply(frame.GetPrim())
    frame.AddTranslateOp().Set(Gf.Vec3d(0, side * 2, 1.3))
    build_strike_pocket(stage, "/Frame", 0.5, 1.5, 1.3, side, 1.0)
    assert (
        not UsdPhysics.CollisionAPI(stage.GetPrimAtPath("/Frame")).GetCollisionEnabledAttr().Get()
    )
    children = stage.GetPrimAtPath("/Frame").GetChildren()
    assert len(children) == 5
    pocket_back = UsdGeom.Xformable(
        stage.GetPrimAtPath("/Frame/pocket_back")
    ).GetLocalTransformation()
    assert side * pocket_back.ExtractTranslation()[1] > 0.522


def test_sibling_joint_frames_remain_coincident_after_parent_translation_and_rotation():
    source = Path(__file__).parents[1] / "isaac_utils/playground/env_rand/door.py"
    tree = ast.parse(source.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_update_joint_transform"
    )
    namespace = {"UsdGeom": UsdGeom, "Gf": Gf}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Env")
    root.AddTranslateOp().Set(Gf.Vec3d(9, -7, 3))
    root.AddRotateZOp().Set(37)
    a = UsdGeom.Xform.Define(stage, "/Env/Panel")
    b = UsdGeom.Xform.Define(stage, "/Env/Latch")
    a.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.2, 0.3))
    b.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.5, 0.6))
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/Env/Panel/Joint")
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.3, 0.4, 0.5))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1))
    namespace["_update_joint_transform"](
        stage, str(joint.GetPath()), str(a.GetPath()), str(b.GetPath())
    )
    cache = UsdGeom.XformCache()
    left = cache.GetLocalToWorldTransform(a.GetPrim()).Transform(
        Gf.Vec3d(joint.GetLocalPos0Attr().Get())
    )
    right = cache.GetLocalToWorldTransform(b.GetPrim()).Transform(
        Gf.Vec3d(joint.GetLocalPos1Attr().Get())
    )
    np.testing.assert_allclose(left, right, atol=1e-6)


def test_old_snapshot_restores_original_prim_types_after_latch_model_change():
    source = Path(__file__).parents[1] / "isaac_utils/playground/env_rand/door.py"
    functions = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {"_source_usd_attribute_value", "_source_usd_type", "_restore_source_asset_snapshot"}
    ]
    namespace = {"Usd": Usd, "Gf": Gf, "Sdf": Sdf, "Vt": Vt}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Door")
    UsdGeom.Mesh.Define(stage, "/Door/bolt")
    UsdGeom.Xform.Define(stage, "/Door/frame")
    UsdGeom.Cube.Define(stage, "/Door/frame/pocket_back")
    snapshot = {
        "stage_prim_path": "/Recorded",
        "prims": [
            {"path": "/Recorded" + suffix, "type": kind, "attributes": attrs, "relationships": {}}
            for suffix, kind, attrs in (
                ("", "Xform", {}),
                ("/bolt", "Cone", {"radius": 0.025}),
                ("/frame", "Cube", {"size": 2.0}),
            )
        ],
    }
    namespace["_restore_source_asset_snapshot"](stage, "/Door", snapshot)
    assert stage.GetPrimAtPath("/Door/bolt").GetTypeName() == "Cone"
    assert stage.GetPrimAtPath("/Door/frame").GetTypeName() == "Cube"
    assert not stage.GetPrimAtPath("/Door/frame/pocket_back").IsActive()


def test_strike_matrix_snapshot_round_trip_and_missing_attribute_inference():
    source = Path(__file__).parents[1] / "isaac_utils/playground/env_rand/door.py"
    functions = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {"_source_usd_attribute_value", "_source_usd_type", "_restore_source_asset_snapshot"}
    ]
    namespace = {"Usd": Usd, "Gf": Gf, "Sdf": Sdf, "Vt": Vt}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    stage = Usd.Stage.CreateInMemory()
    frame = UsdGeom.Cube.Define(stage, "/Door/frame")
    UsdPhysics.CollisionAPI.Apply(frame.GetPrim())
    build_strike_pocket(stage, "/Door/frame", 0.5, 1.5, 1.3, 1, 1.0)
    original = stage.GetPrimAtPath("/Door/frame").GetAttribute("xformOp:transform").Get()
    payload = [[float(x) for x in row] for row in original]
    snapshot = {
        "stage_prim_path": "/Door",
        "prims": [
            {"path": "/Door", "type": "Xform", "attributes": {}, "relationships": {}},
            {
                "path": "/Door/frame",
                "type": "Xform",
                "attributes": {"xformOp:transform": payload, "xformOpOrder": ["xformOp:transform"]},
                "relationships": {},
            },
        ],
    }
    UsdGeom.Xform.Define(stage, "/Copy")
    UsdGeom.Xform.Define(stage, "/Copy/frame")
    namespace["_restore_source_asset_snapshot"](stage, "/Copy", snapshot)
    restored = UsdGeom.Xformable(stage.GetPrimAtPath("/Copy/frame")).GetLocalTransformation()
    np.testing.assert_allclose(np.array(restored), np.array(original))
    namespace["_restore_source_asset_snapshot"](stage, "/Door", snapshot)
    assert stage.GetPrimAtPath("/Door/frame").GetAttribute("xformOp:transform").Get() == original
