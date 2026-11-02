"""Heterogeneous scene references must not inherit a source environment's assets."""

import ast
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def test_rooms_reuse_standard_multi_asset_spawner_in_memory_stage():
    path = Path(__file__).resolve().parents[1] / "pick/environment.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "PickEnvironment")
    setup = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_setup_scene")
    source = ast.unparse(setup)
    assert "MultiUsdFileCfg" in source
    assert "scenes.func('/World/envs/env_.*/VlnverseScene', scenes)" in source
    assert "scene.func(path, scene)" not in source
    assert "_spawn_object(episode, 'PickObject')" in source


def test_direct_room_references_match_template_copies_including_physics_and_transforms():
    rooms = []
    for index in range(2):
        room = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(room, "/Room")
        room.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Cube.Define(room, "/Room/Table")
        mesh.CreateSizeAttr(0.4 + index * 0.1)
        mesh.AddTranslateOp().Set(Gf.Vec3d(0, 0, index))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        rooms.append(room)
    results = []
    for direct in (False, True):
        stage = Usd.Stage.CreateInMemory()
        for index, room in enumerate(rooms):
            parent = UsdGeom.Xform.Define(stage, f"/World/envs/env_{index}")
            parent.AddTranslateOp().Set(Gf.Vec3d(index * 32, 0, 0))
            destination = parent.GetPath().AppendChild("VlnverseScene")
            path = destination if direct else Sdf.Path(f"/World/Template/Asset_{index}")
            prim = stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddReference(room.GetRootLayer().identifier)
            if not direct:
                Sdf.CopySpec(stage.GetRootLayer(), path, stage.GetRootLayer(), destination)
        if not direct:
            stage.RemovePrim("/World/Template")
        results.append(stage.Flatten().ExportToString())
    assert results[0] == results[1]


def test_only_pick_opts_into_independent_environment_copies():
    root = Path(__file__).resolve().parents[1]
    base = ast.parse(
        (root / "third_party/visual_wholebody_lowlevel/legged_gym/envs/base/legged_robot.py")
        .read_text()
    )
    pick = ast.parse((root / "pick/environment.py").read_text())
    for tree, class_name, expected in (
        (base, "LeggedRobotIsaacLab", False),
        (pick, "PickEnvironment", True),
    ):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == class_name)
        assignment = next(node for node in cls.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name)
                                  and target.id == "copy_environment_source"
                                  for target in node.targets))
        assert ast.literal_eval(assignment.value) is expected
    clone_calls = [node for node in ast.walk(base) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "clone_environments"]
    assert len(clone_calls) == 1
    copy_arg = next(keyword.value for keyword in clone_calls[0].keywords
                    if keyword.arg == "copy_from_source")
    assert ast.unparse(copy_arg) == "self.copy_environment_source"


def test_usd_source_inheritance_overrides_a_different_environment_asset_reference():
    assets = []
    for size in (1.0, 2.0):
        asset = Usd.Stage.CreateInMemory()
        asset.DefinePrim("/Object", "Xform")
        UsdGeom.Mesh.Define(asset, "/Object/Scan").CreatePointsAttr([Gf.Vec3f(size, 0, 0)])
        assets.append(asset)

    resolved = []
    for inherit in (True, False):
        stage = Usd.Stage.CreateInMemory()
        source = stage.DefinePrim("/World/envs/env_0", "Xform")
        destination = "/World/envs/env_1"
        if inherit:
            stage.DefinePrim(destination, "Xform").GetInherits().AddInherit(source.GetPath())
        else:
            Sdf.CopySpec(stage.GetRootLayer(), source.GetPath(), stage.GetRootLayer(), destination)
        for index, asset in enumerate(assets):
            actor = stage.DefinePrim(f"/World/envs/env_{index}/PickObject", "Xform")
            actor.GetReferences().AddReference(asset.GetRootLayer().identifier, "/Object")
        points = UsdGeom.Mesh.Get(stage, destination + "/PickObject/Scan").GetPointsAttr().Get()
        resolved.append(float(points[0][0]))
    print(
        {
            "inherited_environment_second_asset_size": resolved[0],
            "independent_environment_second_asset_size": resolved[1],
        }
    )
    assert resolved == [1.0, 2.0]


def test_independent_environments_keep_real_asset_meshes_after_template_copy():
    asset_root = Path(__file__).resolve().parents[1] / "data/pick_assets"
    paths = [asset_root / name / "object.usda" for name in
             ("014_lemon", "gso_Android_Figure_Orange")]
    if not all(path.is_file() for path in paths):
        pytest.skip("downloaded heterogeneous target assets unavailable")

    def mesh_signature(stage, root):
        return {
            str(prim.GetPath().MakeRelativePath(root)): tuple(
                tuple(float(value) for value in point)
                for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()
            )
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root))
            if prim.IsA(UsdGeom.Mesh)
        }

    stage = Usd.Stage.CreateInMemory()
    layer = stage.GetRootLayer()
    source = "/World/envs/env_0"
    destination = "/World/envs/env_1"
    stage.DefinePrim(source, "Xform")
    Sdf.CopySpec(layer, source, layer, destination)
    for index, path in enumerate(paths):
        template = stage.DefinePrim(f"/World/Templates/Asset_{index}", "Xform")
        template.GetReferences().AddReference(str(path))
        object_path = f"/World/envs/env_{index}/PickObject"
        Sdf.CreatePrimInLayer(layer, object_path)
        Sdf.CopySpec(layer, template.GetPath(), layer, object_path)
    expected = mesh_signature(stage, "/World/Templates/Asset_1")
    assert expected
    assert mesh_signature(stage, destination + "/PickObject") == expected
    assert mesh_signature(stage, source + "/PickObject") != expected
    stage.GetPrimAtPath(source + "/PickObject").SetActive(False)
    assert mesh_signature(stage, destination + "/PickObject") == expected
