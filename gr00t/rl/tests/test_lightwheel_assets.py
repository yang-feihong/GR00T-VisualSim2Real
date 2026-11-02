"""Offline Lightwheel imports must not invent scale, missing parts or permission."""

from pathlib import Path

import numpy as np
import pytest

from gr00t.rl.pick.lightwheel_assets import (
    acquire_lightwheel,
    kitchen_visual_source,
    mjcf_visual_source,
)


def test_explicit_source_texture_overrides_obj_placeholder(tmp_path):
    import trimesh
    from PIL import Image

    from gr00t.rl.pick.assets import prepare_mesh_asset

    mesh = trimesh.creation.box(extents=(0.04, 0.06, 0.08))
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)), image=Image.new("RGB", (2, 2), (80, 80, 80))
    )
    source = tmp_path / "source.obj"
    mesh.export(source)
    texture = tmp_path / "original.png"
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:4] = (240, 30, 10)
    pixels[4:] = (10, 200, 80)
    Image.fromarray(pixels).save(texture)
    prepare_mesh_asset("test", tmp_path, source, provenance={}, texture_path=texture)
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "albedo.png")), pixels)


def test_explicit_texture_without_uv_is_not_silently_dropped(tmp_path):
    import trimesh
    from PIL import Image

    from gr00t.rl.pick.assets import prepare_mesh_asset

    source = tmp_path / "source.obj"
    trimesh.creation.box(extents=(0.04, 0.06, 0.08)).export(source)
    texture = tmp_path / "original.png"
    Image.new("RGB", (8, 8), (240, 30, 10)).save(texture)
    with pytest.raises(ValueError, match="no mesh UV"):
        prepare_mesh_asset("test", tmp_path, source, provenance={}, texture_path=texture)


def test_requires_noncommercial_scope_and_attribution(tmp_path):
    with pytest.raises(ValueError, match="non-commercial"):
        acquire_lightwheel({"license": "unknown"}, tmp_path)
    with pytest.raises(ValueError, match="prefix"):
        acquire_lightwheel(
            {"license": "CC-BY-NC-4.0", "usage_scope": "non_commercial_research", "name": "wrong"},
            tmp_path,
        )


@pytest.mark.parametrize("attribute", ['pos="0 0 1"', 'quat="0 1 0 0"'])
def test_no_silent_mjcf_transform_drop(tmp_path, attribute):
    source = tmp_path / "model.xml"
    source.write_text(
        f'<mujoco><worldbody><body {attribute}><geom class="visual"/></body></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="transform"):
        mjcf_visual_source(source)


def test_no_silent_parts_drop(tmp_path):
    source = tmp_path / "model.xml"
    source.write_text(
        '<mujoco><worldbody><body><geom class="visual"/><geom class="visual"/>'
        "</body></worldbody></mujoco>"
    )
    with pytest.raises(ValueError, match="Multi-part"):
        mjcf_visual_source(source)


def test_kitchen_uses_stage_units_and_world_transform(tmp_path):
    import trimesh
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateNew(str(tmp_path / "prop.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World")
    root.AddTranslateOp().Set((30, 40, 50))
    source = trimesh.creation.box(extents=(4, 6, 8))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Prop")
    mesh.CreatePointsAttr(source.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(source.faces))
    mesh.CreateFaceVertexIndicesAttr(source.faces.reshape(-1).tolist())
    material = UsdShade.Material.Define(stage, "/World/Looks/OmniPBR")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/OmniPBR/Shader")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    path, texture = kitchen_visual_source(tmp_path / "prop.usda", tmp_path)
    converted = trimesh.load(path, force="mesh", process=False)
    np.testing.assert_allclose(converted.extents, (0.04, 0.06, 0.08))
    np.testing.assert_allclose(converted.bounds.mean(0), (0.3, 0.4, 0.5))
    assert Path(texture).is_file() and Path(path).is_file()
