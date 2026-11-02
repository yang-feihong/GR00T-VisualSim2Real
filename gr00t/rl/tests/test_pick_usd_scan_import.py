"""USD scan conversion preserves appearance and fails closed on unsupported sources."""

from pathlib import Path

import numpy as np
import pytest

from gr00t.rl.pick.usd_scan_import import usd_visual_source


def source_stage(tmp_path):
    import trimesh
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateNew(str(tmp_path / "source.usda"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    root = UsdGeom.Xform.Define(stage, "/Root")
    root.AddTranslateOp().Set((10, 20, 30))
    box = trimesh.creation.box(extents=(4, 6, 8))
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Scan")
    mesh.CreatePointsAttr(box.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(box.faces))
    mesh.CreateFaceVertexIndicesAttr(box.faces.reshape(-1).tolist())
    uv = np.array([[0.0001, 0.0002], [0.0007, 0.0002], [0.0004, 0.0008]])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
    ).Set(np.tile(uv, (len(box.faces), 1)).tolist())
    material = UsdShade.Material.Define(stage, "/Root/Looks/Material")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    texture = UsdShade.Shader.Define(stage, "/Root/Looks/Material/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("original.png"))
    Image.new("RGB", (8, 8), (240, 30, 80)).save(tmp_path / "original.png")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    reader = UsdShade.Shader.Define(stage, "/Root/Looks/Material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    transform = UsdShade.Shader.Define(stage, "/Root/Looks/Material/Transform")
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1000))
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        transform.ConnectableAPI(), "result"
    )
    stage.GetRootLayer().Save()
    return stage, box, uv


def test_uv_nodes_face_corners_texture_and_world_units(tmp_path):
    import trimesh
    from PIL import Image

    _, box, uv = source_stage(tmp_path)
    path, texture = usd_visual_source(tmp_path / "source.usda", tmp_path)
    result = trimesh.load(path, force="mesh", process=False)
    np.testing.assert_allclose(
        result.vertices[result.faces], box.vertices[box.faces] * 0.01 + (0.1, 0.2, 0.3), atol=1e-7
    )
    np.testing.assert_allclose(
        result.visual.uv[result.faces],
        np.tile((uv * 1000)[None], (len(box.faces), 1, 1)),
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(Image.open(texture)), np.asarray(Image.open(tmp_path / "original.png"))
    )


def test_missing_texture_is_not_replaced_by_gray(tmp_path):
    source_stage(tmp_path)
    (tmp_path / "original.png").unlink()
    with pytest.raises(ValueError, match="texture is not resolved"):
        usd_visual_source(tmp_path / "source.usda", tmp_path)


def test_quad_triangulation_keeps_face_corner_uv(tmp_path):
    import trimesh
    from pxr import UsdGeom

    stage, _, _ = source_stage(tmp_path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Root/Scan"))
    mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    uv = np.array([[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001]])
    UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Set(uv.tolist())
    stage.GetRootLayer().Save()
    path, _ = usd_visual_source(tmp_path / "source.usda", tmp_path)
    converted = trimesh.load(path, force="mesh", process=False)
    corners = trimesh.geometry.triangulate_quads(np.array([[0, 1, 2, 3]]))
    np.testing.assert_allclose(
        converted.visual.uv[converted.faces], (uv * 1000)[corners], atol=1e-7
    )


def test_ngon_triangulation_keeps_uv_and_winding(tmp_path):
    pytest.importorskip("mapbox_earcut")
    import trimesh
    from pxr import UsdGeom

    stage, _, _ = source_stage(tmp_path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Root/Scan"))
    angle = np.arange(5) * 2 * np.pi / 5
    polygon = np.c_[np.cos(angle), np.sin(angle), np.zeros(5)]
    polygon[0, 2] = 0.01
    mesh.GetPointsAttr().Set(polygon.tolist())
    mesh.GetFaceVertexCountsAttr().Set([5])
    mesh.GetFaceVertexIndicesAttr().Set(list(range(5)))
    uv = polygon[:, :2] * 0.0002 + 0.0005
    UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Set(uv.tolist())
    stage.GetRootLayer().Save()
    path, _ = usd_visual_source(tmp_path / "source.usda", tmp_path)
    converted = trimesh.load(path, force="mesh", process=False)
    expected = ((converted.vertices[:, :2] - (0.1, 0.2)) / 0.01 * 0.0002 + 0.0005) * 1000
    np.testing.assert_allclose(converted.visual.uv, expected, atol=1e-7)
    assert len(converted.faces) == 3 and np.all(converted.face_normals[:, 2] > 0)
    assert np.ptp(converted.vertices[:, 2]) == pytest.approx(0.0001)


def test_transparent_material_is_rejected(tmp_path):
    from pxr import UsdShade

    stage, _, _ = source_stage(tmp_path)
    UsdShade.Shader(stage.GetPrimAtPath("/Root/Looks/Material/Surface")).GetInput("opacity").Set(
        0.5
    )
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="Transparent"):
        usd_visual_source(tmp_path / "source.usda", tmp_path)


def test_unsupported_uv_node_is_rejected(tmp_path):
    from pxr import UsdShade

    stage, _, _ = source_stage(tmp_path)
    UsdShade.Shader(stage.GetPrimAtPath("/Root/Looks/Material/Transform")).CreateIdAttr(
        "UnknownNode"
    )
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="coordinate node"):
        usd_visual_source(tmp_path / "source.usda", tmp_path)


def test_additional_material_texture_is_not_silently_dropped(tmp_path):
    from pxr import Sdf, UsdShade

    stage, _, _ = source_stage(tmp_path)
    shader = UsdShade.Shader(stage.GetPrimAtPath("/Root/Looks/Material/Surface"))
    texture = UsdShade.Shader(stage.GetPrimAtPath("/Root/Looks/Material/Texture"))
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="connected surface input"):
        usd_visual_source(tmp_path / "source.usda", tmp_path)


def test_extwin_requires_license_scope_and_prefix(tmp_path):
    from gr00t.rl.pick.extwin_assets import acquire_extwin

    with pytest.raises(ValueError, match="non-commercial"):
        acquire_extwin({"license": "unknown"}, tmp_path)
    with pytest.raises(ValueError, match="prefix"):
        acquire_extwin(
            {"license": "CC-BY-NC-4.0", "usage_scope": "non_commercial_research", "name": "unsafe"},
            tmp_path,
        )


@pytest.mark.parametrize(
    "path,url",
    [
        ("../escape.usda", "https://synthesis-multiverse.extwin.com/source.usda"),
        ("source.usda", "https://untrusted.invalid/source.usda"),
    ],
)
def test_extwin_dependency_paths_and_origin_are_checked(tmp_path, path, url):
    from gr00t.rl.pick.extwin_assets import acquire_extwin

    with pytest.raises(ValueError, match="Unsafe|official"):
        acquire_extwin(
            {
                "license": "CC-BY-NC-4.0",
                "usage_scope": "non_commercial_research",
                "name": "extwin_test",
                "files": [{"path": path, "url": url, "sha256": "0" * 64}],
            },
            tmp_path,
        )


def test_extwin_publishes_disabled_shared_asset_with_original_texture(tmp_path, monkeypatch):
    import hashlib
    import shutil

    from PIL import Image

    from gr00t.rl.pick.extwin_assets import acquire_extwin

    source_stage(tmp_path)
    files = [
        {
            "path": name,
            "url": "https://synthesis-multiverse.extwin.com/" + name,
            "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
        }
        for name in ("source.usda", "original.png")
    ]

    def download(url, target, *, sha256):
        source = tmp_path / url.rsplit("/", 1)[1]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == sha256
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    monkeypatch.setattr("gr00t.rl.pick.asset_library.download", download)
    asset = acquire_extwin(
        {
            "license": "CC-BY-NC-4.0",
            "usage_scope": "non_commercial_research",
            "name": "extwin_test",
            "files": files,
            "member": "source.usda",
            "url": files[0]["url"],
            "source_page": "https://synthesis.extwin.com/",
            "license_evidence": "official README",
            "category": "test",
        },
        tmp_path / "library",
    )
    assert asset["enabled"] is False and asset["physical_grasp_verified"] is False
    np.testing.assert_allclose(asset["size_m"], (0.04, 0.06, 0.08), atol=1e-7)
    np.testing.assert_array_equal(
        np.asarray(Image.open(Path(asset["usd_path"]).with_name("albedo.png"))),
        np.asarray(Image.open(tmp_path / "original.png")),
    )
