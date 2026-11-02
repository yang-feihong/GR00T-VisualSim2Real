"""Download original YCB scans and prepare metric convex rigid-body assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

OBJECTS = ("014_lemon", "015_peach", "018_plum", "012_strawberry")
ASSET_ROOT = Path(__file__).resolve().parents[1] / "data/pick_assets"
SOURCE = "https://ycb-benchmarks.s3.amazonaws.com/data/google/{name}_google_16k.tgz"


def prepare_collision_support(directory):
    """Cache compact convex support vertices without modifying the installed USD."""
    import trimesh
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(directory / "object.usda"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Object/Scan"))
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    hull = trimesh.convex.convex_hull(vertices)
    path = directory / "collision_support.npy"
    np.save(path, np.asarray(hull.vertices, dtype=np.float32))
    faces_path = directory / "collision_faces.npy"
    np.save(faces_path, np.asarray(hull.faces, dtype=np.int32))
    metadata_path = directory / "asset.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        collision_support_path=str(path.resolve()),
        collision_support_vertices=len(hull.vertices),
        collision_faces_path=str(faces_path.resolve()),
    )
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(metadata_path)
    return metadata


def prepare_asset(name: str, root: Path = ASSET_ROOT) -> dict:
    """Preserve scan scale, texture, origin, provenance and download checksum."""
    if name not in OBJECTS:
        raise ValueError(name)
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "source.tgz"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        urllib.request.urlretrieve(SOURCE.format(name=name), temporary)
        temporary.replace(archive)
    with tarfile.open(archive) as package:
        for member in package.getmembers():
            if not member.isfile():
                continue
            target = (directory / member.name).resolve()
            if directory.resolve() not in target.parents:
                raise ValueError(f"Unsafe archive entry: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.extractfile(member) as source, target.open("wb") as output:
                output.write(source.read())
    mesh_path = next(directory.rglob("textured.obj"))
    return prepare_mesh_asset(
        name,
        directory,
        mesh_path,
        provenance={
            "source_url": SOURCE.format(name=name),
            "source_page": "https://www.ycbbenchmarks.com/object-models/",
            "license_page": "https://www.ycbbenchmarks.com/",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        },
    )


def prepare_mesh_asset(name, directory, mesh_path, *, provenance, mass_kg=0.10, texture_path=None):
    """Convert a metric scan, retaining its texture and an exact convex planning hull."""
    import trimesh
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    dimensions = np.ptp(vertices, axis=0)
    if not np.isfinite(vertices).all() or not (
        0.012 <= dimensions.min() <= 0.075
        and np.sort(dimensions)[1] <= 0.10
        and dimensions.max() <= 0.22
    ):
        raise ValueError(f"Scan is not a small graspable object: {name} {dimensions}")
    uv = getattr(mesh.visual, "uv", None)
    if texture_path is not None and uv is None:
        raise ValueError(f"Explicit source texture has no mesh UV coordinates: {mesh_path}")
    vertices -= (vertices.min(0) + vertices.max(0)) / 2
    usd_path = directory / "object.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    body = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(body.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_kg = 0.04 if name == "012_strawberry" else mass_kg
    mass.CreateMassAttr(mass_kg)
    surface = UsdGeom.Mesh.Define(stage, "/Object/Scan")
    surface.CreatePointsAttr(vertices.astype(np.float32).tolist())
    surface.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    surface.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    surface.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(surface.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(surface.GetPrim()).CreateApproximationAttr("convexHull")
    material = UsdShade.Material.Define(stage, "/Object/Material")
    shader = UsdShade.Shader.Define(stage, "/Object/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    if uv is not None:
        # OBJ loaders may invent a placeholder; explicit source metadata takes precedence.
        if texture_path is not None:
            from PIL import Image

            image = Image.open(texture_path)
        else:
            image = mesh.visual.material.image
        if image is None:
            raise ValueError(f"Textured scan has no readable texture: {mesh_path}")
        image.save(directory / "albedo.png")
        UsdGeom.PrimvarsAPI(surface).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(np.asarray(uv, dtype=np.float32).tolist())
        reader = UsdShade.Shader.Define(stage, "/Object/Material/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture = UsdShade.Shader.Define(stage, "/Object/Material/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("albedo.png"))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb"
        )
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.6, 0.1))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(material)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.8)
    physics_material.CreateDynamicFrictionAttr(0.6)
    physics_material.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(material, materialPurpose="physics")
    stage.GetRootLayer().Save()
    metadata = {
        "name": name,
        **provenance,
        "usd_path": str(usd_path.resolve()),
        "size_m": dimensions.tolist(),
        "mass_kg": mass_kg,
        "scale": 1.0,
        "collision": "convexHull",
    }
    (directory / "asset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return prepare_collision_support(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--support-only", action="store_true")
    args = parser.parse_args()
    for name in OBJECTS:
        metadata = (
            prepare_collision_support(args.root / name)
            if args.support_only
            else prepare_asset(name, args.root)
        )
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
