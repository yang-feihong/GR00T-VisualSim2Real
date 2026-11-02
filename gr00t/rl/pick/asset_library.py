"""Licensed scan acquisition and seed-local, geometry-consistent asset variants.

All conversion runs at asset/episode preparation, never in the control loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from gr00t.rl.pick.assets import ASSET_ROOT, prepare_mesh_asset

GSO_REVISION = "6139aa33f6163a9f1c7352dd386760f40691ce0d"
GSO_ROOT = "https://huggingface.co/datasets/suvadityamuk/google-scanned-objects-raw"


def target_shape_eligible(size_m):
    """Screen out plates; actual jaw compatibility is checked separately on the hull."""
    size = np.sort(np.asarray(size_m, dtype=float))
    return bool(
        size.shape == (3,)
        and np.isfinite(size).all()
        and size[0] >= 0.012
        and size[1] <= 0.10
        and size[2] <= 0.22
        and size[0] >= 0.35 * size[1]
    )


def installed_assets(root=ASSET_ROOT, *, include_disabled=False):
    """Only atomically published, geometry-admitted instances enter the pool."""
    assets = []
    for path in sorted(Path(root).glob("*/asset.json")):
        if path.parent.name.startswith("."):
            continue
        asset = json.loads(path.read_text())
        if include_disabled or (
            asset.get("enabled", True) and target_shape_eligible(asset["size_m"])
        ):
            asset.setdefault("category", asset["name"].split("_", 1)[-1])
            asset.setdefault("instance_id", asset["name"])
            assets.append(asset)
    if not assets:
        raise ValueError(f"No installed pick assets in {root}")
    return assets


def sample_instances(assets, seed):
    """One instance per category; more scans do not inflate category probability."""
    generator = np.random.default_rng(np.random.SeedSequence([int(seed), 0x41535354]))
    categories = sorted({asset["category"] for asset in assets})
    return [
        (group := [a for a in assets if a["category"] == category])[
            int(generator.integers(len(group)))
        ]
        for category in categories
    ]


def variant_parameters(asset, seed, slot=0, *, placement_profile=None):
    """Small, identity-preserving deformations, with a bounded real-world size."""
    identity = int.from_bytes(hashlib.sha256(asset["name"].encode()).digest()[:4], "little")
    generator = np.random.default_rng(np.random.SeedSequence([int(seed), identity, slot]))
    organic = asset.get("family") == "produce" or asset["category"] in {
        "apple",
        "orange",
        "lemon",
        "pear",
        "peach",
        "plum",
        "strawberry",
        "banana",
    }
    scale = generator.uniform(0.90, 1.10) * generator.uniform(
        0.96 if organic else 0.99, 1.04 if organic else 1.01, 3
    )
    sizes = np.asarray(asset["size_m"]) * scale
    # Never enlarge a source beyond admission, or rescue an oversized scan by shrinking it.
    if sizes.min() < 0.012 or sizes.max() > 0.22 or np.sort(sizes)[1] > 0.10:
        scale = np.ones(3)
    admitted_scales = asset.get("admitted_scale_xyz")
    if admitted_scales:
        scale = np.asarray(admitted_scales[int(generator.integers(len(admitted_scales)))])
    tint = generator.uniform(0.88, 1.08, 3) if organic else generator.uniform(0.95, 1.05, 3)
    resting = asset.get("stable_placement_rotations")
    rotation = resting[int(generator.integers(len(resting)))] if resting else np.eye(3).tolist()
    supported_shapes = asset.get("supported_placement_shapes")
    if placement_profile is not None:
        supported_shapes = (
            asset.get("placement_profiles", {}).get(placement_profile, {}).get("supported_shapes")
        )
        if not supported_shapes:
            raise ValueError(f"No admitted {placement_profile} placement for {asset['name']}")
    if supported_shapes:
        shape = supported_shapes[int(generator.integers(len(supported_shapes)))]
        scale = np.asarray(shape["scale_xyz"])
        poses = shape["poses"]
        rotation = poses[int(generator.integers(len(poses)))]["rotation"]
    return {
        "schema": "pick_asset_variant_v6",
        "scale_xyz": np.round(scale, 5).tolist(),
        "canonical_rotation": rotation,
        "texture_rgb_multiplier": np.round(tint, 5).tolist(),
        "roughness": round(float(generator.uniform(0.45, 0.75)), 5),
        "support_plane_admitted": None if supported_shapes is None else bool(supported_shapes),
        **({} if placement_profile is None else {"placement_profile": placement_profile}),
    }


@lru_cache(maxsize=128)
def _source_support(path):
    return np.load(Path(path).with_name("collision_support.npy"))


def variant_geometry(asset, params):
    """Match preview, collision and visual origins after arbitrary stable-face rotation."""
    points = (_source_support(asset["usd_path"]) * params["scale_xyz"]) @ np.asarray(
        params["canonical_rotation"]
    ).T
    center = (points.min(0) + points.max(0)) / 2
    return points - center, center


def materialize_variant(asset, seed, slot=0, *, root=ASSET_ROOT, placement_profile=None):
    """Bake scaled visual/collider points and derive the matching planner hull.

    Reference source textures, rather than duplicating multi-megabyte images per episode.
    PhysX derives inertia from the scaled collision mesh and mass at scene creation.
    """
    import fcntl

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    if "_variant_source" in asset:
        seed, slot = asset["_variant_seed"], asset["_variant_slot"]
        placement_profile = asset.get("_variant_profile", placement_profile)
        asset = asset["_variant_source"]
    params = variant_parameters(asset, seed, slot, placement_profile=placement_profile)
    key = hashlib.sha256(
        json.dumps(
            [
                asset["name"],
                asset.get("archive_sha256"),
                asset.get("category"),
                asset["mass_kg"],
                asset.get("grip_aperture_bound_rad"),
                params,
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]
    directory = Path(root) / ".variants" / key
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata_path = directory / "asset.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())
        scale = np.asarray(params["scale_xyz"])
        rotation = np.asarray(params["canonical_rotation"])
        source = Usd.Stage.Open(asset["usd_path"])
        vertices = (
            np.asarray(UsdGeom.Mesh(source.GetPrimAtPath("/Object/Scan")).GetPointsAttr().Get())
            * scale
        )
        vertices = vertices @ rotation.T
        support, center = variant_geometry(asset, params)
        vertices -= center
        path = directory / "object.usda"
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        body = stage.DefinePrim("/Object", "Xform")
        body.GetReferences().AddReference(asset["usd_path"])
        stage.SetDefaultPrim(body)
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Object/Scan"))
        mesh.GetPointsAttr().Set(vertices.astype(np.float32).tolist())
        mesh.CreateExtentAttr().Set([Gf.Vec3f(*vertices.min(0)), Gf.Vec3f(*vertices.max(0))])
        mass = float(asset["mass_kg"] * scale.prod())
        UsdPhysics.MassAPI(body).CreateMassAttr(mass)
        shader = UsdShade.Shader(stage.GetPrimAtPath("/Object/Material/Surface"))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(params["roughness"])
        texture = stage.GetPrimAtPath("/Object/Material/Texture")
        tint = params["texture_rgb_multiplier"]
        if texture:
            UsdShade.Shader(texture).CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(*tint, 1.0)
            )
        else:
            old = shader.GetInput("diffuseColor").Get()
            shader.GetInput("diffuseColor").Set(Gf.Vec3f(*(np.asarray(old) * tint)))
        stage.GetRootLayer().Save()
        source_directory = Path(asset["usd_path"]).parent
        support_path = directory / "collision_support.npy"
        faces_path = directory / "collision_faces.npy"
        np.save(support_path, support.astype(np.float32))
        shutil.copyfile(source_directory / "collision_faces.npy", faces_path)
        metadata = {
            **asset,
            "usd_path": str(path.resolve()),
            "size_m": np.ptp(vertices, axis=0).tolist(),
            "intrinsic_size_m": (np.asarray(asset["size_m"]) * scale).tolist(),
            "mass_kg": mass,
            "source_instance_id": asset.get("instance_id", asset["name"]),
            "variant_id": key,
            "randomization": params,
            "support_plane_admitted": params["support_plane_admitted"],
            "collision_support_path": str(support_path.resolve()),
            "collision_faces_path": str(faces_path.resolve()),
        }
        temporary = metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(metadata_path)
        return metadata


def episode_asset_pool(seed, *, root=ASSET_ROOT, names=None, placement_profile=None):
    assets = installed_assets(root)
    if placement_profile is not None:
        assets = [
            a
            for a in assets
            if a.get("placement_profiles", {}).get(placement_profile, {}).get("supported_shapes")
        ]
    if names is not None:
        assets = [asset for asset in assets if asset["name"] in names]
        if {asset["name"] for asset in assets} != set(names):
            raise ValueError(f"Pick assets are not installed/admitted: {names}")
    assets = sample_instances(assets, seed)
    result = []
    for slot, asset in enumerate(assets):
        params = variant_parameters(asset, seed, slot, placement_profile=placement_profile)
        scale = np.asarray(params["scale_xyz"])
        points, _ = variant_geometry(asset, params)
        result.append(
            {
                **asset,
                "size_m": np.ptp(points, axis=0).tolist(),
                "intrinsic_size_m": (np.asarray(asset["size_m"]) * scale).tolist(),
                "mass_kg": float(asset["mass_kg"] * scale.prod()),
                "support_plane_admitted": params["support_plane_admitted"],
                "_variant_source": asset,
                "_variant_seed": seed,
                "_variant_slot": slot,
                **({} if placement_profile is None else {"_variant_profile": placement_profile}),
            }
        )
    return result


def download(url, destination, *, sha256=None, attempts=3):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        try:
            if not destination.exists():
                temporary = destination.with_suffix(destination.suffix + ".part")
                request = urllib.request.Request(url, headers={"User-Agent": "pick-assets/1.0"})
                with (
                    urllib.request.urlopen(request, timeout=90) as source,
                    temporary.open("wb") as out,
                ):
                    shutil.copyfileobj(source, out)
                os.replace(temporary, destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if sha256 and digest != sha256:
                destination.unlink()
                raise ValueError(f"Archive SHA256 mismatch: {destination}")
            return digest
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 * (attempt + 1))


def safe_extract(archive, directory):
    directory = Path(directory).resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as package:
            for info in package.infolist():
                path = (directory / info.filename).resolve()
                if directory not in path.parents and path != directory:
                    raise ValueError("Unsafe scan archive path")
                if info.is_dir():
                    continue
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Scan archive symlinks are forbidden")
                path.parent.mkdir(parents=True, exist_ok=True)
                with package.open(info) as source, path.open("wb") as out:
                    shutil.copyfileobj(source, out)
    else:
        with tarfile.open(archive) as package:
            package.extractall(directory, filter="data")


def acquire(entry, root=ASSET_ROOT):
    """Validate per-scan license, dimensions and checksums before publication."""
    root = Path(root)
    name = entry["name"]
    directory = root / name
    if (directory / "asset.json").exists():
        path = directory / "asset.json"
        asset = json.loads(path.read_text())
        # Correct category annotations without changing existing geometry or old episodes.
        for key in (
            "category",
            "family",
            "license",
            "license_page",
            "license_evidence",
            "attribution",
            "exclude_reason",
        ):
            if key in entry:
                asset[key] = entry[key]
        asset.setdefault("instance_id", name)
        if asset.get("exclude_reason"):
            asset["enabled"] = False
        temporary_meta = path.with_suffix(".tmp")
        temporary_meta.write_text(json.dumps(asset, indent=2) + "\n")
        temporary_meta.replace(path)
        return asset
    temporary = root / ".downloads" / name
    temporary.mkdir(parents=True, exist_ok=True)
    url = entry["url"]
    archive = temporary / ("source.zip" if entry["source"] == "gso" else "source.tgz")
    digest = download(url, archive, sha256=entry.get("sha256"))
    safe_extract(archive, temporary)
    if entry["source"] == "gso":
        license_text = (temporary / "metadata.pbtxt").read_text()
        if "Creative Commons Attribution 4.0 International" not in license_text:
            raise ValueError(f"Missing per-object CC BY 4.0 license: {name}")
        mesh_path = temporary / "meshes/model.obj"
    else:
        mesh_path = next(temporary.rglob("textured.obj"))
    asset = prepare_mesh_asset(
        name,
        temporary,
        mesh_path,
        mass_kg=entry.get("mass_kg", 0.10),
        texture_path=(temporary / "materials/textures/texture.png")
        if entry["source"] == "gso"
        else None,
        provenance={
            "source_url": url,
            "source_page": entry["source_page"],
            "source": entry["source"],
            "archive_sha256": digest,
            "license": entry["license"],
            "license_page": entry["license_page"],
            "license_evidence": entry.get("license_evidence", "metadata.pbtxt"),
            "attribution": entry["attribution"],
            "category": entry["category"],
            "exclude_reason": entry.get("exclude_reason"),
            "instance_id": name,
            "family": entry.get("family", "household"),
            "enabled": False,
            "admission": "metric_geometry_passed_physical_grasp_not_yet_verified",
        },
    )
    # Relocate before re-generating absolute paths in the published metadata.
    directory.parent.mkdir(parents=True, exist_ok=True)
    for key in ("usd_path", "collision_support_path", "collision_faces_path"):
        asset[key] = str(directory / Path(asset[key]).name)
    (temporary / "asset.json").write_text(json.dumps(asset, indent=2) + "\n")
    temporary.rename(directory)
    return asset


def source_entry(entry):
    if entry["source"] == "gso":
        model = entry["model"]
        return {
            **entry,
            "name": "gso_" + model,
            "url": f"{GSO_ROOT}/resolve/{GSO_REVISION}/{model}.zip",
            "source_page": "https://goo.gle/scanned-objects",
            "license": "CC-BY-4.0",
            "license_page": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "Google LLC, Google Scanned Objects, Copyright 2020 Google LLC",
        }
    if entry["source"] == "ycb":
        model = entry["model"]
        return {
            **entry,
            "name": model,
            "url": f"https://ycb-benchmarks.s3.amazonaws.com/data/google/{model}_google_16k.tgz",
            "source_page": "https://www.ycbbenchmarks.com/object-models/",
            "license": "CC-BY-4.0",
            "license_page": "https://creativecommons.org/licenses/by/4.0/",
            "license_evidence": "https://huggingface.co/datasets/ai-habitat/ycb/blob/29be64fdd95b4881f244152ad653058e0a48c28f/README.md",
            "attribution": (
                "YCB Object and Model Set; Calli, Singh, Walsman, Srinivasa, Abbeel and Dollar"
            ),
        }
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("asset_sources.json")
    )
    parser.add_argument("--root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--sync-metadata", action="store_true")
    args = parser.parse_args()
    if args.audit_output:
        from gr00t.rl.pick.asset_audit import audit_library

        audit_library(args.root, args.audit_output)
        return
    if args.report_only:
        assets = installed_assets(args.root)
        print(
            json.dumps(
                {"instances": len(assets), "categories": sorted({a["category"] for a in assets})},
                indent=2,
            )
        )
        return
    manifest = json.loads(args.manifest.read_text())
    results = []
    for raw in manifest["assets"]:
        entry = source_entry(raw)
        if args.names and entry["name"] not in args.names:
            continue
        if args.sync_metadata and not (args.root / entry["name"] / "asset.json").exists():
            continue
        try:
            asset = acquire(entry, args.root)
            result = {
                "name": entry["name"],
                "category": entry["category"],
                "status": "installed",
                "size_m": asset["size_m"],
            }
        except Exception as error:
            result = {
                "name": entry["name"],
                "category": entry["category"],
                "status": "rejected" if isinstance(error, ValueError) else "error",
                "reason": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        print(json.dumps(result), flush=True)
        report = args.root / (
            "metadata_sync_report.json" if args.sync_metadata else "acquisition_report.json"
        )
        report.write_text(json.dumps(results, indent=2) + "\n")
    if any(r["status"] == "error" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
