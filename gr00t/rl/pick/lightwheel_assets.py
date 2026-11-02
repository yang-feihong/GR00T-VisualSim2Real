"""Offline import of licensed Lightwheel rigid-object packages."""

import json
import shutil
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=4)
def package_directory(url, digest, root):
    from gr00t.rl.pick.asset_library import download, safe_extract

    directory = Path(root) / ".downloads/lightwheel_packages" / digest
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "extracted.json"
    if not marker.exists():
        download(url, directory / "source.zip", sha256=digest)
        safe_extract(directory / "source.zip", directory / "contents")
        marker.write_text(json.dumps({"sha256": digest, "url": url}) + "\n")
    elif json.loads(marker.read_text())["sha256"] != digest:
        raise ValueError("Lightwheel archive marker mismatch")
    return directory / "contents"


def mjcf_visual_source(path):
    """Admit a single opaque, untransformed mesh without changing its appearance."""
    root = ET.parse(path).getroot()
    if root.findall(".//worldbody//joint") or root.findall(".//worldbody//freejoint"):
        raise ValueError("Articulated targets require a separate importer")
    visual = root.findall('.//worldbody//geom[@class="visual"]')
    if len(visual) != 1:
        raise ValueError("Multi-part visual is not admitted by the single-mesh importer")
    for element in root.findall(".//worldbody//body") + visual:
        if any(
            key in element.attrib
            for key in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis")
        ):
            raise ValueError("Transformed MJCF visual requires explicit transform baking")
    mesh = root.find(f'./asset/mesh[@name="{visual[0].get("mesh")}"]')
    if mesh is None or not np.allclose(np.fromstring(mesh.get("scale", "1 1 1"), sep=" "), 1):
        raise ValueError("Scaled or missing MJCF mesh")
    material = root.find(f'./asset/material[@name="{visual[0].get("material")}"]')
    if material is None:
        raise ValueError("Missing source material")
    rgba = np.fromstring(material.get("rgba", "1 1 1 1"), sep=" ")
    if rgba.shape != (4,) or not np.allclose(rgba, 1):
        raise ValueError("Tinted or transparent material requires explicit material baking")
    texture = root.find(f'./asset/texture[@name="{material.get("texture")}"]')
    if texture is None:
        raise ValueError("Missing source texture")
    return path.parent / mesh.get("file"), path.parent / texture.get("file")


def kitchen_visual_source(path, directory):
    """Use the shared opaque USD converter, including inherited material groups."""
    from gr00t.rl.pick.usd_scan_import import usd_visual_source

    return usd_visual_source(path, directory)


def acquire_lightwheel(entry, root):
    """Publish disabled assets; existing GPU audits decide subsequent admission."""
    from gr00t.rl.pick.assets import prepare_mesh_asset

    if (
        entry.get("license") != "CC-BY-NC-4.0"
        or entry.get("usage_scope") != "non_commercial_research"
    ):
        raise ValueError("Lightwheel import requires recorded non-commercial research scope")
    if not entry["name"].startswith("lightwheel_") or Path(entry["name"]).name != entry["name"]:
        raise ValueError("Lightwheel attribution prefix is required")
    contents = package_directory(entry["url"], entry["sha256"], str(root))
    source = (contents / entry["member"]).resolve()
    if contents.resolve() not in source.parents:
        raise ValueError("Unsafe Lightwheel package member")
    temporary = Path(root) / ".downloads" / entry["name"]
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    if entry.get("format", "mjcf") == "kitchen_usd":
        mesh_path, texture_path = kitchen_visual_source(source, temporary)
    else:
        mesh_path, texture_path = mjcf_visual_source(source)
        if any(contents.resolve() not in p.resolve().parents for p in (mesh_path, texture_path)):
            raise ValueError("Unsafe Lightwheel mesh or texture path")
    asset = prepare_mesh_asset(
        entry["name"],
        temporary,
        mesh_path,
        texture_path=texture_path,
        mass_kg=entry.get("mass_kg", 0.10),
        provenance={
            "source": "lightwheel",
            "source_url": entry["url"],
            "source_page": entry["source_page"],
            "archive_sha256": entry["sha256"],
            "source_member": entry["member"],
            "category": entry["category"],
            "instance_id": entry["name"],
            "family": entry.get("family", "household"),
            "license": entry["license"],
            "license_page": entry["license_page"],
            "license_evidence": entry["license_evidence"],
            "attribution": "Lightwheel AI; RoboCasa Lightwheel object collection",
            "usage_scope": entry["usage_scope"],
            "enabled": False,
            "physical_grasp_verified": False,
            "mass_source": "configured_not_measured",
        },
    )
    destination = Path(root) / entry["name"]
    for key in ("usd_path", "collision_support_path", "collision_faces_path"):
        asset[key] = str((destination / Path(asset[key]).name).resolve())
    (temporary / "asset.json").write_text(json.dumps(asset, indent=2) + "\n")
    temporary.rename(destination)
    return asset
