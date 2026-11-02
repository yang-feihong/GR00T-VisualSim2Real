"""Offline acquisition of pinned public Extwin scan dependencies."""

import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit


def acquire_extwin(entry, root):
    import numpy as np
    import trimesh

    from gr00t.rl.pick.asset_library import download, target_shape_eligible
    from gr00t.rl.pick.assets import prepare_mesh_asset
    from gr00t.rl.pick.usd_scan_import import simplify_scan, usd_visual_source

    if (
        entry.get("license") != "CC-BY-NC-4.0"
        or entry.get("usage_scope") != "non_commercial_research"
    ):
        raise ValueError("Extwin import requires recorded non-commercial research scope")
    name = entry["name"]
    if not name.startswith("extwin_") or Path(name).name != name:
        raise ValueError("Extwin attribution prefix is required")
    files = entry["files"]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    source_root = Path(root) / ".downloads/extwin_sources" / digest
    for file in files:
        relative = Path(file["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe Extwin dependency path")
        if urlsplit(file["url"]).hostname != "synthesis-multiverse.extwin.com":
            raise ValueError("Only official public Extwin dependencies are accepted")
        download(file["url"], source_root / relative, sha256=file["sha256"])
    member = Path(entry["member"])
    if (
        member.is_absolute()
        or ".." in member.parts
        or str(member) not in {f["path"] for f in files}
    ):
        raise ValueError("Extwin source member is not in the pinned dependency list")
    temporary = Path(root) / ".downloads" / name
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    mesh_path, texture = usd_visual_source(source_root / member, temporary)
    native = trimesh.load(mesh_path, force="mesh", process=False)
    native_size = np.asarray(native.extents)
    if not target_shape_eligible(native_size) or native_size.min() > 0.075:
        raise ValueError(f"Native Extwin dimensions are outside admission: {native_size}")
    mesh_path, geometry = simplify_scan(mesh_path, temporary)
    asset = prepare_mesh_asset(
        name,
        temporary,
        mesh_path,
        texture_path=texture,
        mass_kg=entry.get("mass_kg", 0.10),
        provenance={
            "source": "extwin",
            "source_url": entry["url"],
            "source_page": entry["source_page"],
            "archive_sha256": digest,
            "source_member": entry["member"],
            "source_files": files,
            "category": entry["category"],
            "instance_id": name,
            "family": entry.get("family", "household"),
            "license": entry["license"],
            "license_evidence": entry["license_evidence"],
            "license_page": "https://creativecommons.org/licenses/by-nc/4.0/",
            "attribution": "Extwin Synthesis",
            "usage_scope": entry["usage_scope"],
            "enabled": False,
            "physical_grasp_verified": False,
            "mass_source": "configured_not_measured",
            "native_size_m": native_size.tolist(),
            "offline_mesh_conversion": geometry,
        },
    )
    destination = Path(root) / name
    for key in ("usd_path", "collision_support_path", "collision_faces_path"):
        asset[key] = str((destination / Path(asset[key]).name).resolve())
    (temporary / "asset.json").write_text(json.dumps(asset, indent=2) + "\n")
    temporary.rename(destination)
    return asset
