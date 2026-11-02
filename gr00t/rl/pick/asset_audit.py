"""GPU pocket admission and source-thumbnail inspection for installed metric scans."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
from gr00t.rl.pick.asset_library import installed_assets, target_shape_eligible, variant_parameters
from gr00t.rl.pick.grasp_quality import convex_grasp_quality, load_grasp_support
from gr00t.rl.pick.pocket import gripper_pocket_geometry


def audit_library(root, output):
    """No simulator: verify the real jaw pocket and friction-cone candidate test.

    This proves only free-space object/gripper compatibility, not wholebody
    reachability, environment clearance or physical retention.
    """
    from PIL import Image, ImageDraw

    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assets = installed_assets(root, include_disabled=True)
    device = "cuda:0"
    roll, pitch, yaw = torch.meshgrid(
        torch.arange(8, device=device) * (math.pi / 4),
        torch.arange(-3, 4, device=device) * (math.pi / 6),
        torch.arange(8, device=device) * (math.pi / 4),
        indexing="ij",
    )
    orientation = _euler_xyz_quat(torch.stack((roll, pitch, yaw), -1).reshape(-1, 3))[None]
    object_orientation = orientation.new_tensor([[1, 0, 0, 0]])
    aperture_bound = float(gripper_pocket_geometry(device)["open_angle"])
    report = []
    thumbnails = []
    for asset in assets:
        points = load_grasp_support(asset["usd_path"], device)
        unconditioned = {k: v for k, v in asset.items() if k != "admitted_scale_xyz"}
        scales = [[1.0, 1.0, 1.0]] + [
            variant_parameters(unconditioned, seed)["scale_xyz"] for seed in range(31)
        ]
        counts = []
        rotation = points.new_tensor(variant_parameters(unconditioned, 0)["canonical_rotation"])
        shape_eligible = [
            target_shape_eligible(np.asarray(asset["size_m"]) * scale) for scale in scales
        ]
        for scale, eligible in zip(points.new_tensor(scales), shape_eligible):
            if not eligible:
                counts.append(0)
                continue
            quality = convex_grasp_quality(
                orientation, object_orientation, (points * scale) @ rotation.T
            )
            feasible = quality["pocket_fit"] & (quality["antipodal_margin"] >= 0)
            counts.append(int(feasible.sum()))
        accepted_scales = [scale for scale, count in zip(scales, counts) if count > 0]
        admitted = bool(accepted_scales) and not asset.get("exclude_reason")
        result = {
            "name": asset["name"],
            "category": asset["category"],
            "size_m": asset["size_m"],
            "source": asset.get("source", "ycb"),
            "archive_sha256": asset.get("archive_sha256"),
            "license": asset.get("license"),
            "candidate_count": orientation.shape[1],
            "shape_proposal_count": len(scales),
            "admitted_shape_count": len(accepted_scales),
            "admitted_scale_xyz": accepted_scales,
            "feasible_counts": counts,
            "shape_eligible": shape_eligible,
            "pocket_admitted": admitted,
            "physical_grasp_verified": False,
            "exclude_reason": asset.get("exclude_reason"),
        }
        # All actual episodes still undergo their existing geometry and physical acceptance.
        metadata_path = root / asset["name"] / "asset.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            enabled=admitted,
            admitted_scale_xyz=accepted_scales,
            pocket_admission=result,
            grip_aperture_bound_rad=aperture_bound,
        )
        temporary = metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(metadata_path)
        report.append(result)
        thumb = root / asset["name"] / "thumbnails/0.jpg"
        if thumb.exists():
            thumbnails.append((thumb, result))
        summary = {
            "installed_instances": len(report),
            "pocket_admitted_instances": sum(r["pocket_admitted"] for r in report),
            "pocket_admitted_categories": sorted(
                {r["category"] for r in report if r["pocket_admitted"]}
            ),
            "note": "Free-space GPU pocket/friction screen, not a physical success rate",
            "objects": report,
        }
        (output / "asset_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if thumbnails:
        sheet = Image.new("RGB", (1000, math.ceil(len(thumbnails) / 5) * 200), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (path, result) in enumerate(thumbnails):
            im = Image.open(path).convert("RGB")
            im.thumbnail((190, 155))
            x, y = i % 5 * 200, i // 5 * 200
            sheet.paste(im, (x, y))
            draw.text((x + 3, y + 157), result["category"][:25], fill="black")
            draw.text(
                (x + 3, y + 175),
                "pocket PASS" if result["pocket_admitted"] else "pocket FAIL",
                fill="green" if result["pocket_admitted"] else "red",
            )
        sheet.save(output / "asset_thumbnails.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "objects"}), flush=True)
    return summary
