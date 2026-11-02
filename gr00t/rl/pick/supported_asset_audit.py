"""Offline stable-face and real-jaw support-plane audit; never a success-rate claim."""

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from gr00t.rl.auto_open.first_pull_wall import _asset_envelopes
from gr00t.rl.auto_open.geometry import _read_collider_points
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.asset_library import installed_assets, target_shape_eligible
from gr00t.rl.pick.assets import ASSET_ROOT
from gr00t.rl.pick.grasp_clearance import GRASP_CLEARANCE_M
from gr00t.rl.pick.grasp_quality import (
    convex_grasp_quality,
    grasp_support_from_hull,
)
from gr00t.rl.pick.pocket import BASKET_MINIMUM_CENTER_DEPTH_M
from gr00t.rl.utils.b2z1_asset import ROBOT_USD


def stable_pose_bank(points, faces, *, max_poses=8):
    """Keep representatives of dominant nearby resting-face basins, not seed fixtures."""
    import trimesh

    if max_poses < 1:
        raise ValueError("At least one stable pose must be requested")
    mesh = trimesh.Trimesh(points, faces, process=False)
    transforms, probability = mesh.compute_stable_poses(sigma=0, n_samples=1)
    rotations = transforms[:, :3, :3]
    up = rotations[:, 2, :]
    nearby = up @ up.T >= math.cos(math.radians(15))
    basin = nearby @ probability
    eligible = basin >= basin.max() / 3
    selected = []
    for index in np.argsort(-basin, kind="stable"):
        if eligible[index] and all(
            up[index] @ up[other] < math.cos(math.radians(20)) for other in selected
        ):
            selected.append(int(index))
            if len(selected) >= max_poses:
                break
    return rotations[selected].copy(), basin[selected].tolist()


def static_resting_pose_bank(points, faces, *, device="cuda:0", max_poses=8):
    """Bounded GPU support-face proposals with the hull COM above the face.

    These are statically stable resting faces, not dynamic basin probabilities.
    Existing measured resting banks remain preferable when available.
    """
    if max_poses < 1:
        raise ValueError("At least one stable pose must be requested")
    vertices = torch.as_tensor(points, device=device, dtype=torch.float64)
    triangles = vertices[torch.as_tensor(faces, device=device, dtype=torch.long)]
    a, b, c = triangles.unbind(1)
    volume6 = (a * torch.cross(b, c, dim=-1)).sum(-1)
    total = volume6.sum()
    if not bool(torch.isfinite(total)) or abs(float(total)) < 1e-12:
        raise ValueError("A closed convex hull is required")
    center = (volume6[:, None] * (a + b + c)).sum(0) / (4 * total)
    v, w = b - a, c - a
    normal = torch.cross(v, w, dim=-1)
    area2 = normal.norm(dim=-1)
    normal = normal / area2[:, None].clamp_min(1e-20)
    # Orient outwards even for a globally reversed hull winding.
    normal *= torch.where(((a - center) * normal).sum(-1) >= 0, 1.0, -1.0)[:, None]
    projected = center - (((center - a) * normal).sum(-1))[:, None] * normal
    relative = projected - a
    vv, vw, ww = (v * v).sum(-1), (v * w).sum(-1), (w * w).sum(-1)
    rv, rw = (relative * v).sum(-1), (relative * w).sum(-1)
    determinant = vv * ww - vw.square()
    u = (rv * ww - rw * vw) / determinant.clamp_min(1e-30)
    t = (rw * vv - rv * vw) / determinant.clamp_min(1e-30)
    remaining = (area2 > 1e-14) & (u >= -1e-8) & (t >= -1e-8) & (u + t <= 1 + 1e-8)
    rotations = []
    for _ in range(max_poses):
        if not bool(remaining.any()):
            break
        index = area2.masked_fill(~remaining, -torch.inf).argmax()
        up = -normal[index]
        axis = torch.eye(3, device=device, dtype=vertices.dtype)[up.abs().argmin()]
        x = torch.cross(axis, up, dim=-1)
        x = x / x.norm()
        rotations.append(torch.stack((x, torch.cross(up, x, dim=-1), up)))
        remaining &= normal @ normal[index] < math.cos(math.radians(20))
    return [
        {
            "rotation": rotation.cpu().tolist(),
            "resting_basin_probability": None,
            "resting_pose_evidence": "convex_hull_COM_projection_inside_support_triangle",
        }
        for rotation in rotations
    ]


@lru_cache(maxsize=4)
def gripper_collision_clouds(device):
    from pxr import Usd

    stage = Usd.Stage.Open(str(ROBOT_USD))
    bodies = {p.GetName(): p for p in stage.Traverse() if p.GetName() in ("link06", "gripperMover")}
    fixed, moving = [
        torch.tensor(
            _read_collider_points(stage, bodies[name], ("/collisions/",)).points_body,
            device=device,
            dtype=torch.float32,
        )
        for name in ("link06", "gripperMover")
    ]
    _, first, second, _ = _asset_envelopes(str(ROBOT_USD))
    closed = fixed.new_tensor(first @ second)
    moving = moving @ closed[:3, :3].T + closed[:3, 3]
    rotation = fixed.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    origin = fixed.new_tensor([0.1, 0, 0])
    return (fixed - origin) @ rotation, (moving - origin) @ rotation


def support_plane_clearance(
    orientations, object_orientations, points, *, command_seating_offset_m=0.0, support_mask=None
):
    """Return a depth-screen callback using the true fixed/moving collision surfaces."""
    fixed, moving = gripper_collision_clouds(str(points.device))
    inverse = orientations.clone()
    inverse[..., 1:] *= -1
    normal = _quat_rotate(inverse, points.new_tensor([0, 0, 1]))
    source = points[None] if points.ndim == 2 else points
    heights = _quat_rotate(object_orientations[:, None], source)[..., 2]
    if support_mask is not None:
        if support_mask.shape != points.shape[:-1]:
            raise ValueError("Support mask must match the support point rows")
        heights = heights.masked_fill(~support_mask, torch.inf)
    bottom = heights.amin(-1)
    fixed_minimum = (normal @ fixed.T).amin(-1)

    def clearance(center, aperture):
        center = center - center.new_tensor([0, 0, command_seating_offset_m])
        origin_z = -bottom[:, None, None] - (normal[:, :, None] * center).sum(-1)
        result = origin_z + fixed_minimum[..., None]
        for weight in torch.linspace(0, 1, 7, device=points.device):
            angle = aperture + weight * (math.pi / 3 - aperture)
            nx, ny, nz = normal[:, :, None].expand_as(center).unbind(-1)
            direction = torch.stack(
                (nx * angle.cos() + nz * angle.sin(), ny, -nx * angle.sin() + nz * angle.cos()),
                -1,
            )
            result = torch.minimum(result, origin_z + (direction @ moving.T).amin(-1))
        return result

    return clearance


def baked_pose_support(points, faces, rotation, device):
    baked = points @ rotation.T
    baked -= (baked.min(0) + baked.max(0)) / 2
    return grasp_support_from_hull(
        torch.as_tensor(baked.astype(np.float32), device=device),
        torch.as_tensor(faces, device=device, dtype=torch.long),
    )


def audit_asset(
    asset,
    device="cuda:0",
    *,
    scale_xyz=None,
    minimum_clearance_m=GRASP_CLEARANCE_M,
    placement_profile=None,
    pose_bank=None,
):

    scale = np.ones(3) if scale_xyz is None else np.asarray(scale_xyz, dtype=float)
    size = (np.asarray(asset.get("intrinsic_size_m", asset["size_m"])) * scale).tolist()
    result = {
        "name": asset["name"],
        "category": asset["category"],
        "size_m": size,
        "scale_xyz": scale.tolist(),
        "support_model": "infinite_horizontal_plane_not_furniture_edge",
        "physical_grasp_verified": False,
        "minimum_environment_clearance_m": minimum_clearance_m,
    }
    if not target_shape_eligible(size) or asset.get("exclude_reason"):
        return {**result, "eligible_shape": False, "poses": []}
    directory = Path(asset["usd_path"]).parent
    points = np.load(directory / "collision_support.npy") * scale
    faces = np.load(directory / "collision_faces.npy")
    if pose_bank is None:
        rotations, basin = stable_pose_bank(points, faces)
    else:
        rotations = [np.asarray(p["rotation"]) for p in pose_bank]
        basin = [p["resting_basin_probability"] for p in pose_bank]
    from gr00t.rl.pick.planning import object_frame_grasp_bank

    tool = object_frame_grasp_bank(torch.tensor([[1.0, 0, 0, 0]], device=device), size)[0]
    if placement_profile == "basket":
        axis = _quat_rotate(tool, tool.new_tensor([1, 0, 0]))
        tool = tool[axis[:, 2] <= -math.cos(math.radians(30))]
    counts, top_counts, basket_counts, diagnostics = [], [], [], []
    # Offline library preprocessing is chunked independently of scene/env control.
    for rotation in rotations:
        support = baked_pose_support(points, faces, rotation, device)
        pose = support.new_tensor([[1, 0, 0, 0]])
        count = top_count = basket_count = 0
        gates = {key: 0 for key in ("pocket_fit", "antipodal", "level_pinch", "clearance")}
        best_clearance = -math.inf
        for chunk in tool.split(128):
            orientation = chunk[None]
            quality = convex_grasp_quality(
                orientation,
                pose,
                support,
                depth_clearance=support_plane_clearance(
                    orientation, pose, support, command_seating_offset_m=0.0005
                ),
                minimum_environment_clearance_m=minimum_clearance_m,
                support_normal_world=support.new_tensor([[0, 0, 1]]),
                return_depth_candidates=True,
            )
            pocket_fit = quality["pocket_fit"]
            antipodal = pocket_fit & (quality["antipodal_margin"] >= 0)
            feasible = antipodal & quality["support_pinch_compatible"]
            clearance = quality["environment_clearance_m"]
            for key, mask in (
                ("pocket_fit", pocket_fit),
                ("antipodal", antipodal),
                ("level_pinch", feasible),
                ("clearance", feasible & (clearance >= minimum_clearance_m)),
            ):
                gates[key] += int(mask.sum())
            best_clearance = max(
                best_clearance, float(clearance.masked_fill(~feasible, -torch.inf).amax())
            )
            accepted = (feasible & (quality["environment_clearance_m"] >= minimum_clearance_m)).any(
                -1
            )
            basket_accepted = (
                feasible
                & (quality["environment_clearance_m"] >= minimum_clearance_m)
                & (quality["center_depth_margin_m"] >= BASKET_MINIMUM_CENTER_DEPTH_M)
            ).any(-1)
            axis = _quat_rotate(chunk, chunk.new_tensor([1, 0, 0]))
            top_down = axis[:, 2] <= -math.cos(math.radians(30))
            count += int(accepted.sum())
            top_count += int((accepted & top_down[None]).sum())
            basket_count += int((basket_accepted & top_down[None]).sum())
        counts.append(count)
        top_counts.append(top_count)
        basket_counts.append(basket_count)
        diagnostics.append(
            {
                "depth_candidate_gate_counts": gates,
                "best_clearance_after_contact_checks_m": best_clearance
                if math.isfinite(best_clearance)
                else None,
                "blocking_gate": next((key for key, value in gates.items() if not value), None),
            }
        )
    result.update(
        eligible_shape=True,
        poses=[
            {
                "rotation": rotation.tolist(),
                "resting_basin_probability": probability,
                "supported_grasp_count": int(count),
                "top_down_grasp_count": int(top_count),
                "basket_top_down_grasp_count": int(basket_count),
                "diagnostics": diagnostic,
            }
            for rotation, probability, count, top_count, basket_count, diagnostic in zip(
                rotations, basin, counts, top_counts, basket_counts, diagnostics
            )
        ],
        physical_grasp_verified=False,
    )
    return result


def shape_bank_fingerprint(asset, *, minimum_clearance_m=GRASP_CLEARANCE_M, placement_profile=None):
    digest = hashlib.sha256(
        b"joint-support-plane-v5-all-hull-and-face-samples-depth-bank-preload-0.5mm"
    )
    digest.update(json.dumps([minimum_clearance_m, placement_profile]).encode())
    digest.update(json.dumps(BASKET_MINIMUM_CENTER_DEPTH_M).encode())
    directory = Path(asset["usd_path"]).parent
    for name in ("collision_support.npy", "collision_faces.npy"):
        digest.update((directory / name).read_bytes())
    for points in gripper_collision_clouds("cpu"):
        digest.update(points.numpy().tobytes())
    digest.update(json.dumps(asset.get("admitted_scale_xyz"), sort_keys=True).encode())
    return digest.hexdigest()


def basket_supported_shapes(shapes):
    return [
        {
            "scale_xyz": shape["scale_xyz"],
            "poses": [
                {
                    **pose,
                    "supported_grasp_count": pose["basket_top_down_grasp_count"],
                    "top_down_grasp_count": pose["basket_top_down_grasp_count"],
                }
                for pose in shape["poses"]
                if pose["basket_top_down_grasp_count"] > 0
            ],
        }
        for shape in shapes
        if any(pose["basket_top_down_grasp_count"] > 0 for pose in shape["poses"])
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--asset-json", type=Path, help="Audit an actual baked variant")
    parser.add_argument("--all-admitted-scales", action="store_true")
    parser.add_argument("--minimum-clearance-m", type=float, default=GRASP_CLEARANCE_M)
    parser.add_argument("--placement-profile", choices=("basket",))
    parser.add_argument("--skip-published-shapes", action="store_true")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--publish-placement-bank",
        action="store_true",
        help="Publish stable faces; does not enable assets or certify physical success",
    )
    args = parser.parse_args()
    if not math.isfinite(args.minimum_clearance_m) or args.minimum_clearance_m < 0.002:
        parser.error("Minimum clearance must be finite and at least 0.002 m")
    if (
        args.publish_placement_bank
        and args.minimum_clearance_m != GRASP_CLEARANCE_M
        and args.placement_profile is None
    ):
        parser.error("Nondefault clearance audits are diagnostic only; do not overwrite the bank")
    if args.skip_published_shapes and not (
        args.all_admitted_scales and args.publish_placement_bank
    ):
        parser.error("Skipping published shapes requires a complete joint-bank publication")
    if not 0 <= args.shard_rank < args.shard_count:
        parser.error("Shard rank must be in [0, shard-count)")
    if args.asset_json is not None and (
        args.names or args.publish_placement_bank or args.all_admitted_scales
    ):
        parser.error("An actual variant audit cannot publish a source bank or select names")
    assets = (
        [json.loads(args.asset_json.read_text())]
        if args.asset_json is not None
        else installed_assets(args.root, include_disabled=True)
    )
    if args.names is not None:
        missing = set(args.names) - {asset["name"] for asset in assets}
        if missing:
            parser.error(f"Unknown installed assets: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for asset_index, asset in enumerate(assets):
        if asset_index % args.shard_count != args.shard_rank:
            continue
        if args.names is not None and asset["name"] not in args.names:
            continue
        fingerprint = (
            shape_bank_fingerprint(
                asset,
                minimum_clearance_m=args.minimum_clearance_m,
                placement_profile=args.placement_profile,
            )
            if args.all_admitted_scales
            else None
        )
        published = asset.get("stable_placement_audit", {})
        if args.placement_profile is not None:
            published = (
                asset.get("placement_profiles", {}).get(args.placement_profile, {}).get("audit", {})
            )
        if args.skip_published_shapes and (published.get("joint_input_sha256") == fingerprint):
            print(f"[pick:shape-audit] reuse {asset['name']}", flush=True)
            continue
        if args.all_admitted_scales:
            scales = asset.get("admitted_scale_xyz") or [[1.0, 1.0, 1.0]]
            shapes = []
            for index, scale in enumerate(scales):
                from gr00t.rl.batch_workflow import check_launch_lease

                check_launch_lease()
                print(f"[pick:shape-audit] {asset['name']} {index + 1}/{len(scales)}", flush=True)
                shapes.append(
                    audit_asset(
                        asset,
                        scale_xyz=scale,
                        minimum_clearance_m=args.minimum_clearance_m,
                        placement_profile=args.placement_profile,
                    )
                )
            result = {**shapes[0], "shape_results": shapes}
        else:
            result = audit_asset(
                asset,
                minimum_clearance_m=args.minimum_clearance_m,
                placement_profile=args.placement_profile,
            )
        if (
            args.publish_placement_bank
            and (result["poses"] or args.placement_profile)
            and asset.get("enabled", True)
        ):
            poses = [p for p in result["poses"] if p["supported_grasp_count"] > 0]
            # A finite furniture/basket edge can permit grasps rejected by an infinite plane.
            poses = poses or result["poses"]
            path = Path(asset["usd_path"]).with_name("asset.json")
            metadata = json.loads(path.read_text())
            if args.placement_profile is not None:
                shapes = result.get("shape_results", [result])
                admitted = basket_supported_shapes(shapes)
                metadata.setdefault("placement_profiles", {})[args.placement_profile] = {
                    "supported_shapes": admitted,
                    "audit": {
                        "support_model": result["support_model"],
                        "minimum_environment_clearance_m": args.minimum_clearance_m,
                        "minimum_center_depth_m": BASKET_MINIMUM_CENTER_DEPTH_M,
                        "admitted_shape_count": len(admitted),
                        "audited_shape_count": len(shapes),
                        "failure": None
                        if admitted
                        else "no_supported_top_down_candidate_at_required_clearance",
                        "joint_input_sha256": fingerprint,
                        "physical_grasp_verified": False,
                    },
                }
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(metadata, indent=2) + "\n")
                temporary.replace(path)
                results.append(result)
                (args.output / "supported_assets.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
                print(json.dumps(result), flush=True)
                continue
            metadata["stable_placement_rotations"] = [p["rotation"] for p in poses]
            metadata["stable_placement_audit"] = {
                "support_model": result["support_model"],
                "nominal_supported_pose_count": sum(
                    p["supported_grasp_count"] > 0 for p in result["poses"]
                ),
                "physical_grasp_verified": False,
            }
            if args.all_admitted_scales:
                metadata["supported_placement_shapes"] = [
                    {
                        "scale_xyz": shape["scale_xyz"],
                        "poses": [p for p in shape["poses"] if p["supported_grasp_count"] > 0],
                    }
                    for shape in result["shape_results"]
                    if any(p["supported_grasp_count"] > 0 for p in shape["poses"])
                ]
                metadata["stable_placement_audit"]["admitted_shape_count"] = len(
                    metadata["supported_placement_shapes"]
                )
                metadata["stable_placement_audit"]["joint_input_sha256"] = fingerprint
                from gr00t.rl.pick.basket import BASKET_GRASP_CLEARANCE_M

                basket_shapes = basket_supported_shapes(result["shape_results"])
                metadata.setdefault("placement_profiles", {})["basket"] = {
                    "supported_shapes": basket_shapes,
                    "audit": {
                        "support_model": result["support_model"],
                        "minimum_environment_clearance_m": BASKET_GRASP_CLEARANCE_M,
                        "minimum_center_depth_m": BASKET_MINIMUM_CENTER_DEPTH_M,
                        "admitted_shape_count": len(basket_shapes),
                        "audited_shape_count": len(result["shape_results"]),
                        "joint_input_sha256": shape_bank_fingerprint(
                            asset,
                            minimum_clearance_m=BASKET_GRASP_CLEARANCE_M,
                            placement_profile="basket",
                        ),
                        "physical_grasp_verified": False,
                    },
                }
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(metadata, indent=2) + "\n")
            temporary.replace(path)
        results.append(result)
        (args.output / "supported_assets.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
