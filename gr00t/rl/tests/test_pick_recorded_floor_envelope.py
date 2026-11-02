"""Audit floor rejections using exact linear extrema, without physics or relaxed gates."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.skipif(not os.environ.get("PICK_GRASP_FAILURE_ROOT"), reason="Recorded batch required")
def test_recorded_floor_box_against_real_collider_extrema(tmp_path):
    from scipy.spatial import ConvexHull

    from gr00t.rl.auto_open.motion_math import _quat_multiply
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.pick.collision import ArmCollisionSamples

    directory = Path(os.environ["PICK_GRASP_FAILURE_ROOT"])

    def numeric(value):
        return torch.from_numpy(np.asarray(value, dtype=np.float32))

    rows, quats, centers, apertures, targets, floors, recorded = [], [], [], [], [], [], []
    seeds, model_path = [], None
    for path in sorted(directory.glob("logs/rank*-control/seed_*/summary.json")):
        summary = json.loads(path.read_text())
        if summary["failure"] != "no_safe_grasp_candidate":
            continue
        if summary["target_semantics"]["support_category"] != "floor":
            continue
        grasp = summary["planning"]["grasp"]
        quality = grasp["convex_contact_model"]
        pocket = torch.tensor(quality["pocket_fit"]).flatten()
        margin = numeric(quality["antipodal_margin"]).flatten()
        selected = torch.where(pocket & (margin >= 0))[0]
        if not len(selected):
            continue
        q = numeric(grasp["candidate_orientations_world_wxyz"]).reshape(-1, 4)[selected]
        center = numeric(quality["command_center_local"]).reshape(-1, 3)[selected]
        angle = numeric(quality["aperture_angle_rad"]).flatten()[selected]
        episode = json.loads(path.with_name("episode.json").read_text())
        record = torch.load(
            path.with_name("control_record.pt"), map_location="cpu", weights_only=False
        )
        position = record["post"]["object_state"][-1, :3]
        rows.append(torch.full((len(selected),), len(seeds), dtype=torch.long))
        seeds.append(summary["seed"])
        quats.append(q)
        centers.append(center)
        apertures.append(angle)
        targets.append(position.expand(len(selected), 3))
        floors.append(torch.full((len(selected),), episode["support_z"]))
        recorded.append(numeric(quality["environment_clearance_m"]).flatten()[selected])
        model_path = model_path or path.with_name("collision_model.pt")
    assert seeds
    owner, q, center, angle, target, floor, old = [
        torch.cat(items).cuda()
        for items in (rows, quats, centers, apertures, targets, floors, recorded)
    ]
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    arm = ArmCollisionSamples.__new__(ArmCollisionSamples)
    arm.parts = [(index, cloud.cuda()) for index, cloud in model["parts"]]
    arm.gripper_surface = [cloud.cuda() for cloud in model["gripper_surface"]]
    fixed, moving = arm._gripper_local_surfaces(target)

    def extrema_cloud(points):
        # An affine minimum over a convex hull equals the minimum over all source points.
        hull = ConvexHull(points.cpu().numpy())
        return points[torch.tensor(hull.vertices, device="cuda", dtype=torch.long)]

    fixed, moving = extrema_cloud(fixed), extrema_cloud(moving)
    assert max(len(fixed), len(moving)) <= 10_000
    dummy = SimpleNamespace(
        visible=lambda start, end, **kw: torch.ones(
            start.shape[:-1], device="cuda", dtype=torch.bool
        )
    )
    arm._moving_envelope_intersects(dummy, target[:1, None], q[:1, None])
    corners = arm._moving_envelope_corners
    tcp_z = (target - _quat_rotate(q, center))[:, 2] - floor
    inverse = q.clone()
    inverse[:, 1:] *= -1
    up = _quat_rotate(inverse, q.new_tensor([0, 0, 1]))
    actual = tcp_z + (up @ fixed.T).amin(-1)
    box = actual.clone()
    for weight in torch.linspace(0, 1, 7, device="cuda"):
        delta = (torch.pi / 3 - angle) * weight
        rotation = torch.stack(((delta / 2).cos(), delta * 0, (delta / 2).sin(), delta * 0), -1)
        local_q = _quat_multiply(q, rotation)
        local_q[:, 1:] *= -1
        normal = _quat_rotate(local_q, q.new_tensor([0, 0, 1]))
        actual = torch.minimum(actual, tcp_z + (normal @ moving.T).amin(-1))
        box = torch.minimum(box, tcp_z + (normal @ corners.T).amin(-1))
    safe = actual >= 0.002
    false_positive = safe & (box < -0.0005) & (old < 0)
    counts = torch.zeros(len(seeds), device="cuda", dtype=torch.long)
    counts.index_add_(0, owner, false_positive.long())
    report = {
        "physics_simulated": False,
        "only_floor_checked": True,
        "other_obstacles_and_ik_not_certified": True,
        "source_batch": str(directory),
        "seeds": seeds,
        "source_candidates": len(owner),
        "fixed_extreme_vertices": len(fixed),
        "moving_extreme_vertices": len(moving),
        "safe_actual_floor_but_box_crosses_counts": counts.cpu().tolist(),
        "rows_with_false_floor_rejection": int((counts > 0).sum()),
    }
    (tmp_path / "floor_envelope_audit.json").write_text(json.dumps(report, indent=2))
