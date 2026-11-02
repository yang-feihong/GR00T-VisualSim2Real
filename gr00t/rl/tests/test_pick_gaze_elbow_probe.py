"""Opt-in metadata counterfactual; does not claim exact replay of unrecorded states."""

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch


def elbow_candidates(roots, initial, target, limits, dt, offsets, roll_step_deg=45):
    from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

    from gr00t.rl.envs.door.b2z1_reset_geometry import quaternion_xyzw_to_matrix
    from gr00t.rl.pick import gaze

    batch, count = roots.shape[:2]
    # Joint-space proposals, not a Cartesian product of elbow/shoulder/roll grids.
    index = torch.arange(512, device=roots.device, dtype=roots.dtype)
    unit = (index[:, None] * roots.new_tensor([0.618033989, 0.414213562, 0.732050808])) % 1
    shoulder = (unit[:, 0] * 2 - 1) * math.pi
    elbow = (unit[:, 1] * 2 - 1) * math.pi
    rolls = (unit[:, 2] * 2 - 1) * math.pi
    candidates = len(index)
    body = roots[:, None].expand(-1, candidates, -1, -1).reshape(-1, count, 13)
    joints = initial[:, None, None].expand(-1, candidates, count, -1).clone()
    joints[..., 1] = (joints[..., 1] + shoulder[None, :, None]).clamp(*limits[1])
    joints[..., 3] = (joints[..., 3] + elbow[None, :, None]).clamp(*limits[3])
    p, q, template = gaze.q3_gaze_path(
        body,
        joints.reshape(-1, count, 6),
        target[:, None].expand(-1, candidates, -1).reshape(-1, 3),
        limits,
        dt,
        blend_s=0,
    )
    rotation = quaternion_xyzw_to_matrix(q[..., [1, 2, 3, 0]])
    mount_offset, mount_rotation = gaze.camera_mount_in_ee(p)
    camera, _ = gaze.camera_pose(p, rotation)
    direction = target[:, None, None].expand(-1, candidates, count, -1).reshape(-1, count, 3)
    aimed = gaze.aimed_rotation(rotation @ mount_rotation, direction - camera)
    rotation = quaternion_xyzw_to_matrix(aimed[..., [1, 2, 3, 0]]) @ mount_rotation.T
    p = camera - (rotation @ mount_offset[:, None]).squeeze(-1)
    noise = offsets[:, None].expand(-1, candidates, -1).clone()
    noise[..., 2] += rolls[None]
    p, q = gaze.perturb_camera_aim(
        p,
        _matrix_to_quaternion_wxyz(rotation),
        noise.reshape(-1, 3),
        dt,
    )
    # This probe uses total template travel as ranking cost, not a q2 diagnostic.
    return p, q, template, body, shoulder.abs() + elbow.abs(), rolls


@pytest.mark.skipif(not os.environ.get("PICK_GAZE_ELBOW_ROOT"), reason="Recorded probe required")
def test_metadata_gaze_elbow_counterfactual(monkeypatch, tmp_path):
    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory
    from gr00t.rl.pick import gaze
    from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
    from gr00t.rl.pick.collision import ArmCollisionSamples
    from gr00t.rl.pick.geometry_views import TranslatedGeometry
    from gr00t.rl.pick.scene_geometry import read_scene_triangles, static_scene_geometry
    from gr00t.rl.pick.scene_object_geometry import SceneObjectGeometryBatch
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    source = Path(os.environ["PICK_GAZE_ELBOW_ROOT"])
    paths = sorted(source.glob("logs/rank*-control/seed_*/episode.json"))
    episodes = [json.loads(path.read_text()) for path in paths]
    details = [json.loads(path.with_name("summary.json").read_text())["planning"] for path in paths]
    batch = len(paths)

    def tensor(value):
        return torch.tensor(value, device="cuda", dtype=torch.float32)

    offsets = gaze.sample_approach_offsets_batch([e["seed"] for e in episodes], "cuda")
    starts = tensor(
        [
            [*d["approach_gaze"]["base_path_start_m"][:2], e["base_yaw"]]
            for e, d in zip(episodes, details, strict=True)
        ]
    )
    ends = tensor(
        [
            d["initial_station"]["station_poses"][d["initial_station"]["selected"]]
            if d["initial_station"]["selected"] >= 0
            else [
                *d["approach_gaze"]["base_path_end_m"][:2],
                e["station_yaw"] + offsets[i, 0].item(),
            ]
            for i, (e, d) in enumerate(zip(episodes, details, strict=True))
        ]
    )
    trajectory = BasePoseTrajectory(batch, "cuda")
    trajectory.install(torch.arange(batch, device="cuda"), starts, ends[:, None], 0.02)
    count = int(trajectory.length.max())
    roots = torch.zeros(batch, count, 13, device="cuda")
    roots[..., :2] = trajectory.poses[:, :count, :2]
    roots[..., 2] = tensor([d["approach_gaze"]["base_path_start_m"][2] for d in details])[:, None]
    yaw = trajectory.poses[:, :count, 2] / 2
    roots[..., 3], roots[..., 6] = yaw.cos(), yaw.sin()
    initial = tensor([e["initial_arm_joints"] for e in episodes])
    target = tensor([e["object_position"] for e in episodes])
    rooms, views = {}, []
    for episode in episodes:
        path = episode["scene_usd_path"]
        if path not in rooms:
            points, faces, labels = read_scene_triangles(
                Path(path),
                Path(episode["geometry_cache_root"]),
                return_labels=True,
            )
            rooms[path] = static_scene_geometry(
                points,
                faces,
                "cuda",
                labels=labels,
                backend="native",
                ground_plane_z_m=0,
            )
        views.append(TranslatedGeometry(rooms[path], tensor(episode["batch_scene_origin_m"])))
    maximum = max(len(e.get("distractors", [])) for e in episodes)
    props = torch.zeros(batch, maximum, 13, device="cuda")
    props[..., 3] = 1
    for row, episode in enumerate(episodes):
        for slot, obj in enumerate(episode.get("distractors", [])):
            props[row, slot, :3] = tensor(obj["object_position"])
            yaw = obj["object_yaw"] / 2
            props[row, slot, 3], props[row, slot, 6] = math.cos(yaw), math.sin(yaw)
    state = {"root_states": roots[:, 0], "distractor_states": props}
    geometry = SceneObjectGeometryBatch(episodes, BatchedSceneGeometry.from_views(views)).update(
        state
    )
    model = torch.load(paths[0].with_name("collision_model.pt"), weights_only=False)
    collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
    collision.parts = [(i, points.cuda()) for i, points in model["parts"]]
    collision.gripper_surface = [points.cuda() for points in model["gripper_surface"]]
    limits = tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    original = gaze.natural_gaze_candidates
    report = {
        "physics_simulated": False,
        "gaze_source": str(gaze.__file__),
        "state_fidelity": (
            "Episode preset and yaw-only roots; actual settled state missing for rejected rows"
        ),
        "seeds": [e["seed"] for e in episodes],
        "cases": {},
    }
    cases = (
        (("production", original),)
        if os.environ.get("PICK_GAZE_PRODUCTION_ONLY")
        else (("q2_only", original), ("joint_sampled_elbow", elbow_candidates))
    )
    for name, candidates in cases:
        monkeypatch.setattr(gaze, "natural_gaze_candidates", candidates)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            _, _, result = gaze.plan_approach_gaze_batch(
                roots,
                initial,
                target,
                limits,
                0.02,
                collision,
                geometry,
                offsets[:, 1:],
                lengths=trajectory.length,
            )
        torch.cuda.synchronize()
        report["cases"][name] = {
            "elapsed_s": time.monotonic() - started,
            **{
                key: result[key].tolist()
                for key in (
                    "valid",
                    "minimum_clearance_m",
                    "minimum_ee_height_above_object_m",
                    "maximum_settled_position_error_m",
                    "minimum_settled_joint_margin_rad",
                )
            },
        }
    (tmp_path / "gaze_elbow_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    assert all(len(case["valid"]) == batch for case in report["cases"].values())
