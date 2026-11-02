"""Replay an approach gaze plan on GPU without starting the simulator."""

import argparse
import json
from pathlib import Path

import torch

from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory
from gr00t.rl.auto_open.motion_math import _quat_multiply
from gr00t.rl.envs.door.b2z1_reset_geometry import (
    b2z1_gripper_pose_and_jacobian,
    quaternion_xyzw_to_matrix,
)
from gr00t.rl.pick.collision import ArmCollisionSamples
from gr00t.rl.pick.gaze import (
    camera_mount_in_ee,
    camera_pose,
    natural_gaze_candidates,
    plan_approach_gaze,
    q3_gaze_path,
    sample_approach_offsets,
)
from gr00t.rl.pick.scene_geometry import SceneGeometry, read_scene_triangles
from gr00t.rl.utils.b2z1_asset import ROBOT_URDF


def free_roll_reach(roots, camera, axis, limits, frames, seed):
    """Offline five-DOF diagnostic; not a production pose-planning fallback."""
    indices = torch.as_tensor(frames, device=roots.device)
    heights = roots.new_tensor([-0.25, -0.18, -0.12, 0, 0.12, 0.18])
    samples = 512
    r = roots[0, indices, None, None].expand(-1, len(heights), samples, -1).clone()
    r[..., 2] += heights[None, :, None]
    r = r.reshape(-1, 13)
    goal = camera[0, indices, None, None].expand(-1, len(heights), samples, -1).reshape(-1, 3)
    direction = axis[0, indices, None, None].expand(-1, len(heights), samples, -1).reshape(-1, 3)
    generator = torch.Generator(device=roots.device).manual_seed(seed)
    q = limits[:, 0] + torch.rand((len(r), 6), generator=generator, device=roots.device) * (
        limits[:, 1] - limits[:, 0]
    )
    offset, _ = camera_mount_in_ee(goal)
    for _ in range(240):
        p, rotation, jacobian = b2z1_gripper_pose_and_jacobian(r[:, :3], r[:, [4, 5, 6, 3]], q)
        eye, optical = camera_pose(p, rotation)
        world_offset = (rotation @ offset[:, None]).squeeze(-1)
        omega = jacobian[:, 3:].transpose(-1, -2)
        eye_j = jacobian[:, :3] + torch.cross(omega, world_offset[:, None], dim=-1).transpose(
            -1, -2
        )
        axis_j = torch.cross(omega, optical[:, None], dim=-1).transpose(-1, -2)
        j = torch.cat((eye_j, 0.3 * axis_j), 1)
        error = torch.cat((goal - eye, 0.3 * (direction - optical)), -1)
        delta = (
            j.transpose(-1, -2)
            @ torch.linalg.solve(
                j @ j.transpose(-1, -2) + 1e-5 * torch.eye(6, device=q.device), error[..., None]
            )
        ).squeeze(-1)
        q = (q + delta.clamp(-0.15, 0.15)).clamp(limits[:, 0], limits[:, 1])
    p, rotation, _ = b2z1_gripper_pose_and_jacobian(r[:, :3], r[:, [4, 5, 6, 3]], q)
    eye, optical = camera_pose(p, rotation)
    error = (goal - eye).norm(dim=-1)
    angle = torch.acos((direction * optical).sum(-1).clamp(-1, 1))
    score = (error + 0.1 * angle).reshape(len(frames), -1)
    best = score.argmin(-1)
    row = torch.arange(len(frames), device=roots.device)
    return {
        "frames": frames,
        "position_error_m": error.reshape(len(frames), -1)[row, best].tolist(),
        "optical_angle_error_rad": angle.reshape(len(frames), -1)[row, best].tolist(),
        "arm_joints_rad": q.reshape(len(frames), -1, 6)[row, best].tolist(),
        "base_height_offset_m": heights[best // samples].tolist(),
        "random_joint_seeds_per_height": samples,
        "seed": seed,
        "note": "Local multistart diagnostic only; failure does not prove infeasibility.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--free-roll-frames", type=int, nargs="+")
    parser.add_argument("--roll-step-deg", type=float, default=45)
    parser.add_argument("--extra-joint-seeds", type=int, default=0)
    parser.add_argument("--station-proposal", type=Path)
    args = parser.parse_args()
    if not 0 < args.roll_step_deg <= 180 or args.extra_joint_seeds < 0:
        parser.error("roll step must be in (0, 180]; extra joint seed count must be nonnegative")
    episode = json.loads((args.run / "episode.json").read_text())
    state = torch.load(args.run / "trajectory.pt", weights_only=False)[0]
    root = state["root_states"].cuda()
    model = torch.load(args.run / "collision_model.pt", weights_only=False)
    collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
    collision.parts = [(i, p.cuda()) for i, p in model["parts"]]
    points, faces = read_scene_triangles(
        Path(episode["scene_usd_path"]), Path(episode["geometry_cache_root"])
    )
    geometry = SceneGeometry(points, faces)
    path = BasePoseTrajectory(1, "cuda:0")
    yaw = root.new_tensor([episode["base_yaw"]])
    start = torch.cat((root[:, :2], yaw[:, None]), -1)
    destination = root.new_tensor([*episode["station_xy"], episode["station_yaw"]])[None, None]
    offsets = sample_approach_offsets(episode["seed"], root.device)
    destination[..., 2] += offsets[0]
    next_joints = None
    if args.station_proposal:
        proposal = json.loads(args.station_proposal.read_text())
        if proposal.get("selected_station") is not None:
            destination = root.new_tensor(proposal["selected_station"])[:, None]
            next_joints = root.new_tensor(proposal["selected_endpoint_joints_rad"])[None, 0]
    path.install(torch.tensor([0], device=root.device), start, destination, 0.02)
    count = int(path.length.max().item())
    poses = path.poses[:, :count]
    roots = root[:, None].expand(-1, count, -1).clone()
    roots[..., :2] = poses[..., :2]
    delta = poses[..., 2] - yaw[:, None]
    q = torch.stack(((delta / 2).cos(), delta * 0, delta * 0, (delta / 2).sin()), -1)
    roots[..., 3:7] = _quat_multiply(q, roots[..., 3:7])
    limits = root.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    initial = state["dof_pos"][:, model["arm_joint_ids"]].cuda()
    template_p, template_q, template = q3_gaze_path(
        roots, initial, root.new_tensor([episode["object_position"]]), limits, 0.02
    )
    if args.geometry_only or args.free_roll_frames:
        p, q, _, _, shoulder, _ = natural_gaze_candidates(
            roots,
            initial,
            root.new_tensor([episode["object_position"]]),
            limits,
            0.02,
            offsets[None, 1:],
        )
        p, q = p[:1], q[:1]
        camera, axis = camera_pose(p, quaternion_xyzw_to_matrix(q[..., [1, 2, 3, 0]]))
        diagnostics = {
            "seed": episode["seed"],
            "camera_position_m": camera.tolist(),
            "camera_axis": axis.tolist(),
            "base_position_m": roots[..., :3].tolist(),
            "base_orientation_wxyz": roots[..., 3:7].tolist(),
            "q3_template_deg": torch.rad2deg(template[..., 2]).tolist(),
        }
        if args.free_roll_frames:
            diagnostics = free_roll_reach(
                roots, camera, axis, limits, args.free_roll_frames, episode["seed"]
            )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(diagnostics, indent=2) + "\n")
            print(str(args.output))
        else:
            print(json.dumps(diagnostics))
        return
    _, _, diagnostics = plan_approach_gaze(
        roots,
        initial,
        root.new_tensor([episode["object_position"]]),
        limits,
        0.02,
        collision,
        geometry,
        offsets[None, 1:],
        roll_step_deg=args.roll_step_deg,
        extra_joint_seeds=args.extra_joint_seeds,
        next_grasp_joints=next_joints,
    )
    diagnostics = {
        "seed": episode["seed"],
        "q3_template_deg": torch.rad2deg(template[..., 2]).tolist(),
        **diagnostics,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(diagnostics, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "valid": diagnostics["valid"]}))
    else:
        print(json.dumps(diagnostics))


if __name__ == "__main__":
    main()
