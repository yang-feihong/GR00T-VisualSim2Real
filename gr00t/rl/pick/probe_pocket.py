"""Offline fixed-wrist pocket-position candidates, without a simulator."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
from gr00t.rl.pick.collision import ArmCollisionSamples
from gr00t.rl.pick.planning import inverse_bank, pocket_pose_targets
from gr00t.rl.pick.pocket import gripper_pocket_geometry
from gr00t.rl.pick.scene_geometry import SceneGeometry, read_scene_triangles
from gr00t.rl.utils.b2z1_asset import ROBOT_URDF


def pocket_position_solve(roots, joints, offsets, target, limits, iterations=100):
    """Generate candidates with a point task; q6 is fixed, not commanded."""
    fixed_wrist = joints[:, 5].clone()
    identity = torch.eye(3, device=joints.device, dtype=joints.dtype)
    for _ in range(iterations):
        position, rotation, jacobian = b2z1_gripper_pose_and_jacobian(
            roots[:, :3], roots[:, [4, 5, 6, 3]], joints
        )
        lever = (rotation @ offsets[..., None]).squeeze(-1)
        angular = jacobian[:, 3:, :5].transpose(-1, -2)
        point_jacobian = jacobian[:, :3, :5] + torch.cross(
            angular, lever[:, None], dim=-1
        ).transpose(-1, -2)
        error = target - position - lever
        delta = (
            point_jacobian.transpose(-1, -2)
            @ torch.linalg.solve(
                point_jacobian @ point_jacobian.transpose(-1, -2) + identity * 1e-4,
                error[..., None],
            )
        ).squeeze(-1)
        joints[:, :5] = (joints[:, :5] + delta.clamp(-0.12, 0.12)).clamp(
            limits[:5, 0], limits[:5, 1]
        )
    joints[:, 5] = fixed_wrist
    return joints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--policy-probe", type=Path, required=True)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--initial-frame", type=int, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--joint-margin-rad", type=float, default=0.03)
    parser.add_argument("--audit-quality-only", action="store_true")
    parser.add_argument("--audit-candidates", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    if not 0 <= args.joint_margin_rad < 0.2:
        parser.error("joint margin must be in [0, 0.2)")
    torch.set_grad_enabled(False)
    device = "cuda:0"
    episode = json.loads((args.run / "episode.json").read_text())
    trace = torch.load(args.run / "trajectory.pt", weights_only=False, map_location=device)
    state = trace[args.initial_frame]
    model = torch.load(args.run / "collision_model.pt", weights_only=False, map_location=device)
    collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
    collision.parts = model["parts"]
    collision.arm_joint_ids = model["arm_joint_ids"]
    collision.gripper_surface = model["gripper_surface"]
    vertices, faces = read_scene_triangles(
        Path(episode["scene_usd_path"]), Path(episode["geometry_cache_root"])
    )
    geometry = SceneGeometry(vertices, faces, device)
    supports = torch.tensor(
        np.load(Path(episode["asset"]["usd_path"]).with_name("collision_support.npy")),
        device=device,
    )
    target, object_quat = state["object_state"][:, :3], state["object_state"][:, 3:7]
    measurement = json.loads((args.policy_probe / "summary.json").read_text())
    i = args.candidate
    if args.audit_quality_only:
        from gr00t.rl.auto_open.planning_geometry import _quat_rotate
        from gr00t.rl.envs.door.b2z1_reset_geometry import _arm_offset_tensors

        indices = args.audit_candidates or [i]
        quats = target.new_tensor(measurement["goal_quaternions_wxyz"])[indices][None]
        positions, quality = pocket_pose_targets(
            quats, target, object_quat, supports, collision, geometry
        )
        quality["opened_clearance_m"] = collision.gripper_clearance(geometry, positions, quats)
        quality["closing_clearance_m"] = collision.gripper_closing_clearance(
            geometry, positions, quats, quality["aperture_angle_rad"]
        )
        offsets = _arm_offset_tensors(positions.dtype, positions.device)
        transform = positions.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        for name, points in zip(("fixed_body", "moving_jaw"), collision.gripper_surface):
            local = (points - offsets[7] - offsets[8]) @ transform
            quality[name + "_clearance_m"] = geometry.rigid_surface_clearance(
                positions, quats, local
            )
            world = positions[..., None, :] + _quat_rotate(quats[..., None, :], local[None, None])
            low = world[..., 2].argmin(-1)
            quality[name + "_lowest_point_world"] = world.gather(
                -2, low[..., None, None].expand(-1, -1, 1, 3)
            ).squeeze(-2)
        result = {
            "note": "Collider geometry only, not measured contact",
            "indices": indices,
            "positions": positions.cpu().tolist(),
            "quality": {key: value.cpu().tolist() for key, value in quality.items()},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[pick:pocket-audit] {args.output}", flush=True)
        return
    reference_q = target.new_tensor(measurement["goal_quaternions_wxyz"][i])[None, None]
    original_goal, _ = pocket_pose_targets(
        reference_q, target, object_quat, supports, collision, geometry
    )
    translation = target.new_tensor(measurement["goals_world"][i]) - original_goal[0, 0]
    root = target.new_tensor(measurement["final_root_states"][i])[None]
    root[:, :3] -= translation
    roots = root.expand(args.samples, -1).clone()
    limits = target.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    # Leave a real joint margin while searching; do not relax any production limit.
    search_limits = limits + limits.new_tensor([args.joint_margin_rad, -args.joint_margin_rad])
    initial = state["dof_pos"][:, collision.arm_joint_ids]
    generator = torch.Generator(device=device).manual_seed(episode["seed"])
    joints = search_limits[:, 0] + torch.rand(
        (args.samples, 6), generator=generator, device=device
    ) * (search_limits[:, 1] - search_limits[:, 0])
    joints[:, 5] = initial[0, 5]
    measured = torch.load(args.policy_probe / "trace.pt", weights_only=False, map_location=device)
    joints[0, :5] = measured[-1, i, 20:25].clamp(search_limits[:5, 0], search_limits[:5, 1])
    pocket = gripper_pocket_geometry(device)
    radius = min(episode["asset"]["size_m"]) / 2
    offset = (
        torch.stack(
            (
                pocket["back_x"] + radius + 0.0015,
                pocket["back_x"] * 0,
                pocket["fixed_plane_z"] + radius,
            )
        )
        .expand(args.samples, -1)
        .clone()
    )
    stages = []
    for iteration in range(3):
        joints = pocket_position_solve(roots, joints, offset, target, search_limits)
        position, rotation, _ = b2z1_gripper_pose_and_jacobian(
            roots[:, :3], roots[:, [4, 5, 6, 3]], joints
        )
        quats = _matrix_to_quaternion_wxyz(rotation)
        point_error = (position + (rotation @ offset[..., None]).squeeze(-1) - target).norm(dim=-1)
        goals, quality = pocket_pose_targets(
            quats[None], target, object_quat, supports, collision, geometry
        )
        residual = (position - goals[0]).norm(dim=-1)
        physical = quality["pocket_fit"][0] & (quality["antipodal_margin"][0] >= 0)
        physical &= quality["environment_clearance_m"][0] >= 0.002
        stages.append(
            {
                "iteration": iteration,
                "point_task_exact_count": int((point_error < 0.003).sum()),
                "minimum_point_task_error_m": float(point_error.min()),
                "physical": int(physical.sum()),
                "position_and_geometry": int((physical & (residual < 0.003)).sum()),
                "minimum_physical_error_m": float(residual.masked_fill(~physical, torch.inf).min()),
            }
        )
        print(f"[pick:pocket-probe] {json.dumps(stages[-1])}", flush=True)
        offset = quality["command_center_local"][0].clone()
    arm_clearance = collision.clearance(geometry, roots, joints, maximum_distance_m=0.0021)
    valid = physical & (residual < 0.003) & (arm_clearance >= 0.002)
    # Final certification uses the same six-DOF inverse as the main workflow.
    solved = inverse_bank(goals[0, :, None], quats[:, None], roots, joints, limits)
    exact = solved.exact.reshape(-1)
    wrist_departure = (solved.arm_joint_positions[:, 5] - initial[0, 5]).abs()
    valid &= exact & (wrist_departure < math.radians(10))
    score = residual + 0.001 * (joints - initial).norm(dim=-1)
    selected = score.masked_fill(~valid, torch.inf).argsort()[:16]
    selected = selected[valid[selected]]
    result = {
        "note": "Offline candidates only; no physical grasp or policy response proof",
        "source_run": str(args.run),
        "policy_probe": str(args.policy_probe),
        "candidate": i,
        "root_state": root[0].cpu().tolist(),
        "samples": args.samples,
        "joint_margin_rad": args.joint_margin_rad,
        "stages": stages,
        "certified_candidates": int(valid.sum()),
        "selected_indices": selected.cpu().tolist(),
        "candidate_orientations_world_wxyz": quats[selected][None].cpu().tolist(),
        "candidate_positions_world": goals[0, selected].cpu().tolist(),
        "candidate_joint_positions_rad": joints[selected].cpu().tolist(),
        "fixed_wrist_rad": float(initial[0, 5]),
        "candidate_position_error_m": residual[selected].cpu().tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "certified": result["certified_candidates"],
                "stages": stages,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
