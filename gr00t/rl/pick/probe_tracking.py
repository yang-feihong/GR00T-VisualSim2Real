"""Offline ideal-servo replay of recorded single-step IK versus an FK joint arc."""

import argparse
import json
from pathlib import Path

import torch
import warp as wp
from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz
from legged_gym.utils.ee_steady_inverse_kernel import (
    V6,
    _constrained_delta,
    _fk,
    _orientation_error,
)

from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
from gr00t.rl.auto_open.pose_planning import solve_pose_targets
from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
from gr00t.rl.pick.collision import ArmCollisionSamples
from gr00t.rl.pick.scene_geometry import SceneGeometry, read_scene_triangles
from gr00t.rl.utils.b2z1_asset import ROBOT_URDF


@wp.kernel
def _replay(
    positions: wp.array3d(dtype=wp.float64),
    orientations: wp.array3d(dtype=wp.float64),
    roots: wp.array2d(dtype=wp.float64),
    initial: wp.array(dtype=wp.float64),
    limits: wp.array2d(dtype=wp.float64),
    damping: wp.float64,
    solved: wp.array3d(dtype=wp.float64),
    errors: wp.array3d(dtype=wp.float64),
):
    row = wp.tid()
    arm, lower, upper = V6(), V6(), V6()
    for j in range(6):
        arm[j] = initial[j]
        lower[j] = limits[j, 0]
        upper[j] = limits[j, 1]
    for step in range(positions.shape[1]):
        base = wp.vec3d(roots[step, 0], roots[step, 1], roots[step, 2])
        base_q = wp.normalize(
            wp.quatd(roots[step, 4], roots[step, 5], roots[step, 6], roots[step, 3])
        )
        point = wp.vec3d(positions[row, step, 0], positions[row, step, 1], positions[row, step, 2])
        quat = wp.normalize(
            wp.quatd(
                orientations[row, step, 1],
                orientations[row, step, 2],
                orientations[row, step, 3],
                orientations[row, step, 0],
            )
        )
        position, orientation, jacobian = _fk(base, base_q, arm)
        rotation_error = _orientation_error(quat, orientation)
        error = V6()
        for j in range(3):
            error[j] = point[j] - position[j]
            error[j + 3] = rotation_error[j]
        arm = arm + _constrained_delta(jacobian, error, arm, lower, upper, damping)
        position, orientation, jacobian = _fk(base, base_q, arm)
        rotation_error = _orientation_error(quat, orientation)
        errors[row, step, 0] = wp.length(point - position)
        errors[row, step, 1] = wp.float64(2.0) * wp.asin(
            wp.min(
                wp.float64(1.0),
                wp.length(wp.vec3d(rotation_error[0], rotation_error[1], rotation_error[2])),
            )
        )
        for j in range(6):
            solved[row, step, j] = arm[j]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--stage", default="MOVE_TO_GRASP")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run / "summary.json").read_text())
    trace = torch.load(args.run / "trajectory.pt", weights_only=False)
    model = torch.load(args.run / "collision_model.pt", weights_only=False)
    start = 0
    for event in summary["transitions"]:
        start += event["step"]
        if event["to"] == args.stage:
            break
    else:
        raise ValueError(f"No recorded entry for stage {args.stage}")
    trace = trace[start:]
    roots = torch.cat([s["root_states"] for s in trace]).cuda()
    positions = torch.cat([s["ee_goal_cart_world"] for s in trace]).cuda()
    quats = torch.cat([s["ee_goal_orn_quat_world"] for s in trace]).cuda()
    initial = trace[0]["dof_pos"][0, model["arm_joint_ids"]].cuda()
    limits = roots.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    sample = torch.arange(1, 33, device=roots.device, dtype=roots.dtype)[:, None]
    fraction = torch.frac(
        sample * roots.new_tensor([0.618034, 0.414214, 0.732051, 0.236068, 0.645751, 0.316625])
    )
    seeds = torch.cat(
        (
            initial[None],
            limits.mean(-1)[None],
            limits[:, 0] + fraction * (limits[:, 1] - limits[:, 0]),
        ),
        0,
    )
    solved = solve_pose_targets(
        positions[-1:].expand(len(seeds), -1),
        quats[-1:].expand(len(seeds), -1),
        base_position_world=roots[-1:, :3].expand(len(seeds), -1),
        base_orientation_world_xyzw=roots[-1:, [4, 5, 6, 3]].expand(len(seeds), -1),
        initial_arm_joint_positions=seeds,
        joint_limits=limits,
    )
    cost = (solved.arm_joint_positions - initial).norm(dim=-1)
    cost = cost.masked_fill(~solved.exact, torch.inf)
    if not torch.isfinite(cost).any().item():
        result = {
            "endpoint_exact": False,
            "best_position_error_m": float(solved.position_error.norm(dim=-1).min()),
        }
    else:
        endpoint = solved.arm_joint_positions[cost.argmin()]
        t = torch.linspace(0, 1, len(trace), device=roots.device)
        t = t.pow(3) * (10 + t * (-15 + 6 * t))
        arc_joints = initial + t[:, None] * (endpoint - initial)
        p, r, _ = b2z1_gripper_pose_and_jacobian(roots[:, :3], roots[:, [4, 5, 6, 3]], arc_joints)
        reference_p = torch.stack((positions, p)).double().contiguous()
        reference_q = torch.stack((quats, _matrix_to_quaternion_wxyz(r))).double().contiguous()
        result_joints = torch.empty((2, len(trace), 6), device=roots.device, dtype=torch.float64)
        error = torch.empty((2, len(trace), 2), device=roots.device, dtype=torch.float64)
        values = [
            reference_p,
            reference_q,
            roots.double().contiguous(),
            initial.double(),
            limits.double().contiguous(),
        ]
        wp.launch(
            _replay,
            dim=2,
            inputs=[
                *[wp.from_torch(v) for v in values],
                args.damping,
                wp.from_torch(result_joints),
                wp.from_torch(error),
            ],
            stream=wp.stream_from_torch(torch.cuda.current_stream()),
        )
        result = {
            "endpoint_exact": True,
            "initial_joints_rad": initial.tolist(),
            "endpoint_joints_rad": endpoint.tolist(),
            "final_position_error_m": error[:, -1, 0].tolist(),
            "final_orientation_error_rad": error[:, -1, 1].tolist(),
            "peak_position_error_m": error[..., 0].amax(-1).tolist(),
            "replayed_final_joints_rad": result_joints[:, -1].tolist(),
            "modes": ["recorded_cartesian_path", "quintic_joint_fk_arc"],
        }
        episode = json.loads((args.run / "episode.json").read_text())
        points, faces = read_scene_triangles(
            Path(episode["scene_usd_path"]), Path(episode["geometry_cache_root"])
        )
        geometry = SceneGeometry(points, faces)
        collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
        collision.parts = [(i, cloud.cuda()) for i, cloud in model["parts"]]
        collision.gripper_surface = [cloud.cuda() for cloud in model["gripper_surface"]]
        result["joint_fk_arc_gripper_minimum_clearance_m"] = float(
            collision.gripper_clearance(
                geometry, p[None], _matrix_to_quaternion_wxyz(r)[None]
            ).min()
        )
        result["joint_fk_arc_arm_minimum_surface_distance_m"] = float(
            collision.clearance(geometry, roots, arc_joints).min()
        )
        torch.save(
            {
                "roots": roots.cpu(),
                "joint_fk_arc": arc_joints.cpu(),
                "positions": reference_p.cpu(),
                "orientations": reference_q.cpu(),
            },
            args.output.with_suffix(".pt"),
        )
    result["note"] = "Offline ideal arm servo with recorded base motion; not physical validation."
    result["stage"] = args.stage
    result["first_frame"] = start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
