"""Reconstruct a recorded panel candidate bank without launching a simulator."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pxr import Usd
from scipy.spatial.transform import Rotation
import torch

from gr00t.rl.auto_open.geometry import ROBOT_COLLIDER_SPECS, _read_collider_points
from gr00t.rl.auto_open.planning_ik import solve_candidate_ik
from gr00t.rl.auto_open.tensor_stage_planning import (
    align_vectors,
    _quat_rotate,
    panel_forward_anchor_distance,
    panel_handle_height_mask,
)
from gr00t.rl.envs.door.b2z1_reset_geometry import quaternion_xyzw_to_matrix
from legged_gym.utils.ee_steady_inverse_gpu import batched_steady_ee_inverse_no_gravity


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--radius", type=float, default=0.20)
    parser.add_argument(
        "--first-contact",
        action="store_true",
        help="Replay first MOVE_TO_PANEL_STANDOFF entry with current baseline and height rule.",
    )
    args = parser.parse_args()
    shard = args.dataset / "shards" / f"episode-seed{args.seed}"
    metadata = json.loads(
        next(args.dataset.glob(f"logs/rank*-control/seed_{args.seed}/metadata.json")).read_text()
    )
    provenance = json.loads((shard / "meta/simulation_provenance.json").read_text())
    episode = next(iter(provenance["episodes"].values()))
    rows = pq.read_table(
        next(shard.rglob("raw_audit/chunk-000/episode_000000.parquet"))
    ).to_pylist()
    row = (
        next(r for r in rows if r["autoopen_stage"] == "MOVE_TO_PANEL_STANDOFF")
        if args.first_contact
        else rows[-1]
    )
    if args.first_contact:
        q = np.asarray(row["root_quaternion_xyzw"])
        root = np.r_[
            row["root_position_world"],
            q[[3, 0, 1, 2]],
            row["root_linear_velocity_world"],
            row["root_angular_velocity_world"],
        ]
        tool = np.asarray(row["gripper_ee_pose_world_xyzw"])
    else:
        root = np.asarray(row["post_root_state"])
        tool = np.asarray(row["post_gripper_ee_pose_world_xyzw"])
    arm_ids = [episode["robot_joint_names"].index(f"joint{i}") for i in range(1, 7)]
    arm = np.asarray(
        row["all_joint_position" if args.first_contact else "post_all_joint_position"]
    )[arm_ids]
    prims = episode["door_asset"]["prims"]
    hinge_prim = next(p["attributes"] for p in prims if p["path"].endswith("/hinge_joint"))
    panel = next(p["attributes"] for p in prims if p["path"].endswith("/door_panel/panel"))
    origin = np.asarray(metadata["robot_state"]["initial_base_pose_w"][:3]) - np.asarray(
        metadata["robot_state"]["initial_base_pose_env"][:3]
    )
    hinge = np.asarray(hinge_prim["physics:localPos0"]) + origin
    hq = hinge_prim.get("physics:localRot0", dict(real=1.0, imaginary=[0.0, 0.0, 0.0]))
    sign = np.sign(Rotation.from_quat(hq["imaginary"] + [hq["real"]]).apply([0.0, 0.0, 1.0])[2])
    center = np.asarray(panel["xformOp:translate"]) + origin
    free = center - hinge
    free[[0, 2]] = 0.0
    free /= np.linalg.norm(free)
    stage = Usd.Stage.Open(provenance["robot_asset"]["path"])
    joints = {p.GetName(): p for p in stage.Traverse() if p.GetName().startswith("joint")}
    limits = np.asarray(
        [
            [
                math.radians(joints[f"joint{i}"].GetAttribute("physics:lowerLimit").Get()),
                math.radians(joints[f"joint{i}"].GetAttribute("physics:upperLimit").Get()),
            ]
            for i in range(1, 7)
        ]
    )
    clouds = {}
    for body_name, key, tokens in ROBOT_COLLIDER_SPECS:
        if key not in ("fixed_finger", "moving_finger"):
            continue
        body = next(
            p
            for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
            if p.GetName().lower() == body_name.lower()
        )
        clouds[key] = _read_collider_points(stage, body, tokens).points_body
    fixed = Rotation.from_euler("x", math.pi / 2.0).apply(clouds["fixed_finger"] - [0.1, 0.0, 0.0])
    moving = Rotation.from_euler("y", row["gripper_feedback"]).apply(clouds["moving_finger"])

    def tensor(value):
        return torch.as_tensor(np.asarray(value), dtype=torch.float64, device=args.device)

    angle = sign * row["door_hinge_position" if args.first_contact else "post_door_hinge_position"]
    rotation = tensor([[math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]])
    normal = _quat_rotate(rotation, tensor([[1.0, 0.0, 0.0]]))
    side = -1.0 if float(((tensor(root[:3] - hinge)) * normal[0]).sum()) < 0.0 else 1.0
    normal *= side
    q = tensor(tool[[6, 3, 4, 5]][None])
    depth = np.array([1.0, 0.0, 0.0])
    if args.first_contact:
        initial = rows[0]
        initial_moving = Rotation.from_euler("y", initial["gripper_feedback"]).apply(
            clouds["moving_finger"]
        )
        separation = initial_moving.mean(0) - fixed.mean(0)
        separation /= np.linalg.norm(separation)
        depth = (
            Rotation.from_quat(initial["gripper_ee_pose_world_xyzw"][3:])
            .inv()
            .apply([-1.0, 0.0, 0.0])
        )
        depth -= separation * (depth @ separation)
        depth /= np.linalg.norm(depth)
    orientation = align_vectors(_quat_rotate(q, tensor(depth[None])), -normal, q)
    points = _quat_rotate(orientation[:, None], tensor(np.concatenate((fixed, moving)))[None])
    minimum = (points * normal[:, None]).sum(-1).amin(-1)
    push = hinge + 0.4 * free
    push[2] = center[2] - panel["xformOp:scale"][2] + 1.0
    push += (center[0] - push[0]) * np.array([1.0, 0.0, 0.0])
    half = panel["xformOp:scale"][0]
    surface = tensor(hinge)[None] + _quat_rotate(
        rotation, tensor(push - hinge + [half * side, 0.0, 0.0])[None]
    )
    baseline = surface + (0.03 - minimum)[:, None] * normal
    tangent = _quat_rotate(rotation, tensor(free)[None])
    nominal_distance = 0.4
    if args.first_contact:
        nominal_distance = float(
            panel_forward_anchor_distance(tensor(root)[None], tensor(hinge)[None], tangent, 0.38)[0]
        )
        baseline += (nominal_distance - 0.4) * tangent
    bound = round(args.radius / 0.05)
    height_bound = min(bound, 4) if args.first_contact else bound
    offsets = tensor(
        [
            [a * 0.05, b * 0.05]
            for a in range(-bound, bound + 1)
            for b in range(-height_bound, height_bound + 1)
        ]
    )
    order = torch.argsort(offsets[:, 1].abs(), stable=True)
    order = order[torch.argsort(offsets[order, 0].abs(), stable=True)]
    order = order[torch.argsort(offsets[order].norm(dim=-1), stable=True)]
    offsets = offsets[order]
    candidates = baseline + offsets[:, :1] * tangent
    candidates[:, 2] += offsets[:, 1]
    width = float((center - hinge) @ free) + panel["xformOp:scale"][1]
    valid = (
        (offsets[:, 0] + nominal_distance >= 0.05)
        & (offsets[:, 0] + nominal_distance <= width - 0.05)
        & (offsets[:, 1] + 1.0 >= 0.05)
        & (offsets[:, 1] + 1.0 <= 2.0 * panel["xformOp:scale"][2] - 0.05)
    )
    candidates, offsets = candidates[valid], offsets[valid]
    count = len(candidates)
    base = tensor(root)[None].expand(count, -1)
    initial = tensor(arm)[None].expand(count, -1)
    target_q = orientation.expand(count, -1)
    lower, upper = tensor(limits[:, 0]), tensor(limits[:, 1])
    solved, position, rotation_error = solve_candidate_ik(
        base[:, :3],
        base[:, [4, 5, 6, 3]],
        candidates,
        quaternion_xyzw_to_matrix(target_q[:, [1, 2, 3, 0]]),
        initial,
        lower.expand(count, -1),
        upper.expand(count, -1),
        iterations=120,
        damping=0.03,
    )
    margin = torch.minimum(solved - lower, upper - solved).amin(-1)
    passed = (position <= 1.0e-4) & (rotation_error <= 2.0e-4) & (margin >= 0.002)
    old_first = torch.where(passed)[0]
    old_first = int(old_first[0]) if len(old_first) else None
    if args.first_contact:
        from gr00t.rl.scripts.render_auto_open_collision_section import load_episode, handle_frame

        _, audit, _, geometry, _, _ = load_episode(Path.cwd(), args.dataset, args.seed)
        center, axis, _, _ = handle_frame(geometry, audit, row["frame_index"])
        handle = next(p["attributes"] for p in prims if p["path"].endswith("/handle_outside"))
        half_height = abs(axis[2]) * handle["height"] / 2 + handle["radius"]
        band = tensor([[center[2] - half_height, center[2] + half_height]])
        jaw = torch.stack((points[..., 2].amin(-1), points[..., 2].amax(-1)), -1)
        clear = panel_handle_height_mask(candidates[:, 2][None], jaw, band)[0]
        passed &= clear
    result = batched_steady_ee_inverse_no_gravity(
        candidates,
        target_q,
        base_position_world=base[:, :3],
        base_orientation_world_xyzw=base[:, [4, 5, 6, 3]],
        initial_arm_joint_positions=initial,
        joint_limits=tensor(limits),
    )
    inverse_position = result.position_error.norm(dim=-1)
    inverse_orientation = result.orientation_error.norm(dim=-1)
    best = int((position / 1.0e-4 + rotation_error / 2.0e-4).argmin())
    selected = torch.where(passed)[0]
    selected = int(selected[0]) if len(selected) else None
    print(
        json.dumps(
            dict(
                seed=args.seed,
                side=side,
                root=root.tolist(),
                old_selected_offset=None if old_first is None else offsets[old_first].tolist(),
                new_selected_offset=None if selected is None else offsets[selected].tolist(),
                new_selected_position=None if selected is None else candidates[selected].tolist(),
                selected_inverse_position_m=None
                if selected is None
                else float(inverse_position[selected]),
                selected_inverse_orientation_rad=None
                if selected is None
                else float(inverse_orientation[selected]),
                candidate_count=count,
                ik_pass=int(passed.sum()),
                best_offset=offsets[best].tolist(),
                feasible_offsets=offsets[passed].tolist(),
                best_position_error=float(position[best]),
                best_orientation_error=float(rotation_error[best]),
                best_joint_margin=float(margin[best]),
                best_joints=solved[best].tolist(),
                inverse_pass=int(
                    ((inverse_position <= 0.010) & (inverse_orientation <= 1.0e-4)).sum()
                ),
                inverse_best_position=float(inverse_position.min()),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
