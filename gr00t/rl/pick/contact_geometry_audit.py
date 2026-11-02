"""Offline signed jaw/object gaps at the end of each grasp attempt."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gr00t.rl.auto_open.motion_math import _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate, _slerp
from gr00t.rl.pick.collision import ArmCollisionSamples
from gr00t.rl.pick.scene_geometry import SceneGeometry


def cartesian_transit_audit(run):
    """Compare a world Cartesian transit with the recorded rejected FK transit."""
    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.pick.distractors import augment_scene_geometry, build_distractor_mesh
    from gr00t.rl.pick.planning import inverse_bank
    from gr00t.rl.pick.scene_geometry import read_scene_triangles
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    episode = json.loads((run / "episode.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if summary.get("task_type") != "pick_external":
        raise ValueError("Transit comparison currently requires an external pick run")
    state = {
        key: value.cuda()
        for key, value in torch.load(run / "trajectory.pt", weights_only=False)[-1].items()
    }
    model = torch.load(run / "collision_model.pt", weights_only=False)
    collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
    collision.parts = [(index, points.cuda()) for index, points in model["parts"]]
    collision.arm_joint_ids = model["arm_joint_ids"].cuda()
    collision.gripper_surface = [points.cuda() for points in model["gripper_surface"]]
    recorded = summary["planning"]["grasp"]
    indices = state["ee_pos"].new_tensor(recorded["inverse_candidate_indices"], dtype=torch.long)
    quats = state["ee_pos"].new_tensor(recorded["candidate_orientations_world_wxyz"])[0, indices]
    center = state["ee_pos"].new_tensor(recorded["convex_contact_model"]["command_center_local"])[
        0, indices
    ]
    goal = state["object_state"][:, :3] - _quat_rotate(quats, center)
    pitch = recorded["capture_axis_pitch_rad"]
    axis = _quat_rotate(quats, goal.new_tensor([np.cos(pitch), 0, -np.sin(pitch)]))
    pre = goal - goal.new_tensor(recorded["pregrasp_standoff_m"])[:, None] * axis
    count = int(recorded["path_samples"][0])
    t = torch.linspace(0, 1, count, device=goal.device)
    positions = state["ee_pos"][:, None] + t[None, :, None] * (pre - state["ee_pos"])[:, None]
    orientations = _slerp(
        state["ee_orn"][:, None].expand(len(pre), count, -1),
        quats[:, None].expand(-1, count, -1),
        t[None],
    )
    points, faces = read_scene_triangles(
        Path(episode["scene_usd_path"]), Path(episode["geometry_cache_root"])
    )
    geometry = augment_scene_geometry(
        SceneGeometry(points, faces),
        build_distractor_mesh(episode.get("distractors", []), state["distractor_states"]),
    )
    target_geometry = augment_scene_geometry(
        geometry, build_distractor_mesh([episode], state["object_state"][:, None])
    )
    scene_clearance = collision.gripper_clearance(geometry, positions, orientations)
    target_clearance = collision.gripper_clearance(target_geometry, positions, orientations)
    elevated = []
    limits = torch.as_tensor(
        read_arm_joint_limits(ROBOT_URDF).copy(), device=goal.device, dtype=goal.dtype
    )
    for height in (0.35, 0.45, 0.55, 0.65):
        high_start = state["ee_pos"].expand_as(pre).clone()
        high_end = pre.clone()
        high_start[:, 2] = torch.maximum(high_start[:, 2], pre.new_tensor(height))
        high_end[:, 2] = high_start[:, 2]
        knots = (state["ee_pos"].expand_as(pre), high_start, high_end, pre)
        rotations = (
            state["ee_orn"].expand_as(quats),
            state["ee_orn"].expand_as(quats),
            quats,
            quats,
        )
        paths, path_quats = [], []
        for start, end, q0, q1 in zip(knots[:-1], knots[1:], rotations[:-1], rotations[1:]):
            paths.append(start[:, None] + t[None, :, None] * (end - start)[:, None])
            path_quats.append(
                _slerp(
                    q0[:, None].expand(-1, count, -1), q1[:, None].expand(-1, count, -1), t[None]
                )
            )
        path, path_q = torch.cat(paths, 1), torch.cat(path_quats, 1)
        gaps = collision.gripper_clearance(target_geometry, path, path_q)
        sample = torch.linspace(0, path.shape[1] - 1, 49, device=goal.device).round().long()
        heights = goal.new_tensor([state["root_states"][0, 2], 0.46, 0.42, 0.38, 0.34])
        actual = state["dof_pos"][0, collision.arm_joint_ids]
        seeds = torch.stack((actual, limits.mean(-1), limits.sum(-1) - actual))
        roots = state["root_states"].expand(len(heights), len(seeds), len(pre), -1).clone()
        roots[..., 2] = heights[:, None, None]
        roots = roots.reshape(-1, 13)
        p = (
            path[:, sample][None, None]
            .expand(len(heights), len(seeds), -1, -1, -1)
            .reshape(-1, len(sample), 3)
        )
        q = (
            path_q[:, sample][None, None]
            .expand(len(heights), len(seeds), -1, -1, -1)
            .reshape(-1, len(sample), 4)
        )
        joints = seeds[None, :, None].expand(len(heights), -1, len(pre), -1).reshape(-1, 6)
        result = inverse_bank(p, q, roots, joints, limits)
        arm_gaps = collision.clearance(
            geometry,
            roots[:, None].expand(-1, len(sample), -1).reshape(-1, 13),
            result.arm_joint_positions,
            include_gripper=False,
        ).reshape(len(heights), len(seeds), len(pre), len(sample))
        exact = result.exact.reshape_as(arm_gaps)
        valid = (exact & (arm_gaps >= 0.002)).any(0).any(0)
        elevated.append(
            {
                "height_m": height,
                "minimum_clearance_m": gaps.amin(-1).tolist(),
                "inverse_and_arm_valid_per_point": valid.tolist(),
                "full_inverse_and_arm_valid": valid.all(-1).tolist(),
                "best_position_error_per_point_m": result.position_error.norm(dim=-1)
                .reshape_as(arm_gaps)
                .amin(0)
                .amin(0)
                .tolist(),
            }
        )
    report = {
        "source_run": str(run),
        "candidate_indices": indices.tolist(),
        "recorded_fk_transit_clearance_m": recorded["gripper_clearance_by_segment_m"]["approach"],
        "cartesian_scene_min_clearance_m": scene_clearance.amin(-1).tolist(),
        "cartesian_scene_initial_clearance_m": scene_clearance[:, 0].tolist(),
        "cartesian_scene_final_clearance_m": scene_clearance[:, -1].tolist(),
        "cartesian_scene_and_target_min_clearance_m": target_clearance.amin(-1).tolist(),
        "cartesian_scene_clearance_per_point_m": scene_clearance.tolist(),
        "not_arm_inverse_or_collision_certified": True,
        "elevated_transit_candidates": elevated,
    }
    torch.save(
        {
            "approach": (
                positions.cpu(),
                orientations.cpu(),
                torch.full((len(pre),), count, dtype=torch.long),
            )
        },
        run / "cartesian_transit.pt",
    )
    return report


def carry_force_audit(trace, summary, model, *, stage="LIFT_OBJECT", stop_at_stage_exit=False):
    frame = 0
    start = None
    stop = len(trace)
    for event in summary["transitions"]:
        frame += event["step"]
        if event["to"] == stage and start is None:
            start = frame
            if not stop_at_stage_exit:
                break
        elif start is not None and stop_at_stage_exit and event["from"] == stage:
            stop = frame
            break
    if start is None:
        return None
    first = max(0, start - 1)
    states = trace[first:stop]

    def values(key):
        return torch.cat([row[key] for row in states]).cuda()

    inverse = values("ee_orn").clone()
    inverse[:, 1:] *= -1
    obj, ee = values("object_state"), values("ee_pos")
    centers = _quat_rotate(inverse, obj[:, :3] - ee)
    relative_orientation = _quat_multiply(inverse, obj[:, 3:7])
    relative_rotation = (
        2 * (relative_orientation * relative_orientation[:1]).sum(-1).abs().clamp_max(1).acos()
    )
    normal = _quat_rotate(inverse[:, None], values("object_finger_forces"))
    friction = _quat_rotate(inverse[:, None], values("object_finger_friction_forces"))
    contacts = _quat_rotate(inverse[:, None], values("object_finger_contact_points") - ee[:, None])
    index = int(model["gripper_joint_ids"][0])
    jaw = values("dof_pos")[:, index]
    from gr00t.rl.pick.pocket import jaw_distal_limits

    edges = jaw_distal_limits(-jaw)
    return {
        "first_frame": first,
        "last_frame_exclusive": stop,
        "stage": stage,
        "simulation_dt_s": summary["simulation_dt_s"],
        "force_convention": "normal/friction forces on robot jaws in measured EE frame",
        "object_center_local_m": centers.tolist(),
        "object_orientation_local_wxyz": relative_orientation.tolist(),
        "object_relative_rotation_rad": relative_rotation.tolist(),
        "center_depth_behind_shorter_jaw_m": (edges.amin(-1) - centers[:, 0]).tolist(),
        "contact_points_local_m": contacts.tolist(),
        "normal_forces_ee_n": normal.tolist(),
        "normal_forces_world_n": values("object_finger_forces").tolist(),
        "friction_forces_ee_n": friction.tolist(),
        "friction_forces_world_n": values("object_finger_friction_forces").tolist(),
        "object_position_world_m": obj[:, :3].tolist(),
        "object_orientation_world_wxyz": obj[:, 3:7].tolist(),
        "ee_position_world_m": ee.tolist(),
        "ee_orientation_world_wxyz": values("ee_orn").tolist(),
        "ee_command_world_m": values("ee_goal_cart_world").tolist(),
        "ee_command_orientation_world_wxyz": values("ee_goal_orn_quat_world").tolist(),
        "friction_to_normal_ratio": (
            friction.norm(dim=-1) / normal.norm(dim=-1).clamp_min(1e-8)
        ).tolist(),
        "jaw_angle_rad": jaw.tolist(),
        "actuator_target_rad": values("gripper_actuator_target").tolist(),
        "contact_hold_active": values("gripper_contact_hold_active").tolist(),
        "object_velocity_world_m_s": obj[:, 7:10].tolist(),
        "ee_tracking_error_world_m": (ee - values("ee_goal_cart_world")).tolist(),
    }


def audit(run):
    run = Path(run)
    episode = json.loads((run / "episode.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    trace = torch.load(run / "trajectory.pt", weights_only=False)
    model = torch.load(run / "collision_model.pt", weights_only=False)
    collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
    collision.parts = [(index, points.cuda()) for index, points in model["parts"]]
    collision.arm_joint_ids = model["arm_joint_ids"].cuda()
    collision.gripper_surface = [points.cuda() for points in model["gripper_surface"]]
    fixed, moving = collision._gripper_local_surfaces(torch.empty(0, device="cuda:0"))
    folder = Path(episode["asset"]["usd_path"]).parent
    surface = SceneGeometry(
        np.load(folder / "collision_support.npy"), np.load(folder / "collision_faces.npy")
    )
    end, planned_at, frames = 0, 0, []
    for event in summary["transitions"]:
        if event["to"] == "MOVE_TO_GRASP":
            planned_at = end + event["step"]
        end += event["step"]
        if event["from"] == "GRASP_OBJECT":
            frames.append((min(end - 1, len(trace) - 1), planned_at))
    if (summary.get("failure") or {}).get("stage") == "GRASP_OBJECT":
        frames.append((len(trace) - 1, planned_at))
    reports = []
    entries = []
    for frame, start in frames:
        rows = trace[start : frame + 1]
        roots = torch.cat([row["root_states"] for row in rows]).cuda()
        objects = torch.cat([row["object_state"] for row in rows]).cuda()
        inverse_objects = objects[:, 3:7].clone()
        inverse_objects[:, 1:] *= -1
        relative_roots = roots.clone()
        relative_roots[:, :3] = _quat_rotate(inverse_objects, roots[:, :3] - objects[:, :3])
        relative_roots[:, 3:7] = _quat_multiply(inverse_objects, roots[:, 3:7])
        joints = torch.cat([row["dof_pos"] for row in rows]).cuda()[:, collision.arm_joint_ids]
        arm_gap = collision.clearance(
            surface, relative_roots, joints, include_gripper=False, maximum_distance_m=0.0021
        )
        blocked = (arm_gap < 0.002).nonzero().flatten()
        commands = torch.cat([row["ee_goal_cart_world"] for row in rows]).cuda()
        command_quats = torch.cat([row["ee_goal_orn_quat_world"] for row in rows]).cuda()
        original_inverse = inverse_objects[:1]
        relative_commands = _quat_rotate(original_inverse, commands - objects[:1, :3])
        relative_quats = _quat_multiply(original_inverse, command_quats)
        open_gap = surface.rigid_surface_clearance(
            relative_commands[:, None],
            relative_quats[:, None],
            torch.cat((fixed, moving)),
            check_rays=False,
            signed_distance=True,
        ).flatten()
        penetrating = (open_gap < -0.001).nonzero().flatten()
        entries.append(
            {
                "first_frame": start,
                "last_frame": frame,
                "arm_excluding_jaws_target_obstructed_frames": int(len(blocked)),
                "first_arm_target_obstruction_frame": None
                if not len(blocked)
                else start + int(blocked[0]),
                "minimum_arm_target_clearance_m": float(arm_gap.min()),
                "command_open_jaw_target_minimum_signed_gap_m": float(open_gap.min()),
                "command_open_jaw_target_final_signed_gap_m": float(open_gap[-1]),
                "first_command_jaw_penetration_over_1mm_frame": (
                    None if not len(penetrating) else start + int(penetrating[0])
                ),
            }
        )
        row = {key: value.cuda() for key, value in trace[frame].items()}
        inverse = row["object_state"][:, 3:7].clone()
        inverse[:, 1:] *= -1
        aperture = -row["dof_pos"][:, int(model["gripper_joint_ids"][0])]
        delta = (torch.pi / 3 - aperture) / 2
        close = torch.stack((delta.cos(), delta * 0, delta.sin(), delta * 0), -1)
        gaps = {}
        for label, position, quat in (
            ("actual", row["ee_pos"], row["ee_orn"]),
            ("command", row["ee_goal_cart_world"], row["ee_goal_orn_quat_world"]),
        ):
            p = _quat_rotate(inverse, position - row["object_state"][:, :3])
            q = _quat_multiply(inverse, quat)
            for jaw, local, orientation in (
                ("fixed", fixed, q),
                ("moving", moving, _quat_multiply(q, close)),
            ):
                gaps[label + "_" + jaw + "_signed_gap_m"] = float(
                    surface.rigid_surface_clearance(
                        p[:, None],
                        orientation[:, None],
                        local,
                        check_rays=False,
                        signed_distance=True,
                    )[0, 0]
                )
        reports.append(
            {
                "frame": frame,
                "planned_at_frame": start,
                **gaps,
                "ee_tracking_error_world_m": (row["ee_pos"] - row["ee_goal_cart_world"])[
                    0
                ].tolist(),
                "object_shift_since_planning_world_m": (
                    row["object_state"][:, :3] - trace[start]["object_state"][:, :3].cuda()
                )[0].tolist(),
                "jaw_force_n": row["object_finger_forces"].norm(dim=-1)[0].tolist(),
                "actual_jaw_angle_rad": -float(aperture[0]),
                "actuator_target_rad": row["gripper_actuator_target"][0].tolist(),
                "contact_hold_active": bool(row["gripper_contact_hold_active"][0]),
            }
        )
    return {
        "run": str(run),
        "definition": "positive=gap, negative=sampled jaw surface inside target convex hull",
        "note": "Uses planning hull and sampled jaw surfaces, not PhysX penetration depth",
        "grasp_attempts": reports,
        "arm_target_geometry": entries,
        "carry_force_audit": carry_force_audit(trace, summary, model),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage")
    parser.add_argument("--compare-cartesian-transit", action="store_true")
    args = parser.parse_args()
    if args.compare_cartesian_transit:
        if args.stage:
            parser.error("Choose a force stage or a Cartesian transit comparison")
        report = cartesian_transit_audit(args.run)
        output = args.run / "cartesian_transit_audit.json"
    elif args.stage:
        summary = json.loads((args.run / "summary.json").read_text())
        trace = torch.load(args.run / "trajectory.pt", weights_only=False)
        model = torch.load(args.run / "collision_model.pt", weights_only=False)
        report = carry_force_audit(trace, summary, model, stage=args.stage, stop_at_stage_exit=True)
        output = args.run / f"force_audit_{args.stage}.json"
    else:
        report = audit(args.run)
        output = args.run / "jaw_clearance_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output)}) if args.stage else json.dumps(report))


if __name__ == "__main__":
    main()
