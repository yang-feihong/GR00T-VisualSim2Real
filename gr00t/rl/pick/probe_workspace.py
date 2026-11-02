"""Offline reachability audit of support height versus base-to-object distance."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
from gr00t.rl.auto_open.motion_math import _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.grasp_clearance import GRASP_CLEARANCE_M
from gr00t.rl.pick.planning import grasp_approach_origin, grasp_pose_bank, inverse_bank
from gr00t.rl.utils.b2z1_asset import ROBOT_URDF


def wrist_route_metrics(wrist, boundaries):
    endpoints = wrist[boundaries]
    travel = endpoints.diff().abs().sum()
    return {
        "phase_endpoint_q6_deg": torch.rad2deg(endpoints),
        "phase_endpoint_travel_deg": torch.rad2deg(travel),
        "phase_endpoint_backtracking_deg": torch.rad2deg(
            (travel - (wrist[-1] - wrist[0]).abs()).clamp_min(0)
        ),
        "episode_q6_span_deg": torch.rad2deg(wrist.max() - wrist.min()),
        "raw_framewise_travel_deg": torch.rad2deg(wrist.diff().abs().sum()),
    }


def probe_recorded_wrist_route(run):
    summary = json.loads((run / "summary.json").read_text())
    trace = torch.load(run / "trajectory.pt", weights_only=False)
    model = torch.load(run / "collision_model.pt", weights_only=False)
    wrist = torch.cat([state["dof_pos"][:, model["arm_joint_ids"][-1]] for state in trace]).cuda()
    lengths = torch.tensor([event["step"] for event in summary["transitions"]], device=wrist.device)
    boundaries = torch.cat(
        (lengths.new_zeros(1), lengths.cumsum(0) - 1, lengths.new_tensor([len(trace) - 1]))
    ).clamp(0, len(trace) - 1)
    return {
        "source_run": str(run),
        "boundary_frames": boundaries.tolist(),
        "boundary_stages": ["INITIAL"]
        + [event["from"] for event in summary["transitions"]]
        + ["FINAL"],
        **{key: value.tolist() for key, value in wrist_route_metrics(wrist, boundaries).items()},
        "note": "Raw finite-joint angles, without periodic wrapping. Framewise travel includes "
        "small solver/physical motion; endpoint backtracking is not itself proof "
        "of unnecessary motion.",
    }


def probe_recorded_states(run, frames):
    """Expose measured posture and FK consistency without frozen-posture inference."""
    from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian

    trace = torch.load(run / "trajectory.pt", weights_only=False)
    model = torch.load(run / "collision_model.pt", weights_only=False)
    selected = [trace[index] for index in frames]
    root = torch.cat([state["root_states"] for state in selected]).cuda()
    joints = torch.cat([state["dof_pos"] for state in selected])[:, model["arm_joint_ids"]].cuda()
    actual = torch.cat([state["ee_pos"] for state in selected]).cuda()
    goal = torch.cat([state["ee_goal_cart_world"] for state in selected]).cuda()
    object_pose = torch.cat([state["object_state"] for state in selected]).cuda()
    w, x, y, z = root[:, 3:7].unbind(-1)
    rpy = torch.stack(
        (
            torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            (2 * (w * y - z * x)).clamp(-1, 1).asin(),
            torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ),
        -1,
    )
    position, _, _ = b2z1_gripper_pose_and_jacobian(root[:, :3], root[:, [4, 5, 6, 3]], joints)
    return {
        "source_run": str(run),
        "frames": frames,
        "root_position_m": root[:, :3].tolist(),
        "root_rpy_deg": torch.rad2deg(rpy).tolist(),
        "arm_joints_deg": torch.rad2deg(joints).tolist(),
        "actual_ee_m": actual.tolist(),
        "command_ee_m": goal.tolist(),
        "object_pose_world": object_pose[:, :7].tolist(),
        "fk_minus_actual_ee_m": (position - actual).tolist(),
        "actual_minus_command_ee_m": (actual - goal).tolist(),
    }


def probe_fresh_station_bank(
    run, frame, wrist_band_deg, yaw_offsets_deg, body_heights_m=None, shifts_m=None
):
    """Rebuild pocket poses for each station; certify endpoints, not a full path."""
    from gr00t.rl.pick.collision import ArmCollisionSamples
    from gr00t.rl.pick.planning import pocket_pose_targets
    from gr00t.rl.pick.scene_geometry import SceneGeometry, read_scene_triangles

    episode = json.loads((run / "episode.json").read_text())
    state = torch.load(run / "trajectory.pt", weights_only=False)[frame]
    model = torch.load(run / "collision_model.pt", weights_only=False)
    collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
    collision.parts = [(index, p.cuda()) for index, p in model["parts"]]
    collision.arm_joint_ids = model["arm_joint_ids"].cuda()
    collision.gripper_surface = [p.cuda() for p in model["gripper_surface"]]
    vertices, faces = read_scene_triangles(
        Path(episode["scene_usd_path"]), Path(episode["geometry_cache_root"])
    )
    geometry = SceneGeometry(vertices, faces)
    root = state["root_states"].cuda()
    target = state["object_state"][:, :3].cuda()
    actual = state["dof_pos"][:, model["arm_joint_ids"]].cuda()
    shift, yaw, height = torch.meshgrid(
        root.new_tensor(
            [-0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20] if shifts_m is None else shifts_m
        ),
        root.new_tensor(yaw_offsets_deg) * (math.pi / 180),
        root.new_tensor(
            [0.22, 0.30, 0.38, 0.46, 0.536] if body_heights_m is None else body_heights_m
        ),
        indexing="ij",
    )
    shift, yaw, height = shift.flatten(), yaw.flatten(), height.flatten()
    roots = root.expand(len(shift), -1).clone()
    toward = target[:, :2] - root[:, :2]
    toward /= toward.norm(dim=-1, keepdim=True)
    roots[:, :2] += shift[:, None] * toward
    roots[:, 2] = height
    yaw_quat = torch.stack(((yaw / 2).cos(), yaw * 0, yaw * 0, (yaw / 2).sin()), -1)
    roots[:, 3:7] = _quat_multiply(yaw_quat, roots[:, 3:7])
    w, x, y, z = roots[:, 3:7].unbind(-1)
    headings = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    _, quats, axes = grasp_pose_bank(
        target.expand(len(roots), -1),
        headings,
        episode["asset"]["size_m"],
        grasp_approach_origin(roots),
    )
    support = root.new_tensor(
        np.load(Path(episode["asset"]["usd_path"]).with_name("collision_support.npy"))
    )
    # Identical world orientations share the same pocket geometry in this scene.
    unique_quats, inverse_quats = torch.unique(quats.reshape(-1, 4), dim=0, return_inverse=True)
    quality_quats = unique_quats[None]

    unique_goals, quality = pocket_pose_targets(
        quality_quats,
        target,
        state["object_state"][:, 3:7].cuda(),
        support,
        collision,
        geometry,
    )
    quality = {
        key: value[0, inverse_quats].reshape(*quats.shape[:-1], *value.shape[2:])
        for key, value in quality.items()
    }
    goals = unique_goals[0, inverse_quats].reshape(*quats.shape[:-1], 3)
    points = torch.stack((goals - 0.08 * axes, goals), -2)
    physical = (
        quality["pocket_fit"]
        & (quality["antipodal_margin"] >= 0)
        & (quality["environment_clearance_m"] >= GRASP_CLEARANCE_M)
    )
    station, candidate = torch.where(physical)
    limits = root.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    if wrist_band_deg is not None:
        band = math.radians(wrist_band_deg)
        limits[5, 0] = torch.maximum(limits[5, 0], actual[0, 5] - band)
        limits[5, 1] = torch.minimum(limits[5, 1], actual[0, 5] + band)
    seeds = torch.cat((actual, limits.mean(-1)[None], limits.sum(-1)[None] - actual))
    result = inverse_bank(
        points[station, candidate, None].expand(-1, len(seeds), -1, -1).flatten(0, 1),
        quats[station, candidate, None, None].expand(-1, len(seeds), 2, -1).flatten(0, 1),
        roots[station, None].expand(-1, len(seeds), -1).flatten(0, 1),
        seeds[None].expand(len(station), -1, -1).flatten(0, 1),
        limits,
    )
    exact_points = result.exact.reshape(-1, len(seeds), 2)
    exact = exact_points.all(-1)
    pocket_index, seed_index = torch.where(exact)
    selected_station = station[pocket_index]
    selected_candidate = candidate[pocket_index]
    joints = result.arm_joint_positions.reshape(-1, len(seeds), 2, 6)[pocket_index, seed_index]
    safe = torch.zeros_like(exact)
    rejection = {}
    if len(pocket_index):
        poses = points[selected_station, selected_candidate]
        orientations = quats[selected_station, selected_candidate]
        arm_clearance = (
            collision.clearance(
                geometry,
                roots[selected_station, None].expand(-1, 2, -1).flatten(0, 1),
                joints.flatten(0, 1),
                include_gripper=False,
            )
            .reshape(-1, 2)
            .amin(-1)
        )
        opened = (
            collision.gripper_clearance(
                geometry,
                poses.flatten(0, 1)[None],
                orientations[:, None].expand(-1, 2, -1).flatten(0, 1)[None],
            )
            .reshape(-1, 2)
            .amin(-1)
        )
        closing = collision.gripper_closing_clearance(
            geometry,
            poses[:, -1][None],
            orientations[None],
            quality["aperture_angle_rad"][selected_station, selected_candidate][None],
        )[0]
        safe[pocket_index, seed_index] = (
            (arm_clearance >= 0.002) & (opened >= 0.002) & (closing >= 0.002)
        )
        rejection = {
            "arm_endpoint_clearance_failed": int((arm_clearance < 0.002).sum()),
            "open_gripper_endpoint_clearance_failed": int((opened < 0.002).sum()),
            "closing_sweep_clearance_failed": int((closing < 0.002).sum()),
        }
    counts = torch.zeros(len(roots), dtype=torch.int64, device=root.device)
    counts.scatter_add_(0, station, safe.any(-1).long())
    viable_pocket, viable_seed = torch.where(safe)
    solutions = result.arm_joint_positions.reshape(-1, len(seeds), 2, 6)[viable_pocket, viable_seed]
    solution_station = station[viable_pocket]
    solution_candidate = candidate[viable_pocket]
    return {
        "source_run": str(run),
        "source_frame": frame,
        "wrist_band_deg": wrist_band_deg,
        "station_xy_m": roots[:, :2].tolist(),
        "station_yaw_deg": torch.rad2deg(headings).tolist(),
        "shift_toward_object_m": shift.tolist(),
        "sampled_body_height_m": height.tolist(),
        "unique_pocket_orientations": len(unique_quats),
        "safe_pregrasp_and_grasp_count": counts.tolist(),
        "exact_pocket_seed_count": int(exact.sum()),
        "exact_pregrasp_seed_count": int(exact_points[..., 0].sum()),
        "exact_grasp_seed_count": int(exact_points[..., 1].sum()),
        "safe_pocket_seed_count": int(safe.sum()),
        "endpoint_rejections": rejection,
        "solutions": {
            "station_index": solution_station.tolist(),
            "candidate_index": solution_candidate.tolist(),
            "pregrasp_and_grasp_joints_rad": solutions.tolist(),
            "q6_max_departure_deg": torch.rad2deg(
                (solutions[..., 5] - actual[0, 5]).abs().amax(-1)
            ).tolist(),
            "pregrasp_and_grasp_position_m": points[solution_station, solution_candidate].tolist(),
            "orientation_world_wxyz": quats[solution_station, solution_candidate].tolist(),
        },
        "note": "Fresh poses and convex pocket fit. Endpoints only: body footprint, gaze, "
        "transition and lift paths are not certified; sampled body height is not predicted.",
    }


def probe_recorded_grasp_bank(
    run, standoff_sweep=False, frame=-1, wrist_band_deg=None, yaw_offsets_deg=(0.0,)
):
    """Audit the actual scene-safe pocket targets, batching every nearby station."""
    summary = json.loads((run / "summary.json").read_text())
    trace = torch.load(run / "trajectory.pt", weights_only=False)
    model = torch.load(run / "collision_model.pt", weights_only=False)
    episode = json.loads((run / "episode.json").read_text())
    start = next(e["step"] for e in summary["transitions"] if e["from"] == "APPROACH_OBJECT")
    entry = trace[start - 1]
    root = trace[frame]["root_states"].cuda()
    target = entry["object_state"][:, :3].cuda()
    entry_root = entry["root_states"].cuda()
    heading = torch.atan2(target[:, 1] - entry_root[:, 1], target[:, 0] - entry_root[:, 0])
    _, quats, axes = grasp_pose_bank(
        target, heading, episode["asset"]["size_m"], grasp_approach_origin(entry_root)
    )
    plan = summary["planning"]["grasp"]
    if "candidate_orientations_world_wxyz" in plan:
        quats = root.new_tensor(plan["candidate_orientations_world_wxyz"])
        axes = _quat_rotate(quats, root.new_tensor([1, 0, 0]).expand(*quats.shape[:-1], 3))
    center = root.new_tensor(plan["convex_contact_model"]["command_center_local"])
    goals = target[:, None] - _quat_rotate(quats, center)
    standoffs = torch.arange(0.08, 0.401, 0.04, device=root.device)
    points = (
        goals[:, :, None] - standoffs[None, None, :, None] * axes[:, :, None]
        if standoff_sweep
        else torch.stack((goals - 0.08 * axes, goals), -2)
    )
    count = points.shape[2]
    points = points.flatten(1, 2)
    quats = quats[:, :, None].expand(-1, -1, count, -1).flatten(1, 2)
    toward = target[:, :2] - root[:, :2]
    toward /= toward.norm(dim=-1, keepdim=True)
    shifts = (
        torch.zeros(2, device=root.device)
        if standoff_sweep
        else torch.arange(-0.10, 0.301, 0.025, device=root.device)
    )
    yaw_offsets = root.new_tensor(yaw_offsets_deg) * (math.pi / 180)
    shifts = shifts[:, None].expand(-1, len(yaw_offsets)).flatten()
    yaw_offsets = yaw_offsets[None].expand(len(shifts) // len(yaw_offsets), -1).flatten()
    limits = root.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    actual = trace[frame]["dof_pos"][:, model["arm_joint_ids"]].cuda()
    if wrist_band_deg is not None:
        band = math.radians(wrist_band_deg)
        limits[5, 0] = torch.maximum(limits[5, 0], actual[0, 5] - band)
        limits[5, 1] = torch.minimum(limits[5, 1], actual[0, 5] + band)
    seeds = torch.cat(
        (
            actual,
            entry["dof_pos"][:, model["arm_joint_ids"]].cuda(),
            limits.mean(-1)[None],
            limits.sum(-1)[None] - actual,
        ),
        0,
    )
    roots = root[:, None].expand(len(shifts), len(seeds), -1).clone()
    roots[..., :2] += shifts[:, None, None] * toward
    yaw_quat = torch.stack(
        ((yaw_offsets / 2).cos(), yaw_offsets * 0, yaw_offsets * 0, (yaw_offsets / 2).sin()), -1
    )
    roots[..., 3:7] = _quat_multiply(yaw_quat[:, None], roots[..., 3:7])
    if standoff_sweep:
        roots[..., 2] = torch.cat((root[:, 2], entry_root[:, 2]))[:, None]
    result = inverse_bank(
        points.expand(len(shifts) * len(seeds), -1, -1),
        quats.expand(len(shifts) * len(seeds), -1, -1),
        roots.flatten(0, 1),
        seeds[None].expand(len(shifts), -1, -1).flatten(0, 1),
        limits,
    )
    valid = result.exact.reshape(len(shifts), len(seeds), -1, count)
    contact = plan["convex_contact_model"]
    physical = torch.tensor(contact["pocket_fit"], device=root.device)[0]
    physical &= torch.tensor(contact["antipodal_margin"], device=root.device)[0] >= 0
    physical &= (
        torch.tensor(contact["environment_clearance_m"], device=root.device)[0] >= GRASP_CLEARANCE_M
    )
    valid &= physical[None, None, :, None]
    grasp_exact = valid[..., -1].any(1).sum(-1)
    if not standoff_sweep:
        valid = valid.all(-1)
    return {
        "base_height_m": float(root[0, 2]),
        "shifts_m": shifts.tolist(),
        "yaw_offsets_deg": torch.rad2deg(yaw_offsets).tolist(),
        "base_heights_m": roots[:, 0, 2].tolist(),
        "standoffs_m": standoffs.tolist() if standoff_sweep else [0.08, 0.0],
        "actual_seed_exact_count": valid[:, 0].sum(1).tolist(),
        "any_seed_exact_count": valid.any(1).sum(1).tolist(),
        "grasp_endpoint_any_seed_exact_count": grasp_exact.tolist(),
        "wrist_band_deg": wrist_band_deg,
        "source_frame": frame,
        "note": (
            "Scene-safe recorded pocket endpoints; shifted-base path and footprint not certified."
        ),
    }


def probe_recorded_station(run):
    """Compare grasp endpoints at measured crouch posture and nearby stations."""
    state = torch.load(run / "trajectory.pt", weights_only=False)[-1]
    episode = json.loads((run / "episode.json").read_text())
    model = torch.load(run / "collision_model.pt", weights_only=False)
    root = state["root_states"].cuda()
    target = state["object_state"][:, :3].cuda()
    toward = target[:, :2] - root[:, :2]
    toward /= toward.norm(dim=-1, keepdim=True)
    shifts = torch.arange(-0.10, 0.201, 0.025, device=root.device)
    roots = root.expand(len(shifts), -1).clone()
    roots[:, :2] += shifts[:, None] * toward
    heading = torch.atan2(toward[:, 1], toward[:, 0]).expand(len(shifts))
    goals, quats, _ = grasp_pose_bank(
        target.expand(len(shifts), -1),
        heading,
        episode["asset"]["size_m"],
        grasp_approach_origin(roots),
    )
    goals[..., 2] += 0.006
    limits = torch.tensor(read_arm_joint_limits(ROBOT_URDF).copy(), device=root.device)
    joints = state["dof_pos"][:, model["arm_joint_ids"]].cuda().expand(len(shifts), -1)
    result = inverse_bank(goals, quats, roots, joints, limits)
    errors = result.position_error.norm(dim=-1).reshape(len(shifts), -1)
    angles = result.orientation_error.norm(dim=-1).reshape_as(errors)
    exact = result.exact.reshape_as(errors)
    commands = result.command_position_world.reshape_as(goals)
    return {
        "source_run": str(run),
        "measured_base_height_m": root[0, 2].item(),
        "measured_goal_m": state["ee_goal_cart_world"][0].tolist(),
        "measured_ee_m": state["ee_pos"][0].tolist(),
        "stations": [
            {
                "toward_object_shift_m": float(shift),
                "position_error_m": errors[i].cpu().tolist(),
                "orientation_error_rad": angles[i].cpu().tolist(),
                "exact": exact[i].cpu().tolist(),
                "inverse_command_delta_z_m": (commands[i, :, 2] - goals[i, :, 2]).cpu().tolist(),
            }
            for i, shift in enumerate(shifts.cpu().tolist())
        ],
        "note": "Fixed measured posture diagnostic only; never a gate on low whole-body goals.",
    }


def probe_recorded_command(run, frame=-1, extra_joint_seeds=0, command_shifts=None):
    state = torch.load(run / "trajectory.pt", weights_only=False)[frame]
    model = torch.load(run / "collision_model.pt", weights_only=False)
    root = state["root_states"].cuda()
    goal = state["ee_goal_cart_world"].cuda()
    toward = goal[:, :2] - root[:, :2]
    toward /= toward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    shifts = (
        torch.arange(0, 0.151, 0.025, device=root.device)
        if command_shifts is None
        else root.new_tensor(command_shifts)
    )
    limits = torch.tensor(read_arm_joint_limits(ROBOT_URDF).copy(), device=root.device)
    actual = state["dof_pos"][:, model["arm_joint_ids"]].cuda()
    seeds = actual
    if extra_joint_seeds:
        index = torch.arange(1, extra_joint_seeds + 1, device=root.device)[:, None]
        fraction = torch.frac(
            index * root.new_tensor([0.618034, 0.414214, 0.732051, 0.236068, 0.645751, 0.316625])
        )
        spread = limits[:, 0] + (0.02 + 0.96 * fraction) * (limits[:, 1] - limits[:, 0])
        seeds = torch.cat((actual, limits.mean(-1)[None], limits.sum(-1)[None] - actual, spread))
    roots = root[:, None].expand(len(shifts), len(seeds), -1).clone()
    roots[..., :2] += shifts[:, None, None] * toward
    count = len(shifts) * len(seeds)
    result = inverse_bank(
        goal[:, None].expand(count, -1, -1),
        state["ee_goal_orn_quat_world"].cuda()[:, None].expand(count, -1, -1),
        roots.flatten(0, 1),
        seeds[None].expand(len(shifts), -1, -1).flatten(0, 1),
        limits,
    )
    error = result.position_error.norm(dim=-1).reshape(len(shifts), len(seeds))
    angle = (2 * result.orientation_error.norm(dim=-1).clamp_max(1).asin()).reshape_as(error)
    valid = result.exact.reshape_as(error)
    cost = error + 0.1 * angle
    cost = torch.where(valid.any(-1, keepdim=True), cost.masked_fill(~valid, torch.inf), cost)
    best = cost.argmin(-1)
    rows = torch.arange(len(shifts), device=root.device)
    return {
        "source_frame": frame,
        "goal": goal.cpu().tolist(),
        "actual_ee": state["ee_pos"].tolist(),
        "base_height_m": float(root[0, 2]),
        "shifts_m": shifts.cpu().tolist(),
        "position_error_m": error[rows, best].tolist(),
        "minimum_position_error_m": error.amin(-1).tolist(),
        "orientation_error_rad": angle[rows, best].tolist(),
        "exact": result.exact.reshape_as(error).any(-1).tolist(),
        "exact_seed_count": result.exact.reshape_as(error).sum(-1).tolist(),
        "joint_seed_count": len(seeds),
        "best_arm_joints_rad": result.arm_joint_positions.reshape(len(shifts), len(seeds), 6)[
            rows, best
        ].tolist(),
        "inverse_commands": result.command_position_world.reshape(len(shifts), len(seeds), 3)[
            rows, best
        ].tolist(),
        "note": "Fixed measured body attitude/height; shifts point toward the EE goal. "
        "This does not certify a base path, collision clearance or whole-body execution.",
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-height", type=float, default=0.536)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--command-reach", action="store_true")
    parser.add_argument("--grasp-bank-reach", action="store_true")
    parser.add_argument("--fresh-station-bank", action="store_true")
    parser.add_argument("--state-audit", action="store_true")
    parser.add_argument("--wrist-travel", action="store_true")
    parser.add_argument("--audit-frames", type=int, nargs="+", default=[0, -1])
    parser.add_argument("--pregrasp-standoff-sweep", action="store_true")
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--wrist-band-deg", type=float)
    parser.add_argument("--station-yaw-offsets", type=float, nargs="+", default=[0.0])
    parser.add_argument("--station-body-heights", type=float, nargs="+")
    parser.add_argument("--station-shifts", type=float, nargs="+")
    parser.add_argument("--extra-joint-seeds", type=int, default=0)
    parser.add_argument("--command-shifts", type=float, nargs="+")
    args = parser.parse_args()
    if args.extra_joint_seeds < 0:
        parser.error("extra joint seed count must be nonnegative")
    if args.run:
        report = (
            probe_recorded_wrist_route(args.run)
            if args.wrist_travel
            else probe_recorded_states(args.run, args.audit_frames)
            if args.state_audit
            else probe_fresh_station_bank(
                args.run,
                args.frame,
                args.wrist_band_deg,
                args.station_yaw_offsets,
                args.station_body_heights,
                args.station_shifts,
            )
            if args.fresh_station_bank
            else probe_recorded_grasp_bank(
                args.run,
                args.pregrasp_standoff_sweep,
                args.frame,
                args.wrist_band_deg,
                args.station_yaw_offsets,
            )
            if args.grasp_bank_reach or args.pregrasp_standoff_sweep
            else probe_recorded_command(
                args.run, args.frame, args.extra_joint_seeds, args.command_shifts
            )
            if args.command_reach
            else probe_recorded_station(args.run)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[pick:probe] {args.output}")
        return
    distance = torch.arange(0.2, 0.951, 0.025, device="cuda:0")
    height = torch.tensor([0.0265, 0.25, 0.45, 0.65, 0.85, 1.0], device="cuda:0")
    d, z = torch.meshgrid(distance, height, indexing="ij")
    centers = torch.stack((d.flatten(), torch.zeros_like(d.flatten()), z.flatten()), -1)
    roots = torch.zeros(len(centers), 13, device="cuda:0")
    roots[:, 2], roots[:, 3] = args.base_height, 1.0
    goals, quats, _ = grasp_pose_bank(
        centers, roots[:, 0], [0.06, 0.06, 0.053], grasp_approach_origin(roots)
    )
    limits = torch.tensor(
        read_arm_joint_limits(ROBOT_URDF).copy(), device="cuda:0", dtype=torch.float32
    )
    joints = roots.new_tensor([-1.57, 2.6, -1.5, 0.4, 0, 0]).expand(len(centers), -1)
    inverse = inverse_bank(goals, quats, roots, joints, limits)
    valid = inverse.exact.reshape(len(distance), len(height), -1).any(-1)
    errors = inverse.position_error.norm(dim=-1).reshape(len(distance), len(height), -1).amin(-1)
    report = []
    for i, h in enumerate(height.cpu().tolist()):
        report.append(
            {
                "height_m": h,
                "feasible_distances_m": distance[valid[:, i]].cpu().tolist(),
                "best_position_error_m": errors[:, i].min().item(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
