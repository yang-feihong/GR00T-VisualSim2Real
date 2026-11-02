"""Inspect recorded approach commands, physical camera gaze, and debug video integrity."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.envs.door.b2z1_reset_geometry import quaternion_xyzw_to_matrix
from gr00t.rl.pick.gaze import camera_pose
from gr00t.rl.pick.pocket import gripper_pocket_geometry, jaw_distal_limits


def closure_timing(angles, dt):
    angles = angles.reshape(-1)
    travel = angles[-1] - angles[0]
    if len(angles) < 2 or float(travel) <= 0:
        return {"closing_travel_rad": float(travel), "motion_10_to_90_s": None}
    fraction = (angles - angles[0]) / travel
    first10 = int(torch.where(fraction >= 0.1)[0][0])
    first90 = int(torch.where(fraction >= 0.9)[0][0])
    return {
        "closing_travel_rad": float(travel),
        "motion_10_to_90_s": (first90 - first10) * dt,
        "entry_to_90_percent_s": first90 * dt,
        "peak_finite_difference_speed_rad_s": float((angles.diff() / dt).abs().max()),
    }


def inspect(run, gripper_joint_index=None, *, mosaic_times_s=()):
    if not (run / "trajectory.pt").is_file():
        return inspect_control_record(run, mosaic_times_s=mosaic_times_s)
    summary = json.loads((run / "summary.json").read_text())
    episode = json.loads((run / "episode.json").read_text())
    trace = torch.load(run / "trajectory.pt", weights_only=False)
    basket_task = episode.get("task_type") == "pick_basket"
    approach = next(
        (event["step"] for event in summary["transitions"] if event["from"] == "APPROACH_OBJECT"),
        0 if basket_task else len(trace),
    )
    states = trace[:approach] if approach else trace

    def values(key):
        return torch.cat([state[key] for state in states]).cuda()

    goal, actual = values("ee_goal_cart_world"), values("ee_pos")
    goal_q, actual_q = values("ee_goal_orn_quat_world"), values("ee_orn")
    target = goal.new_tensor(episode["object_position"])[None].expand_as(goal)

    def angles(p, q):
        eye, forward = camera_pose(p, quaternion_xyzw_to_matrix(q[..., [1, 2, 3, 0]]))
        direction = target - eye
        cosine = (forward * direction).sum(-1) / direction.norm(dim=-1).clamp_min(1e-6)
        return torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))

    desired_angle, actual_angle = angles(goal, goal_q), angles(actual, actual_q)
    if (run / "collision_model.pt").exists():
        model = torch.load(run / "collision_model.pt", weights_only=False)
    else:
        record = torch.load(run / "control_record.pt", weights_only=False)
        model = {"arm_joint_ids": record["arm_joint_ids"]}
    joints = values("dof_pos")[:, model["arm_joint_ids"].cuda()]
    report = {
        "seed": summary["seed"],
        "success": summary["success"],
        "failure": summary["failure"],
        "approach_frames": approach,
        "minimum_command_height_above_object_m": float((goal[:, 2] - target[:, 2]).min()),
        "command_camera_angle_last50_deg_max": float(desired_angle[-50:].max()),
        "actual_camera_angle_last50_deg_max": float(actual_angle[-50:].max()),
        "actual_camera_angle_last_deg": float(actual_angle[-1]),
        "last_actual_minus_command_ee_m": (actual[-1] - goal[-1]).tolist(),
        "maximum_ee_tracking_error_m": float((actual - goal).norm(dim=-1).max()),
        "actual_arm_joint_max_change_deg": torch.rad2deg(
            (joints - joints[:1]).abs().amax(0)
        ).tolist(),
        "videos": {},
    }
    if "arm_position_target" in trace[-1]:
        from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
        from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
        from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

        terminal = trace[-1]
        root = terminal["root_states"].cuda()
        arm_target = terminal["arm_position_target"].cuda()
        target_fk, _, _ = b2z1_gripper_pose_and_jacobian(
            root[:, :3], root[:, [4, 5, 6, 3]], arm_target
        )
        limits = root.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
        w, x, y, z = root[:, 3:7].unbind(-1)
        rpy = torch.stack(
            (
                torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                (2 * (w * y - z * x)).clamp(-1, 1).asin(),
                torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
            ),
            -1,
        )
        command = terminal["ee_goal_cart_world"].cuda()
        report["terminal_tracking"] = {
            "root_position_world_m": root[:, :3].tolist(),
            "root_rpy_deg": torch.rad2deg(rpy).tolist(),
            "ee_command_world_m": command.tolist(),
            "ee_actual_world_m": terminal["ee_pos"].tolist(),
            "arm_target_fk_world_m": target_fk.tolist(),
            "arm_target_fk_minus_ee_command_m": (target_fk - command).tolist(),
            "actual_ee_minus_arm_target_fk_m": (terminal["ee_pos"].cuda() - target_fk).tolist(),
            "arm_joint_target_rad": arm_target.tolist(),
            "arm_joint_target_limit_margin_rad": torch.minimum(
                arm_target - limits[:, 0], limits[:, 1] - arm_target
            ).tolist(),
        }
    if basket_task:
        for name in (
            "command_camera_angle_last50_deg_max",
            "actual_camera_angle_last50_deg_max",
            "actual_camera_angle_last_deg",
        ):
            report.pop(name)
    if "distractors" in episode:
        from gr00t.rl.pick.distractors import object_category, target_semantics

        initial = json.loads((run / "initial_acceptance.json").read_text())
        pairs = [
            (object_category(item["asset"]), item["support_type"])
            for item in [episode, *episode["distractors"]]
        ]
        report["scene_semantics"] = {
            "target": target_semantics(episode),
            "distractor_count": len(episode["distractors"]),
            "category_support_pairs": pairs,
            "unique_pairs": len(pairs) == len(set(pairs)),
            "measured_support_labels_match": (
                initial["checks"]["all_objects_inside_basket"]
                and initial["checks"]["all_objects_supported"]
                if basket_task
                else initial["checks"]["object_support_labels_match"]
            ),
        }
        obstacles = torch.load(run / "planning_obstacles.pt", weights_only=False)
        report["scene_semantics"]["planning_obstacle_vertices"] = len(obstacles["points"])
        if episode["distractors"]:
            poses = torch.cat([row["distractor_states"] for row in trace]).cuda()
            report["scene_semantics"]["maximum_distractor_drift_m"] = float(
                (poses[..., :3] - poses[:1, :, :3]).norm(dim=-1).max()
            )
            if not len(obstacles["points"]):
                raise RuntimeError("Distractors are missing from recorded planning geometry")
        if not report["scene_semantics"]["unique_pairs"] or (
            summary["initial_accepted"]
            and not report["scene_semantics"]["measured_support_labels_match"]
        ):
            raise RuntimeError("Target object/support semantics were not unambiguous at admission")
    dt = summary["simulation_dt_s"]
    durations = {}
    entry_frame, grasp_frame = 0, None
    for event in summary["transitions"]:
        durations[event["from"]] = durations.get(event["from"], 0.0) + event["step"] * dt
        entry_frame += event["step"]
        if event["to"] == "GRASP_OBJECT":
            grasp_frame = entry_frame
    if summary["failure"] and isinstance(summary["failure"], dict):
        failure = summary["failure"]
        durations[failure["stage"]] = (
            durations.get(failure["stage"], 0.0) + failure["stage_step"] * dt
        )
    report["stage_duration_s"] = durations
    from gr00t.rl.pick.probe_workspace import wrist_route_metrics

    wrist = torch.cat([row["dof_pos"][:, model["arm_joint_ids"][-1]] for row in trace]).cuda()
    lengths = torch.tensor(
        [event["step"] for event in summary["transitions"]], device=wrist.device, dtype=torch.long
    )
    boundaries = torch.cat(
        (lengths.new_zeros(1), lengths.cumsum(0) - 1, lengths.new_tensor([len(trace) - 1]))
    ).clamp(0, len(trace) - 1)
    report["episode_wrist_route"] = {
        key: value.tolist() for key, value in wrist_route_metrics(wrist, boundaries).items()
    }
    if grasp_frame is not None:
        wrist = torch.cat(
            [state["dof_pos"] for state in trace[max(0, approach - 1) : grasp_frame + 1]]
        )[:, model["arm_joint_ids"][-1]]
        report["grasp_move_q6_max_change_deg"] = float(
            torch.rad2deg((wrist - wrist[:1]).abs().max())
        )
        start_q = trace[max(0, approach - 1)]["ee_goal_orn_quat_world"]
        end_q = trace[grasp_frame]["ee_goal_orn_quat_world"]
        report["grasp_move_ee_goal_rotation_deg"] = float(
            torch.rad2deg(2 * (start_q * end_q).sum(-1).abs().clamp_max(1).acos())[0]
        )
    object_positions = torch.cat([state["object_state"][:, :3] for state in trace]).cuda()
    report["final_ee_height_world_m"] = float(trace[-1]["ee_pos"][0, 2])
    report["object_net_lift_m"] = float(object_positions[-1, 2] - object_positions[0, 2])
    report["object_max_lift_m"] = float(object_positions[:, 2].max() - object_positions[0, 2])
    report["final_object_height_world_m"] = float(object_positions[-1, 2])
    report["strict_lift_pass"] = bool(
        (object_positions[-1, 2] >= 0.30)
        & (object_positions[-1, 2] - object_positions[0, 2] >= 0.10)
        & (trace[-1]["ee_pos"][0, 2] >= 0.30)
    )
    if basket_task:
        from types import SimpleNamespace

        from gr00t.rl.pick.basket_controller import basket_vertical_clearance

        corners = episode["basket_sampling"]["collider_corners_local_m"]
        points = object_positions.new_tensor(
            np.load(Path(episode["asset"]["usd_path"]).with_name("collision_support.npy"))
        )
        roots = torch.cat([row["root_states"] for row in trace]).cuda()
        poses = torch.cat([row["object_state"] for row in trace]).cuda()
        clearance, _ = basket_vertical_clearance(
            SimpleNamespace(boxes=roots.new_tensor(corners)), roots, poses, points
        )
        record = torch.load(run / "control_record.pt", weights_only=False)
        report["basket"] = {
            "final_object_bottom_above_rim_m": float(clearance[-1]),
            "last50_minimum_object_bottom_above_rim_m": float(clearance[-50:].min()),
            "maximum_dispatched_base_command": float(record["post"]["base_command"].abs().max()),
        }
        report["strict_lift_pass"] &= bool(
            (clearance[-50:] >= episode.get("lift_above_rim_m", 0.10)).all()
        )
    elapsed = 0
    for event in summary["transitions"]:
        elapsed += event["step"]
        if event["from"] != "GRASP_OBJECT":
            continue
        candidates = trace[max(0, elapsed - 20) : min(elapsed, len(trace))]
        forces = torch.cat([state["object_finger_forces"] for state in candidates]).norm(dim=-1)
        frame = max(0, elapsed - 20) + int(forces.amin(-1).argmax())
        state = trace[frame]
        inverse = state["ee_orn"].cuda().clone()
        inverse[:, 1:] *= -1
        center = _quat_rotate(inverse, (state["object_state"][:, :3] - state["ee_pos"]).cuda())
        contacts = _quat_rotate(
            inverse[:, None],
            (state["object_finger_contact_points"] - state["ee_pos"][:, None]).cuda(),
        )
        pocket = gripper_pocket_geometry("cuda:0")
        depth = pocket["open_tip_midpoint_x"] - contacts[..., 0]
        physical_limits = None
        if gripper_joint_index is None and "gripper_joint_ids" in model:
            gripper_joint_index = int(model["gripper_joint_ids"][0])
        if gripper_joint_index is not None:
            aperture = -state["dof_pos"][:, gripper_joint_index].cuda()
            physical_limits = jaw_distal_limits(aperture)
            depth = physical_limits - contacts[..., 0]
            closing_states = trace[max(0, elapsed - event["step"] - 1) : elapsed]
            closing_angles = torch.cat([row["dof_pos"] for row in closing_states])[
                :, gripper_joint_index
            ]
            report["gripper_closure"] = closure_timing(closing_angles, dt)
            report["gripper_closure"]["stage_duration_s"] = event["step"] * dt
        report["grasp_geometry"] = {
            "frame": frame,
            "object_center_local_m": center[0].tolist(),
            "contact_points_local_m": contacts[0].tolist(),
            "contact_depth_behind_mouth_m": depth[0].tolist(),
            "legacy_open_tip_midpoint_x_m": float(pocket["open_tip_midpoint_x"]),
            "contacts_inside_pocket": None
            if physical_limits is None
            else bool(
                torch.isfinite(depth).all()
                & (depth >= 0).all()
                & (contacts[..., 1].abs() <= pocket["half_width"]).all()
            ),
            "center_inside_pocket": None
            if physical_limits is None
            else bool(
                (center[..., 0] >= pocket["back_x"]).all()
                & (center[..., 0] <= physical_limits.amin(-1)).all()
                & (center[..., 1].abs() <= pocket["half_width"]).all()
            ),
            "finger_normal_force_n": state["object_finger_forces"].norm(dim=-1)[0].tolist(),
            "distal_edge_definition": "actual_individual_jaw_edges"
            if physical_limits is not None
            else "legacy_average_open_jaw_projection",
            "jaw_distal_limits_m": None if physical_limits is None else physical_limits[0].tolist(),
            "gripper_joint_index": gripper_joint_index,
            "actual_ee_lift_m": float(
                trace[-1]["ee_pos"][0, 2] - trace[elapsed - 1]["ee_pos"][0, 2]
            ),
        }
        if gripper_joint_index is not None and elapsed < len(trace):
            carried = trace[elapsed:]
            ee = torch.cat([row["ee_pos"] for row in carried]).cuda()
            inverse = torch.cat([row["ee_orn"] for row in carried]).cuda()
            inverse[:, 1:] *= -1
            centers = _quat_rotate(inverse, object_positions[elapsed:] - ee)
            contact_points = torch.cat(
                [row["object_finger_contact_points"] for row in carried]
            ).cuda()
            contact_local = _quat_rotate(inverse[:, None], contact_points - ee[:, None])
            aperture = -torch.cat([row["dof_pos"] for row in carried])[
                :, gripper_joint_index
            ].cuda()
            edges = jaw_distal_limits(aperture)
            forces = torch.cat([row["object_finger_forces"] for row in carried]).cuda().norm(dim=-1)
            bilateral = (forces > 1e-3).all(-1)
            center_inside = (
                (centers[:, 0] >= pocket["back_x"])
                & (centers[:, 0] <= edges.amin(-1))
                & (centers[:, 1].abs() <= pocket["half_width"])
            )
            contacts_inside = (
                torch.isfinite(contact_local).all(dim=(-1, -2))
                & (contact_local[..., 0] <= edges).all(-1)
                & (contact_local[..., 1].abs() <= pocket["half_width"]).all(-1)
            )
            report["carry_geometry"] = {
                "first_frame": elapsed,
                "frames": len(carried),
                "center_outside_pocket_frames": int((~center_inside).sum()),
                "bilateral_contact_missing_frames": int((~bilateral).sum()),
                "bilateral_contact_outside_pocket_frames": int(
                    (bilateral & ~contacts_inside).sum()
                ),
                "maximum_center_drift_m": float((centers - centers[:1]).norm(dim=-1).max()),
                "minimum_center_depth_behind_jaws_m": float((edges - centers[:, :1]).min()),
                "minimum_finger_force_n": forces.amin(0).tolist(),
            }
    for path in sorted(run.glob("*.mp4")):
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = []
        for index in (0, count // 2, count - 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Undecodable frame {index}: {path}")
            frames.append(frame)
        delta = float(np.abs(frames[1].astype(float) - frames[0]).mean())
        motion_expected = (
            len(trace) > 5
            and summary["initial_accepted"]
            and not (basket_task and path.name == "base_0_rgb.mp4")
        )
        if count < 1 or frames[-1].std() < 1 or (motion_expected and (count < 2 or delta < 0.1)):
            raise RuntimeError(f"Blank/static video: {path}")
        report["videos"][path.name] = {
            "frames": count,
            "duration_s": count / fps,
            "motion_mae": delta,
            "motion_expected": motion_expected,
        }
        if path.name == "six_camera_mosaic.mp4":
            cv2.imwrite(str(run / "views_check.png"), frames[-1])
            for seconds in mosaic_times_s:
                if not np.isfinite(seconds) or not 0 <= seconds < count / fps:
                    raise ValueError(f"Mosaic time is outside the recording: {seconds}")
                capture.set(cv2.CAP_PROP_POS_FRAMES, min(count - 1, round(seconds * fps)))
                ok, selected_frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Undecodable requested mosaic frame: {seconds}")
                output = run / f"mosaic_{seconds:.3f}s.png"
                cv2.imwrite(str(output), selected_frame)
                report.setdefault("requested_mosaic_frames", {})[str(seconds)] = str(output)
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                min(count - 1, int(max(0.08, (approach - 20) * 0.02) * fps)),
            )
            ok, frame = capture.read()
            if ok:
                cv2.imwrite(str(run / "approach_check.png"), frame)
        capture.release()
    if len(report["videos"]) != 7:
        raise RuntimeError("Expected six camera streams and a mosaic")
    from gr00t.rl.pick.dataset import episode_dataset_root

    dataset = episode_dataset_root(run)
    if dataset.exists():
        from gr00t.rl.data_collection.auto_open_standard import camera_streams_for_mode
        from gr00t.rl.data_collection.auto_open_writer import _verify_video

        report["training_videos"] = {}
        for stream in camera_streams_for_mode("dual").values():
            paths = list(dataset.rglob(f"{stream['key']}/chunk-000/episode_000000.mp4"))
            if len(paths) != 1:
                raise RuntimeError(f"Missing/duplicate training stream: {stream['key']}")
            _verify_video(paths[0], stream["shape"], len(trace))
            report["training_videos"][stream["key"]] = {
                "frames": len(trace),
                "fps": 50,
                "shape": stream["shape"],
            }
    (run / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "videos"}))


def inspect_control_record(run, *, mosaic_times_s=()):
    """Inspect the same per-episode batch record consumed by official replay."""
    from collections import Counter

    summary = json.loads((run / "summary.json").read_text())
    record = torch.load(run / "control_record.pt", map_location="cpu", weights_only=False)
    peaks = {}
    for row in record["contacts"]:
        a, b = row["body_a_path"], row["body_b_path"]
        if any("foot" in path.lower() for path in (a, b)):
            continue
        pair = tuple(sorted((a, b)))
        if pair not in peaks or row["max_contact_force"] > peaks[pair]["max_contact_force"]:
            peaks[pair] = row
    report = {
        "seed": summary["seed"],
        "success": summary["success"],
        "failure": summary.get("failure"),
        "failed_stage": summary.get("failed_stage"),
        "stage_frames": dict(Counter(record["stages"])),
        "nonfoot_contact_pair_peaks": sorted(
            peaks.values(), key=lambda row: row["max_contact_force"], reverse=True
        )[:20],
        "final_object_height_m": float(record["post"]["object_state"][-1, 2]),
        "minimum_base_height_m": float(record["post"]["root_states"][:, 2].min()),
        "stage_metrics": recorded_stage_metrics(record),
        "terminal_blockers": {
            key: value
            for key, value in summary.get("acceptance_diagnostics", {})
            .get(summary.get("failed_stage"), {})
            .items()
            if value.get("observed") and not value.get("current")
        },
    }
    mosaic = run.parents[2] / "six_camera_mosaics" / f"seed{summary['seed']}_six_camera_mosaic.mp4"
    if mosaic.is_file():
        capture = cv2.VideoCapture(str(mosaic))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if count != len(record["observation_state"]) or fps <= 0:
            raise ValueError(f"Mosaic frame count/rate disagrees with the control record: {mosaic}")
        times = mosaic_times_s or (0, count / fps / 2, (count - 1) / fps)
        try:
            for seconds in times:
                if not np.isfinite(seconds) or not 0 <= seconds < count / fps:
                    raise ValueError(f"Mosaic time is outside the recording: {seconds}")
                capture.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * fps))
                ok, frame = capture.read()
                if not ok or frame.std() < 1:
                    raise RuntimeError(f"Blank/undecodable mosaic frame at {seconds}: {mosaic}")
                if mosaic_times_s:
                    output = run / f"mosaic_{seconds:.3f}s.png"
                    cv2.imwrite(str(output), frame)
                    report.setdefault("requested_mosaic_frames", {})[str(seconds)] = str(output)
            report["mosaic"] = {"frames": count, "fps": fps, "duration_s": count / fps}
        finally:
            capture.release()
    (run / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def recorded_stage_metrics(record):
    """Summarize every recorded stage, without task-specific diagnostic exceptions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    post = record["post"]
    p, q, obj = (post[key].to(device) for key in ("ee_pos", "ee_orn", "object_state"))
    inverse = q.clone()
    inverse[:, 1:] *= -1
    center = _quat_rotate(inverse, obj[:, :3] - p)
    force = post["object_finger_forces"].to(device).norm(dim=-1)
    jaw = post["dof_pos"][:, record["gripper_joint_id"]].to(device)
    error = (
        (p - post["ee_goal_cart_world"].to(device)).norm(dim=-1)
        if "ee_goal_cart_world" in post
        else None
    )
    metrics = {}
    roots = post.get("root_states")
    if roots is not None:
        roots = roots.to(device)
        w, x, y, z = roots[:, 3:7].unbind(-1)
        pitch = torch.rad2deg(torch.asin((2 * (w * y - z * x)).clamp(-1, 1)))
    for stage in dict.fromkeys(record["stages"]):
        rows = torch.tensor(
            [index for index, value in enumerate(record["stages"]) if value == stage], device=device
        )
        metrics[stage] = {
            "frames": len(rows),
            "finger_force_max_n": force[rows].amax(0).tolist(),
            "finger_force_last_n": force[rows[-1]].tolist(),
            "gripper_angle_last_rad": float(jaw[rows[-1]]),
            "gripper_angle_span_rad": float(jaw[rows].max() - jaw[rows].min()),
            "object_center_gripper_local_last_m": center[rows[-1]].tolist(),
            "object_center_gripper_local_min_m": center[rows].amin(0).tolist(),
            "object_center_gripper_local_max_m": center[rows].amax(0).tolist(),
            "ee_tracking_error_last_m": None if error is None else float(error[rows[-1]]),
            "ee_tracking_error_max_m": None if error is None else float(error[rows].max()),
            "object_position_first_m": obj[rows[0], :3].tolist(),
            "object_position_last_m": obj[rows[-1], :3].tolist(),
            "object_displacement_max_m": float(
                (obj[rows, :3] - obj[rows[0], :3]).norm(dim=-1).max()
            ),
        }
        if roots is not None:
            metrics[stage].update(
                body_height_first_last_m=roots[rows[[0, -1]], 2].tolist(),
                body_pitch_first_last_deg=pitch[rows[[0, -1]]].tolist(),
                body_pitch_min_max_deg=[float(pitch[rows].min()), float(pitch[rows].max())],
            )
    return metrics


def inspect_force_history(run, stage):
    """Export measured forces and object slip using the shared record's stage labels."""
    record = torch.load(run / "control_record.pt", map_location="cpu", weights_only=False)
    rows = [index for index, value in enumerate(record["stages"]) if value == stage]
    if not rows:
        raise ValueError(f"No recorded frames for stage: {stage}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def values(key):
        return record["post"][key][rows].to(device)

    p, q, obj = (values(key) for key in ("ee_pos", "ee_orn", "object_state"))
    inverse = q.clone()
    inverse[:, 1:] *= -1
    center = _quat_rotate(inverse, obj[:, :3] - p)
    normal, friction = (
        values(key) for key in ("object_finger_forces", "object_finger_friction_forces")
    )
    report = {
        "stage": stage,
        "frames": rows,
        "force_convention": "forces on robot jaws, not forces on object",
        "object_position_world_m": obj[:, :3].tolist(),
        "object_orientation_world_wxyz": obj[:, 3:7].tolist(),
        "object_velocity_world_m_s": obj[:, 7:10].tolist(),
        "object_angular_velocity_world_rad_s": obj[:, 10:13].tolist(),
        "ee_position_world_m": p.tolist(),
        "ee_orientation_world_wxyz": q.tolist(),
        "object_center_gripper_local_m": center.tolist(),
        "contact_points_gripper_local_m": _quat_rotate(
            inverse[:, None], values("object_finger_contact_points") - p[:, None]
        ).tolist(),
        "normal_forces_world_n": normal.tolist(),
        "friction_forces_world_n": friction.tolist(),
        "normal_forces_gripper_local_n": _quat_rotate(inverse[:, None], normal).tolist(),
        "friction_forces_gripper_local_n": _quat_rotate(inverse[:, None], friction).tolist(),
        "jaw_angle_rad": values("dof_pos")[:, record["gripper_joint_id"]].tolist(),
    }
    for key in ("ee_goal_cart_world", "gripper_actuator_target", "gripper_contact_hold_active"):
        if key in record["post"]:
            report[key] = values(key).tolist()
    output = run / f"force_history_{stage}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "frames": len(rows)}), flush=True)
    return report


def inspect_trunk_transform(run):
    """Reconstruct the deployed transform from recorded pre-step trunk poses."""
    from legged_gym.utils.trunk_control_filter import TrunkControlFilter

    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.data_collection.auto_open_standard import CONTROL_DT
    from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    record = torch.load(run / "control_record.pt", map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    before, after = (
        {key: value.to(device) for key, value in record[phase].items()} for phase in ("pre", "post")
    )
    roots = before["root_states"]
    raw, raw_q = after["ee_goal_cart_world"], after["ee_goal_orn_quat_world"]
    selected = TrunkControlFilter()
    transformed = []
    for index in range(len(roots)):
        pose = roots[index : index + 1]
        p, _ = selected.control_target(
            index * CONTROL_DT,
            pose[:, :3],
            pose[:, 3:7],
            pose.new_zeros(1),
            raw[index : index + 1],
            raw_q[index : index + 1],
        )
        transformed.append(p)
    transformed = torch.cat(transformed)
    fk, _, _ = b2z1_gripper_pose_and_jacobian(
        roots[:, :3], roots[:, [4, 5, 6, 3]], after["arm_position_target"]
    )
    actual = after["ee_pos"]
    arm = after["arm_position_target"]
    limits = arm.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    report = {
        "definition": "reconstructed_trunk_transform_not_recorded_effective_target",
        "limitations": "filter initialized at first control frame; warmup history unavailable; "
        "assumes plane terrain, height-invariant EE mode and one filter update per control tick",
        "frames": len(roots),
        "root_position_world_m": roots[:, :3].tolist(),
        "arm_actual_positions_rad": after["dof_pos"][:, record["arm_joint_ids"]].tolist(),
        "arm_target_positions_rad": arm.tolist(),
        "arm_target_joint_limit_margin_rad": torch.minimum(
            arm - limits[:, 0], limits[:, 1] - arm
        ).tolist(),
        "raw_ee_goal_world_m": raw.tolist(),
        "reconstructed_effective_ee_goal_world_m": transformed.tolist(),
        "arm_target_fk_pre_root_world_m": fk.tolist(),
        "actual_ee_post_world_m": actual.tolist(),
        "trunk_transform_offset_world_m": (transformed - raw).tolist(),
        "arm_target_fk_minus_effective_world_m": (fk - transformed).tolist(),
        "actual_minus_effective_world_m": (actual - transformed).tolist(),
    }
    output = run / "trunk_transform_inspection.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "frames": len(roots)}), flush=True)
    return report


def terminal_inverse_preview(record, *, body_offsets=(0, -0.025, -0.05, -0.075, -0.1), seeds=512):
    """Math-only multi-start inverse; body hypotheses are not policy guarantees."""
    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.auto_open.pose_planning import solve_pose_grid
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = record["pre"]["root_states"][-1:].to(device)
    actual = record["pre"]["dof_pos"][-1:, record["arm_joint_ids"]].to(device)
    limits = root.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    sampled = torch.rand(
        seeds,
        6,
        device=root.device,
        dtype=root.dtype,
        generator=torch.Generator(device=root.device).manual_seed(7919),
    )
    joints = torch.cat(
        (
            actual,
            limits.mean(-1)[None],
            limits.sum(-1)[None] - actual,
            limits[:, 0] + sampled * (limits[:, 1] - limits[:, 0]),
        )
    )
    roots = root.expand(len(body_offsets), len(joints), 1, -1).clone()
    roots[..., 2] += root.new_tensor(body_offsets)[:, None, None]
    position = record["post"]["ee_goal_cart_world"][-1:].to(root)[:, None]
    orientation = record["post"]["ee_goal_orn_quat_world"][-1:].to(root)[:, None]
    result = solve_pose_grid(
        position,
        orientation,
        roots,
        joints[None, :, None].expand(len(body_offsets), -1, -1, -1),
        limits,
    )
    position_error = result.position_error[..., 0, 0, :].norm(dim=-1)
    orientation_error = 2 * result.orientation_error[..., 0, 0, :].norm(dim=-1).clamp_max(1).asin()
    good = (position_error <= 0.002) & (orientation_error <= 0.01)
    cost = position_error + 0.1 * orientation_error
    selected = torch.where(
        good.any(-1), cost.masked_fill(~good, torch.inf).argmin(-1), cost.argmin(-1)
    )
    rows = torch.arange(len(body_offsets), device=root.device)
    return {
        "definition": "deployed_gpu_inverse_no_gravity_multi_start_raw_terminal_goal",
        "not_a_body_policy_or_collision_feasibility_guarantee": True,
        "position_tolerance_m": 0.002,
        "orientation_tolerance_rad": 0.01,
        "joint_seed_count": len(joints),
        "selection": (
            "prefer_satisfying_both_tolerances_then_min_position_plus_0.1_times_orientation"
        ),
        "body_height_world_m": roots[:, 0, 0, 2].tolist(),
        "successful_seed_counts": good.sum(-1).tolist(),
        "minimum_position_error_any_seed_m": position_error.amin(-1).tolist(),
        "best_position_error_m": position_error[rows, selected].tolist(),
        "best_orientation_error_rad": orientation_error[rows, selected].tolist(),
        "selected_arm_joints_rad": result.arm_joint_positions[rows, selected, 0, 0].tolist(),
    }


def inspect_terminal_inverse(run):
    record = torch.load(run / "control_record.pt", map_location="cpu", weights_only=False)
    report = terminal_inverse_preview(record)
    (run / "terminal_inverse_inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def inspect_goal_bounds(run, limits):
    """Audit supplied publication bounds against recorded effective EE goals."""
    record_path = run / "control_record.pt"
    record = (
        torch.load(record_path, map_location="cpu", weights_only=False)
        if record_path.is_file()
        else None
    )
    if record is not None and "ee_goal_cart_local" in record["post"]:
        goals = record["post"]["ee_goal_cart_local"]
    else:
        trace = torch.load(run / "trajectory.pt", map_location="cpu", weights_only=False)
        goals = torch.cat([state["ee_goal_cart_local"] for state in trace])
    bounds = goals.new_tensor(limits).reshape(3, 2)
    violation = torch.maximum(bounds[:, 0] - goals, goals - bounds[:, 1]).clamp_min(0)
    outside = violation > 1e-5
    frames = torch.where(outside.any(-1))[0]
    report = {
        "frames": len(goals),
        "effective_goal_local_min_m": goals.amin(0).tolist(),
        "effective_goal_local_max_m": goals.amax(0).tolist(),
        "reference_limits_m": bounds.tolist(),
        "outside_frames_by_xyz": outside.sum(0).tolist(),
        "maximum_violation_by_xyz_m": violation.amax(0).tolist(),
        "first_outside_frame": None if not len(frames) else int(frames[0]),
        "first_outside_values": goals[frames[:10]].tolist(),
    }
    (run / "goal_bounds_inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def inspect_floor_tracking(run):
    """Compare commanded and actual fixed-jaw collider heights on the GPU."""
    from gr00t.rl.pick.collision import ArmCollisionSamples

    keys = ("ee_pos", "ee_orn", "ee_goal_cart_world", "ee_goal_orn_quat_world")
    record_path = run / "control_record.pt"
    record = (
        torch.load(record_path, map_location="cpu", weights_only=False)
        if record_path.is_file()
        else None
    )
    if record is not None and all(key in record["post"] for key in keys):
        states = {key: record["post"][key].cuda() for key in keys}
    else:
        if not (run / "trajectory.pt").is_file():
            missing = [key for key in keys if key not in record["post"]]
            raise ValueError(f"Recording has no raw debug targets for this audit: {missing}")
        trace = torch.load(run / "trajectory.pt", map_location="cpu", weights_only=False)
        states = {key: torch.cat([state[key] for state in trace]).cuda() for key in keys}
    model = torch.load(run / "collision_model.pt", weights_only=False)
    collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
    collision.gripper_surface = [points.cuda() for points in model["gripper_surface"]]
    fixed, _ = collision._gripper_local_surfaces(states["ee_pos"])

    def heights(position, quaternion):
        return (position[:, None, 2] + _quat_rotate(quaternion[:, None], fixed[None])[..., 2]).amin(
            -1
        )

    actual = heights(states["ee_pos"], states["ee_orn"])
    command = heights(states["ee_goal_cart_world"], states["ee_goal_orn_quat_world"])
    contact = torch.where(actual < 0)[0]
    first = None if not len(contact) else int(contact[0])
    report = {
        "frames": len(actual),
        "fixed_jaw_actual_min_world_z_m": float(actual.min()),
        "fixed_jaw_command_min_world_z_m": float(command.min()),
        "fixed_jaw_max_downward_tracking_error_m": float((command - actual).max()),
        "first_actual_below_world_z_zero_frame": first,
        "at_first_crossing": None
        if first is None
        else {
            "actual_collider_min_z_m": float(actual[first]),
            "command_collider_min_z_m": float(command[first]),
            "actual_ee_world_m": states["ee_pos"][first].tolist(),
            "command_ee_world_m": states["ee_goal_cart_world"][first].tolist(),
        },
    }
    (run / "floor_tracking_inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def compare_recorded_runs(run, other, tolerance=1e-5):
    """Compare one seed across batch layouts, ignoring only inert distractor padding."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = [
        torch.load(path / "control_record.pt", map_location=device, weights_only=False)
        for path in (run, other)
    ]
    episodes = [json.loads((path / "episode.json").read_text()) for path in (run, other)]
    if episodes[0]["seed"] != episodes[1]["seed"]:
        raise ValueError("Batch independence comparison requires the same seed")
    count = min(len(record["observation_state"]) for record in records)
    differences = {}
    first_differences, initial_differences, first_nonzero = {}, {}, {}

    def difference(key, left, right):
        left, right = left[:count].double(), right[:count].double()
        if left.shape != right.shape:
            raise ValueError(f"Different data dimensions for {key}: {left.shape}, {right.shape}")
        if not left.numel():
            first_differences[key], initial_differences[key] = None, 0.0
            first_nonzero[key] = None
            return 0.0
        equal = (left == right) | (left.isnan() & right.isnan())
        delta = torch.where(equal, 0, (left - right).abs()).nan_to_num(nan=torch.inf)
        per_frame = delta.reshape(count, -1).amax(-1)
        failing = torch.where(per_frame > tolerance)[0]
        nonzero = torch.where(per_frame > 0)[0]
        first_nonzero[key] = None if not len(nonzero) else int(nonzero[0])
        first_differences[key] = None if not len(failing) else int(failing[0])
        initial_differences[key] = float(per_frame[0])
        return float(delta.max())

    for key in ("observation_state", "action"):
        differences[key] = difference(key, records[0][key], records[1][key])
    for phase in ("pre", "post"):
        for key in sorted(set(records[0][phase]) & set(records[1][phase])):
            left, right = (record[phase][key][:count] for record in records)
            if key == "distractor_states":
                slots = len(episodes[0].get("distractors", []))
                left, right = left[:, :slots], right[:, :slots]
            if key in ("root_states", "object_state", "distractor_states", "robot_body_state_w"):
                right = right.clone()
                flip = (left[..., 3:7] * right[..., 3:7]).sum(-1) < 0
                right[..., 3:7] = torch.where(flip[..., None], -right[..., 3:7], right[..., 3:7])
            if key in ("ee_orn", "ee_goal_orn_quat_world"):
                right = torch.where((left * right).sum(-1, keepdim=True) < 0, -right, right)
            field = f"{phase}.{key}"
            differences[field] = difference(field, left, right)
    descriptor_keys = (
        "seed",
        "scene_id",
        "asset",
        "object_position",
        "initial_arm_joints",
        "base_xy",
        "base_yaw",
        "station_xy",
        "station_yaw",
        "distractors",
    )
    metadata_equal = all(episodes[0].get(key) == episodes[1].get(key) for key in descriptor_keys)
    physical_components = {}
    for key, start, stop in (
        ("base_position_m", 0, 3),
        ("base_quaternion", 3, 7),
        ("base_linear_velocity_m_s", 7, 10),
        ("base_angular_velocity_rad_s", 10, 13),
    ):
        physical_components[key] = difference(
            key,
            records[0]["pre"]["root_states"][:, start:stop],
            records[1]["pre"]["root_states"][:, start:stop],
        )
    policy_trace = {}
    for key in ("lowlevel_policy_observation", "lowlevel_policy_action"):
        available = [key in record["post"] for record in records]
        detail = {"available_in_both_runs": all(available), "available_by_run": available}
        if all(available):
            field = f"post.{key}"
            frame = first_nonzero[field]
            detail.update(width=int(records[0]["post"][key].shape[-1]), first_nonzero_frame=frame)
            if frame is not None:
                left, right = [record["post"][key][frame].double() for record in records]
                delta = (left - right).abs().nan_to_num(nan=torch.inf)
                equal = (left == right) | (left.isnan() & right.isnan())
                indices = torch.where(~equal)[0]
                indices = indices[torch.argsort(delta[indices], descending=True)]
                detail["different_components"] = len(indices)
                detail["largest_first_frame_differences"] = [
                    {
                        "index": int(index),
                        "current": float(left[index]),
                        "other": float(right[index]),
                        "absolute_difference": float(delta[index]),
                    }
                    for index in indices[:16]
                ]
        policy_trace[key] = detail
    physics_substeps = {}
    for key in (
        "physics_substep_dof_pos",
        "physics_substep_dof_vel",
        "physics_substep_torque",
        "physics_substep_position_target",
    ):
        available = all(key in record["post"] for record in records)
        detail = {"available_in_both_runs": available}
        if available:
            frame = first_nonzero[f"post.{key}"]
            detail["first_nonzero_frame"] = frame
            if frame is not None:
                left, right = [record["post"][key][frame] for record in records]
                equal = (left == right) | (left.isnan() & right.isnan())
                substep = int(torch.where((~equal).any(-1))[0][0])
                detail.update(
                    first_nonzero_substep=substep,
                    different_joint_indices=torch.where(~equal[substep])[0].tolist(),
                    current_substep_values=left[substep].tolist(),
                    other_substep_values=right[substep].tolist(),
                )
        physics_substeps[key] = detail
    frame_counts = [len(record["observation_state"]) for record in records]
    stages_equal = records[0].get("stages") == records[1].get("stages")
    report = {
        "seed": episodes[0]["seed"],
        "compared_frames": count,
        "frame_counts": frame_counts,
        "stage_sequences_equal": stages_equal,
        "descriptors_equal": metadata_equal,
        "tolerance": tolerance,
        "maximum_absolute_difference_by_field": differences,
        "first_frame_exceeding_tolerance_by_field": first_differences,
        "first_nonzero_difference_by_field": first_nonzero,
        "initial_frame_absolute_difference_by_field": initial_differences,
        "base_component_differences": physical_components,
        "raw_policy_trace_comparison": policy_trace,
        "physics_substep_comparison": physics_substeps,
        "within_tolerance": metadata_equal
        and frame_counts[0] == frame_counts[1]
        and stages_equal
        and all(value <= tolerance for value in differences.values()),
        "other_run": str(other),
    }
    (run / "batch_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def compare_initialization_probes(run, other):
    """Locate the first differing initialization physics substep, before task planning."""
    reports = [json.loads((path / "stability_probe.json").read_text()) for path in (run, other)]
    records = [
        torch.load(path / "stability_history.pt", map_location="cpu", weights_only=False)
        for path in (run, other)
    ]
    rows = []
    for seed in sorted(set(reports[0]["seeds"]) & set(reports[1]["seeds"])):
        indices = [report["seeds"].index(seed) for report in reports]
        first, initial = {}, {}
        for key in records[0]["initial"].keys() & records[1]["initial"].keys():
            left, right = [record["initial"][key][index] for record, index in zip(records, indices)]
            if left.shape == right.shape:
                initial[key] = bool(((left == right) | (left.isnan() & right.isnan())).all())
        for key in records[0].keys() & records[1].keys() - {"initial"}:
            left, right = [record[key][:, index] for record, index in zip(records, indices)]
            if key == "distractor_states":
                slots = min(left.shape[1], right.shape[1])
                left, right = left[:, :slots], right[:, :slots]
            equal = (
                (left == right) | (left.isnan() & right.isnan())
                if left.is_floating_point()
                else left == right
            )
            unequal = ~equal.reshape(len(left), -1).all(-1)
            frames = torch.where(unequal)[0]
            detail = None
            if len(frames):
                frame = int(frames[0])
                delta = (left[frame].double() - right[frame].double()).abs()
                detail = {"step": frame + 1, "maximum_absolute_difference": float(delta.max())}
                if key.startswith("physics_substep_"):
                    detail["substep"] = int(torch.where(~equal[frame].all(-1))[0][0])
            first[key] = detail
        prefix_keys = (
            "lowlevel_policy_observation",
            "lowlevel_policy_action",
            "physics_substep_dof_pos",
            "physics_substep_dof_vel",
            "physics_substep_torque",
            "physics_substep_position_target",
        )
        prefix = {}
        for key in prefix_keys:
            if all(key in record for record in records):
                left, right = [record[key][0, index] for record, index in zip(records, indices)]
                if key.startswith("physics_substep_"):
                    left, right = left[0], right[0]
                prefix[key] = bool(torch.equal(left, right))
        velocity = first.get("physics_substep_dof_vel")
        physics_first = (
            all(initial.get(key) for key in ("root_states", "dof_pos", "dof_vel", "object_state"))
            and len(prefix) == len(prefix_keys)
            and all(prefix.values())
            and velocity is not None
            and velocity["step"] == 1
            and velocity["substep"] > 0
        )
        rows.append(
            {
                "seed": seed,
                "initial_equal": initial,
                "first_difference_by_field": first,
                "first_policy_and_prephysics_substep_equal": prefix,
                "origin": "physics_substep_after_equal_inputs"
                if physics_first
                else "requires_further_analysis",
            }
        )
    result = {"current": str(run), "other": str(other), "episodes": rows}
    (run / "initialization_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def compare_door_runs(run, other):
    """Locate exact first differences in standard door audit records, not just outcomes."""
    import pyarrow.parquet as pq

    def episodes(root):
        if (root / "meta/acceptance_report.json").is_file():
            return {root.name: root}
        return {
            path.parent.parent.name: path.parent.parent
            for path in root.glob("shards/episode-seed*/meta/acceptance_report.json")
        }

    current, reference = episodes(run), episodes(other)
    if not current or current.keys() != reference.keys():
        raise ValueError("Door comparisons require the same nonempty seed set")
    rows = []
    for name in sorted(current):
        tables, outcomes = [], []
        for root in (current[name], reference[name]):
            paths = list(root.glob("**/raw_audit/chunk-*/episode_*.parquet"))
            if len(paths) != 1:
                raise ValueError(f"Expected one raw audit for {root}, got {len(paths)}")
            tables.append(pq.read_table(paths[0]).to_pydict())
            outcomes.append(json.loads((root / "meta/acceptance_report.json").read_text()))
        count = min(len(table["frame_index"]) for table in tables)
        if count == 0:
            raise ValueError("Door audit has no control frames")
        first = {}
        for key in sorted(tables[0].keys() & tables[1].keys()):
            values = [np.asarray(table[key][:count]) for table in tables]
            if values[0].shape != values[1].shape:
                first[key] = {"shape_mismatch": [list(value.shape) for value in values]}
                continue
            equal = values[0] == values[1]
            numeric = all(value.dtype.kind in "bifu" for value in values)
            if numeric:
                equal |= np.isnan(values[0]) & np.isnan(values[1])
            indices = np.flatnonzero(~equal.reshape(count, -1).all(-1))
            if not len(indices):
                continue
            index = int(indices[0])
            detail = {"action_frame_index": index}
            if numeric:
                delta = np.where(
                    equal[index], 0,
                    np.abs(values[0][index].astype(float) - values[1][index].astype(float)),
                )
                detail["max_absolute_difference_at_first_frame"] = float(np.nanmax(delta))
                if key.startswith("physics_substep_"):
                    joints = len(tables[0]["all_joint_position"][0])
                    substeps = equal[index].reshape(-1, joints)
                    detail["first_different_substep_index"] = int(
                        np.flatnonzero(~substeps.all(-1))[0]
                    )
                if values[0][index].size <= 19:
                    detail.update(
                        current=values[0][index].tolist(),
                        reference=values[1][index].tolist(),
                    )
            else:
                detail.update(current=values[0][index].tolist(), reference=values[1][index].tolist())
            first[key] = detail
        rows.append({
            "episode": name, "compared_frames": count,
            "frame_counts": [len(table["frame_index"]) for table in tables],
            "first_exact_difference": first,
            "fields_missing_in_reference": sorted(tables[0].keys() - tables[1].keys()),
            "outcomes_current_reference": outcomes,
        })
    report = {"current": str(run), "reference": str(other), "episodes": rows}
    (run / "door_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--gripper-joint-index", type=int, help="Explicit metadata for older recordings"
    )
    parser.add_argument("--mosaic-times-s", type=float, nargs="*", default=[])
    parser.add_argument("--floor-tracking", action="store_true")
    parser.add_argument("--force-stage", help="Export measured contact/force/slip history")
    parser.add_argument("--trunk-transform", action="store_true")
    parser.add_argument("--terminal-inverse", action="store_true")
    parser.add_argument("--compare-run", type=Path)
    parser.add_argument("--compare-initialization", type=Path)
    parser.add_argument("--compare-door-run", type=Path)
    parser.add_argument(
        "--goal-bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    args = parser.parse_args()
    if args.compare_door_run is not None:
        compare_door_runs(args.run, args.compare_door_run)
    elif args.compare_initialization is not None:
        compare_initialization_probes(args.run, args.compare_initialization)
    elif args.terminal_inverse:
        inspect_terminal_inverse(args.run)
    elif args.trunk_transform:
        inspect_trunk_transform(args.run)
    elif args.force_stage is not None:
        inspect_force_history(args.run, args.force_stage)
    elif args.compare_run is not None:
        compare_recorded_runs(args.run, args.compare_run)
    elif args.goal_bounds is not None:
        inspect_goal_bounds(args.run, args.goal_bounds)
    elif args.floor_tracking:
        inspect_floor_tracking(args.run)
    else:
        inspect(args.run, args.gripper_joint_index, mosaic_times_s=args.mosaic_times_s)


if __name__ == "__main__":
    main()
