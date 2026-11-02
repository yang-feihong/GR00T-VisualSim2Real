"""Offline, asset-independent whole-body reach-response feasibility measurements."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from gr00t.rl.auto_open.motion_math import _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate


def reach_features(roots, goals, orientations, initial_arm):
    w, x, y, z = roots[:, 3:7].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    inverse = torch.stack(((yaw / 2).cos(), yaw * 0, yaw * 0, -(yaw / 2).sin()), -1)
    position = _quat_rotate(inverse, goals - roots[:, :3])
    rotation = _quat_multiply(inverse, orientations)
    axes = _quat_rotate(rotation[:, None], goals.new_tensor([[1, 0, 0], [0, 1, 0]])[None]).flatten(
        1
    )
    return torch.cat((position, axes, initial_arm), -1), inverse


def read_probe(path, device):
    report = json.loads((path / "summary.json").read_text())
    if not report.get("synthetic_count") or report["arm_ik_mode"] != "constrained":
        raise ValueError("Calibration requires independent synthetic constrained-controller probes")
    history = torch.load(path / "trace.pt", weights_only=False, map_location=device)[-25:]
    roots = history[..., :13].mean(0)
    initial = roots.new_tensor(report["initial_root_states"])
    goals = roots.new_tensor(report["goals_world"])
    orientations = roots.new_tensor(report["goal_quaternions_wxyz"])
    arm = roots.new_tensor(report["initial_arm_joints_rad"])
    features, inverse = reach_features(initial, goals, orientations, arm)
    w, x, y, z = roots[:, 3:7].unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square()))
    pitch = (2 * (w * y - z * x)).clamp(-1, 1).asin()
    error = _quat_rotate(inverse, history[..., 13:16].mean(0) - goals)
    labels = torch.cat(
        (
            (roots[:, 2] - initial[:, 2])[:, None],
            roll[:, None],
            pitch[:, None],
            error,
            history[..., -1].mean(0)[:, None],
        ),
        -1,
    )
    floor = roots.new_tensor(report["goal_gripper_floor_clearance_m"])
    stable = (history[..., 7:10].norm(dim=-1).amax(0) < 0.10) & (
        history[..., 10:13].norm(dim=-1).amax(0) < 0.35
    )
    usable = torch.isfinite(features).all(-1) & torch.isfinite(labels).all(-1)
    usable &= (floor >= 0.002) & stable & (roots[:, 2] > 0.18)
    success = (history[..., -2].amax(0) < 0.012) & (history[..., -1].amax(0) < 0.06)
    index = (
        torch.as_tensor(report["source_sample_ids"], device=device)
        if "source_sample_ids" in report
        else torch.arange(len(roots), device=device) + report["synthetic_offset"]
    )
    return features[usable], labels[usable], success[usable], index[usable], report


def make_model(inputs=15):
    return nn.Sequential(
        nn.Linear(inputs, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 7),
    )


def export_measured_commands(probes, output, *, device="cuda:0", maximum_position_error_m=0.004):
    """Publish measured commands only, not scene-specific successful grasp poses."""
    import math

    if not math.isfinite(maximum_position_error_m) or not 0 < maximum_position_error_m <= 0.008:
        raise ValueError("Measured position error must leave room within task tolerance")
    keys = (
        "final_root_states",
        "goals_world",
        "goal_quaternions_wxyz",
        "final_position_error_m",
        "final_orientation_error_rad",
        "goal_gripper_floor_clearance_m",
    )
    collected = {key: [] for key in keys}
    ids, sources, checkpoint = [], [], None
    total = 0
    for probe in probes:
        source = probe / "summary.json"
        raw = source.read_bytes()
        report = json.loads(raw)
        digest = report.get("checkpoint_sha256")
        if not digest or checkpoint not in (None, digest):
            raise ValueError("Measured banks require one fingerprinted checkpoint")
        checkpoint = digest
        if not report.get("synthetic_count") or report["arm_ik_mode"] != "constrained":
            raise ValueError("Use asset-independent synthetic constrained-controller probes")
        if abs(report.get("gripper_angle_rad", 0) + torch.pi / 3) > 1e-7:
            raise ValueError("Measured grasp proposals require fully-open-jaw probes")
        values = {key: torch.as_tensor(report[key], device=device) for key in keys}
        valid = (values["final_position_error_m"] <= maximum_position_error_m) & (
            values["final_orientation_error_rad"] <= 0.02
        )
        valid &= (values["goal_gripper_floor_clearance_m"] >= 0.002) & (
            values["final_root_states"][:, 2] > 0.18
        )
        if "final_nonfoot_contact_peak_n" in report:
            valid &= torch.as_tensor(report["final_nonfoot_contact_peak_n"], device=device) < 20
        index = (
            torch.as_tensor(report["source_sample_ids"], device=device)[valid]
            if "source_sample_ids" in report
            else torch.arange(len(valid), device=device)[valid] + report["synthetic_offset"]
        )
        ids.extend(index.tolist())
        for key in keys:
            collected[key].extend(values[key][valid].tolist())
        sources.append(
            {
                "path": str(source),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "synthetic_count": report["synthetic_count"],
            }
        )
        total += report["synthetic_count"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Measured command identifiers must be nonempty and disjoint")
    report = {
        "schema": "pick_measured_low_commands_v1",
        "source_probes": sources,
        "checkpoint_sha256": checkpoint,
        "synthetic_count": total,
        "synthetic_offset": 0,
        "source_sample_ids": ids,
        "measured_command_count": len(ids),
        "maximum_position_error_m": maximum_position_error_m,
        "arm_ik_mode": "constrained",
        "gripper_angle_rad": -torch.pi / 3,
        "note": "Free-space measurements, not a grasp/path acceptance guarantee",
        **collected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def export_robust_measured_commands(probe, output, *, device="cuda:0"):
    """Choose precise centers with independently tested neighboring commands."""
    source = probe / "summary.json"
    raw = source.read_bytes()
    report = json.loads(raw)
    if not report.get("synthetic_catalog") or report.get("synthetic_count", 0) % 5:
        raise ValueError("Robust export requires five longitudinal samples per command")
    if report.get("arm_ik_mode") != "constrained" or not report.get("checkpoint_sha256"):
        raise ValueError("Robust export requires fingerprinted constrained-controller probes")
    if abs(report.get("gripper_angle_rad", 0) + torch.pi / 3) > 1e-7:
        raise ValueError("Robust export requires fully open jaws")
    history = torch.load(probe / "trace.pt", weights_only=False, map_location=device)[-25:]
    error = history[..., -2].amax(0).reshape(-1, 5)
    orientation = history[..., -1].amax(0).reshape(-1, 5)
    stable = (history[..., 7:10].norm(dim=-1).amax(0) < 0.10) & (
        history[..., 10:13].norm(dim=-1).amax(0) < 0.35
    )
    safe = torch.isfinite(history).all((0, 2)) & stable
    safe &= history[..., 2].amin(0) > 0.18
    safe &= history.new_tensor(report["goal_gripper_floor_clearance_m"]) >= 0.002
    safe &= history.new_tensor(report["final_nonfoot_contact_peak_n"]) < 20
    safe = safe.reshape(-1, 5) & (orientation <= 0.02) & (error <= 0.011)
    windows = safe.unfold(1, 3, 1).all(-1) & (error[:, 1:4] <= 0.004)
    worst = error.unfold(1, 3, 1).amax(-1)
    score = worst + 0.01 * error[:, 1:4]
    choice = score.masked_fill(~windows, torch.inf).argmin(-1)
    valid = windows.any(-1)
    index = (torch.arange(len(valid), device=device) * 5 + choice + 1)[valid]
    if not len(index):
        raise ValueError("No precise command has a verified longitudinal neighborhood")
    keys = (
        "final_root_states",
        "goals_world",
        "goal_quaternions_wxyz",
        "final_position_error_m",
        "final_orientation_error_rad",
        "goal_gripper_floor_clearance_m",
        "final_nonfoot_contact_peak_n",
    )
    exported = {key: history.new_tensor(report[key])[index].tolist() for key in keys}
    exported.update(
        schema="pick_measured_low_commands_v1",
        source_probes=[{"path": str(source), "sha256": hashlib.sha256(raw).hexdigest()}],
        checkpoint_sha256=report["checkpoint_sha256"],
        synthetic_count=report["synthetic_count"],
        synthetic_offset=0,
        source_sample_ids=torch.as_tensor(report["source_sample_ids"], device=device)[
            index
        ].tolist(),
        measured_command_count=len(index),
        maximum_position_error_m=0.004,
        arm_ik_mode="constrained",
        gripper_angle_rad=-torch.pi / 3,
        robustness={
            "nominal_longitudinal_radius_m": 0.03,
            "steady_frames": len(history),
            "maximum_neighbor_position_error_m": 0.011,
            "maximum_center_position_error_m": 0.004,
            "center_and_neighbors_share_orientation": True,
            "not_an_arrival_or_grasp_guarantee": True,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(exported, indent=2) + "\n")
    return exported


def measured_grasp_matches(
    probes, run, output, *, device="cuda:0", episode=None, state=None, collision=None, geometry=None
):
    """Offline intersection of observed steady commands and convex grasp geometry."""
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
    from gr00t.rl.pick.grasp_quality import convex_grasp_quality, load_grasp_support
    from gr00t.rl.pick.measured_commands import load_measured_commands
    from gr00t.rl.pick.planning import MAX_POCKET_SAMPLE_EVALUATIONS

    if episode is None:
        episode = json.loads((run / "episode.json").read_text())
    if episode.get("support_type") != "floor":
        raise ValueError("Measured low-reach matching currently requires a floor-supported target")
    if state is None:
        state = torch.load(run / "trajectory.pt", weights_only=False, map_location=device)[-1]
    local_goal, local_quat, sample_id, checkpoint = load_measured_commands(
        probes, state["root_states"]
    )
    root = state["root_states"]
    offsets = torch.arange(24, device=device, dtype=root.dtype) * (torch.pi / 12)
    yaw = offsets[None].expand(len(local_goal), -1).flatten()
    body_quat = _euler_xyz_quat(torch.stack((yaw * 0, yaw * 0, yaw), -1))
    orientations = _quat_multiply(body_quat, local_quat.repeat_interleave(len(offsets), 0))
    source_index = torch.arange(len(local_goal), device=device).repeat_interleave(len(offsets))
    support = load_grasp_support(episode["asset"]["usd_path"], device)
    target, object_quat = state["object_state"][:, :3], state["object_state"][:, 3:7]
    floor_height = target.new_tensor(episode["support_z"])
    chunk_size = max(1, MAX_POCKET_SAMPLE_EVALUATIONS // (17 * len(support)))
    quality_chunks = [
        convex_grasp_quality(chunk[None], object_quat, support, return_depth_candidates=True)
        for chunk in orientations.split(chunk_size)
    ]
    quality = {
        key: torch.cat([part[key] for part in quality_chunks], 1)[0]
        for key in ("pocket_fit", "antipodal_margin", "center_local", "aperture_angle_rad")
    }
    center = quality["center_local"].clone()
    center[..., 2] -= 0.0005
    ee = target[:, None] - _quat_rotate(orientations[:, None], center)
    height_error = ee[..., 2] - floor_height - local_goal[source_index, 2, None]
    matched = quality["pocket_fit"] & (quality["antipodal_margin"] >= 0)
    matched &= height_error.abs() <= 0.002
    row, depth = matched.nonzero(as_tuple=True)
    station_xy = (
        ee[row, depth, :2] - _quat_rotate(body_quat[row], local_goal[source_index[row]])[:, :2]
    )
    stations = torch.cat((station_xy, yaw[row, None]), -1)
    opened = root.new_full((len(row),), -torch.inf)
    closing = opened.clone()
    corridor = torch.zeros(len(row), device=device, dtype=torch.bool)
    if len(row):
        from gr00t.rl.pick.collision import ArmCollisionSamples
        from gr00t.rl.pick.distractors import augment_scene_geometry, build_distractor_mesh
        from gr00t.rl.pick.scene_geometry import SceneGeometry, read_scene_triangles
        from gr00t.rl.pick.station import station_corridors_clear

        if geometry is None:
            points, faces = read_scene_triangles(
                Path(episode["scene_usd_path"]), Path(episode["geometry_cache_root"])
            )
            geometry = augment_scene_geometry(
                SceneGeometry(points, faces, device),
                build_distractor_mesh(episode.get("distractors", []), state["distractor_states"]),
            )
        if collision is None:
            collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
            model = torch.load(run / "collision_model.pt", weights_only=False, map_location=device)
            collision.gripper_surface = model["gripper_surface"]
        pose, rotation = ee[row, depth][None], orientations[row][None]
        opened = collision.gripper_clearance(geometry, pose, rotation)[0]
        closing = collision.gripper_closing_clearance(
            geometry, pose, rotation, quality["aperture_angle_rad"][row, depth][None]
        )[0]
        initial = (
            state
            if run is None
            else torch.load(run / "trajectory.pt", weights_only=False, map_location=device)[0]
        )
        corridor = station_corridors_clear(initial["root_states"], stations[None], geometry)[0]
    scene_safe = (opened >= 0.002) & (closing >= 0.002) & corridor
    if output is None:
        return {
            "stations": stations,
            "orientations": orientations[row],
            "goals": ee[row, depth],
            "depth_preferences": depth.to(root.dtype) / 16,
            "sample_ids": sample_id[source_index[row]],
            "feasible": scene_safe,
            "open_clearance_m": opened,
            "closing_clearance_m": closing,
            "checkpoint_sha256": checkpoint,
            "measured_command_count": len(local_goal),
        }
    result = {
        "source_run": str(run),
        "source_probes": [str(path) for path in probes],
        "checkpoint_sha256": checkpoint,
        "measured_command_count": len(local_goal),
        "candidate_count": int(matched.numel()),
        "matched_geometry_count": len(row),
        "height_tolerance_m": 0.002,
        "not_collision_certified": True,
        "not_integrated_into_collection": True,
        "note": "Observed free-space command match, not a verified grasp or path",
        "station_poses": stations.tolist(),
        "candidate_orientations_world_wxyz": orientations[row].tolist(),
        "candidate_grasp_origins_world_m": ee[row, depth].tolist(),
        "pocket_depth_preferences": (depth / 16).tolist(),
        "source_sample_ids": sample_id[source_index[row]].tolist(),
        "height_error_m": height_error[row, depth].tolist(),
        "open_gripper_scene_clearance_m": opened.tolist(),
        "closing_gripper_scene_clearance_m": closing.tolist(),
        "base_corridor_clear": corridor.tolist(),
        "endpoint_gripper_and_corridor_safe": scene_safe.tolist(),
        "scene_safe_candidate_count": int(scene_safe.sum()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("measured_command_count", "candidate_count", "matched_geometry_count")
            }
        )
    )
    return result


class CalibratedBodyResponse:
    """Stage-entry GPU body-pose prediction; never a control or acceptance override."""

    def __init__(self, path, device, *, allow_unverified=False):
        self.path = str(path)
        bundle = torch.load(path, weights_only=False, map_location=device)
        if not allow_unverified:
            raise ValueError(
                "Body-response models are experimental and not approved for collection"
            )
        self.mean_x, self.std_x = bundle["input_mean"], bundle["input_std"]
        self.mean_y, self.std_y = bundle["output_mean"], bundle["output_std"]
        self.low = bundle.get("input_min", self.mean_x - 2 * self.std_x)
        self.high = bundle.get("input_max", self.mean_x + 2 * self.std_x)
        self.models = []
        for weights in bundle["models"]:
            model = make_model(len(self.mean_x)).to(device)
            model.load_state_dict(weights)
            self.models.append(model.eval())

    @torch.no_grad()
    def __call__(self, reference, positions, orientations, initial_arm):
        from gr00t.rl.auto_open.motion_math import _euler_xyz_quat

        features, _ = reach_features(reference, positions, orientations, initial_arm)
        features = features[:, : len(self.mean_x)]
        normalized = (features - self.mean_x) / self.std_x
        prediction = torch.stack([model(normalized) for model in self.models]).mean(0)
        prediction = prediction * self.std_y + self.mean_y
        valid = (
            (features[:, :3] >= self.low[:3] - 0.01) & (features[:, :3] <= self.high[:3] + 0.01)
        ).all(-1)
        valid &= torch.isfinite(prediction).all(-1) & (prediction[:, :3].abs() < 0.4).all(-1)
        w, x, y, z = reference[:, 3:7].unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        roots = reference.clone()
        roots[:, 2] += prediction[:, 0]
        roots[:, 3:7] = _euler_xyz_quat(torch.stack((prediction[:, 1], prediction[:, 2], yaw), -1))
        return roots, valid


def fit(probes, output, *, steps=2000, ensemble=3, device="cuda:0", include_initial_arm=False):
    if steps < 1 or ensemble < 1:
        raise ValueError("Fit steps and ensemble size must be positive")
    if (output / "model.pt").exists():
        raise FileExistsError(output / "model.pt")
    data = [read_probe(path, device) for path in probes]
    x, y, success, indices = (torch.cat([row[i] for row in data]) for i in range(4))
    if not include_initial_arm:
        x = x[:, :9]
    if len(indices.unique()) != len(indices):
        raise ValueError("Calibration shards overlap")
    test = indices.remainder(5) == 0
    train = ~test
    if int(train.sum()) < 32 or int(test.sum()) < 16:
        raise ValueError("Insufficient independent training/validation commands")
    mean_x, std_x = x[train].mean(0), x[train].std(0).clamp_min(1e-4)
    mean_y, std_y = y[train].mean(0), y[train].std(0).clamp_min(1e-4)
    normalized_x, normalized_y = (x - mean_x) / std_x, (y - mean_y) / std_y
    states, predictions = [], []
    for seed in range(ensemble):
        torch.manual_seed(7919 + seed)
        model = make_model(x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
        model.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(normalized_x[train]) - normalized_y[train]).square().mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions.append(model(normalized_x[test]) * std_y + mean_y)
        states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
    bank = torch.stack(predictions)
    prediction = bank.mean(0)
    residual = prediction - y[test]
    position_bound = prediction[:, 3:6].norm(dim=-1) + 3 * bank[:, :, 3:6].std(
        0, unbiased=False
    ).norm(dim=-1)
    selected = (position_bound < 0.008) & (prediction[:, -1] < 0.04)
    selected_count = int(selected.sum())
    false_positive = int((selected & ~success[test]).sum())
    checkpoint_verified = all(row[4].get("checkpoint_sha256") for row in data)
    fingerprints = {row[4].get("checkpoint_sha256") for row in data}
    if len(fingerprints) > 1:
        raise ValueError("Calibration probes used different checkpoint fingerprints")
    metrics = {
        "samples": len(x),
        "training_samples": int(train.sum()),
        "validation_samples": int(test.sum()),
        "source_samples": sum(row[4]["synthetic_count"] for row in data),
        "validation_successes": int(success[test].sum()),
        "validation_rmse": residual.square().mean(0).sqrt().tolist(),
        "validation_p95_absolute_error": residual.abs().quantile(0.95, dim=0).tolist(),
        "label_units": [
            "height_delta_m",
            "roll_rad",
            "pitch_rad",
            "ee_x_error_m",
            "ee_y_error_m",
            "ee_z_error_m",
            "orientation_error_rad",
        ],
        "predicted_safe_validation_samples": selected_count,
        "false_positive_validation_samples": false_positive,
        "predicted_safe_precision": 1 - false_positive / selected_count if selected_count else None,
        "checkpoint_fingerprint_verified": checkpoint_verified,
        "ready_for_physical_validation": (
            checkpoint_verified and selected_count >= 10 and false_positive / selected_count <= 0.05
        ),
        "not_integrated_into_collection": True,
        "input_includes_initial_arm": include_initial_arm,
        "source_probes": [str(path) for path in probes],
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "pick_wholebody_response_feasibility_v1",
            "models": states,
            "input_mean": mean_x.cpu(),
            "input_std": std_x.cpu(),
            "input_min": x[train].amin(0).cpu(),
            "input_max": x[train].amax(0).cpu(),
            "output_mean": mean_y.cpu(),
            "output_std": std_y.cpu(),
            "metrics": metrics,
        },
        output / "model.pt",
    )
    (output / "report.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)
    return metrics


def audit_roots(model_path, probe, output, *, device="cuda:0"):
    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
    from gr00t.rl.pick.planning import inverse_bank
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    bundle = torch.load(model_path, weights_only=False, map_location=device)
    report = json.loads((probe / "summary.json").read_text())
    roots = torch.tensor(report["initial_root_states"], device=device)
    goals = roots.new_tensor(report["goals_world"])
    orientations = roots.new_tensor(report["goal_quaternions_wxyz"])
    if "initial_arm_joints_rad" in report:
        arms = roots.new_tensor(report["initial_arm_joints_rad"])
    else:
        source = Path(report["source_run"])
        trace = torch.load(source / "trajectory.pt", weights_only=False, map_location=device)
        collision = torch.load(
            source / "collision_model.pt", weights_only=False, map_location=device
        )
        arms = trace[report["initial_frame"]]["dof_pos"][:, collision["arm_joint_ids"]]
    arms = arms.expand(len(roots), -1)
    features, _ = reach_features(roots, goals, orientations, arms)
    features = features[:, : len(bundle["input_mean"])]
    predictions = []
    for state in bundle["models"]:
        model = make_model(features.shape[1]).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            predictions.append(
                model((features - bundle["input_mean"]) / bundle["input_std"])
                * bundle["output_std"]
                + bundle["output_mean"]
            )
    predictions = torch.stack(predictions).mean(0)
    w, x, y, z = roots[:, 3:7].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    predicted_roots = roots.clone()
    predicted_roots[:, 2] += predictions[:, 0]
    predicted_roots[:, 3:7] = _euler_xyz_quat(
        torch.stack((predictions[:, 1], predictions[:, 2], yaw), -1)
    )
    limits = roots.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    seeds = torch.stack((arms, limits.mean(-1).expand_as(arms), limits.sum(-1) - arms), 1)
    result = inverse_bank(
        goals[:, None, None].expand(-1, 3, 1, -1).flatten(0, 1),
        orientations[:, None, None].expand(-1, 3, 1, -1).flatten(0, 1),
        predicted_roots[:, None].expand(-1, 3, -1).flatten(0, 1),
        seeds.flatten(0, 1),
        limits,
    )
    joints = result.arm_joint_positions.reshape(-1, 3, 6)
    margin = torch.minimum(joints - limits[:, 0], limits[:, 1] - joints).amin(-1)
    best = margin.masked_fill(~result.exact.reshape(-1, 3), -torch.inf).amax(-1)
    actual_success = (roots.new_tensor(report["final_position_error_m"]) < 0.012) & (
        roots.new_tensor(report["final_orientation_error_rad"]) < 0.06
    )
    metrics = {}
    for bound in (0.0, 0.02, 0.04, 0.08):
        selected = best >= bound
        count = int(selected.sum())
        metrics[str(bound)] = {
            "predicted_exact_count": count,
            "actual_successes": int((selected & actual_success).sum()),
            "precision": float(actual_success[selected].float().mean()) if count else None,
        }
    selected = int(best.argmax())
    metrics.update(
        source_probe=str(probe),
        selected=selected,
        selected_joint_margin_rad=float(best[selected]),
        selected_actual_position_error_m=report["final_position_error_m"][selected],
        selected_actual_orientation_error_rad=report["final_orientation_error_rad"][selected],
        predicted_roots=predicted_roots.tolist(),
        predicted_joint_margins_rad=best.tolist(),
        not_integrated_into_collection=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({key: value for key, value in metrics.items() if not isinstance(value, list)}))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, nargs="+")
    parser.add_argument("--audit-probe", type=Path)
    parser.add_argument("--match-run", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--include-initial-arm", action="store_true")
    parser.add_argument("--export-commands", action="store_true")
    parser.add_argument("--export-robust-commands", action="store_true")
    parser.add_argument("--maximum-position-error-m", type=float, default=0.004)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.export_robust_commands:
        if (
            not args.probes
            or len(args.probes) != 1
            or args.export_commands
            or args.model
            or args.audit_probe
            or args.match_run
        ):
            parser.error("Robust command export requires exactly one catalog probe")
        result = export_robust_measured_commands(args.probes[0], args.output, device=args.device)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "measured_command_count": result["measured_command_count"],
                }
            )
        )
        return
    if args.export_commands:
        if not args.probes or args.model or args.audit_probe or args.match_run:
            parser.error("Command export requires only source probes and output")
        result = export_measured_commands(
            args.probes,
            args.output,
            device=args.device,
            maximum_position_error_m=args.maximum_position_error_m,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "measured_command_count": result["measured_command_count"],
                }
            )
        )
        return
    if args.match_run:
        if not args.probes or args.model or args.audit_probe:
            parser.error("Measured command matching requires probes, without a fitted model")
        measured_grasp_matches(args.probes, args.match_run, args.output, device=args.device)
        return
    if args.audit_probe:
        if args.model is None or args.probes:
            parser.error("Root audit requires a model and cannot fit at the same time")
        audit_roots(args.model, args.audit_probe, args.output, device=args.device)
    else:
        if not args.probes or args.model is not None:
            parser.error("Fitting requires probes and no existing model")
        fit(
            args.probes,
            args.output,
            steps=args.steps,
            ensemble=args.ensemble,
            device=args.device,
            include_initial_arm=args.include_initial_arm,
        )


if __name__ == "__main__":
    main()
