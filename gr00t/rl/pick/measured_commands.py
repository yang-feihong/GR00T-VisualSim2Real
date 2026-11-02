"""Immutable, fingerprinted low-reach measurements shared by offline and GPU planners."""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import torch

from gr00t.rl.auto_open.motion_math import _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate


@lru_cache(maxsize=32)
def _read_report(path, modified_ns, size):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def report_key(path):
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    path = path.resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def read_measured_report(path):
    return _read_report(*report_key(path))


@lru_cache(maxsize=16)
def _load_commands(keys, device, dtype, maximum_position_error_m=None):
    positions, orientations, identifiers = [], [], []
    checkpoint = None
    for key in keys:
        report, _ = _read_report(*key)
        position_limit = (
            report.get("maximum_position_error_m", 0.004)
            if maximum_position_error_m is None
            else maximum_position_error_m
        )
        if not math.isfinite(position_limit) or not 0 < position_limit <= 0.008:
            raise ValueError("Measured position error must leave room within task tolerance")
        digest = report.get("checkpoint_sha256")
        if not digest or checkpoint not in (None, digest):
            raise ValueError("Measured banks require one fingerprinted checkpoint")
        checkpoint = digest
        if not report.get("synthetic_count") or report["arm_ik_mode"] != "constrained":
            raise ValueError("Use asset-independent synthetic constrained-controller probes")
        if abs(report.get("gripper_angle_rad", 0) + math.pi / 3) > 1e-7:
            raise ValueError("Measured grasp proposals require fully-open-jaw probes")
        roots = torch.tensor(report["final_root_states"], device=device, dtype=dtype)
        goals = roots.new_tensor(report["goals_world"])
        quats = roots.new_tensor(report["goal_quaternions_wxyz"])
        valid = (roots.new_tensor(report["final_position_error_m"]) <= position_limit) & (
            roots.new_tensor(report["final_orientation_error_rad"]) <= 0.02
        )
        valid &= roots.new_tensor(report["goal_gripper_floor_clearance_m"]) >= 0.002
        valid &= roots[:, 2] > 0.18
        if "final_nonfoot_contact_peak_n" in report:
            valid &= roots.new_tensor(report["final_nonfoot_contact_peak_n"]) < 20
        w, x, y, z = roots[:, 3:7].unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        inverse = torch.stack(((yaw / 2).cos(), yaw * 0, yaw * 0, -(yaw / 2).sin()), -1)
        delta = goals - roots[:, :3]
        delta[:, 2] = goals[:, 2]
        positions.append(_quat_rotate(inverse, delta)[valid])
        orientations.append(_quat_multiply(inverse, quats)[valid])
        ids = (
            torch.arange(len(roots), device=device) + report["synthetic_offset"]
            if report.get("source_sample_ids") is None
            else torch.as_tensor(report["source_sample_ids"], device=device, dtype=torch.long)
        )
        identifiers.append(ids[valid])
    local_goal, local_quat, sample_id = map(torch.cat, (positions, orientations, identifiers))
    if not len(local_goal):
        raise ValueError("The supplied probes contain no accurately reached, floor-clear commands")
    if len(sample_id.unique()) != len(sample_id):
        raise ValueError("Measured probe indices overlap")
    return local_goal, local_quat, sample_id, checkpoint


def load_measured_commands(paths, reference, *, maximum_position_error_m=None):
    """Cache only asset metadata; a changed catalog gets a new stat/version key."""
    return _load_commands(
        tuple(report_key(path) for path in paths),
        reference.device,
        reference.dtype,
        maximum_position_error_m,
    )
