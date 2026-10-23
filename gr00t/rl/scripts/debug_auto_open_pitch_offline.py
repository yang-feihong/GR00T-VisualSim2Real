"""Offline pitch counterfactual using logged base motion; never starts Isaac Sim."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[3]
LOWLEVEL = REPO / "gr00t/rl/third_party/visual_wholebody_lowlevel"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(LOWLEVEL))

from gr00t.rl.envs.door.b2z1_reset_geometry import (  # noqa: E402
    b2z1_gripper_pose_and_jacobian as fk,
)
from legged_gym.utils.constrained_ik import (  # noqa: E402
    batched_constrained_dls_delta,
)
from legged_gym.utils.ee_steady_inverse_gpu import (  # noqa: E402
    _matrix_to_quaternion_wxyz,
    _quaternion_multiply_wxyz,
    batched_steady_ee_inverse_no_gravity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=642)
    parser.add_argument("--end-frame", type=int, default=740)
    parser.add_argument("--handle-press-deg", type=float, default=31.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit = pq.read_table(args.audit).to_pydict()
    selection = slice(args.start_frame, args.end_frame + 1)
    positions = np.array(audit["ee_inverse_input_world"])[selection]
    fit = positions[:28, :2]
    center = np.linalg.lstsq(
        np.c_[2 * fit, np.ones(len(fit))], (fit * fit).sum(axis=1), rcond=None
    )[0][:2]
    angles = np.unwrap(np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0]))
    angles -= angles[0]
    positions[:, 2] = positions[0, 2]
    roots = np.array(audit["root_position_world"])[selection]
    root_quats = np.array(audit["root_quaternion_xyzw"])[selection]
    initial = np.array(audit["arm_joint_position"][args.start_frame])
    urdf = LOWLEVEL / "resources/robots/b2z1/urdf/b2z1.urdf"
    tree = ET.parse(urdf).getroot()
    limits = []
    for index in range(1, 7):
        limit = tree.find(f"joint[@name='joint{index}']/limit")
        limits.append([float(limit.get("lower")), float(limit.get("upper"))])

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)

    limits = tensor(limits)
    pitches = list(range(0, 41, 5))
    orientations = np.array([
        (
            Rotation.from_rotvec([0, 0, angle])
            * Rotation.from_euler("x", args.handle_press_deg, degrees=True)
            * Rotation.from_euler("y", -pitch, degrees=True)
            * Rotation.from_euler("xyz", [180, 0, 180], degrees=True)
        ).as_quat()[[3, 0, 1, 2]]
        for pitch in pitches for angle in angles
    ]).reshape(len(pitches), len(positions), 4)
    report = {
        "audit": str(args.audit),
        "frames": [args.start_frame, args.end_frame],
        "assumptions": [
            "Kinematic counterfactual, no physics simulation or contact-force prediction.",
            "Same EE position targets for all pitches; no grasp-center height adjustment.",
            "EE height fixed to first recorded pull command.",
            "Yaw sweep recovered from circular fit of recorded EE commands.",
            "Nominal horizontal zero-pitch orientation, reconstructed handle press angle.",
            "Each pitch starts at its converged steady IK state.",
            "Each subsequent joint target is assumed reached before the next control step.",
            "Recorded base motion is imposed, not predicted for the changed pitch.",
        ],
        "handle_press_deg": args.handle_press_deg,
        "circle_center_xy": center.tolist(),
        "results": {},
    }
    for mode in ("entry_base_fixed", "recorded_base_motion"):
        count = len(pitches)
        desired = tensor(np.tile(positions, (count, 1, 1)))
        orientation = tensor(orientations)
        fixed = mode == "entry_base_fixed"
        base = tensor(np.tile(roots[:1] if fixed else roots, (count, 1, 1)))
        base_q = tensor(np.tile(root_quats[:1] if fixed else root_quats, (count, 1, 1)))
        if fixed:
            base = base.expand(count, len(positions), 3)
            base_q = base_q.expand(count, len(positions), 4)
        start = batched_steady_ee_inverse_no_gravity(
            desired[:, 0], orientation[:, 0],
            base_position_world=base[:, 0],
            base_orientation_world_xyzw=base_q[:, 0],
            initial_arm_joint_positions=tensor(np.tile(initial, (count, 1))),
            joint_limits=limits,
        )
        joints = start.arm_joint_positions
        heights, errors, margins = [], [], []
        for index in range(len(positions)):
            position, rotation, jacobian = fk(base[:, index], base_q[:, index], joints)
            current = _matrix_to_quaternion_wxyz(rotation)
            conjugate = current.clone()
            conjugate[:, 1:] *= -1
            relative = _quaternion_multiply_wxyz(orientation[:, index], conjugate)
            rotation_error = relative[:, 1:] * torch.sign(relative[:, :1]) * 4.0
            error = torch.cat((desired[:, index] - position, rotation_error), dim=-1)
            joints += batched_constrained_dls_delta(jacobian, error, joints, limits, 0.05)
            output, _, _ = fk(base[:, index], base_q[:, index], joints)
            heights.append((output[:, 2] - desired[:, index, 2]).abs().numpy() * 1000)
            errors.append(torch.linalg.vector_norm(output - desired[:, index], dim=-1).numpy() * 1000)
            margins.append((joints[:, 3] - limits[3, 0]).numpy())
        report["results"][mode] = [
            {
                "pitch_deg": pitch,
                "max_height_error_mm": float(np.max(heights, axis=0)[index]),
                "max_position_error_mm": float(np.max(errors, axis=0)[index]),
                "q4_min_lower_limit_margin_rad": float(np.min(margins, axis=0)[index]),
            }
            for index, pitch in enumerate(pitches)
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
