"""Offline MOVE_TO_PANEL_STANDOFF arrival, joint-target FK and contact timing audit."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch

from gr00t.rl.auto_open.geometry import _read_collider_points
from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
from gr00t.rl.scripts.probe_auto_open_arm_wall import Robot, points_at


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    failures = [
        f
        for p in args.dataset.glob("logs/rank*-control/summary.json")
        for f in json.loads(p.read_text())["failures"]
        if f["stage"] == "MOVE_TO_PANEL_STANDOFF"
        and f["reason"] == "contact_panel_acceptance_timeout"
    ]
    robot, shapes = None, None
    result = []
    for failure in failures:
        seed = failure["seed"]
        shard = args.dataset / "shards" / f"episode-seed{seed}"
        meta = json.loads(
            next(args.dataset.glob(f"logs/rank*-control/seed_{seed}/metadata.json")).read_text()
        )
        prov = json.loads((shard / "meta/simulation_provenance.json").read_text())
        if robot is None:
            robot = Robot(prov["robot_asset"]["path"])
            shapes = {}
            for body, tokens, label in [
                ("link06", ("fixed_convex_023", "fixed_convex_024"), "fixed_jaw"),
                ("link06", ("",), "fixed_all"),
                ("gripperMover", ("",), "moving_all"),
            ]:
                prim = next(p for p in robot.stage.Traverse() if p.GetName() == body)
                shapes[label] = (
                    str(prim.GetPath()),
                    _read_collider_points(robot.stage, prim, tokens).points_body,
                )
        d = pq.read_table(next(shard.rglob("raw_audit/**/*.parquet"))).to_pydict()
        contact = np.flatnonzero(np.asarray(d["autoopen_stage"]) == "MOVE_TO_PANEL_STANDOFF")
        ids = np.split(contact, np.flatnonzero(np.diff(contact) > 1) + 1)[-1]
        entry, end = int(ids[0]), int(ids[-1])
        root = np.concatenate((d["root_position_world"], d["root_quaternion_xyzw"]), axis=1)
        before = np.maximum(ids - 1, 0)
        initial = root[max(entry - 1, 0)]
        actual_ee = np.asarray(d["ee_pose_world_xyzw"])
        actual_rot = Rotation.from_quat(actual_ee[ids, 3:]) * Rotation.from_euler("x", -math.pi / 2)
        poses = robot.poses(
            np.r_[initial[:3], initial[[6, 3, 4, 5]]],
            d["arm_joint_position"][max(entry - 1, 0)],
            d["gripper_feedback"][max(entry - 1, 0)],
        )
        start_ee = actual_ee[max(entry - 1, 0)]
        start_rot = Rotation.from_quat(start_ee[3:]) * Rotation.from_euler("x", -math.pi / 2)
        local = {
            label: start_rot.inv().apply(points_at(points, poses[body]) - start_ee[:3])
            for label, (body, points) in shapes.items()
        }
        target_p, target_r, _ = b2z1_gripper_pose_and_jacobian(
            torch.tensor(root[before, :3], dtype=torch.float64),
            torch.tensor(root[before, 3:], dtype=torch.float64),
            torch.tensor(np.asarray(d["arm_joint_position_target"])[ids], dtype=torch.float64),
        )
        target_p, target_r = target_p.numpy(), target_r.numpy()
        prims = next(iter(prov["episodes"].values()))["door_asset"]["prims"]
        attrs = lambda suffix: next(p["attributes"] for p in prims if p["path"].endswith(suffix))
        ha, panel = attrs("/hinge_joint"), attrs("/door_panel/panel")
        origin = (
            np.asarray(meta["robot_state"]["initial_base_pose_w"][:3])
            - meta["robot_state"]["initial_base_pose_env"][:3]
        )
        hinge = origin + ha["physics:localPos0"]
        hq = ha.get("physics:localRot0", dict(real=1.0, imaginary=[0.0, 0.0, 0.0]))
        sign = np.sign(Rotation.from_quat(hq["imaginary"] + [hq["real"]]).apply([0.0, 0.0, 1.0])[2])
        closed = np.array([1.0 if meta["scene"]["robot_side"] == "positive_x" else -1.0, 0.0, 0.0])
        angles = np.asarray(d["door_hinge_position"])
        theta0 = angles[max(entry - 1, 0)]
        rotate = Rotation.from_euler("z", sign * angles[ids])
        normal = rotate.apply(np.broadcast_to(closed, (len(ids), 3)))
        center = hinge + rotate.apply(
            np.broadcast_to(origin + panel["xformOp:translate"] - hinge, (len(ids), 3))
        )
        rotation0 = Rotation.from_euler("z", sign * theta0)
        normal0 = rotation0.apply(closed)
        center0 = hinge + rotation0.apply(origin + panel["xformOp:translate"] - hinge)
        side = np.sign(np.dot(initial[:3] - center0, normal0))
        normal *= side
        normal0 *= side
        clearances = {}
        for label, points in local.items():
            actual_cloud = actual_ee[ids, None, :3] + np.einsum(
                "nij,pj->npi", actual_rot.as_matrix(), points
            )
            target_cloud = target_p[:, None] + np.einsum("nij,pj->npi", target_r, points)
            clearances[label] = dict(
                actual=np.einsum("npi,ni->np", actual_cloud - center[:, None], normal).min(-1)
                - panel["xformOp:scale"][0],
                target=np.einsum("npi,i->np", target_cloud - center0, normal0).min(-1)
                - panel["xformOp:scale"][0],
            )
        events = pq.read_table(next(shard.rglob("raw_contacts/**/*.parquet"))).to_pylist()
        events = [
            e
            for e in events
            if entry <= e["frame_index"] <= end
            and "door_panel" in (e["body_a_path"] + e["body_b_path"])
            and any(
                body in (e["body_a_path"] + e["body_b_path"]) for body in ("link06", "gripperMover")
            )
        ]
        first = min((e["frame_index"] for e in events), default=None)
        item = dict(
            seed=seed,
            entry=entry,
            end=end,
            first_panel_contact=first,
            door_delta_deg=float(np.rad2deg(angles[end] - theta0)),
            door_delta_at_contact_deg=None
            if first is None
            else float(np.rad2deg(angles[first] - theta0)),
            base_drift_m=float(np.linalg.norm(root[end, :2] - initial[:2])),
            target_tracking_error_last_m=float(np.linalg.norm(target_p[-1] - actual_ee[end, :3])),
            clearance={
                label: dict(
                    min_actual_m=float(c["actual"].min()),
                    min_target_frozen_panel_m=float(c["target"].min()),
                    last_actual_m=float(c["actual"][-1]),
                    last_target_frozen_panel_m=float(c["target"][-1]),
                )
                for label, c in clearances.items()
            },
            first_target_panel_crossing=next(
                (
                    int(ids[j])
                    for j in range(len(ids))
                    if min(c["target"][j] for c in clearances.values()) < 0.0
                ),
                None,
            ),
            conditions=failure["acceptance_conditions"],
        )
        result.append(item)
        print("audited", seed, flush=True)
    summary = dict(
        episodes=len(result),
        with_panel_contact=sum(r["first_panel_contact"] is not None for r in result),
        target_crosses_initial_panel=sum(
            r["first_target_panel_crossing"] is not None for r in result
        ),
        door_moves_over_1deg=sum(abs(r["door_delta_deg"]) > 1.0 for r in result),
    )
    args.output.write_text(json.dumps(dict(summary=summary, records=result), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
