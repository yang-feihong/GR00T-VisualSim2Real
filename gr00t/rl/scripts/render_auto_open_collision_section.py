#!/usr/bin/env python3
"""Render true handle/gripper collider cross-sections from an AutoOpen dataset."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import trimesh


ARM_JOINT_ORIGINS = (
    (0.0, 0.0, 0.0585),
    (0.0, 0.0, 0.0450),
    (-0.3500, 0.0, 0.0),
    (0.2180, 0.0, 0.0570),
    (0.0700, 0.0, 0.0),
    (0.0492, 0.0, 0.0),
)
ARM_JOINT_AXES = (2, 1, 1, 1, 2, 0)
COLORS = {
    "fixed": "#2c7fb8",
    "moving": "#d9485f",
    "target": "#6a3d9a",
    "handle": "#b9b9b9",
    "contact": "#f2b701",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--frame", type=int, action="append")
    selection.add_argument("--time", type=float, action="append", dest="times_s")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--actual-only",
        action="store_true",
        help="Omit dashed low-level target colliders.",
    )
    parser.add_argument("--xlim-mm", type=float, nargs=2, default=(-45.0, 65.0))
    parser.add_argument("--ylim-mm", type=float, nargs=2, default=(-50.0, 50.0))
    return parser.parse_args()


def axis_rotation(axis: int, angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == 0:
        return np.asarray(((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)))
    if axis == 1:
        return np.asarray(((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)))
    if axis == 2:
        return np.asarray(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))
    raise ValueError(f"invalid rotation axis: {axis}")


def quaternion_xyzw_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion / max(np.linalg.norm(quaternion), 1.0e-12)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def gripper_pose(
    root_position: np.ndarray,
    root_quaternion_xyzw: np.ndarray,
    arm_joint_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = quaternion_xyzw_matrix(root_quaternion_xyzw)
    from gr00t.rl.utils.b2z1_asset import read_arm_mount

    mount_xyz, mount_yaw = read_arm_mount()
    position = root_position + rotation @ np.asarray(mount_xyz)
    rotation = rotation @ axis_rotation(2, mount_yaw)
    for joint_angle, origin, axis in zip(
        arm_joint_position, ARM_JOINT_ORIGINS, ARM_JOINT_AXES
    ):
        position = position + rotation @ np.asarray(origin)
        rotation = rotation @ axis_rotation(axis, float(joint_angle))
    position = position + rotation @ np.asarray((0.051, 0.0, 0.0))
    rotation = rotation @ axis_rotation(0, -math.pi / 2.0)
    position = position + rotation @ np.asarray((0.049, 0.0, 0.0))
    return position, rotation


def find_parquet(shard: Path, group: str) -> Path:
    matches = sorted(shard.glob(f"**/{group}/chunk-*/episode_*.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {group} parquet below {shard}, found {matches}")
    return matches[0]


def load_hulls(mesh_dir: Path, prefix: str):
    hulls = []
    for path in sorted(mesh_dir.glob(f"{prefix}_*.obj")):
        mesh = trimesh.load_mesh(path, process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if prefix == "fixed":
            vertices = vertices - np.asarray((0.049, 0.0, 0.0))
        hulls.append((path.stem, vertices, np.asarray(mesh.faces, dtype=np.int64)))
    if not hulls:
        raise RuntimeError(f"no {prefix} collision hulls found in {mesh_dir}")
    return hulls


def transform_hulls(hulls, position, rotation, joint_angle=None):
    joint_rotation = np.eye(3) if joint_angle is None else axis_rotation(1, joint_angle)
    return [
        (name, (vertices @ joint_rotation.T) @ rotation.T + position, faces)
        for name, vertices, faces in hulls
    ]


def section_segments(hulls, center, normal, u_axis, v_axis):
    segments = []
    for name, vertices, faces in hulls:
        distances = (vertices - center) @ normal
        for face in faces:
            points, signed = vertices[face], distances[face]
            hits = []
            for edge_a, edge_b in ((0, 1), (1, 2), (2, 0)):
                distance_a, distance_b = signed[edge_a], signed[edge_b]
                point_a, point_b = points[edge_a], points[edge_b]
                if abs(distance_a) <= 1.0e-9:
                    hits.append(point_a)
                if distance_a * distance_b < -1.0e-18:
                    fraction = distance_a / (distance_a - distance_b)
                    hits.append(point_a + fraction * (point_b - point_a))
            unique = []
            for point in hits:
                if not any(np.linalg.norm(point - old) < 1.0e-7 for old in unique):
                    unique.append(point)
            if len(unique) >= 2:
                segments.append(
                    (
                        name,
                        np.asarray(
                            (
                                (unique[0] - center) @ u_axis,
                                (unique[0] - center) @ v_axis,
                                (unique[1] - center) @ u_axis,
                                (unique[1] - center) @ v_axis,
                            )
                        )
                        * 1000.0,
                    )
                )
    return segments


def draw_segments(axis, segments, color, *, alpha, linewidth, linestyle="-"):
    for _, segment in segments:
        axis.plot(
            segment[[0, 2]],
            segment[[1, 3]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
        )


def primitive(prims: list[dict], suffix: str) -> dict:
    return next(row for row in prims if row["path"].endswith(suffix))


def handle_frame(geometry: dict, audit: dict, frame: int):
    hinge_angle = geometry["hinge_sign"] * float(audit["door_hinge_position"][frame])
    panel_rotation = axis_rotation(2, hinge_angle)
    handle_rotation = axis_rotation(
        0, geometry["handle_axis_sign"] * float(audit["handle_position"][frame])
    )
    axle = geometry["hinge_world"] + panel_rotation @ (
        geometry["axle_closed_world"] - geometry["hinge_world"]
    )
    center = axle + panel_rotation @ handle_rotation @ geometry["handle_center_local"]
    handle_axis = panel_rotation @ handle_rotation @ np.asarray((0.0, -1.0, 0.0))
    handle_axis /= np.linalg.norm(handle_axis)
    toward_robot = panel_rotation @ np.asarray((1.0, 0.0, 0.0))
    upward = np.cross(handle_axis, toward_robot)
    if upward[2] < 0.0:
        upward *= -1.0
    upward /= np.linalg.norm(upward)
    return center, handle_axis, toward_robot, upward


def load_episode(repo: Path, dataset: Path, seed: int):
    dataset = dataset.resolve()
    shard = dataset / "shards" / f"episode-seed{seed}"
    with (shard / "meta" / "simulation_provenance.json").open() as stream:
        provenance = json.load(stream)
    episode = provenance["episodes"]["0"]
    columns = [
        "root_position_world",
        "root_quaternion_xyzw",
        "arm_joint_position",
        "arm_joint_target_dispatched",
        "gripper_feedback",
        "gripper_actuator_target",
        "door_hinge_position",
        "handle_position",
        "autoopen_stage",
    ]
    audit = pq.read_table(find_parquet(shard, "raw_audit"), columns=columns).to_pydict()
    contacts = pq.read_table(find_parquet(shard, "raw_contacts")).to_pylist()
    handle_contacts = {}
    for row in contacts:
        paths = row["body_a_path"] + row["body_b_path"]
        if "door_handle" in paths and "Robot" in paths:
            handle_contacts.setdefault(int(row["frame_index"]), []).append(row)

    prims = episode["door_asset"]["prims"]
    handle_joint = primitive(prims, "/handle_joint")["attributes"]
    handle_collider = primitive(prims, "/handle_outside")["attributes"]
    hinge_joint = primitive(prims, "/hinge_joint")["attributes"]
    is_left = "door_left" in episode["door_asset"]["stage_prim_path"]
    initial_root_xy = np.asarray(episode["initial_root_pose_w"][:2])
    environment_origin = np.r_[np.round(initial_root_xy / 8.0) * 8.0, 0.0]
    geometry = {
        "hinge_sign": -1.0 if is_left else 1.0,
        "handle_axis_sign": 1.0 if is_left else -1.0,
        "hinge_world": environment_origin
        + np.asarray((0.02, hinge_joint["physics:localPos0"][1], 0.0)),
        "axle_closed_world": environment_origin
        + np.asarray(handle_joint["physics:localPos0"], dtype=np.float64),
        "handle_center_local": np.asarray(
            handle_collider["xformOp:translate"], dtype=np.float64
        ),
        "handle_radius": float(handle_collider["radius"]),
    }
    mesh_dir = repo / (
        "gr00t/rl/third_party/visual_wholebody_lowlevel/resources/robots/"
        "b2z1/meshes/gripper_collision_hulls"
    )
    return provenance, audit, handle_contacts, geometry, load_hulls(
        mesh_dir, "fixed"
    ), load_hulls(mesh_dir, "moving")


def plot_frame(
    axis,
    frame,
    dt,
    audit,
    contacts_by_frame,
    geometry,
    fixed_hulls,
    moving_hulls,
    *,
    draw_target,
    xlim,
    ylim,
):
    root_position = np.asarray(audit["root_position_world"][frame])
    root_quaternion = np.asarray(audit["root_quaternion_xyzw"][frame])
    actual_position, actual_rotation = gripper_pose(
        root_position,
        root_quaternion,
        np.asarray(audit["arm_joint_position"][frame]),
    )
    center, section_normal, u_axis, v_axis = handle_frame(geometry, audit, frame)
    feedback = float(audit["gripper_feedback"][frame])
    actual_fixed = transform_hulls(fixed_hulls, actual_position, actual_rotation)
    actual_moving = transform_hulls(
        moving_hulls, actual_position, actual_rotation, feedback
    )
    draw_segments(
        axis,
        section_segments(actual_fixed, center, section_normal, u_axis, v_axis),
        COLORS["fixed"],
        alpha=0.85,
        linewidth=1.0,
    )
    draw_segments(
        axis,
        section_segments(actual_moving, center, section_normal, u_axis, v_axis),
        COLORS["moving"],
        alpha=0.90,
        linewidth=1.1,
    )

    actuator = float(audit["gripper_actuator_target"][frame])
    if draw_target:
        target_position, target_rotation = gripper_pose(
            root_position,
            root_quaternion,
            np.asarray(audit["arm_joint_target_dispatched"][frame]),
        )
        target_fixed = transform_hulls(fixed_hulls, target_position, target_rotation)
        target_moving = transform_hulls(
            moving_hulls, target_position, target_rotation, actuator
        )
        draw_segments(
            axis,
            section_segments(target_fixed, center, section_normal, u_axis, v_axis),
            COLORS["target"],
            alpha=0.50,
            linewidth=0.8,
            linestyle="--",
        )
        draw_segments(
            axis,
            section_segments(target_moving, center, section_normal, u_axis, v_axis),
            COLORS["target"],
            alpha=0.75,
            linewidth=1.0,
            linestyle="--",
        )

    axis.add_patch(
        plt.Circle(
            (0.0, 0.0),
            geometry["handle_radius"] * 1000.0,
            facecolor=COLORS["handle"],
            edgecolor="#222222",
            linewidth=1.8,
            alpha=0.72,
            zorder=5,
        )
    )
    frame_contacts = contacts_by_frame.get(frame, [])
    for contact in frame_contacts:
        point = np.asarray(contact["contact_point_world"])
        normal = np.asarray(contact["contact_normal_world"])
        point_2d = np.asarray(
            ((point - center) @ u_axis, (point - center) @ v_axis)
        ) * 1000.0
        normal_2d = np.asarray((normal @ u_axis, normal @ v_axis))
        normal_2d /= max(np.linalg.norm(normal_2d), 1.0e-12)
        force = float(contact["max_contact_force"])
        arrow_length = 7.0 + min(force, 150.0) / 150.0 * 8.0
        axis.scatter(
            [point_2d[0]],
            [point_2d[1]],
            marker="o",
            s=24,
            color=COLORS["contact"],
            edgecolor="#111111",
            zorder=9,
        )
        axis.arrow(
            point_2d[0],
            point_2d[1],
            normal_2d[0] * arrow_length,
            normal_2d[1] * arrow_length,
            width=0.35,
            head_width=2.0,
            head_length=2.5,
            color="#111111",
            length_includes_head=True,
            zorder=9,
        )
        axis.text(
            point_2d[0] + normal_2d[0] * arrow_length,
            point_2d[1] + normal_2d[1] * arrow_length,
            f"{force:.0f} N\n{contact['max_penetration_depth'] * 1000.0:.1f} mm",
            fontsize=7,
        )

    max_penetration = max(
        (row["max_penetration_depth"] for row in frame_contacts), default=0.0
    )
    axis.set_title(
        f"frame {frame} ({frame * dt:.2f} s), {audit['autoopen_stage'][frame]}\n"
        f"q actual={feedback:.4f}, actuator={actuator:.4f} rad, "
        f"max penetration={max_penetration * 1000.0:.2f} mm",
        fontsize=10,
    )
    axis.axhline(0.0, color="#bbbbbb", linewidth=0.5)
    axis.axvline(0.0, color="#bbbbbb", linewidth=0.5)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.grid(alpha=0.15)
    axis.set_xlabel("toward door <- handle center -> toward robot (mm)")
    axis.set_ylabel("section upward (mm)")


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    dataset = args.dataset if args.dataset.is_absolute() else repo / args.dataset
    provenance, audit, contacts, geometry, fixed_hulls, moving_hulls = load_episode(
        repo, dataset, args.seed
    )
    dt = float(provenance["control_dt_s"])
    frames = args.frame or [int(round(value / dt)) for value in args.times_s]
    frames = list(dict.fromkeys(frames))
    frame_count = len(audit["gripper_feedback"])
    invalid = [frame for frame in frames if not 0 <= frame < frame_count]
    if invalid:
        raise ValueError(f"frames outside [0, {frame_count - 1}]: {invalid}")

    columns = min(3, len(frames))
    rows = math.ceil(len(frames) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.5 * columns, 6.0 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, frame in zip(axes.flat, frames):
        plot_frame(
            axis,
            frame,
            dt,
            audit,
            contacts,
            geometry,
            fixed_hulls,
            moving_hulls,
            draw_target=not args.actual_only,
            xlim=args.xlim_mm,
            ylim=args.ylim_mm,
        )
    for axis in axes.flat[len(frames) :]:
        axis.axis("off")
    figure.suptitle(
        f"Seed {args.seed}: true collider section through the outside handle\n"
        "solid blue/red: actual fixed/moving jaw | purple dashed: low-level target\n"
        "black arrows: PhysX force on handle",
        fontsize=13,
    )
    frame_label = "-".join(str(frame) for frame in frames)
    output = args.output or (
        dataset
        / "diagnostics"
        / "collision_sections"
        / f"seed{args.seed}_frames{frame_label}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output.resolve())
    print(f"rendered {len(frames)} frame(s) in {time.perf_counter() - started:.2f} s")


if __name__ == "__main__":
    main()
