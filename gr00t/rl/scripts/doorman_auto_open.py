"""Rule-based DoorMan door opening and raw trajectory collection.

This entry point drives the existing B2Z1 low-level policy through the public
teleop target API.  It intentionally does not synthesize keyboard events or
feed the 10-D high-level action directly to Isaac Lab.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import deque
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch

faulthandler.enable()
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
LOWLEVEL_ROOT = SCRIPT_ROOT / "third_party" / "visual_wholebody_lowlevel"
sys.path.insert(0, str(LOWLEVEL_ROOT))

from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy  # noqa: E402
from legged_gym.utils.ee_steady_state import B2Z1SteadyEeForwardModel  # noqa: E402
from legged_gym.utils.helpers import build_common_arg_parser  # noqa: E402
from legged_gym.utils.isaaclab_app import add_app_launcher_args, launch_app  # noqa: E402
from legged_gym.utils.isaaclab_math import quat_rotate_inverse  # noqa: E402


class Stage(IntEnum):
    RESET = 0
    APPROACH = 1
    ALIGN = 2
    PREGRASP = 3
    CENTER_HANDLE = 4
    GRASP_HANDLE = 5
    TURN_HANDLE = 6
    RELEASE_HANDLE = 7
    RETRACT_ARM = 8
    ALIGN_PANEL = 9
    CONTACT_PANEL = 10
    PUSH_OPEN = 11
    THROUGH = 12
    RESET_ARM = 13
    SUCCESS = 14
    FAILURE = 15


STAGE_NAMES = [
    "reset",
    "approach",
    "align",
    "pregrasp",
    "center_handle",
    "grasp_handle",
    "turn_handle",
    "release_handle",
    "retract_arm",
    "align_panel",
    "contact_panel",
    "push_open",
    "through",
    "reset_arm",
    "success",
    "failure",
]
TASK_PHASE_NAMES = ["position_at_door", "turn_handle", "push_and_traverse"]
STAGE_TO_TASK_PHASE = {
    Stage.APPROACH: 0,
    Stage.ALIGN: 0,
    Stage.PREGRASP: 0,
    Stage.CENTER_HANDLE: 1,
    Stage.GRASP_HANDLE: 1,
    Stage.TURN_HANDLE: 1,
    Stage.RELEASE_HANDLE: 1,
    Stage.RETRACT_ARM: 1,
    Stage.ALIGN_PANEL: 2,
    Stage.CONTACT_PANEL: 2,
    Stage.PUSH_OPEN: 2,
    Stage.THROUGH: 2,
    Stage.RESET_ARM: 2,
}


def _parser() -> argparse.ArgumentParser:
    parser = build_common_arg_parser()
    add_app_launcher_args(parser)
    parser.set_defaults(
        num_envs=1,
        enable_doorman_scene=True,
        doorman_randomize_robot_init=True,
        doorman_arm_joint_init_center=(-1.57, 1.6, -0.63, -1.1, 0.0, 0.0),
        doorman_gripper_joint_init=0.0,
        doorman_randomize_door_init_state=False,
        action_delay_mode="undelayed",
        teleop_mode=True,
        mount_deg=90,
    )
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--save_failures", action="store_true")
    parser.add_argument(
        "--debug_stage",
        choices=("retract_arm",),
        default=None,
        help="Stop after the requested deterministic handle diagnostic.",
    )
    parser.add_argument(
        "--robot_side",
        choices=("positive_x", "negative_x"),
        default="positive_x",
        help="Side of the door where the robot starts; positive_x is the default collection setup.",
    )
    parser.add_argument("--position_only_center", action="store_true", help="Keep EE orientation fixed during CENTER_HANDLE.")
    return parser


def _default_paths(args):
    repo = SCRIPT_ROOT.parent.parent
    run_dir = repo / "ckpt" / "b2z1-low" / "20260918_102500_is-dc7zasdnsm4oadn4-devmachine-0_gpu2_agent1_uhryvo2cn"
    if not args.ckpt_path:
        args.ckpt_path = str(run_dir / "model_54000.pt")
    if not args.run_metadata_path:
        args.run_metadata_path = str(run_dir / "run_metadata.json")
    if not args.robot_usd_path:
        args.robot_usd_path = str(repo / "gr00t" / "rl" / "data" / "robots" / "b2z1_lab30" / "b2z1_mount_90_x0p22_y0_z0p17.usd")
    if not args.rgb_camera_config:
        args.rgb_camera_config = str(LOWLEVEL_ROOT / "legged_gym" / "scripts" / "rgb_cameras.json")


def _single_episode_argv(argv, seed):
    """Build one child invocation without reusing an Isaac Sim process."""
    result = []
    skip_value = False
    for index, value in enumerate(argv):
        if skip_value:
            skip_value = False
            continue
        if value in ("--num_episodes", "--seed"):
            skip_value = True
            continue
        if value.startswith("--num_episodes=") or value.startswith("--seed="):
            continue
        result.append(value)
    result.extend(("--num_episodes", "1", "--seed", str(seed)))
    return result


def _run_episode_children(args):
    """Run each randomized episode in its own SimulationApp process."""
    child_env = os.environ.copy()
    child_env["DOORMAN_SINGLE_EPISODE_CHILD"] = "1"
    successes = 0
    for episode in range(args.num_episodes):
        seed = int(args.seed if args.seed is not None else 0) + episode
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *_single_episode_argv(sys.argv[1:], seed),
        ]
        print(
            f"[doorman][batch] launching episode {episode + 1}/{args.num_episodes} "
            f"seed={seed}",
            flush=True,
        )
        before = set(args.output_dir.glob("episode_*/metadata.json"))
        completed = subprocess.run(command, env=child_env, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"episode child seed={seed} exited with status {completed.returncode}"
            )
        created = sorted(set(args.output_dir.glob("episode_*/metadata.json")) - before)
        if len(created) != 1:
            raise RuntimeError(
                f"episode child seed={seed} created {len(created)} metadata files"
            )
        metadata = json.loads(created[0].read_text(encoding="utf-8"))
        successes += int(bool(metadata.get("success")))
    print(
        f"[doorman][batch] complete: {successes}/{args.num_episodes} successful episodes",
        flush=True,
    )
    if successes != args.num_episodes:
        raise RuntimeError(
            f"batch validation failed: {successes}/{args.num_episodes} episodes succeeded"
        )
    return successes


def _as_np(value, shape_last=None):
    if value is None:
        if shape_last is None:
            return None
        return np.zeros(shape_last, dtype=np.float32)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if shape_last is not None and value.ndim == len(shape_last) + 1:
        value = value[0]
    return value.astype(np.float32, copy=True)


def _quat_yaw(quat):
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_rotate_np(quat, vector):
    """Rotate a 3-D vector by a wxyz quaternion."""
    quat = np.asarray(quat, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    q_vec = quat[1:]
    return vector + 2.0 * (
        quat[0] * np.cross(q_vec, vector) + np.cross(q_vec, np.cross(q_vec, vector))
    )


def _quat_rotate_inverse_np(quat, vector):
    conjugate = np.asarray(quat, dtype=np.float32).copy()
    conjugate[1:] *= -1.0
    return _quat_rotate_np(conjugate, vector)


def _quat_rotate_points_np(quat, points):
    quat = np.asarray(quat, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    q_vec = quat[1:]
    q = np.broadcast_to(q_vec, points.shape)
    return points + 2.0 * (quat[0] * np.cross(q, points) + np.cross(q, np.cross(q, points)))


def _quat_mul_np(left, right):
    lw, lx, ly, lz = np.asarray(left, dtype=np.float32)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float32)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )


def _quat_conjugate_np(quat):
    output = np.asarray(quat, dtype=np.float32).copy()
    output[1:] *= -1.0
    return output


def _quat_axis_angle_np(axis, angle):
    """Return a wxyz quaternion rotating by ``angle`` about ``axis``."""
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-8)
    half = 0.5 * float(angle)
    return np.array(
        [math.cos(half), axis[0] * math.sin(half), axis[1] * math.sin(half), axis[2] * math.sin(half)],
        dtype=np.float32,
    )


def _quat_align_vectors_np(source, target, base_quat):
    """Apply the shortest source->target rotation to ``base_quat``."""
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    source /= max(float(np.linalg.norm(source)), 1.0e-8)
    target /= max(float(np.linalg.norm(target)), 1.0e-8)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot < -0.9999:
        axis = np.cross(source, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(axis) < 1.0e-5:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis /= np.linalg.norm(axis)
        delta = np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float32)
    else:
        delta = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float32)
        delta /= max(float(np.linalg.norm(delta)), 1.0e-8)
    return _quat_mul_np(delta, base_quat)


def _quat_to_euler_xyz_np(quat):
    """Return XYZ Euler angles for a normalized wxyz quaternion."""
    w, x, y, z = np.asarray(quat, dtype=np.float32)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = math.asin(float(sin_pitch))
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _matrix_to_quat_np(matrix):
    m = np.asarray(matrix, dtype=np.float32)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    quat = quat.astype(np.float32)
    return quat / max(float(np.linalg.norm(quat)), 1.0e-8)


def _rgb_array(frame):
    if frame is None:
        return None
    array = frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * (255.0 if array.max(initial=0.0) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _door_gap_angle(hinge_to_free_edge, panel_thickness, clearance):
    """Return the opening angle that leaves ``clearance`` at the free jamb.

    In the horizontal section, the free door corner retreats by
    ``L * (1 - cos(theta))`` while the rotated panel thickness still protrudes
    by ``t * sin(theta)``.  Solve their difference for the requested gap.
    """
    length = float(hinge_to_free_edge)
    thickness = float(panel_thickness)
    requested_gap = float(clearance)
    if length <= 0.0 or thickness <= 0.0 or requested_gap < 0.0:
        raise ValueError(
            "door gap geometry requires positive length/thickness and "
            "non-negative clearance"
        )

    lower = 2.0 * math.atan2(thickness, length)
    upper = 0.5 * math.pi
    if length * (1.0 - math.cos(upper)) - thickness * math.sin(upper) < requested_gap:
        raise ValueError("requested door gap is unreachable before 90 degrees")
    for _ in range(64):
        angle = 0.5 * (lower + upper)
        gap = length * (1.0 - math.cos(angle)) - thickness * math.sin(angle)
        if gap < requested_gap:
            lower = angle
        else:
            upper = angle
    return 0.5 * (lower + upper)


def _save_rgb(frame, path: Path):
    array = _rgb_array(frame)
    if array is None:
        return False
    from PIL import Image

    Image.fromarray(array, mode="RGB").save(path)
    return True


def _save_rgb_video(frames, timestamps, path: Path):
    if not frames:
        return False
    import imageio.v2 as imageio

    if len(timestamps) >= 2:
        intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
        intervals = intervals[intervals > 0.0]
        fps = 1.0 / float(np.median(intervals)) if len(intervals) else 4.0
    else:
        fps = 4.0
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    try:
        for frame in frames:
            writer.append_data(_rgb_array(frame))
    finally:
        writer.close()
    return True


def _save_rgb_mosaic_video(camera_frames, camera_timestamps, path: Path):
    preferred_order = (
        "wrist",
        "trunk_front",
        "robot_top_diagnostic",
        "door_world_diagnostic",
        "door_panel_diagnostic",
        "door_panel_top_diagnostic",
    )
    camera_names = [
        name for name in preferred_order if camera_frames.get(name)
    ]
    camera_names.extend(
        sorted(
            name
            for name, frames in camera_frames.items()
            if frames and name not in camera_names
        )
    )
    if not camera_names:
        return False

    reference_name = max(camera_names, key=lambda name: len(camera_frames[name]))
    timeline = np.asarray(camera_timestamps[reference_name], dtype=np.float64)
    if timeline.size == 0:
        return False
    if timeline.size >= 2:
        intervals = np.diff(timeline)
        intervals = intervals[intervals > 0.0]
        fps = 1.0 / float(np.median(intervals)) if len(intervals) else 4.0
    else:
        fps = 4.0

    from PIL import Image, ImageDraw
    import imageio.v2 as imageio

    tile_width, tile_height = 480, 360
    canvas_width, canvas_height = 3 * tile_width, 2 * tile_height
    timestamp_arrays = {
        name: np.asarray(camera_timestamps[name], dtype=np.float64)
        for name in camera_names
    }

    def nearest_frame(name, timestamp):
        timestamps = timestamp_arrays[name]
        index = int(np.searchsorted(timestamps, timestamp, side="left"))
        if index >= len(timestamps):
            index = len(timestamps) - 1
        elif index > 0 and timestamp - timestamps[index - 1] <= timestamps[index] - timestamp:
            index -= 1
        return camera_frames[name][index]

    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    try:
        for timestamp in timeline:
            canvas = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))
            for tile_index, camera_name in enumerate(camera_names[:6]):
                frame = _rgb_array(nearest_frame(camera_name, timestamp))
                tile = Image.fromarray(frame, mode="RGB")
                if tile.size != (tile_width, tile_height):
                    tile = tile.resize(
                        (tile_width, tile_height), Image.Resampling.BILINEAR
                    )
                draw = ImageDraw.Draw(tile)
                label_box = draw.textbbox((0, 0), camera_name)
                label_width = label_box[2] - label_box[0]
                label_height = label_box[3] - label_box[1]
                draw.rectangle(
                    (8, 8, 20 + label_width, 18 + label_height), fill=(0, 0, 0)
                )
                draw.text((14, 12), camera_name, fill=(255, 255, 255))
                x = (tile_index % 3) * tile_width
                y = (tile_index // 3) * tile_height
                canvas.paste(tile, (x, y))
            writer.append_data(np.asarray(canvas, dtype=np.uint8))
    finally:
        writer.close()
    return True


class ColliderGeometry:
    """Small point-cloud representation of the loaded USD collision geometry.

    The point cloud is deliberately sampled from the collision prims, not from
    visual meshes or rigid-body origins.  It is used for deterministic target
    placement and conservative clearance checks; PhysX remains responsible for
    the final contact response.
    """

    def __init__(self, name, body_name, points_body, source_paths):
        self.name = name
        self.body_name = body_name
        self.points_body = np.asarray(points_body, dtype=np.float32)
        self.source_paths = list(source_paths)
        if self.points_body.ndim != 2 or self.points_body.shape[1] != 3 or not len(self.points_body):
            raise RuntimeError(f"empty collision geometry for {name}: {source_paths}")

    def world_points(self, body_pose):
        pos = np.asarray(body_pose[:3], dtype=np.float32)
        quat = np.asarray(body_pose[3:7], dtype=np.float32)
        return _quat_rotate_points_np(quat, self.points_body) + pos

    def local_centroid(self):
        return self.points_body.mean(axis=0)

    def bounds(self):
        return self.points_body.min(axis=0), self.points_body.max(axis=0)


def _stage_robot_root(stage):
    candidates = []
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if name in {"robot", "b2z1", "b2_description"}:
            path = str(prim.GetPath())
            if "/envs/" in path:
                candidates.append(prim)
    return candidates[0] if candidates else None


def _read_collider_points(stage, body_prim, tokens):
    """Read sampled points from all collision descendants matching tokens."""
    from pxr import Gf, Usd, UsdGeom

    body_xf = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    inv_body_xf = body_xf.GetInverse()
    points = []
    paths = []
    for prim in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies()):
        path_text = str(prim.GetPath())
        low = path_text.lower()
        if "/collisions/" not in low:
            continue
        if not any(token in low for token in tokens):
            continue
        local_points = None
        if prim.IsA(UsdGeom.Mesh):
            raw = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
            if raw:
                local_points = [Gf.Vec3d(p) for p in raw]
        elif prim.IsA(UsdGeom.Cube):
            size = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 1.0) * 0.5
            local_points = [Gf.Vec3d(x, y, z) for x in (-size, size) for y in (-size, size) for z in (-size, size)]
        elif prim.IsA(UsdGeom.Capsule):
            capsule = UsdGeom.Capsule(prim)
            radius = float(capsule.GetRadiusAttr().Get() or 0.0)
            height = float(capsule.GetHeightAttr().Get() or 0.0) * 0.5
            # Conservative extrema for a Z-axis capsule.  The generated
            # robot asset uses capsules only as small wrist colliders.
            local_points = [Gf.Vec3d(x, y, z) for x in (-radius, radius) for y in (-radius, radius) for z in (-height, height)]
        if local_points:
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            transformed = [inv_body_xf.Transform(xf.Transform(p)) for p in local_points]
            # Large jaw meshes are reduced to a deterministic hull sample.
            stride = max(1, len(transformed) // 1024)
            points.extend(np.asarray(transformed[::stride], dtype=np.float32))
            paths.append(path_text)
    if not points:
        return None
    return ColliderGeometry("", body_prim.GetName(), np.asarray(points), paths)


class DoorController:
    def __init__(self, runner: ManipLoco_Policy, max_steps: int, robot_side: str = "positive_x", position_only_center: bool = False):
        self.runner = runner
        self.env = runner.env
        self.max_steps = max_steps
        self.robot_side = robot_side
        self.position_only_center = position_only_center
        self.stage = Stage.RESET
        self.stage_steps = 0
        self.stage_limits = {
            Stage.APPROACH: 700,
            Stage.ALIGN: 250,
            Stage.PREGRASP: 500,
            Stage.CENTER_HANDLE: 800,
            Stage.GRASP_HANDLE: 400,
            Stage.TURN_HANDLE: 900,
            Stage.RELEASE_HANDLE: 450,
            Stage.RETRACT_ARM: 300,
            Stage.ALIGN_PANEL: 400,
            Stage.CONTACT_PANEL: 400,
            Stage.PUSH_OPEN: 900,
            Stage.THROUGH: 700,
            Stage.RESET_ARM: 400,
        }
        self.hinge_idx = None
        self.handle_joint_idx = None
        self.latch_joint_idx = None
        self.handle_body_idx = None
        self.grasp_target_body_idx = None
        self.panel_body_idx = None
        self._introspect()
        self.pregrasp_target_world = None
        self.handle_center_target_world = None
        self.align_settle_steps = 0
        self.pregrasp_settle_steps = 0
        self.handle_align_settle_steps = 0
        self.grasp_settle_steps = 0
        self.grasp_contact_threshold = 1.0
        # The inverse steady-state model removes the systematic load error,
        # but the measured PhysX equilibrium retains about 5.6 mm for B2Z1.
        self.pregrasp_position_tolerance = 0.007
        self.handle_contact_last_seen = np.full(2, -1000, dtype=np.int32)
        self.handle_contact_last_points = np.full((2, 3), np.nan, dtype=np.float32)
        self.grasp_bilateral_contact_seen = False
        self.grasp_waypoint_world = None
        self.center_insert_waypoint_world = None
        self.turn_start_handle_angle = 0.0
        self.turn_start_latch_position = 0.0
        self.turn_start_hinge_angle = 0.0
        self.turn_last_handle_angle = 0.0
        self.turn_last_hinge_angle = 0.0
        self.turn_handle_axis_world = None
        self.turn_axle_world = None
        self.turn_grasp_offset_world = None
        self.turn_grasp_orientation = None
        self.turn_target_handle_angle = None
        self.turn_target_hinge_angle = None
        self.turn_endpoint_world = None
        self.turn_endpoint_quat = None
        self.turn_path_world = None
        self.turn_path_quat = None
        self.turn_path_index = 0
        self.turn_path_stage = None
        self.turn_path_replan_count = 0
        self.turn_gap_started = False
        self.turn_command_bias_world = np.zeros(3, dtype=np.float32)
        self.turn_base_start_xy = None
        self.release_start_world = None
        self.release_substage = "RETURN_HANDLE"
        self.release_start_quat = None
        self.release_endpoint_world = None
        self.release_endpoint_quat = None
        self.release_start_handle_angle = 0.0
        self.release_start_hinge_angle = 0.0
        self.release_return_axle_world = None
        self.release_return_axis_world = None
        self.release_return_offset_world = None
        self.release_return_start_quat = None
        self.release_return_steps = 60
        self.release_vertical_delta = 0.0
        self.release_stable_steps = 0
        self.release_open_start_world = None
        self.release_open_start_step = 0
        self.release_open_command_path = None
        self.release_open_orientation_path = None
        self.release_retreat_start_world = None
        self.release_retreat_start_step = 0
        self.release_retreat_target_world = None
        self.release_retreat_command_path = None
        self.release_retreat_orientation_path = None
        self.release_retreat_path_index = 0
        self.release_restore_command_path = None
        self.release_restore_orientation_path = None
        self.push_target_hinge_angle = None
        self.panel_contact_local = None
        self.panel_center_world = None
        self.panel_push_world = None
        self.panel_tcp_world = None
        self.panel_precontact_clearance = 0.018
        self.panel_contact_clearance = 0.0
        self.panel_contact_established = False
        self.panel_recovery_steps = 0
        self.panel_contact_target_world = None
        self.panel_contact_orientation = None
        self.panel_alignment_established = False
        self.panel_contact_settle_steps = 0
        self.panel_station_xy = None
        self.panel_station_settle_steps = 0
        self.panel_align_ee_goal_local = None
        self.panel_align_ee_orientation_local = None
        self.push_start_hinge_angle = None
        self.push_last_progress_hinge = None
        self.push_no_progress_steps = 0
        self.push_contact_loss_steps = 0
        self.push_pause_steps = 0
        self.push_substage = "ARM_TRACK"
        self.body_align_target_xy = None
        self.body_align_settle_steps = 0
        self.push_contact_world = None
        self.push_normal_world = None
        self.push_orientation = None
        self.push_peak_hinge = 0.0
        self.push_coast_steps = 0
        self.push_depth = 0.0
        self.push_goal_hinge_angle = math.radians(90.0)
        self.push_target_angular_speed = math.radians(7.0)
        self.push_max_angular_speed = math.radians(12.0)
        self.push_command_hinge_angle = None
        self.push_max_observed_angular_speed = 0.0
        self.push_contact_reference_clearance = None
        self.push_finger_clearance = float("nan")
        self.push_finger_approach_speed = float("nan")
        self.push_target_finger_clearance = float("nan")
        self.push_finger_force = 0.0
        self.push_contact_retreat_active = False
        self.push_previous_finger_clearance = None
        self.push_recontact_step = 0.0003
        self.push_nominal_finger_clearance = 0.003
        self.push_speed_settle_steps = 0
        self.push_overspeed_steps = 0
        self.push_release_start_world = None
        self.push_release_normal_world = None
        self.push_release_start_step = 0
        self.push_base_start_xy = None
        self.push_panel_contact_start_world = None
        # A 0.155 m planar envelope around each front hip covers the B2 base
        # box, front camera enclosure, and the laterally offset upper-leg
        # collider in the deployment URDF.  Door thickness is added at run
        # time, so this margin remains a true free-space allowance.
        self.push_shoulder_collision_radius = 0.155
        self.push_shoulder_safety_margin = 0.020
        self.push_shoulder_slowdown_band = 0.080
        self.push_shoulder_min_clearance = float("nan")
        self.push_shoulder_predicted_min_clearance = float("nan")
        self.push_base_safety_scale = 1.0
        self.through_start_base_xy = None
        self.through_best_distance = 0.0
        self.through_no_progress_steps = 0
        self.hinge_world = None
        self.hinge_axis_sign = None
        self.final_min_normal_offset = None
        self.open_loop_trajectory = []
        self.trajectory_step = 0
        self.handle_lower_z = None
        self.handle_upper_z = None
        self.handle_collider_center_local = None
        self.handle_axis_local = None
        self.handle_joint_axis_panel_local = None
        self.handle_half_length = None
        self.handle_collider_path = None
        self.panel_half_thickness = 0.03
        self.panel_hinge_to_free_edge = None
        self.turn_gap_clearance = 0.002
        self.turn_gap_angle = None
        self.grasp_orientation_tolerance = 0.050
        self.ee_orientation_error = 999.0
        self.default_ee_goal_local = None
        self.default_ee_goal_delta_rpy = None
        self.stow_ee_goal_local = None
        self.stow_ee_base_local = None
        # Cartesian targets are rate-limited in world coordinates.  The
        # low-level IK consumes incremental targets, so a large target jump at
        # a stage transition can overshoot the fingers into the door.
        self.ee_command_world = None
        # Rolling-horizon replanning period.  Stage endpoints remain fixed;
        # only the measured path start is refreshed at these boundaries.
        self.replan_interval = 5
        self.replan_counter = 0
        self.finger_separation_local = None
        self.finger_closing_local = None
        self.grasp_orientation_quat = None
        self.grasp_branch = None
        self.ee_orientation_command = None
        self.active_grasp_quat = None
        self.grasp_close_start = -0.30
        self.q_open = None
        self.q_closed = None
        self.q_grasp = None
        self.grasp_latched = False
        self.grasp_close_command = None
        self.grasp_vertical_lift = 0.0
        self.grasp_vertical_lift_target = 0.0
        self.grasp_vertical_balance_started = False
        self.grasp_preload_angle = 0.0
        self.gripper_open_settle = 0
        self.pregrasp_opening = False
        self.center_substage = "CENTER_FORWARD"
        self.center_orientation_settle_steps = 0
        self.pregrasp_clearance_fallback = False
        self.center_clearance_fallback = False
        self.center_vertical_bias = 0.02
        self.center_depth_bias = -0.025
        self.center_standoff_world = None
        self.center_standoff_ee_world = None
        self.orientation_standoff_ee_world = None
        self.center_standoff_nominal_world = None
        self.center_insert_nominal_world = None
        self.grasp_nominal_world = None
        self.center_tangent_feedback_world = np.zeros(3, dtype=np.float32)
        self.center_normal_feedback = 0.0
        self.initial_ee_orientation = None
        self.initial_wrist_roll = None
        self.target_tip_mid_world = None
        self.center_geometry = None
        self.rotation_sweep_radius = None
        self.geometry_report = None
        self.center_metrics = {}
        steady_urdf = (
            LOWLEVEL_ROOT
            / "resources/robots/b2z1/urdf/generated"
            / "b2z1_mount_90_x0p22_y0_z0p17.urdf"
        )
        self.ee_steady_model = B2Z1SteadyEeForwardModel(steady_urdf)
        self.arm_align_command_world = None
        self.arm_align_command_quat = None
        self.arm_align_nominal_command_world = None
        self.arm_command_bias_world = np.zeros(3, dtype=np.float32)
        self.arm_align_command_path = None
        self.arm_align_orientation_path = None
        self.arm_align_path_index = 0
        self.arm_forward_target_world = None
        self.arm_forward_command_world = None
        self.arm_forward_command_quat = None
        self.arm_forward_command_path = None
        self.arm_forward_orientation_path = None
        self.arm_forward_path_index = 0
        self.arm_align_geometry_corrections = 0
        self.arm_align_residual_corrections = 0
        self.arm_stability_window = deque(maxlen=15)
        self.failure_reason = None
        self.replan_counter = 0
        robot_body_names = [name.lower() for name in self.env.body_names]
        self.front_shoulder_body_indices = np.asarray(
            [robot_body_names.index("fl_hip"), robot_body_names.index("fr_hip")],
            dtype=np.int64,
        )
        # The fixed jaw mesh is welded into link06 in the imported USD.  The
        # separate gripper_link body is only the tiny EE reference frame.
        self.fixed_finger_body_idx = robot_body_names.index("link06")
        self.moving_finger_body_idx = robot_body_names.index("grippermover")

    def _introspect(self):
        if self.env.door is None:
            raise RuntimeError("DoorMan scene is disabled; pass --enable_doorman_scene")
        data = self.env.door.data
        self.door_joint_names = list(getattr(data, "joint_names", []))
        self.door_body_names = list(getattr(data, "body_names", []))
        hinge = [i for i, n in enumerate(self.door_joint_names) if "hinge" in n.lower()]
        handle = [i for i, n in enumerate(self.door_joint_names) if "handle" in n.lower()]
        latch = [i for i, n in enumerate(self.door_joint_names) if "latch" in n.lower()]
        body_name_to_idx = {name.lower(): i for i, name in enumerate(self.door_body_names)}
        handle_body_idx = body_name_to_idx.get("door_handle", body_name_to_idx.get("handle"))
        grasp_target_body_idx = body_name_to_idx.get("grasp_target")
        panel_body_idx = body_name_to_idx.get("door_panel")
        if not hinge or not handle or not latch or handle_body_idx is None or panel_body_idx is None:
            raise RuntimeError(
                "Unable to resolve DoorMan articulation names. "
                f"joints={self.door_joint_names}, bodies={self.door_body_names}"
            )
        self.hinge_idx = hinge[0]
        self.handle_joint_idx = handle[0]
        self.latch_joint_idx = latch[0]
        self.handle_body_idx = handle_body_idx
        self.grasp_target_body_idx = grasp_target_body_idx
        self.panel_body_idx = panel_body_idx
        print(f"[doorman] door joints={self.door_joint_names}; bodies={self.door_body_names}")
        print(
            f"[doorman] hinge={self.hinge_idx}, handle_joint={self.handle_joint_idx}, "
            f"latch_joint={self.latch_joint_idx}, "
            f"physical_handle_body={self.handle_body_idx}, grasp_reference_body={self.grasp_target_body_idx}, "
            f"panel_body={self.panel_body_idx}"
        )

    def reset(self):
        self.env.reset()
        self.runner.obs = self.env.get_observations()
        self.stage = Stage.APPROACH
        self.stage_steps = 0
        self.align_settle_steps = 0
        self.pregrasp_settle_steps = 0
        self.handle_align_settle_steps = 0
        self.grasp_settle_steps = 0
        self.handle_contact_last_seen[:] = -1000
        self.handle_contact_last_points[:] = np.nan
        self.grasp_bilateral_contact_seen = False
        self.grasp_waypoint_world = None
        self.center_insert_waypoint_world = None
        self.grasp_branch = None
        self.turn_start_handle_angle = 0.0
        self.turn_start_latch_position = 0.0
        self.turn_start_hinge_angle = 0.0
        self.turn_last_handle_angle = 0.0
        self.turn_last_hinge_angle = 0.0
        self.turn_handle_axis_world = None
        self.turn_axle_world = None
        self.turn_grasp_offset_world = None
        self.turn_grasp_orientation = None
        self.turn_target_handle_angle = None
        self.turn_target_hinge_angle = None
        self.turn_endpoint_world = None
        self.turn_endpoint_quat = None
        self.turn_path_world = None
        self.turn_path_quat = None
        self.turn_path_index = 0
        self.turn_path_stage = None
        self.turn_path_replan_count = 0
        self.turn_gap_started = False
        self.turn_command_bias_world[:] = 0.0
        self.turn_base_start_xy = None
        self.release_start_world = None
        self.release_substage = "RETURN_HANDLE"
        self.release_start_quat = None
        self.release_endpoint_world = None
        self.release_endpoint_quat = None
        self.release_start_handle_angle = 0.0
        self.release_start_hinge_angle = 0.0
        self.release_return_axle_world = None
        self.release_return_axis_world = None
        self.release_return_offset_world = None
        self.release_return_start_quat = None
        self.release_vertical_delta = 0.0
        self.release_stable_steps = 0
        self.release_open_start_world = None
        self.release_open_start_step = 0
        self.release_open_command_path = None
        self.release_open_orientation_path = None
        self.release_retreat_start_world = None
        self.release_retreat_start_step = 0
        self.release_retreat_target_world = None
        self.release_retreat_command_path = None
        self.release_retreat_orientation_path = None
        self.release_retreat_path_index = 0
        self.release_restore_command_path = None
        self.release_restore_orientation_path = None
        self.panel_contact_established = False
        self.panel_recovery_steps = 0
        self.panel_contact_target_world = None
        self.panel_contact_orientation = None
        self.panel_alignment_established = False
        self.panel_contact_settle_steps = 0
        self.panel_station_settle_steps = 0
        self.panel_align_ee_goal_local = None
        self.panel_align_ee_orientation_local = None
        self.push_start_hinge_angle = None
        self.push_last_progress_hinge = None
        self.push_no_progress_steps = 0
        self.push_contact_loss_steps = 0
        self.push_pause_steps = 0
        self.push_substage = "ARM_TRACK"
        self.body_align_target_xy = None
        self.body_align_settle_steps = 0
        self.push_contact_world = None
        self.push_normal_world = None
        self.push_orientation = None
        self.push_peak_hinge = 0.0
        self.push_coast_steps = 0
        self.push_depth = 0.0
        self.push_command_hinge_angle = None
        self.push_max_observed_angular_speed = 0.0
        self.push_contact_reference_clearance = None
        self.push_finger_clearance = float("nan")
        self.push_finger_approach_speed = float("nan")
        self.push_target_finger_clearance = float("nan")
        self.push_finger_force = 0.0
        self.push_contact_retreat_active = False
        self.push_previous_finger_clearance = None
        self.push_speed_settle_steps = 0
        self.push_overspeed_steps = 0
        self.push_release_start_world = None
        self.push_release_normal_world = None
        self.push_release_start_step = 0
        self.push_base_start_xy = None
        self.push_panel_contact_start_world = None
        self.push_shoulder_min_clearance = float("nan")
        self.push_shoulder_predicted_min_clearance = float("nan")
        self.push_base_safety_scale = 1.0
        self.through_start_base_xy = None
        self.through_best_distance = 0.0
        self.through_no_progress_steps = 0
        self.push_target_hinge_angle = None
        self.failure_reason = None
        self.center_substage = "CENTER_FORWARD"
        self.center_orientation_settle_steps = 0
        self.pregrasp_clearance_fallback = False
        self.center_clearance_fallback = False
        self.center_vertical_bias = 0.02
        self.center_depth_bias = -0.025
        self.gripper_open_settle = 0
        self.pregrasp_opening = False
        self.center_tangent_feedback_world[:] = 0.0
        self.center_normal_feedback = 0.0
        self.arm_align_command_world = None
        self.arm_align_command_quat = None
        self.arm_align_nominal_command_world = None
        self.arm_command_bias_world[:] = 0.0
        self.arm_align_command_path = None
        self.arm_align_orientation_path = None
        self.arm_align_path_index = 0
        self.arm_forward_target_world = None
        self.arm_forward_command_world = None
        self.arm_forward_command_quat = None
        self.arm_forward_command_path = None
        self.arm_forward_orientation_path = None
        self.arm_forward_path_index = 0
        self.arm_align_geometry_corrections = 0
        self.arm_align_residual_corrections = 0
        self.arm_stability_window.clear()
        self.grasp_latched = False
        self.grasp_close_command = None
        self.grasp_vertical_lift = 0.0
        self.grasp_vertical_lift_target = 0.0
        self.grasp_vertical_balance_started = False
        state = self.runner.get_control_state()
        door_root = _as_np(state["door_root_pose_w"])[0]
        door_bodies = _as_np(state["door_body_state_w"])[0]
        handle_state = door_bodies[self.handle_body_idx]
        grasp_state = (
            door_bodies[self.grasp_target_body_idx]
            if self.grasp_target_body_idx is not None
            else handle_state
        )
        panel_state = door_bodies[self.panel_body_idx]
        base = _as_np(state["root_states"])[0]
        # The generated asset is an X-thin door spanning Y.  Keep the side
        # convention explicit so all four directions use the same geometry.
        side_sign = 1.0 if self.robot_side == "positive_x" else -1.0
        self.door_normal = np.array([side_sign, 0.0, 0.0], dtype=np.float32)
        normal = self.door_normal[:2]
        # Fail early if reset/configuration put the robot on the wrong side.
        side_error = float((base[0] - door_root[0]) * side_sign)
        if side_error < 0.25:
            raise RuntimeError(
                f"robot is not on requested {self.robot_side} side: "
                f"base_x={base[0]:.3f}, door_x={door_root[0]:.3f}"
            )
        self.handle_pos = handle_state[:3]
        self.handle_collider_center_local, handle_radius, self.handle_collider_path = (
            self._measure_handle_geometry(handle_state, grasp_state)
        )
        self._measure_panel_geometry()
        if self.panel_center_world is None or self.hinge_world is None:
            raise RuntimeError("door panel or hinge geometry is unavailable")
        self.grasp_center_pos = self._physical_handle_grasp_position(handle_state)
        # Push at the panel centre, well away from the handle along the door
        # width.  Moving toward the free edge put the large B2Z1 gripper back
        # into the lever during PUSH_OPEN; the panel centre still provides a
        # useful hinge moment and remains in the arm's reliable workspace.
        self.panel_push_world = self.panel_center_world.copy()
        door_tangent = np.array(
            [-self.door_normal[1], self.door_normal[0]], dtype=np.float32
        )
        panel_contact_world = self.panel_push_world + self.door_normal * self.panel_half_thickness
        self.panel_contact_local = _quat_rotate_inverse_np(
            panel_state[3:7], panel_contact_world - panel_state[:3]
        )
        self.handle_lower_z = float(self.grasp_center_pos[2] - handle_radius)
        self.handle_upper_z = float(self.grasp_center_pos[2] + handle_radius)
        # ``normal`` points from the door toward the robot's initial side, so
        # the pre-door station lies along +normal from the handle.
        # The locomotion policy has a repeatable stopping offset near the
        # frame.  Keep the
        # geometric target here for logging, and use an explicit tolerance
        # below rather than driving the base into the frame.
        # Keep the refined 0.20 m fixed jaw outside the handle while the arm
        # remains in its randomized middle preset.  At 0.65 m the moving jaw
        # reaches the lever during locomotion and invalidates the precomputed
        # closed-door trajectory; 0.77 m retains arm reach with 0.12 m more
        # clearance.
        self.approach_distance = 0.77
        self.station_tolerance = 0.12
        self.station_xy = self.grasp_center_pos[:2] + self.approach_distance * normal
        self.panel_station_xy = self.station_xy.copy()
        self.panel_station_xy += float(
            np.dot(self.panel_center_world[:2] - self.panel_station_xy, door_tangent)
        ) * door_tangent
        self.door_forward = -self.door_normal
        self.initial_handle_pos = self.grasp_center_pos.copy()
        self.initial_door_root = door_root.copy()
        self.default_ee_goal_local = _as_np(state["ee_goal_cart_local"])[0]
        self.default_ee_goal_delta_rpy = _as_np(state["ee_goal_delta_rpy"])[0]
        self.default_ee_base_local = _quat_rotate_inverse_np(
            base[3:7], _as_np(state["ee_pos"])[0] - base[:3]
        )
        # The policy's reset target extends the gripper roughly 0.65 m in
        # front of the trunk.  That is useful for manipulation, but it is not
        # a stow pose: the fingers become the first robot parts to hit the
        # door during body pushing.  Pull the Cartesian goal back by the
        # measured reset-target offset, which places the TCP around local
        # x=0.20 m, behind the front hips at x=0.33 m.
        self.stow_ee_goal_local = self.default_ee_goal_local.copy()
        self.stow_ee_goal_local[0] = 0.0
        self.stow_ee_goal_local[2] = 0.10
        self.stow_ee_base_local = self.default_ee_base_local + (
            self.stow_ee_goal_local - self.default_ee_goal_local
        )
        self.ee_command_world = _as_np(state["ee_pos"])[0].copy()
        self.initial_ee_orientation = _as_np(state["ee_orn"])[0].copy()
        dof_names = list(state["dof_names"])
        self.initial_wrist_roll = float(
            _as_np(state["dof_pos"])[0, dof_names.index("joint6")]
        )
        self.ee_orientation_command = self.initial_ee_orientation.copy()
        # Keep orientation alignment at a fixed, collision-safe point on the
        # robot side of the handle.  Using the measured current EE position
        # as the target lets orientation IK drift move the target itself.
        # Put the open fingertips halfway between the robot-facing panel
        # surface and the handle surface nearest the panel.  This places the
        # tips behind the lever while retaining equal geometric clearance.
        panel_surface = panel_state[:3] + self.panel_half_thickness * self.door_normal
        handle_near_panel = self.grasp_center_pos - handle_radius * self.door_normal
        self.target_tip_mid_world = 0.5 * (panel_surface + handle_near_panel)
        self.target_tip_mid_world[1:] = self.grasp_center_pos[1:]
        self.center_standoff_world = self.target_tip_mid_world + 0.10 * self.door_normal
        self._introspect_robot_colliders(state)
        # Build the q_open capture frame from the real collision meshes.  All
        # points are expressed in gripper_link coordinates so the desired TCP
        # pose can later be solved by one rigid transform.
        robot_bodies = _as_np(state.get("robot_body_state_w"))
        ee_pos = _as_np(state.get("ee_pos"))[0]
        ee_quat = _as_np(state.get("ee_orn"))[0]
        if robot_bodies is not None and ee_pos is not None and ee_quat is not None:
            fixed_pose = robot_bodies[0, self.fixed_finger_body_idx, :7]
            moving_pose = robot_bodies[0, self.moving_finger_body_idx, :7]
            fixed_points = self.center_geometry["fixed_finger"].world_points(fixed_pose)
            moving_points = self.center_geometry["moving_finger"].world_points(moving_pose)
            fixed_local = _quat_rotate_points_np(_quat_conjugate_np(ee_quat), fixed_points - ee_pos)
            moving_local = _quat_rotate_points_np(_quat_conjugate_np(ee_quat), moving_points - ee_pos)
            # Use the two collider clouds to identify the opposing inner
            # surfaces.  The cloud centroids determine only the sign of the
            # separation axis; all contact points come from the mesh itself.
            separation_local = moving_local.mean(axis=0) - fixed_local.mean(axis=0)
            separation_local /= max(float(np.linalg.norm(separation_local)), 1.0e-8)
            fixed_projection = fixed_local @ separation_local
            moving_projection = moving_local @ separation_local
            fixed_inner_indices = np.flatnonzero(
                fixed_projection >= np.quantile(fixed_projection, 0.85)
            )
            moving_inner_indices = np.flatnonzero(
                moving_projection <= np.quantile(moving_projection, 0.15)
            )
            fixed_inner_center = fixed_local[fixed_inner_indices].mean(axis=0)
            moving_inner_center = moving_local[moving_inner_indices].mean(axis=0)
            capture_center = 0.5 * (fixed_inner_center + moving_inner_center)
            # The insertion/closing axis is the measured robot-facing vector,
            # projected orthogonally to the jaw separation axis.
            depth_local = _quat_rotate_inverse_np(ee_quat, -self.door_normal)
            depth_local -= separation_local * float(np.dot(depth_local, separation_local))
            depth_local /= max(float(np.linalg.norm(depth_local)), 1.0e-8)
            # A fingertip is the distal part of the *inner* collision surface,
            # not the rigid-body origin or the centroid of the whole mesh.
            # Select the last 5 percent along the measured insertion axis.
            fixed_inner_points = fixed_local[fixed_inner_indices]
            moving_inner_points = moving_local[moving_inner_indices]
            fixed_depth = fixed_inner_points @ depth_local
            moving_depth = moving_inner_points @ depth_local
            fixed_tip_points = fixed_inner_points[
                fixed_depth >= np.quantile(fixed_depth, 0.95)
            ]
            moving_tip_points = moving_inner_points[
                moving_depth >= np.quantile(moving_depth, 0.95)
            ]
            fixed_tip_center = fixed_tip_points.mean(axis=0)
            moving_tip_center = moving_tip_points.mean(axis=0)
            tip_capture_center = 0.5 * (fixed_tip_center + moving_tip_center)
            # Seat the lever as deeply as its radius permits.  The proximal
            # edge of either inner jaw limits the circular handle center;
            # retain a small open-jaw clearance so insertion itself does not
            # depend on contact.
            fixed_full_depth = fixed_local @ depth_local
            moving_full_depth = moving_local @ depth_local
            pocket_depth = max(
                float(fixed_full_depth.min()), float(moving_full_depth.min())
            )
            pocket_depth += float(handle_radius) + 0.0015
            pocket_capture_center = capture_center.copy()
            pocket_capture_center += (
                pocket_depth - float(np.dot(pocket_capture_center, depth_local))
            ) * depth_local
            self.open_template = {
                "fixed_points": fixed_local,
                "moving_points": moving_local,
                "fixed_inner_points": fixed_local[fixed_inner_indices],
                "moving_inner_points": moving_local[moving_inner_indices],
                "depth_axis": depth_local,
                "separation_axis": separation_local,
                "capture_center": capture_center,
                "fixed_tip_center": fixed_tip_center,
                "moving_tip_center": moving_tip_center,
                "tip_capture_center": tip_capture_center,
                "pocket_capture_center": pocket_capture_center,
                "pocket_depth": pocket_depth,
            }
            all_gripper_points_local = np.concatenate((fixed_local, moving_local), axis=0)
            rotation_sweep_radius = float(
                np.linalg.norm(all_gripper_points_local, axis=1).max()
            )
            self.rotation_sweep_radius = rotation_sweep_radius
            obstacle_front = max(
                float(np.dot(panel_surface, self.door_normal)),
                float(np.dot(self.grasp_center_pos, self.door_normal)) + handle_radius,
            )
            # Keep the aligned arm waypoint far enough from the handle and
            # panel for the complete move from the reset pose to be safe.
            # The refined mount-90 gripper reaches this all-orientations-safe
            # waypoint with a 0.13 m obstacle margin.  The old 0.18 m margin
            # put branch 1 beyond the bounded IK workspace (joint5 saturated),
            # while the full measured rotation_sweep_radius still remains
            # outside the panel and handle envelope.
            safe_ee_projection = obstacle_front + rotation_sweep_radius + 0.13
            self.orientation_standoff_ee_world = ee_pos.copy()
            self.orientation_standoff_ee_world += (
                safe_ee_projection
                - float(np.dot(self.orientation_standoff_ee_world, self.door_normal))
            ) * self.door_normal
            self.orientation_standoff_ee_world[1] = self.grasp_center_pos[1]
            self.orientation_standoff_ee_world[2] = self.grasp_center_pos[2] + 0.05
            initial_offsets_world = _quat_rotate_points_np(
                self.initial_ee_orientation, all_gripper_points_local
            )
            initial_min_normal_offset = float(
                (initial_offsets_world @ self.door_normal).min()
            )
            self.pregrasp_target_world = self.grasp_center_pos.copy()
            pregrasp_projection = obstacle_front + 0.03 - initial_min_normal_offset
            self.pregrasp_target_world += (
                pregrasp_projection
                - float(np.dot(self.pregrasp_target_world, self.door_normal))
            ) * self.door_normal
            self.pregrasp_target_world[2] += 0.05
            self.finger_separation_local = separation_local
            self.finger_closing_local = depth_local
            # The handle must settle in the concave pocket between the two
            # inner jaw surfaces, not at the line joining their distal tips.
            # Recover the TCP pose from that physical pocket center.
            self.grasp_ee_offset_local = pocket_capture_center
            # Keep the measured fingertip geometry for positioning, but do
            # not use its asymmetric centroids to define wrist roll.  The
            # gripper revolute axis is the invariant that must be parallel to
            # the handle axis for the jaws to close in one cross-section.
            tip_sep_local = fixed_tip_center - moving_tip_center
            tip_sep_local /= max(float(np.linalg.norm(tip_sep_local)), 1.0e-8)
            tip_sep_world = _quat_rotate_np(ee_quat, tip_sep_local)
            separation_world = tip_sep_world
            desired_axis_world = -self.door_normal.copy()
            desired_axis_world /= max(
                float(np.linalg.norm(desired_axis_world)), 1.0e-8
            )
            # jointGripper is 86 mm behind the TCP on local -X, so local +X
            # is the physical gripper shaft and local +Y is its revolute
            # axis.  Map those two URDF axes directly instead of inferring a
            # frame from mesh centroids.
            gripper_axis_local = np.array(
                [1.0, 0.0, 0.0], dtype=np.float32
            )
            handle_axis_world = _quat_rotate_np(
                handle_state[3:7], self.handle_axis_local
            )
            handle_axis_world -= desired_axis_world * float(
                np.dot(handle_axis_world, desired_axis_world)
            )
            handle_axis_world /= max(
                float(np.linalg.norm(handle_axis_world)), 1.0e-8
            )

            def _grasp_candidate(joint_axis_world):
                joint_axis_world = np.asarray(joint_axis_world, dtype=np.float32)
                closing_plane_axis = np.cross(
                    desired_axis_world, joint_axis_world
                )
                closing_plane_axis /= max(
                    float(np.linalg.norm(closing_plane_axis)), 1.0e-8
                )
                return _matrix_to_quat_np(
                    np.column_stack(
                        (
                            desired_axis_world,
                            joint_axis_world,
                            closing_plane_axis,
                        )
                    )
                )

            # Candidate 1 retains the branch convention used for a free end
            # pointing right; candidate 0 is the 180-degree roll equivalent.
            grasp_candidates = (
                _grasp_candidate(-handle_axis_world),
                _grasp_candidate(handle_axis_world),
            )
            candidate_distances = [
                self._quat_distance(ee_quat, candidate) for candidate in grasp_candidates
            ]
            # Mirror the wrist-roll branch with the lever direction.  From the
            # robot-facing view, a lever pointing toward the right hinge must
            # turn the gripper right (branch 1); a lever pointing left must
            # turn it left (branch 0).  Choosing the opposite branch still
            # reaches the handle, but puts the gripper in the wrong mirrored
            # configuration.
            door_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            viewer_right = np.cross(-self.door_normal, door_up)
            viewer_right /= max(float(np.linalg.norm(viewer_right)), 1.0e-8)
            lever_free_end_direction = self.hinge_world - handle_state[:3]
            lever_free_end_direction -= (
                float(np.dot(lever_free_end_direction, self.door_normal)) * self.door_normal
            )
            lever_free_end_direction[2] = 0.0
            lever_free_end_direction /= max(
                float(np.linalg.norm(lever_free_end_direction)), 1.0e-8
            )
            free_end_right = float(np.dot(lever_free_end_direction, viewer_right)) > 0.0
            selected_candidate = 1 if free_end_right else 0
            self.grasp_branch = selected_candidate
            self.grasp_orientation_quat = grasp_candidates[selected_candidate]
            target_gripper_axis_world = _quat_rotate_np(
                self.grasp_orientation_quat, gripper_axis_local
            )
            target_joint_axis_world = _quat_rotate_np(
                self.grasp_orientation_quat,
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            )
            gripper_axis_alignment = float(
                np.dot(target_gripper_axis_world, desired_axis_world)
            )
            joint_axis_alignment = abs(
                float(np.dot(target_joint_axis_world, handle_axis_world))
            )
            if gripper_axis_alignment < 1.0 - 1.0e-5:
                raise RuntimeError(
                    "grasp orientation does not align the gripper shaft with "
                    f"the door normal: dot={gripper_axis_alignment:.8f}"
                )
            if joint_axis_alignment < 1.0 - 1.0e-5:
                raise RuntimeError(
                    "gripper joint axis does not align with the handle axis: "
                    f"dot={joint_axis_alignment:.8f}"
                )
            print(
                f"[doorman][geometry] grasp branch={selected_candidate} "
                f"handle_free_end={'right' if free_end_right else 'left'} "
                f"lever_free_end_direction={lever_free_end_direction.round(4).tolist()} "
                f"orientation_distances={np.round(candidate_distances, 4).tolist()} "
                f"gripper_axis_world={target_gripper_axis_world.round(6).tolist()} "
                f"joint_axis_world={target_joint_axis_world.round(6).tolist()} "
                f"axis_alignment={gripper_axis_alignment:.8f} "
                f"joint_axis_alignment={joint_axis_alignment:.8f}",
                flush=True,
            )
            final_offsets_world = _quat_rotate_points_np(
                self.grasp_orientation_quat, all_gripper_points_local
            )
            final_min_normal_offset = float(
                (final_offsets_world @ self.door_normal).min()
            )
            self.final_min_normal_offset = final_min_normal_offset
            final_fixed_offsets_world = _quat_rotate_points_np(
                self.grasp_orientation_quat, fixed_local
            )
            fixed_distal_normal_offset = float(
                (final_fixed_offsets_world @ self.door_normal).min()
            )
            fixed_inner_offset_world = _quat_rotate_np(
                self.grasp_orientation_quat, fixed_inner_center
            )
            fixed_is_upper = bool(fixed_inner_offset_world[2] > 0.0)
            fixed_surface_target_z = float(
                self.grasp_center_pos[2]
                + (handle_radius + 0.013) * (1.0 if fixed_is_upper else -1.0)
            )
            fixed_anchored_ee_z = float(
                fixed_surface_target_z - fixed_inner_offset_world[2]
            )
            self.grasp_waypoint_world = self._effective_grasp_target(
                self.grasp_center_pos, self.grasp_orientation_quat
            )
            self.grasp_waypoint_world[2] = fixed_anchored_ee_z
            # Both wrist-roll branches use the same collider-derived target.
            # Runtime geometry feedback below compensates IK/load residuals;
            # branch-specific Cartesian constants do not generalize across
            # randomized arm states or handle heights.
            self.center_insert_waypoint_world = self.grasp_waypoint_world.copy()
            self.center_standoff_ee_world = self._effective_grasp_target(
                self.target_tip_mid_world, self.grasp_orientation_quat
            )
            self.center_standoff_ee_world[2] = fixed_anchored_ee_z
            center_standoff_projection = obstacle_front + 0.03 - final_min_normal_offset
            self.center_standoff_ee_world += (
                center_standoff_projection
                - float(np.dot(self.center_standoff_ee_world, self.door_normal))
            ) * self.door_normal
            self.center_standoff_nominal_world = self.center_standoff_ee_world.copy()
            self.center_insert_nominal_world = self.center_insert_waypoint_world.copy()
            self.grasp_nominal_world = self.grasp_waypoint_world.copy()
            # Keep the calculated physical frame for diagnostics, but use a
            # low-level-reachable branch for the first position-only solve.
            # The policy tracks this branch reliably; fingertip geometry is
            # still transformed from the real open-jaw template.
            self.reachable_grasp_orientation_quat = self.grasp_orientation_quat.copy()
            tip_separation_local = tip_sep_local
            insertion_local = depth_local
            self.grasp_ee_offset_world = _quat_rotate_np(self.grasp_orientation_quat, tip_capture_center)
            separation_world = _quat_rotate_np(self.grasp_orientation_quat, tip_separation_local)
            measured_sep_world = separation_world / max(float(np.linalg.norm(separation_world)), 1.0e-8)
            target_sep_world = _quat_rotate_np(self.grasp_orientation_quat, tip_separation_local)
            target_depth_world = _quat_rotate_np(self.grasp_orientation_quat, insertion_local)
            direct_sep_world = moving_points.mean(axis=0) - fixed_points.mean(axis=0)
            direct_sep_world /= max(float(np.linalg.norm(direct_sep_world)), 1.0e-8)
            print(
                f"[doorman][geometry] capture_center_local={capture_center.round(4).tolist()} "
                f"pocket_center_local={pocket_capture_center.round(4).tolist()} "
                f"pocket_depth={pocket_depth:.4f} "
                f"depth_local={depth_local.round(4).tolist()} separation_local={separation_local.round(4).tolist()} "
                f"offset_grasp_world={self.grasp_ee_offset_world.round(4).tolist()} "
                f"measured_sep_world={measured_sep_world.round(4).tolist()} "
                f"direct_sep_world={direct_sep_world.round(4).tolist()} "
                f"target_sep_world={target_sep_world.round(4).tolist()} "
                f"target_depth_world={target_depth_world.round(4).tolist()} "
                f"center_insert_waypoint={self.center_insert_waypoint_world.round(4).tolist()} "
                f"grasp_waypoint={self.grasp_waypoint_world.round(4).tolist()} "
                f"fixed_distal_normal_offset={fixed_distal_normal_offset:.4f}",
                flush=True,
            )
            # Keep an explicit, serializable snapshot of the physical capture
            # geometry.  This is a diagnostic truth source, not an IK target
            # guessed from a rigid-body origin.
            def _bounds(points):
                return {
                    "min": points.min(axis=0).astype(float).tolist(),
                    "max": points.max(axis=0).astype(float).tolist(),
                }

            fixed_world = fixed_points
            moving_world = moving_points
            fixed_inner_world = _quat_rotate_np(ee_quat, fixed_inner_center) + ee_pos
            moving_inner_world = _quat_rotate_np(ee_quat, moving_inner_center) + ee_pos
            upper_world, lower_world = (
                (fixed_inner_world, moving_inner_world)
                if fixed_inner_world[2] >= moving_inner_world[2]
                else (moving_inner_world, fixed_inner_world)
            )
            handle_center_world = self.grasp_center_pos.copy()
            handle_axis_world = _quat_rotate_np(handle_state[3:7], self.handle_axis_local)
            handle_axis_world /= max(float(np.linalg.norm(handle_axis_world)), 1.0e-8)
            handle_endpoints = [
                (handle_center_world - self.handle_half_length * handle_axis_world).astype(float).tolist(),
                (handle_center_world + self.handle_half_length * handle_axis_world).astype(float).tolist(),
            ]
            self.geometry_report = {
                "handle": {
                    "body": self.door_body_names[self.handle_body_idx],
                    "collider_path": self.handle_collider_path,
                    "center_world": handle_center_world.astype(float).tolist(),
                    "axis_world": handle_axis_world.astype(float).tolist(),
                    "radius_m": float(handle_radius),
                    "half_length_m": float(self.handle_half_length),
                    "endpoints_world": handle_endpoints,
                    "surface_z_m": [float(self.handle_lower_z), float(self.handle_upper_z)],
                },
                "gripper": {
                    "q_open_rad": float(self.q_open),
                    "tcp_body": "gripper_link",
                    "fixed_finger": {
                        "body": "link06",
                        "collider_paths": self.center_geometry["fixed_finger"].source_paths,
                        "local_bounds": _bounds(self.center_geometry["fixed_finger"].points_body),
                        "world_bounds_at_reset": _bounds(fixed_world),
                        "inner_surface_center_world": fixed_inner_world.astype(float).tolist(),
                    },
                    "moving_finger": {
                        "body": "gripperMover",
                        "collider_paths": self.center_geometry["moving_finger"].source_paths,
                        "local_bounds": _bounds(self.center_geometry["moving_finger"].points_body),
                        "world_bounds_at_reset": _bounds(moving_world),
                        "inner_surface_center_world": moving_inner_world.astype(float).tolist(),
                    },
                    "capture_box_at_reset": {
                        "center_world": (0.5 * (upper_world + lower_world)).astype(float).tolist(),
                        "upper_inner_world": upper_world.astype(float).tolist(),
                        "lower_inner_world": lower_world.astype(float).tolist(),
                        "vertical_opening_m": float(abs(upper_world[2] - lower_world[2])),
                        "horizontal_projection_error_m": float(max(
                            np.linalg.norm((upper_world - handle_center_world)[:2]),
                            np.linalg.norm((lower_world - handle_center_world)[:2]),
                        )),
                    },
                    "tcp_capture_offset_local": pocket_capture_center.astype(float).tolist(),
                    "inner_surface_centroid_offset_local": capture_center.astype(float).tolist(),
                    "pocket_depth_local_m": float(pocket_depth),
                    "tcp_tip_offsets_local": {
                        "fixed_tip": fixed_tip_center.astype(float).tolist(),
                        "moving_tip": moving_tip_center.astype(float).tolist(),
                        "tip_midpoint": tip_capture_center.astype(float).tolist(),
                    },
                    "finger_depth_ranges_from_tcp_m": {
                        "fixed": [float((fixed_local @ depth_local).min()), float((fixed_local @ depth_local).max())],
                        "moving": [float((moving_local @ depth_local).min()), float((moving_local @ depth_local).max())],
                    },
                    "joint_gripper_from_tcp": {
                        "position_local_m": [-0.086, 0.0, 0.0],
                        "axis_local": [0.0, 1.0, 0.0],
                        "source": "URDF fixed transforms: link06->stator 0.051 m, stator->joint 0.049 m, stator->TCP 0.135 m",
                    },
                    "insertion_axis_world": (-self.door_normal).astype(float).tolist(),
                    "rotation_sweep_radius_m": rotation_sweep_radius,
                    "orientation_standoff_ee_world": self.orientation_standoff_ee_world.astype(float).tolist(),
                    "pregrasp_target_ee_world": self.pregrasp_target_world.astype(float).tolist(),
                    "center_standoff_ee_world": self.center_standoff_ee_world.astype(float).tolist(),
                    "center_insert_ee_world": self.center_insert_waypoint_world.astype(float).tolist(),
                },
            }
            print(
                f"[doorman][geometry] handle endpoints={handle_endpoints} "
                f"r={handle_radius:.4f} half_length={self.handle_half_length:.4f}",
                flush=True,
            )
            print(
                f"[doorman][geometry] q_open={self.q_open:.4f} "
                f"fixed_inner={fixed_inner_world.round(4).tolist()} "
                f"moving_inner={moving_inner_world.round(4).tolist()} "
                f"opening={abs(upper_world[2] - lower_world[2]):.4f}m "
                f"capture_center={(0.5 * (upper_world + lower_world)).round(4).tolist()} "
                f"xy_error={self.geometry_report['gripper']['capture_box_at_reset']['horizontal_projection_error_m']:.4f}m",
                flush=True,
            )
            print(
                f"[doorman][geometry] fingertip offsets from TCP: "
                f"fixed={fixed_tip_center.round(4).tolist()} "
                f"moving={moving_tip_center.round(4).tolist()} "
                f"midpoint={tip_capture_center.round(4).tolist()} "
                f"jointGripper=[-0.086, 0.0, 0.0]",
                flush=True,
            )
        else:
            raise RuntimeError("robot body/EE state unavailable while building the open-gripper collider template")
        self._build_open_loop_trajectory(state)
        # Geometry above is intentionally measured at q_open.  Once that
        # template is cached, start the actual episode with the jaw closed;
        # PREGRASP opens it only after the arm is aligned and stationary.
        gripper_joint_id = int(self.env.gripper_joint_ids[0].item())
        joint_pos = self.env.robot.data.joint_pos.clone()
        joint_vel = self.env.robot.data.joint_vel.clone()
        joint_pos[:, gripper_joint_id] = float(self.q_closed)
        joint_vel[:, gripper_joint_id] = 0.0
        self.env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.env.robot.set_joint_position_target(joint_pos)
        self.runner.obs = self.env.get_observations()
        state = self.runner.get_control_state()
        print(
            f"[doorman] RESET physical_handle={self.handle_pos.tolist()} "
            f"grasp_center={self.grasp_center_pos.tolist()} station={self.station_xy.tolist()} "
            f"normal={self.door_normal.tolist()} forward={self.door_forward.tolist()} side={self.robot_side}",
            flush=True,
        )
        return state

    def _introspect_robot_colliders(self, state):
        """Cache collision geometry from the actual loaded robot USD stage."""
        try:
            import omni.usd
            from pxr import Usd
            stage = omni.usd.get_context().get_stage()
        except Exception as exc:
            raise RuntimeError(f"Isaac Sim USD stage is unavailable for collider introspection: {exc}") from exc
        root = _stage_robot_root(stage)
        if root is None:
            raise RuntimeError("Unable to find robot root in the loaded USD stage")
        found = {}
        # The decomposed gripper meshes also contain the palm and linkage.
        # Only these distal hulls form the opposing fingertips.  The names
        # are inherited directly from the lab30 B2Z1 URDF decomposition.
        for body_name, key, tokens in (
            ("link06", "wrist", ("link06/collisions/mesh_0",)),
            (
                "link06",
                "fixed_finger",
                ("fixed_convex_023", "fixed_convex_024"),
            ),
            ("gripperMover", "moving_finger", ("moving_convex_000",)),
            ("gripper_link", "tcp", ("gripper_link/collisions",)),
        ):
            body = None
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if prim.GetName().lower() == body_name.lower() and prim.IsValid():
                    body = prim
                    break
            if body is None:
                raise RuntimeError(f"robot body {body_name} not found below {root.GetPath()}")
            geom = _read_collider_points(stage, body, tokens)
            if geom is None:
                raise RuntimeError(f"required collider geometry not found for {key} under {body.GetPath()}")
            geom.name = key
            found[key] = geom
            lo, hi = geom.bounds()
            print(f"[doorman][geometry] {key}: points={len(geom.points_body)} bounds={lo.round(4).tolist()}..{hi.round(4).tolist()} paths={geom.source_paths[:2]}", flush=True)
        self.center_geometry = found
        # Infer the open endpoint from the actual reset joint and the USD
        # limits.  This avoids assuming a sign while preserving the configured
        # joint semantics.  The opposite endpoint is used for grasp closing.
        joint_id = int(self.env.gripper_joint_ids[0].item())
        q_now = float(_as_np(state["dof_pos"])[0, joint_id])
        limits = _as_np(self.env.dof_pos_limits)[joint_id]
        finite = limits[np.isfinite(limits)]
        if len(finite) < 2:
            raise RuntimeError(f"jointGripper has no finite limits: {limits.tolist()}")
        # Match the B2Z1 control interface: the allowed open command is -0.785
        # rad.  The URDF lower limit is a mechanical bound, not an operating
        # command, and must not be treated as the fully-open target.
        self.q_open = float(np.clip(-0.785, finite.min(), finite.max()))
        self.q_closed = float(np.clip(0.0, finite.min(), finite.max()))
        # q_closed is only the search limit.  GRASP_HANDLE closes toward it
        # and latches the measured joint angle when both jaw faces reach the
        # current handle, so handle diameter is not encoded as a joint angle.
        self.q_grasp = self.q_closed
        self.grasp_latched = False
        print(
            f"[doorman][geometry] jointGripper q_now={q_now:.4f} "
            f"q_open={self.q_open:.4f} q_close_limit={self.q_closed:.4f} "
            f"q_closed={self.q_closed:.4f}",
            flush=True,
        )

    def _configure_grasp_friction(self):
        """Restore the friction used by the original DoorMan robot asset."""
        try:
            import omni.usd
            from isaaclab import sim as sim_utils
            from pxr import UsdPhysics

            stage = omni.usd.get_context().get_stage()
            material_path = "/World/PhysicsMaterials/doorman_gripper"
            material_cfg = sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0,
                dynamic_friction=2.0,
                restitution=0.0,
                friction_combine_mode="average",
            )
            if not stage.GetPrimAtPath(material_path).IsValid():
                material_cfg.func(material_path, material_cfg)
            collider_paths = (
                self.center_geometry["fixed_finger"].source_paths
                + self.center_geometry["moving_finger"].source_paths
            )
            bound_paths = []
            for collider_path in collider_paths:
                collider_prim = stage.GetPrimAtPath(collider_path)
                while collider_prim.IsValid() and not collider_prim.HasAPI(UsdPhysics.CollisionAPI):
                    collider_prim = collider_prim.GetParent()
                if not collider_prim.IsValid():
                    raise RuntimeError(f"no collision ancestor for {collider_path}")
                collision_path = str(collider_prim.GetPath())
                if collision_path not in bound_paths:
                    sim_utils.bind_physics_material(
                        collision_path, material_path, stage=stage
                    )
                    bound_paths.append(collision_path)
            print(
                f"[doorman][geometry] fingertip friction=2.0 bound to {bound_paths}",
                flush=True,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to configure DoorMan fingertip friction: {exc}") from exc

    def _measure_handle_geometry(self, handle_state, grasp_state):
        """Resolve the robot-facing lever capsule, not the handle body origin."""
        try:
            import omni.usd
            from pxr import Gf, Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            collider_name = "handle_outside" if self.robot_side == "positive_x" else "handle_inside"
            path = f"/World/envs/env_0/door/door_handle/{collider_name}"
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Capsule):
                raise RuntimeError(f"missing capsule prim {path}")
            capsule = UsdGeom.Capsule(prim)
            radius = float(capsule.GetRadiusAttr().Get() or 0.0)
            height = float(capsule.GetHeightAttr().Get() or 0.0)
            world_xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            center_w = np.asarray(world_xform.Transform(Gf.Vec3d(0.0, 0.0, 0.0)), dtype=np.float32)
            axis_name = str(capsule.GetAxisAttr().Get() or "Z").upper()
            axis_index = {"X": 0, "Y": 1, "Z": 2}[axis_name]
            axis_primitive = np.zeros(3, dtype=np.float32); axis_primitive[axis_index] = 1.0
            endpoint_w = np.asarray(
                world_xform.Transform(Gf.Vec3d(*[float(v) for v in axis_primitive * (0.5 * height)])),
                dtype=np.float32,
            )
            axis_w = endpoint_w - center_w
            measured_half_length = float(np.linalg.norm(axis_w))
            axis_w /= max(measured_half_length, 1.0e-8)
            center_local = _quat_rotate_inverse_np(handle_state[3:7], center_w - handle_state[:3])
            self.handle_axis_local = _quat_rotate_inverse_np(handle_state[3:7], axis_w)
            self.handle_half_length = measured_half_length
            if radius <= 0.0 or height <= 0.0:
                raise RuntimeError(f"invalid capsule dimensions radius={radius}, height={height}")
            print(
                f"[doorman] grasp collider={path} center_w={center_w.round(4).tolist()} "
                f"center_in_handle={center_local.round(4).tolist()} radius={radius:.4f} height={height:.4f}",
                flush=True,
            )
            return center_local, radius, path
        except Exception as exc:
            raise RuntimeError(f"physical handle_outside collider is unavailable: {exc}") from exc

    def _measure_panel_geometry(self):
        """Read the closed panel and hinge geometry from the loaded USD."""
        try:
            import omni.usd
            from pxr import Gf, Usd, UsdGeom
            stage = omni.usd.get_context().get_stage()
            panel_prim = None
            panel_corners = None
            for prim in stage.Traverse():
                if prim.GetName().lower() == "door_panel" and prim.IsValid():
                    panel_prim = prim
                    break
            if panel_prim is None:
                return
            for prim in Usd.PrimRange(panel_prim, Usd.TraverseInstanceProxies()):
                if prim.IsA(UsdGeom.Cube):
                    cube = UsdGeom.Cube(prim)
                    size = 0.5 * float(cube.GetSizeAttr().Get() or 0.0)
                    if size > 0.0:
                        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        corners = np.asarray([
                            xf.Transform(Gf.Vec3d(x, y, z))
                            for x in (-size, size) for y in (-size, size) for z in (-size, size)
                        ], dtype=np.float32)
                        center = corners.mean(axis=0)
                        self.panel_half_thickness = float(np.max(np.abs((corners - center) @ self.door_normal)))
                        self.panel_center_world = center
                        panel_corners = corners
                        break
            hinge_prim = stage.GetPrimAtPath("/World/envs/env_0/door/root/hinge_joint")
            root_prim = stage.GetPrimAtPath("/World/envs/env_0/door/root")
            if not hinge_prim or not root_prim:
                raise RuntimeError("door hinge or root prim is missing")
            local_pos = hinge_prim.GetAttribute("physics:localPos0").Get()
            local_rot = hinge_prim.GetAttribute("physics:localRot0").Get()
            root_xf = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            self.hinge_world = np.asarray(root_xf.Transform(Gf.Vec3d(local_pos)), dtype=np.float32)
            hinge_axis_root = Gf.Rotation(local_rot).TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
            hinge_axis_world = root_xf.TransformDir(hinge_axis_root)
            self.hinge_axis_sign = 1.0 if float(hinge_axis_world[2]) >= 0.0 else -1.0
            if panel_corners is None:
                raise RuntimeError("door panel collider corners are unavailable")
            panel_tangent = np.cross(
                np.array([0.0, 0.0, 1.0], dtype=np.float32), self.door_normal
            )
            self.panel_hinge_to_free_edge = float(
                np.max(np.abs((panel_corners - self.hinge_world) @ panel_tangent))
            )
            self.turn_gap_angle = _door_gap_angle(
                self.panel_hinge_to_free_edge,
                2.0 * self.panel_half_thickness,
                self.turn_gap_clearance,
            )
            handle_joint_prim = stage.GetPrimAtPath(
                "/World/envs/env_0/door/door_panel/handle_joint"
            )
            if not handle_joint_prim or not handle_joint_prim.IsValid():
                raise RuntimeError("door handle joint prim is missing")
            handle_local_rot = handle_joint_prim.GetAttribute("physics:localRot0").Get()
            handle_axis_panel = Gf.Rotation(handle_local_rot).TransformDir(
                Gf.Vec3d(1.0, 0.0, 0.0)
            )
            self.handle_joint_axis_panel_local = np.asarray(
                handle_axis_panel, dtype=np.float32
            )
            self.handle_joint_axis_panel_local /= max(
                float(np.linalg.norm(self.handle_joint_axis_panel_local)), 1.0e-8
            )
            print(
                f"[doorman][geometry] panel_half_thickness={self.panel_half_thickness:.4f} "
                f"panel_center={self.panel_center_world.round(4).tolist()} "
                f"hinge={self.hinge_world.round(4).tolist()} "
                f"hinge_to_free_edge={self.panel_hinge_to_free_edge:.4f} "
                f"turn_gap={self.turn_gap_clearance:.4f} "
                f"turn_gap_angle_deg={math.degrees(self.turn_gap_angle):.4f} "
                f"hinge_axis_sign={self.hinge_axis_sign:+.0f} "
                f"handle_axis_panel={self.handle_joint_axis_panel_local.round(4).tolist()}",
                flush=True,
            )
        except Exception as exc:
            print(f"[doorman][warn] panel collider introspection unavailable: {exc}", flush=True)

    def _geometry_metrics(self, state, grasp_center):
        """Compute conservative capture and panel-clearance metrics."""
        bodies = _as_np(state.get("robot_body_state_w"))
        if bodies is None or self.center_geometry is None:
            return {"valid": False}
        poses = {
            "fixed_finger": bodies[0, self.fixed_finger_body_idx, :7],
            "moving_finger": bodies[0, self.moving_finger_body_idx, :7],
        }
        points = {name: geom.world_points(poses[name]) for name, geom in self.center_geometry.items() if name in poses}
        if "tcp" in self.center_geometry:
            tcp_idx = next((i for i, n in enumerate(self.env.body_names) if n.lower() == "gripper_link"), None)
            if tcp_idx is not None:
                points["tcp"] = self.center_geometry["tcp"].world_points(bodies[0, tcp_idx, :7])
        if "wrist" in self.center_geometry:
            points["wrist"] = self.center_geometry["wrist"].world_points(bodies[0, self.fixed_finger_body_idx, :7])
        fixed = points["fixed_finger"]
        moving = points["moving_finger"]
        # Recompute the opposing inner surfaces from the current USD collider
        # clouds. The reset template only identifies jaw semantics; using its
        # stale points here can report a centered grasp while IK is off-target.
        separation_world = moving.mean(axis=0) - fixed.mean(axis=0)
        separation_world /= max(float(np.linalg.norm(separation_world)), 1.0e-8)
        fixed_projection = fixed @ separation_world
        moving_projection = moving @ separation_world
        fixed_c = fixed[fixed_projection >= np.quantile(fixed_projection, 0.85)].mean(axis=0)
        moving_c = moving[moving_projection <= np.quantile(moving_projection, 0.15)].mean(axis=0)
        upper_c, lower_c = (fixed_c, moving_c) if fixed_c[2] >= moving_c[2] else (moving_c, fixed_c)
        channel_center = 0.5 * (upper_c + lower_c)
        channel_center_xy_error = float(
            np.linalg.norm((channel_center - grasp_center)[:2])
        )
        upper_xy_error = float(np.linalg.norm((upper_c - grasp_center)[:2]))
        lower_xy_error = float(np.linalg.norm((lower_c - grasp_center)[:2]))
        horizontal_error = max(
            upper_xy_error,
            lower_xy_error,
        )
        z_span = abs(float(fixed_c[2] - moving_c[2]))
        handle_radius = max(float(self.handle_upper_z - self.handle_lower_z) * 0.5, 1.0e-4)
        upper_gap = upper_c[2] - (grasp_center[2] + handle_radius)
        lower_gap = (grasp_center[2] - handle_radius) - lower_c[2]
        depth_axis_w = _quat_rotate_np(_as_np(state["ee_orn"])[0], self.open_template["depth_axis"])
        tip_depth = float(np.quantile(np.concatenate((fixed, moving)) @ depth_axis_w, 0.95))
        depth = abs(tip_depth - float(np.dot(grasp_center, depth_axis_w)))
        tcp_pos = _as_np(state["ee_pos"])[0]
        tcp_quat = _as_np(state["ee_orn"])[0]
        pocket_center = tcp_pos + _quat_rotate_np(
            tcp_quat, self.open_template["pocket_capture_center"]
        )
        pocket_error = pocket_center - grasp_center
        panel_center = _as_np(state["door_body_state_w"])[0, self.panel_body_idx, :3]
        all_points = np.concatenate(list(points.values()), axis=0)
        clearance = float(np.min((all_points - panel_center) @ self.door_normal) - self.panel_half_thickness)
        handle_state = _as_np(state["door_body_state_w"])[0, self.handle_body_idx, :7]
        handle_axis = _quat_rotate_np(handle_state[3:7], self.handle_axis_local)
        handle_axis /= max(float(np.linalg.norm(handle_axis)), 1.0e-8)
        fixed_to_handle = grasp_center - fixed_c
        handle_to_moving = moving_c - grasp_center
        fixed_axial_offset = float(np.dot(fixed_c - grasp_center, handle_axis))
        handle_between_jaws = bool(
            float(np.dot(fixed_to_handle, separation_world)) > handle_radius
            and float(np.dot(handle_to_moving, separation_world)) > handle_radius
        )
        handle_radius = max(float(self.handle_upper_z - self.handle_lower_z) * 0.5, 1.0e-4)

        def _capsule_clearance(cloud):
            delta = cloud - grasp_center
            axial = np.clip(delta @ handle_axis, -self.handle_half_length, self.handle_half_length)
            closest_axis = grasp_center + axial[:, None] * handle_axis
            return float(np.min(np.linalg.norm(cloud - closest_axis, axis=1) - handle_radius))

        return {
            "valid": True,
            "horizontal_error": horizontal_error,
            "channel_center_xy_error": channel_center_xy_error,
            "upper_xy_error": upper_xy_error,
            "lower_xy_error": lower_xy_error,
            "z_span": z_span,
            "upper_gap": float(upper_gap),
            "lower_gap": float(lower_gap),
            "grasp_depth": float(depth),
            "pocket_center_error": float(np.linalg.norm(pocket_error)),
            "pocket_normal_error": float(np.dot(pocket_error, self.door_normal)),
            "pocket_vertical_error": float(pocket_error[2]),
            "pocket_center": pocket_center,
            "panel_clearance": clearance,
            "fixed_handle_clearance": _capsule_clearance(fixed),
            "moving_handle_clearance": _capsule_clearance(moving),
            "fixed_center": fixed_c,
            "moving_center": moving_c,
            "channel_center": channel_center,
            "fixed_axial_offset": fixed_axial_offset,
            "handle_between_jaws": handle_between_jaws,
        }

    def _current_capture_center(self, state):
        """Return the midpoint of the two measured open-jaw inner surfaces."""
        bodies = _as_np(state.get("robot_body_state_w"))
        if bodies is None or self.center_geometry is None:
            return None
        fixed_pose = bodies[0, self.fixed_finger_body_idx, :7]
        moving_pose = bodies[0, self.moving_finger_body_idx, :7]
        fixed = self.center_geometry["fixed_finger"].world_points(fixed_pose)
        moving = self.center_geometry["moving_finger"].world_points(moving_pose)
        separation = moving.mean(axis=0) - fixed.mean(axis=0)
        separation /= max(float(np.linalg.norm(separation)), 1.0e-8)
        fixed_proj = fixed @ separation
        moving_proj = moving @ separation
        fixed_inner = fixed[fixed_proj >= np.quantile(fixed_proj, 0.85)].mean(axis=0)
        moving_inner = moving[moving_proj <= np.quantile(moving_proj, 0.15)].mean(axis=0)
        return 0.5 * (fixed_inner + moving_inner)

    def _panel_finger_clearance(self, state, hinge_joint=None):
        """Return the closest finger-collider clearance from the robot-side panel face."""
        bodies = _as_np(state.get("robot_body_state_w"))
        door_bodies = _as_np(state.get("door_body_state_w"))
        if bodies is None or door_bodies is None or self.center_geometry is None:
            return float("nan")
        if hinge_joint is None:
            hinge_joint = float(
                _as_np(state["door_joint_position"])[0, self.hinge_idx]
            )
        panel_center = door_bodies[0, self.panel_body_idx, :3]
        panel_normal = _quat_rotate_np(
            self._yaw_quat(self.hinge_axis_sign * hinge_joint),
            self.door_normal,
        )
        fixed = self.center_geometry["fixed_finger"].world_points(
            bodies[0, self.fixed_finger_body_idx, :7]
        )
        moving = self.center_geometry["moving_finger"].world_points(
            bodies[0, self.moving_finger_body_idx, :7]
        )
        finger_points = np.concatenate((fixed, moving), axis=0)
        return float(
            np.min((finger_points - panel_center) @ panel_normal)
            - self.panel_half_thickness
        )

    @staticmethod
    def _center_capture_ready(metrics):
        """Return whether the open jaws geometrically enclose the handle."""
        return bool(
            metrics.get("valid")
            and metrics.get("channel_center_xy_error", 999.0) <= 0.018
            and metrics.get("horizontal_error", 999.0) <= 0.022
            and abs(metrics.get("pocket_normal_error", 999.0)) <= 0.012
            and metrics.get("upper_gap", -1.0) >= 0.003
            and metrics.get("lower_gap", -1.0) >= 0.003
            and metrics.get("grasp_depth", 0.0) >= 0.015
            and metrics.get("panel_clearance", -1.0) >= 0.005
        )

    def _update_center_geometry_feedback(
        self, state, metrics, *, allow_normal=False, unilateral_contact=False
    ):
        """Correct nominal grasp targets from measured jaw/handle geometry.

        This feedback is deliberately independent of the door handedness and
        random seed.  At the standoff it only aligns the open channel in the
        door tangent plane.  Normal correction is enabled after insertion has
        slowed down and remains constrained by measured panel clearance.
        """
        if (
            not metrics.get("valid")
            or self.center_standoff_nominal_world is None
            or self.stage_steps % 5 != 0
        ):
            return False
        ee_speed = float(np.linalg.norm(_as_np(state["ee_vel"])[0, :3]))
        if ee_speed > 0.06:
            return False

        channel_error = (
            np.asarray(metrics["channel_center"], dtype=np.float32)
            - self.grasp_center_pos
        )
        normal_error = float(np.dot(channel_error, self.door_normal))
        tangent_error = channel_error - normal_error * self.door_normal
        tangent_step = -0.45 * tangent_error
        tangent_step_norm = float(np.linalg.norm(tangent_step))
        if tangent_step_norm > 0.004:
            tangent_step *= 0.004 / tangent_step_norm

        changed = tangent_step_norm > 0.00025
        if changed:
            proposed = self.center_tangent_feedback_world + tangent_step
            proposed_norm = float(np.linalg.norm(proposed))
            if proposed_norm > 0.15:
                proposed *= 0.15 / proposed_norm
            self.center_tangent_feedback_world = proposed.astype(np.float32)

        if allow_normal and float(np.linalg.norm(tangent_error)) <= 0.015:
            pocket_normal_error = float(metrics["pocket_normal_error"])
            normal_step = float(np.clip(-0.35 * pocket_normal_error, -0.002, 0.002))
            if unilateral_contact:
                # A single jaw touching during insertion means the channel is
                # not ready to go deeper.  Back away while tangent feedback
                # recenters it instead of loading that jaw against the lever.
                normal_step = max(normal_step, 0.002)
            proposed_normal = float(
                np.clip(self.center_normal_feedback + normal_step, -0.04, 0.04)
            )
            clearance_delta = proposed_normal - self.center_normal_feedback
            if float(metrics["panel_clearance"]) + clearance_delta >= 0.006:
                if abs(clearance_delta) > 0.00025:
                    changed = True
                self.center_normal_feedback = proposed_normal

        tangent = self.center_tangent_feedback_world
        normal = self.center_normal_feedback * self.door_normal
        self.center_standoff_ee_world = self.center_standoff_nominal_world + tangent
        self.center_insert_waypoint_world = (
            self.center_insert_nominal_world + tangent + normal
        )
        self.grasp_waypoint_world = self.grasp_nominal_world + tangent + normal
        return changed

    def _update_fixed_jaw_feedback(self, state, metrics, *, fixed_contact=False):
        """Bring the non-actuated jaw gently to one side of the handle."""
        if fixed_contact or not metrics.get("valid") or self.stage_steps % 5 != 0:
            return False
        if float(np.linalg.norm(_as_np(state["ee_vel"])[0, :3])) > 0.06:
            return False

        fixed_center = np.asarray(metrics["fixed_center"], dtype=np.float32)
        handle_state = _as_np(state["door_body_state_w"])[
            0, self.handle_body_idx, :7
        ]
        handle_axis = _quat_rotate_np(handle_state[3:7], self.handle_axis_local)
        handle_axis /= max(float(np.linalg.norm(handle_axis)), 1.0e-8)
        center_delta = fixed_center - self.grasp_center_pos
        closest_axis = self.grasp_center_pos + np.clip(
            float(np.dot(center_delta, handle_axis)),
            -self.handle_half_length,
            self.handle_half_length,
        ) * handle_axis
        toward_handle = closest_axis - fixed_center
        radial_distance = float(np.linalg.norm(toward_handle))
        if radial_distance < 1.0e-6:
            return False
        toward_handle /= radial_distance

        # Stop within the contact manifold's spatial noise rather than
        # integrating a sub-millimetre residual into increasing force.
        target_clearance = 0.0005
        distance_error = float(metrics["fixed_handle_clearance"]) - target_clearance
        if abs(distance_error) <= 0.001:
            return False
        step_distance = float(np.clip(0.45 * distance_error, -0.003, 0.003))
        if abs(step_distance) <= 0.0002:
            return False
        delta = step_distance * toward_handle
        normal_delta = float(np.dot(delta, self.door_normal))
        tangent_delta = delta - normal_delta * self.door_normal

        proposed_normal = float(
            np.clip(self.center_normal_feedback + normal_delta, -0.04, 0.04)
        )
        applied_normal_delta = proposed_normal - self.center_normal_feedback
        if float(metrics["panel_clearance"]) + applied_normal_delta < 0.006:
            return False
        proposed_tangent = self.center_tangent_feedback_world + tangent_delta
        tangent_norm = float(np.linalg.norm(proposed_tangent))
        if tangent_norm > 0.15:
            proposed_tangent *= 0.15 / tangent_norm

        self.center_normal_feedback = proposed_normal
        self.center_tangent_feedback_world = proposed_tangent.astype(np.float32)
        tangent = self.center_tangent_feedback_world
        normal = self.center_normal_feedback * self.door_normal
        self.center_insert_waypoint_world = (
            self.center_insert_nominal_world + tangent + normal
        )
        self.grasp_waypoint_world = self.grasp_nominal_world + tangent + normal
        return True

    def _tip_metrics(self, state, handle_center):
        """Measure the actual open-jaw fingertip pair against the handle."""
        bodies = _as_np(state.get("robot_body_state_w"))
        ee_pos = _as_np(state.get("ee_pos"))
        ee_orn = _as_np(state.get("ee_orn"))
        if bodies is None or ee_pos is None or ee_orn is None or not self.open_template:
            return {"valid": False}
        tcp_pos = ee_pos[0]
        tcp_quat = ee_orn[0]
        fixed_tip = _quat_rotate_np(tcp_quat, self.open_template["fixed_tip_center"]) + tcp_pos
        moving_tip = _quat_rotate_np(tcp_quat, self.open_template["moving_tip_center"]) + tcp_pos
        upper, lower = (fixed_tip, moving_tip) if fixed_tip[2] >= moving_tip[2] else (moving_tip, fixed_tip)
        radius = max(float(self.handle_upper_z - self.handle_lower_z) * 0.5, 1.0e-4)
        return {
            "valid": True,
            "fixed_tip": fixed_tip,
            "moving_tip": moving_tip,
            "upper_tip": upper,
            "lower_tip": lower,
            "horizontal_error": float(max(np.linalg.norm((upper - handle_center)[:2]), np.linalg.norm((lower - handle_center)[:2]))),
            "upper_gap": float(upper[2] - (handle_center[2] + radius)),
            "lower_gap": float((handle_center[2] - radius) - lower[2]),
            "tip_midpoint_error": float(np.linalg.norm(0.5 * (fixed_tip + moving_tip) - self.target_tip_mid_world)),
        }

    def _physical_handle_grasp_position(self, handle_state):
        return handle_state[:3] + _quat_rotate_np(handle_state[3:7], self.handle_collider_center_local)

    def _contact_is_on_handle_capsule(self, point, handle_state):
        if point is None or not np.isfinite(point).all() or self.handle_axis_local is None:
            return False
        center = self._physical_handle_grasp_position(handle_state)
        axis = _quat_rotate_np(handle_state[3:7], self.handle_axis_local)
        delta = np.asarray(point, dtype=np.float32) - center
        axial = float(np.clip(np.dot(delta, axis), -self.handle_half_length, self.handle_half_length))
        radial_distance = float(np.linalg.norm(delta - axial * axis))
        return radial_distance <= 0.018 and abs(float(np.dot(delta, axis))) <= self.handle_half_length + 0.008

    def _contacts_form_opposed_grasp(self, points, handle_state):
        """Validate that two contacts oppose each other in one handle section."""
        points = np.asarray(points, dtype=np.float32).reshape(2, 3)
        if not np.isfinite(points).all():
            return False, {}
        center = self._physical_handle_grasp_position(handle_state)
        axis = _quat_rotate_np(handle_state[3:7], self.handle_axis_local)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-8)
        delta = points - center
        axial = delta @ axis
        radial = delta - axial[:, None] * axis
        closing_axis = np.cross(-self.door_normal, axis)
        closing_axis /= max(float(np.linalg.norm(closing_axis)), 1.0e-8)
        closing_coordinates = radial @ closing_axis
        radius = max(float(self.handle_upper_z - self.handle_lower_z) * 0.5, 1.0e-4)
        metrics = {
            "axial": axial.copy(),
            "axial_delta": abs(float(axial[0] - axial[1])),
            "closing_coordinates": closing_coordinates.copy(),
        }
        valid = bool(
            metrics["axial_delta"] <= 0.020
            and np.all(np.abs(axial) <= max(self.handle_half_length - 0.005, 0.0))
            and float(closing_coordinates[0] * closing_coordinates[1]) < 0.0
            and np.all(np.abs(closing_coordinates) >= 0.25 * radius)
        )
        return valid, metrics

    def _robot_side_grasp_position(self, handle_pos, reference_pos):
        """Choose the physical handle grasp point on the robot-facing side.

        The generated marker is 0.15 m along the handle axle.  Older assets
        place it on -X, so using it unchanged after moving the robot to +X
        would put the palm behind the lever.  Preserve its measured offset
        while selecting the sign that faces the configured robot side.
        """
        handle_pos = np.asarray(handle_pos, dtype=np.float32)
        reference_pos = np.asarray(reference_pos, dtype=np.float32)
        offset = float(np.dot(reference_pos[:3] - handle_pos[:3], self.door_normal))
        magnitude = max(abs(offset), 0.10)
        return handle_pos[:3] + self.door_normal * magnitude

    def _base_command(self, state):
        base = _as_np(state["root_states"])[0]
        error_w = np.array([self.station_xy[0] - base[0], self.station_xy[1] - base[1], 0.0], dtype=np.float32)
        yaw = _quat_yaw(base[3:7])
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([c * error_w[0] + s * error_w[1], -s * error_w[0] + c * error_w[1]])
        target_yaw = math.atan2(-self.door_normal[1], -self.door_normal[0])
        yaw_error = ((target_yaw - yaw + math.pi) % (2 * math.pi)) - math.pi
        # Position-only commands let the low-level locomotion controller coast
        # past the station.  Add measured velocity damping so the base is
        # actually stationary before arm contact is attempted.
        vel_w = base[7:10]
        local_vel = np.array([c * vel_w[0] + s * vel_w[1], -s * vel_w[0] + c * vel_w[1]])
        cmd = np.array(
            [
                np.clip(1.2 * local[0] - 1.0 * local_vel[0], -0.8, 0.8),
                np.clip(1.2 * local[1] - 1.0 * local_vel[1], -0.8, 0.8),
                np.clip(1.5 * yaw_error - 0.35 * base[12], -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        station_dist = float(np.linalg.norm(error_w[:2]))
        planar_command_norm = float(np.linalg.norm(cmd[:2]))
        # The imported locomotion policy has an effective planar dead-zone
        # around 0.2 m/s.  Preserve direction but keep moving until the base is
        # well inside the station tolerance, then let velocity damping settle.
        if station_dist > 0.08 and 1.0e-6 < planar_command_norm < 0.25:
            cmd[:2] *= 0.25 / planar_command_norm
        return cmd, station_dist, abs(yaw_error)

    def _ee_local_target(self, world_target):
        origin = self.env.get_goal_reference_origin()
        quat = self.env.get_goal_reference_quat()
        target = torch.as_tensor(world_target, device=self.env.device, dtype=torch.float32).reshape(1, 3)
        local = quat_rotate_inverse(quat, target - origin)
        offset = self.env.get_goal_center_offset_local()
        return local - offset

    def _grasp_orientation_delta(self, ee_goal_local, grasp_quat_world):
        """Convert the configured B2Z1 grasp orientation into teleop RPY deltas."""
        desired_world = (
            self.active_grasp_quat.copy()
            if self.active_grasp_quat is not None
            else self._desired_grasp_quat(grasp_quat_world)
        )
        goal_ref_quat = _as_np(self.env.get_goal_reference_quat())[0]
        local_quat = _quat_mul_np(_quat_conjugate_np(goal_ref_quat), desired_world)
        local_rpy = _quat_to_euler_xyz_np(local_quat)
        cart = np.asarray(ee_goal_local, dtype=np.float32).reshape(3)
        radius = max(float(np.linalg.norm(cart)), 1e-8)
        sphere_pitch = math.asin(float(np.clip(cart[2] / radius, -1.0, 1.0)))
        sphere_yaw = math.atan2(float(cart[1]), float(cart[0]))
        return np.array(
            [
                float(((local_rpy[0] + math.pi / 2.0 + math.pi) % (2.0 * math.pi)) - math.pi),
                float(((local_rpy[1] + sphere_pitch - self.env.arm_induced_pitch + math.pi) % (2.0 * math.pi)) - math.pi),
                float(((local_rpy[2] - sphere_yaw + math.pi) % (2.0 * math.pi)) - math.pi),
            ],
            dtype=np.float32,
        )

    def _desired_grasp_quat(self, grasp_quat_world):
        """Orient the real jaws: separation vertical, closing direction at the door.

        In the B2Z1 URDF the open-finger separation is the gripper local +X
        axis.  We map +X to world +Z (upper/lower fingers), and map local +Z
        toward the door so the finger contact faces point along the approach
        normal.  The handle body's quaternion is intentionally ignored: this
        asset's lever axis is vertical and its visual marker has a different
        frame.
        """
        if self.grasp_orientation_quat is not None:
            return self.grasp_orientation_quat.copy()
        del grasp_quat_world
        # Columns are world directions of the measured local EE axes.  The
        # first local axis is the actual two-finger separation measured from
        # the USD body states, not an assumed URDF axis.
        x_local = self.finger_separation_local
        y_local = self.finger_closing_local
        if x_local is None or y_local is None:
            x_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            y_local = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # Either sign is geometrically valid for the two-finger vertical
        # grasp.  The negative sign is the reachable configuration for the
        # B2Z1 reset pose (it avoids an unnecessary large shoulder rotation).
        x_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        y_axis = -np.asarray(self.door_normal, dtype=np.float32)
        y_axis /= max(float(np.linalg.norm(y_axis)), 1.0e-8)
        # Complete both frames with a right-handed third axis.  The source
        # local z is inferred from the measured local separation/closing axes.
        z_local = np.cross(x_local, y_local)
        z_local /= max(float(np.linalg.norm(z_local)), 1.0e-8)
        y_axis = y_axis - x_axis * float(np.dot(y_axis, x_axis))
        y_axis /= max(float(np.linalg.norm(y_axis)), 1.0e-8)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= max(float(np.linalg.norm(z_axis)), 1.0e-8)
        # Rotation mapping local basis to world basis: R = W * L^T.
        local_basis = np.column_stack((x_local, y_local, z_local))
        world_basis = np.column_stack((x_axis, y_axis, z_axis))
        matrix = world_basis @ local_basis.T
        m00, m01, m02 = float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])
        m10, m11, m12 = float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])
        m20, m21, m22 = float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])
        trace = m00 + m11 + m22
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            q = np.array([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s], dtype=np.float32)
        elif m00 > m11 and m00 > m22:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            q = np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s], dtype=np.float32)
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            q = np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s], dtype=np.float32)
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            q = np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s], dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1.0e-8)
        return q

    def _dynamic_grasp_quat(self, state):
        """Build a grasp orientation from the current measured jaw geometry."""
        bodies = _as_np(state.get("robot_body_state_w"))
        ee_pos = _as_np(state.get("ee_pos"))
        ee_orn = _as_np(state.get("ee_orn"))
        if bodies is None or ee_pos is None or ee_orn is None:
            return self.grasp_orientation_quat.copy()
        fixed = bodies[0, self.fixed_finger_body_idx, :3]
        moving = bodies[0, self.moving_finger_body_idx, :3]
        separation = moving - fixed
        if float(np.linalg.norm(separation)) < 1.0e-5:
            return self.grasp_orientation_quat.copy()
        separation /= np.linalg.norm(separation)
        current = ee_orn[0]
        desired = _quat_align_vectors_np(
            separation,
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            current,
        )
        closing = _quat_rotate_np(current, self.open_template["depth_axis"])
        closing[2] = 0.0
        desired_closing = -self.door_normal.copy()
        desired_closing[2] = 0.0
        if np.linalg.norm(closing) > 1.0e-5:
            closing /= np.linalg.norm(closing)
            desired_closing /= max(float(np.linalg.norm(desired_closing)), 1.0e-8)
            angle = math.atan2(
                float(np.cross(closing, desired_closing)[2]),
                float(np.clip(np.dot(closing, desired_closing), -1.0, 1.0)),
            )
            half = 0.5 * angle
            desired = _quat_mul_np(
                np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32),
                desired,
            )
        return desired / max(float(np.linalg.norm(desired)), 1.0e-8)

    @staticmethod
    def _quat_distance(q0, q1):
        q0 = np.asarray(q0, dtype=np.float32)
        q1 = np.asarray(q1, dtype=np.float32)
        q0 /= max(float(np.linalg.norm(q0)), 1e-8)
        q1 /= max(float(np.linalg.norm(q1)), 1e-8)
        return 2.0 * math.acos(float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0)))

    def _slew_orientation(self, target, max_angle=0.025):
        target = np.asarray(target, dtype=np.float32)
        target /= max(float(np.linalg.norm(target)), 1.0e-8)
        if self.ee_orientation_command is None:
            self.ee_orientation_command = target.copy()
            return target.copy()
        current = self.ee_orientation_command / max(float(np.linalg.norm(self.ee_orientation_command)), 1.0e-8)
        dot = float(np.dot(current, target))
        if dot < 0.0:
            target = -target; dot = -dot
        angle = 2.0 * math.acos(float(np.clip(dot, 0.0, 1.0)))
        if angle <= max_angle:
            output = target
        else:
            fraction = max_angle / angle
            theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
            if theta < 1.0e-6:
                output = target
            else:
                output = (math.sin((1.0 - fraction) * theta) * current + math.sin(fraction * theta) * target) / math.sin(theta)
        self.ee_orientation_command = output.astype(np.float32) / max(float(np.linalg.norm(output)), 1.0e-8)
        return self.ee_orientation_command.copy()

    @staticmethod
    def _rotate_about_z(point, origin, angle):
        point = np.asarray(point, dtype=np.float32)
        origin = np.asarray(origin, dtype=np.float32)
        c, s = math.cos(float(angle)), math.sin(float(angle))
        delta = point - origin
        return origin + np.array(
            [c * delta[0] - s * delta[1], s * delta[0] + c * delta[1], delta[2]],
            dtype=np.float32,
        )

    @staticmethod
    def _yaw_quat(angle):
        half = 0.5 * float(angle)
        return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)

    @staticmethod
    def _quat_nlerp(start, end, fraction):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        if float(np.dot(start, end)) < 0.0:
            end = -end
        result = (1.0 - float(fraction)) * start + float(fraction) * end
        return result / max(float(np.linalg.norm(result)), 1.0e-8)

    def _build_open_loop_trajectory(self, state):
        """Precompute every high-level reference before simulation resumes."""
        if self.hinge_world is None or self.hinge_axis_sign is None or self.panel_center_world is None:
            raise RuntimeError("panel and hinge geometry are required for open-loop planning")
        if self.final_min_normal_offset is None:
            raise RuntimeError("gripper collision geometry is required for open-loop planning")

        trajectory = []
        zero_base = np.zeros(3, dtype=np.float32)
        initial_ee = _as_np(state["ee_pos"])[0].copy()
        initial_quat = self.initial_ee_orientation.copy()
        grasp_quat = self.grasp_orientation_quat.copy()
        dt = float(self.env.dt)

        def append_segment(stage, count, start_world, end_world, start_quat, end_quat,
                           start_gripper, end_gripper, base_commands=None,
                           local_override=None, substage=None):
            if count <= 0:
                return
            if base_commands is None:
                base_commands = np.repeat(zero_base[None, :], count, axis=0)
            for index in range(count):
                fraction = 1.0 if count == 1 else index / float(count - 1)
                world = None
                if start_world is not None and end_world is not None:
                    world = ((1.0 - fraction) * start_world + fraction * end_world).astype(np.float32)
                quat = self._quat_nlerp(start_quat, end_quat, fraction)
                trajectory.append({
                    "stage": stage,
                    "substage": substage,
                    "base": np.asarray(base_commands[index], dtype=np.float32).copy(),
                    "ee_world": world,
                    "ee_local": None if local_override is None else np.asarray(local_override, dtype=np.float32).copy(),
                    "quat": quat,
                    "gripper": float((1.0 - fraction) * start_gripper + fraction * end_gripper),
                })

        base = _as_np(state["root_states"])[0]
        initial_yaw = _quat_yaw(base[3:7])
        target_yaw = math.atan2(float(-self.door_normal[1]), float(-self.door_normal[0]))
        yaw_delta = ((target_yaw - initial_yaw + math.pi) % (2.0 * math.pi)) - math.pi
        approach_steps = 45
        weights = np.sin(math.pi * (np.arange(approach_steps, dtype=np.float32) + 0.5) / approach_steps)
        displacement_world = self.station_xy - base[:2]
        c, s = math.cos(initial_yaw), math.sin(initial_yaw)
        displacement_local = np.array(
            [c * displacement_world[0] + s * displacement_world[1],
             -s * displacement_world[0] + c * displacement_world[1]],
            dtype=np.float32,
        )
        # The velocity policy realizes about 70 percent of a short command
        # pulse in this operating region.  This fixed calibration is part of
        # the planned feed-forward trajectory, not an online correction.
        approach_base = np.zeros((approach_steps, 3), dtype=np.float32)
        approach_base[:, :2] = weights[:, None] * displacement_local[None, :] / max(dt * float(weights.sum()) * 0.70, 1.0e-6)
        approach_base[:, 2] = weights * yaw_delta / max(dt * float(weights.sum()) * 0.70, 1.0e-6)
        approach_base[:, :2] = np.clip(approach_base[:, :2], -0.8, 0.8)
        approach_base[:, 2] = np.clip(approach_base[:, 2], -1.0, 1.0)
        append_segment(
            Stage.APPROACH, approach_steps, None, None, initial_quat, initial_quat,
            self.q_open, self.q_open, approach_base, self.default_ee_goal_local,
        )
        append_segment(
            Stage.ALIGN, 20, None, None, initial_quat, initial_quat,
            self.q_open, self.q_open, local_override=self.default_ee_goal_local,
        )

        predicted_ee = initial_ee.copy()
        predicted_ee[:2] += displacement_world
        append_segment(Stage.PREGRASP, 180, predicted_ee, self.center_standoff_ee_world,
                       initial_quat, grasp_quat, self.q_open, self.q_open)
        forward_target = self.grasp_waypoint_world.copy()
        forward_target[2] += 0.010
        append_segment(Stage.CENTER_HANDLE, 100, self.center_standoff_ee_world, forward_target,
                       grasp_quat, grasp_quat, self.q_open, self.q_open,
                       substage="CENTER_FORWARD")
        append_segment(Stage.GRASP_HANDLE, 80, forward_target, forward_target,
                       grasp_quat, grasp_quat, self.q_open, self.q_grasp)

        handle_state = _as_np(state["door_body_state_w"])[0, self.handle_body_idx, :7]
        handle_axle = handle_state[:3].copy()
        turn_gripper = float(self.q_grasp)
        # Closing the fingers deterministically loads the mechanism before
        # this segment starts.  These calibrated offsets keep the first TURN
        # sample continuous with that precomputed closing sequence.
        initial_handle_angle = 0.10
        initial_door_joint_angle = 0.005
        handle_fast_steps = 60
        handle_finish_steps = 100
        handle_target_angle = 0.74
        coupled_turn_steps = 320
        door_open_steps = 160
        release_joint_angle = math.radians(8.0)
        release_angle = self.hinge_axis_sign * release_joint_angle
        handle_angles = np.concatenate(
            [
                np.linspace(
                    initial_handle_angle,
                    0.68,
                    handle_fast_steps,
                    endpoint=False,
                    dtype=np.float32,
                ),
                np.linspace(
                    0.68,
                    handle_target_angle,
                    handle_finish_steps,
                    dtype=np.float32,
                ),
            ]
        )
        handle_angles = np.pad(
            handle_angles,
            (0, coupled_turn_steps - len(handle_angles)),
            mode="edge",
        )
        door_progress = np.minimum(
            np.arange(coupled_turn_steps, dtype=np.float32)
            / float(door_open_steps - 1),
            1.0,
        )
        door_progress = door_progress * door_progress * (3.0 - 2.0 * door_progress)
        turn_angles = self.hinge_axis_sign * (
            initial_door_joint_angle
            + door_progress * (release_joint_angle - initial_door_joint_angle)
        )
        base_follow_delay = 60
        base_turn_angles = np.concatenate(
            [
                np.full(base_follow_delay, turn_angles[0], dtype=np.float32),
                turn_angles[:-base_follow_delay],
            ]
        )
        turn_base_points = np.asarray([
            self._rotate_about_z(
                np.r_[self.station_xy, base[2]], self.hinge_world, angle
            )[:2]
            for angle in base_turn_angles
        ])
        turn_base = self._planned_base_commands(
            turn_base_points, target_yaw + base_turn_angles, dt
        )
        release_grasp = None
        release_quat = None
        for index, (handle_angle, door_angle) in enumerate(
            zip(handle_angles, turn_angles)
        ):
            handle_fraction = float(handle_angle / handle_target_angle)
            handle_quat = np.array(
                [
                    math.cos(0.5 * handle_angle),
                    math.sin(0.5 * handle_angle),
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            # First follow the lever around its axle in the closed-door frame,
            # then carry that complete pose around the door hinge.  The fixed
            # vertical offset is the measured lab30 IK load compensation.
            point_closed = handle_axle + _quat_rotate_np(
                handle_quat, self.grasp_waypoint_world - handle_axle
            )
            point_closed[2] -= 0.052 * handle_fraction
            point = self._rotate_about_z(
                point_closed, self.hinge_world, door_angle
            )
            tangent_compensation = 0.02 * min(index / 20.0, 1.0)
            point += tangent_compensation * self.door_forward
            quat = _quat_mul_np(
                self._yaw_quat(door_angle),
                _quat_mul_np(handle_quat, grasp_quat),
            )
            trajectory.append({
                "stage": Stage.TURN_HANDLE,
                "substage": None,
                "base": turn_base[index],
                "ee_world": point,
                "ee_local": None,
                "quat": quat,
                "gripper": float(
                    (1.0 - handle_fraction) * self.q_grasp
                    + handle_fraction * turn_gripper
                ),
            })
            release_grasp = point
            release_quat = quat

        append_segment(Stage.RELEASE_HANDLE, 50, release_grasp, release_grasp,
                       release_quat, release_quat, turn_gripper, self.q_open)
        release_normal = _quat_rotate_np(self._yaw_quat(release_angle), self.door_normal)
        retract_target = release_grasp + 0.20 * release_normal
        retract_target[2] += 0.10
        append_segment(Stage.RETRACT_ARM, 70, release_grasp, retract_target,
                       release_quat, release_quat, self.q_open, 0.0)
        # Once the gripper is clear of the handle, undo the wrist rotation used
        # to align the jaws with the lever.  Restore it in place before moving
        # the base toward the panel so the orientation target stays continuous.
        append_segment(
            Stage.RETRACT_ARM,
            120,
            retract_target,
            retract_target,
            release_quat,
            initial_quat,
            0.0,
            0.0,
            substage="RESTORE_ORIENTATION",
        )
        align_panel_steps = 120
        align_panel_points = np.linspace(
            self.station_xy, self.panel_station_xy, align_panel_steps, dtype=np.float32
        )
        align_panel_yaws = np.full(align_panel_steps, target_yaw, dtype=np.float32)
        align_panel_base = self._planned_base_commands(
            align_panel_points, align_panel_yaws, dt
        )
        append_segment(
            Stage.ALIGN_PANEL,
            align_panel_steps,
            None,
            None,
            initial_quat,
            initial_quat,
            0.0,
            0.0,
            align_panel_base,
            self.stow_ee_goal_local,
        )

        # Push on the central panel, away from the handle and its return path.
        # The old 35% interpolation left the target close enough to the lever
        # for the fixed finger to strike it during the diagonal approach.
        panel_surface = self.panel_push_world.copy()
        surface_projection = float(np.dot(self.panel_center_world, self.door_normal)) + self.panel_half_thickness
        panel_surface += (surface_projection - float(np.dot(panel_surface, self.door_normal))) * self.door_normal
        panel_tcp = panel_surface.copy()
        # Compensate the low-level arm's roughly 2.5 mm steady-state normal
        # tracking residual.  The 4 mm geometric preload yields about 1.5 mm
        # of physical panel compression before the guarded push starts.
        tcp_projection = surface_projection - self.final_min_normal_offset - 0.004
        panel_tcp += (tcp_projection - float(np.dot(panel_tcp, self.door_normal))) * self.door_normal
        # Closed-door TCP pose whose most forward gripper collider lies on the
        # panel surface.  CONTACT_PANEL approaches this pose from outside.
        self.panel_tcp_world = panel_tcp.copy()
        contact_target = self._rotate_about_z(panel_tcp, self.hinge_world, release_angle)
        panel_push_quat = _quat_mul_np(self._yaw_quat(release_angle), grasp_quat)
        append_segment(Stage.CONTACT_PANEL, 140, retract_target, contact_target,
                       release_quat, panel_push_quat, 0.0, 0.0)

        push_steps = 800
        final_joint_angle = math.radians(85.0)
        final_angle = self.hinge_axis_sign * final_joint_angle
        push_angles = np.linspace(release_angle, final_angle, push_steps, dtype=np.float32)
        push_base_points = np.asarray([
            self._rotate_about_z(np.r_[self.station_xy, base[2]], self.hinge_world, angle)[:2]
            for angle in push_angles
        ])
        push_base = self._planned_base_commands(push_base_points, target_yaw + push_angles, dt)
        for index, angle in enumerate(push_angles):
            trajectory.append({
                "stage": Stage.PUSH_OPEN, "substage": None,
                "base": push_base[index],
                "ee_world": self._rotate_about_z(panel_tcp, self.hinge_world, angle),
                "ee_local": None,
                "quat": _quat_mul_np(self._yaw_quat(angle), grasp_quat),
                "gripper": 0.0,
            })

        through_steps = 240
        through_yaws = np.linspace(target_yaw + final_angle, target_yaw, through_steps, dtype=np.float32)
        through_base = np.zeros((through_steps, 3), dtype=np.float32)
        for index, yaw in enumerate(through_yaws):
            c, s = math.cos(float(yaw)), math.sin(float(yaw))
            velocity_world = 0.45 * self.door_forward[:2]
            through_base[index, :2] = [c * velocity_world[0] + s * velocity_world[1],
                                       -s * velocity_world[0] + c * velocity_world[1]]
        through_base[:, 2] = np.gradient(through_yaws, dt)
        append_segment(Stage.THROUGH, through_steps, None, None, release_quat, initial_quat,
                       0.0, 0.0, through_base, self.default_ee_goal_local)
        append_segment(Stage.RESET_ARM, 100, None, None, initial_quat, initial_quat,
                       0.0, 0.0, local_override=self.default_ee_goal_local)

        self.open_loop_trajectory = trajectory
        self.trajectory_step = 0
        self.stage = trajectory[0]["stage"]
        self.stage_steps = 0
        self.center_substage = trajectory[0]["substage"] or self.center_substage
        stage_counts = {
            stage.name: sum(sample["stage"] == stage for sample in trajectory)
            for stage in Stage if stage not in (Stage.RESET, Stage.SUCCESS, Stage.FAILURE)
        }
        print(
            f"[doorman][plan] precomputed {len(trajectory)} open-loop samples; "
            f"panel_tcp_closed={panel_tcp.round(4).tolist()} "
            f"release_joint_angle_deg={math.degrees(release_joint_angle):.1f} "
            f"world_rotation_deg={math.degrees(release_angle):.1f} "
            f"final_joint_angle_deg={math.degrees(final_joint_angle):.1f} stages={stage_counts}",
            flush=True,
        )

    @staticmethod
    def _planned_base_commands(points_world, yaws, dt):
        points_world = np.asarray(points_world, dtype=np.float32)
        yaws = np.asarray(yaws, dtype=np.float32)
        velocity_world = np.gradient(points_world, float(dt), axis=0)
        yaw_rate = np.gradient(np.unwrap(yaws), float(dt))
        commands = np.zeros((len(points_world), 3), dtype=np.float32)
        for index, yaw in enumerate(yaws):
            c, s = math.cos(float(yaw)), math.sin(float(yaw))
            commands[index, 0] = c * velocity_world[index, 0] + s * velocity_world[index, 1]
            commands[index, 1] = -s * velocity_world[index, 0] + c * velocity_world[index, 1]
            commands[index, 2] = yaw_rate[index]
        commands[:, :2] = np.clip(commands[:, :2], -0.8, 0.8)
        commands[:, 2] = np.clip(commands[:, 2], -1.0, 1.0)
        return commands

    def command(self, state):
        """Publish the next precomputed reference without changing the plan."""
        if not self.open_loop_trajectory:
            raise RuntimeError("open-loop trajectory was not built at reset")
        index = min(self.trajectory_step, len(self.open_loop_trajectory) - 1)
        sample = self.open_loop_trajectory[index]
        self.stage = sample["stage"]
        if sample["substage"] is not None:
            self.center_substage = sample["substage"]
        base_cmd = sample["base"]
        self.active_grasp_quat = sample["quat"]
        if sample["ee_local"] is not None:
            ee_local = torch.as_tensor(
                sample["ee_local"], device=self.env.device, dtype=torch.float32
            ).reshape(1, 3)
        else:
            ee_local = self._ee_local_target(sample["ee_world"])
        orn = self._grasp_orientation_delta(
            ee_local[0].detach().cpu().numpy(), self.active_grasp_quat
        ).reshape(1, 3)
        self.runner.set_teleop_targets(
            torch.as_tensor(base_cmd, device=self.env.device, dtype=torch.float32).reshape(1, 3),
            ee_goal_cart=ee_local,
            ee_orn_delta_rpy=torch.as_tensor(orn, device=self.env.device),
            gripper_target=torch.tensor([[sample["gripper"]]], device=self.env.device),
            ee_goal_orn_quat_world=torch.as_tensor(
                self.active_grasp_quat, device=self.env.device
            ).reshape(1, 4),
            ee_goal_orn_quat_world_enabled=True,
        )
        base = _as_np(state["root_states"])[0]
        station_dist = float(np.linalg.norm(base[:2] - self.station_xy))
        yaw = _quat_yaw(base[3:7])
        target_yaw = math.atan2(float(-self.door_normal[1]), float(-self.door_normal[0]))
        yaw_error = abs(((target_yaw - yaw + math.pi) % (2.0 * math.pi)) - math.pi)
        self.ee_orientation_error = self._quat_distance(
            _as_np(state["ee_orn"])[0], self.active_grasp_quat
        )
        high = np.concatenate(
            [base_cmd, ee_local.detach().cpu().numpy()[0], orn[0], [sample["gripper"]]]
        ).astype(np.float32)
        return high, station_dist, yaw_error

    def _legacy_feedback_command(self, state):
        """Generate the next target for the measured-state feedback controller.

        TURN_HANDLE uses a fixed stage endpoint and rebuilds the complete
        interpolated path from the measured pose every five steps.  Physical
        feedback can end the stage early, but cannot move that endpoint.
        """
        door_bodies = _as_np(state["door_body_state_w"])[0]
        handle = door_bodies[self.handle_body_idx, :3]
        grasp_center = self._physical_handle_grasp_position(door_bodies[self.handle_body_idx, :7])
        # The grasp point comes from grasp_target, but its orientation must
        # follow the physical door_handle body.  The target marker may retain
        # an identity/default rotation even when the lever is mirrored by the
        # door-open direction or has rotated around its X-axis.
        grasp_quat = door_bodies[self.handle_body_idx, 3:7]
        panel = door_bodies[self.panel_body_idx]
        panel_contact = panel[:3] + _quat_rotate_np(panel[3:7], self.panel_contact_local)
        self.handle_pos = handle
        self.grasp_center_pos = grasp_center
        base_cmd, station_dist, yaw_error = self._base_command(state)
        # Keep one fixed target derived from the reset USD collider template.
        # A target recomputed from the moving jaws makes the reference chase
        # the IK error and can drive the arm away from the handle.
        desired_grasp_quat = self._desired_grasp_quat(grasp_quat)
        if self.stage == Stage.CENTER_HANDLE and self.position_only_center:
            desired_grasp_quat = _as_np(state["ee_orn"])[0].copy()
        rotate_for_grasp = self.stage in (
            Stage.PREGRASP,
            Stage.CENTER_HANDLE,
            Stage.GRASP_HANDLE,
            Stage.TURN_HANDLE,
            Stage.RELEASE_HANDLE,
            Stage.RETRACT_ARM,
            Stage.CONTACT_PANEL,
            Stage.PUSH_OPEN,
        )
        desired_orientation = desired_grasp_quat if rotate_for_grasp else self.initial_ee_orientation
        if (
            self.stage == Stage.ALIGN_PANEL
            and self.panel_align_ee_orientation_local is not None
        ):
            goal_ref_quat = _as_np(self.env.get_goal_reference_quat())[0]
            desired_orientation = _quat_mul_np(
                goal_ref_quat, self.panel_align_ee_orientation_local
            )
        if self.stage in (Stage.THROUGH, Stage.RESET_ARM, Stage.SUCCESS):
            desired_orientation = self.initial_ee_orientation
        command_orientation = desired_orientation
        if self.stage == Stage.PREGRASP and self.arm_align_command_quat is not None:
            command_orientation = self.arm_align_command_quat
        elif self.stage in (Stage.CENTER_HANDLE, Stage.GRASP_HANDLE):
            if self.arm_forward_command_quat is not None:
                command_orientation = self.arm_forward_command_quat
        self.active_grasp_quat = self._slew_orientation(command_orientation)
        ee_orientation_error = 999.0
        if state.get("ee_orn") is not None:
            ee_orientation_error = self._quat_distance(
                _as_np(state["ee_orn"])[0], desired_orientation
            )
        self.ee_orientation_error = ee_orientation_error
        if self.stage == Stage.CENTER_HANDLE and self.stage_steps <= 3 and state.get("ee_orn") is not None:
            print(
                f"[doorman] grasp_orn_target={desired_grasp_quat.round(4).tolist()} "
                f"ee_orn_actual={_as_np(state['ee_orn'])[0].round(4).tolist()}",
                flush=True,
            )
        # During base approach keep the arm at its reset pose.  Moving the arm
        # to the handle while the body is still walking caused the old script
        # to skip the physical pre-grasp motion entirely.
        ee_local_override = None
        if self.stage in (Stage.APPROACH, Stage.ALIGN):
            ee_local_override = self.default_ee_goal_local
            ee_target_world = _as_np(state["ee_goal_cart_world"])[0]
        else:
            ee_target_world = grasp_center.copy()
        gripper = float(self.q_open if self.q_open is not None else -0.785)
        if self.stage in (Stage.APPROACH, Stage.ALIGN) or (
            self.stage == Stage.PREGRASP and not self.pregrasp_opening
        ):
            gripper = float(self.q_closed if self.q_closed is not None else 0.0)
        if self.stage == Stage.PREGRASP:
            # One arm-alignment stage: reach the final standoff position and
            # final grasp orientation, then become stationary before moving
            # toward the handle.
            if (
                self.arm_align_command_path is not None
                and self.arm_align_path_index
                < len(self.arm_align_command_path) - 1
            ):
                path_index = min(
                    self.arm_align_path_index,
                    len(self.arm_align_command_path) - 1,
                )
                ee_target_world = self.arm_align_command_path[path_index].copy()
                self.active_grasp_quat = self._slew_orientation(
                    self.arm_align_orientation_path[path_index]
                )
                self.arm_align_path_index = min(
                    path_index + 1, len(self.arm_align_command_path) - 1
                )
            else:
                ee_target_world = (
                    self.arm_align_command_world.copy()
                    if self.arm_align_command_world is not None
                    else self.center_standoff_ee_world.copy()
                )
        elif self.stage == Stage.CENTER_HANDLE:
            # Forward is a separate motion with the jaws fully open.  Its
            # physical endpoint already includes the requested 10 mm depth;
            # the command is the one-shot inverse steady-state solution.
            if self.arm_forward_command_path is not None:
                path_index = min(
                    self.arm_forward_path_index,
                    len(self.arm_forward_command_path) - 1,
                )
                ee_target_world = self.arm_forward_command_path[path_index].copy()
                self.active_grasp_quat = self._slew_orientation(
                    self.arm_forward_orientation_path[path_index]
                )
                self.arm_forward_path_index = min(
                    path_index + 1, len(self.arm_forward_command_path) - 1
                )
            else:
                ee_target_world = self.arm_forward_command_world.copy()
            self.handle_center_target_world = self.arm_forward_target_world.copy()
        elif self.stage == Stage.GRASP_HANDLE:
            # Keep the insertion endpoint unchanged.  The single moving jaw
            # sweeps toward the fixed jaw while closing.  Once it approaches
            # the handle, move the arm toward the fixed jaw so both sides meet
            # actively.  Mirrored grasps therefore lower the arm when the
            # fixed jaw is above the handle and lift it when the fixed jaw is
            # below the handle.
            ee_target_world = self.arm_forward_command_world.copy()
            if self.grasp_latched:
                gripper = float(self.q_grasp)
            else:
                # Close from measured position, slowing sharply near the
                # handle surface.  This remains diameter-independent while
                # avoiding a one-step impact when the contact manifold appears.
                actual_q = float(_as_np(state["gripper_position"])[0, 0])
                if self.grasp_close_command is None:
                    self.grasp_close_command = actual_q
                moving_clearance = float(
                    self.center_metrics.get("moving_handle_clearance", 999.0)
                )
                fixed_clearance = float(
                    self.center_metrics.get("fixed_handle_clearance", 999.0)
                )
                contact_forces = state.get("handle_finger_contact_forces_w")
                contact_norms = (
                    np.linalg.norm(_as_np(contact_forces)[0], axis=-1)
                    if contact_forces is not None
                    else np.zeros(2, dtype=np.float32)
                )
                fixed_contact = bool(
                    contact_norms[0] >= self.grasp_contact_threshold
                )
                moving_contact = bool(
                    contact_norms[1] >= self.grasp_contact_threshold
                )
                fixed_center = np.asarray(
                    self.center_metrics.get("fixed_center"), dtype=np.float32
                )
                moving_center = np.asarray(
                    self.center_metrics.get("moving_center"), dtype=np.float32
                )
                fixed_is_upper = bool(
                    fixed_center.shape == (3,)
                    and moving_center.shape == (3,)
                    and np.isfinite(fixed_center[2])
                    and np.isfinite(moving_center[2])
                    and fixed_center[2] > moving_center[2]
                )
                balance_direction = -1.0 if fixed_is_upper else 1.0
                if (
                    not self.grasp_vertical_balance_started
                    and (
                        moving_clearance <= 0.003
                        or moving_contact
                    )
                ):
                    fixed_gap = max(
                        float(
                            self.center_metrics.get(
                                "upper_gap" if fixed_is_upper else "lower_gap",
                                0.0,
                            )
                        ),
                        0.0,
                    )
                    self.grasp_vertical_lift_target = float(
                        balance_direction * np.clip(fixed_gap, 0.0, 0.010)
                    )
                    self.grasp_vertical_balance_started = True
                    print(
                        f"[doorman][grasp-balance] "
                        f"fixed_is_upper={fixed_is_upper} "
                        f"fixed_gap={fixed_gap:.4f} "
                        f"vertical_target={self.grasp_vertical_lift_target:.4f}",
                        flush=True,
                    )
                if moving_contact != fixed_contact:
                    self.grasp_vertical_lift_target = float(
                        np.clip(
                            self.grasp_vertical_lift_target
                            + balance_direction * 0.00005,
                            -0.010,
                            0.010,
                        )
                    )
                lift_delta = np.clip(
                    self.grasp_vertical_lift_target
                    - self.grasp_vertical_lift,
                    -0.0002,
                    0.0002,
                )
                self.grasp_vertical_lift += float(lift_delta)
                if moving_clearance > 0.008:
                    close_step = 0.016
                    command_lead = 0.080
                elif moving_clearance > 0.003:
                    close_step = 0.012
                    command_lead = 0.045
                else:
                    close_step = 0.008
                    command_lead = 0.024
                if moving_contact:
                    # A single moving-jaw contact is the start of capture, not
                    # completion.  Keep closing quickly until the fixed jaw is
                    # geometrically close, then cushion only the final contact.
                    if fixed_contact or fixed_clearance <= 0.004:
                        close_step = 0.004
                        command_lead = 0.012
                    else:
                        close_step = 0.012
                        command_lead = 0.045
                if float(np.max(contact_norms)) > 50.0:
                    close_step = -0.001
                    command_lead = -0.001
                next_command = min(
                    self.grasp_close_command + close_step,
                    actual_q + command_lead,
                )
                self.grasp_close_command = float(
                    np.clip(next_command, self.q_open, self.q_closed)
                )
                gripper = self.grasp_close_command
            ee_target_world[2] += self.grasp_vertical_lift
        elif self.stage == Stage.TURN_HANDLE:
            # Follow the fixed stage-entry handle-rotation trajectory.
            ee_target_world, self.active_grasp_quat = self._turn_path_target(state)
            gripper = float(self.q_grasp)
        elif self.stage == Stage.RELEASE_HANDLE:
            if self.release_substage == "RETURN_HANDLE":
                return_fraction = float(
                    np.clip(self.stage_steps / self.release_return_steps, 0.0, 1.0)
                )
                desired_handle_angle = (
                    (1.0 - return_fraction) * self.release_start_handle_angle
                )
                if return_fraction >= 1.0:
                    current_handle_angle = float(
                        _as_np(state["door_joint_position"])[0, self.handle_joint_idx]
                    )
                    desired_handle_angle -= float(
                        np.clip(current_handle_angle, -0.06, 0.06)
                    )
                return_delta = desired_handle_angle - self.release_start_handle_angle
                return_rotation = _quat_axis_angle_np(
                    self.release_return_axis_world, return_delta
                )
                ee_target_world = (
                    self.release_return_axle_world
                    + _quat_rotate_np(
                        return_rotation, self.release_return_offset_world
                    )
                )
                self.active_grasp_quat = _quat_mul_np(
                    return_rotation, self.release_return_start_quat
                )
                gripper = float(self.q_grasp)
            elif self.release_substage == "OPEN_AND_CLEAR":
                open_steps = max(0, self.stage_steps - self.release_open_start_step)
                gripper_fraction = float(np.clip(open_steps / 35.0, 0.0, 1.0))
                if (
                    self.release_open_command_path is None
                    or self.release_open_orientation_path is None
                ):
                    raise RuntimeError("release open inverse path is unavailable")
                path_index = min(open_steps, len(self.release_open_command_path) - 1)
                ee_target_world = self.release_open_command_path[path_index].copy()
                self.active_grasp_quat = self.release_open_orientation_path[
                    path_index
                ].copy()
                gripper = (1.0 - gripper_fraction) * float(self.q_grasp) + gripper_fraction * float(self.q_open)
            elif self.release_substage == "RETREAT":
                if (
                    self.release_retreat_command_path is None
                    or self.release_retreat_orientation_path is None
                ):
                    raise RuntimeError("release retreat inverse path is unavailable")
                path_index = min(
                    self.release_retreat_path_index,
                    len(self.release_retreat_command_path) - 1,
                )
                ee_target_world = self.release_retreat_command_path[
                    path_index
                ].copy()
                self.active_grasp_quat = self.release_retreat_orientation_path[
                    path_index
                ].copy()
                self.release_retreat_path_index = min(
                    path_index + 1,
                    len(self.release_retreat_command_path) - 1,
                )
                gripper = float(self.q_open)
            else:
                raise RuntimeError(
                    f"Unsupported release substage: {self.release_substage}"
                )
        elif self.stage == Stage.RETRACT_ARM:
            # Hold the measured retreat endpoint while restoring orientation
            # and closing.  Position and orientation commands both come from
            # the same steady-state inverse path so wrist rotation cannot
            # pull the physical EE back toward the handle.
            if (
                self.release_restore_command_path is None
                or self.release_restore_orientation_path is None
            ):
                raise RuntimeError("release restore inverse path is unavailable")
            restore_fraction = float(
                np.clip(self.stage_steps / 50.0, 0.0, 1.0)
            )
            path_index = min(
                self.stage_steps, len(self.release_restore_command_path) - 1
            )
            ee_target_world = self.release_restore_command_path[path_index].copy()
            self.active_grasp_quat = self.release_restore_orientation_path[
                path_index
            ].copy()
            gripper = (
                (1.0 - restore_fraction) * float(self.q_open)
                + restore_fraction * float(self.q_closed)
            )
        elif self.stage == Stage.ALIGN_PANEL:
            # Carry the post-release arm pose with the trunk while the base
            # moves laterally.  ALIGN_PANEL must not introduce another arm
            # retreat or an unsolicited stow motion.
            if self.panel_align_ee_goal_local is None:
                self.panel_align_ee_goal_local = _as_np(
                    state["ee_goal_cart_local"]
                )[0].copy()
            ee_local_override = self.panel_align_ee_goal_local
            # The arm pose remains unchanged, but close the jaws after they
            # have cleared the lever so the open finger span does not rake
            # across the panel during the lateral base translation.
            gripper = 0.0
        elif self.stage in (Stage.CONTACT_PANEL, Stage.PUSH_OPEN):
            if self.stage == Stage.PUSH_OPEN and self.panel_tcp_world is not None:
                current_hinge_joint = float(
                    _as_np(state["door_joint_position"])[0, self.hinge_idx]
                )
                hinge_velocity = float(
                    _as_np(state["door_joint_velocity"])[0, self.hinge_idx]
                )
                current_hinge_world = self.hinge_axis_sign * current_hinge_joint
                if self.push_orientation is None:
                    self.push_orientation = self.active_grasp_quat.copy()
                if self.push_substage == "ARM_TRACK":
                    remaining = max(
                        0.0, self.push_goal_hinge_angle - current_hinge_joint
                    )
                    desired_speed = min(
                        self.push_target_angular_speed, 0.8 * remaining
                    )
                    # Velocity feedback becomes a small angular lead/lag for
                    # the Cartesian impedance target.  Positive error pushes;
                    # an over-speeding panel drives the target behind the
                    # measured panel and lets the arm absorb its momentum.
                    angular_offset = np.clip(
                        0.25 * (desired_speed - hinge_velocity),
                        -math.radians(2.0),
                        math.radians(2.0),
                    )
                    target_hinge_joint = float(
                        np.clip(
                            current_hinge_joint + angular_offset,
                            self.push_start_hinge_angle - math.radians(2.0),
                            self.push_goal_hinge_angle,
                        )
                    )
                    self.push_command_hinge_angle = target_hinge_joint
                    target_hinge_delta_world = self.hinge_axis_sign * (
                        target_hinge_joint - self.push_start_hinge_angle
                    )
                    ee_target_world = self._rotate_about_z(
                        self.push_contact_world,
                        self.hinge_world,
                        target_hinge_delta_world,
                    )
                    # Tangential lead controls door angular speed.  Normal
                    # contact is controlled separately from the measured
                    # finger collision hulls so a detached arm cannot build
                    # centimetres of tracking error and then strike the panel.
                    finger_clearance = self._panel_finger_clearance(
                        state, current_hinge_joint
                    )
                    previous_clearance = self.push_previous_finger_clearance
                    if previous_clearance is None or not np.isfinite(finger_clearance):
                        approach_speed = 0.0
                    else:
                        approach_speed = (
                            finger_clearance - previous_clearance
                        ) / float(self.runner.env.dt)
                    self.push_previous_finger_clearance = finger_clearance
                    finger_forces = _as_np(state.get("finger_contact_forces_w"))
                    finger_force = (
                        float(np.linalg.norm(finger_forces[0], axis=-1).max())
                        if finger_forces is not None
                        else 0.0
                    )
                    desired_clearance = self.push_nominal_finger_clearance
                    if finger_force > 20.0:
                        desired_clearance += min(
                            0.030, 0.00050 * (finger_force - 20.0)
                        )
                    if np.isfinite(finger_clearance):
                        # At most 0.3 mm inward per 20 ms control cycle.  The
                        # resulting 1.5 cm/s recontact speed stays in effect
                        # until the real collider cloud reaches the panel.
                        desired_clearance = max(
                            desired_clearance,
                            finger_clearance - self.push_recontact_step,
                        )
                        # The panel may be the object closing the gap, for
                        # example while springing back after contact loss.  In
                        # that case an inward-only rate limit cannot prevent a
                        # collision: command enough outward clearance to
                        # absorb 250 ms of measured relative closing motion.
                        closing_speed = max(0.0, -approach_speed - 0.015)
                        desired_clearance = max(
                            desired_clearance,
                            finger_clearance
                            + min(0.040, 0.25 * closing_speed),
                        )
                    reference_clearance = (
                        self.push_contact_reference_clearance
                        if self.push_contact_reference_clearance is not None
                        else 0.0
                    )
                    current_normal = _quat_rotate_np(
                        self._yaw_quat(current_hinge_world), self.door_normal
                    )
                    ee_target_world += (
                        desired_clearance - reference_clearance
                    ) * current_normal
                    self.push_finger_clearance = finger_clearance
                    self.push_finger_approach_speed = approach_speed
                    self.push_target_finger_clearance = desired_clearance
                    self.push_finger_force = finger_force
                    self.push_contact_retreat_active = bool(
                        approach_speed < -0.03 or finger_force > 20.0
                    )
                    self.active_grasp_quat = _quat_mul_np(
                        self._yaw_quat(
                            self.hinge_axis_sign
                            * (target_hinge_joint - self.push_start_hinge_angle)
                        ),
                        self.push_orientation,
                    )
                elif self.push_substage == "ARM_RELEASE":
                    release_start = (
                        self.push_release_start_world
                        if self.push_release_start_world is not None
                        else _as_np(state["ee_pos"])[0]
                    )
                    release_normal = (
                        self.push_release_normal_world
                        if self.push_release_normal_world is not None
                        else _quat_rotate_np(
                            self._yaw_quat(current_hinge_world),
                            self.door_normal,
                        )
                    )
                    release_steps = max(
                        0, self.stage_steps - self.push_release_start_step
                    )
                    release_fraction = float(
                        np.clip(release_steps / 50.0, 0.0, 1.0)
                    )
                    ee_target_world = (
                        release_start
                        + 0.10 * release_fraction * release_normal
                        + np.array(
                            [0.0, 0.0, 0.02 * release_fraction],
                            dtype=np.float32,
                        )
                    )
                    self.active_grasp_quat = self.push_orientation.copy()
                else:
                    raise RuntimeError(
                        f"Unsupported arm push substage: {self.push_substage}"
                    )
            else:
                current_hinge_joint = float(
                    _as_np(state["door_joint_position"])[0, self.hinge_idx]
                )
                current_hinge_world = self.hinge_axis_sign * current_hinge_joint
                current_normal = _quat_rotate_np(
                    self._yaw_quat(current_hinge_world), self.door_normal
                )
                # CONTACT_PANEL starts while the released door is still
                # settling.  Track the preselected panel-local point until
                # contact is established; freezing its world pose at stage
                # entry leaves the hand in empty space as the door returns.
                # PUSH_OPEN snapshots its own contact frame after the guarded
                # handoff, so no feedback replanning is introduced there.
                if (
                    self.panel_contact_target_world is None
                    or not self.panel_contact_established
                ):
                    contact_tcp = self._rotate_about_z(
                        self.panel_tcp_world, self.hinge_world, current_hinge_world
                    )
                    self.panel_contact_target_world = (
                        contact_tcp
                        + self.panel_contact_clearance * current_normal
                    )
                    self.panel_contact_orientation = _quat_mul_np(
                        self._yaw_quat(current_hinge_world),
                        self.grasp_orientation_quat,
                    )
                clearance = (
                    self.panel_contact_clearance
                    if self.panel_contact_established
                    else self.panel_precontact_clearance
                )
                if not self.panel_alignment_established:
                    clearance = 0.10
                if self.panel_recovery_steps > 0:
                    clearance = max(clearance, 0.025)
                ee_target_world = (
                    self.panel_contact_target_world
                    + (clearance - self.panel_contact_clearance) * current_normal
                )
                if self.panel_contact_orientation is not None:
                    self.active_grasp_quat = self.panel_contact_orientation.copy()
            gripper = 0.0
        elif self.stage == Stage.THROUGH:
            # Keep the arm compact while walking through; the explicit reset
            # stage below verifies the final default pose after passage.
            ee_local_override = self.stow_ee_goal_local
            gripper = 0.0
        elif self.stage in (Stage.RESET_ARM, Stage.SUCCESS):
            ee_local_override = self.default_ee_goal_local
            gripper = 0.0

        # Keep the reset-pose override exact, but slew every Cartesian target
        # used for physical interaction so the arm approaches the handle and
        # panel continuously instead of teleporting its IK goal.
        follows_precomputed_inverse_path = bool(
            self.stage == Stage.CENTER_HANDLE
            and self.arm_forward_command_path is not None
        )
        if follows_precomputed_inverse_path:
            self.ee_command_world = ee_target_world.copy()
        elif ee_local_override is None and self.stage != Stage.TURN_HANDLE:
            if self.stage == Stage.CONTACT_PANEL:
                max_step = 0.0015 if self.panel_alignment_established else 0.003
            elif self.stage == Stage.PUSH_OPEN:
                if (
                    self.push_substage == "ARM_TRACK"
                    and self.push_contact_retreat_active
                ):
                    max_step = 0.004
                else:
                    max_step = 0.0015 if self.push_substage == "ARM_TRACK" else 0.002
            else:
                max_step = 0.002 if self.stage == Stage.CENTER_HANDLE else 0.012
            ee_target_world = self._slew_ee_target(ee_target_world, max_step=max_step)
        else:
            self.ee_command_world = _as_np(state["ee_pos"])[0].copy()

        if self.stage == Stage.TURN_HANDLE:
            joints = _as_np(state["door_joint_position"])[0]
            latch_released = float(joints[self.latch_joint_idx]) >= 0.02
            hinge_now = float(joints[self.hinge_idx])
            hinge_goal = self.turn_target_hinge_angle
            pushing_to_handoff = (
                latch_released
                and hinge_goal is not None
                and abs(hinge_now - self.turn_start_hinge_angle)
                < abs(hinge_goal - self.turn_start_hinge_angle) * 0.98
            )
            base_cmd = np.zeros(3, dtype=np.float32)
            if self.turn_base_start_xy is None:
                self.turn_base_start_xy = _as_np(state["root_states"])[0, :2].copy()
            # Keep the base fixed through initial lever engagement.  Once the
            # arm is well into the fixed turn trajectory, unfold a slow,
            # preplanned 80 mm forward translation over four seconds.  Yaw is
            # held fixed; the base only restores arm workspace near the lower
            # end of the lever arc.
            assist_start_step = 240
            assist_duration_steps = 200
            if self.stage_steps >= assist_start_step:
                fraction = min(
                    1.0,
                    (self.stage_steps - assist_start_step)
                    / float(assist_duration_steps),
                )
                turn_base_target = (
                    self.turn_base_start_xy
                    + (0.08 * fraction) * self.door_forward[:2]
                )
                target_yaw = math.atan2(
                    float(self.door_forward[1]), float(self.door_forward[0])
                )
                base_cmd = self._pose_command(
                    state, turn_base_target, target_yaw
                )
                base_cmd[:2] = np.clip(base_cmd[:2], -0.04, 0.04)
                base_cmd[2] = 0.0
            if pushing_to_handoff:
                travel_cmd = self._travel_command(state, 0.35)
                base_cmd[:2] = travel_cmd[:2]
        elif self.stage == Stage.PUSH_OPEN:
            base_cmd = (
                self._arm_push_base_command(state)
                if self.push_substage == "ARM_TRACK"
                else np.zeros(3, dtype=np.float32)
            )
        elif self.stage == Stage.ALIGN_PANEL:
            panel_station_xy, panel_station_yaw = self._current_panel_station(state)
            base_cmd = self._pose_command(
                state,
                panel_station_xy,
                panel_station_yaw,
            )
        elif self.stage == Stage.THROUGH:
            through_ramp = min(1.0, self.stage_steps / 40.0)
            base_cmd = self._travel_command(state, 0.20 + 0.35 * through_ramp)
        elif self.stage not in (Stage.APPROACH, Stage.ALIGN):
            base_cmd[:2] = 0.0
            if self.stage != Stage.ALIGN:
                base_cmd[2] = 0.0

        # Hold the accepted station against arm reaction forces while the
        # measured base itself remains stationary.
        if self.stage in (
            Stage.PREGRASP,
            Stage.CENTER_HANDLE,
            Stage.GRASP_HANDLE,
            Stage.TURN_HANDLE,
        ):
            base_cmd, _, _ = self._base_command(state)
            base_cmd[:2] = np.clip(base_cmd[:2], -0.25, 0.25)
            base_cmd[2] = float(np.clip(base_cmd[2], -0.40, 0.40))
        elif self.stage in (
            Stage.CENTER_HANDLE,
            Stage.GRASP_HANDLE,
            Stage.RELEASE_HANDLE,
            Stage.RETRACT_ARM,
            Stage.CONTACT_PANEL,
            Stage.RESET_ARM,
        ):
            base_cmd[:] = 0.0

        # Dampen residual base momentum at the TURN->RELEASE handoff.  This
        # brakes the robot in place; it is not a door-pushing command.
        if self.stage == Stage.RELEASE_HANDLE and self.stage_steps < 20:
            base = _as_np(state["root_states"])[0]
            yaw = _quat_yaw(base[3:7])
            c, s = math.cos(yaw), math.sin(yaw)
            vel = base[7:10]
            local_vel = np.array(
                [c * vel[0] + s * vel[1], -s * vel[0] + c * vel[1]],
                dtype=np.float32,
            )
            base_cmd[:2] = np.clip(-1.5 * local_vel, -0.25, 0.25)
            base_cmd[2] = np.clip(-0.8 * float(base[12]), -0.25, 0.25)

        ee_local = (
            torch.as_tensor(ee_local_override, device=self.env.device, dtype=torch.float32).reshape(1, 3)
            if ee_local_override is not None
            else self._ee_local_target(ee_target_world)
        )
        orn = np.zeros((1, 3), dtype=np.float32)
        use_world_orientation = self.stage not in (Stage.RESET_ARM, Stage.SUCCESS)
        if state.get("ee_orn") is not None:
            self.ee_orientation_error = self._quat_distance(
                _as_np(state["ee_orn"])[0], self.active_grasp_quat
            )
        if use_world_orientation:
            orn[0] = self._grasp_orientation_delta(
                ee_local[0].detach().cpu().numpy(), self.active_grasp_quat
            )
        else:
            orn[0] = self.default_ee_goal_delta_rpy
        self.runner.set_teleop_targets(
            torch.as_tensor(base_cmd, device=self.env.device).reshape(1, 3),
            ee_goal_cart=ee_local,
            ee_orn_delta_rpy=torch.as_tensor(orn, device=self.env.device),
            gripper_target=torch.tensor([[gripper]], device=self.env.device),
            ee_goal_orn_quat_world=torch.as_tensor(
                self.active_grasp_quat, device=self.env.device
            ).reshape(1, 4),
            ee_goal_orn_quat_world_enabled=use_world_orientation,
        )
        high = np.concatenate([base_cmd, ee_local.detach().cpu().numpy()[0], orn[0], [gripper]]).astype(np.float32)
        return high, station_dist, yaw_error

    def _slew_ee_target(self, target, max_step=0.012):
        target = np.asarray(target, dtype=np.float32)
        if self.ee_command_world is None:
            self.ee_command_world = target.copy()
        delta = target - self.ee_command_world
        distance = float(np.linalg.norm(delta))
        if distance > max_step:
            delta *= max_step / distance
        self.ee_command_world = self.ee_command_world + delta
        return self.ee_command_world.copy()

    def _travel_command(self, state, speed):
        """Convert a world-frame door-through direction into base commands."""
        base = _as_np(state["root_states"])[0]
        yaw = _quat_yaw(base[3:7])
        c, s = math.cos(yaw), math.sin(yaw)
        direction = self.door_forward[:2]
        local_x = c * direction[0] + s * direction[1]
        local_y = -s * direction[0] + c * direction[1]
        target_yaw = math.atan2(float(direction[1]), float(direction[0]))
        yaw_error = ((target_yaw - yaw + math.pi) % (2 * math.pi)) - math.pi
        return np.array(
            [speed * local_x, speed * local_y, np.clip(1.2 * yaw_error, -0.6, 0.6)],
            dtype=np.float32,
        )

    def _pose_command(self, state, target_xy, target_yaw):
        """Drive to a world-frame planar pose through the low-level policy."""
        base = _as_np(state["root_states"])[0]
        yaw = _quat_yaw(base[3:7])
        error_world = np.asarray(target_xy, dtype=np.float32) - base[:2]
        c, s = math.cos(yaw), math.sin(yaw)
        command = np.array(
            [
                c * error_world[0] + s * error_world[1],
                -s * error_world[0] + c * error_world[1],
                ((target_yaw - yaw + math.pi) % (2.0 * math.pi)) - math.pi,
            ],
            dtype=np.float32,
        )
        planar_norm = float(np.linalg.norm(command[:2]))
        if planar_norm <= 0.02:
            command[:2] = 0.0
        elif planar_norm > 0.0:
            # Treat planar position error as a proportional velocity command.
            # The previous 0.06--0.21 m branch normalized every command to
            # 0.21 m/s, so the base repeatedly overshot the target before
            # falling abruptly into a 6 cm dead band.
            command_speed = min(0.21, 1.5 * planar_norm)
            measured_speed = float(np.linalg.norm(base[7:9]))
            if planar_norm > 0.08 and measured_speed < 0.03:
                # The imported locomotion policy has a load-dependent planar
                # dead zone near 0.2 m/s.  Only boost a command after measured
                # motion has stalled outside the acceptance radius; retain
                # proportional slowdown once the base is close enough.
                command_speed = 0.21
            command[:2] *= command_speed / planar_norm
        command[2] = np.clip(1.2 * command[2], -0.6, 0.6)
        return command

    def _current_panel_station(self, state):
        """Return the base station facing the panel at its current hinge pose."""
        hinge = float(_as_np(state["door_joint_position"])[0, self.hinge_idx])
        panel_angle = self.hinge_axis_sign * hinge
        station_reference = np.array(
            [self.panel_station_xy[0], self.panel_station_xy[1], self.hinge_world[2]],
            dtype=np.float32,
        )
        station_world = self._rotate_about_z(
            station_reference, self.hinge_world, panel_angle
        )
        panel_forward = _quat_rotate_np(
            self._yaw_quat(panel_angle), self.door_forward
        )
        target_yaw = math.atan2(
            float(panel_forward[1]), float(panel_forward[0])
        )
        return station_world[:2], target_yaw

    def _arm_push_base_command(self, state):
        """Advance along door-forward while keeping both front shoulders clear."""
        if (
            self.push_base_start_xy is None
            or self.push_panel_contact_start_world is None
        ):
            return np.zeros(3, dtype=np.float32)
        base = _as_np(state["root_states"])[0]
        hinge = float(
            _as_np(state["door_joint_position"])[0, self.hinge_idx]
        )
        current_contact = self._rotate_about_z(
            self.push_contact_world,
            self.hinge_world,
            self.hinge_axis_sign * (hinge - self.push_start_hinge_angle),
        )
        forward = np.asarray(self.door_forward[:2], dtype=np.float32)
        panel_forward_displacement = max(
            0.0,
            float(
                np.dot(
                    current_contact[:2]
                    - self.push_panel_contact_start_world[:2],
                    forward,
                )
            ),
        )
        target_xy = (
            self.push_base_start_xy
            + panel_forward_displacement * forward
        )
        forward_error = max(
            0.0, float(np.dot(target_xy - base[:2], forward))
        )
        if forward_error <= 0.025:
            forward_speed = 0.0
        else:
            forward_speed = min(0.18, 1.2 * forward_error)
            measured_speed = float(np.linalg.norm(base[7:9]))
            if forward_error > 0.08 and measured_speed < 0.03:
                forward_speed = 0.21
        # Do not move the support platform while the arm is detached or
        # rapidly closing its panel-normal gap.  Base assistance resumes once
        # the arm is back in a controlled near-contact state.
        if (
            np.isfinite(self.push_finger_clearance)
            and self.push_finger_clearance > 0.012
        ) or (
            np.isfinite(self.push_finger_approach_speed)
            and self.push_finger_approach_speed < -0.03
        ) or (
            self.push_finger_force > 10.0
        ):
            forward_speed = 0.0

        # The contact point is farther from the hinge than the near shoulder,
        # so blindly copying its forward displacement can sweep that shoulder
        # into the smaller-radius part of the panel.  Treat the current panel
        # as a finite XY segment and conservatively sweep both measured front
        # hip positions through the command horizon.  Keeping the panel fixed
        # over this prediction is conservative because a push-opening panel
        # rotates away from the robot.
        closed_free_end = 2.0 * self.panel_center_world - self.hinge_world
        panel_free_end = self._rotate_about_z(
            closed_free_end,
            self.hinge_world,
            self.hinge_axis_sign * hinge,
        )
        panel_start = np.asarray(self.hinge_world[:2], dtype=np.float32)
        panel_end = np.asarray(panel_free_end[:2], dtype=np.float32)
        shoulders = _as_np(state.get("robot_body_state_w"))
        if shoulders is None:
            raise RuntimeError(
                "robot_body_state_w is required for shoulder-safe push assistance"
            )
        shoulder_xy = shoulders[0, self.front_shoulder_body_indices, :2]
        collision_envelope = (
            self.push_shoulder_collision_radius + self.panel_half_thickness
        )

        def shoulder_clearance(advance):
            points = shoulder_xy + float(advance) * forward[None, :]
            segment = panel_end - panel_start
            segment_norm_sq = float(np.dot(segment, segment))
            if segment_norm_sq < 1.0e-10:
                center_distances = np.linalg.norm(points - panel_start, axis=1)
            else:
                projection = np.clip(
                    ((points - panel_start) @ segment) / segment_norm_sq,
                    0.0,
                    1.0,
                )
                closest = panel_start + projection[:, None] * segment[None, :]
                center_distances = np.linalg.norm(points - closest, axis=1)
            return float(np.min(center_distances) - collision_envelope)

        current_clearance = shoulder_clearance(0.0)
        measured_forward_speed = max(
            0.0, float(np.dot(np.asarray(base[7:9], dtype=np.float32), forward))
        )
        braking_distance = (
            0.15 * measured_forward_speed
            + measured_forward_speed * measured_forward_speed / (2.0 * 0.60)
        )
        command_horizon_distance = forward_speed * 0.45
        predicted_advance = max(braking_distance, command_horizon_distance)
        advances = np.linspace(0.0, predicted_advance, 17, dtype=np.float32)
        clearances = np.asarray(
            [shoulder_clearance(advance) for advance in advances],
            dtype=np.float32,
        )
        predicted_clearance = float(np.min(clearances))
        unsafe = np.flatnonzero(clearances < self.push_shoulder_safety_margin)
        swept_scale = 1.0
        if unsafe.size:
            first_unsafe = int(unsafe[0])
            swept_scale = max(0.0, (first_unsafe - 1) / max(len(advances) - 1, 1))
        clearance_scale = float(
            np.clip(
                (current_clearance - self.push_shoulder_safety_margin)
                / self.push_shoulder_slowdown_band,
                0.0,
                1.0,
            )
        )
        self.push_shoulder_min_clearance = current_clearance
        self.push_shoulder_predicted_min_clearance = predicted_clearance
        self.push_base_safety_scale = min(swept_scale, clearance_scale)
        forward_speed *= self.push_base_safety_scale
        # Tiny commands sit below the walking policy's useful range and add
        # foot motion without creating clearance.  Stop and let the arm rotate
        # the door farther away before trying the same forward-only move again.
        if forward_speed < 0.08:
            forward_speed = 0.0

        yaw = _quat_yaw(base[3:7])
        c, s = math.cos(yaw), math.sin(yaw)
        velocity_world = forward_speed * forward
        return np.array(
            [
                c * velocity_world[0] + s * velocity_world[1],
                -s * velocity_world[0] + c * velocity_world[1],
                0.0,
            ],
            dtype=np.float32,
        )

    def _body_push_command(self, state, forward_speed):
        """Push forward while rejecting lateral sliding along the panel."""
        base = _as_np(state["root_states"])[0]
        yaw = _quat_yaw(base[3:7])
        hinge = float(_as_np(state["door_joint_position"])[0, self.hinge_idx])
        panel_center = self._rotate_about_z(
            self.panel_center_world,
            self.hinge_world,
            self.hinge_axis_sign * hinge,
        )
        velocity_world = np.asarray(self.door_forward[:2], dtype=np.float32) * float(
            forward_speed
        )
        velocity_world[1] += np.clip(
            1.5 * float(panel_center[1] - base[1]), -0.30, 0.30
        )
        c, s = math.cos(yaw), math.sin(yaw)
        target_yaw = math.atan2(float(self.door_forward[1]), float(self.door_forward[0]))
        yaw_error = ((target_yaw - yaw + math.pi) % (2.0 * math.pi)) - math.pi
        return np.array(
            [
                c * velocity_world[0] + s * velocity_world[1],
                -s * velocity_world[0] + c * velocity_world[1],
                np.clip(1.2 * yaw_error, -0.6, 0.6),
            ],
            dtype=np.float32,
        )

    def _push_arc_command(self, state, speed):
        """Follow the panel contact point's circular path about the hinge."""
        base = _as_np(state["root_states"])[0]
        yaw = _quat_yaw(base[3:7])
        hinge_angle = float(_as_np(state["door_joint_position"])[0, self.hinge_idx])
        if self.panel_tcp_world is None:
            return self._travel_command(state, speed)
        panel_point = self._rotate_about_z(
            self.panel_tcp_world,
            self.hinge_world,
            self.hinge_axis_sign * hinge_angle,
        )
        radius = panel_point[:2] - self.hinge_world[:2]
        tangent = self.hinge_axis_sign * np.array(
            [-radius[1], radius[0]], dtype=np.float32
        )
        tangent /= max(float(np.linalg.norm(tangent)), 1.0e-8)
        c, s = math.cos(yaw), math.sin(yaw)
        local_x = c * tangent[0] + s * tangent[1]
        local_y = -s * tangent[0] + c * tangent[1]
        target_yaw = math.atan2(float(tangent[1]), float(tangent[0]))
        yaw_error = ((target_yaw - yaw + math.pi) % (2 * math.pi)) - math.pi
        return np.array(
            [speed * local_x, speed * local_y, np.clip(0.8 * yaw_error, -0.35, 0.35)],
            dtype=np.float32,
        )

    def _reset_turn_path(self):
        """Invalidate the rolling path whenever TURN_HANDLE is left."""
        self.turn_path_world = None
        self.turn_path_quat = None
        self.turn_path_index = 0
        self.turn_path_stage = None

    def _ensure_turn_endpoint(self, state):
        """Create TURN_HANDLE's fixed geometric endpoint exactly once.

        The endpoint is the original open-loop handle-rotation construction:
        rotate the grasp offset around the physical handle axle by the
        preselected target handle angle.  Feedback may change the path start,
        but never this endpoint.
        """
        if self.turn_endpoint_world is not None and self.turn_endpoint_quat is not None:
            return
        door_bodies = _as_np(state["door_body_state_w"])[0]
        handle_state = door_bodies[self.handle_body_idx, :7]
        # The revolute handle joint is defined at the handle body's axle, as
        # in the original open-loop construction.  Do not rotate around the
        # grasp point itself.
        axle = handle_state[:3].copy()
        panel_state = door_bodies[self.panel_body_idx, :7]
        axis = _quat_rotate_np(
            panel_state[3:7], self.handle_joint_axis_panel_local
        )
        axis /= max(float(np.linalg.norm(axis)), 1.0e-8)
        start_pos = _as_np(state["ee_pos"])[0].copy()
        start_quat = _as_np(state["ee_orn"])[0].copy()
        current_angle = float(_as_np(state["door_joint_position"])[0, self.handle_joint_idx])
        self.turn_start_handle_angle = current_angle
        # This is the fixed local goal inherited from the original trajectory.
        self.turn_target_handle_angle = max(float(self.turn_target_handle_angle or 0.74), current_angle)
        handle_delta = self.turn_target_handle_angle - current_angle
        current_hinge = float(_as_np(state["door_joint_position"])[0, self.hinge_idx])
        self.turn_start_hinge_angle = current_hinge
        # Keep the latch depressed until the randomized door edge has fully
        # cleared the frame.  Plan beyond this episode's geometric threshold;
        # the margin absorbs steady Cartesian tracking error at the endpoint.
        if self.turn_gap_angle is None:
            raise RuntimeError("door gap angle is unavailable")
        gap_delta = self.hinge_axis_sign * (
            self.turn_gap_angle + math.radians(2.0)
        )
        self.turn_target_hinge_angle = gap_delta
        self.turn_handle_axis_world = axis.copy()
        self.turn_axle_world = axle.copy()
        self.turn_grasp_offset_world = start_pos - axle
        self.turn_grasp_orientation = start_quat.copy()
        current_goal = _as_np(state.get("ee_goal_cart_world"))
        if current_goal is not None:
            self.turn_command_bias_world = current_goal[0] - start_pos
        else:
            self.turn_command_bias_world[:] = 0.0
        handle_endpoint = axle + _quat_rotate_np(
            _quat_axis_angle_np(axis, handle_delta), self.turn_grasp_offset_world
        )
        self.turn_endpoint_world = self._rotate_about_z(
            handle_endpoint, self.hinge_world, gap_delta
        )
        self.turn_endpoint_quat = _quat_mul_np(
            self._yaw_quat(gap_delta),
            _quat_mul_np(
                _quat_axis_angle_np(axis, handle_delta), self.turn_grasp_orientation
            ),
        )
        print(
            f"[doorman][turn] fixed endpoint angle={self.turn_target_handle_angle:.4f} "
            f"hinge_target={self.turn_target_hinge_angle:.4f} "
            f"start_angle={current_angle:.4f} axle={axle.round(4).tolist()} "
            f"endpoint={self.turn_endpoint_world.round(4).tolist()}",
            flush=True,
        )

    def _replan_turn_path(self, state):
        """Generate a complete handle-axis arc to the fixed endpoint."""
        self._ensure_turn_endpoint(state)
        start = _as_np(state["ee_pos"])[0].copy()
        start_q = _as_np(state["ee_orn"])[0].copy()
        current_angle = float(_as_np(state["door_joint_position"])[0, self.handle_joint_idx])
        remaining_angle = max(0.0, self.turn_target_handle_angle - current_angle)
        current_hinge = float(_as_np(state["door_joint_position"])[0, self.hinge_idx])
        current_hinge_delta = self.hinge_axis_sign * (
            current_hinge - self.turn_start_hinge_angle
        )
        remaining_hinge = (
            self.turn_target_hinge_angle - current_hinge_delta
            if self.turn_gap_started
            else 0.0
        )
        distance = float(np.linalg.norm(self.turn_endpoint_world - start))
        angle = self._quat_distance(start_q, self.turn_endpoint_quat)
        count = max(120, int(math.ceil(distance / 0.004)) + 1, int(math.ceil(angle / 0.03)) + 1)
        fractions = np.linspace(0.0, 1.0, count, dtype=np.float32)
        nominal_handle_current = self.turn_axle_world + _quat_rotate_np(
            _quat_axis_angle_np(
                self.turn_handle_axis_world,
                current_angle - self.turn_start_handle_angle,
            ),
            self.turn_grasp_offset_world,
        )
        nominal_current = self._rotate_about_z(
            nominal_handle_current,
            self.hinge_world,
            current_hinge_delta,
        )
        tracking_offset = start - nominal_current
        path = []
        orientations = []
        for fraction in fractions:
            planned_angle = (
                current_angle - self.turn_start_handle_angle
                + float(fraction) * remaining_angle
            )
            nominal_handle = self.turn_axle_world + _quat_rotate_np(
                _quat_axis_angle_np(self.turn_handle_axis_world, planned_angle),
                self.turn_grasp_offset_world,
            )
            planned_hinge = (
                current_hinge_delta
                + float(fraction) * remaining_hinge
            )
            nominal = self._rotate_about_z(
                nominal_handle, self.hinge_world, planned_hinge
            )
            # Start at the measured EE pose, converge continuously onto the
            # exact handle/hinge arc while retaining the steady-state command
            # bias measured at the accepted grasp.
            path.append(
                nominal
                + self.turn_command_bias_world
                + (1.0 - float(fraction)) * tracking_offset
            )
            orientations.append(
                _quat_mul_np(
                    self._yaw_quat(planned_hinge),
                    _quat_mul_np(
                        _quat_axis_angle_np(
                            self.turn_handle_axis_world, planned_angle
                        ),
                        self.turn_grasp_orientation,
                    ),
                )
            )
        self.turn_path_world = np.asarray(path, dtype=np.float32)
        self.turn_path_quat = np.asarray(orientations, dtype=np.float32)
        # Index zero is the measured pose itself; command the next sample so
        # replanning does not insert a one-step pause every five frames.
        self.turn_path_index = 1
        self.turn_path_stage = self.stage
        self.turn_path_replan_count += 1

    def _turn_path_target(self, state):
        """Return the next sample from the fixed stage-entry turn path."""
        latch_position = float(
            _as_np(state["door_joint_position"])[0, self.latch_joint_idx]
        )
        if not self.turn_gap_started and latch_position >= 0.02:
            self.turn_gap_started = True
            self._reset_turn_path()
            print(
                f"[doorman][turn] latch released at {latch_position:.4f} m; "
                "starting door-gap arc",
                flush=True,
            )
        if (
            self.turn_path_stage != self.stage
            or self.turn_path_world is None
        ):
            self._replan_turn_path(state)
        index = min(self.turn_path_index, len(self.turn_path_world) - 1)
        target = self.turn_path_world[index].copy()
        quat = self.turn_path_quat[index].copy()
        self.turn_path_index = min(self.turn_path_index + 1, len(self.turn_path_world) - 1)
        return target, quat

    def _effective_grasp_target(self, grasp_center, ee_quat=None):
        target = np.asarray(grasp_center, dtype=np.float32).copy()
        if ee_quat is None:
            offset_world = getattr(self, "grasp_ee_offset_world", np.zeros(3, dtype=np.float32))
        else:
            offset_world = _quat_rotate_np(ee_quat, self.grasp_ee_offset_local)
        target -= offset_world
        return target

    def _steady_compensated_ee_command(
        self, state, desired_position_world, desired_orientation_world, label
    ):
        """Invert the mathematical steady EE map once at a stable stage boundary."""
        dof_names = list(state["dof_names"])
        dof_positions = _as_np(state["dof_pos"])[0]
        joint_positions = {
            name: float(dof_positions[index])
            for index, name in enumerate(dof_names)
        }
        root = _as_np(state["root_states"])[0]
        result = self.ee_steady_model.inverse(
            desired_position_world,
            desired_orientation_world,
            base_position_world=root[:3],
            base_orientation_world_wxyz=root[3:7],
            joint_positions=joint_positions,
        )
        print(
            f"[doorman][steady-map] {label} desired="
            f"{np.asarray(desired_position_world).round(5).tolist()} command="
            f"{result.command_position_world.round(5).tolist()} compensation_mm="
            f"{((result.command_position_world - desired_position_world) * 1000.0).round(3).tolist()} "
            f"predicted={result.forward_result.position_world.round(5).tolist()} "
            f"predicted_error_mm={(result.position_error * 1000.0).round(3).tolist()} "
            f"predicted_q={result.forward_result.arm_joint_positions.round(5).tolist()} "
            f"inverse_nfev={result.solver_iterations}",
            flush=True,
        )
        return (
            result.command_position_world.astype(np.float32),
            result.command_orientation_world_wxyz.astype(np.float32),
        )

    def _build_steady_compensated_path(
        self,
        state,
        start_position_world,
        end_position_world,
        orientation_world,
        *,
        start_orientation_world=None,
        label="arm_forward_path",
        command_bias_world=None,
    ):
        """Map a straight measured-EE path through the mathematical inverse."""
        start = np.asarray(start_position_world, dtype=np.float64)
        end = np.asarray(end_position_world, dtype=np.float64)
        end_orientation = np.asarray(orientation_world, dtype=np.float64)
        start_orientation = (
            end_orientation
            if start_orientation_world is None
            else np.asarray(start_orientation_world, dtype=np.float64)
        )
        orientation_distance = 2.0 * math.acos(
            float(np.clip(abs(np.dot(start_orientation, end_orientation)), 0.0, 1.0))
        )
        count = max(
            1,
            int(math.ceil(float(np.linalg.norm(end - start)) / 0.002)),
            int(math.ceil(orientation_distance / 0.03)),
        )
        dof_names = list(state["dof_names"])
        dof_positions = _as_np(state["dof_pos"])[0]
        joint_positions = {
            name: float(dof_positions[index])
            for index, name in enumerate(dof_names)
        }
        root = _as_np(state["root_states"])[0]
        command_positions = []
        command_orientations = []
        for fraction in np.linspace(1.0 / count, 1.0, count):
            desired = (1.0 - fraction) * start + fraction * end
            desired_orientation = self._quat_nlerp(
                start_orientation, end_orientation, float(fraction)
            )
            result = self.ee_steady_model.inverse(
                desired,
                desired_orientation,
                base_position_world=root[:3],
                base_orientation_world_wxyz=root[3:7],
                joint_positions=joint_positions,
            )
            command_position = result.command_position_world
            if command_bias_world is not None:
                command_position = command_position + np.asarray(
                    command_bias_world, dtype=np.float64
                )
            command_positions.append(command_position.astype(np.float32))
            command_orientations.append(
                result.command_orientation_world_wxyz.astype(np.float32)
            )
            for name, value in zip(
                self.ee_steady_model.arm_joint_names,
                result.forward_result.arm_joint_positions,
                strict=True,
            ):
                joint_positions[name] = float(value)
        print(
            f"[doorman][steady-map] {label} samples={count} "
            f"actual_start={start.round(5).tolist()} actual_end={end.round(5).tolist()} "
            f"command_end={command_positions[-1].round(5).tolist()}",
            flush=True,
        )
        return (
            np.asarray(command_positions, dtype=np.float32),
            np.asarray(command_orientations, dtype=np.float32),
        )

    def _build_release_inverse_path(
        self,
        state,
        start_position_world,
        end_position_world,
        start_orientation_world,
        end_orientation_world,
        start_gripper_position,
        end_gripper_position,
        steps,
        label,
    ):
        """Invert a measured-EE release path, including changing jaw load."""
        start = np.asarray(start_position_world, dtype=np.float64)
        end = np.asarray(end_position_world, dtype=np.float64)
        start_orientation = np.asarray(start_orientation_world, dtype=np.float64)
        end_orientation = np.asarray(end_orientation_world, dtype=np.float64)
        dof_names = list(state["dof_names"])
        dof_positions = _as_np(state["dof_pos"])[0]
        joint_positions = {
            name: float(dof_positions[index])
            for index, name in enumerate(dof_names)
        }
        gripper_joint_id = int(self.env.gripper_joint_ids[0].item())
        gripper_joint_name = dof_names[gripper_joint_id]
        root = _as_np(state["root_states"])[0]
        command_positions = []
        command_orientations = []
        predicted_positions = []

        for fraction in np.linspace(0.0, 1.0, int(steps) + 1):
            desired = (1.0 - fraction) * start + fraction * end
            desired_orientation = self._quat_nlerp(
                start_orientation, end_orientation, float(fraction)
            )
            joint_positions[gripper_joint_name] = (
                (1.0 - fraction) * float(start_gripper_position)
                + fraction * float(end_gripper_position)
            )
            result = self.ee_steady_model.inverse(
                desired,
                desired_orientation,
                base_position_world=root[:3],
                base_orientation_world_wxyz=root[3:7],
                joint_positions=joint_positions,
                position_tolerance_m=self.pregrasp_position_tolerance,
            )
            command_positions.append(
                result.command_position_world.astype(np.float32)
            )
            command_orientations.append(
                result.command_orientation_world_wxyz.astype(np.float32)
            )
            predicted_positions.append(result.forward_result.position_world.copy())
            for name, value in zip(
                self.ee_steady_model.arm_joint_names,
                result.forward_result.arm_joint_positions,
                strict=True,
            ):
                joint_positions[name] = float(value)

        predicted_positions = np.asarray(predicted_positions)
        prediction_error = predicted_positions - np.linspace(
            start, end, int(steps) + 1
        )
        command_positions = np.asarray(command_positions, dtype=np.float32)
        print(
            f"[doorman][steady-map] {label} "
            f"samples={len(command_positions)} "
            f"actual_start={start.round(5).tolist()} "
            f"actual_end={end.round(5).tolist()} "
            f"command_start={command_positions[0].round(5).tolist()} "
            f"command_end={command_positions[-1].round(5).tolist()} "
            f"command_delta_mm={((command_positions[-1] - command_positions[0]) * 1000.0).round(3).tolist()} "
            f"max_prediction_error_mm={float(np.max(np.linalg.norm(prediction_error, axis=1))) * 1000.0:.4f}",
            flush=True,
        )
        return (
            command_positions,
            np.asarray(command_orientations, dtype=np.float32),
        )

    def _arm_pose_is_stable(self, state):
        """Require an unchanged measured EE pose over 0.3 seconds."""
        position = _as_np(state["ee_pos"])[0].copy()
        orientation = _as_np(state["ee_orn"])[0].copy()
        self.arm_stability_window.append((position, orientation))
        if len(self.arm_stability_window) < self.arm_stability_window.maxlen:
            return False
        positions = np.asarray([sample[0] for sample in self.arm_stability_window])
        orientations = [sample[1] for sample in self.arm_stability_window]
        position_span = float(
            np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1))
        )
        orientation_span = max(
            self._quat_distance(orientation, sample) for sample in orientations
        )
        return bool(
            position_span <= 0.0003
            and orientation_span <= 0.003
        )

    def advance(self, state, station_dist, yaw_error):
        """Advance only the trajectory clock; measured state never retimes it."""
        previous_stage = self.stage
        door_bodies = _as_np(state["door_body_state_w"])[0]
        grasp_center = self._physical_handle_grasp_position(
            door_bodies[self.handle_body_idx, :7]
        )
        if self.stage in (Stage.CENTER_HANDLE, Stage.GRASP_HANDLE):
            self.center_metrics = self._geometry_metrics(state, grasp_center)
        handle_contacts = state.get("handle_finger_contact_forces_w")
        contact_points = state.get("handle_finger_contact_points_w")
        if handle_contacts is not None and contact_points is not None:
            norms = np.linalg.norm(_as_np(handle_contacts)[0], axis=-1)
            points = _as_np(contact_points)[0]
            present = np.asarray([
                norms[index] >= self.grasp_contact_threshold
                and self._contact_is_on_handle_capsule(
                    points[index], door_bodies[self.handle_body_idx, :7]
                )
                for index in range(2)
            ])
            if np.all(present):
                self.grasp_bilateral_contact_seen = True

        joints = _as_np(state["door_joint_position"])[0]
        self.turn_last_handle_angle = float(joints[self.handle_joint_idx])
        self.turn_last_hinge_angle = float(joints[self.hinge_idx])
        self.trajectory_step += 1
        self.stage_steps += 1
        if self.trajectory_step >= len(self.open_loop_trajectory):
            self.stage = Stage.SUCCESS
            self.stage_steps = 0
            return self.stage

        next_sample = self.open_loop_trajectory[self.trajectory_step]
        self.stage = next_sample["stage"]
        if next_sample["substage"] is not None:
            self.center_substage = next_sample["substage"]
        if self.stage != previous_stage:
            print(
                f"[doorman][plan] stage {previous_stage.name} -> {self.stage.name} "
                f"at trajectory_step={self.trajectory_step} "
                f"hinge={self.turn_last_hinge_angle:.3f} "
                f"handle_joint={self.turn_last_handle_angle:.3f} "
                f"station_dist={station_dist:.3f} yaw_error={yaw_error:.3f}",
                flush=True,
            )
            self.stage_steps = 0
        return self.stage

    def _legacy_feedback_advance(self, state, station_dist, yaw_error):
        """Advance stages exclusively from measured physical conditions."""
        previous_stage = self.stage
        hinge = float(_as_np(state["door_joint_position"])[0, self.hinge_idx])
        handle_joint = float(_as_np(state["door_joint_position"])[0, self.handle_joint_idx])
        latch_joint = float(_as_np(state["door_joint_position"])[0, self.latch_joint_idx])
        ee = _as_np(state["ee_pos"])[0]
        door_bodies = _as_np(state["door_body_state_w"])[0]
        handle_position = door_bodies[self.handle_body_idx, :3]
        grasp_center = self._physical_handle_grasp_position(
            door_bodies[self.handle_body_idx, :7]
        )
        panel = door_bodies[self.panel_body_idx]
        panel_contact = panel[:3] + _quat_rotate_np(panel[3:7], self.panel_contact_local)
        pregrasp_dist = (
            float(np.linalg.norm(ee - self.center_standoff_ee_world))
            if ee is not None and self.center_standoff_ee_world is not None
            else 999.0
        )
        grasp_target = self._effective_grasp_target(grasp_center)
        ee_dist = float(np.linalg.norm(ee - grasp_target)) if ee is not None else 999.0
        ee_xy_dist = float(np.linalg.norm((ee - grasp_target)[:2])) if ee is not None else 999.0
        ee_z_dist = abs(float(ee[2] - grasp_target[2])) if ee is not None else 999.0
        panel_dist = float(np.linalg.norm(ee - panel_contact)) if ee is not None else 999.0
        gripper_pos = float(_as_np(state["gripper_position"])[0, 0])
        gripper_contact = state.get("gripper_contact_force_w")
        grasp_contact_force = (
            float(np.linalg.norm(_as_np(gripper_contact)[0]))
            if gripper_contact is not None
            else 0.0
        )
        finger_contacts = state.get("finger_contact_forces_w")
        finger_contact_norms = np.zeros(2, dtype=np.float32)
        if finger_contacts is not None:
            finger_contact_norms = np.linalg.norm(_as_np(finger_contacts)[0], axis=-1)
        handle_contacts = state.get("handle_finger_contact_forces_w")
        handle_contact_norms = np.zeros(2, dtype=np.float32)
        handle_contact_present = np.zeros(2, dtype=bool)
        handle_points = None
        if handle_contacts is not None:
            handle_contact_norms = np.linalg.norm(_as_np(handle_contacts)[0], axis=-1)
            handle_points = state.get("handle_finger_contact_points_w")
            handle_points = _as_np(handle_points)[0] if handle_points is not None else None
            if handle_points is None:
                handle_contact_norms[:] = 0.0
            else:
                for finger_idx in range(2):
                    on_capsule = self._contact_is_on_handle_capsule(
                        handle_points[finger_idx], door_bodies[self.handle_body_idx, :7]
                    )
                    force_present = (
                        handle_contact_norms[finger_idx]
                        >= self.grasp_contact_threshold
                    )
                    handle_contact_present[finger_idx] = on_capsule and force_present
                    if not handle_contact_present[finger_idx]:
                        handle_contact_norms[finger_idx] = 0.0
        speed = float(np.linalg.norm(_as_np(state["root_states"])[0, 7:10]))
        base = _as_np(state["root_states"])[0]
        base_xy = base[:2]
        ee_base_local = _quat_rotate_inverse_np(base[3:7], ee - base[:3])
        reset_arm_dist = float(np.linalg.norm(ee_base_local - self.default_ee_base_local))
        stow_arm_dist = (
            float(np.linalg.norm(ee_base_local - self.stow_ee_base_local))
            if self.stow_ee_base_local is not None
            else 999.0
        )
        stow_arm_compact = bool(
            ee_base_local[0] < 0.42
            and abs(float(ee_base_local[1])) < 0.12
            and 0.20 < ee_base_local[2] < 0.45
        )
        ee_orientation_error = float(self.ee_orientation_error)
        metrics = self._geometry_metrics(state, grasp_center)
        if self.stage in (Stage.CENTER_HANDLE, Stage.GRASP_HANDLE):
            self.center_metrics = metrics
        if self.stage == Stage.APPROACH and station_dist < self.station_tolerance and yaw_error < 0.10 and speed < 0.03:
            self.stage = Stage.ALIGN; self.stage_steps = 0
            self.align_settle_steps = 0
        elif self.stage == Stage.ALIGN:
            angular_speed = float(
                np.linalg.norm(_as_np(state["root_states"])[0, 10:13])
            )
            base_stable = bool(
                station_dist < self.station_tolerance
                and yaw_error < 0.08
                and speed < 0.02
                and angular_speed < 0.05
            )
            self.align_settle_steps = (
                self.align_settle_steps + 1 if base_stable else 0
            )
            if self.align_settle_steps >= 20:
                (
                    self.arm_align_command_path,
                    self.arm_align_orientation_path,
                ) = self._build_steady_compensated_path(
                    state,
                    ee,
                    self.center_standoff_ee_world,
                    self.grasp_orientation_quat,
                    start_orientation_world=_as_np(state["ee_orn"])[0],
                    label="arm_align_path",
                )
                self.arm_align_path_index = 0
                self.arm_align_command_world = self.arm_align_command_path[-1].copy()
                self.arm_align_command_quat = self.arm_align_orientation_path[-1].copy()
                self.arm_align_nominal_command_world = (
                    self.arm_align_command_world.copy()
                )
                self.arm_align_geometry_corrections = 0
                self.arm_stability_window.clear()
                self.stage = Stage.PREGRASP; self.stage_steps = 0
        elif self.stage == Stage.PREGRASP:
            arm_stable = self._arm_pose_is_stable(state)
            align_path_complete = bool(
                self.arm_align_command_path is not None
                and self.arm_align_path_index
                >= len(self.arm_align_command_path) - 1
            )
            pocket_error = np.asarray(
                metrics.get("pocket_center", grasp_center), dtype=np.float32
            ) - grasp_center
            tangent_error = pocket_error - float(
                np.dot(pocket_error, self.door_normal)
            ) * self.door_normal
            # Height is anchored by the fixed-jaw geometry above.  The closed
            # reset template must not move that target when centering laterally.
            tangent_error[2] = 0.0
            tangent_error_norm = float(np.linalg.norm(tangent_error))
            if tangent_error_norm > 0.012:
                tangent_error *= 0.012 / tangent_error_norm
            if (
                align_path_complete
                and arm_stable
                and tangent_error_norm > 0.003
                and self.arm_align_geometry_corrections < 1
            ):
                self.center_standoff_ee_world = (
                    self.center_standoff_ee_world - tangent_error
                ).astype(np.float32)
                self.center_insert_waypoint_world = (
                    self.center_insert_waypoint_world - tangent_error
                ).astype(np.float32)
                self.grasp_waypoint_world = (
                    self.grasp_waypoint_world - tangent_error
                ).astype(np.float32)
                (
                    self.arm_align_command_world,
                    self.arm_align_command_quat,
                ) = self._steady_compensated_ee_command(
                    state,
                    self.center_standoff_ee_world,
                    self.grasp_orientation_quat,
                    "arm_align_geometry",
                )
                self.arm_align_nominal_command_world = (
                    self.arm_align_command_world.copy()
                )
                self.arm_command_bias_world[:] = 0.0
                self.arm_align_geometry_corrections += 1
                self.arm_stability_window.clear()
                self.pregrasp_settle_steps = 0
                arm_stable = False
                print(
                    "[doorman][align-geometry] correction_mm="
                    f"{(-tangent_error * 1000.0).round(3).tolist()}",
                    flush=True,
                )
            steady_error = ee - self.center_standoff_ee_world
            if (
                align_path_complete
                and arm_stable
                and float(np.linalg.norm(steady_error)) > 0.003
                and self.arm_align_residual_corrections < 1
                and not np.any(
                    handle_contact_norms >= self.grasp_contact_threshold
                )
            ):
                correction = steady_error.astype(np.float32)
                correction_norm = float(np.linalg.norm(correction))
                if correction_norm > 0.015:
                    correction *= 0.015 / correction_norm
                self.arm_align_command_world = (
                    self.arm_align_command_world - correction
                ).astype(np.float32)
                self.arm_command_bias_world = (
                    self.arm_align_command_world
                    - self.arm_align_nominal_command_world
                ).astype(np.float32)
                self.arm_align_residual_corrections += 1
                self.arm_stability_window.clear()
                self.pregrasp_settle_steps = 0
                arm_stable = False
                print(
                    "[doorman][steady-map] residual_correction "
                    f"measured_error_mm={(steady_error * 1000.0).round(3).tolist()} "
                    f"count={self.arm_align_residual_corrections}",
                    flush=True,
                )
            arm_aligned = (
                align_path_complete
                and
                pregrasp_dist <= self.pregrasp_position_tolerance
                and ee_orientation_error <= self.grasp_orientation_tolerance
                and station_dist < self.station_tolerance
                and yaw_error < 0.08
                and speed < 0.02
                and not np.any(handle_contact_norms >= self.grasp_contact_threshold)
                and arm_stable
            )
            if not self.pregrasp_opening:
                self.pregrasp_settle_steps = (
                    self.pregrasp_settle_steps + 1 if arm_aligned else 0
                )
                if self.pregrasp_settle_steps >= 5:
                    self.pregrasp_opening = True
                    self.gripper_open_settle = 0
                    print(
                        "[doorman][pregrasp] arm aligned and stable; opening gripper",
                        flush=True,
                    )
            gripper_joint_id = int(self.env.gripper_joint_ids[0].item())
            actual_q = float(_as_np(state["gripper_position"])[0, 0])
            gripper_speed = abs(
                float(_as_np(state["dof_vel"])[0, gripper_joint_id])
            )
            gripper_open_stable = bool(
                self.pregrasp_opening
                and arm_aligned
                and abs(actual_q - self.q_open) < 0.02
                and gripper_speed < 0.05
            )
            self.gripper_open_settle = (
                self.gripper_open_settle + 1 if gripper_open_stable else 0
            )
            if self.gripper_open_settle >= 10:
                self.arm_forward_target_world = (
                    self.grasp_waypoint_world - 0.010 * self.door_normal
                ).astype(np.float32)
                (
                    self.arm_forward_command_path,
                    self.arm_forward_orientation_path,
                ) = self._build_steady_compensated_path(
                    state,
                    ee,
                    self.arm_forward_target_world,
                    self.grasp_orientation_quat,
                    command_bias_world=self.arm_command_bias_world,
                )
                self.arm_forward_path_index = 0
                self.arm_forward_command_world = (
                    self.arm_forward_command_path[-1].copy()
                )
                self.arm_forward_command_quat = (
                    self.arm_forward_orientation_path[-1].copy()
                )
                self.center_substage = "CENTER_FORWARD"
                self.arm_stability_window.clear()
                self.stage = Stage.CENTER_HANDLE; self.stage_steps = 0
        elif self.stage == Stage.CENTER_HANDLE:
            if self.q_open is None:
                self.failure_reason = "geometry_q_open_unavailable"
                self.stage = Stage.FAILURE
            elif self.arm_forward_target_world is None:
                self.failure_reason = "arm_forward_target_unavailable"
                self.stage = Stage.FAILURE
            elif metrics.get("valid") and metrics["panel_clearance"] < 0.005:
                self.failure_reason = (
                    f"center_panel_clearance={metrics['panel_clearance']:.4f}"
                )
                self.stage = Stage.FAILURE
            else:
                actual_q = float(_as_np(state["gripper_position"])[0, 0])
                forward_distance = float(
                    np.linalg.norm(ee - self.arm_forward_target_world)
                )
                arm_stable = self._arm_pose_is_stable(state)
                forward_ready = bool(
                    forward_distance <= 0.012
                    and ee_orientation_error <= self.grasp_orientation_tolerance
                    and abs(actual_q - self.q_open) < 0.02
                    and self._center_capture_ready(metrics)
                    and speed < 0.02
                    and arm_stable
                )
                self.handle_align_settle_steps = (
                    self.handle_align_settle_steps + 1 if forward_ready else 0
                )
                if self.handle_align_settle_steps >= 5:
                    self.stage = Stage.GRASP_HANDLE; self.stage_steps = 0
                    self.grasp_settle_steps = 0
                    self.grasp_close_command = actual_q
        elif self.stage == Stage.GRASP_HANDLE:
            if metrics.get("valid") and metrics["panel_clearance"] < 0.005:
                self.failure_reason = f"grasp_panel_clearance={metrics['panel_clearance']:.4f}"
                self.stage = Stage.FAILURE
            for finger_idx, present in enumerate(handle_contact_present):
                if present:
                    self.handle_contact_last_seen[finger_idx] = self.stage_steps
                    if handle_points is not None:
                        self.handle_contact_last_points[finger_idx] = handle_points[finger_idx]
            # GPU PhysX may alternate the two active contact manifolds across
            # adjacent frames even while a lightly loaded grasp is static.
            # Count only force-qualified handle contacts above, but accept each
            # side being observed within a short physical-time window.
            bilateral_recent = bool(
                np.all(self.stage_steps - self.handle_contact_last_seen <= 5)
            )
            opposed_grasp, opposed_metrics = self._contacts_form_opposed_grasp(
                self.handle_contact_last_points,
                door_bodies[self.handle_body_idx, :7],
            )
            if (
                not self.grasp_latched
                and bilateral_recent
                and opposed_grasp
            ):
                # Preserve a small force preload, but derive the actual hold
                # angle from this handle instead of assuming its diameter.
                self.q_grasp = float(
                    np.clip(
                        gripper_pos + self.grasp_preload_angle,
                        self.q_open,
                        self.q_closed,
                    )
                )
                self.grasp_latched = True
                self.grasp_settle_steps = 0
                print(
                    f"[doorman][grasp] latched q_actual={gripper_pos:.4f} "
                    f"q_hold={self.q_grasp:.4f} "
                    f"fixed_clearance={metrics['fixed_handle_clearance']:.4f} "
                    f"moving_clearance={metrics['moving_handle_clearance']:.4f} "
                    f"contact_axial_delta={opposed_metrics['axial_delta']:.4f} "
                    f"contact_closing={opposed_metrics['closing_coordinates'].round(4).tolist()} "
                    f"forces={handle_contact_norms.round(3).tolist()}",
                    flush=True,
                )
            peak_handle_force = float(np.max(handle_contact_norms))
            if self.grasp_latched and peak_handle_force > 50.0:
                # Relax only the measured hold angle until the force returns to
                # a usable range; never substitute a diameter-specific angle.
                self.q_grasp = float(
                    np.clip(self.q_grasp - 0.001, self.q_open, self.q_closed)
                )
                self.grasp_settle_steps = 0
            joint_id = int(self.env.gripper_joint_ids[0].item())
            gripper_speed = abs(float(_as_np(state["dof_vel"])[0, joint_id]))
            grasp_stable = (
                self.grasp_latched
                and bilateral_recent
                and opposed_grasp
                and gripper_speed < 0.08
                and peak_handle_force <= 50.0
            )
            self.grasp_settle_steps = (
                self.grasp_settle_steps + 1 if grasp_stable else 0
            )
            if self.grasp_settle_steps >= 10:
                self.grasp_bilateral_contact_seen = True
            if self.grasp_settle_steps >= 10:
                self.turn_start_handle_angle = handle_joint
                self.turn_start_latch_position = latch_joint
                self.turn_start_hinge_angle = hinge
                self.turn_last_handle_angle = handle_joint
                self.turn_last_hinge_angle = hinge
                self.turn_target_handle_angle = 0.74
                self.turn_endpoint_world = None
                self.turn_endpoint_quat = None
                self.turn_path_world = None
                self.turn_path_quat = None
                self.turn_path_index = 0
                self.turn_path_stage = None
                self.turn_base_start_xy = base_xy.copy()
                self.stage = Stage.TURN_HANDLE; self.stage_steps = 0
        elif self.stage == Stage.TURN_HANDLE:
            # Reaching the latch release is only an intermediate condition.
            # Keep the bilateral grasp and continue the same geometric path
            # until the door has been actively opened past the latch-clearance
            # handoff target.
            latch_released = latch_joint >= 0.02
            if self.turn_gap_angle is None:
                raise RuntimeError("door gap angle is unavailable")
            door_gap_open = abs(hinge) >= self.turn_gap_angle
            # GRASP_HANDLE already requires ten consecutive bilateral-contact
            # frames before TURN_HANDLE can start.  Near the end of the fixed
            # lever arc one pad naturally unloads as the latch bottoms out;
            # requiring bilateral contact again here traps a physically
            # completed turn at the endpoint.  Keep requiring contact with
            # the handle so an accidental door collision cannot pass.
            grasp_retained = (
                self.grasp_bilateral_contact_seen
                and bool(np.any(handle_contact_present))
            )
            if latch_released and door_gap_open and grasp_retained:
                self._reset_turn_path()
                self.release_start_world = ee.copy()
                self.release_start_handle_angle = handle_joint
                self.release_start_hinge_angle = hinge
                handle_state = door_bodies[self.handle_body_idx, :7]
                panel_state = door_bodies[self.panel_body_idx, :7]
                release_axis = _quat_rotate_np(
                    panel_state[3:7], self.handle_joint_axis_panel_local
                )
                release_axis /= max(float(np.linalg.norm(release_axis)), 1.0e-8)
                self.release_return_axle_world = handle_state[:3].copy()
                self.release_return_axis_world = release_axis
                self.release_return_offset_world = (
                    ee - self.release_return_axle_world
                )
                self.release_return_start_quat = _as_np(state["ee_orn"])[0].copy()
                self.release_vertical_delta = -float(self.grasp_vertical_lift)
                self.release_substage = "RETURN_HANDLE"
                self.ee_command_world = ee.copy()
                self.release_stable_steps = 0
                self.stage = Stage.RELEASE_HANDLE; self.stage_steps = 0
        if self.stage == Stage.TURN_HANDLE:
            self.turn_last_handle_angle = handle_joint
            self.turn_last_hinge_angle = hinge
        elif self.stage == Stage.RELEASE_HANDLE:
            release_contact_clear = not np.any(handle_contact_present)
            gripper_open = abs(gripper_pos - float(self.q_open)) < 0.08
            if self.release_substage == "RETURN_HANDLE":
                handle_horizontal = abs(handle_joint) < 0.05
                self.release_stable_steps = (
                    self.release_stable_steps + 1
                    if self.stage_steps >= self.release_return_steps
                    and handle_horizontal
                    else 0
                )
                if self.release_stable_steps >= 10:
                    self.release_substage = "OPEN_AND_CLEAR"
                    self.release_open_start_world = ee.copy()
                    self.release_open_start_step = self.stage_steps
                    self.release_start_quat = _as_np(state["ee_orn"])[0].copy()
                    release_open_end_world = (
                        self.release_open_start_world
                        + np.array(
                            [0.0, 0.0, self.release_vertical_delta],
                            dtype=np.float32,
                        )
                    )
                    (
                        self.release_open_command_path,
                        self.release_open_orientation_path,
                    ) = self._build_release_inverse_path(
                        state,
                        self.release_open_start_world,
                        release_open_end_world,
                        self.release_start_quat,
                        self.release_start_quat,
                        self.q_grasp,
                        self.q_open,
                        steps=35,
                        label="release_open_path",
                    )
                    self.ee_command_world = (
                        self.release_open_command_path[0].copy()
                    )
                    self.release_stable_steps = 0
                    print(
                        f"[doorman][release] handle horizontal; opening with "
                        f"vertical_delta={self.release_vertical_delta:.4f}",
                        flush=True,
                    )
            elif self.release_substage == "OPEN_AND_CLEAR":
                fully_released = gripper_open and release_contact_clear
                self.release_stable_steps = (
                    self.release_stable_steps + 1 if fully_released else 0
                )
                if self.release_stable_steps >= 10:
                    self.release_substage = "RETREAT"
                    self.release_retreat_start_world = ee.copy()
                    self.release_retreat_start_step = self.stage_steps
                    self.release_retreat_target_world = (
                        self.release_retreat_start_world
                        + 0.10 * self.door_normal
                        + np.array([0.0, 0.0, 0.02], dtype=np.float32)
                    )
                    retreat_steps = max(
                        50,
                        int(
                            math.ceil(
                                float(
                                    np.linalg.norm(
                                        self.release_retreat_target_world
                                        - self.release_retreat_start_world
                                    )
                                )
                                / 0.002
                            )
                        ),
                    )
                    (
                        self.release_retreat_command_path,
                        self.release_retreat_orientation_path,
                    ) = self._build_release_inverse_path(
                        state,
                        self.release_retreat_start_world,
                        self.release_retreat_target_world,
                        self.release_start_quat,
                        self.release_start_quat,
                        self.q_open,
                        self.q_open,
                        steps=retreat_steps,
                        label="release_retreat_path",
                    )
                    self.release_retreat_path_index = 0
                    self.ee_command_world = (
                        self.release_retreat_command_path[0].copy()
                    )
                    self.release_stable_steps = 0
                    print(
                        f"[doorman][release] both jaws clear; begin retreat "
                        f"hinge={hinge:.4f}",
                        flush=True,
                    )
            elif self.release_substage == "RETREAT":
                retreat_path_complete = bool(
                    self.release_retreat_command_path is not None
                    and self.release_retreat_path_index
                    >= len(self.release_retreat_command_path) - 1
                )
                retreat_position_error = float(
                    np.linalg.norm(ee - self.release_retreat_target_world)
                )
                retreat_complete = bool(
                    retreat_path_complete
                    and retreat_position_error <= 0.010
                    and release_contact_clear
                )
                self.release_stable_steps = (
                    self.release_stable_steps + 1 if retreat_complete else 0
                )
                if self.release_stable_steps >= 10:
                    self.release_endpoint_world = (
                        self.release_retreat_target_world.copy()
                    )
                    current_orientation = _as_np(state["ee_orn"])[0].copy()
                    dof_names = list(state["dof_names"])
                    current_wrist_roll = float(
                        _as_np(state["dof_pos"])[
                            0, dof_names.index("joint6")
                        ]
                    )
                    wrist_roll_delta = float(
                        self.initial_wrist_roll - current_wrist_roll
                    )
                    wrist_axis_world = _quat_rotate_np(
                        current_orientation,
                        np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    )
                    self.release_endpoint_quat = _quat_mul_np(
                        _quat_axis_angle_np(wrist_axis_world, wrist_roll_delta),
                        current_orientation,
                    )
                    (
                        self.release_restore_command_path,
                        self.release_restore_orientation_path,
                    ) = self._build_release_inverse_path(
                        state,
                        ee,
                        self.release_endpoint_world,
                        current_orientation,
                        self.release_endpoint_quat,
                        self.q_open,
                        self.q_closed,
                        steps=50,
                        label="release_restore_path",
                    )
                    print(
                        "[doorman][release] restoring wrist roll only "
                        f"q6_current={current_wrist_roll:.5f} "
                        f"q6_initial={self.initial_wrist_roll:.5f} "
                        f"delta={wrist_roll_delta:.5f}",
                        flush=True,
                    )
                    self.ee_command_world = (
                        self.release_restore_command_path[0].copy()
                    )
                    self.arm_stability_window.clear()
                    self.release_stable_steps = 0
                    self.stage = Stage.RETRACT_ARM; self.stage_steps = 0
        elif self.stage == Stage.RETRACT_ARM:
            gripper_closed = abs(gripper_pos - float(self.q_closed)) < 0.03
            dof_names = list(state["dof_names"])
            wrist_roll = float(
                _as_np(state["dof_pos"])[0, dof_names.index("joint6")]
            )
            orientation_restored = self._quat_distance(
                _as_np(state["ee_orn"])[0], self.release_endpoint_quat
            ) < 0.05
            wrist_roll_restored = abs(wrist_roll - self.initial_wrist_roll) < 0.05
            arm_stable = self._arm_pose_is_stable(state)
            restored = bool(
                self.stage_steps >= 50
                and gripper_closed
                and orientation_restored
                and wrist_roll_restored
                and np.linalg.norm(ee - self.release_endpoint_world) <= 0.010
                and arm_stable
            )
            self.release_stable_steps = (
                self.release_stable_steps + 1 if restored else 0
            )
            if self.release_stable_steps >= 10:
                self.panel_align_ee_goal_local = _as_np(
                    state["ee_goal_cart_local"]
                )[0].copy()
                goal_ref_quat = _as_np(self.env.get_goal_reference_quat())[0]
                self.panel_align_ee_orientation_local = _quat_mul_np(
                    _quat_conjugate_np(goal_ref_quat),
                    _as_np(state["ee_orn"])[0],
                )
                self.panel_station_settle_steps = 0
                self.stage = Stage.ALIGN_PANEL; self.stage_steps = 0
        elif self.stage == Stage.ALIGN_PANEL:
            panel_station_xy, target_yaw = self._current_panel_station(state)
            current_yaw = _quat_yaw(base[3:7])
            panel_station_error = float(
                np.linalg.norm(base_xy - panel_station_xy)
            )
            panel_yaw_error = abs(
                ((target_yaw - current_yaw + math.pi) % (2.0 * math.pi)) - math.pi
            )
            panel_station_settled = (
                panel_station_error < 0.08
                and panel_yaw_error < 0.08
                and speed < 0.08
            )
            self.panel_station_settle_steps = (
                self.panel_station_settle_steps + 1
                if panel_station_settled
                else 0
            )
            if self.panel_station_settle_steps >= 15:
                print(
                    f"[doorman][align-panel] centred base at "
                    f"xy={base_xy.round(4).tolist()} target="
                    f"{panel_station_xy.round(4).tolist()} "
                    f"yaw_error={panel_yaw_error:.4f}",
                    flush=True,
                )
                self.panel_contact_target_world = None
                self.panel_contact_orientation = None
                self.panel_alignment_established = False
                self.panel_contact_established = False
                self.panel_contact_settle_steps = 0
                self.panel_recovery_steps = 0
                self.stage = Stage.CONTACT_PANEL; self.stage_steps = 0
        elif self.stage == Stage.CONTACT_PANEL:
            ee_speed = float(np.linalg.norm(_as_np(state["ee_vel"])[0, :3]))
            hinge_speed = abs(float(_as_np(state["door_joint_velocity"])[0, self.hinge_idx]))
            finger_force = float(np.max(finger_contact_norms)) if finger_contact_norms.size else 0.0
            if self.panel_contact_target_world is not None:
                current_hinge = float(_as_np(state["door_joint_position"])[0, self.hinge_idx])
                current_hinge_world = self.hinge_axis_sign * current_hinge
                current_normal = _quat_rotate_np(
                    self._yaw_quat(current_hinge_world), self.door_normal
                )
                signed_clearance = float(
                    np.dot(ee - self.panel_contact_target_world, current_normal)
                )
                lateral_error = float(np.linalg.norm(
                    (ee - self.panel_contact_target_world)
                    - signed_clearance * current_normal
                ))
            else:
                signed_clearance = 999.0
                lateral_error = 999.0
            # CONTACT_PANEL is a guarded physical handoff.  The old 6 cm
            # threshold allowed the arm to enter PUSH_OPEN while still
            # travelling toward the panel; the first base command then
            # created an impulse and opened the door by inertia.  First enter
            # a 4 mm panel-clearance target, then require a settled, low-force
            # contact before enabling the pushing controller.
            if self.panel_recovery_steps > 0:
                self.panel_recovery_steps -= 1
            if (
                not self.panel_alignment_established
                and lateral_error < 0.025
                and abs(signed_clearance - 0.10) < 0.025
                and ee_speed < 0.06
            ):
                self.panel_alignment_established = True
            if finger_force > 120.0:
                self.panel_contact_established = False
                self.panel_contact_settle_steps = 0
                self.panel_recovery_steps = max(self.panel_recovery_steps, 8)
            elif (
                self.panel_alignment_established
                and signed_clearance <= 0.022
                and lateral_error < 0.035
                and ee_speed < 0.08
            ):
                self.panel_contact_established = True
            contact_sample = (
                self.panel_contact_established
                and abs(signed_clearance) <= 0.012
                and lateral_error < 0.025
                and ee_speed < 0.06
                and hinge_speed < 0.20
                and 0.5 <= finger_force < 120.0
            )
            self.panel_contact_settle_steps = (
                self.panel_contact_settle_steps + 1 if contact_sample else 0
            )
            panel_contact_ready = (
                self.panel_contact_settle_steps >= 8
                and abs(gripper_pos) < 0.20
            )
            # Do not treat passive post-release inertia or a distant arm as a
            # valid push.  PUSH_OPEN starts only after this guarded contact.
            if panel_contact_ready:
                self.push_start_hinge_angle = hinge
                self.push_last_progress_hinge = hinge
                self.push_no_progress_steps = 0
                self.push_contact_loss_steps = 0
                self.push_pause_steps = 0
                self.push_substage = "ARM_TRACK"
                self.push_contact_world = ee.copy()
                self.push_contact_reference_clearance = (
                    self._panel_finger_clearance(state, hinge)
                )
                self.push_finger_clearance = self.push_contact_reference_clearance
                self.push_finger_approach_speed = 0.0
                self.push_target_finger_clearance = (
                    self.push_nominal_finger_clearance
                )
                self.push_finger_force = finger_force
                self.push_contact_retreat_active = False
                self.push_previous_finger_clearance = (
                    self.push_contact_reference_clearance
                )
                self.push_normal_world = _quat_rotate_np(
                    self._yaw_quat(self.hinge_axis_sign * hinge),
                    self.door_normal,
                )
                self.push_orientation = _as_np(state["ee_orn"])[0].copy()
                self.push_peak_hinge = hinge
                self.push_coast_steps = 0
                self.push_depth = 0.0
                self.push_command_hinge_angle = hinge
                self.push_max_observed_angular_speed = abs(
                    float(
                        _as_np(state["door_joint_velocity"])[
                            0, self.hinge_idx
                        ]
                    )
                )
                self.push_speed_settle_steps = 0
                self.push_overspeed_steps = 0
                self.push_release_start_world = None
                self.push_release_normal_world = None
                self.push_release_start_step = 0
                self.push_base_start_xy = base_xy.copy()
                self.push_panel_contact_start_world = self.push_contact_world.copy()
                self.stage = Stage.PUSH_OPEN; self.stage_steps = 0
        elif self.stage == Stage.PUSH_OPEN:
            hinge_velocity = float(
                _as_np(state["door_joint_velocity"])[0, self.hinge_idx]
            )
            opening_speed = hinge_velocity
            opening_angle = hinge
            self.push_peak_hinge = max(self.push_peak_hinge, opening_angle)
            self.push_max_observed_angular_speed = max(
                self.push_max_observed_angular_speed,
                max(0.0, opening_speed),
            )
            self.push_overspeed_steps = (
                self.push_overspeed_steps + 1
                if opening_speed > self.push_max_angular_speed
                else 0
            )
            if self.push_overspeed_steps >= 3:
                self.failure_reason = (
                    f"push_angular_speed_limit:velocity={opening_speed:.4f},"
                    f"limit={self.push_max_angular_speed:.4f}"
                )
                self.stage = Stage.FAILURE
            elif opening_angle > math.radians(94.0):
                self.failure_reason = (
                    f"push_angle_overshoot:hinge={opening_angle:.4f}"
                )
                self.stage = Stage.FAILURE
            elif self.push_substage == "ARM_TRACK":
                near_goal = opening_angle >= math.radians(89.0)
                speed_settled = abs(opening_speed) <= math.radians(2.5)
                self.push_speed_settle_steps = (
                    self.push_speed_settle_steps + 1
                    if near_goal and speed_settled
                    else 0
                )
                if self.push_speed_settle_steps >= 15:
                    self.push_substage = "ARM_RELEASE"
                    self.push_release_start_world = ee.copy()
                    self.push_release_normal_world = _quat_rotate_np(
                        self._yaw_quat(
                            self.hinge_axis_sign * opening_angle
                        ),
                        self.door_normal,
                    )
                    self.push_release_start_step = self.stage_steps
                    self.ee_command_world = ee.copy()
                    print(
                        f"[doorman][push] reached controlled 90 deg; "
                        f"begin arm release hinge={opening_angle:.4f} "
                        f"speed={opening_speed:.4f} max_speed="
                        f"{self.push_max_observed_angular_speed:.4f}",
                        flush=True,
                    )
                if self.push_last_progress_hinge is None:
                    self.push_last_progress_hinge = opening_angle
                if opening_angle >= self.push_last_progress_hinge + math.radians(0.25):
                    self.push_last_progress_hinge = opening_angle
                    self.push_no_progress_steps = 0
                elif (
                    self.push_command_hinge_angle is not None
                    and self.push_command_hinge_angle
                    > opening_angle + math.radians(0.5)
                ):
                    self.push_no_progress_steps += 1
                else:
                    self.push_no_progress_steps = 0
                if self.push_no_progress_steps > 300:
                    self.failure_reason = (
                        f"arm_push_no_progress:hinge={opening_angle:.4f},"
                        f"command={self.push_command_hinge_angle:.4f}"
                    )
                    self.stage = Stage.FAILURE
            elif self.push_substage == "ARM_RELEASE":
                release_steps = self.stage_steps - self.push_release_start_step
                if opening_angle < math.radians(85.0):
                    self.failure_reason = (
                        f"push_release_swingback:hinge={opening_angle:.4f}"
                    )
                    self.stage = Stage.FAILURE
                elif release_steps >= 50 and abs(opening_speed) <= math.radians(3.0):
                    self.through_start_base_xy = base_xy.copy()
                    self.through_best_distance = 0.0
                    self.through_no_progress_steps = 0
                    self.stage = Stage.THROUGH; self.stage_steps = 0
            else:
                raise RuntimeError(
                    f"Unsupported arm push substage: {self.push_substage}"
                )
        elif self.stage == Stage.THROUGH:
            through_origin = (
                self.through_start_base_xy
                if self.through_start_base_xy is not None
                else base_xy
            )
            through_distance = float(
                np.dot(base_xy - through_origin, self.door_forward[:2])
            )
            opening_angle = hinge
            base_up = _quat_rotate_np(
                base[3:7], np.array([0.0, 0.0, 1.0], dtype=np.float32)
            )
            upright = float(base_up[2])
            if through_distance > self.through_best_distance + 0.01:
                self.through_best_distance = through_distance
                self.through_no_progress_steps = 0
            else:
                self.through_no_progress_steps += 1
            if upright < 0.75 or base[2] < 0.30:
                self.failure_reason = (
                    f"through_robot_fell:up={upright:.3f},height={base[2]:.3f}"
                )
                self.stage = Stage.FAILURE
            elif opening_angle < math.radians(65.0) and through_distance < 0.8:
                self.failure_reason = (
                    f"through_door_swingback:hinge={hinge:.4f},"
                    f"distance={through_distance:.3f}"
                )
                self.stage = Stage.FAILURE
            elif self.through_no_progress_steps > 150:
                self.failure_reason = (
                    f"through_no_progress:distance={through_distance:.3f}"
                )
                self.stage = Stage.FAILURE
            elif through_distance > 1.0:
                self.stage = Stage.RESET_ARM; self.stage_steps = 0
        elif self.stage == Stage.RESET_ARM:
            # The same Cartesian command can converge to a different bounded
            # IK solution after the mirrored grasp branch and traversal.  The
            # interface contract is that the default command is restored;
            # validate the resulting physical arm against a safe body-frame
            # envelope instead of requiring the initial FK solution.
            reset_arm_safe = bool(
                0.25 < ee_base_local[0] < 0.70
                and abs(float(ee_base_local[1])) < 0.15
                and 0.25 < ee_base_local[2] < 0.65
            )
            reset_ee_speed = float(
                np.linalg.norm(_as_np(state["ee_vel"])[0, :3])
            )
            if (
                self.stage_steps >= 80
                and reset_arm_safe
                and reset_ee_speed < 0.08
                and speed < 0.10
            ):
                self.stage = Stage.SUCCESS; self.stage_steps = 0
        self.stage_steps += 1
        if self.stage in self.stage_limits and self.stage_steps > self.stage_limits[self.stage]:
            if self.stage == Stage.TURN_HANDLE:
                self.failure_reason = (
                    f"turn_timeout:handle_delta={abs(self.turn_last_handle_angle - self.turn_start_handle_angle):.4f},"
                    f"hinge_delta={abs(self.turn_last_hinge_angle - self.turn_start_hinge_angle):.4f}"
                )
            else:
                self.failure_reason = f"stage_timeout:{self.stage.name.lower()}"
            self.stage = Stage.FAILURE
        if self.stage != previous_stage:
            print(
                f"[doorman] stage {previous_stage.name} -> {self.stage.name} "
                f"after {self.stage_steps} steps: station_dist={station_dist:.3f} "
                f"yaw_error={yaw_error:.3f} speed={speed:.3f} hinge={hinge:.3f} "
                f"handle_joint={handle_joint:.3f} latch_joint={latch_joint:.4f} "
                f"handle_center_dist={ee_dist:.3f} "
                f"handle_xy_dist={ee_xy_dist:.3f} handle_z_error={ee_z_dist:.3f} "
                f"panel_dist={panel_dist:.3f} pregrasp_dist={pregrasp_dist:.3f} "
                f"panel_settle={self.panel_contact_settle_steps} "
                f"push_progress={0.0 if self.push_start_hinge_angle is None else hinge - self.push_start_hinge_angle:.3f} "
                f"through_distance={0.0 if self.through_start_base_xy is None else float(np.dot(base_xy - self.through_start_base_xy, self.door_forward[:2])):.3f} "
                f"grasp_contact_force={grasp_contact_force:.3f} "
                f"finger_contact_forces={finger_contact_norms.round(3).tolist()} "
                f"handle_contact_forces={handle_contact_norms.round(3).tolist()} "
                f"grasp_orientation_error={ee_orientation_error:.3f}",
                flush=True,
            )
        if self.stage == Stage.CENTER_HANDLE and self.stage_steps % 10 == 0:
            scalar_metrics = {
                key: round(float(value), 4)
                for key, value in self.center_metrics.items()
                if isinstance(value, (float, int, np.floating))
            }
            print(
                f"[doorman][center] substage={self.center_substage} q={actual_q if 'actual_q' in locals() else float('nan'):.4f} "
                f"metrics={scalar_metrics}",
                flush=True,
            )
            if self.center_metrics.get("valid"):
                print(
                    f"[doorman][center-geom] fixed_inner={np.asarray(self.center_metrics.get('fixed_center')).round(4).tolist()} "
                    f"moving_inner={np.asarray(self.center_metrics.get('moving_center')).round(4).tolist()} "
                    f"handle={np.asarray(grasp_center).round(4).tolist()}",
                    flush=True,
                )
            tip_metrics = self._tip_metrics(state, grasp_center)
            if tip_metrics.get("valid"):
                print(
                    f"[doorman][tip-geom] fixed={np.asarray(tip_metrics['fixed_tip']).round(4).tolist()} "
                    f"moving={np.asarray(tip_metrics['moving_tip']).round(4).tolist()} "
                    f"mid_err={tip_metrics['tip_midpoint_error']:.4f} "
                    f"xy={tip_metrics['horizontal_error']:.4f} "
                    f"upper_gap={tip_metrics['upper_gap']:.4f} lower_gap={tip_metrics['lower_gap']:.4f}",
                    flush=True,
                )
        elif self.stage == Stage.GRASP_HANDLE and self.stage_steps % 10 == 0:
            print(
                f"[doorman][grasp] q={gripper_pos:.4f} "
                f"fixed_clearance={metrics.get('fixed_handle_clearance', float('nan')):.4f} "
                f"moving_clearance={metrics.get('moving_handle_clearance', float('nan')):.4f} "
                f"panel_clearance={metrics.get('panel_clearance', float('nan')):.4f} "
                f"fixed_center={np.asarray(metrics.get('fixed_center')).round(4).tolist()} "
                f"moving_center={np.asarray(metrics.get('moving_center')).round(4).tolist()} "
                f"handle_contact_forces={handle_contact_norms.round(3).tolist()}",
                flush=True,
            )
        return self.stage


def _serialize_episode(controller, runner, episode_id, seed, output_dir, max_steps, save_failures, debug_stage=None):
    state = controller.reset()
    records = {k: [] for k in (
        "timestamp", "step", "stage_id", "task_phase_id", "done", "success", "base_pose_w", "base_velocity_w",
        "joint_position", "joint_velocity", "joint_position_target", "arm_position_target",
        "applied_torque", "ee_pose_w", "ee_velocity_w", "ee_goal_cart_world",
        "ee_goal_orn_quat_world", "ee_goal_orn_quat_world_enabled", "ee_jacobian_singular_values",
        "ee_jacobian_condition", "gripper_position", "robot_body_state_w", "contact_forces_w",
        "handle_contact_points_w",
        "door_root_pose_w", "door_joint_position", "door_joint_velocity", "handle_pose_w",
        "base_command", "ee_goal_cart_local", "ee_goal_delta_rpy", "gripper_command",
        "high_level_action", "low_level_action", "center_substage_id",
        "q_open", "gripper_actual", "center_horizontal_error", "center_upper_gap",
        "center_lower_gap", "center_grasp_depth", "center_pocket_error",
        "center_pocket_normal_error", "center_pocket_vertical_error", "center_panel_clearance",
        "handle_contact_force", "push_command_hinge_angle",
        "push_shoulder_min_clearance", "push_shoulder_predicted_min_clearance",
        "push_base_safety_scale", "push_finger_clearance",
        "push_finger_approach_speed", "push_target_finger_clearance",
        "push_finger_force", "push_contact_retreat_active")}
    camera_frames = {}
    camera_timestamps = {}
    rgb_index_rows = []
    debug_retract_hold_steps = None
    for step in range(max_steps):
        if controller.stage in (Stage.SUCCESS, Stage.FAILURE):
            break
        finger_force = state.get("finger_contact_forces_w")
        if finger_force is not None:
            max_finger_force = float(np.linalg.norm(_as_np(finger_force)[0], axis=-1).max())
            if max_finger_force > 180.0:
                controller.failure_reason = f"safety_force_limit:{max_finger_force:.1f}N"
                controller.stage = Stage.FAILURE
                finger_vectors = _as_np(finger_force)[0]
                handle_vectors = _as_np(
                    state.get("handle_finger_contact_forces_w")
                )
                handle_points = _as_np(
                    state.get("handle_finger_contact_points_w")
                )
                door_joints = _as_np(state.get("door_joint_position"))
                print(
                    f"[doorman][safety] aborting fixed trajectory at step={step}: "
                    f"finger force {max_finger_force:.1f} N exceeds 180 N; "
                    f"finger_vectors={finger_vectors.round(2).tolist()} "
                    f"handle_vectors={None if handle_vectors is None else handle_vectors[0].round(2).tolist()} "
                    f"handle_points={None if handle_points is None else handle_points[0].round(4).tolist()} "
                    f"door_joints={None if door_joints is None else door_joints[0].round(5).tolist()}",
                    flush=True,
                )
                break
        # Closed-loop rolling-horizon control.  A local target is recomputed
        # from the measured state every cycle; the target slew provides the
        # interpolation and naturally re-plans at the five-step boundary.
        high, dist, yaw_error = controller._legacy_feedback_command(state)
        if step == 0 or step % 10 == 0:
            ee_dbg = _as_np(state.get("ee_pos"))[0].round(3).tolist() if state.get("ee_pos") is not None else None
            goal_dbg = _as_np(state.get("ee_goal_cart_world"))[0].round(3).tolist() if state.get("ee_goal_cart_world") is not None else None
            contact_dbg = state.get("gripper_contact_force_w")
            contact_dbg = float(np.linalg.norm(_as_np(contact_dbg)[0])) if contact_dbg is not None else 0.0
            base_dbg = _as_np(state["root_states"])[0]
            base_pos_dbg = base_dbg[:3].round(3).tolist()
            base_speed_dbg = float(np.linalg.norm(base_dbg[7:10]))
            robot_bodies_dbg = _as_np(state.get("robot_body_state_w"))
            finger_pos_dbg = None
            if robot_bodies_dbg is not None:
                finger_pos_dbg = robot_bodies_dbg[0, [
                    controller.fixed_finger_body_idx,
                    controller.moving_finger_body_idx,
                ], :3].round(3).tolist()
            finger_force_dbg = state.get("finger_contact_forces_w")
            if finger_force_dbg is not None:
                finger_force_dbg = np.linalg.norm(_as_np(finger_force_dbg)[0], axis=-1).round(3).tolist()
            handle_force_dbg = state.get("handle_finger_contact_forces_w")
            if handle_force_dbg is not None:
                handle_force_dbg = np.linalg.norm(_as_np(handle_force_dbg)[0], axis=-1).round(3).tolist()
            door_joint_dbg = _as_np(state["door_joint_position"])[0].round(5).tolist()
            push_progress_dbg = (
                0.0
                if controller.push_start_hinge_angle is None
                else float(door_joint_dbg[controller.hinge_idx])
                - controller.push_start_hinge_angle
            )
            through_distance_dbg = (
                0.0
                if controller.through_start_base_xy is None
                else float(np.dot(
                    base_dbg[:2] - controller.through_start_base_xy,
                    controller.door_forward[:2],
                ))
            )
            print(
                f"[doorman] step={step} stage={controller.stage.name} "
                f"base_command={high[:3].round(3).tolist()} station_dist={dist:.3f} "
                f"yaw_error={yaw_error:.3f} ee={ee_dbg} goal={goal_dbg} "
                f"base_pos={base_pos_dbg} base_speed={base_speed_dbg:.3f} "
                f"stage_steps={controller.stage_steps} grasp_contact_force={contact_dbg:.3f} "
                f"grasp_orientation_error={controller.ee_orientation_error:.3f} "
                f"door_joints={door_joint_dbg} "
                f"panel_settle={controller.panel_contact_settle_steps} "
                f"push_substage={controller.push_substage} "
                f"push_progress={push_progress_dbg:.3f} "
                f"push_command_hinge={controller.push_command_hinge_angle} "
                f"shoulder_clearance={controller.push_shoulder_min_clearance:.3f} "
                f"shoulder_predicted={controller.push_shoulder_predicted_min_clearance:.3f} "
                f"base_safety_scale={controller.push_base_safety_scale:.2f} "
                f"finger_clearance={controller.push_finger_clearance:.4f} "
                f"finger_approach={controller.push_finger_approach_speed:.3f} "
                f"finger_clearance_target={controller.push_target_finger_clearance:.4f} "
                f"push_finger_force={controller.push_finger_force:.1f} "
                f"contact_retreat={int(controller.push_contact_retreat_active)} "
                f"through_distance={through_distance_dbg:.3f} "
                f"finger_positions={finger_pos_dbg} finger_contact_forces={finger_force_dbg} "
                f"handle_contact_forces={handle_force_dbg}",
                flush=True,
            )
        stage_before = int(controller.stage)
        tensor_state_keys = (
            "root_states",
            "dof_pos",
            "dof_vel",
            "ee_pos",
            "ee_orn",
            "ee_vel",
            "gripper_position",
            "gripper_command",
            "door_root_pose_w",
            "door_joint_position",
            "door_joint_velocity",
            "door_body_state_w",
            "joint_position_target",
            "arm_position_target",
            "applied_torque",
            "ee_goal_cart_world",
            "ee_goal_orn_quat_world",
            "ee_goal_orn_quat_world_enabled",
            "ee_jacobian_arm",
            "robot_body_state_w",
            "contact_forces_w",
            "handle_finger_contact_points_w",
        )
        state_np = {key: _as_np(state[key]) for key in tensor_state_keys}
        records["timestamp"].append(step * float(runner.env.dt)); records["step"].append(step)
        records["stage_id"].append(stage_before)
        records["task_phase_id"].append(STAGE_TO_TASK_PHASE.get(controller.stage, -1))
        records["done"].append(False); records["success"].append(False)
        records["base_pose_w"].append(state_np["root_states"][0, :7]); records["base_velocity_w"].append(state_np["root_states"][0, 7:])
        records["joint_position"].append(state_np["dof_pos"][0]); records["joint_velocity"].append(state_np["dof_vel"][0])
        records["joint_position_target"].append(state_np["joint_position_target"][0])
        records["arm_position_target"].append(state_np["arm_position_target"][0])
        records["applied_torque"].append(state_np["applied_torque"][0])
        ee_pos = state_np["ee_pos"] if state_np["ee_pos"] is not None else np.zeros((1, 3), dtype=np.float32)
        ee_orn = state_np["ee_orn"] if state_np["ee_orn"] is not None else np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (1, 1))
        ee_vel = state_np["ee_vel"] if state_np["ee_vel"] is not None else np.zeros((1, 6), dtype=np.float32)
        records["ee_pose_w"].append(np.concatenate([ee_pos[0], ee_orn[0]])); records["ee_velocity_w"].append(ee_vel[0])
        records["ee_goal_cart_world"].append(state_np["ee_goal_cart_world"][0])
        records["ee_goal_orn_quat_world"].append(state_np["ee_goal_orn_quat_world"][0])
        records["ee_goal_orn_quat_world_enabled"].append(bool(state_np["ee_goal_orn_quat_world_enabled"][0]))
        jacobian = state_np["ee_jacobian_arm"]
        if jacobian is None:
            singular_values = np.full(6, np.nan, dtype=np.float32)
        else:
            singular_values = np.linalg.svd(jacobian[0], compute_uv=False).astype(np.float32)
        records["ee_jacobian_singular_values"].append(singular_values)
        records["ee_jacobian_condition"].append(
            float(singular_values[0] / singular_values[-1])
            if np.isfinite(singular_values).all() and singular_values[-1] > 1.0e-8
            else float("inf")
        )
        robot_body_state = state_np["robot_body_state_w"]
        records["robot_body_state_w"].append(
            robot_body_state[0]
            if robot_body_state is not None
            else np.empty((0, 13), dtype=np.float32)
        )
        contact_forces = state_np["contact_forces_w"]
        records["contact_forces_w"].append(
            contact_forces[0]
            if contact_forces is not None
            else np.empty((0, 3), dtype=np.float32)
        )
        contact_points = state_np["handle_finger_contact_points_w"]
        records["handle_contact_points_w"].append(
            contact_points[0] if contact_points is not None else np.full((2, 3), np.nan, dtype=np.float32)
        )
        records["gripper_position"].append(state_np["gripper_position"][0]); records["door_root_pose_w"].append(state_np["door_root_pose_w"][0])
        records["door_joint_position"].append(state_np["door_joint_position"][0]); records["door_joint_velocity"].append(state_np["door_joint_velocity"][0])
        handle_pose = state_np["door_body_state_w"][0, controller.handle_body_idx, :7]; records["handle_pose_w"].append(handle_pose)
        records["base_command"].append(high[:3]); records["ee_goal_cart_local"].append(high[3:6]); records["ee_goal_delta_rpy"].append(high[6:9]); records["gripper_command"].append(high[9:10]); records["high_level_action"].append(high)
        records["push_command_hinge_angle"].append(
            float(
                controller.push_command_hinge_angle
                if controller.push_command_hinge_angle is not None
                else np.nan
            )
        )
        records["push_shoulder_min_clearance"].append(
            float(controller.push_shoulder_min_clearance)
        )
        records["push_shoulder_predicted_min_clearance"].append(
            float(controller.push_shoulder_predicted_min_clearance)
        )
        records["push_base_safety_scale"].append(
            float(controller.push_base_safety_scale)
        )
        records["push_finger_clearance"].append(
            float(controller.push_finger_clearance)
        )
        records["push_finger_approach_speed"].append(
            float(controller.push_finger_approach_speed)
        )
        records["push_target_finger_clearance"].append(
            float(controller.push_target_finger_clearance)
        )
        records["push_finger_force"].append(
            float(controller.push_finger_force)
        )
        records["push_contact_retreat_active"].append(
            bool(controller.push_contact_retreat_active)
        )
        center_substages = (
            "CENTER_FORWARD",
        )
        metrics = controller.center_metrics
        records["center_substage_id"].append(center_substages.index(controller.center_substage) if controller.center_substage in center_substages else -1)
        records["q_open"].append(float(controller.q_open if controller.q_open is not None else np.nan))
        records["gripper_actual"].append(float(state_np["gripper_position"][0, 0]))
        for key, record_key in (
            ("horizontal_error", "center_horizontal_error"),
            ("upper_gap", "center_upper_gap"),
            ("lower_gap", "center_lower_gap"),
            ("grasp_depth", "center_grasp_depth"),
            ("pocket_center_error", "center_pocket_error"),
            ("pocket_normal_error", "center_pocket_normal_error"),
            ("pocket_vertical_error", "center_pocket_vertical_error"),
            ("panel_clearance", "center_panel_clearance"),
        ):
            records[record_key].append(float(metrics.get(key, np.nan)))
        handle_force = state.get("handle_finger_contact_forces_w")
        records["handle_contact_force"].append(
            np.linalg.norm(_as_np(handle_force)[0], axis=-1) if handle_force is not None else np.zeros(2, dtype=np.float32)
        )
        if step and step % 100 == 0:
            print(f"[doorman] env step begin step={step}", flush=True)
        result = runner.step_once()
        if step and step % 100 == 0:
            print(f"[doorman] env step end step={step}", flush=True)
        low = _as_np(result["low_level_action"])[0]; records["low_level_action"].append(low)
        frame_indices = {}
        for camera_name, frame in result.get("rgb_frames", {}).items():
            frames = camera_frames.setdefault(camera_name, [])
            timestamps = camera_timestamps.setdefault(camera_name, [])
            frame_indices[camera_name] = len(frames)
            frames.append(frame)
            timestamps.append(step * float(runner.env.dt))
        rgb_index_rows.append(frame_indices)
        finite_state = all(
            state_np[key] is None or np.isfinite(state_np[key]).all()
            for key in (
                "root_states",
                "dof_pos",
                "dof_vel",
                "ee_pos",
                "ee_orn",
                "ee_vel",
                "door_root_pose_w",
                "door_joint_position",
                "door_joint_velocity",
                "door_body_state_w",
                "joint_position_target",
                "arm_position_target",
                "applied_torque",
                "ee_goal_cart_world",
                "ee_goal_orn_quat_world",
                "ee_jacobian_arm",
            )
        )
        if not finite_state or not np.isfinite(low).all() or not np.isfinite(high).all():
            controller.failure_reason = "non_finite_state_or_action"
            controller.stage = Stage.FAILURE
            print("[doorman] FAILURE: non-finite state or action", flush=True)
            break
        state = result["state"]
        controller.replan_counter = (controller.replan_counter + 1) % controller.replan_interval
        if debug_retract_hold_steps is None:
            controller._legacy_feedback_advance(state, dist, yaw_error)
            if debug_stage == "retract_arm" and controller.stage == Stage.ALIGN_PANEL:
                controller.stage = Stage.RETRACT_ARM
                controller.stage_steps = 100
                debug_retract_hold_steps = 50
                print(
                    "[doorman][debug] handle released, arm retreated, and "
                    "initial closed-gripper orientation restored; holding for 1.0 s",
                    flush=True,
                )
        else:
            debug_retract_hold_steps -= 1
            if debug_retract_hold_steps <= 0:
                controller.stage = Stage.SUCCESS
                controller.failure_reason = None
                print(
                    "[doorman][debug] RELEASE_HANDLE and RETRACT_ARM validation passed",
                    flush=True,
                )
    success = controller.stage == Stage.SUCCESS
    if records["done"]:
        records["done"][-1] = True; records["success"][-1] = success
    if not success and not save_failures:
        print(f"[doorman] episode {episode_id} failed at stage={controller.stage.name}")
        return False
    episode_dir = output_dir / f"episode_{episode_id:06d}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    rgb_dir = episode_dir / "rgb"; rgb_dir.mkdir()
    video_dir = episode_dir / "video"; video_dir.mkdir()
    camera_names = sorted(camera_frames)
    for camera_name in camera_names:
        camera_dir = rgb_dir / camera_name
        camera_dir.mkdir()
        for i, frame in enumerate(camera_frames[camera_name]):
            _save_rgb(frame, camera_dir / f"{i:06d}.png")
        _save_rgb_video(
            camera_frames[camera_name],
            camera_timestamps[camera_name],
            video_dir / f"{camera_name}.mp4",
        )
    mosaic_video_name = "all_cameras.mp4"
    mosaic_video_written = _save_rgb_mosaic_video(
        camera_frames,
        camera_timestamps,
        video_dir / mosaic_video_name,
    )
    records["rgb_frame_index"] = [
        [row.get(camera_name, -1) for camera_name in camera_names]
        for row in rgb_index_rows
    ]
    for camera_name in camera_names:
        records[f"rgb_{camera_name}_timestamp"] = camera_timestamps[camera_name]
    arrays = {k: np.asarray(v) for k, v in records.items()}
    np.savez_compressed(episode_dir / "trajectory.npz", **arrays)
    metadata = {
        "schema_version": "doorman_raw_v1", "episode_id": episode_id, "seed": seed, "robot": "b2z1",
        "sim_dt": float(runner.env.dt), "control_dt": float(runner.env.dt), "success": bool(success),
        "stage_names": STAGE_NAMES,
        "task_phase_names": TASK_PHASE_NAMES,
        "camera": {
            "enabled": bool(camera_frames),
            "format": ["png", "mp4"],
            "channels": "rgb",
            "names": camera_names,
            "frame_count": {
                name: len(camera_frames[name]) for name in camera_names
            },
            "mosaic_video": (
                f"video/{mosaic_video_name}" if mosaic_video_written else None
            ),
        },
        "scene": {
            "door_prim_path": "/World/envs/env_0/door",
            "door_joint_names": controller.door_joint_names,
            "door_body_names": controller.door_body_names,
            "hinge_joint_index": controller.hinge_idx,
            "handle_joint_index": controller.handle_joint_idx,
            "latch_joint_index": controller.latch_joint_idx,
            "robot_side": controller.robot_side,
            "approach_normal_world": controller.door_normal.tolist(),
            "push_and_traverse_direction_world": controller.door_forward.tolist(),
            "handle_surface_z": [controller.handle_lower_z, controller.handle_upper_z],
            "grasp_center_world": controller.grasp_center_pos.tolist(),
            "geometry_report": controller.geometry_report,
        },
        "robot_state": {
            "joint_names": list(state.get("dof_names", [])),
            "body_names": list(state.get("body_names", [])),
            "arm_joint_indices": _as_np(runner.env.arm_joint_ids).astype(int).tolist(),
            "gripper_joint_indices": _as_np(runner.env.gripper_joint_ids).astype(int).tolist(),
        },
        "control": {
            "action_delay_mode": "undelayed",
            "high_level_action_dim": 10,
            "low_level_action_dim": int(arrays["low_level_action"].shape[-1]) if arrays["low_level_action"].ndim > 1 else 0,
            "push_controller": "arm_velocity_feedback",
            "push_goal_angle_deg": math.degrees(controller.push_goal_hinge_angle),
            "push_target_angular_speed_deg_s": math.degrees(controller.push_target_angular_speed),
            "push_max_angular_speed_deg_s": math.degrees(controller.push_max_angular_speed),
            "push_max_observed_angular_speed_deg_s": math.degrees(controller.push_max_observed_angular_speed),
            "base_push_assist": "forward_only_no_yaw_shoulder_swept_clearance",
            "base_push_shoulder_bodies": ["FL_hip", "FR_hip"],
            "base_push_shoulder_collision_radius_m": controller.push_shoulder_collision_radius,
            "base_push_panel_half_thickness_m": controller.panel_half_thickness,
            "base_push_shoulder_safety_margin_m": controller.push_shoulder_safety_margin,
            "base_push_prediction_horizon_s": 0.45,
            "base_push_assumed_deceleration_m_s2": 0.60,
            "push_nominal_finger_clearance_m": controller.push_nominal_finger_clearance,
            "push_recontact_step_m": controller.push_recontact_step,
        },
        "assist_mode": "none",
        "door_effort_assist": False,
        "kinematic_assist": False,
        "data_quality": "physical_only",
        "failure_reason": controller.failure_reason,
    }
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"[doorman] wrote {episode_dir} success={success} steps={len(arrays['step'])} "
        f"frames={{{', '.join(f'{name}: {len(camera_frames[name])}' for name in camera_names)}}}"
    )
    return success


def main():
    args = _parser().parse_args()
    if args.num_episodes < 1 or args.max_steps < 1:
        raise SystemExit("--num_episodes and --max_steps must be positive")
    if args.num_episodes > 1 and os.environ.get("DOORMAN_SINGLE_EPISODE_CHILD") != "1":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _run_episode_children(args)
        return
    _default_paths(args)
    for label, path in (("checkpoint", args.ckpt_path), ("run metadata", args.run_metadata_path), ("robot USD", args.robot_usd_path)):
        if path and not Path(path).is_file():
            raise SystemExit(f"{label} not found: {path}")
    if args.rgb_camera_config and not Path(args.rgb_camera_config).is_file():
        raise SystemExit(f"RGB camera config not found: {args.rgb_camera_config}")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        suggested = Path.cwd() / "doorman_dataset"
        raise SystemExit(
            f"Cannot write output directory {args.output_dir}: permission denied. "
            f"Use a writable path such as {suggested} or /tmp/doorman_dataset."
        ) from exc
    simulation_app = launch_app(args)
    successes = 0
    try:
        next_episode_id = 1
        base_seed = int(args.seed if args.seed is not None else 0)
        for episode in range(args.num_episodes):
            while (args.output_dir / f"episode_{next_episode_id:06d}").exists():
                next_episode_id += 1
            episode_seed = base_seed + episode
            args.seed = episode_seed
            torch.manual_seed(episode_seed)
            runner = ManipLoco_Policy(args)
            controller = DoorController(runner, args.max_steps, args.robot_side, args.position_only_center)
            successes += int(_serialize_episode(
                controller, runner, next_episode_id, episode_seed, args.output_dir,
                args.max_steps, args.save_failures, args.debug_stage,
            ))
            next_episode_id += 1
            if hasattr(runner.env, "close"):
                runner.env.close()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        # Isaac Sim 5.1 can deadlock in the headless render callback during
        # SimulationApp.close().  Process teardown releases those resources;
        # interactive runs still use the normal close path.
        if not args.headless and simulation_app is not None and hasattr(simulation_app, "close"):
            simulation_app.close()
    print(
        f"[doorman] complete: {successes}/{args.num_episodes} successful episodes",
        flush=True,
    )
    if args.headless:
        # Isaac Sim 5.1 may hang while unloading rendering plugins even after
        # all episode data has been flushed.  Each headless episode owns its
        # process, so deterministic process teardown is safe here.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
