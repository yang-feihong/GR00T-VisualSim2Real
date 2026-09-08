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
import signal
import sys
import time
import traceback
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
    CONTACT_PANEL = 9
    PUSH_OPEN = 10
    THROUGH = 11
    RESET_ARM = 12
    SUCCESS = 13
    FAILURE = 14


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
        doorman_randomize_door_init_state=False,
        action_delay_mode="undelayed",
        teleop_mode=True,
    )
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output root. Defaults to <repo>/datasets/doorman/episodes.",
    )
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--save_failures", action="store_true")
    parser.add_argument(
        "--debug_stage",
        choices=("center_handle",),
        default=None,
        help="Run only the deterministic CENTER_HANDLE/GRASP diagnostic.",
    )
    parser.set_defaults(doorman_door_open_lr="left")
    parser.add_argument(
        "--robot_side",
        choices=("positive_x", "negative_x"),
        default="positive_x",
        help="Side of the door where the robot starts; positive_x is the default collection setup.",
    )
    parser.add_argument("--position_only_center", action="store_true", help="Keep EE orientation fixed during CENTER_HANDLE.")
    parser.add_argument(
        "--rgb_camera_name",
        type=str,
        default=None,
        help="Camera rig used for the saved RGB stream; defaults to door_panel_diagnostic when available.",
    )
    return parser


def _default_paths(args):
    repo = SCRIPT_ROOT.parent.parent
    run_dir = repo / "ckpt" / "b2z1-low" / "20260817_154058_is-dc6jrxz4suufnmlp-devmachine-0_gpu1_agent2_z8e0p66v"
    if not args.ckpt_path:
        args.ckpt_path = str(run_dir / "model_40000.pt")
    if not args.run_metadata_path:
        args.run_metadata_path = str(run_dir / "run_metadata.json")
    if not args.robot_usd_path:
        args.robot_usd_path = str(repo / "gr00t" / "rl" / "data" / "robots" / "b2z1_lab30" / "b2z1_mount_0_x0p22_y0_z0p17.usd")
    if not args.rgb_camera_config:
        default_rgb_config = (
            repo
            / "gr00t"
            / "rl"
            / "third_party"
            / "visual_wholebody_lowlevel"
            / "legged_gym"
            / "scripts"
            / "rgb_cameras.json"
        )
        if default_rgb_config.is_file():
            args.rgb_camera_config = str(default_rgb_config)


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


def _save_rgb(frame, path: Path):
    if frame is None:
        return False
    array = frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * (255.0 if array.max(initial=0.0) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    from PIL import Image

    Image.fromarray(array, mode="RGB").save(path)
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


def _read_collider_points(stage, body_prim, tokens, reference_prim=None):
    """Read sampled points from all collision descendants matching tokens."""
    from pxr import Gf, Usd, UsdGeom

    if reference_prim is None:
        reference_prim = body_prim
    body_xf = UsdGeom.Xformable(reference_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
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
    return ColliderGeometry("", reference_prim.GetName(), np.asarray(points), paths)


def _read_robot_collider_by_path(stage, robot_root, tokens, reference_prim):
    """Read collision prims anywhere below the robot, including instance proxies."""
    from pxr import Gf, Usd, UsdGeom
    ref_xf = UsdGeom.Xformable(reference_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    inv_ref = ref_xf.GetInverse()
    points, paths = [], []
    for prim in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies()):
        low = str(prim.GetPath()).lower()
        if "/collisions/" not in low or not any(tok in low for tok in tokens):
            continue
        local = None
        if prim.IsA(UsdGeom.Mesh):
            raw = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
            local = [Gf.Vec3d(p) for p in raw] if raw else None
        elif prim.IsA(UsdGeom.Cube):
            h = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 1.0) * 0.5
            local = [Gf.Vec3d(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)]
        if local:
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            transformed = [inv_ref.Transform(xf.Transform(p)) for p in local]
            stride = max(1, len(transformed) // 1024)
            points.extend(np.asarray(transformed[::stride], dtype=np.float32))
            paths.append(str(prim.GetPath()))
    if not points:
        return None
    return ColliderGeometry("", reference_prim.GetName(), np.asarray(points), paths)


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
            Stage.CENTER_HANDLE: 300,
            Stage.GRASP_HANDLE: 250,
            Stage.TURN_HANDLE: 500,
            Stage.RELEASE_HANDLE: 150,
            Stage.RETRACT_ARM: 300,
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
        self.handle_align_settle_steps = 0
        self.grasp_settle_steps = 0
        self.grasp_contact_threshold = 1.0
        self.handle_contact_last_seen = np.full(2, -1000, dtype=np.int32)
        self.turn_start_handle_angle = 0.0
        self.turn_start_latch_position = 0.0
        self.turn_start_hinge_angle = 0.0
        self.turn_last_handle_angle = 0.0
        self.turn_last_hinge_angle = 0.0
        self.panel_contact_local = None
        self.handle_lower_z = None
        self.handle_upper_z = None
        self.handle_collider_center_local = None
        self.handle_axis_local = None
        self.handle_half_length = None
        self.handle_collider_path = None
        self.panel_half_thickness = 0.03
        self.ee_orientation_error = 999.0
        self.default_ee_goal_local = None
        # Cartesian targets are rate-limited in world coordinates.  The
        # low-level IK consumes incremental targets, so a large target jump at
        # a stage transition can overshoot the fingers into the door.
        self.ee_command_world = None
        self.finger_separation_local = None
        self.finger_closing_local = None
        self.grasp_orientation_quat = None
        self.ee_orientation_command = None
        self.active_grasp_quat = None
        self.grasp_close_start = -0.30
        self.q_open = None
        self.q_closed = None
        self.gripper_open_settle = 0
        self.center_substage = "CENTER_ORIENT"
        self.center_orientation_settle_steps = 0
        self.center_vertical_bias = 0.02
        self.center_depth_bias = -0.025
        self.center_standoff_world = None
        self.target_tip_mid_world = None
        self.center_geometry = None
        self.geometry_report = None
        self.center_metrics = {}
        self.failure_reason = None
        robot_body_names = [name.lower() for name in self.env.body_names]
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
        self.handle_align_settle_steps = 0
        self.grasp_settle_steps = 0
        self.handle_contact_last_seen[:] = -1000
        self.turn_start_handle_angle = 0.0
        self.turn_start_latch_position = 0.0
        self.turn_start_hinge_angle = 0.0
        self.turn_last_handle_angle = 0.0
        self.turn_last_hinge_angle = 0.0
        self.failure_reason = None
        self.center_substage = "CENTER_ORIENT"
        self.center_orientation_settle_steps = 0
        self.center_vertical_bias = 0.02
        self.center_depth_bias = -0.025
        self.gripper_open_settle = 0
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
        self.grasp_center_pos = self._physical_handle_grasp_position(handle_state)
        self.handle_lower_z = float(self.grasp_center_pos[2] - handle_radius)
        self.handle_upper_z = float(self.grasp_center_pos[2] + handle_radius)
        # ``normal`` points from the door toward the robot's initial side, so
        # the pre-door station lies along +normal from the handle.
        # The locomotion policy has a repeatable stopping offset near the
        # frame: the commanded 0.65 m standoff settles roughly 0.1--0.15 m
        # farther out once the feet touch the collision margin.  Keep the
        # geometric target here for logging, and use an explicit tolerance
        # below rather than driving the base into the frame.
        # Keep the base outside the door-frame collision margin.  The loaded
        # low-level policy settles near x=0.74 m when the target is roughly
        # 0.65 m from the handle; a 0.50 m target is inside that margin and
        # can never be reached by the locomotion controller.
        self.approach_distance = 0.65
        self.station_tolerance = 0.12
        self.station_xy = self.grasp_center_pos[:2] + self.approach_distance * normal
        self.door_forward = -self.door_normal
        self.initial_handle_pos = self.grasp_center_pos.copy()
        self.initial_door_root = door_root.copy()
        self.default_ee_goal_local = _as_np(state["ee_goal_cart_local"])[0]
        self.default_ee_base_local = _quat_rotate_inverse_np(
            base[3:7], _as_np(state["ee_pos"])[0] - base[:3]
        )
        self.ee_command_world = _as_np(state["ee_pos"])[0].copy()
        self.ee_orientation_command = _as_np(state["ee_orn"])[0].copy()
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
            }
            self.finger_separation_local = separation_local
            self.finger_closing_local = depth_local
            # The primary control point is the midpoint of the two real
            # fingertips.  gripper_link is recovered from this point by one
            # rigid transform; it is not placed at the handle center.
            self.grasp_ee_offset_local = tip_capture_center
            # Use the measured fingertip separation to choose which physical
            # jaw is upper/lower.  At q_open the fixed jaw is not guaranteed
            # to be the upper jaw; this is determined from the actual mesh.
            tip_sep_local = fixed_tip_center - moving_tip_center
            tip_sep_local /= max(float(np.linalg.norm(tip_sep_local)), 1.0e-8)
            tip_sep_world = _quat_rotate_np(ee_quat, tip_sep_local)
            separation_world = tip_sep_world
            align = _quat_align_vectors_np(
                separation_world,
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
                ee_quat,
            )
            aligned_closing = _quat_rotate_np(align, depth_local)
            desired_closing = -self.door_normal.copy(); desired_closing[2] = 0.0
            desired_closing /= max(float(np.linalg.norm(desired_closing)), 1.0e-8)
            aligned_closing[2] = 0.0
            aligned_closing /= max(float(np.linalg.norm(aligned_closing)), 1.0e-8)
            yaw_delta = math.atan2(
                float(np.cross(aligned_closing, desired_closing)[2]),
                float(np.dot(aligned_closing, desired_closing)),
            )
            half = 0.5 * yaw_delta
            roll_about_z = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)
            self.grasp_orientation_quat = _quat_mul_np(roll_about_z, align)
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
                f"depth_local={depth_local.round(4).tolist()} separation_local={separation_local.round(4).tolist()} "
                f"offset_grasp_world={self.grasp_ee_offset_world.round(4).tolist()} "
                f"measured_sep_world={measured_sep_world.round(4).tolist()} "
                f"direct_sep_world={direct_sep_world.round(4).tolist()} "
                f"target_sep_world={target_sep_world.round(4).tolist()} "
                f"target_depth_world={target_depth_world.round(4).tolist()}",
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
                    "tcp_capture_offset_local": capture_center.astype(float).tolist(),
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
        # Contact the panel away from the handle, toward the panel center.  The
        # target is stored in the panel frame so it follows the swinging door.
        push_point_world = self.grasp_center_pos.copy()
        push_point_world[:2] += 0.35 * (door_root[:2] - push_point_world[:2])
        push_point_world -= 0.12 * self.door_normal
        self.panel_contact_local = _quat_rotate_inverse_np(
            panel_state[3:7], push_point_world - panel_state[:3]
        )
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
        for body_name, key, tokens in (
            ("link06", "wrist", ("link06/collisions/mesh_0",)),
            # In the lab30 USD the fixed jaw is welded into link06 and its
            # decomposed collision meshes are fixed_convex_* descendants.
            ("link06", "fixed_finger", ("fixed_convex_",)),
            ("gripperMover", "moving_finger", ("moving_convex_",)),
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
                candidates = [str(p.GetPath()) for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())]
                raise RuntimeError(
                    f"required collider geometry not found for {key} under {body.GetPath()}; "
                    f"collision candidates={candidates[:20]}"
                )
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
        # The imported B2Z1 gripper uses the negative endpoint for maximum
        # opening (the two endpoints are otherwise symmetric around zero).
        # Prefer the endpoint farthest from the configured closed/default
        # position; ties are resolved toward the negative endpoint.
        self.q_open = float(finite[np.argmin(finite)])
        self.q_closed = float(np.clip(0.0, finite.min(), finite.max()))
        print(f"[doorman][geometry] jointGripper q_now={q_now:.4f} q_open={self.q_open:.4f} q_closed={self.q_closed:.4f}", flush=True)

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
        """Read the door-panel thickness from the loaded USD when available."""
        try:
            import omni.usd
            from pxr import Gf, Usd, UsdGeom
            stage = omni.usd.get_context().get_stage()
            panel_prim = None
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
                        break
            print(f"[doorman][geometry] panel_half_thickness={self.panel_half_thickness:.4f}", flush=True)
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
        horizontal_error = max(
            float(np.linalg.norm((upper_c - grasp_center)[:2])),
            float(np.linalg.norm((lower_c - grasp_center)[:2])),
        )
        z_span = abs(float(fixed_c[2] - moving_c[2]))
        handle_radius = max(float(self.handle_upper_z - self.handle_lower_z) * 0.5, 1.0e-4)
        upper_gap = upper_c[2] - (grasp_center[2] + handle_radius)
        lower_gap = (grasp_center[2] - handle_radius) - lower_c[2]
        depth_axis_w = _quat_rotate_np(_as_np(state["ee_orn"])[0], self.open_template["depth_axis"])
        tip_depth = float(np.quantile(np.concatenate((fixed, moving)) @ depth_axis_w, 0.95))
        depth = abs(tip_depth - float(np.dot(grasp_center, depth_axis_w)))
        panel_center = _as_np(state["door_body_state_w"])[0, self.panel_body_idx, :3]
        all_points = np.concatenate(list(points.values()), axis=0)
        clearance = float(np.min((all_points - panel_center) @ self.door_normal) - self.panel_half_thickness)
        return {
            "valid": True,
            "horizontal_error": horizontal_error,
            "z_span": z_span,
            "upper_gap": float(upper_gap),
            "lower_gap": float(lower_gap),
            "grasp_depth": float(depth),
            "panel_clearance": clearance,
            "fixed_center": fixed_c,
            "moving_center": moving_c,
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
        return cmd, float(np.linalg.norm(error_w[:2])), abs(yaw_error)

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
                float(((local_rpy[0] - math.pi / 2.0 + math.pi) % (2.0 * math.pi)) - math.pi),
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

    def command(self, state):
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
        ee_orientation_error = 999.0
        if state.get("ee_orn") is not None:
            ee_orientation_error = self._quat_distance(
                _as_np(state["ee_orn"])[0], desired_grasp_quat
            )
        self.ee_orientation_error = ee_orientation_error
        if self.stage in (Stage.CENTER_HANDLE, Stage.GRASP_HANDLE, Stage.TURN_HANDLE):
            self.active_grasp_quat = self._slew_orientation(desired_grasp_quat)
        else:
            self.active_grasp_quat = None
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
        if self.stage == Stage.PREGRASP:
            # Stop 10 cm on the robot side of the physical capsule before
            # inserting the jaws.  The distal-jaw X offset is only applied
            # after this standoff is reached.
            ee_target_world = grasp_center + 0.10 * self.door_normal
            ee_target_world[2] += 0.05
            self.pregrasp_target_world = ee_target_world.copy()
        elif self.stage == Stage.CENTER_HANDLE:
            # Keep the real jaws at q_open.  The target is derived from the
            # sampled collider capture frame and is approached in sub-stages.
            desired_capture = (
                self.target_tip_mid_world.copy()
                if self.target_tip_mid_world is not None
                else grasp_center.copy()
            )
            if self.center_substage in ("CENTER_ORIENT", "CENTER_STANDOFF"):
                desired_capture = self.center_standoff_world.copy()
            # Align the open jaws vertically while they are still at the
            # robot-side standoff.  Waiting until CENTER_INSERT leaves the
            # lower jaw below the handle and causes the upper jaw to strike
            # the lever first.
            if self.target_tip_mid_world is None:
                desired_capture[2] += float(self.center_vertical_bias)
            current_capture = self._current_capture_center(state)
            current_ee = _as_np(state["ee_pos"])[0]
            if self.center_substage == "CENTER_ORIENT":
                # Hold the fixed outer standoff while rotating.  The target
                # must not follow any transient IK-induced EE drift.
                ee_target_world = self._effective_grasp_target(
                    self.center_standoff_world, self.grasp_orientation_quat
                )
            elif current_capture is not None:
                # Closed-loop rigid translation using the actual jaw surfaces.
                # This compensates only for measured IK/body-frame tracking;
                # it is not a fixed grasp offset and cannot alter door state.
                # The height correction is latched while the jaws are still
                # clear of the handle.  Recomputing it after contact would
                # feed collision reaction forces back into the target.
                if self.target_tip_mid_world is None:
                    desired_capture[2] += float(self.center_vertical_bias)
                if self.center_substage == "CENTER_INSERT":
                    insert_metrics = self._geometry_metrics(state, grasp_center)
                    contact_values = state.get("handle_finger_contact_forces_w")
                    contact_norms = (
                        np.linalg.norm(_as_np(contact_values)[0], axis=-1)
                        if contact_values is not None
                        else np.zeros(2, dtype=np.float32)
                    )
                    if insert_metrics.get("valid") and not np.any(contact_norms >= self.grasp_contact_threshold):
                        # Before first physical contact, keep the two measured
                        # inner surfaces symmetrically above/below the real
                        # capsule.  Once either jaw touches the handle this
                        # correction is frozen to avoid feeding reaction force
                        # back into the target.
                        z_bias = 0.5 * (
                            float(insert_metrics.get("lower_gap", 0.0))
                            - float(insert_metrics.get("upper_gap", 0.0))
                        )
                        if self.target_tip_mid_world is None:
                            desired_capture[2] += float(np.clip(z_bias, -0.08, 0.08))
                # The measured jaw capture surfaces sit a few centimetres on
                # the robot side of the handle even when the TCP target is
                # nominally centered.  Apply the measured -X insertion
                # offset before contact; panel clearance remains a hard
                # safety check in ``advance``.
                # The control point is the measured fingertip midpoint.  Do
                # not close the loop on the old inner-surface centroid: that
                # point is not rigidly coincident with the fingertip target.
                desired_capture = self.target_tip_mid_world.copy()
                ee_target_world = self._effective_grasp_target(
                    desired_capture, self.grasp_orientation_quat
                )
            else:
                ee_target_world = self._effective_grasp_target(
                    desired_capture, self.grasp_orientation_quat
                )
            self.handle_center_target_world = ee_target_world.copy()
        elif self.stage == Stage.GRASP_HANDLE:
            ee_target_world = self._effective_grasp_target(grasp_center)
            # First close to an intermediate jaw opening.  This lets both
            # fingers settle around the lever instead of sweeping one finger
            # through it when the target jumps from fully open to closed.
            if self.stage_steps < 35:
                fraction = min(1.0, self.stage_steps / 34.0)
                gripper = (1.0 - fraction) * float(self.q_open) + fraction * float(self.q_closed)
            else:
                gripper = float(self.q_closed)
        elif self.stage == Stage.TURN_HANDLE:
            # A real handle response is required by advance(); no door state is
            # written here.  Downward motion rotates the X-axis handle joint.
            ee_target_world = self._effective_grasp_target(grasp_center)
            ee_target_world[2] -= min(0.025, 0.00015 * float(self.stage_steps))
            gripper = 0.0
        elif self.stage == Stage.RELEASE_HANDLE:
            ee_target_world = self._effective_grasp_target(grasp_center)
            gripper = -0.785
        elif self.stage == Stage.RETRACT_ARM:
            ee_target_world = grasp_center + 0.20 * self.door_normal
            ee_target_world[2] += 0.10
            # Open while separating from the handle, then close before pushing.
            gripper = -0.785 if self.stage_steps < 40 else 0.0
        elif self.stage in (Stage.CONTACT_PANEL, Stage.PUSH_OPEN):
            ee_target_world = panel_contact
            gripper = 0.0
        elif self.stage == Stage.THROUGH:
            # Keep the arm compact while walking through; the explicit reset
            # stage below verifies the final default pose after passage.
            ee_local_override = self.default_ee_goal_local
            gripper = 0.0
        elif self.stage in (Stage.RESET_ARM, Stage.SUCCESS):
            ee_local_override = self.default_ee_goal_local
            gripper = 0.0

        # Keep the reset-pose override exact, but slew every Cartesian target
        # used for physical interaction so the arm approaches the handle and
        # panel continuously instead of teleporting its IK goal.
        if ee_local_override is None:
            max_step = 0.002 if self.stage == Stage.CENTER_HANDLE and self.center_substage == "CENTER_INSERT" else 0.012
            ee_target_world = self._slew_ee_target(ee_target_world, max_step=max_step)
        else:
            self.ee_command_world = _as_np(state["ee_pos"])[0].copy()

        if self.stage == Stage.TURN_HANDLE:
            # Keep the handle depressed while applying a small world-aligned
            # push.  This retracts the latch and creates a real gap before the
            # gripper is allowed to release.
            base_cmd = self._travel_command(state, 0.08)
        elif self.stage == Stage.PUSH_OPEN:
            base_cmd = self._travel_command(state, 0.30)
        elif self.stage == Stage.THROUGH:
            base_cmd = self._travel_command(state, 0.55)
        elif self.stage not in (Stage.APPROACH, Stage.ALIGN):
            base_cmd[:2] = 0.0
            if self.stage != Stage.ALIGN:
                base_cmd[2] = 0.0

        # The arm must settle against a stationary base before a grasp or
        # handle torque is attempted.  This also prevents contact reaction
        # forces from moving the robot out of its station.
        if self.stage in (
            Stage.PREGRASP,
            Stage.CENTER_HANDLE,
            Stage.GRASP_HANDLE,
            Stage.TURN_HANDLE,
            Stage.RELEASE_HANDLE,
            Stage.RETRACT_ARM,
            Stage.CONTACT_PANEL,
            Stage.RESET_ARM,
        ):
            base_cmd[:] = 0.0

        ee_local = (
            torch.as_tensor(ee_local_override, device=self.env.device, dtype=torch.float32).reshape(1, 3)
            if ee_local_override is not None
            else self._ee_local_target(ee_target_world)
        )
        orn = np.zeros((1, 3), dtype=np.float32)
        if self.stage in (Stage.CENTER_HANDLE, Stage.GRASP_HANDLE, Stage.TURN_HANDLE):
            orn[0] = self._grasp_orientation_delta(ee_local[0].detach().cpu().numpy(), grasp_quat)
        self.runner.set_teleop_targets(
            torch.as_tensor(base_cmd, device=self.env.device).reshape(1, 3),
            ee_goal_cart=ee_local,
            ee_orn_delta_rpy=torch.as_tensor(orn, device=self.env.device),
            gripper_target=torch.tensor([[gripper]], device=self.env.device),
            ee_goal_orn_quat_world=(
                torch.as_tensor(self.active_grasp_quat, device=self.env.device)
                .reshape(1, 4)
                if self.stage in (Stage.CENTER_HANDLE, Stage.GRASP_HANDLE, Stage.TURN_HANDLE)
                else None
            ),
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

    def _effective_grasp_target(self, grasp_center, ee_quat=None):
        target = np.asarray(grasp_center, dtype=np.float32).copy()
        if ee_quat is None:
            offset_world = getattr(self, "grasp_ee_offset_world", np.zeros(3, dtype=np.float32))
        else:
            offset_world = _quat_rotate_np(ee_quat, self.grasp_ee_offset_local)
        target -= offset_world
        return target

    def advance(self, state, station_dist, yaw_error):
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
            float(np.linalg.norm(ee - self.pregrasp_target_world))
            if ee is not None and self.pregrasp_target_world is not None
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
        if handle_contacts is not None:
            handle_contact_norms = np.linalg.norm(_as_np(handle_contacts)[0], axis=-1)
            handle_points = state.get("handle_finger_contact_points_w")
            handle_points = _as_np(handle_points)[0] if handle_points is not None else None
            if handle_points is None:
                handle_contact_norms[:] = 0.0
            else:
                for finger_idx in range(2):
                    if not self._contact_is_on_handle_capsule(
                        handle_points[finger_idx], door_bodies[self.handle_body_idx, :7]
                    ):
                        handle_contact_norms[finger_idx] = 0.0
        speed = float(np.linalg.norm(_as_np(state["root_states"])[0, 7:10]))
        base = _as_np(state["root_states"])[0]
        base_xy = base[:2]
        ee_base_local = _quat_rotate_inverse_np(base[3:7], ee - base[:3])
        reset_arm_dist = float(np.linalg.norm(ee_base_local - self.default_ee_base_local))
        ee_orientation_error = float(self.ee_orientation_error)
        if self.stage == Stage.APPROACH and station_dist < self.station_tolerance and yaw_error < 0.10 and speed < 0.05:
            self.stage = Stage.ALIGN; self.stage_steps = 0
            self.align_settle_steps = 0
        elif self.stage == Stage.ALIGN:
            self.align_settle_steps = self.align_settle_steps + 1 if yaw_error < 0.10 and speed < 0.10 else 0
            if self.align_settle_steps >= 10:
                self.stage = Stage.PREGRASP; self.stage_steps = 0
        elif self.stage == Stage.PREGRASP and pregrasp_dist < 0.10 and speed < 0.10:
            self.stage = Stage.CENTER_HANDLE; self.stage_steps = 0
        elif self.stage == Stage.CENTER_HANDLE:
            if self.q_open is None:
                self.failure_reason = "geometry_q_open_unavailable"
                self.stage = Stage.FAILURE
            actual_q = float(_as_np(state["gripper_position"])[0, 0])
            metrics = self._geometry_metrics(state, grasp_center)
            self.center_metrics = metrics
            if self.center_substage == "CENTER_ORIENT":
                # Do not start translating the jaws while the arm is still
                # rotating.  A single low orientation error can be a
                # transient IK crossing; require a stationary, continuously
                # settled pose before any insertion toward the panel.
                angular_speed = float(np.linalg.norm(_as_np(state["root_states"])[0, 10:13]))
                # This controller's damped IK settles at about 0.10 rad on
                # this arm; use a margin above that measured floor while
                # still requiring 20 consecutive stationary samples.
                if ee_orientation_error < 0.13 and speed < 0.01 and angular_speed < 0.01:
                    self.center_orientation_settle_steps += 1
                else:
                    self.center_orientation_settle_steps = 0
                if self.center_orientation_settle_steps >= 20:
                    self.center_substage = "CENTER_STANDOFF"; self.stage_steps = 0
            elif self.center_substage == "CENTER_STANDOFF":
                if metrics.get("valid") and not np.any(handle_contact_norms >= self.grasp_contact_threshold):
                    # Balance the measured open-jaw clearances once, before
                    # insertion.  The nominal 20 mm value is only a fallback
                    # for the first few frames before the collider metrics are
                    # populated.
                    measured_bias = 0.5 * (
                        float(metrics.get("lower_gap", 0.0))
                        - float(metrics.get("upper_gap", 0.0))
                    )
                    self.center_vertical_bias = float(np.clip(measured_bias + 0.02, -0.06, 0.06))
                angular_speed = float(np.linalg.norm(_as_np(state["root_states"])[0, 10:13]))
                if speed < 0.01 and angular_speed < 0.01 and abs(actual_q - self.q_open) < 0.02:
                    self.gripper_open_settle += 1
                else:
                    self.gripper_open_settle = 0
                if self.gripper_open_settle >= 20:
                    self.center_substage = "CENTER_INSERT"; self.stage_steps = 0
            elif self.center_substage == "CENTER_INSERT" and metrics.get("valid"):
                tip_metrics = self._tip_metrics(state, grasp_center)
                if metrics["panel_clearance"] < 0.005:
                    self.failure_reason = f"center_panel_clearance={metrics['panel_clearance']:.4f}"
                    self.stage = Stage.FAILURE
                elif tip_metrics.get("valid") and tip_metrics["tip_midpoint_error"] < 0.025 and ee_orientation_error < 0.20:
                    self.center_substage = "CENTER_SETTLE"; self.stage_steps = 0
            elif self.center_substage == "CENTER_SETTLE":
                tip_metrics = self._tip_metrics(state, grasp_center)
                aligned = (
                    abs(actual_q - self.q_open) < 0.02
                    and speed < 0.01
                    and tip_metrics.get("valid", False)
                    and tip_metrics["horizontal_error"] <= 0.012
                    and tip_metrics["upper_gap"] >= 0.003
                    and tip_metrics["lower_gap"] >= 0.003
                    and tip_metrics["tip_midpoint_error"] <= 0.012
                    and metrics["panel_clearance"] >= 0.005
                    and ee_orientation_error < 0.20
                )
                self.handle_align_settle_steps = self.handle_align_settle_steps + 1 if aligned else 0
                if self.handle_align_settle_steps >= 20:
                    self.stage = Stage.GRASP_HANDLE; self.stage_steps = 0
                    self.grasp_settle_steps = 0
        elif self.stage == Stage.GRASP_HANDLE:
            for finger_idx, force in enumerate(handle_contact_norms):
                if force >= self.grasp_contact_threshold:
                    self.handle_contact_last_seen[finger_idx] = self.stage_steps
            bilateral_recent = bool(np.all(self.stage_steps - self.handle_contact_last_seen <= 12))
            centered_and_closed = bilateral_recent and abs(gripper_pos - self.q_closed) < 0.20
            self.grasp_settle_steps = self.grasp_settle_steps + 1 if centered_and_closed else 0
            if self.grasp_settle_steps >= 20:
                self.turn_start_handle_angle = handle_joint
                self.turn_start_latch_position = latch_joint
                self.turn_start_hinge_angle = hinge
                self.turn_last_handle_angle = handle_joint
                self.turn_last_hinge_angle = hinge
                self.stage = Stage.TURN_HANDLE; self.stage_steps = 0
        elif (
            self.stage == Stage.TURN_HANDLE
            and ee_dist < 0.10
            and abs(handle_joint - self.turn_start_handle_angle) > 0.20
            and abs(hinge - self.turn_start_hinge_angle) > math.radians(1.0)
        ):
            self.stage = Stage.RELEASE_HANDLE; self.stage_steps = 0
        if self.stage == Stage.TURN_HANDLE:
            self.turn_last_handle_angle = handle_joint
            self.turn_last_hinge_angle = hinge
        elif self.stage == Stage.RELEASE_HANDLE and gripper_pos < -0.55 and self.stage_steps >= 20:
            self.stage = Stage.RETRACT_ARM; self.stage_steps = 0
        elif (
            self.stage == Stage.RETRACT_ARM
            and ee_dist > 0.12
            and abs(gripper_pos) < 0.20
            and self.stage_steps >= 50
        ):
            self.stage = Stage.CONTACT_PANEL; self.stage_steps = 0
        elif self.stage == Stage.CONTACT_PANEL and panel_dist < 0.06 and abs(gripper_pos) < 0.20:
            self.stage = Stage.PUSH_OPEN; self.stage_steps = 0
        elif self.stage == Stage.PUSH_OPEN and abs(hinge) > math.radians(80.0):
            self.stage = Stage.THROUGH; self.stage_steps = 0
        elif self.stage == Stage.THROUGH and float(np.dot(base_xy - self.initial_handle_pos[:2], self.door_forward[:2])) > 1.5:
            self.stage = Stage.RESET_ARM; self.stage_steps = 0
        elif self.stage == Stage.RESET_ARM and reset_arm_dist < 0.10 and speed < 0.10:
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
        return self.stage


def _serialize_episode(controller, runner, episode_id, seed, output_dir, max_steps, save_failures, debug_stage=None):
    state = controller.reset()
    records = {k: [] for k in (
        "timestamp", "step", "stage_id", "task_phase_id", "done", "success", "base_pose_w", "base_velocity_w",
        "joint_position", "joint_velocity", "ee_pose_w", "ee_velocity_w", "gripper_position",
        "door_root_pose_w", "door_joint_position", "door_joint_velocity", "handle_pose_w",
        "base_command", "ee_goal_cart_local", "ee_goal_delta_rpy", "gripper_command",
        "high_level_action", "low_level_action", "rgb_frame_index", "center_substage_id",
        "q_open", "gripper_actual", "center_horizontal_error", "center_upper_gap",
        "center_lower_gap", "center_grasp_depth", "center_panel_clearance",
        "handle_contact_force")}
    frames = []
    for step in range(max_steps):
        if controller.stage in (Stage.SUCCESS, Stage.FAILURE):
            break
        high, dist, yaw_error = controller.command(state)
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
            print(
                f"[doorman] step={step} stage={controller.stage.name} "
                f"base_command={high[:3].round(3).tolist()} station_dist={dist:.3f} "
                f"yaw_error={yaw_error:.3f} ee={ee_dbg} goal={goal_dbg} "
                f"base_pos={base_pos_dbg} base_speed={base_speed_dbg:.3f} "
                f"stage_steps={controller.stage_steps} grasp_contact_force={contact_dbg:.3f} "
                f"grasp_orientation_error={controller.ee_orientation_error:.3f} "
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
        )
        state_np = {key: _as_np(state[key]) for key in tensor_state_keys}
        records["timestamp"].append(step * float(runner.env.dt)); records["step"].append(step)
        records["stage_id"].append(stage_before)
        records["task_phase_id"].append(STAGE_TO_TASK_PHASE.get(controller.stage, -1))
        records["done"].append(False); records["success"].append(False)
        records["base_pose_w"].append(state_np["root_states"][0, :7]); records["base_velocity_w"].append(state_np["root_states"][0, 7:])
        records["joint_position"].append(state_np["dof_pos"][0]); records["joint_velocity"].append(state_np["dof_vel"][0])
        ee_pos = state_np["ee_pos"] if state_np["ee_pos"] is not None else np.zeros((1, 3), dtype=np.float32)
        ee_orn = state_np["ee_orn"] if state_np["ee_orn"] is not None else np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (1, 1))
        ee_vel = state_np["ee_vel"] if state_np["ee_vel"] is not None else np.zeros((1, 6), dtype=np.float32)
        records["ee_pose_w"].append(np.concatenate([ee_pos[0], ee_orn[0]])); records["ee_velocity_w"].append(ee_vel[0])
        records["gripper_position"].append(state_np["gripper_position"][0]); records["door_root_pose_w"].append(state_np["door_root_pose_w"][0])
        records["door_joint_position"].append(state_np["door_joint_position"][0]); records["door_joint_velocity"].append(state_np["door_joint_velocity"][0])
        handle_pose = state_np["door_body_state_w"][0, controller.handle_body_idx, :7]; records["handle_pose_w"].append(handle_pose)
        records["base_command"].append(high[:3]); records["ee_goal_cart_local"].append(high[3:6]); records["ee_goal_delta_rpy"].append(high[6:9]); records["gripper_command"].append(high[9:10]); records["high_level_action"].append(high)
        center_substages = ("CENTER_ORIENT", "CENTER_STANDOFF", "CENTER_INSERT", "CENTER_SETTLE")
        metrics = controller.center_metrics
        records["center_substage_id"].append(center_substages.index(controller.center_substage) if controller.center_substage in center_substages else -1)
        records["q_open"].append(float(controller.q_open if controller.q_open is not None else np.nan))
        records["gripper_actual"].append(float(state_np["gripper_position"][0, 0]))
        for key, record_key in (
            ("horizontal_error", "center_horizontal_error"),
            ("upper_gap", "center_upper_gap"),
            ("lower_gap", "center_lower_gap"),
            ("grasp_depth", "center_grasp_depth"),
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
        frame_idx = len(frames)
        frame = result.get("rgb")
        records["rgb_frame_index"].append(frame_idx if frame is not None else -1)
        if frame is not None:
            frames.append(frame)
        finite_state = all(
            value is None or np.isfinite(value).all()
            for value in state_np.values()
            if isinstance(value, np.ndarray)
        )
        if not finite_state or not np.isfinite(low).all() or not np.isfinite(high).all():
            controller.failure_reason = "non_finite_state_or_action"
            controller.stage = Stage.FAILURE
            print("[doorman] FAILURE: non-finite state or action", flush=True)
            break
        state = result["state"]
        controller.advance(state, dist, yaw_error)
        if debug_stage == "center_handle" and controller.stage == Stage.TURN_HANDLE:
            controller.stage = Stage.SUCCESS
            controller.failure_reason = None
            print("[doorman][debug] CENTER_HANDLE and bilateral GRASP_HANDLE validation passed", flush=True)
    success = controller.stage == Stage.SUCCESS
    if records["done"]:
        records["done"][-1] = True; records["success"][-1] = success
    if not success and not save_failures:
        print(f"[doorman] episode {episode_id} failed at stage={controller.stage.name}")
        return False
    episode_dir = output_dir / f"episode_{episode_id:06d}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    rgb_dir = episode_dir / "rgb"; rgb_dir.mkdir()
    for i, frame in enumerate(frames): _save_rgb(frame, rgb_dir / f"{i:06d}.png")
    arrays = {k: np.asarray(v) for k, v in records.items()}
    np.savez_compressed(episode_dir / "trajectory.npz", **arrays)
    metadata = {
        "schema_version": "doorman_raw_v1", "episode_id": episode_id, "seed": seed, "robot": "b2z1",
        "sim_dt": float(runner.env.dt), "control_dt": float(runner.env.dt), "success": bool(success),
        "stage_names": STAGE_NAMES,
        "task_phase_names": TASK_PHASE_NAMES,
        "camera": {
            "enabled": bool(frames),
            "format": "png",
            "channels": "rgb",
            "frame_count": len(frames),
            "source": getattr(runner, "_rgb_frame_source", None),
            "config": str(getattr(args, "rgb_camera_config", "") or "") or None,
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
        "control": {"action_delay_mode": "undelayed", "high_level_action_dim": 10, "low_level_action_dim": int(arrays["low_level_action"].shape[-1]) if arrays["low_level_action"].ndim > 1 else 0},
        "assist_mode": "none",
        "door_effort_assist": False,
        "kinematic_assist": False,
        "data_quality": "physical_only",
        "failure_reason": controller.failure_reason,
    }
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[doorman] wrote {episode_dir} success={success} steps={len(arrays['step'])} frames={len(frames)}")
    return success


def main():
    args = _parser().parse_args()
    if args.num_episodes < 1 or args.max_steps < 1:
        raise SystemExit("--num_episodes and --max_steps must be positive")
    _default_paths(args)
    if args.output_dir is None:
        args.output_dir = SCRIPT_ROOT.parent.parent / "datasets" / "doorman" / "episodes"
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
    print(f"[doorman] complete: {successes}/{args.num_episodes} successful episodes")


if __name__ == "__main__":
    main()
