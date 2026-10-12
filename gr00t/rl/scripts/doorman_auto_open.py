"""Rule-based DoorMan door opening and raw trajectory collection.

This entry point drives the existing B2Z1 low-level policy through the public
teleop target API.  It intentionally does not synthesize keyboard events or
feed the 10-D high-level action directly to Isaac Lab.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import faulthandler
import hashlib
import json
import math
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import deque
from enum import IntEnum
from pathlib import Path

if (
    __name__ == "__main__"
    and os.environ.get("AUTO_OPEN_FIXED_ENTRY") != "run_auto_open.sh"
):
    raise SystemExit(
        "Direct AutoOpen invocation is disabled. Use the maintained "
        "gr00t/rl/scripts/run_auto_open.sh entry."
    )

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
from legged_gym.utils.steady_path_worker import (  # noqa: E402
    initialize_steady_path_worker,
    solve_deferred_path_request,
    solve_steady_inverse_request,
)
from gr00t.rl.envs.door.b2z1_reset_geometry import (  # noqa: E402
    quaternion_xyzw_to_matrix,
    solve_b2z1_arm_ik,
)
from gr00t.rl.scripts.auto_open_target_noise import (  # noqa: E402
    EpisodeTargetNoise,
    SemanticTargetNoiseConfig,
)
from gr00t.rl.data_collection.auto_open_standard import (  # noqa: E402
    TERMINAL_ARM_JOINT_SPEED_RAD_S,
    TERMINAL_BASE_ANGULAR_SPEED_RAD_S,
    TERMINAL_BASE_LINEAR_SPEED_M_S,
    TERMINAL_DOOR_SPEED_RAD_S,
    TERMINAL_GRIPPER_SPEED_RAD_S,
    TERMINAL_STABILITY_FRAMES,
)


class Stage(IntEnum):
    APPROACH = 1
    PREGRASP = 2
    OPEN_GRIPPER = 3
    CENTER_HANDLE = 4
    GRASP_HANDLE = 5
    TURN_HANDLE = 6
    RETURN_HANDLE = 7
    OPEN_AND_CLEAR = 8
    RETREAT_ARM = 9
    RESTORE_GRIPPER = 10
    ALIGN_PANEL = 11
    CONTACT_PANEL = 12
    PUSH_OPEN = 13
    PUSH_ARM_RETRACT = 14
    BASE_ADVANCE = 15
    THROUGH = 16
    SUCCESS = 17
    FAILURE = 18


class DeferredSteadyInverse(RuntimeError):
    def __init__(
        self,
        controller,
        state,
        desired_position_world,
        desired_orientation_world,
        label,
        position_tolerance_m,
    ):
        super().__init__(f"deferred steady inverse: {label}")
        self.controller = controller
        self.state = state
        self.desired_position_world = np.asarray(
            desired_position_world, dtype=np.float64
        ).copy()
        self.desired_orientation_world = np.asarray(
            desired_orientation_world, dtype=np.float64
        ).copy()
        self.label = str(label)
        self.position_tolerance_m = position_tolerance_m


class DeferredPathSolve(Exception):
    def __init__(self, controller, kind, label, request):
        super().__init__(f"deferred steady path: {label}")
        self.controller = controller
        self.kind = str(kind)
        self.label = str(label)
        self.request = request


# These stages do not prescribe base travel.  They inherit and hold the last
# target owned by the preceding movement stage.
BASE_HOLD_STAGES = frozenset(
    {
        Stage.PREGRASP,
        Stage.OPEN_GRIPPER,
        Stage.CENTER_HANDLE,
        Stage.GRASP_HANDLE,
        Stage.TURN_HANDLE,
        Stage.RETURN_HANDLE,
        Stage.OPEN_AND_CLEAR,
        Stage.RETREAT_ARM,
        Stage.RESTORE_GRIPPER,
        Stage.CONTACT_PANEL,
        Stage.PUSH_OPEN,
        Stage.PUSH_ARM_RETRACT,
    }
)

MAX_BASE_PLANAR_SPEED = 0.4
MAX_BASE_YAW_RATE = 0.6
BASE_STATION_TOLERANCE_M = 0.12
BASE_YAW_TOLERANCE_RAD = 0.08
THROUGH_STOP_DISTANCE_M = 0.08
THROUGH_STOP_YAW_RAD = 0.05
ROBOT_TRUNK_REAR_X = -0.284
THROUGH_TAIL_TARGET_DISTANCE = 1.0
CENTER_HANDLE_MIN_AXIAL_OVERLAP = 0.010
CENTER_HANDLE_CHANNEL_DESIGN_ERROR = 0.008
GRASP_FIXED_JAW_BALANCE_LIMIT = 0.020
GRASP_FIXED_JAW_DESIGN_GAP = 0.019


def _base_pose_accepted(distance_m, yaw_error_rad):
    return bool(
        float(distance_m) < BASE_STATION_TOLERANCE_M
        and abs(float(yaw_error_rad)) < BASE_YAW_TOLERANCE_RAD
    )


def _resolved_door_prim_path(env, env_id):
    """Return the scene-owned door root path for one environment."""
    resolver = getattr(env.door, "prim_path_for_env", None)
    if resolver is not None:
        return str(resolver(int(env_id)))
    return f"/World/envs/env_{int(env_id)}/door"


def _through_stop_latch(latched, distance_m, yaw_error_rad):
    """Latch zero velocity inside the target core; release outside acceptance."""
    if latched:
        return _base_pose_accepted(distance_m, yaw_error_rad)
    return bool(
        float(distance_m) < THROUGH_STOP_DISTANCE_M
        and abs(float(yaw_error_rad)) < THROUGH_STOP_YAW_RAD
    )


def _approach_stop_latch(
    latched,
    distance_m,
    yaw_error_rad,
    planar_speed_m_s,
    angular_speed_rad_s,
):
    """Own the commanded stop while APPROACH verifies measured stability."""
    if latched:
        return _base_pose_accepted(distance_m, yaw_error_rad)
    return bool(
        _base_pose_accepted(distance_m, yaw_error_rad)
        and float(planar_speed_m_s) < 0.02
        and float(angular_speed_rad_s) < 0.05
    )


def _solve_center_vertical_shift(channel_error, fixed_jaw_gap):
    """Closest safe vertical shift satisfying channel and grasp travel limits."""
    error = float(channel_error)
    gap = max(float(fixed_jaw_gap), 0.0)
    needed = max(abs(error) - CENTER_HANDLE_CHANNEL_DESIGN_ERROR, 0.0)
    available = max(GRASP_FIXED_JAW_DESIGN_GAP - gap, 0.0)
    magnitude = min(needed, available)
    shift = -math.copysign(magnitude, error) if magnitude > 0.0 else 0.0
    residual = error + shift
    return {
        "shift_m": shift,
        "predicted_channel_error_m": residual,
        "predicted_fixed_jaw_gap_m": gap + magnitude,
        "acceptance_feasible": abs(residual) <= 0.010,
    }


def _limit_base_commands(commands):
    """Apply the shared base command contract to one or more commands."""
    limited = np.asarray(commands, dtype=np.float32).copy()
    planar = limited[..., :2]
    planar_norm = np.linalg.norm(planar, axis=-1, keepdims=True)
    scale = np.minimum(
        1.0,
        MAX_BASE_PLANAR_SPEED / np.maximum(planar_norm, 1.0e-8),
    )
    limited[..., :2] = planar * scale
    limited[..., 2] = np.clip(
        limited[..., 2], -MAX_BASE_YAW_RATE, MAX_BASE_YAW_RATE
    )
    return limited


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
        clear_stage_on_close=True,
    )
    parser.add_argument(
        "--batched_envs",
        type=int,
        default=0,
        help="Run independent AutoOpen controllers in one vectorized SimulationApp.",
    )
    parser.add_argument(
        "--batch_active_env_ids",
        default=None,
        help=(
            "Comma-separated env ids to execute inside a fully initialized batch. "
            "Diagnostic only; all environments are still created to preserve randomization."
        ),
    )
    parser.add_argument(
        "--steady_path_workers",
        type=int,
        default=0,
        help="Exact CPU workers used to batch PREGRASP steady-path solves.",
    )
    parser.add_argument(
        "--batch_contact_ik_backend",
        choices=("scalar", "exact_workers"),
        default="scalar",
        help="CONTACT_PANEL steady-inverse backend used by batched collection.",
    )
    parser.add_argument(
        "--batch_path_backend",
        choices=("scalar", "exact_workers"),
        default="scalar",
        help=(
            "Backend for PREGRASP and RELEASE path solves in batched collection."
        ),
    )
    parser.add_argument(
        "--batch_panel_clearance_backend",
        choices=("scalar", "gpu"),
        default="gpu",
        help="Backend for batched CONTACT_PANEL/PUSH_OPEN finger clearance.",
    )
    parser.add_argument(
        "--batch_geometry_fallback_backend",
        choices=("scalar", "numpy_batch"),
        default="numpy_batch",
        help="Backend for guarded CENTER_HANDLE geometry fallback rows.",
    )
    parser.add_argument(
        "--target_noise",
        action="append",
        default=[],
        metavar="NAME=MAX_ABS",
        help=(
            "Set one episode-scoped task-semantic target variation bound. "
            "Repeat for independent variables; omitted variables remain zero."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--standard_dataset_dir",
        type=Path,
        default=None,
        help=(
            "Opt-in DATA_STANARD.md output root. This passive collector is "
            "disabled by default and does not alter AutoOpen control."
        ),
    )
    parser.add_argument(
        "--camera_mount_position_noise_m",
        type=float,
        default=0.0,
        help=(
            "Standard-data-only radius of the episode-constant camera mount "
            "position error in each parent-link frame."
        ),
    )
    parser.add_argument(
        "--camera_mount_rotation_noise_deg",
        type=float,
        default=0.0,
        help=(
            "Standard-data-only maximum episode-constant camera mount "
            "rotation angle in each nominal camera frame."
        ),
    )
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument(
        "--disable_rgb_cameras",
        action="store_true",
        help="Run physics-only without creating RGB sensors or render products.",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Write per-episode metadata without trajectories or RGB artifacts.",
    )
    parser.add_argument(
        "--robot_side",
        choices=("positive_x", "negative_x"),
        default="positive_x",
        help="Side of the door where the robot starts; positive_x is the default collection setup.",
    )
    parser.add_argument("--position_only_center", action="store_true", help="Keep EE orientation fixed during CENTER_HANDLE.")
    parser.add_argument(
        "--disable_push_base_assist",
        action="store_true",
        help="Keep the base stationary while the arm pushes the door open.",
    )
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
    if args.standard_dataset_dir is not None:
        if args.disable_rgb_cameras:
            raise SystemExit("standard data collection requires both RGB cameras")
        nominal_camera_config = Path(
            LOWLEVEL_ROOT / "legged_gym" / "scripts" / "rgb_cameras_dataset.json"
        )
        args.nominal_rgb_camera_config = str(nominal_camera_config)
        from gr00t.rl.data_collection.camera_mount_randomization import (
            randomize_camera_mounts,
        )

        nominal_config = json.loads(nominal_camera_config.read_text(encoding="utf-8"))
        actual_config, mount_randomization = randomize_camera_mounts(
            nominal_config,
            seed=int(args.seed if args.seed is not None else 0),
            position_radius_m=args.camera_mount_position_noise_m,
            rotation_angle_deg=args.camera_mount_rotation_noise_deg,
        )
        args.camera_mount_randomization = mount_randomization
        runtime_config = (
            Path(args.output_dir)
            / f"runtime_rgb_cameras_seed{int(args.seed if args.seed is not None else 0)}.json"
        )
        runtime_config.parent.mkdir(parents=True, exist_ok=True)
        runtime_config.write_text(
            json.dumps(actual_config, indent=2), encoding="utf-8"
        )
        args.rgb_camera_config = str(runtime_config)
    elif args.disable_rgb_cameras:
        args.rgb_camera_config = None
    elif not args.rgb_camera_config:
        args.rgb_camera_config = str(LOWLEVEL_ROOT / "legged_gym" / "scripts" / "rgb_cameras.json")
    if not hasattr(args, "nominal_rgb_camera_config"):
        args.nominal_rgb_camera_config = args.rgb_camera_config
    if not hasattr(args, "camera_mount_randomization"):
        args.camera_mount_randomization = {
            "enabled": False,
            "position_radius_m": 0.0,
            "rotation_angle_deg": 0.0,
            "episode_constant": True,
            "samples": {},
        }


def _standard_simulation_provenance(args, runner, base_seed):
    """Build reproducibility metadata without changing simulator state."""
    from gr00t.rl.data_collection.auto_open_writer import file_sha256

    repo = SCRIPT_ROOT.parent.parent

    def git(*git_args):
        try:
            return subprocess.check_output(
                ("git", *git_args), cwd=repo, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def hashed(path):
        path = Path(path)
        return {"path": str(path), "sha256": file_sha256(path)} if path.is_file() else None

    try:
        import omni.kit.app

        isaac_sim_version = str(omni.kit.app.get_app().get_app_version())
    except Exception:
        isaac_sim_version = "unknown"
    try:
        driver_version = subprocess.check_output(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver_version = "unknown"

    robot_spawn = runner.env.cfg.robot.spawn
    robot_props = robot_spawn.articulation_props
    collision_props = robot_spawn.collision_props
    rigid_props = robot_spawn.rigid_props
    sim_cfg = runner.env.cfg.sim
    physx_cfg = sim_cfg.physx
    camera_path = Path(args.rgb_camera_config)
    nominal_camera_path = Path(args.nominal_rgb_camera_config)
    door_source = repo / "gr00t" / "rl" / "isaac_utils" / "playground" / "env_rand" / "door.py"
    camera_configuration = json.loads(camera_path.read_text(encoding="utf-8"))
    return {
        "collector": "auto_open_standard_sidecar",
        "repository_commit": git("rev-parse", "HEAD"),
        "repository_dirty": bool(git("status", "--porcelain")),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "nvidia_driver_version": driver_version,
        "isaac_sim_version": isaac_sim_version,
        "scene_id": "door_open_b2z1_autoopen",
        "robot_asset": hashed(args.robot_usd_path),
        "door_asset": {
            "kind": "procedural_usd_stage",
            "generator_source": hashed(door_source),
            "episode_stage_snapshots": "stored in provenance.episodes.*.door_asset",
        },
        "checkpoint": hashed(args.ckpt_path),
        "run_metadata": hashed(args.run_metadata_path),
        "camera_config": hashed(camera_path),
        "nominal_camera_config": hashed(nominal_camera_path),
        "camera_calibration_id": file_sha256(camera_path),
        "camera_mount_randomization": args.camera_mount_randomization,
        "camera_configuration": camera_configuration,
        "camera_rotation_convention": {
            "input": "roll_pitch_yaw_radians_relative_to_parent_body",
            "composition": "q = qz(yaw) * qy(pitch) * qx(roll)",
            "runtime_quaternion_order": "wxyz",
            "dataset_pose_quaternion_order": "xyzw",
        },
        "camera_rendering": {
            "renderer": "RTX",
            "color_output": "RGB_uint8_sRGB",
            "exposure": "Isaac Sim renderer default; no per-camera override",
            "reflections": bool(args.rtx_reflections),
            "translucency": bool(args.rtx_translucency),
            "subsurface_scattering": bool(args.rtx_subsurface_scattering),
            "caustics": bool(args.rtx_caustics),
            "indirect_diffuse": bool(args.rtx_indirect_diffuse),
            "ambient_occlusion": bool(args.rtx_ambient_occlusion),
            "direct_lighting_samples_per_pixel": int(args.rtx_direct_lighting_spp),
            "reflection_samples_per_pixel": int(args.rtx_reflections_spp),
        },
        "physics_dt_s": float(runner.env.physics_dt),
        "control_dt_s": float(runner.env.dt),
        "decimation": int(runner.env.cfg.decimation),
        "solver_position_iterations": int(
            getattr(robot_props, "solver_position_iteration_count", 0)
        ),
        "solver_velocity_iterations": int(
            getattr(robot_props, "solver_velocity_iteration_count", 0)
        ),
        "ccd_enabled": bool(getattr(robot_props, "enable_ccd", False)),
        "self_collisions_enabled": bool(robot_props.enabled_self_collisions),
        "gravity_m_s2": list(sim_cfg.gravity),
        "physx_scene": {
            name: getattr(physx_cfg, name, None)
            for name in (
                "solver_type",
                "max_position_iteration_count",
                "max_velocity_iteration_count",
                "bounce_threshold_velocity",
                "friction_offset_threshold",
                "friction_correlation_distance",
                "enable_stabilization",
                "enable_enhanced_determinism",
                "gpu_max_rigid_contact_count",
                "gpu_max_rigid_patch_count",
                "gpu_found_lost_pairs_capacity",
                "gpu_found_lost_aggregate_pairs_capacity",
                "gpu_total_aggregate_pairs_capacity",
                "gpu_collision_stack_size",
                "gpu_heap_capacity",
                "gpu_temp_buffer_capacity",
            )
        },
        "robot_rigid_body": {
            name: getattr(rigid_props, name, None)
            for name in (
                "disable_gravity",
                "max_linear_velocity",
                "max_angular_velocity",
                "max_depenetration_velocity",
                "enable_gyroscopic_forces",
            )
        },
        "robot_collision": {
            "contact_offset_m": getattr(collision_props, "contact_offset", None),
            "rest_offset_m": getattr(collision_props, "rest_offset", None),
            "source": "robot_asset collider subtree",
        },
        "seed_start": int(base_seed),
        "randomization": "DoorSpawner metadata recorded per episode",
        "control_backend": "B2Z1 low-level policy plus AutoOpen FSM",
        "rendering": "RTX headless, two DATA_STANARD.md RGB sensors at 50 Hz",
    }


def _usd_json_value(value):
    """Convert authored USD values to stable JSON without changing the stage."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "path"):
        return str(value.path)
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return {
            "real": float(value.GetReal()),
            "imaginary": [float(component) for component in imaginary],
        }
    try:
        return [_usd_json_value(item) for item in value]
    except TypeError:
        return str(value)


def _door_asset_stage_snapshot(stage, door_prim):
    """Describe and hash the actual generated door subtree for one episode."""
    from pxr import Usd

    prims = []
    for prim in Usd.PrimRange(door_prim, Usd.TraverseInstanceProxies()):
        attributes = {}
        for attribute in prim.GetAttributes():
            if attribute.HasAuthoredValueOpinion():
                attributes[attribute.GetName()] = _usd_json_value(attribute.Get())
        relationships = {}
        for relationship in prim.GetRelationships():
            targets = relationship.GetTargets()
            if targets:
                relationships[relationship.GetName()] = [str(path) for path in targets]
        prims.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "attributes": attributes,
                "relationships": relationships,
                "custom_data": _usd_json_value(prim.GetCustomData()),
            }
        )
    payload = {
        "stage_prim_path": str(door_prim.GetPath()),
        "prims": prims,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


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


def _arm_drive_stiffness(env):
    """Read arm stiffness from the actuator that drives this environment."""
    stiffness = _as_np(env.robot.data.default_joint_stiffness)
    if stiffness.ndim == 2:
        stiffness = stiffness[0]
    arm_joint_ids = _as_np(env.arm_joint_ids).astype(np.int64, copy=False)
    arm_stiffness = np.asarray(stiffness[arm_joint_ids], dtype=np.float64)
    if arm_stiffness.shape != (len(env.arm_joint_names),):
        raise RuntimeError(
            "active arm actuator stiffness has an unexpected shape: "
            f"{arm_stiffness.shape}"
        )
    if not np.all(np.isfinite(arm_stiffness)) or np.any(arm_stiffness <= 0.0):
        raise RuntimeError(
            f"active arm actuator stiffness is invalid: {arm_stiffness.tolist()}"
        )
    return arm_stiffness


def _quat_yaw(quat):
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_rotate_np(quat, vector):
    """Rotate a 3-D vector by a wxyz quaternion."""
    quat = np.asarray(quat, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    w, x, y, z = quat
    vx, vy, vz = vector
    cross_x = y * vz - z * vy
    cross_y = z * vx - x * vz
    cross_z = x * vy - y * vx
    nested_x = y * cross_z - z * cross_y
    nested_y = z * cross_x - x * cross_z
    nested_z = x * cross_y - y * cross_x
    return np.array(
        (
            vx + 2.0 * (w * cross_x + nested_x),
            vy + 2.0 * (w * cross_y + nested_y),
            vz + 2.0 * (w * cross_z + nested_z),
        ),
        dtype=np.float32,
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

    def bounds(self):
        return self.points_body.min(axis=0), self.points_body.max(axis=0)


def _stage_robot_root(stage, env_id=0):
    env_prefix = f"/World/envs/env_{int(env_id)}/"
    candidates = []
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if name in {"robot", "b2z1", "b2_description"}:
            path = str(prim.GetPath())
            if path.startswith(env_prefix):
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


def _slice_control_state(state, env_id, num_envs):
    """Return a one-environment view while preserving shared name metadata."""
    sliced = {}
    for key, value in state.items():
        if (
            (torch.is_tensor(value) or isinstance(value, np.ndarray))
            and value.ndim > 0
            and value.shape[0] == num_envs
        ):
            sliced[key] = value[env_id : env_id + 1]
        else:
            sliced[key] = value
    return sliced


def _slice_all_control_states(state, num_envs):
    """Materialize row views once for command and post-step advance."""
    return [
        _slice_control_state(state, env_id, num_envs)
        for env_id in range(num_envs)
    ]


def _control_state_to_host(state, num_envs):
    """Copy vectorized controller observations to host once per simulation step."""
    hosted = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == num_envs:
            hosted[key] = value.detach().cpu().numpy()
        else:
            hosted[key] = value
    return hosted


def _update_batched_reference_buffers(env, buffers):
    buffers["get_goal_reference_origin"] = _as_np(env.get_goal_reference_origin())
    buffers["get_goal_reference_quat"] = _as_np(env.get_goal_reference_quat())
    buffers["get_goal_center_offset_local"] = _as_np(
        env.get_goal_center_offset_local()
    )


class _BatchedEnvView:
    """Expose one row of a vectorized ManipLoco environment to a scalar controller."""

    _ROW_METHODS = {
        "get_goal_reference_origin",
        "get_goal_reference_quat",
        "get_goal_center_offset_local",
    }

    def __init__(self, env, env_id, reference_buffers=None, arm_mode_buffer=None):
        self._env = env
        self._env_id = int(env_id)
        self._reference_buffers = reference_buffers
        self._arm_mode_buffer = arm_mode_buffer

    def __getattr__(self, name):
        if name == "teleop_arm_joint_pos_targets":
            return self._env.teleop_arm_joint_pos_targets[
                self._env_id : self._env_id + 1
            ]
        attribute = getattr(self._env, name)
        if name in self._ROW_METHODS:
            if self._reference_buffers is not None:
                return lambda: self._reference_buffers[name][
                    self._env_id : self._env_id + 1
                ]
            return lambda: attribute()[self._env_id : self._env_id + 1]
        return attribute

    def _set_teleop_arm_control_mode(self, mode):
        if self._arm_mode_buffer is not None:
            joint_mode = mode == "joint"
            if bool(self._arm_mode_buffer[self._env_id]) == joint_mode:
                return
            self._arm_mode_buffer[self._env_id] = joint_mode
        self._env._set_teleop_arm_control_mode(mode, [self._env_id])

    def begin_cartesian_transition(self):
        self._env.begin_cartesian_transition([self._env_id])

    def cartesian_transition_is_active(self):
        return self._env.cartesian_transition_is_active([self._env_id])


class _BatchedRunnerView:
    """Collect one controller's commands into shared vectorized teleop buffers."""

    def __init__(
        self,
        runner,
        env_id,
        state,
        reference_buffers=None,
        arm_mode_buffer=None,
        command_buffers=None,
    ):
        self._runner = runner
        self.env_id = int(env_id)
        self.env = _BatchedEnvView(
            runner.env, env_id, reference_buffers, arm_mode_buffer
        )
        self.state = state
        self.command_buffers = command_buffers

    @property
    def obs(self):
        return self._runner.obs

    @obs.setter
    def obs(self, value):
        self._runner.obs = value

    def set_state(self, state):
        self.state = state

    def get_control_state(self):
        return self.state

    def set_teleop_targets(
        self,
        base_command,
        ee_goal_cart=None,
        ee_orn_delta_rpy=None,
        gripper_target=None,
        ee_goal_orn_quat_world=None,
        ee_goal_orn_quat_world_enabled=None,
    ):
        env = self._runner.env
        row = slice(self.env_id, self.env_id + 1)
        base_command = _limit_base_commands(base_command)

        if self.command_buffers is not None:
            def _store(name, value):
                if value is not None:
                    target = self.command_buffers[name][row]
                    target[...] = _as_np(value).reshape(target.shape)

            _store("base", base_command)
            _store("ee_cart", ee_goal_cart)
            _store("ee_rpy", ee_orn_delta_rpy)
            _store("gripper", gripper_target)
            if ee_goal_orn_quat_world is not None:
                quat = _as_np(ee_goal_orn_quat_world).reshape(1, 4)
                quat /= max(float(np.linalg.norm(quat)), 1.0e-8)
                self.command_buffers["ee_quat"][row] = quat
                if ee_goal_orn_quat_world_enabled is None:
                    self.command_buffers["ee_quat_enabled"][row] = True
            if ee_goal_orn_quat_world_enabled is not None:
                self.command_buffers["ee_quat_enabled"][row] = bool(
                    np.asarray(ee_goal_orn_quat_world_enabled).reshape(-1)[0]
                )
            return

        def _copy(target, value):
            if value is not None:
                target[row] = torch.as_tensor(
                    value, device=env.device, dtype=target.dtype
                ).reshape_as(target[row])

        _copy(env.teleop_raw_commands, base_command)
        _copy(env.teleop_raw_ee_goal_cart, ee_goal_cart)
        _copy(env.teleop_raw_ee_goal_orn_delta_rpy, ee_orn_delta_rpy)
        _copy(env.gripper_pos_targets, gripper_target)
        if ee_goal_orn_quat_world is not None:
            quat = torch.as_tensor(
                ee_goal_orn_quat_world, device=env.device, dtype=torch.float32
            ).reshape(1, 4)
            quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
            env.teleop_world_ee_goal_orn_quat[row] = quat
            if ee_goal_orn_quat_world_enabled is None:
                env.teleop_world_ee_goal_orn_quat_enabled[row] = True
        if ee_goal_orn_quat_world_enabled is not None:
            env.teleop_world_ee_goal_orn_quat_enabled[row] = bool(
                torch.as_tensor(ee_goal_orn_quat_world_enabled).reshape(-1)[0]
            )
        env._teleop_inputs_dirty = True


def _make_batched_command_buffers(env):
    return {
        "base": _as_np(env.teleop_raw_commands),
        "ee_cart": _as_np(env.teleop_raw_ee_goal_cart),
        "ee_rpy": _as_np(env.teleop_raw_ee_goal_orn_delta_rpy),
        "gripper": _as_np(env.gripper_pos_targets),
        "ee_quat": _as_np(env.teleop_world_ee_goal_orn_quat),
        "ee_quat_enabled": env.teleop_world_ee_goal_orn_quat_enabled.detach().cpu().numpy().copy(),
    }


def _flush_batched_command_buffers(env, buffers):
    for target, name in (
        (env.teleop_raw_commands, "base"),
        (env.teleop_raw_ee_goal_cart, "ee_cart"),
        (env.teleop_raw_ee_goal_orn_delta_rpy, "ee_rpy"),
        (env.gripper_pos_targets, "gripper"),
        (env.teleop_world_ee_goal_orn_quat, "ee_quat"),
        (env.teleop_world_ee_goal_orn_quat_enabled, "ee_quat_enabled"),
    ):
        target.copy_(torch.as_tensor(buffers[name], device=env.device, dtype=target.dtype))
    env._teleop_inputs_dirty = True


def _batched_grasp_orientation_delta_gpu(
    command_buffers, reference_buffers, arm_induced_pitch, env_ids, device
):
    """Compute teleop grasp-orientation RPY deltas for selected envs on GPU."""
    if not env_ids:
        return np.empty((0, 3), dtype=np.float32)
    rows = np.asarray(env_ids, dtype=np.int64)
    cart = torch.as_tensor(
        command_buffers["ee_cart"][rows], device=device, dtype=torch.float32
    )
    desired = torch.as_tensor(
        command_buffers["ee_quat"][rows], device=device, dtype=torch.float32
    )
    reference = torch.as_tensor(
        reference_buffers["get_goal_reference_quat"][rows],
        device=device,
        dtype=torch.float32,
    ).clone()
    reference[:, 1:] *= -1.0
    lw, lx, ly, lz = reference.unbind(dim=-1)
    rw, rx, ry, rz = desired.unbind(dim=-1)
    local = torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )
    w, x, y, z = local.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    radius = torch.linalg.vector_norm(cart, dim=-1).clamp_min(1.0e-8)
    sphere_pitch = torch.asin((cart[:, 2] / radius).clamp(-1.0, 1.0))
    sphere_yaw = torch.atan2(cart[:, 1], cart[:, 0])
    induced_pitch = torch.as_tensor(
        np.asarray(arm_induced_pitch, dtype=np.float32)[rows],
        device=device,
        dtype=torch.float32,
    )
    two_pi = 2.0 * math.pi
    result = torch.stack(
        (
            torch.remainder(roll + math.pi / 2.0 + math.pi, two_pi) - math.pi,
            torch.remainder(pitch + sphere_pitch - induced_pitch + math.pi, two_pi)
            - math.pi,
            torch.remainder(yaw - sphere_yaw + math.pi, two_pi) - math.pi,
        ),
        dim=-1,
    )
    return result.detach().cpu().numpy()


def _base_commands_exact(
    state,
    station_xy,
    station_yaw,
    acceptance_xy=None,
    acceptance_yaw=None,
):
    """Vectorize the scalar controller's mixed-precision default base command."""
    roots = np.asarray(state["root_states"], dtype=np.float32)
    error = np.zeros((len(roots), 3), dtype=np.float32)
    error[:, :2] = np.asarray(station_xy, dtype=np.float32) - roots[:, :2]
    w, x, y, z = (roots[:, index] for index in range(3, 7))
    yaw = np.asarray(
        [
            math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            for qw, qx, qy, qz in zip(w, x, y, z, strict=True)
        ],
        dtype=np.float64,
    )
    cosine_sine = np.asarray(
        [(math.cos(value), math.sin(value)) for value in yaw],
        dtype=np.float64,
    )
    cosine, sine = cosine_sine[:, 0], cosine_sine[:, 1]
    local_x = cosine * error[:, 0] + sine * error[:, 1]
    local_y = -sine * error[:, 0] + cosine * error[:, 1]
    local_vx = cosine * roots[:, 7] + sine * roots[:, 8]
    local_vy = -sine * roots[:, 7] + cosine * roots[:, 8]
    yaw_error = np.asarray(
        [
            ((target - current + math.pi) % (2.0 * math.pi)) - math.pi
            for target, current in zip(station_yaw, yaw, strict=True)
        ],
        dtype=np.float64,
    )
    command = np.empty((len(roots), 3), dtype=np.float32)
    command[:, 0] = 1.2 * local_x - local_vx
    command[:, 1] = 1.2 * local_y - local_vy
    command[:, 2] = 1.5 * yaw_error - 0.35 * roots[:, 12]
    command = _limit_base_commands(command)
    command_distance = np.linalg.norm(error[:, :2], axis=1).astype(np.float32)
    planar_norm = np.linalg.norm(command[:, :2], axis=1).astype(np.float32)
    deadzone = (
        (command_distance > 0.08)
        & (planar_norm > 1.0e-6)
        & (planar_norm < 0.15)
    )
    command[deadzone, :2] *= (0.15 / planar_norm[deadzone])[:, None]
    yaw_deadzone = (
        (np.abs(yaw_error) > BASE_YAW_TOLERANCE_RAD)
        & (np.abs(command[:, 2]) > 1.0e-6)
        & (np.abs(command[:, 2]) < 0.2)
    )
    command[yaw_deadzone, 2] = np.copysign(0.2, command[yaw_deadzone, 2])
    acceptance_xy = station_xy if acceptance_xy is None else acceptance_xy
    acceptance_yaw = station_yaw if acceptance_yaw is None else acceptance_yaw
    acceptance_error = (
        np.asarray(acceptance_xy, dtype=np.float32) - roots[:, :2]
    )
    distance = np.linalg.norm(acceptance_error, axis=1).astype(np.float32)
    acceptance_yaw_error = np.asarray(
        [
            ((target - current + math.pi) % (2.0 * math.pi)) - math.pi
            for target, current in zip(acceptance_yaw, yaw, strict=True)
        ],
        dtype=np.float64,
    )
    return command, distance, np.abs(acceptance_yaw_error)


def _merge_tolerance_warning_stats(total, update):
    count = int(update.get("count", 0))
    if count == 0:
        return total
    total["count"] = int(total.get("count", 0)) + count
    for name in (
        "position_error_m",
        "orientation_error",
        "equilibrium_error_nm",
    ):
        low = float(update[f"{name}_min"])
        high = float(update[f"{name}_max"])
        total[f"{name}_min"] = min(total.get(f"{name}_min", math.inf), low)
        total[f"{name}_max"] = max(total.get(f"{name}_max", 0.0), high)
    return total


def _quat_rotate_points_torch(quaternion_wxyz, points):
    """Rotate one point cloud per quaternion with float32 Isaac semantics."""
    vector = quaternion_wxyz[:, None, 1:]
    scalar = quaternion_wxyz[:, None, :1]
    points = points.expand(quaternion_wxyz.shape[0], -1, -1)
    first_cross = torch.cross(vector, points, dim=-1)
    return points + 2.0 * (
        scalar * first_cross + torch.cross(vector, first_cross, dim=-1)
    )


def _make_batched_geometry_cache(controllers, device):
    first = controllers[0]
    for controller in controllers[1:]:
        if (
            controller.fixed_finger_body_idx != first.fixed_finger_body_idx
            or controller.moving_finger_body_idx != first.moving_finger_body_idx
            or controller.handle_body_idx != first.handle_body_idx
            or controller.panel_body_idx != first.panel_body_idx
            or controller.hinge_idx != first.hinge_idx
        ):
            raise RuntimeError("batched geometry requires common body indices")
    tcp_idx = next(
        index
        for index, name in enumerate(first.env.body_names)
        if name.lower() == "gripper_link"
    )

    def stacked(name, dtype=torch.float32):
        return torch.as_tensor(
            np.stack([np.asarray(getattr(controller, name)) for controller in controllers]),
            dtype=dtype,
            device=device,
        )

    geometry = first.center_geometry
    return {
        "fixed_body_idx": first.fixed_finger_body_idx,
        "moving_body_idx": first.moving_finger_body_idx,
        "handle_body_idx": first.handle_body_idx,
        "panel_body_idx": first.panel_body_idx,
        "hinge_idx": first.hinge_idx,
        "tcp_body_idx": tcp_idx,
        "fixed_points": torch.as_tensor(
            geometry["fixed_finger"].points_body, device=device
        ),
        "fixed_points_np": geometry["fixed_finger"].points_body.copy(),
        "moving_points": torch.as_tensor(
            geometry["moving_finger"].points_body, device=device
        ),
        "moving_points_np": geometry["moving_finger"].points_body.copy(),
        "tcp_points": (
            torch.as_tensor(geometry["tcp"].points_body, device=device)
            if "tcp" in geometry
            else None
        ),
        "tcp_points_np": (
            geometry["tcp"].points_body.copy() if "tcp" in geometry else None
        ),
        "wrist_points": (
            torch.as_tensor(geometry["wrist"].points_body, device=device)
            if "wrist" in geometry
            else None
        ),
        "wrist_points_np": (
            geometry["wrist"].points_body.copy() if "wrist" in geometry else None
        ),
        "handle_center_local": stacked("handle_collider_center_local"),
        "handle_axis_local": stacked("handle_axis_local"),
        "handle_axis_local_np": np.stack(
            [controller.handle_axis_local for controller in controllers]
        ).astype(np.float32),
        "door_normal": stacked("door_normal"),
        "door_normal_np": np.stack(
            [controller.door_normal for controller in controllers]
        ).astype(np.float32),
        "handle_radius": torch.as_tensor(
            [
                max((controller.handle_upper_z - controller.handle_lower_z) * 0.5, 1.0e-4)
                for controller in controllers
            ],
            dtype=torch.float32,
            device=device,
        ),
        "handle_radius_np": np.asarray(
            [
                max((controller.handle_upper_z - controller.handle_lower_z) * 0.5, 1.0e-4)
                for controller in controllers
            ],
            dtype=np.float32,
        ),
        "handle_half_length": torch.as_tensor(
            [controller.handle_half_length for controller in controllers],
            dtype=torch.float32,
            device=device,
        ),
        "handle_half_length_np": np.asarray(
            [controller.handle_half_length for controller in controllers],
            dtype=np.float32,
        ),
        "panel_half_thickness": torch.as_tensor(
            [controller.panel_half_thickness for controller in controllers],
            dtype=torch.float32,
            device=device,
        ),
        "panel_half_thickness_np": np.asarray(
            [controller.panel_half_thickness for controller in controllers],
            dtype=np.float32,
        ),
        "hinge_axis_sign": torch.as_tensor(
            [controller.hinge_axis_sign for controller in controllers],
            dtype=torch.float32,
            device=device,
        ),
        "depth_axis": torch.as_tensor(
            np.stack([controller.open_template["depth_axis"] for controller in controllers]),
            dtype=torch.float32,
            device=device,
        ),
        "depth_axis_np": np.stack(
            [controller.open_template["depth_axis"] for controller in controllers]
        ).astype(np.float32),
        "pocket_center_local": torch.as_tensor(
            np.stack(
                [controller.open_template["pocket_capture_center"] for controller in controllers]
            ),
            dtype=torch.float32,
            device=device,
        ),
        "pocket_center_local_np": np.stack(
            [
                controller.open_template["pocket_capture_center"]
                for controller in controllers
            ]
        ).astype(np.float32),
    }


def _batched_panel_finger_clearance_gpu(state, cache, env_ids):
    """Compute panel-to-finger clearance for selected rows on the GPU."""
    if not env_ids:
        return None
    device = state["robot_body_state_w"].device
    rows = torch.as_tensor(env_ids, dtype=torch.long, device=device)
    bodies = state["robot_body_state_w"][rows]
    door_bodies = state["door_body_state_w"][rows]

    def world_points(body_idx, local_points):
        pose = bodies[:, body_idx, :7]
        return (
            _quat_rotate_points_torch(pose[:, 3:7], local_points)
            + pose[:, None, :3]
        )

    fixed = world_points(cache["fixed_body_idx"], cache["fixed_points"])
    moving = world_points(cache["moving_body_idx"], cache["moving_points"])
    finger_points = torch.cat((fixed, moving), dim=1)
    panel_center = door_bodies[:, cache["panel_body_idx"], :3]
    hinge_angle = (
        state["door_joint_position"][rows, cache["hinge_idx"]]
        * cache["hinge_axis_sign"][rows]
    )
    c = torch.cos(hinge_angle)
    s = torch.sin(hinge_angle)
    door_normal = cache["door_normal"][rows]
    panel_normal = torch.stack(
        (
            c * door_normal[:, 0] - s * door_normal[:, 1],
            s * door_normal[:, 0] + c * door_normal[:, 1],
            door_normal[:, 2],
        ),
        dim=-1,
    )
    signed_distance = torch.einsum(
        "npi,ni->np", finger_points - panel_center[:, None, :], panel_normal
    )
    return signed_distance.min(dim=1).values - cache["panel_half_thickness"][rows]


def _batched_handle_centers_gpu(state, cache):
    door_bodies = state["door_body_state_w"]
    if not torch.is_tensor(door_bodies):
        door_bodies = torch.as_tensor(
            door_bodies,
            device=cache["handle_center_local"].device,
            dtype=torch.float32,
        )
    handle_state = door_bodies[:, cache["handle_body_idx"], :7]
    rotated = _quat_rotate_points_torch(
        handle_state[:, 3:7], cache["handle_center_local"][:, None, :]
    )[:, 0]
    return (handle_state[:, :3] + rotated).detach().cpu().numpy()


def _batched_geometry_metrics_gpu(state, cache, env_ids):
    """GPU shadow of DoorController._geometry_metrics for selected env rows."""
    if not env_ids:
        return None
    device = state["robot_body_state_w"].device
    rows = torch.as_tensor(env_ids, dtype=torch.long, device=device)
    bodies = state["robot_body_state_w"][rows]
    door_bodies = state["door_body_state_w"][rows]

    def select(name):
        return cache[name][rows]

    def world_points(body_idx, local_points):
        pose = bodies[:, body_idx, :7]
        return _quat_rotate_points_torch(pose[:, 3:7], local_points) + pose[:, None, :3]

    fixed = world_points(cache["fixed_body_idx"], cache["fixed_points"])
    moving = world_points(cache["moving_body_idx"], cache["moving_points"])
    separation = moving.mean(dim=1) - fixed.mean(dim=1)
    separation = separation / torch.linalg.vector_norm(
        separation, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    fixed_projection = torch.einsum("npi,ni->np", fixed, separation)
    moving_projection = torch.einsum("npi,ni->np", moving, separation)
    fixed_threshold = torch.quantile(fixed_projection, 0.85, dim=1, keepdim=True)
    moving_threshold = torch.quantile(moving_projection, 0.15, dim=1, keepdim=True)
    fixed_mask = fixed_projection >= fixed_threshold
    moving_mask = moving_projection <= moving_threshold
    fixed_center = (fixed * fixed_mask.unsqueeze(-1)).sum(dim=1) / fixed_mask.sum(
        dim=1, keepdim=True
    )
    moving_center = (moving * moving_mask.unsqueeze(-1)).sum(dim=1) / moving_mask.sum(
        dim=1, keepdim=True
    )

    handle_pose = door_bodies[:, cache["handle_body_idx"], :7]
    grasp_center = handle_pose[:, :3] + _quat_rotate_points_torch(
        handle_pose[:, 3:7], select("handle_center_local")[:, None, :]
    )[:, 0]
    fixed_is_upper = fixed_center[:, 2] >= moving_center[:, 2]
    upper_center = torch.where(fixed_is_upper[:, None], fixed_center, moving_center)
    lower_center = torch.where(fixed_is_upper[:, None], moving_center, fixed_center)
    channel_center = 0.5 * (upper_center + lower_center)
    channel_center_xy_error = torch.linalg.vector_norm(
        (channel_center - grasp_center)[:, :2], dim=-1
    )
    upper_xy_error = torch.linalg.vector_norm(
        (upper_center - grasp_center)[:, :2], dim=-1
    )
    lower_xy_error = torch.linalg.vector_norm(
        (lower_center - grasp_center)[:, :2], dim=-1
    )
    radius = select("handle_radius")
    upper_gap = upper_center[:, 2] - (grasp_center[:, 2] + radius)
    lower_gap = (grasp_center[:, 2] - radius) - lower_center[:, 2]

    ee_quaternion = state["ee_orn"][rows]
    ee_position = state["ee_pos"][rows]
    depth_axis = _quat_rotate_points_torch(
        ee_quaternion, select("depth_axis")[:, None, :]
    )[:, 0]
    jaw_points = torch.cat((fixed, moving), dim=1)
    tip_depth = torch.quantile(
        torch.einsum("npi,ni->np", jaw_points, depth_axis), 0.95, dim=1
    )
    grasp_depth = torch.abs(
        tip_depth - torch.sum(grasp_center * depth_axis, dim=-1)
    )
    pocket_center = ee_position + _quat_rotate_points_torch(
        ee_quaternion, select("pocket_center_local")[:, None, :]
    )[:, 0]
    pocket_error = pocket_center - grasp_center

    panel_center = door_bodies[:, cache["panel_body_idx"], :3]
    clearance_clouds = [fixed, moving]
    if cache["tcp_points"] is not None:
        clearance_clouds.append(world_points(cache["tcp_body_idx"], cache["tcp_points"]))
    if cache["wrist_points"] is not None:
        clearance_clouds.append(world_points(cache["fixed_body_idx"], cache["wrist_points"]))
    all_points = torch.cat(clearance_clouds, dim=1)
    panel_clearance = torch.einsum(
        "npi,ni->np", all_points - panel_center[:, None, :], select("door_normal")
    ).amin(dim=1) - select("panel_half_thickness")

    handle_axis = _quat_rotate_points_torch(
        handle_pose[:, 3:7], select("handle_axis_local")[:, None, :]
    )[:, 0]
    handle_axis = handle_axis / torch.linalg.vector_norm(
        handle_axis, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)

    def capsule_clearance(cloud):
        delta = cloud - grasp_center[:, None, :]
        axial = torch.einsum("npi,ni->np", delta, handle_axis).clamp(
            -select("handle_half_length")[:, None],
            select("handle_half_length")[:, None],
        )
        closest = grasp_center[:, None, :] + axial.unsqueeze(-1) * handle_axis[:, None, :]
        return (
            torch.linalg.vector_norm(cloud - closest, dim=-1) - radius[:, None]
        ).amin(dim=1)

    fixed_to_handle = grasp_center - fixed_center
    handle_to_moving = moving_center - grasp_center
    result = {
        "horizontal_error": torch.maximum(upper_xy_error, lower_xy_error),
        "channel_center_xy_error": channel_center_xy_error,
        "upper_xy_error": upper_xy_error,
        "lower_xy_error": lower_xy_error,
        "z_span": torch.abs(fixed_center[:, 2] - moving_center[:, 2]),
        "upper_gap": upper_gap,
        "lower_gap": lower_gap,
        "grasp_depth": grasp_depth,
        "pocket_center_error": torch.linalg.vector_norm(pocket_error, dim=-1),
        "pocket_normal_error": torch.sum(pocket_error * select("door_normal"), dim=-1),
        "pocket_axial_error": torch.sum(pocket_error * handle_axis, dim=-1),
        "pocket_vertical_error": pocket_error[:, 2],
        "channel_vertical_error": channel_center[:, 2] - grasp_center[:, 2],
        "pocket_center": pocket_center,
        "panel_clearance": panel_clearance,
        "fixed_handle_clearance": capsule_clearance(fixed),
        "moving_handle_clearance": capsule_clearance(moving),
        "fixed_center": fixed_center,
        "moving_center": moving_center,
        "channel_center": channel_center,
        "fixed_axial_offset": torch.sum(
            (fixed_center - grasp_center) * handle_axis, dim=-1
        ),
        "handle_between_jaws": (
            torch.sum(fixed_to_handle * separation, dim=-1) > radius
        )
        & (torch.sum(handle_to_moving * separation, dim=-1) > radius),
    }
    return {
        name: value.detach().cpu().numpy()
        for name, value in result.items()
    }


def _quat_rotate_points_batch_np(quaternion_wxyz, points):
    """Rotate a shared point cloud with one float32 quaternion per row."""
    quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)[None, :, :]
    vector = quaternion_wxyz[:, None, 1:]
    first_cross = np.cross(vector, points)
    return points + 2.0 * (
        quaternion_wxyz[:, None, :1] * first_cross
        + np.cross(vector, first_cross)
    )


def _batched_geometry_metrics_numpy(state, cache, env_ids, grasp_centers):
    """Vectorized NumPy mirror of the scalar jaw geometry for selected rows."""
    if not env_ids:
        return None
    rows = np.asarray(env_ids, dtype=np.int64)
    bodies = np.asarray(state["robot_body_state_w"], dtype=np.float32)[rows]
    door_bodies = np.asarray(state["door_body_state_w"], dtype=np.float32)[rows]
    ee_position = np.asarray(state["ee_pos"], dtype=np.float32)[rows]
    ee_quaternion = np.asarray(state["ee_orn"], dtype=np.float32)[rows]
    grasp_center = np.asarray(grasp_centers, dtype=np.float32)

    def world_points(body_idx, local_points):
        pose = bodies[:, body_idx, :7]
        return (
            _quat_rotate_points_batch_np(pose[:, 3:7], local_points)
            + pose[:, None, :3]
        )

    fixed = world_points(cache["fixed_body_idx"], cache["fixed_points_np"])
    moving = world_points(cache["moving_body_idx"], cache["moving_points_np"])
    selection_axis = moving.mean(axis=1) - fixed.mean(axis=1)
    separation = selection_axis / np.maximum(
        np.linalg.norm(selection_axis, axis=1, keepdims=True), 1.0e-8
    )
    fixed_projection = np.einsum("npi,ni->np", fixed, separation)
    moving_projection = np.einsum("npi,ni->np", moving, separation)
    fixed_threshold = np.quantile(fixed_projection, 0.85, axis=1, keepdims=True)
    moving_threshold = np.quantile(moving_projection, 0.15, axis=1, keepdims=True)
    fixed_mask = fixed_projection >= fixed_threshold
    moving_mask = moving_projection <= moving_threshold
    fixed_center = (fixed * fixed_mask[..., None]).sum(axis=1) / fixed_mask.sum(
        axis=1, keepdims=True
    )
    moving_center = (moving * moving_mask[..., None]).sum(axis=1) / moving_mask.sum(
        axis=1, keepdims=True
    )
    fixed_is_upper = fixed_center[:, 2] >= moving_center[:, 2]
    upper_center = np.where(fixed_is_upper[:, None], fixed_center, moving_center)
    lower_center = np.where(fixed_is_upper[:, None], moving_center, fixed_center)
    channel_center = 0.5 * (upper_center + lower_center)
    channel_center_xy_error = np.linalg.norm(
        (channel_center - grasp_center)[:, :2], axis=1
    )
    upper_xy_error = np.linalg.norm((upper_center - grasp_center)[:, :2], axis=1)
    lower_xy_error = np.linalg.norm((lower_center - grasp_center)[:, :2], axis=1)
    radius = cache["handle_radius_np"][rows]
    upper_gap = upper_center[:, 2] - (grasp_center[:, 2] + radius)
    lower_gap = (grasp_center[:, 2] - radius) - lower_center[:, 2]

    depth_vector = cache["depth_axis_np"][rows]
    q_vector = ee_quaternion[:, 1:]
    first_cross = np.cross(q_vector, depth_vector)
    depth_axis = depth_vector + 2.0 * (
        ee_quaternion[:, :1] * first_cross + np.cross(q_vector, first_cross)
    )
    jaw_points = np.concatenate((fixed, moving), axis=1)
    tip_depth = np.quantile(
        np.einsum("npi,ni->np", jaw_points, depth_axis), 0.95, axis=1
    )
    grasp_depth = np.abs(
        tip_depth - np.einsum("ni,ni->n", grasp_center, depth_axis)
    )
    pocket_local = cache["pocket_center_local_np"][rows]
    first_cross = np.cross(q_vector, pocket_local)
    pocket_center = ee_position + pocket_local + 2.0 * (
        ee_quaternion[:, :1] * first_cross + np.cross(q_vector, first_cross)
    )
    pocket_error = pocket_center - grasp_center

    panel_center = door_bodies[:, cache["panel_body_idx"], :3]
    clearance_clouds = [fixed, moving]
    if cache["tcp_points_np"] is not None:
        clearance_clouds.append(
            world_points(cache["tcp_body_idx"], cache["tcp_points_np"])
        )
    if cache["wrist_points_np"] is not None:
        clearance_clouds.append(
            world_points(cache["fixed_body_idx"], cache["wrist_points_np"])
        )
    all_points = np.concatenate(clearance_clouds, axis=1)
    door_normal = cache["door_normal_np"][rows]
    panel_clearance = np.einsum(
        "npi,ni->np", all_points - panel_center[:, None, :], door_normal
    ).min(axis=1) - cache["panel_half_thickness_np"][rows]

    handle_pose = door_bodies[:, cache["handle_body_idx"], :7]
    handle_axis_local = cache["handle_axis_local_np"][rows]
    handle_q_vector = handle_pose[:, 4:7]
    first_cross = np.cross(handle_q_vector, handle_axis_local)
    handle_axis = handle_axis_local + 2.0 * (
        handle_pose[:, 3:4] * first_cross
        + np.cross(handle_q_vector, first_cross)
    )
    handle_axis /= np.maximum(
        np.linalg.norm(handle_axis, axis=1, keepdims=True), 1.0e-8
    )
    half_length = cache["handle_half_length_np"][rows]

    def capsule_clearance(cloud):
        delta = cloud - grasp_center[:, None, :]
        axial = np.einsum("npi,ni->np", delta, handle_axis)
        axial = np.clip(axial, -half_length[:, None], half_length[:, None])
        closest = (
            grasp_center[:, None, :] + axial[..., None] * handle_axis[:, None, :]
        )
        return (
            np.linalg.norm(cloud - closest, axis=-1) - radius[:, None]
        ).min(axis=1)

    fixed_to_handle = grasp_center - fixed_center
    handle_to_moving = moving_center - grasp_center
    return {
        "horizontal_error": np.maximum(upper_xy_error, lower_xy_error),
        "channel_center_xy_error": channel_center_xy_error,
        "upper_xy_error": upper_xy_error,
        "lower_xy_error": lower_xy_error,
        "z_span": np.abs(fixed_center[:, 2] - moving_center[:, 2]),
        "upper_gap": upper_gap,
        "lower_gap": lower_gap,
        "grasp_depth": grasp_depth,
        "pocket_center_error": np.linalg.norm(pocket_error, axis=1),
        "pocket_normal_error": np.einsum("ni,ni->n", pocket_error, door_normal),
        "pocket_axial_error": np.einsum("ni,ni->n", pocket_error, handle_axis),
        "pocket_vertical_error": pocket_error[:, 2],
        "channel_vertical_error": channel_center[:, 2] - grasp_center[:, 2],
        "pocket_center": pocket_center,
        "panel_clearance": panel_clearance,
        "fixed_handle_clearance": capsule_clearance(fixed),
        "moving_handle_clearance": capsule_clearance(moving),
        "fixed_center": fixed_center,
        "moving_center": moving_center,
        "channel_center": channel_center,
        "fixed_axial_offset": np.einsum(
            "ni,ni->n", fixed_center - grasp_center, handle_axis
        ),
        "handle_between_jaws": (
            np.einsum("ni,ni->n", fixed_to_handle, separation) > radius
        )
        & (np.einsum("ni,ni->n", handle_to_moving, separation) > radius),
    }


def _numpy_geometry_requires_exact_scalar(controller, metrics):
    """Keep approximate batch values away from control-sensitive boundaries."""
    if controller.stage != Stage.CENTER_HANDLE:
        return True
    numerical_guard_m = 1.0e-4
    boundaries = (
        ("pocket_normal_error", 0.005, True),
        ("channel_vertical_error", 0.010, True),
        ("grasp_depth", 0.015, False),
        ("panel_clearance", 0.005, False),
    )
    for name, threshold, absolute_value in boundaries:
        value = float(metrics[name])
        if not math.isfinite(value):
            return True
        if absolute_value:
            value = abs(value)
        if abs(value - threshold) <= numerical_guard_m:
            return True
    axial_overlap = controller._center_handle_axial_overlap(metrics)
    if not math.isfinite(axial_overlap):
        return True
    if (
        abs(axial_overlap - CENTER_HANDLE_MIN_AXIAL_OVERLAP)
        <= numerical_guard_m
    ):
        return True
    return False


class _BatchedStabilityTracker:
    """GPU mirror of the per-controller 15-sample EE stability deque."""

    def __init__(self, num_envs, device, window_size=15):
        self.window_size = int(window_size)
        self.positions = torch.zeros(
            (num_envs, self.window_size, 3), dtype=torch.float32, device=device
        )
        self.orientations = torch.zeros(
            (num_envs, self.window_size, 4), dtype=torch.float32, device=device
        )
        self.counts = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update(self, state, controllers, env_ids):
        if not env_ids:
            return {}
        device = self.positions.device
        rows = torch.as_tensor(env_ids, dtype=torch.long, device=device)
        cleared = [
            env_id
            for env_id in env_ids
            if len(controllers[env_id].arm_stability_window) == 0
        ]
        if cleared:
            cleared_rows = torch.as_tensor(cleared, dtype=torch.long, device=device)
            self.counts[cleared_rows] = 0
        self.positions[rows] = torch.roll(self.positions[rows], -1, dims=1)
        self.orientations[rows] = torch.roll(
            self.orientations[rows], -1, dims=1
        )
        self.positions[rows, -1] = state["ee_pos"][rows]
        self.orientations[rows, -1] = state["ee_orn"][rows]
        self.counts[rows] = torch.clamp(self.counts[rows] + 1, max=self.window_size)

        positions = self.positions[rows]
        position_span = torch.linalg.vector_norm(
            positions - positions.mean(dim=1, keepdim=True), dim=-1
        ).amax(dim=1)
        orientations = self.orientations[rows]
        orientations = orientations / torch.linalg.vector_norm(
            orientations, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        current = orientations[:, -1]
        dots = torch.abs(torch.sum(orientations * current[:, None, :], dim=-1))
        orientation_span = (2.0 * torch.acos(dots.clamp(0.0, 1.0))).amax(dim=1)
        enough = self.counts[rows] >= self.window_size
        stable = enough & (position_span <= 0.001) & (orientation_span <= 0.003)
        position_span = position_span.cpu().numpy()
        orientation_span = orientation_span.cpu().numpy()
        enough = enough.cpu().numpy()
        stable = stable.cpu().numpy()
        return {
            env_id: {
                "enough": bool(enough[row]),
                "position_span": float(position_span[row]),
                "orientation_span": float(orientation_span[row]),
                "stable": bool(stable[row]),
            }
            for row, env_id in enumerate(env_ids)
        }


class DoorController:
    def __init__(
        self,
        runner: ManipLoco_Policy,
        robot_side: str = "positive_x",
        position_only_center: bool = False,
        disable_push_base_assist: bool = False,
        env_id: int = 0,
        shared_steady_model=None,
        shared_robot_geometry=None,
        target_noise_seed=0,
        target_noise_config=None,
    ):
        self.runner = runner
        self.env = runner.env
        self.robot_side = robot_side
        self.position_only_center = position_only_center
        self.disable_push_base_assist = disable_push_base_assist
        self.env_id = int(env_id)
        self.shared_robot_geometry = shared_robot_geometry
        self.target_noise = EpisodeTargetNoise(
            seed=int(target_noise_seed), config=target_noise_config
        )
        self.grasp_semantic_position_delta_world = np.zeros(
            3, dtype=np.float32
        )
        self.stage = Stage.APPROACH
        self.stage_steps = 0
        self.stage_limits = {
            Stage.APPROACH: 1000,
            Stage.PREGRASP: 500,
            Stage.OPEN_GRIPPER: 100,
            Stage.CENTER_HANDLE: 800,
            Stage.GRASP_HANDLE: 400,
            Stage.TURN_HANDLE: 900,
            Stage.RETURN_HANDLE: 150,
            Stage.OPEN_AND_CLEAR: 100,
            Stage.RETREAT_ARM: 200,
            Stage.RESTORE_GRIPPER: 300,
            Stage.ALIGN_PANEL: 400,
            Stage.CONTACT_PANEL: 400,
            Stage.PUSH_OPEN: 900,
            Stage.PUSH_ARM_RETRACT: 300,
            Stage.BASE_ADVANCE: 900,
            Stage.THROUGH: 1100,
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
        self.approach_settle_steps = 0
        self.approach_stop_latched = False
        self.pregrasp_settle_steps = 0
        self.handle_align_settle_steps = 0
        self.grasp_settle_steps = 0
        self.grasp_contact_threshold = 1.0
        self.turn_release_max_hinge_speed = math.radians(2.0)
        self.turn_release_slow_duration = 2.0
        self.turn_release_slow_steps_required = int(
            math.ceil(self.turn_release_slow_duration / float(self.env.dt))
        )
        self.turn_release_slow_steps = 0
        # The inverse steady-state model removes the systematic load error,
        # but the measured PhysX equilibrium retains about 5.6 mm for B2Z1.
        self.pregrasp_position_tolerance = 0.010
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
        self.release_start_world = None
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
        self.release_open_command_path = None
        self.release_open_orientation_path = None
        self.release_retreat_start_world = None
        self.release_retreat_target_world = None
        self.release_retreat_command_target_world = None
        self.release_retreat_command_path = None
        self.release_retreat_orientation_path = None
        self.release_retreat_path_index = 0
        self.release_restore_command_path = None
        self.release_restore_orientation_path = None
        self.push_target_hinge_angle = None
        self.panel_contact_local = None
        self.panel_center_world = None
        self.panel_push_world = None
        self.panel_bottom_z = None
        self.panel_top_z = None
        self.panel_hinge_to_free_direction = None
        self.panel_push_tangent_offset = 0.4
        self.panel_push_height_offset = 1.0
        self.panel_contact_grid_step = 0.05
        self.panel_contact_grid_radius = 0.2
        self.panel_contact_edge_margin = 0.05
        self.panel_contact_selected_offset = None
        self.panel_tcp_world = None
        self.panel_push_preload = 0.004
        self.panel_standoff_target_clearance = 0.03
        self.panel_standoff_acceptance_clearance = 0.05
        self.panel_contact_target_world = None
        self.panel_contact_desired_ee_world = None
        self.panel_contact_command_target_world = None
        self.panel_contact_inverse_target_world = None
        self.panel_contact_inverse_orientation_world = None
        self.panel_contact_inverse_command_world = None
        self.panel_contact_inverse_command_orientation_world = None
        self.prepared_panel_contact_command = None
        self.batched_panel_finger_clearance = None
        self.defer_steady_inverse = False
        self.defer_path_solve = False
        self.panel_contact_orientation = None
        self.panel_contact_settle_steps = 0
        self.panel_station_xy = None
        self.panel_station_settle_steps = 0
        self.base_hold_stage = None
        self.base_hold_target_xy = None
        self.base_hold_target_yaw = None
        self.base_hold_nominal_xy = None
        self.base_hold_nominal_yaw = None
        self.panel_align_ee_goal_local = None
        self.panel_align_ee_orientation_local = None
        self.push_start_hinge_angle = None
        self.push_last_progress_hinge = None
        self.push_no_progress_steps = 0
        self.push_pause_steps = 0
        self.body_align_target_xy = None
        self.body_align_settle_steps = 0
        self.push_contact_world = None
        self.push_normal_world = None
        self.push_orientation = None
        self.push_peak_hinge = 0.0
        self.push_coast_steps = 0
        self.push_depth = 0.0
        self.push_goal_hinge_angle = math.radians(90.0)
        self.push_target_angular_speed = math.radians(20.0)
        self.push_max_angular_speed = math.radians(30.0)
        self.push_speed_kp = 0.45
        self.push_speed_ki = 0.50
        self.push_speed_integral_limit = math.radians(8.0)
        self.push_max_angular_lead = math.radians(4.0)
        self.push_max_ee_step = 0.003
        self.push_speed_error_integral = 0.0
        self.push_command_hinge_angle = None
        self.push_max_observed_angular_speed = 0.0
        self.push_finger_clearance = float("nan")
        self.push_finger_approach_speed = float("nan")
        self.push_target_finger_clearance = float("nan")
        self.push_finger_force = 0.0
        self.push_contact_retreat_active = False
        self.push_previous_finger_clearance = None
        self.push_arm_detached = False
        self.push_detached_ee_goal_local = None
        self.push_retracted_ee_goal_local = None
        self.push_recovery_settle_steps = 0
        self.push_comfort_reach_radius = None
        self.push_reengage_reach_radius = 0.4
        self.push_reengage_release_radius = 0.45
        self.push_reengage_target_lead = 0.05
        self.push_reengage_reachable = False
        self.push_overspeed_steps = 0
        self.push_base_start_xy = None
        self.push_panel_contact_start_world = None
        self.base_advance_target_longitudinal = None
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
        self.through_station_xy = None
        self.through_station_yaw = None
        self.through_best_station_error = math.inf
        self.through_no_progress_steps = 0
        self.through_settle_steps = 0
        self.through_stop_latched = False
        self.through_stability_failure_counts = {}
        self.hinge_world = None
        self.hinge_axis_sign = None
        self.final_min_normal_offset = None
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
        self.initial_arm_joint_target = None
        self.middle_arm_joint_target = None
        self.align_panel_arm_joint_target = None
        self.base_advance_arm_joint_target = None
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
        self.fixed_inner_center_local = None
        self.fixed_surface_target_z = None
        self.grasp_orientation_quat = None
        self.pregrasp_pitch_deg = 0.0
        self.grasp_branch = None
        self.ee_orientation_command = None
        self.active_grasp_quat = None
        self.grasp_close_start = -0.30
        self.grasp_contact_close_step = 0.004
        self.q_open = None
        self.q_closed = None
        self.grasp_latched = False
        self.grasp_vertical_lift = 0.0
        self.grasp_vertical_lift_target = 0.0
        self.grasp_vertical_balance_started = False
        self.gripper_open_settle = 0
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
        arm_drive_stiffness = _arm_drive_stiffness(self.env)
        if shared_steady_model is not None:
            if not np.array_equal(
                shared_steady_model.stiffness, arm_drive_stiffness
            ):
                raise RuntimeError(
                    "shared steady EE model stiffness does not match the active "
                    f"arm actuator: model={shared_steady_model.stiffness.tolist()} "
                    f"actuator={arm_drive_stiffness.tolist()}"
                )
            self.ee_steady_model = shared_steady_model
        else:
            self.ee_steady_model = B2Z1SteadyEeForwardModel(
                steady_urdf,
                stiffness=arm_drive_stiffness,
            )
        self.arm_align_command_world = None
        self.arm_align_command_quat = None
        self.arm_align_command_path = None
        self.arm_align_orientation_path = None
        self.arm_align_path_index = 0
        self.arm_forward_target_world = None
        self.arm_forward_desired_target_world = None
        self.arm_align_desired_target_world = None
        self.arm_forward_command_world = None
        self.arm_forward_command_quat = None
        self.arm_forward_command_path = None
        self.arm_forward_orientation_path = None
        self.arm_forward_path_index = 0
        self.center_vertical_plan = None
        self.arm_stability_window = deque(maxlen=15)
        self.failure_reason = None
        self.failure_stage = None
        self.last_acceptance_diagnostics = {}
        self.last_stability_diagnostics = {}
        self.stage_exit_diagnostics = {}
        self.precomputed_steady_paths = {}
        self.gpu_geometry_uses = 0
        self.gpu_geometry_fallbacks = 0
        self.batched_stability_result = None
        self.gpu_stability_uses = 0
        self.gpu_stability_fallbacks = 0
        self.batched_handle_center = None
        self.defer_batched_orientation_delta = False
        self.batched_default_base_command = None
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

    def _door_prim_path(self):
        return _resolved_door_prim_path(self.env, self.env_id)

    def reset(self, state=None, reset_env=True, close_gripper=True):
        if reset_env:
            self.env.reset()
            self.runner.obs = self.env.get_observations()
        self.stage = Stage.APPROACH
        self.stage_steps = 0
        self.base_hold_stage = None
        self.base_hold_target_xy = None
        self.base_hold_target_yaw = None
        self.base_hold_nominal_xy = None
        self.base_hold_nominal_yaw = None
        self.approach_settle_steps = 0
        self.approach_stop_latched = False
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
        self.turn_release_slow_steps = 0
        self.turn_command_bias_world[:] = 0.0
        self.release_start_world = None
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
        self.release_open_command_path = None
        self.release_open_orientation_path = None
        self.release_retreat_start_world = None
        self.release_retreat_target_world = None
        self.release_retreat_command_target_world = None
        self.release_retreat_command_path = None
        self.release_retreat_orientation_path = None
        self.release_retreat_path_index = 0
        self.release_restore_command_path = None
        self.release_restore_orientation_path = None
        self.panel_contact_target_world = None
        self.panel_contact_desired_ee_world = None
        self.panel_contact_command_target_world = None
        self.panel_contact_inverse_target_world = None
        self.panel_contact_inverse_orientation_world = None
        self.panel_contact_inverse_command_world = None
        self.panel_contact_inverse_command_orientation_world = None
        self.prepared_panel_contact_command = None
        self.panel_contact_selected_offset = None
        self.batched_panel_finger_clearance = None
        self.defer_path_solve = False
        self.panel_contact_orientation = None
        self.panel_contact_settle_steps = 0
        self.panel_station_settle_steps = 0
        self.panel_align_ee_goal_local = None
        self.panel_align_ee_orientation_local = None
        self.align_panel_arm_joint_target = None
        self.base_advance_arm_joint_target = None
        self.push_start_hinge_angle = None
        self.push_last_progress_hinge = None
        self.push_no_progress_steps = 0
        self.push_pause_steps = 0
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
        self.push_speed_error_integral = 0.0
        self.push_finger_clearance = float("nan")
        self.push_finger_approach_speed = float("nan")
        self.push_target_finger_clearance = float("nan")
        self.push_finger_force = 0.0
        self.push_contact_retreat_active = False
        self.push_previous_finger_clearance = None
        self.push_arm_detached = False
        self.push_detached_ee_goal_local = None
        self.push_recovery_settle_steps = 0
        self.push_comfort_reach_radius = None
        self.push_reengage_reachable = False
        self.push_overspeed_steps = 0
        self.push_base_start_xy = None
        self.push_panel_contact_start_world = None
        self.base_advance_target_longitudinal = None
        self.push_shoulder_min_clearance = float("nan")
        self.push_shoulder_predicted_min_clearance = float("nan")
        self.push_base_safety_scale = 1.0
        self.through_best_station_error = math.inf
        self.through_no_progress_steps = 0
        self.through_settle_steps = 0
        self.through_stop_latched = False
        self.through_stability_failure_counts = {}
        self.push_target_hinge_angle = None
        self.failure_reason = None
        self.failure_stage = None
        self.last_acceptance_diagnostics = {}
        self.last_stability_diagnostics = {}
        self.stage_exit_diagnostics = {}
        self.precomputed_steady_paths = {}
        self.center_orientation_settle_steps = 0
        self.pregrasp_clearance_fallback = False
        self.center_clearance_fallback = False
        self.center_vertical_bias = 0.02
        self.center_depth_bias = -0.025
        self.gripper_open_settle = 0
        self.pregrasp_pitch_deg = 0.0
        self.center_tangent_feedback_world[:] = 0.0
        self.center_normal_feedback = 0.0
        self.arm_align_command_world = None
        self.arm_align_command_quat = None
        self.arm_align_command_path = None
        self.arm_align_orientation_path = None
        self.arm_align_path_index = 0
        self.arm_forward_target_world = None
        self.arm_forward_desired_target_world = None
        self.arm_align_desired_target_world = None
        self.arm_forward_command_world = None
        self.arm_forward_command_quat = None
        self.arm_forward_command_path = None
        self.arm_forward_orientation_path = None
        self.arm_forward_path_index = 0
        self.center_vertical_plan = None
        self.arm_stability_window.clear()
        self.grasp_latched = False
        self.grasp_vertical_lift = 0.0
        self.grasp_vertical_lift_target = 0.0
        self.grasp_vertical_balance_started = False
        if state is None:
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
        # Define the push point in door-local physical coordinates rather than
        # as a fraction of the randomized panel dimensions.
        self.panel_push_world = self.hinge_world.copy()
        self.panel_push_world += (
            self.panel_push_tangent_offset
            * self.panel_hinge_to_free_direction
        )
        self.panel_push_world[2] = (
            self.panel_bottom_z + self.panel_push_height_offset
        )
        panel_midplane_projection = float(
            np.dot(self.panel_center_world, self.door_normal)
        )
        self.panel_push_world += (
            panel_midplane_projection
            - float(np.dot(self.panel_push_world, self.door_normal))
        ) * self.door_normal
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
        self.station_tolerance = BASE_STATION_TOLERANCE_M
        self.station_xy = self.grasp_center_pos[:2] + self.approach_distance * normal
        self.panel_station_xy = self.station_xy.copy()
        self.panel_station_xy += float(
            np.dot(self.panel_center_world[:2] - self.panel_station_xy, door_tangent)
        ) * door_tangent
        self.door_forward = -self.door_normal
        self.through_station_yaw = math.atan2(
            float(self.door_forward[1]), float(self.door_forward[0])
        )
        c = math.cos(self.through_station_yaw)
        s = math.sin(self.through_station_yaw)
        rear_offset_world = np.array(
            [c * ROBOT_TRUNK_REAR_X, s * ROBOT_TRUNK_REAR_X],
            dtype=np.float32,
        )
        target_tail_xy = (
            door_root[:2]
            + THROUGH_TAIL_TARGET_DISTANCE * self.door_forward[:2]
        )
        self.through_station_xy = target_tail_xy - rear_offset_world
        self.initial_handle_pos = self.grasp_center_pos.copy()
        self.initial_door_root = door_root.copy()
        self.default_ee_goal_local = _as_np(state["ee_goal_cart_local"])[0]
        self.default_ee_goal_delta_rpy = _as_np(state["ee_goal_delta_rpy"])[0]
        self.push_retracted_ee_goal_local = self.default_ee_goal_local.copy()
        self.push_retracted_ee_goal_local[0] = 0.36
        middle_arm_joint_target = np.asarray(
            self.env.cfg.doorman_arm_joint_init_center, dtype=np.float32
        )
        if middle_arm_joint_target.shape != (len(self.env.arm_joint_names),):
            raise ValueError(
                "DoorMan middle arm preset must contain one value per arm joint: "
                f"expected {len(self.env.arm_joint_names)}, "
                f"got {middle_arm_joint_target.shape}"
            )
        self.middle_arm_joint_target = middle_arm_joint_target
        self.ee_command_world = _as_np(state["ee_pos"])[0].copy()
        self.initial_ee_orientation = _as_np(state["ee_orn"])[0].copy()
        dof_names = list(state["dof_names"])
        dof_position = _as_np(state["dof_pos"])[0]
        self.initial_arm_joint_target = np.asarray(
            [dof_position[dof_names.index(name)] for name in self.env.arm_joint_names],
            dtype=np.float32,
        )
        self.initial_wrist_roll = float(
            dof_position[dof_names.index("joint6")]
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
        if self.shared_robot_geometry is None:
            self._introspect_robot_colliders(state)
        else:
            self.center_geometry = self.shared_robot_geometry
            self._initialize_gripper_limits(state)
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
            # This point is the handle-center target carried by the TCP.  The
            # grasp frame maps local +X to the door-facing shaft direction,
            # so include the former 10 mm CENTER_HANDLE insertion in the
            # local geometry instead of adding a separate world-frame offset.
            pocket_capture_center[0] -= 0.010
            pocket_depth = float(np.dot(pocket_capture_center, depth_local))
            # A grasp may slide along the lever without becoming invalid.
            # Record the axial band covered by both inner jaw surfaces; the
            # CENTER_HANDLE gate later checks its physical overlap with the
            # finite handle segment instead of forcing both centers to match.
            joint_axis_local = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            fixed_axial = (
                fixed_local[fixed_inner_indices] - pocket_capture_center
            ) @ joint_axis_local
            moving_axial = (
                moving_local[moving_inner_indices] - pocket_capture_center
            ) @ joint_axis_local
            common_axial_min = max(
                float(fixed_axial.min()), float(moving_axial.min())
            )
            common_axial_max = min(
                float(fixed_axial.max()), float(moving_axial.max())
            )
            if common_axial_max <= common_axial_min:
                raise RuntimeError(
                    "fixed and moving jaw inner surfaces have no common axial band"
                )
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
                "common_axial_min": common_axial_min,
                "common_axial_max": common_axial_max,
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
            self.fixed_inner_center_local = fixed_inner_center.copy()
            self.fixed_surface_target_z = fixed_surface_target_z
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
        self._initialize_panel_push_tcp()
        # Geometry above is intentionally measured at q_open.  Once that
        # template is cached, start the actual episode with the jaw closed;
        # PREGRASP opens it only after the arm is aligned and stationary.
        gripper_joint_id = int(self.env.gripper_joint_ids[0].item())
        if close_gripper:
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
        root = _stage_robot_root(stage, self.env_id)
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
        self._initialize_gripper_limits(state)

    def _initialize_gripper_limits(self, state):
        # Infer the open endpoint from the actual reset joint and the USD
        # limits.  This avoids assuming a sign while preserving the configured
        # joint semantics.  The opposite endpoint is used for grasp closing.
        joint_id = int(self.env.gripper_joint_ids[0].item())
        q_now = float(_as_np(state["dof_pos"])[0, joint_id])
        limits = _as_np(self.env.dof_pos_limits)[joint_id]
        finite = limits[np.isfinite(limits)]
        if len(finite) < 2:
            raise RuntimeError(f"jointGripper has no finite limits: {limits.tolist()}")
        # One physical command vocabulary is shared by capability and data:
        # 0 rad is closed and -pi/3 rad is fully open.
        requested_open = -math.pi / 3
        self.q_open = float(np.clip(requested_open, finite.min(), finite.max()))
        self.q_closed = float(np.clip(0.0, finite.min(), finite.max()))
        self.grasp_latched = False
        print(
            f"[doorman][geometry] jointGripper q_now={q_now:.4f} "
            f"q_open={self.q_open:.4f} q_close_limit={self.q_closed:.4f} "
            f"q_closed={self.q_closed:.4f}",
            flush=True,
        )

    def _measure_handle_geometry(self, handle_state, grasp_state):
        """Resolve the robot-facing lever capsule, not the handle body origin."""
        try:
            import omni.usd
            from pxr import Gf, Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            collider_name = "handle_outside" if self.robot_side == "positive_x" else "handle_inside"
            path = f"{self._door_prim_path()}/door_handle/{collider_name}"
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
            panel_prim = stage.GetPrimAtPath(f"{self._door_prim_path()}/door_panel")
            panel_corners = None
            if not panel_prim or not panel_prim.IsValid():
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
            door_prim = stage.GetPrimAtPath(self._door_prim_path())
            hinge_prims = [
                prim
                for prim in Usd.PrimRange(door_prim, Usd.TraverseInstanceProxies())
                if prim.GetName() == "hinge_joint"
            ]
            if len(hinge_prims) != 1:
                raise RuntimeError(
                    f"expected one door hinge joint, found {len(hinge_prims)}"
                )
            hinge_prim = hinge_prims[0]
            root_targets = hinge_prim.GetRelationship("physics:body0").GetTargets()
            root_prim = (
                stage.GetPrimAtPath(root_targets[0]) if len(root_targets) == 1 else None
            )
            if not hinge_prim or not root_prim or not root_prim.IsValid():
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
            hinge_to_center = self.panel_center_world - self.hinge_world
            hinge_to_center[2] = 0.0
            hinge_to_center -= (
                float(np.dot(hinge_to_center, self.door_normal))
                * self.door_normal
            )
            self.panel_hinge_to_free_direction = hinge_to_center / max(
                float(np.linalg.norm(hinge_to_center)), 1.0e-8
            )
            self.panel_bottom_z = float(panel_corners[:, 2].min())
            self.panel_top_z = float(panel_corners[:, 2].max())
            self.panel_hinge_to_free_edge = float(
                np.max(
                    (panel_corners - self.hinge_world)
                    @ self.panel_hinge_to_free_direction
                )
            )
            self.turn_gap_angle = _door_gap_angle(
                self.panel_hinge_to_free_edge,
                2.0 * self.panel_half_thickness,
                self.turn_gap_clearance,
            )
            handle_joint_prim = stage.GetPrimAtPath(
                f"{self._door_prim_path()}/door_panel/handle_joint"
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
                f"panel_z=[{self.panel_bottom_z:.4f},{self.panel_top_z:.4f}] "
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
            "pocket_axial_error": float(np.dot(pocket_error, handle_axis)),
            "pocket_vertical_error": float(pocket_error[2]),
            "channel_vertical_error": float(channel_center[2] - grasp_center[2]),
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

    def _panel_finger_clearance(self, state, hinge_joint=None):
        """Return the closest finger-collider clearance from the robot-side panel face."""
        if self.batched_panel_finger_clearance is not None:
            return float(self.batched_panel_finger_clearance)
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

    def _center_handle_axial_overlap(self, metrics):
        """Length of handle covered by both inner jaw surfaces."""
        if not metrics.get("valid") or not self.open_template:
            return -math.inf
        pocket_axial_error = float(metrics.get("pocket_axial_error", math.inf))
        jaw_min = (
            pocket_axial_error
            + float(self.open_template["common_axial_min"])
        )
        jaw_max = (
            pocket_axial_error
            + float(self.open_template["common_axial_max"])
        )
        overlap_min = max(-float(self.handle_half_length), jaw_min)
        overlap_max = min(float(self.handle_half_length), jaw_max)
        return max(0.0, overlap_max - overlap_min)

    def _center_capture_ready(self, metrics):
        """Return whether the open jaws geometrically enclose the handle."""
        return bool(
            metrics.get("valid")
            and abs(metrics.get("pocket_normal_error", 999.0)) <= 0.005
            and self._center_handle_axial_overlap(metrics)
            >= CENTER_HANDLE_MIN_AXIAL_OVERLAP
            and abs(metrics.get("channel_vertical_error", 999.0)) <= 0.010
            and metrics.get("grasp_depth", 0.0) >= 0.015
            and metrics.get("panel_clearance", -1.0) >= 0.005
        )

    def _gpu_geometry_needs_scalar(self, metrics):
        """Protect every control threshold from GPU reduction ambiguity."""
        scalar_values = [
            value
            for value in metrics.values()
            if np.asarray(value).ndim == 0
            and np.asarray(value).dtype != np.bool_
        ]
        if not all(np.isfinite(float(value)) for value in scalar_values):
            return True
        if self.stage == Stage.CENTER_HANDLE:
            guarded_thresholds = (
                ("pocket_normal_error", 0.005, 0.0001),
                ("channel_vertical_error", 0.010, 0.0001),
                ("grasp_depth", 0.015, 0.0001),
                ("panel_clearance", 0.005, 0.0001),
            )
            for name, threshold, margin in guarded_thresholds:
                value = (
                    abs(float(metrics[name]))
                    if name in {
                        "pocket_normal_error",
                        "pocket_axial_error",
                        "channel_vertical_error",
                    }
                    else float(metrics[name])
                )
                if abs(value - threshold) <= margin:
                    return True
            axial_overlap = self._center_handle_axial_overlap(metrics)
            if (
                not math.isfinite(axial_overlap)
                or abs(
                    axial_overlap - CENTER_HANDLE_MIN_AXIAL_OVERLAP
                ) <= 0.0001
            ):
                return True
        elif self.stage == Stage.GRASP_HANDLE:
            guarded_thresholds = (
                ("panel_clearance", 0.005),
                ("moving_handle_clearance", 0.003),
                ("moving_handle_clearance", 0.008),
                ("fixed_handle_clearance", 0.004),
            )
            if any(
                abs(float(metrics[name]) - threshold) <= 0.0001
                for name, threshold in guarded_thresholds
            ):
                return True
            if abs(float(metrics["fixed_center"][2] - metrics["moving_center"][2])) <= 0.010:
                return True
        return False

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
        if self.batched_handle_center is not None:
            return self.batched_handle_center.copy()
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

    def _task_axes_world(self):
        normal = np.asarray(self.door_normal, dtype=np.float32)
        tangent = np.array([-normal[1], normal[0], 0.0], dtype=np.float32)
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.column_stack((normal, tangent, vertical))

    def _grasp_semantic_axes_world(self, orientation_world):
        """Return insertion, handle-axis, and jaw-closing axes for this pose."""
        depth = _quat_rotate_np(
            orientation_world, self.open_template["depth_axis"]
        )
        closing = _quat_rotate_np(
            orientation_world, self.open_template["separation_axis"]
        )
        depth /= max(float(np.linalg.norm(depth)), 1.0e-8)
        closing -= depth * float(np.dot(closing, depth))
        closing /= max(float(np.linalg.norm(closing)), 1.0e-8)
        handle_axis = np.cross(closing, depth)
        handle_axis /= max(float(np.linalg.norm(handle_axis)), 1.0e-8)
        return np.column_stack((depth, handle_axis, closing))

    def _perturb_base_target(self, label, target_xy, target_yaw):
        config = self.target_noise.config
        if label == "approach":
            bounds = (
                config.approach_normal_m,
                config.approach_tangent_m,
                config.approach_yaw_rad,
            )
        elif label == "align_panel":
            bounds = (
                config.align_panel_normal_m,
                config.align_panel_tangent_m,
                config.align_panel_yaw_rad,
            )
        elif label == "through":
            bounds = (
                config.through_forward_m,
                config.through_tangent_m,
                config.through_yaw_rad,
            )
        else:
            raise ValueError(f"unknown base target-noise semantic {label!r}")
        normal_xy = (
            self.door_forward[:2]
            if label == "through"
            else self.door_normal[:2]
        )
        return self.target_noise.perturb_base(
            label,
            target_xy,
            target_yaw,
            normal_xy,
            normal_m=bounds[0],
            tangent_m=bounds[1],
            yaw_rad=bounds[2],
        )

    def _perturb_ee_target(
        self, label, target_world, orientation_world, semantic_axes_world=None
    ):
        config = self.target_noise.config
        sample_label = label
        if label in ("pregrasp", "center_handle"):
            # Both stages describe one reach-and-insert action. Reusing this
            # sample keeps the CENTER_HANDLE path collinear with PREGRASP.
            sample_label = "grasp_chain"
            position_bounds = (
                config.grasp_depth_m,
                config.grasp_handle_axis_m,
                config.grasp_closing_m,
            )
            rotation_bounds = (config.grasp_roll_rad, 0.0, 0.0)
            position_names = ("depth_m", "handle_axis_m", "closing_m")
            rotation_names = ("roll_rad", "forbidden_tilt_h_rad", "forbidden_tilt_z_rad")
            semantic_axes_world = self._grasp_semantic_axes_world(
                orientation_world
            )
            rotation_semantic_axes_world = np.column_stack(
                (
                    semantic_axes_world[:, 1],
                    semantic_axes_world[:, 0],
                    semantic_axes_world[:, 2],
                )
            )
        elif label == "contact_panel":
            sample_label = "panel_push_chain"
            position_bounds = (
                config.panel_depth_m,
                config.panel_tangent_m,
                config.panel_vertical_m,
            )
            rotation_bounds = (
                config.panel_roll_rad,
                config.panel_tilt_horizontal_rad,
                config.panel_tilt_vertical_rad,
            )
            position_names = ("depth_m", "tangent_m", "vertical_m")
            rotation_names = (
                "roll_rad",
                "tilt_horizontal_rad",
                "tilt_vertical_rad",
            )
        elif label == "open_and_clear":
            position_bounds = (0.0, 0.0, config.release_lift_m)
            rotation_bounds = (0.0, 0.0, 0.0)
            position_names = ("normal_m", "tangent_m", "lift_m")
            rotation_names = ("normal_rad", "tangent_rad", "vertical_rad")
        elif label == "retreat_arm":
            position_bounds = (
                config.release_retreat_normal_m,
                0.0,
                config.release_retreat_vertical_m,
            )
            rotation_bounds = (0.0, 0.0, 0.0)
            position_names = ("normal_m", "tangent_m", "vertical_m")
            rotation_names = ("normal_rad", "tangent_rad", "vertical_rad")
        elif label in ("return_handle", "restore_gripper", "turn_handle"):
            # These poses are constrained by a physical pivot or by a
            # previously accepted endpoint. Independent Cartesian noise has
            # no task-level meaning and would create a discontinuity.
            return (
                np.asarray(target_world).reshape(3).copy(),
                np.asarray(orientation_world).reshape(4).copy(),
            )
        else:
            raise ValueError(f"unknown EE target-noise semantic {label!r}")
        if label not in ("pregrasp", "center_handle"):
            rotation_semantic_axes_world = None
        perturbed = self.target_noise.perturb_pose(
            sample_label,
            target_world,
            orientation_world,
            self._task_axes_world()
            if semantic_axes_world is None
            else semantic_axes_world,
            position_bounds=position_bounds,
            rotation_bounds=rotation_bounds,
            position_names=position_names,
            rotation_names=rotation_names,
            rotation_semantic_axes_world=rotation_semantic_axes_world,
        )
        if sample_label == "grasp_chain":
            self.grasp_semantic_position_delta_world = (
                np.asarray(perturbed[0], dtype=np.float32)
                - np.asarray(target_world, dtype=np.float32)
            )
        return perturbed

    def _base_command(
        self,
        state,
        target_xy=None,
        target_yaw=None,
        acceptance_xy=None,
        acceptance_yaw=None,
    ):
        default_target = target_xy is None and target_yaw is None
        if default_target and self.batched_default_base_command is not None:
            command, distance, yaw_error = self.batched_default_base_command
            return command.copy(), float(distance), float(yaw_error)
        if default_target:
            nominal_xy, nominal_yaw = self._approach_base_target()
            target_xy, target_yaw = self._perturb_base_target(
                "approach", nominal_xy, nominal_yaw
            )
            acceptance_xy = nominal_xy
            acceptance_yaw = nominal_yaw
        elif target_xy is None:
            target_xy = self.station_xy
        if target_yaw is None:
            target_yaw = math.atan2(-self.door_normal[1], -self.door_normal[0])
        acceptance_xy = target_xy if acceptance_xy is None else acceptance_xy
        acceptance_yaw = target_yaw if acceptance_yaw is None else acceptance_yaw
        command, distance, yaw_error = _base_commands_exact(
            state,
            np.asarray(target_xy, dtype=np.float32).reshape(1, 2),
            np.asarray((target_yaw,), dtype=np.float64),
            np.asarray(acceptance_xy, dtype=np.float32).reshape(1, 2),
            np.asarray((acceptance_yaw,), dtype=np.float64),
        )
        return command[0], float(distance[0]), float(yaw_error[0])

    def _set_base_tracking_target(
        self, target_xy, target_yaw, nominal_xy=None, nominal_yaw=None
    ):
        """Record the target currently owned by a base movement stage."""
        self.base_hold_target_xy = np.asarray(target_xy, dtype=np.float32).copy()
        self.base_hold_target_yaw = float(target_yaw)
        self.base_hold_nominal_xy = np.asarray(
            target_xy if nominal_xy is None else nominal_xy,
            dtype=np.float32,
        ).copy()
        self.base_hold_nominal_yaw = float(
            target_yaw if nominal_yaw is None else nominal_yaw
        )

    def _approach_base_target(self):
        """Return the base target shared by scalar and batched APPROACH."""
        target_yaw = math.atan2(
            float(-self.door_normal[1]), float(-self.door_normal[0])
        )
        return self.station_xy, target_yaw

    def _base_hold_target(self):
        """Return the preceding movement target unchanged for a hold stage."""
        if self.stage not in BASE_HOLD_STAGES:
            self.base_hold_stage = None
            return None
        if (
            self.base_hold_target_xy is None
            or self.base_hold_target_yaw is None
            or self.base_hold_nominal_xy is None
            or self.base_hold_nominal_yaw is None
        ):
            raise RuntimeError(
                f"{self.stage.name} entered without a base target to hold"
            )
        if self.base_hold_stage != self.stage:
            self.base_hold_stage = self.stage
            print(
                f"[doorman][base-hold] stage={self.stage.name} "
                f"inherited_xy={self.base_hold_target_xy.round(4).tolist()} "
                f"inherited_yaw={self.base_hold_target_yaw:.4f}",
                flush=True,
            )
        return (
            self.base_hold_target_xy,
            self.base_hold_target_yaw,
            self.base_hold_nominal_xy,
            self.base_hold_nominal_yaw,
        )

    def _ee_local_target(self, world_target):
        origin = self.env.get_goal_reference_origin()
        quat = self.env.get_goal_reference_quat()
        if isinstance(origin, np.ndarray):
            target = np.asarray(world_target, dtype=np.float32).reshape(1, 3)
            local = _quat_rotate_inverse_np(quat[0], target[0] - origin[0]).reshape(1, 3)
            return local - self.env.get_goal_center_offset_local()
        target = torch.as_tensor(world_target, device=self.env.device, dtype=torch.float32).reshape(1, 3)
        local = quat_rotate_inverse(quat, target - origin)
        offset = self.env.get_goal_center_offset_local()
        return local - offset

    def _ee_world_target(self, local_target):
        """Map one goal-frame EE target into world coordinates."""
        origin = _as_np(self.env.get_goal_reference_origin())[0]
        quat = _as_np(self.env.get_goal_reference_quat())[0]
        offset = _as_np(self.env.get_goal_center_offset_local())[0]
        local_with_center = (
            np.asarray(local_target, dtype=np.float32).reshape(3) + offset
        )
        return origin + _quat_rotate_np(quat, local_with_center)

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

    def _initialize_panel_push_tcp(self):
        """Build geometry shared by feedback and open-loop panel pushing."""
        if (
            self.panel_push_world is None
            or self.panel_center_world is None
            or self.final_min_normal_offset is None
        ):
            raise RuntimeError("panel push geometry is incomplete")
        panel_surface = self.panel_push_world.copy()
        surface_projection = (
            float(np.dot(self.panel_center_world, self.door_normal))
            + self.panel_half_thickness
        )
        panel_surface += (
            surface_projection
            - float(np.dot(panel_surface, self.door_normal))
        ) * self.door_normal
        panel_tcp = panel_surface.copy()
        tcp_projection = (
            surface_projection
            - self.final_min_normal_offset
            - self.panel_push_preload
        )
        panel_tcp += (
            tcp_projection - float(np.dot(panel_tcp, self.door_normal))
        ) * self.door_normal
        self.panel_tcp_world = panel_tcp.astype(np.float32)

    def _panel_tcp_for_orientation(self, state, hinge_world, orientation_world):
        """Place the EE from the current gripper geometry at the panel face."""
        if self.panel_push_world is None or self.center_geometry is None:
            raise RuntimeError("panel or gripper collision geometry is incomplete")
        bodies = _as_np(state.get("robot_body_state_w"))
        ee_position = _as_np(state.get("ee_pos"))
        ee_orientation = _as_np(state.get("ee_orn"))
        if bodies is None or ee_position is None or ee_orientation is None:
            raise RuntimeError("current gripper pose is unavailable")
        fixed_points_world = self.center_geometry["fixed_finger"].world_points(
            bodies[0, self.fixed_finger_body_idx, :7]
        )
        moving_points_world = self.center_geometry["moving_finger"].world_points(
            bodies[0, self.moving_finger_body_idx, :7]
        )
        gripper_points_local = _quat_rotate_points_np(
            _quat_conjugate_np(ee_orientation[0]),
            np.concatenate((fixed_points_world, moving_points_world), axis=0)
            - ee_position[0],
        )
        hinge_world = float(hinge_world)
        current_normal = _quat_rotate_np(
            self._yaw_quat(hinge_world), self.door_normal
        )
        panel_surface = self._rotate_about_z(
            self.panel_push_world + self.door_normal * self.panel_half_thickness,
            self.hinge_world,
            hinge_world,
        )
        offsets_world = _quat_rotate_points_np(
            orientation_world, gripper_points_local
        )
        min_normal_offset = float((offsets_world @ current_normal).min())
        panel_tcp = panel_surface.copy()
        target_projection = (
            float(np.dot(panel_surface, current_normal))
            - min_normal_offset
            - self.panel_push_preload
        )
        panel_tcp += (
            target_projection - float(np.dot(panel_tcp, current_normal))
        ) * current_normal
        return panel_tcp.astype(np.float32)

    def _command_approach(self, state):
        """Publish the sole APPROACH command for scalar and batched runners."""
        base_cmd, station_dist, yaw_error = self._base_command(state)
        root_state = _as_np(state["root_states"])[0]
        self.approach_stop_latched = _approach_stop_latch(
            self.approach_stop_latched,
            station_dist,
            yaw_error,
            np.linalg.norm(root_state[7:9]),
            np.linalg.norm(root_state[10:13]),
        )
        if self.approach_stop_latched:
            base_cmd[:] = 0.0
        target_xy, target_yaw = self._perturb_base_target(
            "approach", *self._approach_base_target()
        )
        nominal_xy, nominal_yaw = self._approach_base_target()
        self._set_base_tracking_target(
            target_xy, target_yaw, nominal_xy, nominal_yaw
        )
        self.base_hold_stage = None
        self.active_grasp_quat = self._slew_orientation(
            self.initial_ee_orientation
        )
        self.ee_command_world = _as_np(state["ee_pos"])[0].copy()
        self.ee_orientation_error = self._quat_distance(
            _as_np(state["ee_orn"])[0], self.active_grasp_quat
        )
        self.env._set_teleop_arm_control_mode("joint")
        hold_target = torch.as_tensor(
            self.initial_arm_joint_target,
            device=self.env.device,
            dtype=torch.float32,
        ).reshape(1, -1)
        self.env.teleop_arm_joint_pos_targets.copy_(hold_target)
        ee_local = self.default_ee_goal_local.reshape(1, 3)
        orientation_delta = np.zeros((1, 3), dtype=np.float32)
        if not self.defer_batched_orientation_delta:
            orientation_delta[0] = self._grasp_orientation_delta(
                ee_local[0], self.active_grasp_quat
            )
        gripper = float(self.q_closed if self.q_closed is not None else 0.0)
        self.runner.set_teleop_targets(
            base_cmd.reshape(1, 3),
            gripper_target=np.asarray([[gripper]], dtype=np.float32),
        )
        high = np.concatenate(
            (base_cmd, ee_local[0], orientation_delta[0], [gripper])
        ).astype(np.float32)
        return high, station_dist, yaw_error

    def command(self, state):
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
        if self.stage == Stage.APPROACH:
            return self._command_approach(state)
        base_cmd, station_dist, yaw_error = self._base_command(state)
        # Keep one fixed target derived from the reset USD collider template.
        # A target recomputed from the moving jaws makes the reference chase
        # the IK error and can drive the arm away from the handle.
        desired_grasp_quat = self._desired_grasp_quat(grasp_quat)
        if self.stage == Stage.CENTER_HANDLE and self.position_only_center:
            desired_grasp_quat = _as_np(state["ee_orn"])[0].copy()
        rotate_for_grasp = self.stage in (
            Stage.PREGRASP,
            Stage.OPEN_GRIPPER,
            Stage.CENTER_HANDLE,
            Stage.GRASP_HANDLE,
            Stage.TURN_HANDLE,
            Stage.RETURN_HANDLE,
            Stage.OPEN_AND_CLEAR,
            Stage.RETREAT_ARM,
            Stage.RESTORE_GRIPPER,
        )
        desired_orientation = desired_grasp_quat if rotate_for_grasp else self.initial_ee_orientation
        if (
            self.stage
            in (
                Stage.ALIGN_PANEL,
                Stage.CONTACT_PANEL,
                Stage.PUSH_OPEN,
                Stage.PUSH_ARM_RETRACT,
                Stage.BASE_ADVANCE,
            )
            and self.panel_align_ee_orientation_local is not None
        ):
            goal_ref_quat = _as_np(self.env.get_goal_reference_quat())[0]
            desired_orientation = _quat_mul_np(
                goal_ref_quat, self.panel_align_ee_orientation_local
            )
        if self.stage in (Stage.THROUGH, Stage.SUCCESS):
            desired_orientation = self.initial_ee_orientation
        command_orientation = desired_orientation
        if (
            self.stage in (Stage.PREGRASP, Stage.OPEN_GRIPPER)
        ):
            if (
                self.arm_align_command_path is not None
                and self.arm_align_orientation_path is not None
                and self.arm_align_path_index
                < len(self.arm_align_command_path) - 1
            ):
                command_orientation = self.arm_align_orientation_path[
                    min(
                        self.arm_align_path_index,
                        len(self.arm_align_orientation_path) - 1,
                    )
                ]
            elif self.arm_align_command_quat is not None:
                command_orientation = self.arm_align_command_quat
        elif self.stage in (Stage.CENTER_HANDLE, Stage.GRASP_HANDLE):
            if (
                self.stage == Stage.CENTER_HANDLE
                and self.arm_forward_command_path is not None
                and self.arm_forward_orientation_path is not None
            ):
                command_orientation = self.arm_forward_orientation_path[
                    min(
                        self.arm_forward_path_index,
                        len(self.arm_forward_orientation_path) - 1,
                    )
                ]
            elif self.arm_forward_command_quat is not None:
                command_orientation = self.arm_forward_command_quat
        if self.stage == Stage.CONTACT_PANEL:
            # CONTACT_PANEL's final orientation is produced by the one-shot
            # steady inverse below.  Preserve the previous command here so the
            # prepared target is slewed exactly once after it is available.
            self.active_grasp_quat = self.ee_orientation_command.copy()
        else:
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
        ee_target_world = grasp_center.copy()
        gripper = float(self.q_open if self.q_open is not None else -0.785)
        if self.stage == Stage.PREGRASP:
            gripper = float(self.q_closed if self.q_closed is not None else 0.0)
        if self.stage in (Stage.PREGRASP, Stage.OPEN_GRIPPER):
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
                self.arm_forward_path_index = min(
                    path_index + 1, len(self.arm_forward_command_path) - 1
                )
            else:
                ee_target_world = self.arm_forward_command_world.copy()
            self.handle_center_target_world = self.arm_forward_target_world.copy()
        elif self.stage == Stage.GRASP_HANDLE:
            # High-level semantics are simply "grasp": keep commanding the
            # closed position. Contact-aware force limiting belongs to the
            # low-level gripper controller; measured opening is feedback only.
            ee_target_world = self.arm_forward_command_world.copy()
            moving_clearance = float(
                self.center_metrics.get("moving_handle_clearance", 999.0)
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
                and (moving_clearance <= 0.003 or moving_contact)
            ):
                fixed_gap = max(
                    float(
                        self.center_metrics.get(
                            "upper_gap" if fixed_is_upper else "lower_gap", 0.0
                        )
                    ),
                    0.0,
                )
                self.grasp_vertical_lift_target = float(
                    balance_direction
                    * np.clip(
                        fixed_gap, 0.0, GRASP_FIXED_JAW_BALANCE_LIMIT
                    )
                )
                self.grasp_vertical_balance_started = True
                print(
                    f"[doorman][grasp-balance] fixed_is_upper={fixed_is_upper} "
                    f"fixed_gap={fixed_gap:.4f} "
                    f"vertical_target={self.grasp_vertical_lift_target:.4f}",
                    flush=True,
                )
            if moving_contact != fixed_contact:
                self.grasp_vertical_lift_target = float(
                    np.clip(
                        self.grasp_vertical_lift_target
                        + balance_direction * 0.00005,
                        -GRASP_FIXED_JAW_BALANCE_LIMIT,
                        GRASP_FIXED_JAW_BALANCE_LIMIT,
                    )
                )
            lift_delta = np.clip(
                self.grasp_vertical_lift_target - self.grasp_vertical_lift,
                -0.0002,
                0.0002,
            )
            self.grasp_vertical_lift += float(lift_delta)
            gripper = 0.0
            ee_target_world[2] += self.grasp_vertical_lift
        elif self.stage == Stage.TURN_HANDLE:
            # Follow the fixed stage-entry handle-rotation trajectory.
            ee_target_world, self.active_grasp_quat = self._turn_path_target(state)
            gripper = 0.0
        elif self.stage == Stage.RETURN_HANDLE:
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
            ee_target_world = self.release_return_axle_world + _quat_rotate_np(
                return_rotation, self.release_return_offset_world
            )
            self.active_grasp_quat = _quat_mul_np(
                return_rotation, self.release_return_start_quat
            )
            ee_target_world, self.active_grasp_quat = self._perturb_ee_target(
                "return_handle", ee_target_world, self.active_grasp_quat
            )
            gripper = 0.0
        elif self.stage == Stage.OPEN_AND_CLEAR:
            open_steps = self.stage_steps
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
            gripper = (
                (1.0 - gripper_fraction) * float(self.q_closed)
                + gripper_fraction * float(self.q_open)
            )
        elif self.stage == Stage.RETREAT_ARM:
            if (
                self.release_retreat_command_path is None
                or self.release_retreat_orientation_path is None
            ):
                raise RuntimeError("release retreat inverse path is unavailable")
            path_index = min(
                self.release_retreat_path_index,
                len(self.release_retreat_command_path) - 1,
            )
            ee_target_world = self.release_retreat_command_path[path_index].copy()
            self.active_grasp_quat = self.release_retreat_orientation_path[
                path_index
            ].copy()
            self.release_retreat_path_index = min(
                path_index + 1,
                len(self.release_retreat_command_path) - 1,
            )
            gripper = float(self.q_open)
        elif self.stage == Stage.RESTORE_GRIPPER:
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
            # Hold the measured entry joint pose while the base moves.
            gripper = 0.0
        elif self.stage in (
            Stage.CONTACT_PANEL,
            Stage.PUSH_OPEN,
            Stage.PUSH_ARM_RETRACT,
            Stage.BASE_ADVANCE,
        ):
            if (
                self.stage
                in (Stage.PUSH_OPEN, Stage.PUSH_ARM_RETRACT, Stage.BASE_ADVANCE)
                and self.panel_tcp_world is not None
            ):
                current_hinge_joint = float(
                    _as_np(state["door_joint_position"])[0, self.hinge_idx]
                )
                hinge_velocity = float(
                    _as_np(state["door_joint_velocity"])[0, self.hinge_idx]
                )
                current_hinge_world = self.hinge_axis_sign * current_hinge_joint
                if self.push_orientation is None:
                    self.push_orientation = self.active_grasp_quat.copy()
                if self.stage in (
                    Stage.PUSH_OPEN,
                    Stage.PUSH_ARM_RETRACT,
                    Stage.BASE_ADVANCE,
                ):
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
                    current_goal_local = _as_np(state["ee_goal_cart_local"])[0]
                    comfort_reach_radius = (
                        self.push_comfort_reach_radius
                        if self.push_comfort_reach_radius is not None
                        else float(np.linalg.norm(current_goal_local[:2]))
                    )
                    arm_overextended = bool(
                        self.stage == Stage.PUSH_OPEN
                        and float(np.linalg.norm(current_goal_local[:2]))
                        > comfort_reach_radius
                    )
                    tracked_contact_delta = None
                    tracked_contact_world = None
                    tracked_contact_orientation = None
                    if self.stage in (Stage.PUSH_ARM_RETRACT, Stage.BASE_ADVANCE):
                        tracked_contact_delta = self.hinge_axis_sign * (
                            current_hinge_joint - self.push_start_hinge_angle
                        )
                        tracked_contact_world = self._rotate_about_z(
                            self.push_contact_world,
                            self.hinge_world,
                            tracked_contact_delta,
                        )
                        tracked_contact_orientation = _quat_mul_np(
                            self._yaw_quat(tracked_contact_delta),
                            self.push_orientation,
                        )
                    if self.stage == Stage.PUSH_OPEN and arm_overextended:
                        self.push_arm_detached = True
                        self.push_recovery_settle_steps = 0
                        self.push_detached_ee_goal_local = current_goal_local.copy()
                        self.stage = Stage.PUSH_ARM_RETRACT
                        self.stage_steps = 0
                        print(
                            f"[doorman][push] begin arm/base reposition "
                            f"reason=reach "
                            f"reach={np.linalg.norm(current_goal_local[:2]):.4f} "
                            f"limit={comfort_reach_radius:.4f} "
                            f"clearance={finger_clearance:.4f}",
                            flush=True,
                        )

                    if self.stage in (Stage.PUSH_ARM_RETRACT, Stage.BASE_ADVANCE):
                        if tracked_contact_world is None:
                            tracked_contact_delta = self.hinge_axis_sign * (
                                current_hinge_joint - self.push_start_hinge_angle
                            )
                            tracked_contact_world = self._rotate_about_z(
                                self.push_contact_world,
                                self.hinge_world,
                                tracked_contact_delta,
                            )
                            tracked_contact_orientation = _quat_mul_np(
                                self._yaw_quat(tracked_contact_delta),
                                self.push_orientation,
                            )
                        tracked_contact_local = _as_np(
                            self._ee_local_target(tracked_contact_world)
                        )[0]
                        if self.stage == Stage.PUSH_ARM_RETRACT:
                            retract_delta = (
                                self.push_retracted_ee_goal_local
                                - self.push_detached_ee_goal_local
                            )
                            retract_distance = float(np.linalg.norm(retract_delta))
                            if retract_distance > 0.004:
                                self.push_detached_ee_goal_local += (
                                    0.004 / retract_distance
                                ) * retract_delta
                            else:
                                self.push_detached_ee_goal_local = (
                                    self.push_retracted_ee_goal_local.copy()
                                )
                            ee_local_override = self.push_detached_ee_goal_local.copy()
                            arm_speed = float(
                                np.linalg.norm(_as_np(state["ee_vel"])[0, :3])
                            )
                            arm_retracted = bool(
                                retract_distance <= 0.006 and arm_speed < 0.04
                            )
                            self.push_recovery_settle_steps = (
                                self.push_recovery_settle_steps + 1
                                if arm_retracted
                                else 0
                            )
                            if self.push_recovery_settle_steps >= 10:
                                arm_joint_ids = _as_np(
                                    self.env.arm_joint_ids
                                ).astype(int)
                                self.base_advance_arm_joint_target = _as_np(
                                    state["dof_pos"]
                                )[0, arm_joint_ids].copy()
                                self.stage = Stage.BASE_ADVANCE
                                self.stage_steps = 0
                                self.push_recovery_settle_steps = 0
                                self.push_reengage_reachable = False
                                self.base_advance_target_longitudinal = None
                                print(
                                    "[doorman][push] arm comfortable; "
                                    "begin base-only advance",
                                    flush=True,
                                )
                        elif self.stage == Stage.BASE_ADVANCE:
                            ee_local_override = self.push_retracted_ee_goal_local.copy()
                            target_distance = float(
                                np.linalg.norm(tracked_contact_local[:2])
                            )
                            if self.push_reengage_reachable:
                                if target_distance > self.push_reengage_release_radius:
                                    self.push_reengage_reachable = False
                            elif target_distance <= self.push_reengage_reach_radius:
                                self.push_reengage_reachable = True
                            target_reachable = self.push_reengage_reachable
                            if target_reachable:
                                base_cmd[:] = 0.0
                            hinge_speed = abs(
                                float(
                                    _as_np(state["door_joint_velocity"])[
                                        0, self.hinge_idx
                                    ]
                                )
                            )
                            base_speed = float(
                                np.linalg.norm(_as_np(state["root_states"])[0, 7:9])
                            )
                            ready_for_contact = bool(
                                target_reachable
                                and hinge_speed < self.turn_release_max_hinge_speed
                                and base_speed < 0.04
                            )
                            self.push_recovery_settle_steps = (
                                self.push_recovery_settle_steps + 1
                                if ready_for_contact
                                else 0
                            )
                            if self.push_recovery_settle_steps >= 10:
                                self.ee_command_world = _as_np(state["ee_pos"])[0].copy()
                                print(
                                    f"[doorman][push] panel reachable and settled; "
                                    f"reach={target_distance:.4f} "
                                    f"hinge_speed={math.degrees(hinge_speed):.3f}deg/s "
                                    f"base_speed={base_speed:.4f}m/s; "
                                    "restart panel standoff positioning",
                                    flush=True,
                                )
                                self._begin_panel_contact(
                                    tracked_contact_orientation
                                )
                        self.push_command_hinge_angle = current_hinge_joint
                        self.push_speed_error_integral = 0.0
                        self.push_finger_clearance = finger_clearance
                        self.push_finger_approach_speed = approach_speed
                        self.push_target_finger_clearance = float("nan")
                        self.push_finger_force = finger_force
                        self.push_contact_retreat_active = False
                        self.active_grasp_quat = self.push_orientation.copy()
                    else:
                        desired_speed = self.push_target_angular_speed
                        speed_error = desired_speed - hinge_velocity
                        self.push_speed_error_integral = float(
                            np.clip(
                                self.push_speed_error_integral
                                + speed_error * float(self.runner.env.dt),
                                -self.push_speed_integral_limit,
                                self.push_speed_integral_limit,
                            )
                        )
                        # Convert velocity error into a bounded angular lead/lag
                        # for the Cartesian impedance target.  The integral term
                        # removes the steady-state speed deficit under panel load.
                        angular_offset = np.clip(
                            self.push_speed_kp * speed_error
                            + self.push_speed_ki * self.push_speed_error_integral,
                            -self.push_max_angular_lead,
                            self.push_max_angular_lead,
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
                        self.push_finger_clearance = finger_clearance
                        self.push_finger_approach_speed = approach_speed
                        self.push_target_finger_clearance = float("nan")
                        self.push_finger_force = finger_force
                        self.push_contact_retreat_active = False
                        current_hinge_delta_world = self.hinge_axis_sign * (
                            current_hinge_joint - self.push_start_hinge_angle
                        )
                        self.active_grasp_quat = _quat_mul_np(
                            self._yaw_quat(current_hinge_delta_world),
                            self.push_orientation,
                        )
                    self.active_grasp_quat = self.push_orientation.copy()
            else:
                if self.prepared_panel_contact_command is None:
                    self._prepare_panel_contact_command(state)
                (
                    ee_target_world,
                    prepared_orientation,
                ) = self.prepared_panel_contact_command
                self.active_grasp_quat = self._slew_orientation(
                    prepared_orientation
                )
            gripper = 0.0
        elif self.stage == Stage.THROUGH:
            # Traverse with the arm held at the same middle joint preset used
            # for the final state; do not introduce a separate stow pose.
            gripper = 0.0
        elif self.stage == Stage.SUCCESS:
            gripper = 0.0

        # Keep the reset-pose override exact, but slew every Cartesian target
        # used for physical interaction so the arm approaches the handle and
        # panel continuously instead of teleporting its IK goal.
        follows_precomputed_inverse_path = bool(
            self.stage == Stage.CENTER_HANDLE
            and self.arm_forward_command_path is not None
        )
        if self.stage in (Stage.ALIGN_PANEL, Stage.BASE_ADVANCE):
            # These stages hold measured joint angles while the base moves.
            # Keep the Cartesian slew origin at the actual EE pose so the
            # next EE-controlled stage starts continuously from that pose.
            self.ee_command_world = _as_np(state["ee_pos"])[0].copy()
        elif follows_precomputed_inverse_path:
            self.ee_command_world = ee_target_world.copy()
        elif ee_local_override is None and self.stage != Stage.TURN_HANDLE:
            if self.stage == Stage.CONTACT_PANEL:
                max_step = 0.003
            elif self.stage in (
                Stage.PUSH_OPEN,
                Stage.PUSH_ARM_RETRACT,
                Stage.BASE_ADVANCE,
            ):
                max_step = self.push_max_ee_step
            else:
                max_step = 0.002 if self.stage == Stage.CENTER_HANDLE else 0.012
            ee_target_world = self._slew_ee_target(ee_target_world, max_step=max_step)
        else:
            self.ee_command_world = _as_np(state["ee_pos"])[0].copy()

        if self.stage == Stage.BASE_ADVANCE:
            if self.disable_push_base_assist:
                base_advance_target = _as_np(state["root_states"])[0, :2].copy()
                base_advance_yaw = _quat_yaw(_as_np(state["root_states"])[0, 3:7])
            else:
                base_advance_target, base_advance_yaw = (
                    self._base_advance_target(state)
                )
            base_cmd, station_dist, yaw_error = self._base_command(
                state,
                base_advance_target,
                base_advance_yaw,
            )
            self._set_base_tracking_target(
                base_advance_target, base_advance_yaw
            )
            self.base_hold_stage = None
        elif self.stage == Stage.ALIGN_PANEL:
            panel_nominal_xy, panel_nominal_yaw = self._current_panel_station(state)
            panel_station_xy, panel_station_yaw = self._perturb_base_target(
                "align_panel", panel_nominal_xy, panel_nominal_yaw
            )
            base_cmd, station_dist, yaw_error = self._base_command(
                state,
                panel_station_xy,
                panel_station_yaw,
                panel_nominal_xy,
                panel_nominal_yaw,
            )
            self._set_base_tracking_target(
                panel_station_xy,
                panel_station_yaw,
                panel_nominal_xy,
                panel_nominal_yaw,
            )
            self.base_hold_stage = None
        elif self.stage in (Stage.THROUGH, Stage.SUCCESS):
            through_target_xy, through_target_yaw = self._perturb_base_target(
                "through", self.through_station_xy, self.through_station_yaw
            )
            base_cmd, station_dist, yaw_error = self._base_command(
                state,
                through_target_xy,
                through_target_yaw,
                self.through_station_xy,
                self.through_station_yaw,
            )
            self.through_stop_latched = _through_stop_latch(
                self.through_stop_latched, station_dist, yaw_error
            )
            if self.stage == Stage.THROUGH and self.through_stop_latched:
                base_cmd[:] = 0.0
            self._set_base_tracking_target(
                through_target_xy,
                through_target_yaw,
                self.through_station_xy,
                self.through_station_yaw,
            )
            self.base_hold_stage = None
        elif self.stage in BASE_HOLD_STAGES:
            hold_target = self._base_hold_target()
            base_cmd, station_dist, yaw_error = self._base_command(
                state,
                hold_target[0],
                hold_target[1],
                hold_target[2],
                hold_target[3],
            )
            base_cmd[:] = 0.0
        else:
            base_cmd[:] = 0.0

        base_cmd = _limit_base_commands(base_cmd)

        ee_local = (
            torch.as_tensor(ee_local_override, device=self.env.device, dtype=torch.float32).reshape(1, 3)
            if ee_local_override is not None
            else self._ee_local_target(ee_target_world)
        )
        # Stage-specific inverse paths may replace the generic slew output.
        # Keep the slew state aligned with the quaternion actually sent so a
        # later stage cannot resume from a stale grasp orientation.
        self.ee_orientation_command = self.active_grasp_quat.copy()
        orn = np.zeros((1, 3), dtype=np.float32)
        use_world_orientation = self.stage != Stage.SUCCESS
        if state.get("ee_orn") is not None:
            self.ee_orientation_error = self._quat_distance(
                _as_np(state["ee_orn"])[0], self.active_grasp_quat
            )
        if use_world_orientation:
            if not self.defer_batched_orientation_delta:
                orn[0] = self._grasp_orientation_delta(
                    _as_np(ee_local)[0], self.active_grasp_quat
                )
        else:
            orn[0] = self.default_ee_goal_delta_rpy
        if self.stage in (Stage.ALIGN_PANEL, Stage.BASE_ADVANCE):
            self.env._set_teleop_arm_control_mode("joint")
            arm_target = (
                self.align_panel_arm_joint_target
                if self.stage == Stage.ALIGN_PANEL
                else self.base_advance_arm_joint_target
            )
            if arm_target is None:
                raise RuntimeError(
                    f"{self.stage.name} arm joint hold target is unavailable"
                )
            hold_target = torch.as_tensor(
                arm_target,
                device=self.env.device,
                dtype=torch.float32,
            ).reshape(1, -1)
            self.env.teleop_arm_joint_pos_targets.copy_(hold_target)
            self.runner.set_teleop_targets(
                base_cmd.reshape(1, 3),
                gripper_target=np.asarray([[gripper]], dtype=np.float32),
            )
        elif self.stage in (Stage.THROUGH, Stage.SUCCESS):
            self.env._set_teleop_arm_control_mode("joint")
            middle_target = torch.as_tensor(
                self.middle_arm_joint_target,
                device=self.env.device,
                dtype=torch.float32,
            ).reshape(1, -1)
            self.env.teleop_arm_joint_pos_targets.copy_(middle_target)
            self.runner.set_teleop_targets(
                base_cmd.reshape(1, 3),
                gripper_target=np.asarray([[gripper]], dtype=np.float32),
            )
        else:
            self.env._set_teleop_arm_control_mode("ee")
            self.runner.set_teleop_targets(
                base_cmd.reshape(1, 3),
                ee_goal_cart=ee_local,
                ee_orn_delta_rpy=orn,
                gripper_target=np.asarray([[gripper]], dtype=np.float32),
                ee_goal_orn_quat_world=self.active_grasp_quat.reshape(1, 4),
                ee_goal_orn_quat_world_enabled=use_world_orientation,
            )
        high = np.concatenate([base_cmd, _as_np(ee_local)[0], orn[0], [gripper]]).astype(np.float32)
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

    def _current_panel_station(self, state):
        """Return the fixed doorway-centred station facing the door frame."""
        target_yaw = math.atan2(
            float(self.door_forward[1]), float(self.door_forward[0])
        )
        return self.panel_station_xy.copy(), target_yaw

    def _begin_panel_contact(self, orientation_world=None):
        """Move the gripper to the single panel-standoff pose."""
        self.panel_contact_target_world = None
        self.panel_contact_command_target_world = None
        self.panel_contact_inverse_target_world = None
        self.panel_contact_inverse_orientation_world = None
        self.panel_contact_inverse_command_world = None
        self.panel_contact_inverse_command_orientation_world = None
        self.prepared_panel_contact_command = None
        self.panel_contact_selected_offset = None
        self.panel_contact_orientation = (
            None
            if orientation_world is None
            else np.asarray(orientation_world, dtype=np.float32).copy()
        )
        self.panel_contact_settle_steps = 0
        self.stage = Stage.CONTACT_PANEL
        self.stage_steps = 0

    def _select_reachable_panel_contact_target(
        self,
        state,
        baseline_position_world,
        orientation_world,
        current_hinge_world,
        baseline_tangent_shift=0.0,
        baseline_vertical_shift=0.0,
    ):
        """Select the nearest reachable panel-grid target with one batched IK."""
        if (
            self.panel_bottom_z is None
            or self.panel_top_z is None
            or self.panel_hinge_to_free_direction is None
            or self.panel_hinge_to_free_edge is None
        ):
            raise RuntimeError("panel grid geometry is incomplete")

        step = float(self.panel_contact_grid_step)
        radius = float(self.panel_contact_grid_radius)
        margin = float(self.panel_contact_edge_margin)
        tangent_min = max(
            -radius,
            margin
            - self.panel_push_tangent_offset
            - float(baseline_tangent_shift),
        )
        tangent_max = min(
            radius,
            self.panel_hinge_to_free_edge
            - margin
            - self.panel_push_tangent_offset
            - float(baseline_tangent_shift),
        )
        vertical_min = max(
            -radius,
            margin
            - self.panel_push_height_offset
            - float(baseline_vertical_shift),
        )
        panel_height = self.panel_top_z - self.panel_bottom_z
        vertical_max = min(
            radius,
            panel_height
            - margin
            - self.panel_push_height_offset
            - float(baseline_vertical_shift),
        )

        tangent_indices = np.arange(
            math.ceil(tangent_min / step),
            math.floor(tangent_max / step) + 1,
            dtype=np.int32,
        )
        vertical_indices = np.arange(
            math.ceil(vertical_min / step),
            math.floor(vertical_max / step) + 1,
            dtype=np.int32,
        )
        tangent_grid, vertical_grid = np.meshgrid(
            tangent_indices, vertical_indices, indexing="ij"
        )
        offsets = np.stack(
            (tangent_grid.reshape(-1), vertical_grid.reshape(-1)), axis=-1
        ).astype(np.float64)
        offsets *= step
        distances = np.linalg.norm(offsets, axis=1)
        order = np.lexsort(
            (
                np.abs(offsets[:, 1]),
                np.abs(offsets[:, 0]),
                distances,
            )
        )
        offsets = offsets[order]
        distances = distances[order]

        current_tangent = _quat_rotate_np(
            self._yaw_quat(current_hinge_world),
            self.panel_hinge_to_free_direction,
        ).astype(np.float64)
        candidates = (
            np.asarray(baseline_position_world, dtype=np.float64)[None, :]
            + offsets[:, :1] * current_tangent[None, :]
        )
        candidates[:, 2] += offsets[:, 1]

        dof_names = list(state["dof_names"])
        dof_positions = _as_np(state["dof_pos"])[0]
        dof_index = {name: index for index, name in enumerate(dof_names)}
        initial_arm = np.asarray(
            [dof_positions[dof_index[name]] for name in self.ee_steady_model.arm_joint_names],
            dtype=np.float64,
        )
        root = _as_np(state["root_states"])[0].astype(np.float64)
        root_xyzw = root[3:7][[1, 2, 3, 0]]
        orientation_xyzw = np.asarray(
            orientation_world, dtype=np.float64
        )[[1, 2, 3, 0]]
        device = torch.device(self.env.device)
        dtype = torch.float64
        count = len(candidates)
        root_position = torch.as_tensor(
            root[:3], device=device, dtype=dtype
        ).expand(count, -1)
        root_quaternion = torch.as_tensor(
            root_xyzw, device=device, dtype=dtype
        ).expand(count, -1)
        target_position = torch.as_tensor(
            candidates, device=device, dtype=dtype
        )
        target_rotation = quaternion_xyzw_to_matrix(
            torch.as_tensor(
                orientation_xyzw, device=device, dtype=dtype
            ).expand(count, -1)
        )
        initial_joint_position = torch.as_tensor(
            initial_arm, device=device, dtype=dtype
        ).expand(count, -1)
        joint_lower = torch.as_tensor(
            self.ee_steady_model.lower_limits, device=device, dtype=dtype
        )
        joint_upper = torch.as_tensor(
            self.ee_steady_model.upper_limits, device=device, dtype=dtype
        )

        solutions, position_error, orientation_error = solve_b2z1_arm_ik(
            root_position,
            root_quaternion,
            target_position,
            target_rotation,
            initial_joint_position,
            joint_lower,
            joint_upper,
            iterations=120,
            damping=0.03,
        )
        joint_margin = torch.minimum(
            solutions - joint_lower,
            joint_upper - solutions,
        ).amin(dim=1)
        reachable = (
            (position_error <= 1.0e-4)
            & (orientation_error <= 2.0e-4)
            & (joint_margin >= 2.0e-3)
        )
        reachable_np = reachable.detach().cpu().numpy()
        if not np.any(reachable_np):
            score = (
                position_error / 0.01
            ) ** 2 + (
                orientation_error / self.grasp_orientation_tolerance
            ) ** 2
            best = int(torch.argmin(score).item())
            raise RuntimeError(
                "panel contact grid has no exact IK solution: "
                f"candidates={count} best_offset_m={offsets[best].round(4).tolist()} "
                f"position_error_m={float(position_error[best]):.6f} "
                f"orientation_error_rad={float(orientation_error[best]):.6f} "
                f"joint_margin_rad={float(joint_margin[best]):.6f}"
            )

        selected = int(np.flatnonzero(reachable_np)[0])
        selected_offset = offsets[selected].astype(np.float32)
        self.panel_contact_selected_offset = selected_offset.copy()
        print(
            "[doorman][panel-grid] "
            f"candidates={count} baseline_offset_m="
            f"[{self.panel_push_tangent_offset:.3f},"
            f"{self.panel_push_height_offset:.3f}] "
            f"selected_delta_m={selected_offset.round(4).tolist()} "
            f"distance_m={distances[selected]:.4f} "
            f"position_error_m={float(position_error[selected]):.8f} "
            f"orientation_error_rad={float(orientation_error[selected]):.8f} "
            f"joint_margin_rad={float(joint_margin[selected]):.6f}",
            flush=True,
        )
        return candidates[selected].astype(np.float32)

    def _prepare_panel_contact_command(self, state):
        """Prepare CONTACT_PANEL's target before the stateful command runs."""
        if self.prepared_panel_contact_command is not None:
            return
        current_hinge_joint = float(
            _as_np(state["door_joint_position"])[0, self.hinge_idx]
        )
        current_hinge_world = self.hinge_axis_sign * current_hinge_joint
        current_normal = _quat_rotate_np(
            self._yaw_quat(current_hinge_world), self.door_normal
        )
        orientation_seed = (
            self.panel_contact_orientation
            if self.panel_contact_orientation is not None
            else _as_np(state["ee_orn"])[0]
        )
        insertion_axis = _quat_rotate_np(
            orientation_seed, self.open_template["depth_axis"]
        )
        self.panel_contact_orientation = _quat_align_vectors_np(
            insertion_axis, -current_normal, orientation_seed
        )
        self.panel_contact_target_world = self._panel_tcp_for_orientation(
            state, current_hinge_world, self.panel_contact_orientation
        )
        baseline_position_world = (
            self.panel_contact_target_world
            + (self.panel_push_preload + self.panel_standoff_target_clearance)
            * current_normal
        )
        current_tangent = np.array(
            [-current_normal[1], current_normal[0], 0.0],
            dtype=np.float32,
        )
        semantic_axes = np.column_stack(
            (
                current_normal,
                current_tangent,
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
            )
        )
        sampled_position_world, command_orientation_world = (
            self._perturb_ee_target(
                "contact_panel",
                baseline_position_world,
                self.panel_contact_orientation,
                semantic_axes,
            )
        )
        semantic_position_delta = sampled_position_world - baseline_position_world
        panel_orientation_varies = any(
            value > 0.0
            for value in (
                self.target_noise.config.panel_roll_rad,
                self.target_noise.config.panel_tilt_horizontal_rad,
                self.target_noise.config.panel_tilt_vertical_rad,
            )
        )
        oriented_panel_tcp_world = (
            self._panel_tcp_for_orientation(
                state, current_hinge_world, command_orientation_world
            )
            if panel_orientation_varies
            else self.panel_contact_target_world
        )
        oriented_baseline_world = (
            oriented_panel_tcp_world
            + (self.panel_push_preload + self.panel_standoff_target_clearance)
            * current_normal
        )
        randomized_baseline_world = (
            oriented_baseline_world + semantic_position_delta
        )
        desired_position_world = self._select_reachable_panel_contact_target(
            state,
            randomized_baseline_world,
            command_orientation_world,
            current_hinge_world,
            baseline_tangent_shift=float(
                np.dot(semantic_position_delta, current_tangent)
            ),
            baseline_vertical_shift=float(semantic_position_delta[2]),
        )
        self.panel_contact_target_world = (
            desired_position_world
            - (self.panel_push_preload + self.panel_standoff_target_clearance)
            * current_normal
        )
        self.panel_contact_desired_ee_world = desired_position_world.copy()
        self.panel_contact_orientation = command_orientation_world.copy()
        self.panel_contact_command_target_world = (
            self.panel_contact_target_world.copy()
        )
        self.prepared_panel_contact_command = self._panel_contact_inverse_command(
            state,
            desired_position_world,
            command_orientation_world,
        )

    def _panel_contact_inverse_command(
        self, state, desired_position_world, desired_orientation_world
    ):
        """Map CONTACT_PANEL's geometric EE target to its steady command."""
        desired_position = np.asarray(desired_position_world, dtype=np.float32)
        desired_orientation = np.asarray(
            desired_orientation_world, dtype=np.float32
        )
        target_changed = bool(
            self.panel_contact_inverse_target_world is None
            or np.linalg.norm(
                desired_position - self.panel_contact_inverse_target_world
            )
            > 1.0e-4
            or self.panel_contact_inverse_orientation_world is None
            or self._quat_distance(
                desired_orientation,
                self.panel_contact_inverse_orientation_world,
            )
            > 1.0e-4
        )
        if target_changed:
            (
                self.panel_contact_inverse_command_world,
                self.panel_contact_inverse_command_orientation_world,
            ) = self._steady_compensated_ee_command(
                state,
                desired_position,
                desired_orientation,
                "panel_contact",
                position_tolerance_m=0.010,
            )
            self.panel_contact_inverse_target_world = desired_position.copy()
            self.panel_contact_inverse_orientation_world = (
                desired_orientation.copy()
            )
        return (
            self.panel_contact_inverse_command_world.copy(),
            self.panel_contact_inverse_command_orientation_world.copy(),
        )

    def _begin_through(self, opening_angle, opening_speed):
        """Start traversing as soon as the door crosses the accepted angle."""
        self.through_best_station_error = math.inf
        self.through_no_progress_steps = 0
        self.through_settle_steps = 0
        self.through_stability_failure_counts = {}
        self.stage = Stage.THROUGH
        self.stage_steps = 0
        print(
            f"[doorman][push] reached 89 deg threshold; begin traversal "
            f"hinge={opening_angle:.4f} "
            f"speed={opening_speed:.4f} max_speed="
            f"{self.push_max_observed_angular_speed:.4f}",
            flush=True,
        )

    def _through_tail_clearance(self, state):
        """Signed distance from the trunk rear edge to the door-frame plane."""
        base = _as_np(state["root_states"])[0]
        rear_offset_world = _quat_rotate_np(
            base[3:7],
            np.array([ROBOT_TRUNK_REAR_X, 0.0, 0.0], dtype=np.float32),
        )
        tail_xy = base[:2] + rear_offset_world[:2]
        return float(
            np.dot(
                tail_xy - self.initial_door_root[:2],
                self.door_forward[:2],
            )
        )

    def _base_advance_target(self, state):
        """Target the nearer of panel reachability and shoulder clearance."""
        if (
            self.push_base_start_xy is None
            or self.push_panel_contact_start_world is None
        ):
            base = _as_np(state["root_states"])[0]
            return base[:2].copy(), _quat_yaw(base[3:7])
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
        target_yaw = math.atan2(float(forward[1]), float(forward[0]))
        line_origin = np.asarray(self.panel_station_xy, dtype=np.float32)
        current_longitudinal = float(
            np.dot(base[:2] - line_origin, forward)
        )
        line_position = line_origin + current_longitudinal * forward

        def monotonic_target(proposed_longitudinal):
            previous = self.base_advance_target_longitudinal
            target_longitudinal = max(
                current_longitudinal,
                float(proposed_longitudinal),
                -math.inf if previous is None else float(previous),
            )
            self.base_advance_target_longitudinal = target_longitudinal
            return (
                line_origin + target_longitudinal * forward
            ).astype(np.float32)

        if self.push_reengage_reachable:
            # The forward lead only keeps the locomotion command outside its
            # dead zone while approaching.  Once the physical reach condition
            # is met, latch the current centre-line position and settle here.
            self.base_advance_target_longitudinal = current_longitudinal
            return line_position.astype(np.float32), target_yaw

        contact_local = _as_np(self._ee_local_target(current_contact))[0]
        contact_xy = np.asarray(contact_local[:2], dtype=np.float64)
        goal_ref_quat = _as_np(self.env.get_goal_reference_quat())[0]
        forward_local = _quat_rotate_inverse_np(
            goal_ref_quat,
            np.array([forward[0], forward[1], 0.0], dtype=np.float32),
        )[:2].astype(np.float64)
        lateral_shift_world = line_position - base[:2]
        lateral_shift_local = _quat_rotate_inverse_np(
            goal_ref_quat,
            np.array(
                [lateral_shift_world[0], lateral_shift_world[1], 0.0],
                dtype=np.float32,
            ),
        )[:2].astype(np.float64)
        contact_xy -= lateral_shift_local
        radius = float(self.push_reengage_reach_radius)
        quadratic_a = float(np.dot(forward_local, forward_local))
        quadratic_b = float(np.dot(contact_xy, forward_local))
        quadratic_c = float(np.dot(contact_xy, contact_xy) - radius * radius)
        discriminant = (
            quadratic_b * quadratic_b - quadratic_a * quadratic_c
        )
        if quadratic_c <= 0.0:
            required_advance = 0.0
        elif quadratic_a <= 1.0e-10 or discriminant < 0.0:
            self.push_base_safety_scale = 0.0
            return monotonic_target(current_longitudinal), target_yaw
        else:
            root = math.sqrt(max(0.0, discriminant))
            interval_start = (quadratic_b - root) / quadratic_a
            interval_end = (quadratic_b + root) / quadratic_a
            if interval_end < 0.0:
                self.push_base_safety_scale = 0.0
                return monotonic_target(current_longitudinal), target_yaw
            required_advance = max(0.0, interval_start)
        required_advance += self.push_reengage_target_lead

        # Search shoulder clearance only at candidate target positions.  The
        # target remains on the fixed doorway-centre line for every candidate.
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
        shoulder_xy = np.asarray(
            shoulders[0, self.front_shoulder_body_indices, :2],
            dtype=np.float32,
        )
        required_clearance = (
            self.push_shoulder_collision_radius + self.panel_half_thickness
            + self.push_shoulder_safety_margin
        )
        candidate_count = max(
            2, int(math.ceil(required_advance / 0.005)) + 1
        )
        candidate_advances = np.linspace(
            0.0, required_advance, candidate_count, dtype=np.float32
        )
        candidate_targets = (
            line_position[None, :]
            + candidate_advances[:, None] * forward[None, :]
        )
        candidate_displacements = candidate_targets - base[None, :2]
        points = (
            shoulder_xy[None, :, :]
            + candidate_displacements[:, None, :]
        )
        segment = panel_end - panel_start
        segment_norm_sq = float(np.dot(segment, segment))
        if segment_norm_sq < 1.0e-10:
            center_distances = np.linalg.norm(
                points - panel_start[None, None, :], axis=-1
            )
        else:
            projection = np.clip(
                np.sum(
                    (points - panel_start[None, None, :])
                    * segment[None, None, :],
                    axis=-1,
                )
                / segment_norm_sq,
                0.0,
                1.0,
            )
            closest = (
                panel_start[None, None, :]
                + projection[:, :, None] * segment[None, None, :]
            )
            center_distances = np.linalg.norm(points - closest, axis=-1)
        candidate_min_distances = np.min(center_distances, axis=1)
        candidate_clearances = candidate_min_distances - (
            self.push_shoulder_collision_radius + self.panel_half_thickness
        )
        candidate_safe = candidate_min_distances >= required_clearance
        unsafe = np.flatnonzero(~candidate_safe)
        if unsafe.size:
            last_safe = int(unsafe[0]) - 1
            target_index = max(0, last_safe)
        else:
            target_index = len(candidate_targets) - 1
        target_advance = float(candidate_advances[target_index])
        current_projection = np.clip(
            ((shoulder_xy - panel_start[None, :]) @ segment)
            / max(segment_norm_sq, 1.0e-10),
            0.0,
            1.0,
        )
        current_closest = (
            panel_start[None, :]
            + current_projection[:, None] * segment[None, :]
        )
        current_center_distances = np.linalg.norm(
            shoulder_xy - current_closest, axis=1
        )
        self.push_shoulder_min_clearance = float(
            np.min(current_center_distances)
            - self.push_shoulder_collision_radius
            - self.panel_half_thickness
        )
        self.push_shoulder_predicted_min_clearance = float(
            candidate_clearances[target_index]
        )
        if required_advance > 1.0e-6:
            self.push_base_safety_scale = float(
                np.clip(target_advance / required_advance, 0.0, 1.0)
            )
        else:
            self.push_base_safety_scale = 1.0

        proposed_longitudinal = current_longitudinal + target_advance
        return monotonic_target(proposed_longitudinal), target_yaw

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
        self.turn_target_handle_angle = max(
            float(self.turn_target_handle_angle or 0.74)
            + self.target_noise.positive_margin(
                "turn_handle_overtravel_rad",
                self.target_noise.config.turn_handle_overtravel_rad,
            ),
            current_angle,
        )
        handle_delta = self.turn_target_handle_angle - current_angle
        current_hinge = float(_as_np(state["door_joint_position"])[0, self.hinge_idx])
        self.turn_start_hinge_angle = current_hinge
        # Keep the latch depressed until the randomized door edge has fully
        # cleared the frame.  Plan beyond this episode's geometric threshold;
        # the margin absorbs steady Cartesian tracking error at the endpoint.
        if self.turn_gap_angle is None:
            raise RuntimeError("door gap angle is unavailable")
        gap_delta = self.hinge_axis_sign * (
            self.turn_gap_angle
            + math.radians(2.0)
            + self.target_noise.positive_margin(
                "turn_gap_overtravel_rad",
                self.target_noise.config.turn_gap_overtravel_rad,
            )
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

    def _pregrasp_pitch_candidates(self):
        """Return downward wrist-pitch candidates in increasing order."""
        horizontal = np.asarray(self.grasp_orientation_quat, dtype=np.float32)
        joint_axis_world = _quat_rotate_np(
            horizontal, np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
        shaft_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = []
        for pitch_deg in (0.0, 5.0, 10.0, 15.0, 20.0):
            if pitch_deg == 0.0:
                orientation = horizontal.copy()
            else:
                angle = math.radians(pitch_deg)
                alternatives = [
                    _quat_mul_np(
                        _quat_axis_angle_np(joint_axis_world, signed_angle),
                        horizontal,
                    )
                    for signed_angle in (-angle, angle)
                ]
                orientation = min(
                    alternatives,
                    key=lambda quat: float(_quat_rotate_np(quat, shaft_local)[2]),
                )
            orientation /= max(float(np.linalg.norm(orientation)), 1.0e-8)
            end_position = self.center_standoff_ee_world.copy()
            fixed_inner_offset_world = _quat_rotate_np(
                orientation, self.fixed_inner_center_local
            )
            end_position[2] = (
                self.fixed_surface_target_z - fixed_inner_offset_world[2]
            )
            candidates.append(
                {
                    "pitch_deg": pitch_deg,
                    "orientation": orientation.astype(np.float32),
                    "end_position": end_position.astype(np.float32),
                }
            )
        return candidates

    def _apply_pregrasp_candidate(self, candidate):
        """Commit one selected pitch and its fixed-jaw-anchored target height."""
        orientation = np.asarray(candidate["orientation"], dtype=np.float32)
        end_position = np.asarray(candidate["end_position"], dtype=np.float32)
        self.pregrasp_pitch_deg = float(candidate["pitch_deg"])
        self.grasp_orientation_quat = orientation.copy()
        self.reachable_grasp_orientation_quat = orientation.copy()
        self.grasp_ee_offset_world = _quat_rotate_np(
            orientation, self.open_template["tip_capture_center"]
        )
        for name in (
            "center_standoff_ee_world",
            "center_standoff_nominal_world",
            "center_insert_waypoint_world",
            "center_insert_nominal_world",
            "grasp_waypoint_world",
            "grasp_nominal_world",
        ):
            target = getattr(self, name)
            target[2] = end_position[2]

    def _plan_pregrasp_path(self, state, start_position_world):
        """Select one reachable wrist pitch, then solve only that full path."""
        if "arm_align_path" in self.precomputed_steady_paths:
            command_target, command_orientation = self._perturb_ee_target(
                "pregrasp",
                self.center_standoff_ee_world,
                self.grasp_orientation_quat,
            )
            self.arm_align_desired_target_world = command_target.copy()
            return self._build_steady_compensated_path(
                state,
                start_position_world,
                command_target,
                command_orientation,
                start_orientation_world=_as_np(state["ee_orn"])[0],
                label="arm_align_path",
            )
        candidates = self._pregrasp_pitch_candidates()
        dof_names = list(state["dof_names"])
        dof_positions = _as_np(state["dof_pos"])[0]
        joint_positions = {
            name: float(dof_positions[index])
            for index, name in enumerate(dof_names)
        }
        root = _as_np(state["root_states"])[0]
        selected = None
        best = None
        diagnostics = []
        for candidate in candidates:
            command_position, command_orientation = self._perturb_ee_target(
                "pregrasp",
                candidate["end_position"],
                candidate["orientation"],
            )
            result = self.ee_steady_model.inverse(
                command_position,
                command_orientation,
                base_position_world=root[:3],
                base_orientation_world_wxyz=root[3:7],
                joint_positions=joint_positions,
                position_tolerance_m=float("inf"),
                orientation_tolerance=float("inf"),
            )
            position_error = float(np.linalg.norm(result.position_error))
            orientation_error = float(np.linalg.norm(result.orientation_error))
            diagnostics.append(
                (candidate["pitch_deg"], position_error, orientation_error)
            )
            score = position_error + orientation_error
            if best is None or score < best[0]:
                best = (score, candidate)
            if position_error <= 1.0e-4 and orientation_error <= 1.0e-4:
                selected = candidate
                break
        if selected is None:
            selected = best[1]
            print(
                "[doorman][pregrasp-pitch][warn] no exact candidate; "
                f"using best pitch={selected['pitch_deg']:.0f}deg",
                flush=True,
            )
        self._apply_pregrasp_candidate(selected)
        print(
            f"[doorman][pregrasp-pitch] selected={self.pregrasp_pitch_deg:.0f}deg "
            f"candidates={diagnostics}",
            flush=True,
        )
        command_target, command_orientation = self._perturb_ee_target(
            "pregrasp",
            self.center_standoff_ee_world,
            self.grasp_orientation_quat,
        )
        self.arm_align_desired_target_world = command_target.copy()
        return self._build_steady_compensated_path(
            state,
            start_position_world,
            command_target,
            command_orientation,
            start_orientation_world=_as_np(state["ee_orn"])[0],
            label="arm_align_path",
        )

    def _steady_compensated_ee_command(
        self,
        state,
        desired_position_world,
        desired_orientation_world,
        label,
        position_tolerance_m=None,
    ):
        """Invert the mathematical steady EE map once at a stable stage boundary."""
        if self.defer_steady_inverse:
            raise DeferredSteadyInverse(
                self,
                state,
                desired_position_world,
                desired_orientation_world,
                label,
                position_tolerance_m,
            )
        dof_names = list(state["dof_names"])
        dof_positions = _as_np(state["dof_pos"])[0]
        joint_positions = {
            name: float(dof_positions[index])
            for index, name in enumerate(dof_names)
        }
        root = _as_np(state["root_states"])[0]
        desired_position = np.asarray(desired_position_world, dtype=np.float32)
        desired_orientation = np.asarray(
            desired_orientation_world, dtype=np.float32
        )
        finite_inputs = bool(
            np.all(np.isfinite(desired_position))
            and np.all(np.isfinite(desired_orientation))
            and np.all(np.isfinite(root[:7]))
            and np.all(np.isfinite(dof_positions))
        )
        if not finite_inputs:
            raise RuntimeError(
                f"steady_inverse_non_finite:{label}:"
                f"desired_position={desired_position.tolist()}:"
                f"desired_orientation={desired_orientation.tolist()}:"
                f"root={root[:7].tolist()}:"
                f"finite_dof={bool(np.all(np.isfinite(dof_positions)))}"
            )
        inverse_kwargs = {}
        if position_tolerance_m is not None:
            inverse_kwargs["position_tolerance_m"] = position_tolerance_m
        result = self.ee_steady_model.inverse(
            desired_position_world,
            desired_orientation_world,
            base_position_world=root[:3],
            base_orientation_world_wxyz=root[3:7],
            joint_positions=joint_positions,
            **inverse_kwargs,
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
    ):
        """Map a straight measured-EE path through the mathematical inverse."""
        precomputed = self.precomputed_steady_paths.pop(label, None)
        if precomputed is not None:
            return precomputed
        if self.defer_path_solve and label == "arm_forward_path":
            request = _steady_path_request(
                state,
                start_position_world,
                end_position_world,
                orientation_world,
                start_orientation_world_wxyz=(
                    orientation_world
                    if start_orientation_world is None
                    else start_orientation_world
                ),
            )
            request["path_kind"] = "steady"
            raise DeferredPathSolve(self, "steady", label, request)
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
        if self.defer_path_solve:
            raise DeferredPathSolve(
                self,
                "release",
                label,
                _release_path_request(
                    state,
                    start_position_world,
                    end_position_world,
                    start_orientation_world,
                    end_orientation_world,
                    start_gripper_position,
                    end_gripper_position,
                    steps,
                    self.pregrasp_position_tolerance,
                    int(self.env.gripper_joint_ids[0].item()),
                ),
            )
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

    def _complete_deferred_path(self, label, positions, orientations):
        """Commit one worker result at the same FSM boundary as scalar planning."""
        positions = np.asarray(positions, dtype=np.float32)
        orientations = np.asarray(orientations, dtype=np.float32)
        previous_stage = self.stage
        if label == "arm_forward_path":
            self.arm_forward_command_path = positions
            self.arm_forward_orientation_path = orientations
            self.arm_forward_path_index = 0
            self.arm_forward_command_world = positions[-1].copy()
            self.arm_forward_command_quat = orientations[-1].copy()
            self.arm_stability_window.clear()
            self.stage = Stage.CENTER_HANDLE
            self.stage_steps = 0
        elif label == "release_open_path":
            self.release_open_command_path = positions
            self.release_open_orientation_path = orientations
            self.ee_command_world = positions[0].copy()
            self.release_stable_steps = 0
        elif label == "release_retreat_path":
            self.release_retreat_command_path = positions
            self.release_retreat_orientation_path = orientations
            self.release_retreat_path_index = 0
            self.ee_command_world = positions[0].copy()
            self.release_stable_steps = 0
        elif label == "release_restore_path":
            self.release_restore_command_path = positions
            self.release_restore_orientation_path = orientations
            self.ee_command_world = positions[0].copy()
            self.arm_stability_window.clear()
            self.release_stable_steps = 0
            self.stage = Stage.RESTORE_GRIPPER
            self.stage_steps = 0
        else:
            raise RuntimeError(f"unsupported deferred path label: {label}")

        self.stage_steps += 1
        if (
            self.stage in self.stage_limits
            and self.stage_steps > self.stage_limits[self.stage]
        ):
            self.failure_reason = f"stage_timeout:{self.stage.name.lower()}"
            self.failure_stage = self.stage.name
            self.stage = Stage.FAILURE
        if self.stage != previous_stage:
            self.stage_exit_diagnostics[previous_stage.name] = dict(
                self.last_acceptance_diagnostics
            )

    def _arm_pose_is_stable(self, state):
        """Require an unchanged measured EE pose over 0.3 seconds."""
        batched_result = self.batched_stability_result
        self.batched_stability_result = None
        position = _as_np(state["ee_pos"])[0].copy()
        orientation = _as_np(state["ee_orn"])[0].copy()
        self.arm_stability_window.append((position, orientation))
        if len(self.arm_stability_window) < self.arm_stability_window.maxlen:
            self.last_stability_diagnostics = {
                "samples": len(self.arm_stability_window),
                "required_samples": int(self.arm_stability_window.maxlen),
                "stable": False,
            }
            if batched_result is not None and not batched_result["enough"]:
                self.gpu_stability_uses += 1
            return False
        if batched_result is not None:
            position_span = batched_result["position_span"]
            orientation_span = batched_result["orientation_span"]
            if (
                abs(position_span - 0.001) > 0.00005
                and abs(orientation_span - 0.003) > 0.0005
            ):
                self.gpu_stability_uses += 1
                self.last_stability_diagnostics = {
                    "samples": int(self.arm_stability_window.maxlen),
                    "position_span_m": float(position_span),
                    "orientation_span_rad": float(orientation_span),
                    "stable": bool(batched_result["stable"]),
                    "source": "gpu",
                }
                return batched_result["stable"]
            self.gpu_stability_fallbacks += 1
        positions = np.asarray([sample[0] for sample in self.arm_stability_window])
        orientations = [sample[1] for sample in self.arm_stability_window]
        position_span = float(
            np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1))
        )
        orientation_span = max(
            self._quat_distance(orientation, sample) for sample in orientations
        )
        stable = bool(
            position_span <= 0.001
            and orientation_span <= 0.003
        )
        self.last_stability_diagnostics = {
            "samples": int(self.arm_stability_window.maxlen),
            "position_span_m": position_span,
            "orientation_span_rad": float(orientation_span),
            "stable": stable,
            "source": "scalar",
        }
        return stable

    def advance(
        self,
        state,
        station_dist,
        yaw_error,
        precomputed_geometry_metrics=None,
        precomputed_geometry_authoritative=False,
    ):
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
        pregrasp_acceptance_target = (
            self.arm_align_desired_target_world
            if self.arm_align_desired_target_world is not None
            else self.center_standoff_ee_world
        )
        pregrasp_dist = (
            float(np.linalg.norm(ee - pregrasp_acceptance_target))
            if ee is not None and pregrasp_acceptance_target is not None
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
        ee_orientation_error = float(self.ee_orientation_error)
        self.last_acceptance_diagnostics = {
            "stage": previous_stage.name,
            "stage_steps": int(self.stage_steps),
            "station_distance_m": float(station_dist),
            "yaw_error_rad": float(yaw_error),
            "base_linear_speed_m_s": speed,
            "pregrasp_distance_m": pregrasp_dist,
            "ee_orientation_error_rad": ee_orientation_error,
            "handle_contact_force_n": handle_contact_norms.astype(float).tolist(),
        }
        metrics = {}
        if self.stage in (Stage.CENTER_HANDLE, Stage.GRASP_HANDLE):
            if (
                precomputed_geometry_metrics is not None
                and (
                    precomputed_geometry_authoritative
                    or not self._gpu_geometry_needs_scalar(
                        precomputed_geometry_metrics
                    )
                )
            ):
                metrics = precomputed_geometry_metrics
                if precomputed_geometry_authoritative:
                    self.gpu_geometry_fallbacks += 1
                else:
                    self.gpu_geometry_uses += 1
            else:
                metrics = self._geometry_metrics(state, grasp_center)
                if precomputed_geometry_metrics is not None:
                    self.gpu_geometry_fallbacks += 1
        if self.stage in (Stage.CENTER_HANDLE, Stage.GRASP_HANDLE):
            self.center_metrics = metrics
            if metrics.get("valid"):
                self.last_acceptance_diagnostics.update(
                    {
                        "pocket_normal_error_m": float(
                            metrics["pocket_normal_error"]
                        ),
                        "pocket_axial_error_m": float(
                            metrics["pocket_axial_error"]
                        ),
                        "channel_vertical_error_m": float(
                            metrics["channel_vertical_error"]
                        ),
                        "channel_span_m": float(metrics["z_span"]),
                        "upper_gap_m": float(metrics["upper_gap"]),
                        "lower_gap_m": float(metrics["lower_gap"]),
                        "fixed_handle_clearance_m": float(
                            metrics["fixed_handle_clearance"]
                        ),
                        "moving_handle_clearance_m": float(
                            metrics["moving_handle_clearance"]
                        ),
                        "handle_axial_overlap_m": float(
                            self._center_handle_axial_overlap(metrics)
                        ),
                        "center_vertical_plan": dict(
                            self.center_vertical_plan or {}
                        ),
                    }
                )
        if (
            self.stage == Stage.CONTACT_PANEL
            and self.push_start_hinge_angle is not None
            and hinge >= math.radians(89.0)
        ):
            opening_speed = float(
                _as_np(state["door_joint_velocity"])[0, self.hinge_idx]
            )
            self.push_max_observed_angular_speed = max(
                self.push_max_observed_angular_speed,
                max(0.0, opening_speed),
            )
            self._begin_through(hinge, opening_speed)

        if self.stage == Stage.APPROACH:
            angular_speed = float(
                np.linalg.norm(_as_np(state["root_states"])[0, 10:13])
            )
            base_stable = bool(
                station_dist < self.station_tolerance
                and yaw_error < BASE_YAW_TOLERANCE_RAD
                and speed < 0.02
                and angular_speed < 0.05
            )
            self.approach_settle_steps = (
                self.approach_settle_steps + 1 if base_stable else 0
            )
            if self.approach_settle_steps >= 25:
                (
                    self.arm_align_command_path,
                    self.arm_align_orientation_path,
                ) = self._plan_pregrasp_path(state, ee)
                self.arm_align_path_index = 0
                self.arm_align_command_world = self.arm_align_command_path[-1].copy()
                self.arm_align_command_quat = self.arm_align_orientation_path[-1].copy()
                self.arm_stability_window.clear()
                self.env.begin_cartesian_transition()
                self.stage = Stage.PREGRASP; self.stage_steps = 0
        elif self.stage in (Stage.PREGRASP, Stage.OPEN_GRIPPER):
            arm_stable = self._arm_pose_is_stable(state)
            align_path_complete = bool(
                self.arm_align_command_path is not None
                and self.arm_align_path_index
                >= len(self.arm_align_command_path) - 1
            )
            steady_error = ee - self.center_standoff_ee_world
            arm_aligned = (
                align_path_complete
                and
                pregrasp_dist <= self.pregrasp_position_tolerance
                and ee_orientation_error <= self.grasp_orientation_tolerance
                and station_dist < self.station_tolerance
                and yaw_error < BASE_YAW_TOLERANCE_RAD
                and speed < 0.02
                and not np.any(handle_contact_norms >= self.grasp_contact_threshold)
                and arm_stable
            )
            arm_joint_ids = _as_np(self.env.arm_joint_ids).astype(int)
            self.last_acceptance_diagnostics.update(
                {
                    "align_path_complete": align_path_complete,
                    "arm_stable": bool(arm_stable),
                    "steady_error_m": float(np.linalg.norm(steady_error)),
                    "position_tolerance_m": float(self.pregrasp_position_tolerance),
                    "orientation_tolerance_rad": float(self.grasp_orientation_tolerance),
                    "station_tolerance_m": float(self.station_tolerance),
                    "arm_aligned": bool(arm_aligned),
                    "path_index": int(self.arm_align_path_index),
                    "path_length": int(
                        0
                        if self.arm_align_command_path is None
                        else len(self.arm_align_command_path)
                    ),
                    "stability": dict(self.last_stability_diagnostics),
                    "ee_position_world_m": ee.astype(float).tolist(),
                    "pregrasp_target_world_m": pregrasp_acceptance_target.astype(float).tolist(),
                    "ee_linear_speed_m_s": float(
                        np.linalg.norm(_as_np(state["ee_vel"])[0, :3])
                    ),
                    "arm_joint_speed_max_rad_s": float(
                        np.max(np.abs(_as_np(state["dof_vel"])[0, arm_joint_ids]))
                    ),
                    "arm_joint_position_rad": _as_np(state["dof_pos"])[
                        0, arm_joint_ids
                    ].astype(float).tolist(),
                }
            )
            if self.stage == Stage.PREGRASP:
                self.pregrasp_settle_steps = (
                    self.pregrasp_settle_steps + 1 if arm_aligned else 0
                )
                if self.pregrasp_settle_steps >= 5:
                    self.stage = Stage.OPEN_GRIPPER
                    self.stage_steps = 0
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
                self.stage == Stage.OPEN_GRIPPER
                and arm_aligned
                and abs(actual_q - self.q_open) < 0.02
                and gripper_speed < 0.05
            )
            self.gripper_open_settle = (
                self.gripper_open_settle + 1 if gripper_open_stable else 0
            )
            if self.stage == Stage.OPEN_GRIPPER and self.gripper_open_settle >= 10:
                self.arm_forward_target_world = self.grasp_waypoint_world.copy()
                open_metrics = self._geometry_metrics(state, grasp_center)
                if not open_metrics.get("valid"):
                    self.failure_reason = "open_channel_geometry_unavailable"
                    self.stage = Stage.FAILURE
                    return self.stage
                fixed_is_upper = bool(
                    np.asarray(open_metrics["fixed_center"])[2]
                    >= grasp_center[2]
                )
                fixed_gap = float(
                    open_metrics["upper_gap" if fixed_is_upper else "lower_gap"]
                )
                self.center_vertical_plan = _solve_center_vertical_shift(
                    open_metrics["channel_vertical_error"], fixed_gap
                )
                if not self.center_vertical_plan["acceptance_feasible"]:
                    self.failure_reason = (
                        "center_vertical_geometry_infeasible:"
                        f"channel={open_metrics['channel_vertical_error']:.6f},"
                        f"fixed_gap={fixed_gap:.6f}"
                    )
                    self.stage = Stage.FAILURE
                    return self.stage
                self.arm_forward_target_world[2] += self.center_vertical_plan[
                    "shift_m"
                ]
                (
                    arm_forward_command_target,
                    arm_forward_command_orientation,
                ) = self._perturb_ee_target(
                    "center_handle",
                    self.arm_forward_target_world,
                    self.grasp_orientation_quat,
                )
                self.arm_forward_desired_target_world = (
                    arm_forward_command_target.copy()
                )
                (
                    self.arm_forward_command_path,
                    self.arm_forward_orientation_path,
                ) = self._build_steady_compensated_path(
                    state,
                    ee,
                    arm_forward_command_target,
                    arm_forward_command_orientation,
                )
                self.arm_forward_path_index = 0
                self.arm_forward_command_world = (
                    self.arm_forward_command_path[-1].copy()
                )
                self.arm_forward_command_quat = (
                    self.arm_forward_orientation_path[-1].copy()
                )
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
                forward_acceptance_target = (
                    self.arm_forward_desired_target_world
                    if self.arm_forward_desired_target_world is not None
                    else self.arm_forward_target_world
                )
                forward_distance = float(
                    np.linalg.norm(ee - forward_acceptance_target)
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
                self.grasp_latched = True
                self.grasp_settle_steps = 0
                print(
                    f"[doorman][grasp] latched q_actual={gripper_pos:.4f} "
                    f"semantic_command={self.q_closed:.4f} "
                    f"fixed_clearance={metrics['fixed_handle_clearance']:.4f} "
                    f"moving_clearance={metrics['moving_handle_clearance']:.4f} "
                    f"contact_axial_delta={opposed_metrics['axial_delta']:.4f} "
                    f"contact_closing={opposed_metrics['closing_coordinates'].round(4).tolist()} "
                    f"forces={handle_contact_norms.round(3).tolist()}",
                    flush=True,
                )
            if self.grasp_latched and not bilateral_recent:
                self.grasp_settle_steps = 0
            peak_handle_force = float(np.max(handle_contact_norms))
            if self.grasp_latched and peak_handle_force > 50.0:
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
            self.last_acceptance_diagnostics.update(
                {
                    "gripper_position_rad": gripper_pos,
                    "gripper_speed_rad_s": gripper_speed,
                    "grasp_latched": bool(self.grasp_latched),
                    "bilateral_recent": bilateral_recent,
                    "opposed_grasp": bool(opposed_grasp),
                    "grasp_stable": bool(grasp_stable),
                    "grasp_settle_steps": int(self.grasp_settle_steps),
                    "contact_age_steps": (
                        self.stage_steps - self.handle_contact_last_seen
                    ).astype(int).tolist(),
                    "fixed_handle_clearance_m": float(
                        metrics.get("fixed_handle_clearance", math.inf)
                    ),
                    "moving_handle_clearance_m": float(
                        metrics.get("moving_handle_clearance", math.inf)
                    ),
                    "handle_between_jaws": bool(
                        metrics.get("handle_between_jaws", False)
                    ),
                    "axial_error_m": float(metrics.get("axial_error", math.inf)),
                    "vertical_error_m": float(
                        metrics.get("vertical_error", math.inf)
                    ),
                    "panel_clearance_m": float(
                        metrics.get("panel_clearance", math.inf)
                    ),
                    "opposed_axial_delta_m": float(
                        opposed_metrics.get("axial_delta", math.inf)
                    ),
                    "opposed_closing_coordinates_m": np.asarray(
                        opposed_metrics.get(
                            "closing_coordinates", [math.inf, math.inf]
                        )
                    ).astype(float).tolist(),
                }
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
                self.turn_release_slow_steps = 0
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
            hinge_speed = abs(
                float(
                    _as_np(state["door_joint_velocity"])[
                        0, self.hinge_idx
                    ]
                )
            )
            hinge_speed_below_limit = (
                hinge_speed < self.turn_release_max_hinge_speed
            )
            self.turn_release_slow_steps = (
                self.turn_release_slow_steps + 1
                if latch_released and door_gap_open and hinge_speed_below_limit
                else 0
            )
            door_hinge_slow = (
                self.turn_release_slow_steps
                >= self.turn_release_slow_steps_required
            )
            self.last_acceptance_diagnostics.update(
                {
                    "hinge_speed_rad_s": hinge_speed,
                    "turn_release_max_hinge_speed_rad_s": (
                        self.turn_release_max_hinge_speed
                    ),
                    "hinge_speed_below_turn_release_limit": bool(
                        hinge_speed_below_limit
                    ),
                    "turn_release_slow_steps": int(
                        self.turn_release_slow_steps
                    ),
                    "turn_release_slow_steps_required": int(
                        self.turn_release_slow_steps_required
                    ),
                    "turn_release_slow_duration_s": (
                        self.turn_release_slow_duration
                    ),
                    "door_hinge_slow": bool(door_hinge_slow),
                }
            )
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
            if (
                latch_released
                and door_gap_open
                and door_hinge_slow
                and grasp_retained
            ):
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
                self.ee_command_world = ee.copy()
                self.release_stable_steps = 0
                self.stage = Stage.RETURN_HANDLE; self.stage_steps = 0
        if self.stage == Stage.TURN_HANDLE:
            self.turn_last_handle_angle = handle_joint
            self.turn_last_hinge_angle = hinge
        elif self.stage in (
            Stage.RETURN_HANDLE,
            Stage.OPEN_AND_CLEAR,
            Stage.RETREAT_ARM,
        ):
            release_contact_clear = not np.any(handle_contact_present)
            gripper_open = abs(gripper_pos - float(self.q_open)) < 0.08
            if self.stage == Stage.RETURN_HANDLE:
                handle_horizontal = abs(handle_joint) < 0.05
                self.release_stable_steps = (
                    self.release_stable_steps + 1
                    if self.stage_steps >= self.release_return_steps
                    and handle_horizontal
                    else 0
                )
                if self.release_stable_steps >= 10:
                    self.stage = Stage.OPEN_AND_CLEAR
                    self.stage_steps = 0
                    self.release_open_start_world = ee.copy()
                    self.release_start_quat = _as_np(state["ee_orn"])[0].copy()
                    release_open_end_world = (
                        self.release_open_start_world
                        + np.array(
                            [0.0, 0.0, self.release_vertical_delta],
                            dtype=np.float32,
                        )
                    )
                    (
                        release_open_command_end,
                        release_open_command_orientation,
                    ) = self._perturb_ee_target(
                        "open_and_clear",
                        release_open_end_world,
                        self.release_start_quat,
                    )
                    (
                        self.release_open_command_path,
                        self.release_open_orientation_path,
                    ) = self._build_release_inverse_path(
                        state,
                        self.release_open_start_world,
                        release_open_command_end,
                        self.release_start_quat,
                        release_open_command_orientation,
                        self.q_closed,
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
            elif self.stage == Stage.OPEN_AND_CLEAR:
                fully_released = gripper_open and release_contact_clear
                self.release_stable_steps = (
                    self.release_stable_steps + 1 if fully_released else 0
                )
                if self.release_stable_steps >= 10:
                    self.stage = Stage.RETREAT_ARM
                    self.stage_steps = 0
                    self.release_retreat_start_world = ee.copy()
                    self.release_retreat_target_world = (
                        self.release_retreat_start_world
                        + 0.10 * self.door_normal
                        + np.array([0.0, 0.0, 0.02], dtype=np.float32)
                    )
                    (
                        self.release_retreat_command_target_world,
                        release_retreat_command_orientation,
                    ) = self._perturb_ee_target(
                        "retreat_arm",
                        self.release_retreat_target_world,
                        self.release_start_quat,
                    )
                    retreat_steps = max(
                        50,
                        int(
                            math.ceil(
                                float(
                                    np.linalg.norm(
                                        self.release_retreat_command_target_world
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
                        self.release_retreat_command_target_world,
                        self.release_start_quat,
                        release_retreat_command_orientation,
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
            elif self.stage == Stage.RETREAT_ARM:
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
                    self.release_endpoint_world = self._ee_world_target(
                        self.push_retracted_ee_goal_local
                    )
                    current_orientation = _as_np(state["ee_orn"])[0].copy()
                    dof_names = list(state["dof_names"])
                    current_wrist_roll = float(
                        _as_np(state["dof_pos"])[
                            0, dof_names.index("joint6")
                        ]
                    )
                    wrist_roll_target = 0.0
                    wrist_roll_delta = wrist_roll_target - current_wrist_roll
                    wrist_axis_world = _quat_rotate_np(
                        current_orientation,
                        np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    )
                    self.release_endpoint_quat = _quat_mul_np(
                        _quat_axis_angle_np(wrist_axis_world, wrist_roll_delta),
                        current_orientation,
                    )
                    (
                        release_command_endpoint_world,
                        release_command_endpoint_quat,
                    ) = self._perturb_ee_target(
                        "restore_gripper",
                        self.release_endpoint_world,
                        self.release_endpoint_quat,
                    )
                    restore_steps = max(
                        50,
                        int(
                            math.ceil(
                                float(
                                    np.linalg.norm(
                                        release_command_endpoint_world - ee
                                    )
                                )
                                / 0.004
                            )
                        ),
                    )
                    (
                        self.release_restore_command_path,
                        self.release_restore_orientation_path,
                    ) = self._build_release_inverse_path(
                        state,
                        ee,
                        release_command_endpoint_world,
                        current_orientation,
                        release_command_endpoint_quat,
                        self.q_open,
                        self.q_closed,
                        steps=restore_steps,
                        label="release_restore_path",
                    )
                    print(
                        "[doorman][release] restoring wrist roll only "
                        f"q6_current={current_wrist_roll:.5f} "
                        f"q6_target={wrist_roll_target:.5f} "
                        f"delta={wrist_roll_delta:.5f}",
                        flush=True,
                    )
                    self.ee_command_world = (
                        self.release_restore_command_path[0].copy()
                    )
                    self.arm_stability_window.clear()
                    self.release_stable_steps = 0
                    self.stage = Stage.RESTORE_GRIPPER; self.stage_steps = 0
        elif self.stage == Stage.RESTORE_GRIPPER:
            gripper_closed = abs(gripper_pos - float(self.q_closed)) < 0.03
            orientation_restored = self._quat_distance(
                _as_np(state["ee_orn"])[0], self.release_endpoint_quat
            ) < 0.05
            arm_stable = self._arm_pose_is_stable(state)
            restored = bool(
                self.stage_steps >= 50
                and gripper_closed
                and orientation_restored
                and np.linalg.norm(ee - self.release_endpoint_world) <= 0.010
                and arm_stable
            )
            self.release_stable_steps = (
                self.release_stable_steps + 1 if restored else 0
            )
            if self.release_stable_steps >= 10:
                arm_joint_ids = _as_np(self.env.arm_joint_ids).astype(int)
                self.align_panel_arm_joint_target = _as_np(
                    state["dof_pos"]
                )[0, arm_joint_ids].copy()
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
                and panel_yaw_error < BASE_YAW_TOLERANCE_RAD
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
                self._begin_panel_contact()
        elif self.stage == Stage.CONTACT_PANEL:
            ee_speed = float(np.linalg.norm(_as_np(state["ee_vel"])[0, :3]))
            finger_force = float(np.max(finger_contact_norms)) if finger_contact_norms.size else 0.0
            current_hinge = float(_as_np(state["door_joint_position"])[0, self.hinge_idx])
            current_hinge_world = self.hinge_axis_sign * current_hinge
            current_normal = _quat_rotate_np(
                self._yaw_quat(current_hinge_world), self.door_normal
            )
            if self.panel_contact_target_world is not None:
                signed_clearance = float(
                    np.dot(ee - self.panel_contact_target_world, current_normal)
                )
                lateral_error = float(np.linalg.norm(
                    (ee - self.panel_contact_target_world)
                    - signed_clearance * current_normal
                ))
                insertion_axis = _quat_rotate_np(
                    _as_np(state["ee_orn"])[0], self.open_template["depth_axis"]
                )
                insertion_axis /= max(float(np.linalg.norm(insertion_axis)), 1.0e-8)
                panel_axis_error = math.acos(
                    float(np.clip(np.dot(insertion_axis, -current_normal), -1.0, 1.0))
                )
            else:
                signed_clearance = 999.0
                lateral_error = 999.0
                panel_axis_error = math.pi
            finger_clearance = self._panel_finger_clearance(state, hinge)
            standoff_sample = (
                0.0 <= finger_clearance <= self.panel_standoff_acceptance_clearance
                and lateral_error < 0.035
                and panel_axis_error <= self.grasp_orientation_tolerance
                and ee_speed < 0.06
            )
            self.last_acceptance_diagnostics.update(
                {
                    "panel_finger_clearance_m": float(finger_clearance),
                    "panel_lateral_error_m": float(lateral_error),
                    "panel_axis_error_rad": float(panel_axis_error),
                    "ee_linear_speed_m_s": float(ee_speed),
                    "panel_standoff_sample": bool(standoff_sample),
                }
            )
            if self.panel_contact_desired_ee_world is not None:
                ee_target_error = (
                    ee - self.panel_contact_desired_ee_world
                )
                target_normal_error = float(
                    np.dot(ee_target_error, current_normal)
                )
                self.last_acceptance_diagnostics.update(
                    {
                        "panel_ee_target_error_m": float(
                            np.linalg.norm(ee_target_error)
                        ),
                        "panel_ee_target_normal_error_m": target_normal_error,
                        "panel_ee_target_tangent_error_m": float(
                            np.linalg.norm(
                                ee_target_error
                                - target_normal_error * current_normal
                            )
                        ),
                    }
                )
            self.panel_contact_settle_steps = (
                self.panel_contact_settle_steps + 1 if standoff_sample else 0
            )
            panel_standoff_ready = (
                self.panel_contact_settle_steps >= 8
                and abs(gripper_pos) < 0.20
            )
            if panel_standoff_ready:
                self.push_start_hinge_angle = hinge
                self.push_last_progress_hinge = hinge
                self.push_no_progress_steps = 0
                self.push_pause_steps = 0
                self.push_contact_world = (
                    self.panel_contact_command_target_world.copy()
                )
                self.push_finger_clearance = self._panel_finger_clearance(state, hinge)
                self.push_finger_approach_speed = 0.0
                self.push_target_finger_clearance = float("nan")
                self.push_finger_force = finger_force
                self.push_contact_retreat_active = False
                self.push_arm_detached = False
                self.push_detached_ee_goal_local = None
                self.push_recovery_settle_steps = 0
                self.push_comfort_reach_radius = 0.7
                self.push_reengage_reachable = False
                self.push_previous_finger_clearance = (
                    self.push_finger_clearance
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
                self.push_speed_error_integral = 0.0
                self.push_overspeed_steps = 0
                self.push_base_start_xy = base_xy.copy()
                self.push_panel_contact_start_world = self.push_contact_world.copy()
                self.stage = Stage.PUSH_OPEN; self.stage_steps = 0
        elif self.stage in (
            Stage.PUSH_OPEN,
            Stage.PUSH_ARM_RETRACT,
            Stage.BASE_ADVANCE,
        ):
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
            else:
                near_goal = opening_angle >= math.radians(89.0)
                if near_goal:
                    self._begin_through(opening_angle, opening_speed)
                else:
                    if self.push_arm_detached:
                        self.push_no_progress_steps = 0
                    elif self.push_last_progress_hinge is None:
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
        elif self.stage == Stage.THROUGH:
            tail_clearance = self._through_tail_clearance(state)
            opening_angle = hinge
            base_up = _quat_rotate_np(
                base[3:7], np.array([0.0, 0.0, 1.0], dtype=np.float32)
            )
            upright = float(base_up[2])
            if station_dist < self.through_best_station_error - 0.01:
                self.through_best_station_error = station_dist
                self.through_no_progress_steps = 0
            elif station_dist >= self.station_tolerance:
                self.through_no_progress_steps += 1
            else:
                self.through_no_progress_steps = 0
            arm_joint_ids = _as_np(self.env.arm_joint_ids).astype(int)
            arm_position = _as_np(state["dof_pos"])[0, arm_joint_ids]
            arm_velocity = _as_np(state["dof_vel"])[0, arm_joint_ids]
            middle_joint_error = float(
                np.max(np.abs(arm_position - self.middle_arm_joint_target))
            )
            middle_joint_speed = float(np.max(np.abs(arm_velocity)))
            reset_ee_speed = float(
                np.linalg.norm(_as_np(state["ee_vel"])[0, :3])
            )
            reset_ee_angular_speed = float(
                np.linalg.norm(_as_np(state["ee_vel"])[0, 3:6])
            )
            gripper_joint_id = int(self.env.gripper_joint_ids[0].item())
            gripper_speed = abs(float(_as_np(state["dof_vel"])[0, gripper_joint_id]))
            door_joint_speed = float(
                np.max(np.abs(_as_np(state["door_joint_velocity"])[0]))
            )
            terminal_checks = {
                "station": station_dist < self.station_tolerance,
                "yaw": yaw_error < BASE_YAW_TOLERANCE_RAD,
                "base_linear_speed": speed < TERMINAL_BASE_LINEAR_SPEED_M_S,
                "base_angular_speed": float(np.linalg.norm(base[10:13]))
                < TERMINAL_BASE_ANGULAR_SPEED_RAD_S,
                "middle_joint_error": middle_joint_error < 0.05,
                "arm_joint_speed": middle_joint_speed
                < TERMINAL_ARM_JOINT_SPEED_RAD_S,
                "gripper_speed": gripper_speed < TERMINAL_GRIPPER_SPEED_RAD_S,
                "door_joint_speed": door_joint_speed < TERMINAL_DOOR_SPEED_RAD_S,
            }
            actual_terminal_stable = all(terminal_checks.values())
            for name, passed in terminal_checks.items():
                if not passed:
                    self.through_stability_failure_counts[name] = (
                        self.through_stability_failure_counts.get(name, 0) + 1
                    )
            self.through_settle_steps = (
                self.through_settle_steps + 1
                if actual_terminal_stable
                else 0
            )
            self.last_acceptance_diagnostics.update(
                {
                    "terminal_actual_stable": bool(actual_terminal_stable),
                    "terminal_stable_steps": int(self.through_settle_steps),
                    "terminal_failed_checks": [
                        name for name, passed in terminal_checks.items() if not passed
                    ],
                    "terminal_failure_counts": dict(
                        self.through_stability_failure_counts
                    ),
                    "middle_joint_error_rad": middle_joint_error,
                    "base_angular_speed_rad_s": float(
                        np.linalg.norm(base[10:13])
                    ),
                    "arm_joint_speed_max_rad_s": middle_joint_speed,
                    "ee_linear_speed_m_s": reset_ee_speed,
                    "ee_angular_speed_rad_s": reset_ee_angular_speed,
                    "gripper_speed_rad_s": gripper_speed,
                    "door_joint_speed_max_rad_s": door_joint_speed,
                }
            )
            if upright < 0.75 or base[2] < 0.30:
                self.failure_reason = (
                    f"through_robot_fell:up={upright:.3f},height={base[2]:.3f}"
                )
                self.stage = Stage.FAILURE
            elif opening_angle < math.radians(65.0) and tail_clearance < 0.8:
                self.failure_reason = (
                    f"through_door_swingback:hinge={hinge:.4f},"
                    f"tail_clearance={tail_clearance:.3f}"
                )
                self.stage = Stage.FAILURE
            elif (
                station_dist >= self.station_tolerance
                and self.through_no_progress_steps > 150
            ):
                self.failure_reason = (
                    f"through_no_progress:station_error={station_dist:.3f},"
                    f"tail_clearance={tail_clearance:.3f}"
                )
                self.stage = Stage.FAILURE
            elif (
                # The collector records the pre-step observation.  One extra
                # stable transition leaves 50 complete stable observations in
                # the half-open episode timeline.
                self.through_settle_steps > TERMINAL_STABILITY_FRAMES
            ):
                self.stage = Stage.SUCCESS; self.stage_steps = 0
        self.stage_steps += 1
        if (
            self.stage in self.stage_limits
            and self.stage_steps > self.stage_limits[self.stage]
        ):
            if self.stage == Stage.TURN_HANDLE:
                self.failure_reason = (
                    f"turn_timeout:handle_delta={abs(self.turn_last_handle_angle - self.turn_start_handle_angle):.4f},"
                    f"hinge_delta={abs(self.turn_last_hinge_angle - self.turn_start_hinge_angle):.4f}"
                )
            else:
                self.failure_reason = f"stage_timeout:{self.stage.name.lower()}"
            self.failure_stage = self.stage.name
            self.stage = Stage.FAILURE
        if self.stage != previous_stage:
            self.stage_exit_diagnostics[previous_stage.name] = dict(
                self.last_acceptance_diagnostics
            )
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
                f"through_tail_clearance={self._through_tail_clearance(state):.3f} "
                f"through_station_error={float(np.linalg.norm(base_xy - self.through_station_xy)):.3f} "
                f"through_settle={self.through_settle_steps} "
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
                f"[doorman][center] q={actual_q if 'actual_q' in locals() else float('nan'):.4f} "
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


def _write_batched_episode_result(
    args, controller, runner, state, env_id, seed, step_count, max_finger_force,
    max_abs_hinge_speed, initial_root_state, initial_dof_position,
):
    episode_dir = args.output_dir / f"seed_{seed}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    door_path = _resolved_door_prim_path(runner.env, env_id)
    door_prim = runner.env.sim.stage.GetPrimAtPath(door_path)
    door_randomization = dict(door_prim.GetMetadata("customData") or {})
    arm_joint_ids = _as_np(runner.env.arm_joint_ids).astype(int)
    metadata = {
        "schema_version": "doorman_boundary_v1",
        "seed": seed,
        "batch_env_id": env_id,
        "batch_seed": int(args.seed),
        "success": controller.stage == Stage.SUCCESS,
        "failure_reason": controller.failure_reason,
        "failure_stage": controller.failure_stage,
        "acceptance_diagnostics": controller.last_acceptance_diagnostics,
        "stage_exit_diagnostics": controller.stage_exit_diagnostics,
        "target_noise": controller.target_noise.metadata(),
        "camera": {"enabled": False, "names": [], "frame_count": {}},
        "scene": {
            "door_prim_path": door_path,
            "robot_side": controller.robot_side,
            "door_randomization": door_randomization,
        },
        "robot_state": {
            "initial_base_pose_w": initial_root_state.tolist(),
            "initial_arm_joint_position": initial_dof_position[arm_joint_ids].tolist(),
            "initial_gripper_position": float(
                initial_dof_position[int(runner.env.gripper_joint_ids[0].item())]
            ),
        },
        "result": {
            "final_stage": controller.stage.name,
            "steps": int(step_count),
            "max_finger_force_n": float(max_finger_force),
            "max_abs_hinge_speed_deg_s": math.degrees(max_abs_hinge_speed),
            "summary_only": True,
        },
    }
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _steady_path_request(
    state,
    start_position_world,
    end_position_world,
    end_orientation_world_wxyz,
    *,
    start_orientation_world_wxyz,
):
    return {
        "dof_names": list(state["dof_names"]),
        "dof_positions": _as_np(state["dof_pos"])[0].astype(float).tolist(),
        "root_state": _as_np(state["root_states"])[0].astype(float).tolist(),
        "start_position_world": np.asarray(
            start_position_world, dtype=np.float64
        ).tolist(),
        "end_position_world": np.asarray(
            end_position_world, dtype=np.float64
        ).tolist(),
        "start_orientation_world_wxyz": np.asarray(
            start_orientation_world_wxyz, dtype=np.float64
        ).tolist(),
        "end_orientation_world_wxyz": np.asarray(
            end_orientation_world_wxyz, dtype=np.float64
        ).tolist(),
    }


def _steady_inverse_worker_request(request):
    state = request.state
    return {
        "dof_names": list(state["dof_names"]),
        "dof_positions": _as_np(state["dof_pos"])[0].astype(float).tolist(),
        "root_state": _as_np(state["root_states"])[0].astype(float).tolist(),
        "desired_position_world": request.desired_position_world.astype(float).tolist(),
        "desired_orientation_world_wxyz": (
            request.desired_orientation_world.astype(float).tolist()
        ),
        "position_tolerance_m": request.position_tolerance_m,
    }


def _release_path_request(
    state,
    start_position_world,
    end_position_world,
    start_orientation_world_wxyz,
    end_orientation_world_wxyz,
    start_gripper_position,
    end_gripper_position,
    steps,
    position_tolerance_m,
    gripper_joint_id,
):
    dof_names = list(state["dof_names"])
    return {
        "path_kind": "release",
        "dof_names": dof_names,
        "dof_positions": _as_np(state["dof_pos"])[0].astype(float).tolist(),
        "root_state": _as_np(state["root_states"])[0].astype(float).tolist(),
        "start_position_world": np.asarray(
            start_position_world, dtype=np.float64
        ).tolist(),
        "end_position_world": np.asarray(
            end_position_world, dtype=np.float64
        ).tolist(),
        "start_orientation_world_wxyz": np.asarray(
            start_orientation_world_wxyz, dtype=np.float64
        ).tolist(),
        "end_orientation_world_wxyz": np.asarray(
            end_orientation_world_wxyz, dtype=np.float64
        ).tolist(),
        "start_gripper_position": float(start_gripper_position),
        "end_gripper_position": float(end_gripper_position),
        "steps": int(steps),
        "position_tolerance_m": float(position_tolerance_m),
        "gripper_joint_name": dof_names[int(gripper_joint_id)],
    }


def _run_batched(args):
    num_envs = int(args.batched_envs)
    if num_envs < 1:
        raise ValueError("--batched_envs must be positive")
    args.num_envs = num_envs
    if args.env_spacing is None:
        args.env_spacing = 8.0
    base_seed = int(args.seed if args.seed is not None else 0)
    if args.batch_active_env_ids is None:
        active_env_ids = tuple(range(num_envs))
    else:
        active_env_ids = tuple(
            sorted(
                {
                    int(value.strip())
                    for value in args.batch_active_env_ids.split(",")
                    if value.strip()
                }
            )
        )
        if not active_env_ids:
            raise ValueError("--batch_active_env_ids must select at least one env")
        if active_env_ids[0] < 0 or active_env_ids[-1] >= num_envs:
            raise ValueError(
                "--batch_active_env_ids contains an id outside --batched_envs"
            )
    active_env_mask = np.zeros(num_envs, dtype=bool)
    active_env_mask[list(active_env_ids)] = True
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    runner = ManipLoco_Policy(args)
    env = runner.env
    door_handle_paths = [
        str(prim.GetPath())
        for prim in env.sim.stage.Traverse()
        if prim.GetName() == "door_handle"
        and str(prim.GetPath()).startswith("/World/envs/")
    ]
    print(
        f"[doorman][batch-init] usd_door_handle_count={len(door_handle_paths)} "
        f"paths={door_handle_paths if num_envs <= 32 else 'omitted'}",
        flush=True,
    )
    # Generate one current PhysX contact frame from the collision-free reset
    # pose before policy inference. ContactSensor.reset() marks its buffers
    # outdated, so reading immediately would reload the previous solve.
    env.reset()
    neutral_actions = torch.zeros(
        (num_envs, env.num_actions), dtype=torch.float32, device=env.device
    )
    runner.obs, *_ = env.step(neutral_actions)
    device_state = runner.get_control_state()
    full_state = _control_state_to_host(device_state, num_envs)
    state_rows = _slice_all_control_states(full_state, num_envs)
    shared_steady_model = B2Z1SteadyEeForwardModel(
        LOWLEVEL_ROOT
        / "resources/robots/b2z1/urdf/generated"
        / "b2z1_mount_90_x0p22_y0_z0p17.urdf",
        stiffness=_arm_drive_stiffness(env),
        emit_tolerance_warnings=not args.summary_only,
    )
    steady_tolerance_stats = {"count": 0}
    steady_path_executor = None
    if args.steady_path_workers > 0:
        steady_path_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=int(args.steady_path_workers),
            mp_context=mp.get_context("spawn"),
            initializer=initialize_steady_path_worker,
            initargs=(
                str(shared_steady_model.urdf_path),
                shared_steady_model.stiffness,
            ),
        )
    if (
        (
            args.batch_contact_ik_backend == "exact_workers"
            or args.batch_path_backend == "exact_workers"
        )
        and steady_path_executor is None
    ):
        raise ValueError(
            "exact worker backends require "
            "--steady_path_workers greater than zero"
        )
    views = []
    controllers = []
    reference_buffers = {}
    _update_batched_reference_buffers(env, reference_buffers)
    command_buffers = _make_batched_command_buffers(env)
    arm_mode_buffer = (
        env.teleop_arm_joint_control_mask.detach().cpu().numpy().copy()
    )
    shared_robot_geometry = None
    controller_stdout = open(os.devnull, "w") if args.summary_only else None
    with (
        contextlib.redirect_stdout(controller_stdout)
        if controller_stdout is not None
        else contextlib.nullcontext()
    ):
        for env_id in range(num_envs):
            sliced_state = state_rows[env_id]
            view = _BatchedRunnerView(
                runner,
                env_id,
                sliced_state,
                reference_buffers=reference_buffers,
                arm_mode_buffer=arm_mode_buffer,
                command_buffers=command_buffers,
            )
            controller = DoorController(
                view,
                robot_side=args.robot_side,
                position_only_center=args.position_only_center,
                disable_push_base_assist=args.disable_push_base_assist,
                env_id=env_id,
                shared_steady_model=shared_steady_model,
                shared_robot_geometry=shared_robot_geometry,
                target_noise_seed=base_seed + env_id,
                target_noise_config=args.target_noise_config,
            )
            controller.defer_batched_orientation_delta = True
            controller.reset(
                state=sliced_state,
                reset_env=False,
                close_gripper=False,
            )
            views.append(view)
            controllers.append(controller)
            if shared_robot_geometry is None:
                shared_robot_geometry = controller.center_geometry
    geometry_cache = _make_batched_geometry_cache(controllers, env.device)
    stability_tracker = _BatchedStabilityTracker(num_envs, env.device)

    # DoorController.reset() measures the open-jaw geometry and then closes
    # the jaw before a scalar episode starts.  Do that once for the complete
    # vectorized batch so every environment has the same initial semantics.
    gripper_joint_id = int(env.gripper_joint_ids[0].item())
    joint_pos = env.robot.data.joint_pos.clone()
    joint_vel = env.robot.data.joint_vel.clone()
    joint_pos[:, gripper_joint_id] = torch.as_tensor(
        [controller.q_closed for controller in controllers],
        device=env.device,
        dtype=joint_pos.dtype,
    )
    command_buffers["gripper"][:, 0] = np.asarray(
        [controller.q_closed for controller in controllers], dtype=np.float32
    )
    joint_vel[:, gripper_joint_id] = 0.0
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    env.robot.set_joint_position_target(joint_pos)
    runner.obs = env.get_observations()
    device_state = runner.get_control_state()
    full_state = _control_state_to_host(device_state, num_envs)
    handle_center_gpu_s = 0.0
    phase_start = time.perf_counter()
    handle_centers = _batched_handle_centers_gpu(full_state, geometry_cache)
    handle_center_gpu_s += time.perf_counter() - phase_start
    initial_root_states = _as_np(full_state["root_states"])[:, :7].copy()
    initial_dof_positions = _as_np(full_state["dof_pos"]).copy()
    initial_gripper_positions = _as_np(full_state["gripper_position"]).reshape(-1).copy()
    standard_collector = None
    standard_door_randomization = None
    standard_door_asset_snapshot = None
    video_frames = {}
    video_timestamps = {}
    if args.record_video and num_envs != 1:
        raise ValueError("video regression uses the unified runner with batch size one")
    if args.standard_dataset_dir is not None:
        if num_envs != 1 or active_env_ids != (0,):
            raise ValueError(
                "standard collection uses the unified runner with exactly one active environment"
            )
        from gr00t.rl.data_collection.auto_open_collector import (
            AutoOpenStandardEpisodeCollector,
        )

        standard_collector = AutoOpenStandardEpisodeCollector(
            runner,
            controllers[0],
            episode_index=0,
            seed=base_seed,
            video_staging_root=(
                Path(args.standard_dataset_dir) / ".staging" / "episode_000000"
            ),
        )
        door_prim = env.sim.stage.GetPrimAtPath(
            _resolved_door_prim_path(env, 0)
        )
        standard_door_randomization = dict(
            door_prim.GetMetadata("customData") or {}
        )
        standard_door_asset_snapshot = _door_asset_stage_snapshot(
            env.sim.stage, door_prim
        )
    print(
        "[doorman][batch-init] door_joint_position="
        f"{_as_np(full_state['door_joint_position']).round(5).tolist()} "
        "gripper_position="
        f"{_as_np(full_state['gripper_position']).round(5).reshape(-1).tolist()}",
        flush=True,
    )

    step_counts = np.zeros(num_envs, dtype=np.int32)
    max_finger_forces = np.zeros(num_envs, dtype=np.float32)
    max_abs_hinge_speeds = np.zeros(num_envs, dtype=np.float32)
    pending_metrics = np.zeros((num_envs, 2), dtype=np.float32)
    batch_timing = {
        "command_s": 0.0,
        "teleop_prepare_s": 0.0,
        "policy_physics_s": 0.0,
        "state_to_host_s": 0.0,
        "advance_s": 0.0,
        "steady_path_workers_s": 0.0,
        "geometry_gpu_s": 0.0,
        "geometry_numpy_backend_s": 0.0,
        "stability_gpu_s": 0.0,
        "orientation_gpu_s": 0.0,
        "contact_ik_backend_s": 0.0,
        "path_backend_s": 0.0,
        "panel_clearance_backend_s": 0.0,
    }
    contact_ik_backend_stats = {
        "batches": 0,
        "requests": 0,
        "worker_errors": 0,
    }
    path_backend_stats = {
        "batches": 0,
        "requests": 0,
        "worker_errors": 0,
        "labels": {},
    }
    geometry_numpy_backend_stats = {
        "batches": 0,
        "requests": 0,
        "accepted_requests": 0,
        "exact_scalar_requests": 0,
        "max_batch": 0,
    }
    orientation_gpu_calls = 0
    arm_induced_pitch = np.asarray(
        [controller.env.arm_induced_pitch for controller in controllers],
        dtype=np.float64,
    )
    approach_nominal_targets = [
        controller._approach_base_target() for controller in controllers
    ]
    approach_command_targets = [
        controller._perturb_base_target("approach", *target)
        for controller, target in zip(
            controllers, approach_nominal_targets, strict=True
        )
    ]
    station_xy = np.stack([target[0] for target in approach_command_targets])
    station_yaw = np.asarray(
        [target[1] for target in approach_command_targets], dtype=np.float64
    )
    nominal_station_xy = np.stack(
        [target[0] for target in approach_nominal_targets]
    )
    nominal_station_yaw = np.asarray(
        [target[1] for target in approach_nominal_targets], dtype=np.float64
    )
    base_command_exact_s = 0.0
    pending_high_actions = [None] * num_envs
    for step in range(args.max_steps):
        pending_high_actions = [None] * num_envs
        phase_start = time.perf_counter()
        batched_base_command = _base_commands_exact(
            full_state,
            station_xy,
            station_yaw,
            nominal_station_xy,
            nominal_station_yaw,
        )
        base_command_exact_s += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        active = False
        deferred_contact_requests = []
        if args.batch_panel_clearance_backend == "gpu":
            for controller in controllers:
                controller.batched_panel_finger_clearance = None
            panel_clearance_env_ids = [
                env_id
                for env_id, controller in enumerate(controllers)
                if active_env_mask[env_id]
                and controller.stage
                in (
                    Stage.CONTACT_PANEL,
                    Stage.PUSH_OPEN,
                    Stage.PUSH_ARM_RETRACT,
                    Stage.BASE_ADVANCE,
                )
            ]
            if panel_clearance_env_ids:
                panel_start = time.perf_counter()
                panel_values = (
                    _batched_panel_finger_clearance_gpu(
                        device_state, geometry_cache, panel_clearance_env_ids
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                batch_timing["panel_clearance_backend_s"] += (
                    time.perf_counter() - panel_start
                )
                for env_id, value in zip(
                    panel_clearance_env_ids, panel_values, strict=True
                ):
                    controllers[env_id].batched_panel_finger_clearance = float(value)
        for env_id, (view, controller) in enumerate(zip(views, controllers)):
            sliced_state = state_rows[env_id]
            view.set_state(sliced_state)
            controller.batched_handle_center = handle_centers[env_id]
            controller.batched_default_base_command = tuple(
                value[env_id] for value in batched_base_command
            )
            if not active_env_mask[env_id]:
                command_buffers["base"][env_id] = 0.0
                continue
            if controller.stage in (Stage.SUCCESS, Stage.FAILURE):
                command_buffers["base"][env_id] = 0.0
                continue
            active = True
            step_counts[env_id] = step + 1
            finger_force = sliced_state.get("finger_contact_forces_w")
            if finger_force is not None:
                force = float(np.linalg.norm(_as_np(finger_force)[0], axis=-1).max())
                max_finger_forces[env_id] = max(max_finger_forces[env_id], force)
                if force > 600.0:
                    controller.failure_reason = f"safety_force_limit:{force:.1f}N"
                    controller.failure_stage = controller.stage.name
                    controller.stage = Stage.FAILURE
                    command_buffers["base"][env_id] = 0.0
                    continue
            try:
                with (
                    contextlib.redirect_stdout(controller_stdout)
                    if controller_stdout is not None
                    else contextlib.nullcontext()
                ):
                    controller.defer_steady_inverse = (
                        args.batch_contact_ik_backend == "exact_workers"
                    )
                    if (
                        controller.stage == Stage.CONTACT_PANEL
                        and controller.defer_steady_inverse
                    ):
                        controller._prepare_panel_contact_command(sliced_state)
                    high, dist, yaw_error = controller.command(sliced_state)
                    pending_high_actions[env_id] = high
            except DeferredSteadyInverse as request:
                deferred_contact_requests.append((env_id, request))
                continue
            except RuntimeError as exc:
                controller.failure_reason = f"controller_runtime_error:{exc}"
                controller.failure_stage = controller.stage.name
                controller.stage = Stage.FAILURE
                command_buffers["base"][env_id] = 0.0
                continue
            finally:
                controller.defer_steady_inverse = False
            pending_metrics[env_id] = (dist, yaw_error)


        if deferred_contact_requests:
            requests = [item[1] for item in deferred_contact_requests]
            backend_start = time.perf_counter()
            worker_results = list(
                steady_path_executor.map(
                    solve_steady_inverse_request,
                    [_steady_inverse_worker_request(request) for request in requests],
                )
            )
            batch_timing["contact_ik_backend_s"] += (
                time.perf_counter() - backend_start
            )
            contact_ik_backend_stats["batches"] += 1
            contact_ik_backend_stats["requests"] += len(requests)

            for (env_id, request), worker_result in zip(
                deferred_contact_requests, worker_results, strict=True
            ):
                controller = request.controller
                _merge_tolerance_warning_stats(
                    steady_tolerance_stats,
                    worker_result["warning_stats"],
                )
                if worker_result["error"] is not None:
                    contact_ik_backend_stats["worker_errors"] += 1
                    controller.failure_reason = (
                        "controller_runtime_error:" + worker_result["error"]
                    )
                    controller.failure_stage = controller.stage.name
                    controller.stage = Stage.FAILURE
                    command_buffers["base"][env_id] = 0.0
                    continue
                controller.panel_contact_inverse_command_world = np.asarray(
                    worker_result["command_position"], dtype=np.float32
                ).copy()
                controller.panel_contact_inverse_command_orientation_world = (
                    np.asarray(
                        worker_result["command_orientation"], dtype=np.float32
                    ).copy()
                )
                controller.panel_contact_inverse_target_world = (
                    request.desired_position_world.astype(np.float32)
                )
                controller.panel_contact_inverse_orientation_world = (
                    request.desired_orientation_world.astype(np.float32)
                )
                controller.prepared_panel_contact_command = (
                    controller.panel_contact_inverse_command_world.copy(),
                    controller.panel_contact_inverse_command_orientation_world.copy(),
                )
                sliced_state = state_rows[env_id]
                try:
                    with (
                        contextlib.redirect_stdout(controller_stdout)
                        if controller_stdout is not None
                        else contextlib.nullcontext()
                    ):
                        high, dist, yaw_error = controller.command(
                            sliced_state
                        )
                        pending_high_actions[env_id] = high
                except RuntimeError as exc:
                    controller.failure_reason = f"controller_runtime_error:{exc}"
                    controller.failure_stage = controller.stage.name
                    controller.stage = Stage.FAILURE
                    command_buffers["base"][env_id] = 0.0
                    continue
                pending_metrics[env_id] = (dist, yaw_error)
        batch_timing["command_s"] += time.perf_counter() - phase_start
        if not active:
            break

        orientation_env_ids = [
            env_id
            for env_id, controller in enumerate(controllers)
            if active_env_mask[env_id]
            and controller.stage not in (Stage.SUCCESS, Stage.FAILURE)
        ]
        phase_start = time.perf_counter()
        orientation_delta = _batched_grasp_orientation_delta_gpu(
            command_buffers,
            reference_buffers,
            arm_induced_pitch,
            orientation_env_ids,
            env.device,
        )
        batch_timing["orientation_gpu_s"] += time.perf_counter() - phase_start
        if orientation_env_ids:
            orientation_gpu_calls += len(orientation_env_ids)
            command_buffers["ee_rpy"][orientation_env_ids] = orientation_delta

        phase_start = time.perf_counter()
        _flush_batched_command_buffers(env, command_buffers)
        env._clip_teleop_targets()
        env._update_effective_teleop_inputs(force=True)
        batch_timing["teleop_prepare_s"] += time.perf_counter() - phase_start
        if standard_collector is not None and pending_high_actions[0] is None:
            if controllers[0].stage in (Stage.SUCCESS, Stage.FAILURE):
                break
            raise RuntimeError(
                "unified standard collector has no command for an active frame"
            )
        if standard_collector is not None:
            high = pending_high_actions[0]
            standard_collector.capture_before_step(
                state_rows[0], high, stage_name=controllers[0].stage.name
            )
        phase_start = time.perf_counter()
        result = runner.step_once()
        if standard_collector is not None:
            standard_collector.complete_after_step(result)
        elif args.record_video:
            manager = runner.rgb_camera_manager
            if manager is None or not manager.enabled:
                raise RuntimeError("video regression requires enabled RGB cameras")
            images_by_name = manager.render_images(env_id=0, update=False)
            timestamp = step * float(env.dt)
            for name, images in images_by_name.items():
                if len(images) != 1:
                    raise RuntimeError(
                        f"camera {name} returned {len(images)} frames for one environment"
                    )
                video_frames.setdefault(name, []).append(images[0])
                video_timestamps.setdefault(name, []).append(timestamp)
        batch_timing["policy_physics_s"] += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        handle_centers = _batched_handle_centers_gpu(result["state"], geometry_cache)
        handle_center_gpu_s += time.perf_counter() - phase_start
        geometry_env_ids = [
            env_id
            for env_id, controller in enumerate(controllers)
            if controller.stage in (Stage.CENTER_HANDLE, Stage.GRASP_HANDLE)
        ]
        phase_start = time.perf_counter()
        geometry_metrics = _batched_geometry_metrics_gpu(
            result["state"], geometry_cache, geometry_env_ids
        )
        batch_timing["geometry_gpu_s"] += time.perf_counter() - phase_start
        geometry_rows = {
            env_id: row for row, env_id in enumerate(geometry_env_ids)
        }
        stability_env_ids = [
            env_id
            for env_id, controller in enumerate(controllers)
            if controller.stage
            in (
                Stage.PREGRASP,
                Stage.OPEN_GRIPPER,
                Stage.CENTER_HANDLE,
                Stage.RESTORE_GRIPPER,
            )
        ]
        phase_start = time.perf_counter()
        stability_results = stability_tracker.update(
            result["state"], controllers, stability_env_ids
        )
        batch_timing["stability_gpu_s"] += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        device_state = result["state"]
        full_state = _control_state_to_host(device_state, num_envs)
        state_rows = _slice_all_control_states(full_state, num_envs)
        _update_batched_reference_buffers(env, reference_buffers)
        batch_timing["state_to_host_s"] += time.perf_counter() - phase_start

        if args.batch_panel_clearance_backend == "gpu":
            for controller in controllers:
                controller.batched_panel_finger_clearance = None
            panel_clearance_env_ids = [
                env_id
                for env_id, controller in enumerate(controllers)
                if active_env_mask[env_id]
                and controller.stage
                in (
                    Stage.CONTACT_PANEL,
                    Stage.PUSH_OPEN,
                    Stage.PUSH_ARM_RETRACT,
                    Stage.BASE_ADVANCE,
                )
            ]
            if panel_clearance_env_ids:
                panel_start = time.perf_counter()
                panel_values = (
                    _batched_panel_finger_clearance_gpu(
                        device_state, geometry_cache, panel_clearance_env_ids
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                batch_timing["panel_clearance_backend_s"] += (
                    time.perf_counter() - panel_start
                )
                for env_id, value in zip(
                    panel_clearance_env_ids, panel_values, strict=True
                ):
                    controllers[env_id].batched_panel_finger_clearance = float(value)

        geometry_numpy_metrics = None
        geometry_numpy_rows = {}
        if args.batch_geometry_fallback_backend == "numpy_batch":
            fallback_env_ids = []
            for env_id in geometry_env_ids:
                row = geometry_rows[env_id]
                candidate_metrics = {
                    name: value[row] for name, value in geometry_metrics.items()
                }
                candidate_metrics["valid"] = True
                if controllers[env_id]._gpu_geometry_needs_scalar(
                    candidate_metrics
                ):
                    fallback_env_ids.append(env_id)
            if fallback_env_ids:
                numpy_start = time.perf_counter()
                geometry_numpy_metrics = _batched_geometry_metrics_numpy(
                    full_state,
                    geometry_cache,
                    fallback_env_ids,
                    handle_centers[fallback_env_ids],
                )
                batch_timing["geometry_numpy_backend_s"] += (
                    time.perf_counter() - numpy_start
                )
                candidate_rows = {
                    env_id: row for row, env_id in enumerate(fallback_env_ids)
                }
                for env_id, row in candidate_rows.items():
                    numpy_metrics = {
                        name: value[row]
                        for name, value in geometry_numpy_metrics.items()
                    }
                    if _numpy_geometry_requires_exact_scalar(
                        controllers[env_id], numpy_metrics
                    ):
                        geometry_numpy_backend_stats["exact_scalar_requests"] += 1
                    else:
                        geometry_numpy_rows[env_id] = row
                geometry_numpy_backend_stats["batches"] += 1
                geometry_numpy_backend_stats["requests"] += len(fallback_env_ids)
                geometry_numpy_backend_stats["accepted_requests"] += len(
                    geometry_numpy_rows
                )
                geometry_numpy_backend_stats["max_batch"] = max(
                    geometry_numpy_backend_stats["max_batch"],
                    len(fallback_env_ids),
                )

        if steady_path_executor is not None:
            phase_start = time.perf_counter()
            planning = []
            for env_id, controller in enumerate(controllers):
                if not active_env_mask[env_id] or controller.stage != Stage.APPROACH:
                    continue
                sliced_state = state_rows[env_id]
                root_state = _as_np(sliced_state["root_states"])[0]
                dist, yaw_error = pending_metrics[env_id]
                base_stable = bool(
                    dist < controller.station_tolerance
                    and yaw_error < BASE_YAW_TOLERANCE_RAD
                    and float(np.linalg.norm(root_state[7:10])) < 0.02
                    and float(np.linalg.norm(root_state[10:13])) < 0.05
                )
                if not base_stable or controller.approach_settle_steps + 1 < 25:
                    continue
                request = _steady_path_request(
                    sliced_state,
                    _as_np(sliced_state["ee_pos"])[0],
                    controller.center_standoff_ee_world,
                    controller.grasp_orientation_quat,
                    start_orientation_world_wxyz=_as_np(sliced_state["ee_orn"])[0],
                )
                request_candidates = []
                for candidate in controller._pregrasp_pitch_candidates():
                    command_position, command_orientation = (
                        controller._perturb_ee_target(
                            "pregrasp",
                            candidate["end_position"],
                            candidate["orientation"],
                        )
                    )
                    request_candidates.append(
                        {
                            "pitch_deg": candidate["pitch_deg"],
                            "orientation": command_orientation.astype(float).tolist(),
                            "end_position": command_position.astype(float).tolist(),
                            "nominal_orientation": candidate["orientation"].astype(float).tolist(),
                            "nominal_end_position": candidate["end_position"].astype(float).tolist(),
                        }
                    )
                request["pregrasp_candidates"] = request_candidates
                request["path_kind"] = "steady"
                planning.append((controller, request))
            if planning:
                solved = steady_path_executor.map(
                    solve_deferred_path_request,
                    [request for _, request in planning],
                )
                for (controller, _), solved_path in zip(
                    planning, solved, strict=True
                ):
                    _merge_tolerance_warning_stats(
                        steady_tolerance_stats, solved_path["warning_stats"]
                    )
                    if solved_path["error"] is not None:
                        path_backend_stats["worker_errors"] += 1
                        controller.failure_reason = (
                            "controller_runtime_error:" + solved_path["error"]
                        )
                        controller.failure_stage = controller.stage.name
                        controller.stage = Stage.FAILURE
                        continue
                    selection = solved_path.get("pregrasp_selection")
                    if selection is None:
                        raise RuntimeError("PREGRASP worker omitted pitch selection")
                    controller._apply_pregrasp_candidate(
                        {
                            "pitch_deg": selection["pitch_deg"],
                            "orientation": selection["nominal_orientation"],
                            "end_position": selection["nominal_end_position"],
                        }
                    )
                    controller.arm_align_desired_target_world = np.asarray(
                        selection["end_position"], dtype=np.float32
                    ).copy()
                    print(
                        f"[doorman][pregrasp-pitch] "
                        f"selected={controller.pregrasp_pitch_deg:.0f}deg "
                        f"exact={selection['exact']} "
                        f"candidates={selection['diagnostics']}",
                        flush=True,
                    )
                    controller.precomputed_steady_paths["arm_align_path"] = (
                        solved_path["positions"],
                        solved_path["orientations"],
                    )
            batch_timing["steady_path_workers_s"] += (
                time.perf_counter() - phase_start
            )

        phase_start = time.perf_counter()
        deferred_path_requests = []
        for env_id, (view, controller) in enumerate(zip(views, controllers)):
            if not active_env_mask[env_id]:
                continue
            if controller.stage in (Stage.SUCCESS, Stage.FAILURE):
                continue
            sliced_state = state_rows[env_id]
            view.set_state(sliced_state)
            controller.batched_handle_center = handle_centers[env_id]
            hinge_velocity = _as_np(sliced_state.get("door_joint_velocity"))
            if hinge_velocity is not None:
                max_abs_hinge_speeds[env_id] = max(
                    max_abs_hinge_speeds[env_id],
                    abs(float(hinge_velocity[0, controller.hinge_idx])),
                )
            controller.replan_counter = (
                controller.replan_counter + 1
            ) % controller.replan_interval
            dist, yaw_error = pending_metrics[env_id]
            try:
                precomputed_geometry_metrics = None
                precomputed_geometry_authoritative = False
                if env_id in geometry_rows:
                    if env_id in geometry_numpy_rows:
                        row = geometry_numpy_rows[env_id]
                        precomputed_geometry_metrics = {
                            name: value[row]
                            for name, value in geometry_numpy_metrics.items()
                        }
                        precomputed_geometry_authoritative = True
                    else:
                        row = geometry_rows[env_id]
                        precomputed_geometry_metrics = {
                            name: value[row]
                            for name, value in geometry_metrics.items()
                        }
                    precomputed_geometry_metrics["valid"] = True
                controller.batched_stability_result = stability_results.get(env_id)
                controller.defer_path_solve = (
                    args.batch_path_backend == "exact_workers"
                )
                with (
                    contextlib.redirect_stdout(controller_stdout)
                    if controller_stdout is not None
                    else contextlib.nullcontext()
                ):
                    controller.advance(
                        sliced_state,
                        dist,
                        yaw_error,
                        precomputed_geometry_metrics=precomputed_geometry_metrics,
                        precomputed_geometry_authoritative=(
                            precomputed_geometry_authoritative
                        ),
                    )
            except DeferredPathSolve as request:
                deferred_path_requests.append((env_id, request))
            except RuntimeError as exc:
                controller.failure_reason = f"controller_runtime_error:{exc}"
                controller.failure_stage = controller.stage.name
                controller.stage = Stage.FAILURE
            finally:
                controller.defer_path_solve = False

        if deferred_path_requests:
            path_start = time.perf_counter()
            path_results = list(
                steady_path_executor.map(
                    solve_deferred_path_request,
                    [request.request for _, request in deferred_path_requests],
                )
            )
            batch_timing["path_backend_s"] += time.perf_counter() - path_start
            path_backend_stats["batches"] += 1
            path_backend_stats["requests"] += len(deferred_path_requests)
            for (env_id, request), result in zip(
                deferred_path_requests, path_results, strict=True
            ):
                labels = path_backend_stats["labels"]
                labels[request.label] = labels.get(request.label, 0) + 1
                _merge_tolerance_warning_stats(
                    steady_tolerance_stats, result["warning_stats"]
                )
                controller = request.controller
                if result["error"] is not None:
                    path_backend_stats["worker_errors"] += 1
                    controller.failure_reason = (
                        "controller_runtime_error:" + result["error"]
                    )
                    controller.failure_stage = controller.stage.name
                    controller.stage = Stage.FAILURE
                    continue
                controller._complete_deferred_path(
                    request.label,
                    result["positions"],
                    result["orientations"],
                )
        batch_timing["advance_s"] += time.perf_counter() - phase_start

    print(
        "[doorman][batch-timing] "
        + " ".join(f"{name}={value:.3f}" for name, value in batch_timing.items()),
        flush=True,
    )
    print(
        "[doorman][geometry-gpu] "
        f"uses={sum(c.gpu_geometry_uses for c in controllers)} "
        f"fallbacks={sum(c.gpu_geometry_fallbacks for c in controllers)}",
        flush=True,
    )
    print(
        "[doorman][stability-gpu] "
        f"uses={sum(c.gpu_stability_uses for c in controllers)} "
        f"fallbacks={sum(c.gpu_stability_fallbacks for c in controllers)}",
        flush=True,
    )
    print(
        f"[doorman][handle-center-gpu] wall_s={handle_center_gpu_s:.3f}",
        flush=True,
    )
    print(
        f"[doorman][orientation-gpu] calls={orientation_gpu_calls}",
        flush=True,
    )
    print(
        f"[doorman][base-command-exact] wall_s={base_command_exact_s:.3f}",
        flush=True,
    )
    if steady_path_executor is not None:
        steady_path_executor.shutdown(wait=True)

    successes = 0
    for env_id in active_env_ids:
        controller = controllers[env_id]
        if controller.stage not in (Stage.SUCCESS, Stage.FAILURE):
            controller.failure_reason = f"max_steps:{args.max_steps}"
            controller.stage = Stage.FAILURE
        successes += int(controller.stage == Stage.SUCCESS)
        sliced_state = state_rows[env_id]
        _write_batched_episode_result(
            args,
            controller,
            runner,
            sliced_state,
            env_id,
            base_seed + env_id,
            step_counts[env_id],
            max_finger_forces[env_id],
            max_abs_hinge_speeds[env_id],
            initial_root_states[env_id],
            initial_dof_positions[env_id],
        )
    if args.record_video:
        episode_dir = args.output_dir / f"seed_{base_seed}"
        video_dir = episode_dir / "video"
        video_dir.mkdir(exist_ok=True)
        for name, frames in video_frames.items():
            _save_rgb_video(
                frames,
                video_timestamps[name],
                video_dir / f"{name}.mp4",
            )
        _save_rgb_mosaic_video(
            video_frames,
            video_timestamps,
            video_dir / "all_cameras_mosaic.mp4",
        )
    if standard_collector is not None:
        controller = controllers[0]
        success = controller.stage == Stage.SUCCESS
        reasons = () if success else (
            f"autoopen_failure:{controller.failure_stage or controller.stage.name}:"
            f"{controller.failure_reason or 'unknown'}",
        )
        standard_episode = standard_collector.finish(
            acceptance_reasons=reasons,
            provenance={
                "seed": base_seed,
                "initial_root_pose_w": initial_root_states[0].tolist(),
                "initial_dof_position": initial_dof_positions[0].tolist(),
                "initial_gripper_position": float(initial_gripper_positions[0]),
                "door_randomization": standard_door_randomization,
                "door_asset": standard_door_asset_snapshot,
                "camera_mount_randomization": dict(
                    args.camera_mount_randomization or {}
                ),
                "contact_audit": standard_collector.contact_view_description,
            },
        )
        standard_collector.close()
        from gr00t.rl.data_collection.auto_open_writer import AutoOpenDatasetWriter

        camera_calibration = json.loads(
            Path(args.rgb_camera_config).read_text(encoding="utf-8")
        )
        standard_writer = AutoOpenDatasetWriter(
            args.standard_dataset_dir,
            camera_calibration=camera_calibration,
            simulation_provenance=_standard_simulation_provenance(
                args, runner, base_seed
            ),
        )
        standard_writer.write_episode(standard_episode)
        standard_writer.finalize()
    _merge_tolerance_warning_stats(
        steady_tolerance_stats,
        shared_steady_model.consume_tolerance_warning_stats(),
    )
    summary = {
        "batch_seed": base_seed,
        "num_envs": num_envs,
        "active_env_ids": list(active_env_ids),
        "successes": successes,
        "success_rate": successes / len(active_env_ids),
        "steady_path_workers": int(args.steady_path_workers),
        "steady_inverse_tolerance": steady_tolerance_stats,
        "batch_timing": batch_timing,
        "panel_clearance_backend": args.batch_panel_clearance_backend,
        "geometry_fallback_backend": args.batch_geometry_fallback_backend,
        "geometry_numpy_backend": (
            geometry_numpy_backend_stats
            if args.batch_geometry_fallback_backend == "numpy_batch"
            else None
        ),
        "contact_ik_backend": (
            contact_ik_backend_stats
            if args.batch_contact_ik_backend == "exact_workers"
            else None
        ),
        "path_backend": (
            path_backend_stats
            if args.batch_path_backend == "exact_workers"
            else None
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"[doorman][batch] complete: {successes}/{len(active_env_ids)} active "
        f"({num_envs} initialized)",
        flush=True,
    )
    print(
        "[doorman][steady-inverse-tolerance] "
        + " ".join(f"{key}={value}" for key, value in steady_tolerance_stats.items()),
        flush=True,
    )
    if args.headless:
        # Isaac Sim 5.1 spends tens of seconds unloading a large vectorized
        # stage after all results are durable.  The process owns this App, so
        # deterministic process teardown is the correct headless lifecycle.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    runner.close()
    return successes


def main():
    args = _parser().parse_args()
    try:
        args.target_noise_config = SemanticTargetNoiseConfig.from_assignments(
            args.target_noise
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.max_steps < 1:
        raise SystemExit("--max_steps must be positive")
    _default_paths(args)
    for label, path in (("checkpoint", args.ckpt_path), ("run metadata", args.run_metadata_path), ("robot USD", args.robot_usd_path)):
        if path and not Path(path).is_file():
            raise SystemExit(f"{label} not found: {path}")
    if args.rgb_camera_config and not Path(args.rgb_camera_config).is_file():
        raise SystemExit(f"RGB camera config not found: {args.rgb_camera_config}")
    if args.batched_envs < 1:
        raise SystemExit(
            "AutoOpen has one vectorized execution path; use the fixed launcher "
            "which supplies --batched_envs (one for a scalar shard)."
        )
    if args.standard_dataset_dir is not None and args.batched_envs != 1:
        raise SystemExit(
            "standard camera collection requires --batched_envs 1 per shard"
        )
    camera_mount_noise_enabled = bool(
        args.camera_mount_position_noise_m > 0.0
        or args.camera_mount_rotation_noise_deg > 0.0
    )
    if camera_mount_noise_enabled and args.standard_dataset_dir is None:
        raise SystemExit("camera mount randomization is only supported for standard data collection")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        suggested = Path.cwd() / "doorman_dataset"
        raise SystemExit(
            f"Cannot write output directory {args.output_dir}: permission denied. "
            f"Use a writable path such as {suggested} or /tmp/doorman_dataset."
        ) from exc
    simulation_app = launch_app(args)
    try:
        _run_batched(args)
        return
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        # Isaac Sim 5.1 can deadlock in the headless render callback during
        # SimulationApp.close().  Process teardown releases those resources;
        # interactive runs still use the normal close path.
        if not args.headless and simulation_app is not None and hasattr(simulation_app, "close"):
            simulation_app.close()


if __name__ == "__main__":
    main()
