"""Fixed AutoOpen command-line contract and runtime path resolution."""

import argparse
import json
from pathlib import Path

from legged_gym.utils.helpers import build_common_arg_parser
from legged_gym.utils.isaaclab_app import add_app_launcher_args

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
LOWLEVEL_ROOT = SCRIPT_ROOT / "third_party" / "visual_wholebody_lowlevel"

# [x, y, z, roll, pitch, yaw] in the B2Z1 goal-reference frame.  This is the
# shared physical endpoint for RESTORE_GRIPPER and PUSH_ARM_RETRACT.
DEFAULT_ARM_RETRACT_POSE_LOCAL = (
    0.30,
    0.0,
    0.30,
    -1.5707963267948966,
    0.0,
    0.0,
)

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
    parser.add_argument(
        "--arm_retract_pose_local",
        type=float,
        nargs=6,
        default=DEFAULT_ARM_RETRACT_POSE_LOCAL,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help=(
            "Shared six-DoF EE endpoint for RESTORE_GRIPPER and "
            "PUSH_ARM_RETRACT, expressed in the B2Z1 goal-reference frame."
        ),
    )
    parser.add_argument(
        "--arm_retract_position_noise_radius_m",
        type=float,
        default=0.05,
        help=(
            "Per-entry uniform-volume position randomization radius for "
            "RESTORE_GRIPPER and PUSH_ARM_RETRACT."
        ),
    )
    parser.add_argument(
        "--arm_retract_rotation_noise_deg",
        type=float,
        default=5.0,
        help=(
            "Per-entry independent absolute XYZ Euler randomization bound "
            "for RESTORE_GRIPPER and PUSH_ARM_RETRACT."
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
    parser.add_argument(
        "--camera_focal_scale_noise_fraction",
        type=float,
        default=0.0,
        help="Standard-data-only shared fx/fy scale variation per camera.",
    )
    parser.add_argument(
        "--camera_principal_point_noise_px",
        type=float,
        default=0.0,
        help="Standard-data-only principal-point variation radius in pixels.",
    )
    parser.add_argument(
        "--camera_distortion_noise_fraction",
        type=float,
        default=0.0,
        help="Standard-data-only relative variation of fisheye k1 through k4.",
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


def _resolve_runtime_camera_config(
    args,
    nominal_camera_config: Path,
    *,
    position_radius_m: float = 0.0,
    rotation_angle_deg: float = 0.0,
    focal_scale_fraction: float = 0.0,
    principal_point_radius_px: float = 0.0,
    distortion_relative_fraction: float = 0.0,
):
    """Materialize the sole renderer-facing camera configuration."""
    from gr00t.rl.data_collection.camera_mount_randomization import (
        randomize_camera_mounts,
    )

    nominal_camera_config = Path(nominal_camera_config)
    nominal_config = json.loads(
        nominal_camera_config.read_text(encoding="utf-8")
    )
    actual_config, camera_randomization = randomize_camera_mounts(
        nominal_config,
        seed=int(args.seed if args.seed is not None else 0),
        position_radius_m=position_radius_m,
        rotation_angle_deg=rotation_angle_deg,
        focal_scale_fraction=focal_scale_fraction,
        principal_point_radius_px=principal_point_radius_px,
        distortion_relative_fraction=distortion_relative_fraction,
    )
    runtime_config = (
        Path(args.output_dir)
        / f"runtime_rgb_cameras_seed{int(args.seed if args.seed is not None else 0)}.json"
    )
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(
        json.dumps(actual_config, indent=2), encoding="utf-8"
    )
    args.nominal_rgb_camera_config = str(nominal_camera_config)
    args.rgb_camera_config = str(runtime_config)
    args.camera_mount_randomization = camera_randomization


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
        _resolve_runtime_camera_config(
            args,
            nominal_camera_config,
            position_radius_m=args.camera_mount_position_noise_m,
            rotation_angle_deg=args.camera_mount_rotation_noise_deg,
            focal_scale_fraction=args.camera_focal_scale_noise_fraction,
            principal_point_radius_px=args.camera_principal_point_noise_px,
            distortion_relative_fraction=args.camera_distortion_noise_fraction,
        )
    elif args.disable_rgb_cameras:
        args.rgb_camera_config = None
    else:
        nominal_camera_config = Path(
            args.rgb_camera_config
            if args.rgb_camera_config
            else (
                LOWLEVEL_ROOT / "legged_gym" / "scripts" / "rgb_cameras.json"
            )
        )
        _resolve_runtime_camera_config(
            args,
            nominal_camera_config,
        )
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
