"""Rule-based DoorMan door opening and raw trajectory collection.

This entry point drives the existing B2Z1 low-level policy through the public
teleop target API.  It intentionally does not synthesize keyboard events or
feed the 10-D high-level action directly to Isaac Lab.
"""

from __future__ import annotations

import faulthandler
import math
import os
import signal
import sys
import traceback
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple

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
from legged_gym.utils.isaaclab_app import launch_app  # noqa: E402
from gr00t.rl.auto_open.cli import (  # noqa: E402
    DEFAULT_ARM_RETRACT_POSE_LOCAL,
    _default_paths,
    _parser,
)
from gr00t.rl.auto_open.control import (  # noqa: E402
    BASE_STATION_TOLERANCE_M,
    BASE_YAW_TOLERANCE_RAD,
    THROUGH_STOP_DISTANCE_M,
    THROUGH_STOP_YAW_RAD,
    _approach_base_target,
    _arm_drive_stiffness,
    _arm_pose_is_stable,
    _base_advance_target,
    _base_command,
    _base_velocity_damping,
    _current_panel_station,
    _desired_grasp_quat,
    _effective_grasp_target,
    _ee_local_target,
    _ee_world_orientation_target,
    _ee_world_target,
    _grasp_orientation_delta,
    _limit_base_commands,
    _perturb_base_target,
    _perturb_ee_target,
    _quat_distance,
    _set_base_tracking_target,
    _stabilize_base_command,
    _slew_ee_target,
    _slew_orientation,
    _through_tail_clearance,
    _zero_cross_base_reversal,
)
from gr00t.rl.auto_open.geometry import (  # noqa: E402
    HANDLE_AXLE_MARGIN_MIN_M,
    HANDLE_FREE_MARGIN_MIN_M,
    _center_handle_axial_overlap,
    _contact_is_on_handle_capsule,
    _contacts_form_opposed_grasp,
    _geometry_metrics,
    _panel_finger_clearance,
    _physical_handle_grasp_position,
    _tip_metrics,
    _quat_axis_angle_np,
    _quat_conjugate_np,
    _rotate_about_z,
    _quat_mul_np,
    _quat_rotate_inverse_np,
    _quat_rotate_np,
    _quat_yaw,
    _resolved_door_prim_path,
    _yaw_quat,
)
from gr00t.rl.auto_open.planning import (  # noqa: E402
    _build_release_inverse_path,
    _build_steady_compensated_path,
    _plan_pregrasp_path,
    _prepare_panel_contact_command,
    _reset_turn_path,
    _sample_reachable_arm_retract_pose,
    _turn_path_target,
)
from gr00t.rl.auto_open.target_noise import (  # noqa: E402
    EpisodeTargetNoise,
    SemanticTargetNoiseConfig,
)
from gr00t.rl.auto_open.utils import as_numpy as _as_np  # noqa: E402
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


@dataclass(frozen=True)
class StageSpec:
    control: str
    acceptance: str
    budget_steps: Optional[int]
    successors: Tuple[Stage, ...]


# This is the executable AutoOpen stage graph. DoorController derives timeout
# budgets from it and validates every runtime stage transition against it.
STAGE_SPECS = {
    Stage.APPROACH: StageSpec(
        "track the door-relative base station; hold the initial arm pose",
        "base pose and measured velocity remain inside the approach window",
        1000,
        (Stage.PREGRASP,),
    ),
    Stage.PREGRASP: StageSpec(
        "hold the base station; follow the planned pregrasp EE path",
        "path complete and the measured EE/base pose is stable and contact-free",
        500,
        (Stage.OPEN_GRIPPER,),
    ),
    Stage.OPEN_GRIPPER: StageSpec(
        "hold the pregrasp EE pose and command the gripper open",
        "pregrasp remains valid and measured gripper opening is stable",
        100,
        (Stage.CENTER_HANDLE,),
    ),
    Stage.CENTER_HANDLE: StageSpec(
        "follow the planned insertion path with the gripper open",
        "handle is centered in the capture pocket/channel and EE pose is stable",
        800,
        (Stage.GRASP_HANDLE,),
    ),
    Stage.GRASP_HANDLE: StageSpec(
        "hold EE pose and command gripper closure",
        "stable force-qualified opposed contacts establish a grasp",
        400,
        (Stage.TURN_HANDLE,),
    ),
    Stage.TURN_HANDLE: StageSpec(
        "follow the handle-turn path while retaining the grasp",
        "latch and door release conditions hold with sufficiently slow hinge motion",
        900,
        (Stage.RETURN_HANDLE,),
    ),
    Stage.RETURN_HANDLE: StageSpec(
        "return the handle along its release path while retaining the grasp",
        "handle has returned horizontally and remained stable",
        150,
        (Stage.OPEN_AND_CLEAR,),
    ),
    Stage.OPEN_AND_CLEAR: StageSpec(
        "open the gripper and move clear of the handle",
        "gripper is open and physical handle contact is absent",
        100,
        (Stage.RETREAT_ARM,),
    ),
    Stage.RETREAT_ARM: StageSpec(
        "follow the planned arm retreat path",
        "retreat path is complete and EE motion is stable",
        200,
        (Stage.RESTORE_GRIPPER,),
    ),
    Stage.RESTORE_GRIPPER: StageSpec(
        "restore the arm/gripper posture along the planned path",
        "restore path is complete and measured arm pose is stable",
        300,
        (Stage.ALIGN_PANEL,),
    ),
    Stage.ALIGN_PANEL: StageSpec(
        "track the panel-aligned base station while holding arm joints",
        "base reaches and stabilizes at the panel station",
        400,
        (Stage.CONTACT_PANEL,),
    ),
    Stage.CONTACT_PANEL: StageSpec(
        "hold the base and slew the EE to the selected panel standoff",
        "panel-relative EE standoff pose is reached and stable",
        400,
        (Stage.PUSH_OPEN, Stage.THROUGH),
    ),
    Stage.PUSH_OPEN: StageSpec(
        "advance the EE push target while holding the base",
        "door opens sufficiently, or arm extension requests repositioning",
        900,
        (Stage.PUSH_ARM_RETRACT, Stage.THROUGH),
    ),
    Stage.PUSH_ARM_RETRACT: StageSpec(
        "retract the arm from the panel while holding the base",
        "arm is clear and stable for base repositioning",
        300,
        (Stage.BASE_ADVANCE, Stage.THROUGH),
    ),
    Stage.BASE_ADVANCE: StageSpec(
        "advance toward the monotonic shoulder-safe recontact station",
        "base is stable and the panel contact point is back in arm reach",
        900,
        (Stage.CONTACT_PANEL, Stage.THROUGH),
    ),
    Stage.THROUGH: StageSpec(
        "track the door-frame-relative tail-clear target while restoring the arm",
        "tail clearance, pose, robot, arm, gripper, and door speeds are stable",
        1100,
        (Stage.SUCCESS,),
    ),
    Stage.SUCCESS: StageSpec("hold final commands", "terminal success", None, ()),
    Stage.FAILURE: StageSpec("hold failure commands", "terminal failure", None, ()),
}

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

CENTER_HANDLE_CHANNEL_DESIGN_ERROR = 0.008
GRASP_FIXED_JAW_BALANCE_LIMIT = 0.030
GRASP_FIXED_JAW_DESIGN_GAP = 0.019
THROUGH_ENTRY_STABILITY_FRAMES = 10
THROUGH_ENTRY_UPRIGHT_COS = math.cos(math.radians(10.0))
PUSH_RETRACT_COMMAND_HOLD_FRAMES = 5


def _base_pose_accepted(distance_m, yaw_error_rad):
    return bool(
        float(distance_m) < BASE_STATION_TOLERANCE_M
        and abs(float(yaw_error_rad)) < BASE_YAW_TOLERANCE_RAD
    )


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
        arm_retract_pose_local=DEFAULT_ARM_RETRACT_POSE_LOCAL,
        arm_retract_position_noise_radius_m=0.05,
        arm_retract_rotation_noise_deg=5.0,
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
        self.arm_retract_pose_local = np.asarray(
            arm_retract_pose_local, dtype=np.float32
        )
        if self.arm_retract_pose_local.shape != (6,) or not np.isfinite(
            self.arm_retract_pose_local
        ).all():
            raise ValueError(
                "arm_retract_pose_local must contain six finite values: "
                "x, y, z, roll, pitch, yaw"
            )
        self.arm_retract_position_noise_radius_m = float(
            arm_retract_position_noise_radius_m
        )
        self.arm_retract_rotation_noise_deg = float(
            arm_retract_rotation_noise_deg
        )
        if self.arm_retract_position_noise_radius_m < 0.0:
            raise ValueError("arm retract position noise radius must be non-negative")
        if not 0.0 <= self.arm_retract_rotation_noise_deg <= 180.0:
            raise ValueError("arm retract rotation noise must be within [0, 180] deg")
        self.arm_retract_random_seed = int(target_noise_seed)
        self.arm_retract_entry_count = 0
        self.arm_retract_randomization_records = []
        self.active_arm_retract_pose_local = self.arm_retract_pose_local.copy()
        self.restore_retract_pose_sampled = False
        self.grasp_semantic_position_delta_world = np.zeros(
            3, dtype=np.float32
        )
        self.stage = Stage.APPROACH
        self.stage_steps = 0
        self.stage_limits = {
            stage: spec.budget_steps
            for stage, spec in STAGE_SPECS.items()
            if spec.budget_steps is not None
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
        self.base_command_previous = np.zeros(3, dtype=np.float32)
        self.base_reversal_pending_sign = np.zeros(3, dtype=np.int8)
        self.base_reversal_pending_steps = np.zeros(3, dtype=np.int32)
        self.base_reversal_lock_steps = np.zeros(3, dtype=np.int32)
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
        self.through_motion_ready = False
        self.through_entry_stable_steps = 0
        self.through_min_base_height = float(
            self.env.cfg.rewards.base_height_target_min
        )
        self.hinge_world = None
        self.hinge_axis_sign = None
        self.final_min_normal_offset = None
        self.handle_lower_z = None
        self.handle_upper_z = None
        self.handle_collider_center_local = None
        self.handle_axis_local = None
        self.handle_free_axis_local = None
        self.handle_free_axis_world = None
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

    def _transition_to(self, next_stage, *, reset_steps=True):
        if next_stage != Stage.FAILURE:
            allowed = STAGE_SPECS[self.stage].successors
            if next_stage not in allowed:
                allowed_names = ", ".join(stage.name for stage in allowed) or "none"
                raise RuntimeError(
                    f"illegal AutoOpen transition {self.stage.name} -> "
                    f"{next_stage.name}; allowed: {allowed_names}"
                )
        self.stage = next_stage
        if reset_steps:
            self.stage_steps = 0

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


    def _center_capture_ready(self, metrics):
        """Return whether the open jaws geometrically enclose the handle."""
        return bool(
            metrics.get("valid")
            and abs(metrics.get("pocket_normal_error", 999.0)) <= 0.005
            and metrics.get("handle_axle_margin", -999.0)
            >= HANDLE_AXLE_MARGIN_MIN_M
            and metrics.get("handle_free_margin", -999.0)
            >= HANDLE_FREE_MARGIN_MIN_M
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
                ("handle_axle_margin", HANDLE_AXLE_MARGIN_MIN_M, 0.0001),
                ("handle_free_margin", HANDLE_FREE_MARGIN_MIN_M, 0.0001),
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


    def _command_approach(self, state):
        """Publish the sole APPROACH command for scalar and batched runners."""
        base_cmd, station_dist, yaw_error = _base_command(self, state)
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
        base_cmd = _zero_cross_base_reversal(
            self, _limit_base_commands(base_cmd)
        )
        target_xy, target_yaw = _perturb_base_target(
            self,
            "approach", *_approach_base_target(self)
        )
        nominal_xy, nominal_yaw = _approach_base_target(self)
        _set_base_tracking_target(
            self,
            target_xy, target_yaw, nominal_xy, nominal_yaw
        )
        self.base_hold_stage = None
        self.active_grasp_quat = _slew_orientation(
            self,
            self.initial_ee_orientation
        )
        self.ee_command_world = _as_np(state["ee_pos"])[0].copy()
        self.ee_orientation_error = _quat_distance(
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
            orientation_delta[0] = _grasp_orientation_delta(
                self,
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
        grasp_center = _physical_handle_grasp_position(self, door_bodies[self.handle_body_idx, :7])
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
        base_cmd, station_dist, yaw_error = _base_command(self, state)
        # Keep one fixed target derived from the reset USD collider template.
        # A target recomputed from the moving jaws makes the reference chase
        # the IK error and can drive the arm away from the handle.
        desired_grasp_quat = _desired_grasp_quat(self, grasp_quat)
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
        if self.stage == Stage.PUSH_ARM_RETRACT:
            desired_orientation = _ee_world_orientation_target(
                self, self.active_arm_retract_pose_local[3:]
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
            self.active_grasp_quat = _slew_orientation(self, command_orientation)
        ee_orientation_error = 999.0
        if state.get("ee_orn") is not None:
            ee_orientation_error = _quat_distance(
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
            ee_target_world, self.active_grasp_quat = _turn_path_target(self, state)
            gripper = 0.0
        elif self.stage == Stage.RETURN_HANDLE:
            return_fraction = float(
                np.clip(self.stage_steps / self.release_return_steps, 0.0, 1.0)
            )
            desired_handle_angle = (
                (1.0 - return_fraction) * self.release_start_handle_angle
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
            ee_target_world, self.active_grasp_quat = _perturb_ee_target(
                self,
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
                    finger_clearance = _panel_finger_clearance(
                        self,
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
                        tracked_contact_world = _rotate_about_z(
                            self.push_contact_world,
                            self.hinge_world,
                            tracked_contact_delta,
                        )
                        tracked_contact_orientation = _quat_mul_np(
                            _yaw_quat(tracked_contact_delta),
                            self.push_orientation,
                        )
                    if self.stage == Stage.PUSH_OPEN and arm_overextended:
                        self.push_arm_detached = True
                        self.push_recovery_settle_steps = 0
                        self.push_detached_ee_goal_local = current_goal_local.copy()
                        _sample_reachable_arm_retract_pose(
                            self, state, Stage.PUSH_ARM_RETRACT.name
                        )
                        self._transition_to(Stage.PUSH_ARM_RETRACT)
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
                            tracked_contact_world = _rotate_about_z(
                                self.push_contact_world,
                                self.hinge_world,
                                tracked_contact_delta,
                            )
                            tracked_contact_orientation = _quat_mul_np(
                                _yaw_quat(tracked_contact_delta),
                                self.push_orientation,
                            )
                        tracked_contact_local = _as_np(
                            _ee_local_target(self, tracked_contact_world)
                        )[0]
                        if self.stage == Stage.PUSH_ARM_RETRACT:
                            retract_delta = (
                                self.active_arm_retract_pose_local[:3]
                                - self.push_detached_ee_goal_local
                            )
                            retract_distance = float(np.linalg.norm(retract_delta))
                            if self.stage_steps >= PUSH_RETRACT_COMMAND_HOLD_FRAMES:
                                if retract_distance > 0.004:
                                    self.push_detached_ee_goal_local += (
                                        0.004 / retract_distance
                                    ) * retract_delta
                                else:
                                    self.push_detached_ee_goal_local = (
                                        self.active_arm_retract_pose_local[:3].copy()
                                    )
                            ee_local_override = self.push_detached_ee_goal_local.copy()
                            actual_ee_world = _as_np(state["ee_pos"])[0]
                            actual_ee_local = _as_np(
                                _ee_local_target(self, actual_ee_world)
                            )[0]
                            retract_position_error = float(
                                np.linalg.norm(
                                    actual_ee_local
                                    - self.active_arm_retract_pose_local[:3]
                                )
                            )
                            retract_orientation_target = (
                                _ee_world_orientation_target(
                                    self,
                                    self.active_arm_retract_pose_local[3:],
                                )
                            )
                            retract_orientation_error = _quat_distance(
                                _as_np(state["ee_orn"])[0],
                                retract_orientation_target,
                            )
                            arm_speed = float(
                                np.linalg.norm(_as_np(state["ee_vel"])[0, :3])
                            )
                            arm_retracted = bool(
                                retract_position_error <= 0.010
                                and retract_orientation_error < 0.05
                                and arm_speed < 0.04
                            )
                            self.last_acceptance_diagnostics.update(
                                {
                                    "push_retract_position_error_m": (
                                        retract_position_error
                                    ),
                                    "push_retract_orientation_error_rad": (
                                        retract_orientation_error
                                    ),
                                    "push_retract_command_remaining_m": (
                                        retract_distance
                                    ),
                                    "push_retract_arm_speed_m_s": arm_speed,
                                }
                            )
                            self.push_recovery_settle_steps = (
                                self.push_recovery_settle_steps + 1
                                if arm_retracted
                                else 0
                            )
                            if self.push_recovery_settle_steps >= 10:
                                self.stage_exit_diagnostics[
                                    Stage.PUSH_ARM_RETRACT.name
                                ] = dict(self.last_acceptance_diagnostics)
                                arm_joint_ids = _as_np(
                                    self.env.arm_joint_ids
                                ).astype(int)
                                self.base_advance_arm_joint_target = _as_np(
                                    state["dof_pos"]
                                )[0, arm_joint_ids].copy()
                                self._transition_to(Stage.BASE_ADVANCE)
                                self.push_recovery_settle_steps = 0
                                self.push_reengage_reachable = False
                                self.base_advance_target_longitudinal = None
                                print(
                                    "[doorman][push] arm comfortable; "
                                    "begin base-only advance",
                                    flush=True,
                                )
                        elif self.stage == Stage.BASE_ADVANCE:
                            ee_local_override = (
                                self.active_arm_retract_pose_local[:3].copy()
                            )
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
                        if self.stage != Stage.PUSH_ARM_RETRACT:
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
                        ee_target_world = _rotate_about_z(
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
                            _yaw_quat(current_hinge_delta_world),
                            self.push_orientation,
                        )
                    if self.stage != Stage.PUSH_ARM_RETRACT:
                        self.active_grasp_quat = self.push_orientation.copy()
            else:
                if self.prepared_panel_contact_command is None:
                    _prepare_panel_contact_command(self, state)
                (
                    ee_target_world,
                    prepared_orientation,
                ) = self.prepared_panel_contact_command
                self.active_grasp_quat = _slew_orientation(
                    self,
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
            ee_target_world = _slew_ee_target(self, ee_target_world, max_step=max_step)
        else:
            self.ee_command_world = _as_np(state["ee_pos"])[0].copy()

        if self.stage == Stage.BASE_ADVANCE:
            if self.disable_push_base_assist:
                base_advance_target = _as_np(state["root_states"])[0, :2].copy()
                base_advance_yaw = _quat_yaw(_as_np(state["root_states"])[0, 3:7])
            else:
                base_advance_target, base_advance_yaw = (
                    _base_advance_target(self, state)
                )
            base_cmd, station_dist, yaw_error = _base_command(
                self,
                state,
                base_advance_target,
                base_advance_yaw,
            )
            _set_base_tracking_target(
                self,
                base_advance_target, base_advance_yaw
            )
            self.base_hold_stage = None
        elif self.stage == Stage.ALIGN_PANEL:
            panel_nominal_xy, panel_nominal_yaw = _current_panel_station(self, state)
            panel_station_xy, panel_station_yaw = _perturb_base_target(
                self,
                "align_panel", panel_nominal_xy, panel_nominal_yaw
            )
            base_cmd, station_dist, yaw_error = _base_command(
                self,
                state,
                panel_station_xy,
                panel_station_yaw,
                panel_nominal_xy,
                panel_nominal_yaw,
            )
            _set_base_tracking_target(
                self,
                panel_station_xy,
                panel_station_yaw,
                panel_nominal_xy,
                panel_nominal_yaw,
            )
            self.base_hold_stage = None
        elif self.stage in (Stage.THROUGH, Stage.SUCCESS):
            through_target_xy, through_target_yaw = _perturb_base_target(
                self,
                "through", self.through_station_xy, self.through_station_yaw
            )
            base_cmd, station_dist, yaw_error = _base_command(
                self,
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
                base_cmd[:] = _base_velocity_damping(state)[0]
            if self.stage == Stage.THROUGH and not self.through_motion_ready:
                base_cmd[:] = 0.0
            _set_base_tracking_target(
                self,
                through_target_xy,
                through_target_yaw,
                self.through_station_xy,
                self.through_station_yaw,
            )
            self.base_hold_stage = None
        elif self.stage in BASE_HOLD_STAGES:
            hold_target = self._base_hold_target()
            base_cmd, station_dist, yaw_error = _base_command(
                self,
                state,
                hold_target[0],
                hold_target[1],
                hold_target[2],
                hold_target[3],
            )
            if (
                self.stage not in (Stage.PREGRASP, Stage.OPEN_GRIPPER)
                or _base_pose_accepted(station_dist, yaw_error)
            ):
                base_cmd[:] = 0.0
        else:
            base_cmd[:] = 0.0

        base_cmd = _stabilize_base_command(self, _limit_base_commands(base_cmd))

        ee_local = (
            torch.as_tensor(ee_local_override, device=self.env.device, dtype=torch.float32).reshape(1, 3)
            if ee_local_override is not None
            else _ee_local_target(self, ee_target_world)
        )
        # Stage-specific inverse paths may replace the generic slew output.
        # Keep the slew state aligned with the quaternion actually sent so a
        # later stage cannot resume from a stale grasp orientation.
        self.ee_orientation_command = self.active_grasp_quat.copy()
        orn = np.zeros((1, 3), dtype=np.float32)
        use_world_orientation = self.stage != Stage.SUCCESS
        if state.get("ee_orn") is not None:
            self.ee_orientation_error = _quat_distance(
                _as_np(state["ee_orn"])[0], self.active_grasp_quat
            )
        if use_world_orientation:
            if not self.defer_batched_orientation_delta:
                orn[0] = _grasp_orientation_delta(
                    self,
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
        self._transition_to(Stage.CONTACT_PANEL)

    def _begin_through(self, opening_angle, opening_speed):
        """Start traversing as soon as the door crosses the accepted angle."""
        self.through_best_station_error = math.inf
        self.through_no_progress_steps = 0
        self.through_settle_steps = 0
        self.through_stability_failure_counts = {}
        self.through_motion_ready = False
        self.through_entry_stable_steps = 0
        self._transition_to(Stage.THROUGH)
        print(
            f"[doorman][push] reached 89 deg threshold; begin traversal "
            f"hinge={opening_angle:.4f} "
            f"speed={opening_speed:.4f} max_speed="
            f"{self.push_max_observed_angular_speed:.4f}",
            flush=True,
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
            self._transition_to(Stage.CENTER_HANDLE)
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
            self._transition_to(Stage.RESTORE_GRIPPER)
        else:
            raise RuntimeError(f"unsupported deferred path label: {label}")

        self.stage_steps += 1
        if (
            self.stage in self.stage_limits
            and self.stage_steps > self.stage_limits[self.stage]
        ):
            self.failure_reason = f"stage_timeout:{self.stage.name.lower()}"
            self.failure_stage = self.stage.name
            self._transition_to(Stage.FAILURE, reset_steps=False)
        if self.stage != previous_stage:
            diagnostics = dict(
                self.stage_exit_diagnostics.get(previous_stage.name, {})
            )
            diagnostics.update(self.last_acceptance_diagnostics)
            self.stage_exit_diagnostics[previous_stage.name] = diagnostics


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
        grasp_center = _physical_handle_grasp_position(
            self,
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
        grasp_target = _effective_grasp_target(self, grasp_center)
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
                    on_capsule = _contact_is_on_handle_capsule(
                        self,
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
                metrics = _geometry_metrics(self, state, grasp_center)
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
                            _center_handle_axial_overlap(self, metrics)
                        ),
                        "handle_axle_margin_m": float(
                            metrics["handle_axle_margin"]
                        ),
                        "handle_free_margin_m": float(
                            metrics["handle_free_margin"]
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
                ) = _plan_pregrasp_path(self, state, ee)
                self.arm_align_path_index = 0
                self.arm_align_command_world = self.arm_align_command_path[-1].copy()
                self.arm_align_command_quat = self.arm_align_orientation_path[-1].copy()
                self.arm_stability_window.clear()
                self._transition_to(Stage.PREGRASP)
        elif self.stage in (Stage.PREGRASP, Stage.OPEN_GRIPPER):
            arm_stable = _arm_pose_is_stable(self, state)
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
                    self._transition_to(Stage.OPEN_GRIPPER)
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
                open_metrics = _geometry_metrics(self, state, grasp_center)
                if not open_metrics.get("valid"):
                    self.failure_reason = "open_channel_geometry_unavailable"
                    self._transition_to(Stage.FAILURE, reset_steps=False)
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
                    self._transition_to(Stage.FAILURE, reset_steps=False)
                    return self.stage
                self.arm_forward_target_world[2] += self.center_vertical_plan[
                    "shift_m"
                ]
                (
                    arm_forward_command_target,
                    arm_forward_command_orientation,
                ) = _perturb_ee_target(
                    self,
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
                ) = _build_steady_compensated_path(
                    self,
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
                self._transition_to(Stage.CENTER_HANDLE)
        elif self.stage == Stage.CENTER_HANDLE:
            if self.q_open is None:
                self.failure_reason = "geometry_q_open_unavailable"
                self._transition_to(Stage.FAILURE, reset_steps=False)
            elif self.arm_forward_target_world is None:
                self.failure_reason = "arm_forward_target_unavailable"
                self._transition_to(Stage.FAILURE, reset_steps=False)
            elif metrics.get("valid") and metrics["panel_clearance"] < 0.005:
                self.failure_reason = (
                    f"center_panel_clearance={metrics['panel_clearance']:.4f}"
                )
                self._transition_to(Stage.FAILURE, reset_steps=False)
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
                arm_stable = _arm_pose_is_stable(self, state)
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
                    self._transition_to(Stage.GRASP_HANDLE)
                    self.grasp_settle_steps = 0
        elif self.stage == Stage.GRASP_HANDLE:
            if metrics.get("valid") and metrics["panel_clearance"] < 0.005:
                self.failure_reason = f"grasp_panel_clearance={metrics['panel_clearance']:.4f}"
                self._transition_to(Stage.FAILURE, reset_steps=False)
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
            opposed_grasp, opposed_metrics = _contacts_form_opposed_grasp(
                self,
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
                self._transition_to(Stage.TURN_HANDLE)
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
                _reset_turn_path(self)
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
                self._transition_to(Stage.RETURN_HANDLE)
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
                    self._transition_to(Stage.OPEN_AND_CLEAR)
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
                    ) = _perturb_ee_target(
                        self,
                        "open_and_clear",
                        release_open_end_world,
                        self.release_start_quat,
                    )
                    (
                        self.release_open_command_path,
                        self.release_open_orientation_path,
                    ) = _build_release_inverse_path(
                        self,
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
                    self._transition_to(Stage.RETREAT_ARM)
                    self.release_retreat_start_world = ee.copy()
                    self.release_retreat_target_world = (
                        self.release_retreat_start_world
                        + 0.10 * self.door_normal
                        + np.array([0.0, 0.0, 0.02], dtype=np.float32)
                    )
                    (
                        self.release_retreat_command_target_world,
                        release_retreat_command_orientation,
                    ) = _perturb_ee_target(
                        self,
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
                    ) = _build_release_inverse_path(
                        self,
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
                    if not self.restore_retract_pose_sampled:
                        _sample_reachable_arm_retract_pose(
                            self, state, Stage.RESTORE_GRIPPER.name
                        )
                        self.restore_retract_pose_sampled = True
                    self.release_endpoint_world = _ee_world_target(
                        self,
                        self.active_arm_retract_pose_local[:3],
                    )
                    self.release_endpoint_quat = _ee_world_orientation_target(
                        self, self.active_arm_retract_pose_local[3:]
                    )
                    current_orientation = _as_np(state["ee_orn"])[0].copy()
                    (
                        release_command_endpoint_world,
                        release_command_endpoint_quat,
                    ) = _perturb_ee_target(
                        self,
                        "restore_gripper",
                        self.release_endpoint_world,
                        self.release_endpoint_quat,
                    )
                    # These are the final geometric targets presented to the
                    # inverse path.  Acceptance must use the same targets.
                    self.release_endpoint_world = (
                        release_command_endpoint_world.copy()
                    )
                    self.release_endpoint_quat = (
                        release_command_endpoint_quat.copy()
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
                    ) = _build_release_inverse_path(
                        self,
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
                        "[doorman][release] restoring configured six-DoF pose "
                        f"pose_local={self.active_arm_retract_pose_local.round(5).tolist()}",
                        flush=True,
                    )
                    self.ee_command_world = (
                        self.release_restore_command_path[0].copy()
                    )
                    self.arm_stability_window.clear()
                    self.release_stable_steps = 0
                    self._transition_to(Stage.RESTORE_GRIPPER)
        elif self.stage == Stage.RESTORE_GRIPPER:
            gripper_closed = abs(gripper_pos - float(self.q_closed)) < 0.03
            actual_orientation = _as_np(state["ee_orn"])[0]
            orientation_error = _quat_distance(
                actual_orientation, self.release_endpoint_quat
            )
            position_error = float(
                np.linalg.norm(ee - self.release_endpoint_world)
            )
            command_position_error = float(
                np.linalg.norm(ee - self.release_restore_command_path[-1])
            )
            command_orientation_error = _quat_distance(
                actual_orientation,
                self.release_restore_orientation_path[-1],
            )
            arm_stable = _arm_pose_is_stable(self, state)
            restored = bool(
                self.stage_steps >= 50
                and gripper_closed
                and orientation_error < 0.05
                and position_error <= 0.010
                and arm_stable
            )
            self.release_stable_steps = (
                self.release_stable_steps + 1 if restored else 0
            )
            self.last_acceptance_diagnostics.update(
                {
                    "restore_position_error_m": position_error,
                    "restore_orientation_error_rad": orientation_error,
                    "restore_actual_position_world_m": ee.astype(float).tolist(),
                    "restore_target_position_world_m": self.release_endpoint_world.astype(
                        float
                    ).tolist(),
                    "restore_actual_orientation_world_wxyz": actual_orientation.astype(
                        float
                    ).tolist(),
                    "restore_target_orientation_world_wxyz": self.release_endpoint_quat.astype(
                        float
                    ).tolist(),
                    "restore_actual_to_command_position_error_m": (
                        command_position_error
                    ),
                    "restore_actual_to_command_orientation_error_rad": (
                        command_orientation_error
                    ),
                    "restore_gripper_error_rad": abs(
                        gripper_pos - float(self.q_closed)
                    ),
                    "restore_arm_stable": bool(arm_stable),
                    "restore_settle_steps": int(self.release_stable_steps),
                }
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
                self._transition_to(Stage.ALIGN_PANEL)
        elif self.stage == Stage.ALIGN_PANEL:
            panel_station_xy, target_yaw = _current_panel_station(self, state)
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
                _yaw_quat(current_hinge_world), self.door_normal
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
            finger_clearance = _panel_finger_clearance(self, state, hinge)
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
                self.push_finger_clearance = _panel_finger_clearance(self, state, hinge)
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
                    _yaw_quat(self.hinge_axis_sign * hinge),
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
                self._transition_to(Stage.PUSH_OPEN)
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
                self._transition_to(Stage.FAILURE, reset_steps=False)
            elif opening_angle > math.radians(94.0):
                self.failure_reason = (
                    f"push_angle_overshoot:hinge={opening_angle:.4f}"
                )
                self._transition_to(Stage.FAILURE, reset_steps=False)
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
                        self._transition_to(Stage.FAILURE, reset_steps=False)
        elif self.stage == Stage.THROUGH:
            tail_clearance = _through_tail_clearance(self, state)
            opening_angle = hinge
            opening_speed = float(
                _as_np(state["door_joint_velocity"])[0, self.hinge_idx]
            )
            base_up = _quat_rotate_np(
                base[3:7], np.array([0.0, 0.0, 1.0], dtype=np.float32)
            )
            upright = float(base_up[2])
            entry_checks = {
                "base_height": base[2] >= self.through_min_base_height,
                "upright": upright >= THROUGH_ENTRY_UPRIGHT_COS,
                "base_linear_speed": speed < TERMINAL_BASE_LINEAR_SPEED_M_S,
                "base_angular_speed": float(np.linalg.norm(base[10:13]))
                < TERMINAL_BASE_ANGULAR_SPEED_RAD_S,
                "door_joint_speed": abs(opening_speed) < TERMINAL_DOOR_SPEED_RAD_S,
            }
            self.through_entry_stable_steps = (
                self.through_entry_stable_steps + 1
                if all(entry_checks.values())
                else 0
            )
            if self.through_entry_stable_steps >= THROUGH_ENTRY_STABILITY_FRAMES:
                self.through_motion_ready = True
            if not self.through_motion_ready:
                self.through_no_progress_steps = 0
            elif station_dist < self.through_best_station_error - 0.01:
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
                    "through_motion_ready": bool(self.through_motion_ready),
                    "through_entry_stable_steps": int(
                        self.through_entry_stable_steps
                    ),
                    "through_entry_failed_checks": [
                        name for name, passed in entry_checks.items() if not passed
                    ],
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
                self._transition_to(Stage.FAILURE, reset_steps=False)
            elif opening_angle < math.radians(65.0) and tail_clearance < 0.8:
                self.failure_reason = (
                    f"through_door_swingback:hinge={hinge:.4f},"
                    f"tail_clearance={tail_clearance:.3f}"
                )
                self._transition_to(Stage.FAILURE, reset_steps=False)
            elif (
                station_dist >= self.station_tolerance
                and self.through_no_progress_steps > 150
            ):
                self.failure_reason = (
                    f"through_no_progress:station_error={station_dist:.3f},"
                    f"tail_clearance={tail_clearance:.3f}"
                )
                self._transition_to(Stage.FAILURE, reset_steps=False)
            elif (
                # The collector records the pre-step observation.  One extra
                # stable transition leaves 50 complete stable observations in
                # the half-open episode timeline.
                self.through_settle_steps > TERMINAL_STABILITY_FRAMES
            ):
                self._transition_to(Stage.SUCCESS)
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
            self._transition_to(Stage.FAILURE, reset_steps=False)
        if self.stage != previous_stage:
            diagnostics = dict(
                self.stage_exit_diagnostics.get(previous_stage.name, {})
            )
            diagnostics.update(self.last_acceptance_diagnostics)
            self.stage_exit_diagnostics[previous_stage.name] = diagnostics
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
                f"through_tail_clearance={_through_tail_clearance(self, state):.3f} "
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
            tip_metrics = _tip_metrics(self, state, grasp_center)
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
        or args.camera_focal_scale_noise_fraction > 0.0
        or args.camera_principal_point_noise_px > 0.0
        or args.camera_distortion_noise_fraction > 0.0
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
        from gr00t.rl.auto_open.runtime import _run_batched

        _run_batched(args, controller_cls=DoorController, stage_enum=Stage)
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
