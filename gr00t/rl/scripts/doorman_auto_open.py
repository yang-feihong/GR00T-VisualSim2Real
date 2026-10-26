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
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple

if __name__ == "__main__" and os.environ.get("AUTO_OPEN_FIXED_ENTRY") not in {
    "run_auto_open.sh",
    "run_auto_open_pitch_probe.sh",
}:
    raise SystemExit(
        "Direct AutoOpen invocation is disabled. Use a maintained entry in gr00t/rl/scripts/."
    )

import numpy as np

faulthandler.enable()
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
LOWLEVEL_ROOT = SCRIPT_ROOT / "third_party" / "visual_wholebody_lowlevel"
sys.path.insert(0, str(LOWLEVEL_ROOT))

from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy  # noqa: E402
from legged_gym.utils.isaaclab_app import launch_app  # noqa: E402

from gr00t.rl.auto_open.arm_model import read_arm_joint_limits  # noqa: E402
from gr00t.rl.auto_open.cli import (  # noqa: E402
    DEFAULT_ARM_RETRACT_POSE_LOCAL,
    DEFAULT_ARM_RETRACT_POSITION_NOISE_RADIUS_M,
    DEFAULT_ARM_RETRACT_ROTATION_NOISE_DEG,
    _default_paths,
    _parser,
)
from gr00t.rl.auto_open.geometry import _resolved_door_prim_path  # noqa: E402
from gr00t.rl.auto_open.target_noise import (  # noqa: E402
    EpisodeTargetNoise,
    SemanticTargetNoiseConfig,
)


class Stage(IntEnum):
    APPROACH_HANDLE = 1
    MOVE_TO_PREGRASP = 2
    OPEN_GRIPPER = 3
    INSERT_AROUND_HANDLE = 4
    GRASP_HANDLE = 5
    PRESS_HANDLE = 6
    OPEN_DOOR_WITH_HANDLE = 7
    RESET_HANDLE = 8
    RELEASE_HANDLE = 9
    WITHDRAW_FROM_HANDLE = 10
    STOW_ARM_AND_CLOSE_GRIPPER = 11
    MOVE_TO_GAP_STATION = 20
    INSERT_INTO_DOOR_GAP = 21
    PRY_DOOR_OPEN = 22
    STOW_ARM_AFTER_PULL = 23
    FOLLOW_WALL_TO_PANEL = 24
    MOVE_TO_PANEL_STATION = 12
    MOVE_TO_PANEL_STANDOFF = 13
    PUSH_PANEL_OPEN = 14
    STOW_ARM_AFTER_PUSH = 15
    ADVANCE_TO_NEXT_PUSH_STATION = 16
    PASS_THROUGH_DOORWAY = 17
    SUCCESS = 18
    FAILURE = 19


@dataclass(frozen=True)
class StageSpec:
    control: str
    acceptance: str
    budget_steps: Optional[int]
    successors: Tuple[Stage, ...]


# This is the executable AutoOpen stage graph. DoorController derives timeout
# budgets from it and validates every runtime stage transition against it.
STAGE_SPECS = {
    Stage.APPROACH_HANDLE: StageSpec(
        "track the door-relative base station; hold the initial arm pose",
        "base pose and measured velocity remain inside the approach window",
        1000,
        (Stage.MOVE_TO_PREGRASP,),
    ),
    Stage.MOVE_TO_PREGRASP: StageSpec(
        "hold the base station; follow the planned pregrasp EE path",
        "path complete and the measured EE/base pose is stable and contact-free",
        500,
        (Stage.OPEN_GRIPPER,),
    ),
    Stage.OPEN_GRIPPER: StageSpec(
        "hold the pregrasp EE pose and command the gripper open",
        "pregrasp remains valid and measured gripper opening is stable",
        100,
        (Stage.INSERT_AROUND_HANDLE,),
    ),
    Stage.INSERT_AROUND_HANDLE: StageSpec(
        "follow the planned insertion path with the gripper open",
        "handle is centered in the capture pocket/channel and EE pose is stable",
        800,
        (Stage.GRASP_HANDLE,),
    ),
    Stage.GRASP_HANDLE: StageSpec(
        "hold EE pose and command gripper closure",
        "stable force-qualified opposed contacts establish a grasp",
        400,
        (Stage.PRESS_HANDLE,),
    ),
    Stage.PRESS_HANDLE: StageSpec(
        "rotate the grasped handle until the latch releases",
        "the measured latch angle reaches the release threshold",
        900,
        (Stage.OPEN_DOOR_WITH_HANDLE,),
    ),
    Stage.OPEN_DOOR_WITH_HANDLE: StageSpec(
        "push or pull the grasped handle along the door-hinge arc",
        "door gap and slow-hinge release conditions hold while grasp is retained",
        900,
        (Stage.RESET_HANDLE,),
    ),
    Stage.RESET_HANDLE: StageSpec(
        "return the handle along its release path while retaining the grasp",
        "handle has returned horizontally and remained stable",
        150,
        (Stage.RELEASE_HANDLE,),
    ),
    Stage.RELEASE_HANDLE: StageSpec(
        "open the gripper to release the handle",
        "gripper is open and blocking handle contacts on both jaws are absent",
        100,
        (Stage.WITHDRAW_FROM_HANDLE,),
    ),
    Stage.WITHDRAW_FROM_HANDLE: StageSpec(
        "follow the planned arm retreat path",
        "retreat path is complete and EE motion is stable",
        200,
        (Stage.STOW_ARM_AND_CLOSE_GRIPPER,),
    ),
    Stage.STOW_ARM_AND_CLOSE_GRIPPER: StageSpec(
        "restore the arm/gripper posture along the planned path",
        "restore path is complete and measured arm pose is stable",
        300,
        (Stage.MOVE_TO_PANEL_STATION, Stage.MOVE_TO_GAP_STATION),
    ),
    Stage.MOVE_TO_GAP_STATION: StageSpec(
        "back up 0.20 m at entry heading, then move to a clearance- and insertion-validated "
        "station while holding restored arm joints",
        "base is stable at its sampled station and its collider projection clears the frame",
        1000,
        (Stage.INSERT_INTO_DOOR_GAP,),
    ),
    Stage.INSERT_INTO_DOOR_GAP: StageSpec(
        "hold the base and insert closed jaws behind the free panel edge",
        "closed jaws settle at the inner-panel standoff; physical contact is not required",
        900,
        (Stage.PRY_DOOR_OPEN, Stage.STOW_ARM_AFTER_PULL, Stage.MOVE_TO_GAP_STATION),
    ),
    Stage.PRY_DOOR_OPEN: StageSpec(
        "hold the base, follow a hinge-centered position arc with speed feedback, "
        "and hold world orientation",
        "door is at least 30 degrees open and reaches the opening goal "
        "or the arm comfort-workspace boundary",
        900,
        (Stage.STOW_ARM_AFTER_PULL,),
    ),
    Stage.STOW_ARM_AFTER_PULL: StageSpec(
        "hold the base and retract closed jaws along the shared restore planner",
        "restore path is complete and measured arm pose is stable",
        300,
        (Stage.FOLLOW_WALL_TO_PANEL, Stage.PASS_THROUGH_DOORWAY),
    ),
    Stage.FOLLOW_WALL_TO_PANEL: StageSpec(
        "hold retracted joints and follow the precomputed near-wall shoulder/arcsin-heading "
        "curve; advance samples only while the published heading angle is below the door angle",
        "door-angle gate holds the reference and base settles with door speed below 2 deg/s "
        "throughout the shared stability window, or the door reaches the opening threshold "
        "and enters PASS_THROUGH_DOORWAY directly",
        1000,
        (Stage.MOVE_TO_PANEL_STANDOFF, Stage.PASS_THROUGH_DOORWAY),
    ),
    Stage.MOVE_TO_PANEL_STATION: StageSpec(
        "track the panel-aligned base station while holding arm joints",
        "base reaches and stabilizes at the panel station",
        400,
        (Stage.MOVE_TO_PANEL_STANDOFF,),
    ),
    Stage.MOVE_TO_PANEL_STANDOFF: StageSpec(
        "hold the base and slew the EE to the selected panel standoff",
        "panel-relative EE standoff pose is reached and stable",
        400,
        (Stage.PUSH_PANEL_OPEN, Stage.PASS_THROUGH_DOORWAY),
    ),
    Stage.PUSH_PANEL_OPEN: StageSpec(
        "advance the EE push target while holding the base",
        "door opens sufficiently, or arm extension requests repositioning",
        900,
        (Stage.STOW_ARM_AFTER_PUSH, Stage.PASS_THROUGH_DOORWAY, Stage.STOW_ARM_AFTER_PULL),
    ),
    Stage.STOW_ARM_AFTER_PUSH: StageSpec(
        "retract the arm from the panel while holding the base",
        "arm is clear and stable for base repositioning",
        300,
        (Stage.ADVANCE_TO_NEXT_PUSH_STATION, Stage.PASS_THROUGH_DOORWAY),
    ),
    Stage.ADVANCE_TO_NEXT_PUSH_STATION: StageSpec(
        "advance toward the monotonic shoulder-safe recontact station",
        "base is stable and the panel contact point is back in arm reach",
        900,
        (Stage.MOVE_TO_PANEL_STANDOFF, Stage.PASS_THROUGH_DOORWAY),
    ),
    Stage.PASS_THROUGH_DOORWAY: StageSpec(
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
        Stage.MOVE_TO_PREGRASP,
        Stage.OPEN_GRIPPER,
        Stage.INSERT_AROUND_HANDLE,
        Stage.GRASP_HANDLE,
        Stage.PRESS_HANDLE,
        Stage.OPEN_DOOR_WITH_HANDLE,
        Stage.RESET_HANDLE,
        Stage.RELEASE_HANDLE,
        Stage.WITHDRAW_FROM_HANDLE,
        Stage.STOW_ARM_AND_CLOSE_GRIPPER,
        Stage.MOVE_TO_PANEL_STANDOFF,
        Stage.PUSH_PANEL_OPEN,
        Stage.STOW_ARM_AFTER_PUSH,
        Stage.INSERT_INTO_DOOR_GAP,
        Stage.PRY_DOOR_OPEN,
        Stage.STOW_ARM_AFTER_PULL,
    }
)

CENTER_HANDLE_CHANNEL_DESIGN_ERROR = 0.008
GRASP_FIXED_JAW_BALANCE_LIMIT = 0.030
GRASP_FIXED_JAW_DESIGN_GAP = 0.019
THROUGH_ENTRY_STABILITY_FRAMES = 10
THROUGH_ENTRY_UPRIGHT_COS = math.cos(math.radians(10.0))
PUSH_RETRACT_COMMAND_HOLD_FRAMES = 5


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
        shared_arm_joint_limits=None,
        shared_robot_geometry=None,
        target_noise_seed=0,
        target_noise_config=None,
        arm_retract_pose_local=DEFAULT_ARM_RETRACT_POSE_LOCAL,
        arm_retract_position_noise_radius_m=DEFAULT_ARM_RETRACT_POSITION_NOISE_RADIUS_M,
        arm_retract_rotation_noise_deg=DEFAULT_ARM_RETRACT_ROTATION_NOISE_DEG,
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
        self.arm_retract_pose_local = np.asarray(arm_retract_pose_local, dtype=np.float32)
        if (
            self.arm_retract_pose_local.shape != (6,)
            or not np.isfinite(self.arm_retract_pose_local).all()
        ):
            raise ValueError(
                "arm_retract_pose_local must contain six finite values: x, y, z, roll, pitch, yaw"
            )
        self.arm_retract_position_noise_radius_m = float(arm_retract_position_noise_radius_m)
        self.arm_retract_rotation_noise_deg = float(arm_retract_rotation_noise_deg)
        if self.arm_retract_position_noise_radius_m < 0.0:
            raise ValueError("arm retract position noise radius must be non-negative")
        if not 0.0 <= self.arm_retract_rotation_noise_deg <= 180.0:
            raise ValueError("arm retract rotation noise must be within [0, 180] deg")
        self.arm_retract_random_seed = int(target_noise_seed)
        self.arm_retract_entry_count = 0
        self.arm_retract_randomization_records = []
        self.active_arm_retract_pose_local = self.arm_retract_pose_local.copy()
        self.restore_retract_pose_sampled = False
        self.grasp_semantic_position_delta_world = np.zeros(3, dtype=np.float32)
        self.stage = Stage.APPROACH_HANDLE
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
        self.base_arrival_settle_steps = 0
        self.base_arrival_confirmed = False
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
        self.release_return_steps = 60
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
        self.panel_contact_orientation = None
        self.panel_contact_settle_steps = 0
        self.panel_station_xy = None
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
        self.push_max_angular_lead = math.radians(2.0)  # Initial PRY_DOOR_OPEN arc only.
        from gr00t.rl.auto_open.panel_motion import PUSH_TIP_MAX_ANGULAR_LEAD_RAD

        self.push_tip_max_angular_lead = PUSH_TIP_MAX_ANGULAR_LEAD_RAD
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
        self.through_stability_failure_counts = {}
        self.through_motion_ready = False
        self.through_entry_stable_steps = 0
        self.through_min_base_height = float(self.env.cfg.rewards.base_height_target_min)
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
        arm_urdf = LOWLEVEL_ROOT / "resources/robots/b2z1/urdf" / "b2z1.urdf"
        self.arm_joint_limits = (
            read_arm_joint_limits(arm_urdf)
            if shared_arm_joint_limits is None
            else shared_arm_joint_limits
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
        self.failure_reason = None
        self.failure_stage = None
        self.last_acceptance_diagnostics = {}
        self.last_stability_diagnostics = {}
        self.stage_exit_diagnostics = {}
        self.gpu_geometry_uses = 0
        self.gpu_geometry_fallbacks = 0
        self.batched_handle_center = None
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
        if (
            not hinge
            or not handle
            or not latch
            or handle_body_idx is None
            or panel_body_idx is None
        ):
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
            f"physical_handle_body={self.handle_body_idx}, "
            f"grasp_reference_body={self.grasp_target_body_idx}, "
            f"panel_body={self.panel_body_idx}"
        )

    def _door_prim_path(self):
        return _resolved_door_prim_path(self.env, self.env_id)


def main():
    args = _parser().parse_args()
    try:
        args.target_noise_config = SemanticTargetNoiseConfig.from_assignments(args.target_noise)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.max_steps < 1:
        raise SystemExit("--max_steps must be positive")
    _default_paths(args)
    for label, path in (
        ("checkpoint", args.ckpt_path),
        ("run metadata", args.run_metadata_path),
        ("robot USD", args.robot_usd_path),
    ):
        if path and not Path(path).is_file():
            raise SystemExit(f"{label} not found: {path}")
    if args.rgb_camera_config and not Path(args.rgb_camera_config).is_file():
        raise SystemExit(f"RGB camera config not found: {args.rgb_camera_config}")
    if args.batched_envs < 1:
        raise SystemExit(
            "AutoOpen has one vectorized execution path; use the fixed launcher "
            "which supplies --batched_envs."
        )
    if args.episode_seeds is not None:
        if len(args.episode_seeds) != args.batched_envs:
            raise SystemExit("--episode_seeds must contain exactly --batched_envs values")
        if len(set(args.episode_seeds)) != len(args.episode_seeds):
            raise SystemExit("--episode_seeds must not contain duplicates")
    if args.replay_source_dataset_dir is not None:
        if args.standard_dataset_dir is None:
            raise SystemExit("trajectory replay requires --standard_dataset_dir")
        if args.collection_camera_mode != "dual" or args.disable_rgb_cameras:
            raise SystemExit("trajectory replay requires dual camera mode")
        if not (args.replay_source_dataset_dir / "manifest.json").is_file():
            raise SystemExit(
                "trajectory replay source must contain manifest.json: "
                f"{args.replay_source_dataset_dir}"
            )
    camera_mount_noise_enabled = bool(
        args.camera_mount_position_noise_m > 0.0
        or args.camera_mount_rotation_noise_deg > 0.0
        or args.camera_focal_scale_noise_fraction > 0.0
        or args.camera_principal_point_noise_px > 0.0
        or args.camera_distortion_noise_fraction > 0.0
    )
    if camera_mount_noise_enabled and args.standard_dataset_dir is None:
        raise SystemExit(
            "camera mount randomization is only supported for standard data collection"
        )
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
        if args.replay_source_dataset_dir is not None:
            from gr00t.rl.auto_open.replay import run_trajectory_replay

            run_trajectory_replay(args)
            return
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
