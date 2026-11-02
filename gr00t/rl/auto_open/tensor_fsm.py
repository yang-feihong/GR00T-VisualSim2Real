"""GPU-resident batched state machine for AutoOpen.

This module is AutoOpen's only state-machine execution path. Per-environment
controller objects provide immutable task configuration and sparse planning
context during initialization and stage-boundary requests only.
"""

from __future__ import annotations

import math

import torch

from gr00t.rl.auto_open.acceptance import (
    StageAcceptanceMixin,
    TensorStabilityWindow,
    consecutive_steps,
    update_robot_stability,
)
from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory, frame_traversal_waypoints
from gr00t.rl.auto_open.contact_surface import separating_rear_contact
from gr00t.rl.auto_open.control import (
    ALIGN_PANEL_ARRIVAL,
    APPROACH_ARRIVAL,
    BASE_STATION_TOLERANCE_M,
    THROUGH_TAIL_RELEASE_DISTANCE,
    THROUGH_TAIL_TARGET_DISTANCE,
)
from gr00t.rl.auto_open.execution import StageExecutionMixin, TensorCommandBuffers
from gr00t.rl.auto_open.geometry import _quat_rotate_points_torch
from gr00t.rl.auto_open.motion_math import (
    _axis_angle_quat,
    _euler_xyz_quat,
    _quat_conjugate,
    _quat_distance,
    _quat_multiply,
    _quat_rotate,
    _quat_rotate_inverse,
    _rotate_about_z,
    _signed_quat_rotation_about_axis,
    _slew_position,
    _slew_quaternion,
    _yaw_from_wxyz,
    _yaw_quat,
)
from gr00t.rl.auto_open.motion_targets import PaddedPathBank
from gr00t.rl.auto_open.panel_motion import (
    PullPanelConfig,
    fingertip_arc_position,
    hinge_speed_target,
    outside_comfort_workspace,
)
from gr00t.rl.auto_open.pregrasp_reachability import PRESS_HANDLE_LATCH_RELEASE_M
from gr00t.rl.auto_open.pull_station import (
    PullStationConfig,
    angle_to_wall,
    base_collider_points,
    base_projection_clearances,
    door_hardware_height_bounds,
    frame_boundaries,
    robot_collider_cache,
    robot_motion_envelope,
    shoulder_wall_curve,
)
from gr00t.rl.auto_open.stage_interfaces import stage_interfaces
from gr00t.rl.auto_open.stages import TensorStageLifecycle, validate_stage_graph
from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_dot,
    fixed_norm,
    fixed_normalize,
    fixed_sum,
)

THROUGH_ENTRY_STABILITY_FRAMES = 10
THROUGH_ENTRY_UPRIGHT_COS = math.cos(math.radians(10.0))
TERMINAL_BASE_LINEAR_SPEED_M_S = 0.05
TERMINAL_BASE_ANGULAR_SPEED_RAD_S = 0.05
TERMINAL_ARM_JOINT_SPEED_RAD_S = 0.08
TERMINAL_GRIPPER_SPEED_RAD_S = 0.05
TERMINAL_DOOR_SPEED_RAD_S = math.radians(2.0)
TERMINAL_STABILITY_FRAMES = 50
ROBOT_TRUNK_REAR_X = -0.284


class PullTensorHistory:
    """Append device batches online; materialize one environment only at export."""

    def __init__(self, num_envs, formatter=None):
        self.num_envs = num_envs
        self.formatter = formatter
        self.batches = []

    def append_batch(self, rows, record):
        def snapshot(value):
            if isinstance(value, torch.Tensor):
                if value.shape[0] != len(rows):
                    raise ValueError("history tensor batch length disagrees with rows")
                return value.detach().clone()
            if isinstance(value, dict):
                return {key: snapshot(item) for key, item in value.items()}
            return value

        self.batches.append((rows.detach().clone(), snapshot(record)))

    def __getitem__(self, env_id):
        # This is the final-output boundary, never a planning input.
        def finite(value):
            if isinstance(value, list):
                return [finite(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        def materialize(value, index):
            if isinstance(value, torch.Tensor):
                return finite(value[index].detach().cpu().tolist())
            if isinstance(value, dict):
                return {key: materialize(item, index) for key, item in value.items()}
            return value

        result = []
        for rows, record in self.batches:
            for index in (rows == env_id).nonzero().flatten().cpu().tolist():
                item = materialize(record, index)
                result.append(self.formatter(item) if self.formatter else item)
        return result


def _pull_station_history_record(values):
    return {
        "entry_root_pose_world_wxyz": values[:7],
        "entry_hinge_angle_rad": values[7],
        "waypoint_count": int(values[8]),
        "sweep_clear": bool(values[9]),
        "robot_envelope_min_local_m": values[10:13],
        "robot_envelope_max_local_m": values[13:16],
        "waypoints_xy_yaw": [values[16:19], values[19:22]][: int(values[8])],
        "tracking_margin_m": values[25],
        "sampled_station_angle_noise_rad": values[22],
        "effective_station_angle_noise_rad": values[23],
        "rotation_target_admissible": bool(values[24]),
    }


def _pull_cycle_uniform(seeds, cycles, candidate_count):
    """Keep the existing pull stream bitwise unchanged through the shared GPU RNG."""
    from gr00t.rl.auto_open.tensor_random import episode_uniform

    return episode_uniform(seeds, cycles, candidate_count)


class TensorDoorFSM(TensorStageLifecycle, StageExecutionMixin, StageAcceptanceMixin):
    """Batched AutoOpen controller whose mutable per-env state is on device."""

    def __init__(
        self,
        *,
        stage_enum,
        controllers,
        env,
        approach_command_xy=None,
        approach_command_yaw=None,
        approach_nominal_xy=None,
        approach_nominal_yaw=None,
        terminal_stage=None,
        pull_door=False,
        pull_panel_contact_baseline="trunk_front_minus10",
    ):
        self.Stage = stage_enum
        self.terminal_stage = (
            stage_enum.PASS_THROUGH_DOORWAY if terminal_stage is None else terminal_stage
        )
        self.env = env
        self.pull_door = bool(pull_door)
        self.interfaces = stage_interfaces(stage_enum, pull=self.pull_door)
        self.stage_specs = getattr(controllers[0], "stage_specs", None)
        if self.stage_specs is not None:
            validate_stage_graph(stage_enum, self.stage_specs, self.interfaces)
        self.device = env.device
        self.num_envs = len(controllers)
        device = self.device
        self.previous_dof_position = env.dof_pos.detach().clone()
        self.control_joint_velocity = torch.zeros_like(self.previous_dof_position)

        def stack(name, *, dtype=torch.float32):
            return torch.as_tensor(
                [getattr(controller, name) for controller in controllers],
                dtype=dtype,
                device=device,
            )

        self.stage = torch.full(
            (self.num_envs,), int(stage_enum.APPROACH_HANDLE), dtype=torch.int64, device=device
        )
        self.stage_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self.base_arrival_settle_steps = torch.zeros_like(self.stage_steps)
        self.base_arrival_max_settle_steps = torch.zeros_like(self.stage_steps)
        self.base_arrival_confirmed = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.base_stability_confirmed = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.base_stability_steps = torch.zeros_like(self.stage_steps)
        self.base_arrival_inner_distance = torch.full(
            (self.num_envs,), APPROACH_ARRIVAL.inner_distance_m, device=device
        )
        self.base_arrival_outer_distance = torch.full(
            (self.num_envs,), APPROACH_ARRIVAL.outer_distance_m, device=device
        )
        self.base_arrival_inner_yaw = torch.full(
            (self.num_envs,), APPROACH_ARRIVAL.inner_yaw_rad, device=device
        )
        self.base_arrival_outer_yaw = torch.full(
            (self.num_envs,), APPROACH_ARRIVAL.outer_yaw_rad, device=device
        )
        self.base_arrival_required_frames = torch.full(
            (self.num_envs,),
            APPROACH_ARRIVAL.settle_frames,
            dtype=torch.int32,
            device=device,
        )
        self.base_arrival_max_linear_speed = torch.full(
            (self.num_envs,), APPROACH_ARRIVAL.max_linear_speed_m_s, device=device
        )
        self.base_arrival_max_angular_speed = torch.full(
            (self.num_envs,),
            APPROACH_ARRIVAL.max_angular_speed_rad_s,
            device=device,
        )
        self.base_command_previous = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=device
        )
        self.base_reversal_pending_sign = torch.zeros(
            (self.num_envs, 3), dtype=torch.int8, device=device
        )
        self.base_reversal_pending_steps = torch.zeros(
            (self.num_envs, 3), dtype=torch.int32, device=device
        )
        self.base_reversal_lock_steps = torch.zeros_like(self.base_reversal_pending_steps)
        door_normal = stack("door_normal")
        default_xy = stack("station_xy")
        default_yaw = torch.atan2(-door_normal[:, 1], -door_normal[:, 0])

        def supplied(value, default):
            if value is None:
                return default.clone()
            return torch.as_tensor(value, dtype=torch.float32, device=device).clone()

        self.station_xy = supplied(approach_command_xy, default_xy)
        self.station_yaw = supplied(approach_command_yaw, default_yaw)
        self.nominal_station_xy = supplied(approach_nominal_xy, default_xy)
        self.nominal_station_yaw = supplied(approach_nominal_yaw, default_yaw)
        self.initial_arm_joint_target = stack("initial_arm_joint_target")
        self.default_ee_goal_local = stack("default_ee_goal_local")
        self.initial_ee_orientation = stack("initial_ee_orientation")
        self.q_closed = stack("q_closed")
        self.q_open = stack("q_open")
        self.arm_induced_pitch = torch.as_tensor(
            [controller.env.arm_induced_pitch for controller in controllers],
            dtype=torch.float32,
            device=device,
        )
        self.pregrasp_position_tolerance = stack("pregrasp_position_tolerance")
        self.grasp_orientation_tolerance = stack("grasp_orientation_tolerance")
        self.arm_align_desired_target_world = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=device
        )
        self.arm_align_target_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.pregrasp_settle_steps = torch.zeros_like(self.stage_steps)
        self.gripper_open_settle = torch.zeros_like(self.stage_steps)
        self.handle_align_settle_steps = torch.zeros_like(self.stage_steps)
        self.grasp_settle_steps = torch.zeros_like(self.stage_steps)
        self.grasp_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.grasp_bilateral_contact_seen = torch.zeros_like(self.grasp_latched)
        self.grasp_vertical_balance_started = torch.zeros_like(self.grasp_latched)
        self.grasp_vertical_lift = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.grasp_vertical_lift_target = torch.zeros_like(self.grasp_vertical_lift)
        self.handle_contact_last_seen = torch.full(
            (self.num_envs, 2), -1000, dtype=torch.int32, device=device
        )
        self.handle_contact_last_points = torch.full(
            (self.num_envs, 2, 3), torch.nan, dtype=torch.float32, device=device
        )
        self.arm_forward_desired_target_world = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=device
        )
        self.arm_forward_command_world = torch.zeros_like(self.arm_forward_desired_target_world)
        self.arm_forward_command_quat = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=device
        )
        self.arm_forward_target_valid = torch.zeros_like(self.grasp_latched)
        self.handle_body_idx = int(controllers[0].handle_body_idx)
        self.handle_center_local = stack("handle_collider_center_local")
        self.handle_axis_local = stack("handle_axis_local")
        self.handle_half_length = stack("handle_half_length")
        self.handle_radius = 0.5 * (stack("handle_upper_z") - stack("handle_lower_z"))
        self.door_normal = stack("door_normal")
        self.release_contact_body_indices = (
            int(controllers[0].fixed_finger_body_idx),
            int(controllers[0].moving_finger_body_idx),
        )
        self.release_contact_triangles = tuple(
            torch.as_tensor(
                controllers[0].center_geometry[name].triangles_body,
                dtype=torch.float32,
                device=device,
            )
            for name in ("fixed_contact_surfaces", "moving_contact_surfaces")
        )
        self.depth_axis_local = torch.as_tensor(
            [controller.open_template["depth_axis"] for controller in controllers],
            dtype=torch.float32,
            device=device,
        )
        self.base_hold_target_xy = self.station_xy.clone()
        self.base_hold_target_yaw = self.station_yaw.clone()
        self.base_hold_nominal_xy = self.nominal_station_xy.clone()
        self.base_hold_nominal_yaw = self.nominal_station_yaw.clone()
        self.pending_station_distance = torch.full(
            (self.num_envs,), torch.inf, dtype=torch.float32, device=device
        )
        self.pending_yaw_error = self.pending_station_distance.clone()
        self.arm_align_path = PaddedPathBank(self.num_envs, 512, device)
        if not self.pull_door:
            self.panel_arrival_path = PaddedPathBank(self.num_envs, 384, device)
        self.arm_forward_path = PaddedPathBank(self.num_envs, 512, device)
        self.release_open_path = PaddedPathBank(self.num_envs, 512, device)
        self.release_retreat_path = PaddedPathBank(self.num_envs, 512, device)
        self.release_restore_path = PaddedPathBank(self.num_envs, 512, device)
        self.turn_path = PaddedPathBank(self.num_envs, 512, device)
        self.arm_stability = TensorStabilityWindow(self.num_envs, device)
        self.hinge_idx = int(controllers[0].hinge_idx)
        self.handle_joint_idx = int(controllers[0].handle_joint_idx)
        self.latch_joint_idx = int(controllers[0].latch_joint_idx)
        self.panel_body_idx = int(controllers[0].panel_body_idx)
        self.handle_joint_axis_panel_local = stack("handle_joint_axis_panel_local")
        self.hinge_world = stack("hinge_world")
        self.hinge_axis_sign = stack("hinge_axis_sign")
        self.turn_gap_angle = stack("turn_gap_angle")
        self.turn_target_margin = torch.as_tensor(
            [
                controller.target_noise.positive_margin(
                    "turn_handle_overtravel_rad",
                    controller.target_noise.config.turn_handle_overtravel_rad,
                )
                for controller in controllers
            ],
            dtype=torch.float32,
            device=device,
        )
        self.turn_gap_margin = torch.as_tensor(
            [
                controller.target_noise.positive_margin(
                    "turn_gap_overtravel_rad",
                    controller.target_noise.config.turn_gap_overtravel_rad,
                )
                for controller in controllers
            ],
            dtype=torch.float32,
            device=device,
        )
        scalar_shape = (self.num_envs,)
        vector_shape = (self.num_envs, 3)
        quat_shape = (self.num_envs, 4)
        self.turn_start_handle_angle = torch.zeros(scalar_shape, device=device)
        self.turn_start_hinge_angle = torch.zeros(scalar_shape, device=device)
        self.turn_target_handle_angle = torch.zeros(scalar_shape, device=device)
        self.turn_target_hinge_angle = torch.zeros(scalar_shape, device=device)
        self.turn_handle_axis_world = torch.zeros(vector_shape, device=device)
        self.turn_axle_world = torch.zeros(vector_shape, device=device)
        self.turn_grasp_offset_world = torch.zeros(vector_shape, device=device)
        self.turn_grasp_orientation = torch.zeros(quat_shape, device=device)
        self.turn_gap_gripper_angle = torch.zeros(scalar_shape, device=device)
        self.turn_release_slow_steps = torch.zeros_like(self.stage_steps)
        self.turn_release_slow_steps_required = torch.as_tensor(
            [controller.turn_release_slow_steps_required for controller in controllers],
            dtype=torch.int32,
            device=device,
        )
        self.turn_release_max_hinge_speed = stack("turn_release_max_hinge_speed")
        self.release_return_steps = torch.as_tensor(
            [controller.release_return_steps for controller in controllers],
            dtype=torch.int32,
            device=device,
        )
        self.release_return_axle_world = torch.zeros(vector_shape, device=device)
        self.release_return_axis_world = torch.zeros(vector_shape, device=device)
        self.release_return_offset_world = torch.zeros(vector_shape, device=device)
        self.release_return_start_quat = torch.zeros(quat_shape, device=device)
        self.release_return_angle = torch.zeros(scalar_shape, device=device)
        self.release_vertical_delta = torch.zeros(scalar_shape, device=device)
        self.release_stable_steps = torch.zeros_like(self.stage_steps)
        self.release_retreat_target_world = torch.zeros(vector_shape, device=device)
        self.release_endpoint_world = torch.zeros(vector_shape, device=device)
        self.release_endpoint_quat = torch.zeros(quat_shape, device=device)
        self.align_panel_arm_joint_target = torch.zeros_like(self.initial_arm_joint_target)
        self.panel_align_ee_goal_local = torch.zeros(vector_shape, device=device)
        self.panel_align_ee_orientation_local = torch.zeros(quat_shape, device=device)
        self.panel_station_xy = stack("panel_station_xy")
        self.door_forward = stack("door_forward")
        align_targets = [
            controller.target_noise.perturb_base(
                "align_panel",
                controller.panel_station_xy,
                math.atan2(controller.door_forward[1], controller.door_forward[0]),
                controller.door_normal[:2],
                normal_m=controller.target_noise.config.align_panel_normal_m,
                tangent_m=controller.target_noise.config.align_panel_tangent_m,
                yaw_rad=controller.target_noise.config.align_panel_yaw_rad,
            )
            for controller in controllers
        ]
        self.panel_station_command_xy = torch.as_tensor(
            [target[0] for target in align_targets], dtype=torch.float32, device=device
        )
        self.panel_station_command_yaw = torch.as_tensor(
            [target[1] for target in align_targets], dtype=torch.float32, device=device
        )
        self.panel_station_nominal_yaw = torch.atan2(
            self.door_forward[:, 1], self.door_forward[:, 0]
        )
        points, indices = robot_collider_cache(controllers[0])
        self.base_motion_points = torch.as_tensor(points, device=device)
        self.base_motion_body_indices = torch.as_tensor(indices, device=device)
        if self.pull_door and self.terminal_stage != stage_enum.STOW_ARM_AND_CLOSE_GRIPPER:
            self.pull_station_config = PullStationConfig(
                panel_contact_baseline=pull_panel_contact_baseline
            )
            self.pull_base_points = torch.tensor(
                base_collider_points(controllers[0]), dtype=torch.float32, device=device
            )
            self.pull_motion_points = self.base_motion_points
            self.pull_motion_body_indices = self.base_motion_body_indices
            boundaries = torch.tensor(
                [frame_boundaries(c) for c in controllers], dtype=torch.float32, device=device
            )
            self.pull_frame_edge, self.pull_wall_front = boundaries[:, 0], boundaries[:, 1]
            self.pull_hardware_z = torch.tensor(
                [door_hardware_height_bounds(c) for c in controllers],
                dtype=torch.float32,
                device=device,
            )
            self.pull_free_direction = stack("panel_hinge_to_free_direction")[:, :2]
            self.pull_free_edge = stack("panel_hinge_to_free_edge")
            self.pull_station_xy = torch.zeros((self.num_envs, 2), device=device)
            self.pull_shoulder_anchor = torch.zeros_like(self.pull_station_xy)
            self.pull_shoulder_local = torch.zeros_like(self.pull_station_xy)
            self.pull_panel_side = torch.ones(self.num_envs, device=device)
            self.pull_follow_holding = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self.pull_follow_beta = torch.zeros(self.num_envs, device=device)
            self.pull_follow_velocity = torch.zeros((self.num_envs, 3), device=device)
            self.pull_door_opened = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self.pull_station_yaw = torch.zeros(self.num_envs, device=device)
            self.pull_initial_station_xy = torch.zeros_like(self.pull_station_xy)
            self.pull_initial_station_yaw = torch.zeros_like(self.pull_station_yaw)
            self.pull_base_path = torch.zeros((self.num_envs, 2, 3), device=device)
            self.pull_base_path_length = torch.ones_like(self.stage_steps)
            self.pull_base_path_index = torch.zeros_like(self.stage_steps)
            self.pull_base_trajectory = BasePoseTrajectory(self.num_envs, device)
            self.pull_base_entry_pose = torch.zeros((self.num_envs, 3), device=device)
            self.pull_gap_xy = torch.zeros_like(self.pull_station_xy)
            self.pull_planned_frame_clearance = torch.zeros(self.num_envs, device=device)
            self.pull_planned_wall_clearance = torch.zeros(self.num_envs, device=device)
            self.pull_initial_station_history = PullTensorHistory(self.num_envs)
            self.pull_first_station_attempts = torch.zeros(
                self.num_envs, dtype=torch.int64, device=device
            )
            self.pull_frame_clear_steps = torch.zeros_like(self.stage_steps)
            self.pull_panel_config = PullPanelConfig()
            self.pull_insert_path = PaddedPathBank(self.num_envs, 1024, device)
            self.pull_open_path = PaddedPathBank(self.num_envs, 256, device)
            self.pull_insert_goal = torch.zeros((self.num_envs, 3), device=device)
            self.pull_insert_orientation = torch.zeros((self.num_envs, 4), device=device)
            self.pull_face_offset_local = torch.zeros((self.num_envs, 3), device=device)
            self.pull_panel_contact = torch.zeros((self.num_envs, 3), device=device)
            self.pull_insert_stable_steps = torch.zeros_like(self.stage_steps)
            self.pull_finished_for_reposition = torch.zeros(
                self.num_envs, dtype=torch.bool, device=device
            )
            self.pull_cycle = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
            self.pull_cycle_samples = torch.zeros((self.num_envs, 4), device=device)
            self.prepare_pull_cycle_samples(controllers)
            self.pull_cycle_history = PullTensorHistory(self.num_envs)
            self.pull_station_history = PullTensorHistory(
                self.num_envs, _pull_station_history_record
            )
            self.pull_move_start_hinge = torch.zeros(self.num_envs, device=device)
        self.panel_contact_target_world = torch.zeros(vector_shape, device=device)
        self.panel_contact_desired_ee_world = torch.zeros(vector_shape, device=device)
        self.panel_push_tip_offset_local = torch.zeros(vector_shape, device=device)
        self.panel_push_tip_surface_world = torch.zeros(vector_shape, device=device)
        self.panel_push_tip_hinge_angle = torch.zeros(scalar_shape, device=device)
        self.panel_contact_command_target_world = torch.zeros(vector_shape, device=device)
        self.panel_contact_command_orientation_world = torch.zeros(quat_shape, device=device)
        self.panel_contact_orientation_seed = torch.zeros(quat_shape, device=device)
        self.panel_contact_orientation_seed_valid = torch.zeros_like(self.grasp_latched)
        self.panel_contact_plan_valid = torch.zeros_like(self.grasp_latched)
        self.panel_contact_normal_sign = torch.ones(self.num_envs, device=device)
        self.panel_contact_settle_steps = torch.zeros_like(self.stage_steps)
        self.panel_last_finger_clearance = torch.full(
            scalar_shape, torch.nan, dtype=torch.float32, device=device
        )
        self.panel_last_lateral_error = torch.full_like(self.panel_last_finger_clearance, torch.nan)
        self.panel_last_axis_error = torch.full_like(self.panel_last_finger_clearance, torch.nan)
        self.panel_last_ee_speed = torch.full_like(self.panel_last_finger_clearance, torch.nan)
        self.panel_standoff_acceptance_clearance = stack("panel_standoff_acceptance_clearance")
        self.ee_command_world = torch.zeros(vector_shape, device=device)
        self.ee_orientation_command = torch.zeros(quat_shape, device=device)
        self.ee_command_valid = torch.zeros_like(self.grasp_latched)
        self.push_start_hinge_angle = torch.zeros(scalar_shape, device=device)
        self.push_planned_stop_hinge = torch.full(scalar_shape, math.radians(89.0), device=device)
        self.push_last_progress_hinge = torch.zeros(scalar_shape, device=device)
        self.push_no_progress_steps = torch.zeros_like(self.stage_steps)
        self.push_contact_world = torch.zeros(vector_shape, device=device)
        self.push_tip_offset_world = torch.zeros(vector_shape, device=device)
        self.push_orientation = torch.zeros(quat_shape, device=device)
        self.push_command_hinge_angle = torch.zeros(scalar_shape, device=device)
        self.push_speed_error_integral = torch.zeros(scalar_shape, device=device)
        self.push_overspeed_steps = torch.zeros_like(self.stage_steps)
        self.push_max_observed_angular_speed = torch.zeros(scalar_shape, device=device)
        self.push_comfort_reach_radius = torch.full(
            scalar_shape, 0.7, dtype=torch.float32, device=device
        )
        self.push_arm_detached = torch.zeros_like(self.grasp_latched)
        self.push_detached_ee_goal_local = torch.zeros(vector_shape, device=device)
        self.active_arm_retract_pose_local = stack("active_arm_retract_pose_local")
        self.push_recovery_settle_steps = torch.zeros_like(self.stage_steps)
        self.push_reengage_reachable = torch.zeros_like(self.grasp_latched)
        self.push_previous_finger_clearance = torch.full(
            scalar_shape, torch.nan, dtype=torch.float32, device=device
        )
        self.push_target_angular_speed = stack("push_target_angular_speed")
        self.push_max_angular_speed = stack("push_max_angular_speed")
        self.push_speed_kp = stack("push_speed_kp")
        self.push_speed_ki = stack("push_speed_ki")
        self.push_speed_integral_limit = stack("push_speed_integral_limit")
        self.push_max_angular_lead = stack("push_max_angular_lead")
        self.push_tip_max_angular_lead = stack("push_tip_max_angular_lead")
        self.push_goal_hinge_angle = stack("push_goal_hinge_angle")
        self.push_reengage_reach_radius = stack("push_reengage_reach_radius")
        self.push_reengage_release_radius = stack("push_reengage_release_radius")
        self.push_reengage_target_lead = stack("push_reengage_target_lead")
        self.push_base_start_xy = torch.zeros((self.num_envs, 2), device=device)
        self.push_entry_ee_world = torch.zeros(vector_shape, device=device)
        self.push_entry_command_world = torch.zeros(vector_shape, device=device)
        self.base_advance_arm_joint_target = torch.zeros_like(self.initial_arm_joint_target)
        self.base_advance_target_longitudinal = torch.full(
            scalar_shape, -torch.inf, dtype=torch.float32, device=device
        )
        self.panel_center_world = stack("panel_center_world")
        self.panel_half_thickness = stack("panel_half_thickness")
        self.front_shoulder_body_indices = torch.as_tensor(
            controllers[0].front_shoulder_body_indices,
            dtype=torch.int64,
            device=device,
        )
        self.push_shoulder_collision_radius = stack("push_shoulder_collision_radius")
        self.push_shoulder_safety_margin = stack("push_shoulder_safety_margin")
        self.dt = float(env.dt)
        self.middle_arm_joint_target = stack("middle_arm_joint_target")
        self.through_station_xy = stack("through_station_xy")
        self.through_station_yaw = stack("through_station_yaw")
        through_targets = [
            controller.target_noise.perturb_base(
                "through",
                controller.through_station_xy,
                controller.through_station_yaw,
                controller.door_forward[:2],
                normal_m=controller.target_noise.config.through_forward_m,
                tangent_m=controller.target_noise.config.through_tangent_m,
                yaw_rad=controller.target_noise.config.through_yaw_rad,
            )
            for controller in controllers
        ]
        self.through_command_xy = torch.as_tensor(
            [target[0] for target in through_targets], dtype=torch.float32, device=device
        )
        self.through_command_yaw = torch.as_tensor(
            [target[1] for target in through_targets], dtype=torch.float32, device=device
        )
        self.through_best_station_error = torch.full(
            scalar_shape, torch.inf, dtype=torch.float32, device=device
        )
        self.through_no_progress_steps = torch.zeros_like(self.stage_steps)
        self.through_settle_steps = torch.zeros_like(self.stage_steps)
        self.through_stop_latched = torch.zeros_like(self.grasp_latched)
        self.through_motion_ready = torch.zeros_like(self.grasp_latched)
        self.through_entry_stable_steps = torch.zeros_like(self.stage_steps)
        self.through_trajectory = BasePoseTrajectory(self.num_envs, device)
        self.through_path_previous_index = torch.zeros_like(self.through_trajectory.index)
        self.through_min_base_height = stack("through_min_base_height")
        self.initial_door_root = stack("initial_door_root")
        self.failure_code = torch.zeros(self.num_envs, dtype=torch.int16, device=device)
        self.failure_from_stage = torch.zeros(self.num_envs, dtype=torch.int16, device=device)
        # Every stage reports acceptance through this one observer.  Entries
        # are GPU tensors so diagnostics do not add per-frame host copies.
        self._acceptance = {}
        stage_slots = max(int(stage) for stage in stage_enum) + 1
        self.stage_visit = torch.zeros(
            (self.num_envs, stage_slots), dtype=torch.int32, device=device
        )
        self.stage_visit[:, int(stage_enum.APPROACH_HANDLE)] = 1
        self.stage_limits = torch.zeros(
            stage_slots,
            dtype=torch.int32,
            device=device,
        )
        for stage, limit in controllers[0].stage_limits.items():
            self.stage_limits[int(stage)] = int(limit)

    def update_control_joint_velocity(self, state: dict[str, torch.Tensor]) -> None:
        """Measure joint velocity from consecutive 50 Hz control states.

        PhysX ``dof_vel`` is sampled at the final physics substep and can expose
        a contact-rebound spike that produced almost no control-frame motion.
        Acceptance uses this control-rate displacement instead.
        """
        position = state["dof_pos"]
        self.control_joint_velocity.copy_((position - self.previous_dof_position) / self.dt)
        self.previous_dof_position.copy_(position)

    def control_joint_speed(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor, joint_ids
    ) -> torch.Tensor:
        """Return absolute control-rate joint speed for acceptance checks."""
        if hasattr(self, "control_joint_velocity"):
            return self.control_joint_velocity[rows, joint_ids].abs()
        # Focused unit fixtures may bypass __init__; keep observation passive.
        return torch.zeros_like(state["dof_pos"][rows, joint_ids])

    def command_approach_handle(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        mask = self.mask(self.Stage.APPROACH_HANDLE)
        if not torch.any(mask):
            return
        rows = torch.where(mask)[0]
        command, distance, yaw_error = self._track_station(
            state,
            rows,
            self.station_xy[rows],
            self.station_yaw[rows],
            self.nominal_station_xy[rows],
            self.nominal_station_yaw[rows],
            buffers=buffers,
        )
        self.pending_station_distance[rows] = distance
        self.pending_yaw_error[rows] = yaw_error

        self.arm_joint_target(
            rows,
            self.initial_arm_joint_target[rows],
            buffers,
            idle_position=self.default_ee_goal_local[rows],
            idle_orientation=self.initial_ee_orientation[rows],
        )
        self.gripper_target(rows, self.q_closed[rows], buffers)

    def advance_approach_handle(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Advance APPROACH_HANDLE and return env ids requesting pregrasp planning."""
        mask = self.mask(self.Stage.APPROACH_HANDLE)
        if not torch.any(mask):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        rows = torch.where(mask)[0]
        self.record_acceptance(
            self.Stage.APPROACH_HANDLE,
            rows,
            {
                "base_arrival_confirmed": (
                    self.base_arrival_confirmed[rows],
                    self.base_arrival_settle_steps[rows],
                    ">=frames",
                    self.base_arrival_required_frames[rows],
                ),
                "base_stability_confirmed": (
                    self.base_stability_confirmed[rows],
                    self.base_stability_steps[rows],
                    ">=frames",
                    self.base_arrival_required_frames[rows],
                ),
            },
        )
        self.stage_steps[rows] += 1
        return rows[self.base_arrival_confirmed[rows] & self.base_stability_confirmed[rows]]

    def command_pregrasp_and_open_gripper(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        mask = self.mask(self.Stage.MOVE_TO_PREGRASP) | self.mask(self.Stage.OPEN_GRIPPER)
        if not torch.any(mask):
            return
        rows = torch.where(mask)[0]
        command, _, _ = self._base_hold_command(state, rows, buffers)
        self.arm_trajectory(rows, self.arm_align_path, buffers)
        pregrasp = self.stage[rows] == int(self.Stage.MOVE_TO_PREGRASP)
        self.gripper_target(
            rows, torch.where(pregrasp, self.q_closed[rows], self.q_open[rows]), buffers
        )

    def advance_pregrasp_and_open_gripper(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Advance both stages and return rows requesting INSERT_AROUND_HANDLE paths."""
        mask = self.mask(self.Stage.MOVE_TO_PREGRASP) | self.mask(self.Stage.OPEN_GRIPPER)
        if not torch.any(mask):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        rows = torch.where(mask)[0]
        stable, _, _ = self.arm_stability.update(state, rows)
        position_error = fixed_norm(
            state["ee_pos"][rows] - self.arm_align_desired_target_world[rows]
        )
        orientation_error = _quat_distance(
            state["ee_orn"][rows],
            self.arm_align_path.orientations[
                rows, (self.arm_align_path.length[rows] - 1).to(torch.int64)
            ],
        )
        root_speed = fixed_norm(state["root_states"][rows, 7:10])
        contacts = state.get("handle_finger_contact_forces_w")
        no_contact = torch.ones(len(rows), dtype=torch.bool, device=self.device)
        if contacts is not None:
            no_contact = torch.all(fixed_norm(contacts[rows]) < 1.0, dim=-1)
        aligned = (
            self.arm_align_path.complete(rows)
            & (position_error <= self.pregrasp_position_tolerance[rows])
            & (orientation_error <= self.grasp_orientation_tolerance[rows])
            & self.base_arrival_confirmed[rows]
            & self.base_stability_confirmed[rows]
            & (root_speed < 0.02)
            & no_contact
            & stable
        )

        common_conditions = {
            "path_complete": (
                self.arm_align_path.complete(rows),
                self.arm_align_path.complete(rows).to(torch.float32),
                "==",
                1.0,
            ),
            "ee_position_error_m": (
                position_error <= self.pregrasp_position_tolerance[rows],
                position_error,
                "<=",
                self.pregrasp_position_tolerance[rows],
            ),
            "ee_orientation_error_rad": (
                orientation_error <= self.grasp_orientation_tolerance[rows],
                orientation_error,
                "<=",
                self.grasp_orientation_tolerance[rows],
            ),
            "base_arrival_confirmed": (
                self.base_arrival_confirmed[rows],
                self.base_arrival_settle_steps[rows],
                ">=frames",
                self.base_arrival_required_frames[rows],
            ),
            "base_stability_confirmed": (
                self.base_stability_confirmed[rows],
                self.base_stability_steps[rows],
                ">=frames",
                self.base_arrival_required_frames[rows],
            ),
            "base_linear_speed_m_s": (root_speed < 0.02, root_speed, "<", 0.02),
            "no_handle_contact": (
                no_contact,
                no_contact.to(torch.float32),
                "==",
                1.0,
            ),
            "arm_stable": (stable, stable.to(torch.float32), "==", 1.0),
        }

        pregrasp = self.stage[rows] == int(self.Stage.MOVE_TO_PREGRASP)
        pre_rows = rows[pregrasp]
        if len(pre_rows):
            self.record_acceptance(
                self.Stage.MOVE_TO_PREGRASP,
                pre_rows,
                {
                    name: tuple(
                        value[pregrasp] if torch.is_tensor(value) and value.ndim else value
                        for value in item
                    )
                    for name, item in common_conditions.items()
                },
                required_streak=5,
            )
        pre_settled = consecutive_steps(aligned[pregrasp], self.pregrasp_settle_steps[pre_rows])
        self.pregrasp_settle_steps[pre_rows] = pre_settled
        to_open = pre_rows[pre_settled >= 5]
        if len(to_open):
            self.transition(to_open, self.Stage.OPEN_GRIPPER)
            self.gripper_open_settle[to_open] = 0

        open_mask = self.stage[rows] == int(self.Stage.OPEN_GRIPPER)
        open_rows = rows[open_mask]
        if len(open_rows):
            gripper_joint_id = int(self.env.gripper_joint_ids[0].item())
            gripper_error = (state["gripper_position"][open_rows, 0] - self.q_open[open_rows]).abs()
            gripper_speed = self.control_joint_speed(state, open_rows, gripper_joint_id)
            open_stable = aligned[open_mask] & (gripper_error < 0.02) & (gripper_speed < 0.05)
            open_conditions = {
                name: tuple(
                    value[open_mask] if torch.is_tensor(value) and value.ndim else value
                    for value in item
                )
                for name, item in common_conditions.items()
            }
            open_conditions.update(
                {
                    "gripper_open_error_rad": (gripper_error < 0.02, gripper_error, "<", 0.02),
                    "gripper_speed_rad_s": (gripper_speed < 0.05, gripper_speed, "<", 0.05),
                }
            )
            self.record_acceptance(
                self.Stage.OPEN_GRIPPER,
                open_rows,
                open_conditions,
                required_streak=10,
            )
            open_settled = consecutive_steps(open_stable, self.gripper_open_settle[open_rows])
            self.gripper_open_settle[open_rows] = open_settled
            requests = open_rows[open_settled >= 10]
        else:
            requests = torch.empty(0, dtype=torch.int64, device=self.device)

        active = self.mask(self.Stage.MOVE_TO_PREGRASP) | self.mask(self.Stage.OPEN_GRIPPER)
        self.stage_steps[active] += 1
        return requests

    def command_insert_and_grasp_handle(
        self,
        state: dict[str, torch.Tensor],
        geometry: dict[str, torch.Tensor],
        buffers: TensorCommandBuffers,
    ) -> None:
        mask = self.mask(self.Stage.INSERT_AROUND_HANDLE) | self.mask(self.Stage.GRASP_HANDLE)
        if not torch.any(mask):
            return
        rows = torch.where(mask)[0]
        command, _, _ = self._base_hold_command(state, rows, buffers)
        center = self.stage[rows] == int(self.Stage.INSERT_AROUND_HANDLE)
        target = self.arm_forward_command_world[rows].clone()
        orientation = self.arm_forward_command_quat[rows]
        if torch.any(center):
            center_rows = rows[center]
            path_position, path_orientation = self.arm_forward_path.current(
                center_rows, advance=True
            )
            target[center] = path_position
            orientation = orientation.clone()
            orientation[center] = path_orientation
        grasp = ~center
        if torch.any(grasp):
            grasp_rows = rows[grasp]
            moving_clearance = geometry["moving_handle_clearance"][grasp]
            contact_forces = state.get("handle_finger_contact_forces_w")
            if contact_forces is None:
                contact_norms = torch.zeros(
                    (len(grasp_rows), 2), dtype=torch.float32, device=self.device
                )
            else:
                contact_norms = fixed_norm(contact_forces[grasp_rows])
            fixed_contact = contact_norms[:, 0] >= 1.0
            moving_contact = contact_norms[:, 1] >= 1.0
            fixed_center = geometry["fixed_center"][grasp]
            moving_center = geometry["moving_center"][grasp]
            fixed_is_upper = fixed_center[:, 2] > moving_center[:, 2]
            balance_direction = torch.where(
                fixed_is_upper,
                -torch.ones_like(moving_clearance),
                torch.ones_like(moving_clearance),
            )
            start = (~self.grasp_vertical_balance_started[grasp_rows]) & (
                (moving_clearance <= 0.003) | moving_contact
            )
            upper_gap = geometry["upper_gap"][grasp]
            lower_gap = geometry["lower_gap"][grasp]
            fixed_gap = torch.where(fixed_is_upper, upper_gap, lower_gap).clamp_min(0.0)
            proposed = balance_direction * fixed_gap.clamp(0.0, 0.030)
            self.grasp_vertical_lift_target[grasp_rows] = torch.where(
                start, proposed, self.grasp_vertical_lift_target[grasp_rows]
            )
            self.grasp_vertical_balance_started[grasp_rows] |= start
            imbalance = moving_contact != fixed_contact
            adjusted = (
                self.grasp_vertical_lift_target[grasp_rows] + balance_direction * 0.00005
            ).clamp(-0.030, 0.030)
            self.grasp_vertical_lift_target[grasp_rows] = torch.where(
                imbalance, adjusted, self.grasp_vertical_lift_target[grasp_rows]
            )
            lift_delta = (
                self.grasp_vertical_lift_target[grasp_rows] - self.grasp_vertical_lift[grasp_rows]
            ).clamp(-0.0002, 0.0002)
            self.grasp_vertical_lift[grasp_rows] += lift_delta
            target[grasp, 2] += self.grasp_vertical_lift[grasp_rows]
        self.arm_pose_target(rows, target, orientation, buffers)
        self.gripper_target(
            rows, torch.where(center, self.q_open[rows], self.q_closed[rows]), buffers
        )

    def _handle_contacts(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forces = state.get("handle_finger_contact_forces_w")
        points = state.get("handle_finger_contact_points_w")
        if forces is None or points is None:
            shape = (len(rows), 2)
            return (
                torch.zeros(shape, dtype=torch.bool, device=self.device),
                torch.zeros(shape, dtype=torch.float32, device=self.device),
                torch.full((*shape, 3), torch.nan, dtype=torch.float32, device=self.device),
            )
        points = points[rows]
        norms = fixed_norm(forces[rows])
        handle_pose = state["door_body_state_w"][rows, self.handle_body_idx, :7]
        center = handle_pose[:, :3] + _quat_rotate(
            handle_pose[:, 3:7], self.handle_center_local[rows]
        )
        axis = _quat_rotate(handle_pose[:, 3:7], self.handle_axis_local[rows])
        delta = points - center[:, None, :]
        axial_raw = fixed_dot(delta, axis[:, None, :])
        axial = torch.maximum(
            torch.minimum(axial_raw, self.handle_half_length[rows, None]),
            -self.handle_half_length[rows, None],
        )
        radial = fixed_norm(delta - axial[..., None] * axis[:, None, :])
        present = (
            torch.isfinite(points).all(dim=-1)
            & (radial <= 0.018)
            & (axial_raw.abs() <= self.handle_half_length[rows, None] + 0.008)
            & (norms >= 1.0)
        )
        return present, torch.where(present, norms, 0.0), points

    def _opposed_grasp(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        handle_pose = state["door_body_state_w"][rows, self.handle_body_idx, :7]
        center = handle_pose[:, :3] + _quat_rotate(
            handle_pose[:, 3:7], self.handle_center_local[rows]
        )
        axis = _quat_rotate(handle_pose[:, 3:7], self.handle_axis_local[rows])
        axis = fixed_normalize(axis, eps=1e-8)
        delta = points - center[:, None, :]
        axial = fixed_dot(delta, axis[:, None, :])
        radial = delta - axial[..., None] * axis[:, None, :]
        closing_axis = torch.cross(-self.door_normal[rows], axis, dim=-1)
        closing_axis = fixed_normalize(closing_axis, eps=1e-8)
        closing = fixed_dot(radial, closing_axis[:, None, :])
        return (
            torch.isfinite(points).all(dim=(1, 2))
            & (closing[:, 0] * closing[:, 1] < 0.0)
            & torch.all(closing.abs() >= 0.25 * self.handle_radius[rows, None], dim=-1)
        )

    def advance_insert_and_grasp_handle(
        self,
        state: dict[str, torch.Tensor],
        geometry: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Advance CENTER/GRASP and return rows entering PRESS_HANDLE."""
        mask = self.mask(self.Stage.INSERT_AROUND_HANDLE) | self.mask(self.Stage.GRASP_HANDLE)
        if not torch.any(mask):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        rows = torch.where(mask)[0]
        stable, _, _ = self.arm_stability.update(state, rows)
        center = self.stage[rows] == int(self.Stage.INSERT_AROUND_HANDLE)
        if torch.any(center):
            center_rows = rows[center]
            panel_collision = geometry["panel_clearance"][center] < 0.005
            self.fail(center_rows[panel_collision], 10)
            center_position_error = fixed_norm(
                state["ee_pos"][center_rows] - self.arm_forward_desired_target_world[center_rows]
            )
            center_orientation_error = _quat_distance(
                state["ee_orn"][center_rows],
                self.arm_forward_command_quat[center_rows],
            )
            center_gripper_error = (
                state["gripper_position"][center_rows, 0] - self.q_open[center_rows]
            ).abs()
            center_root_speed = fixed_norm(state["root_states"][center_rows, 7:10])
            capture_ready = (
                (geometry["pocket_normal_error"][center].abs() <= 0.008)
                & (geometry["handle_axle_margin"][center] >= 0.020)
                & (geometry["handle_free_margin"][center] >= 0.010)
                & (geometry["channel_vertical_error"][center].abs() <= 0.010)
                & (geometry["grasp_depth"][center] >= 0.015)
                & (geometry["panel_clearance"][center] >= 0.005)
            )
            ready = (
                (center_position_error <= 0.012)
                & (center_orientation_error <= self.grasp_orientation_tolerance[center_rows])
                & (center_gripper_error < 0.02)
                & capture_ready
                & (center_root_speed < 0.02)
                & stable[center]
                & ~panel_collision
            )
            self.record_acceptance(
                self.Stage.INSERT_AROUND_HANDLE,
                center_rows,
                {
                    "ee_position_error_m": (
                        center_position_error <= 0.012,
                        center_position_error,
                        "<=",
                        0.012,
                    ),
                    "ee_orientation_error_rad": (
                        center_orientation_error <= self.grasp_orientation_tolerance[center_rows],
                        center_orientation_error,
                        "<=",
                        self.grasp_orientation_tolerance[center_rows],
                    ),
                    "gripper_open_error_rad": (
                        center_gripper_error < 0.02,
                        center_gripper_error,
                        "<",
                        0.02,
                    ),
                    "pocket_normal_error_m": (
                        geometry["pocket_normal_error"][center].abs() <= 0.008,
                        geometry["pocket_normal_error"][center].abs(),
                        "<=",
                        0.008,
                    ),
                    "handle_axle_margin_m": (
                        geometry["handle_axle_margin"][center] >= 0.020,
                        geometry["handle_axle_margin"][center],
                        ">=",
                        0.020,
                    ),
                    "handle_free_margin_m": (
                        geometry["handle_free_margin"][center] >= 0.010,
                        geometry["handle_free_margin"][center],
                        ">=",
                        0.010,
                    ),
                    "channel_vertical_error_m": (
                        geometry["channel_vertical_error"][center].abs() <= 0.010,
                        geometry["channel_vertical_error"][center].abs(),
                        "<=",
                        0.010,
                    ),
                    "grasp_depth_m": (
                        geometry["grasp_depth"][center] >= 0.015,
                        geometry["grasp_depth"][center],
                        ">=",
                        0.015,
                    ),
                    "panel_clearance_m": (
                        ~panel_collision,
                        geometry["panel_clearance"][center],
                        ">=",
                        0.005,
                    ),
                    "base_linear_speed_m_s": (
                        center_root_speed < 0.02,
                        center_root_speed,
                        "<",
                        0.02,
                    ),
                    "arm_stable": (stable[center], stable[center].to(torch.float32), "==", 1.0),
                },
                required_streak=5,
            )
            settled = consecutive_steps(ready, self.handle_align_settle_steps[center_rows])
            self.handle_align_settle_steps[center_rows] = settled
            to_grasp = center_rows[settled >= 5]
            if len(to_grasp):
                self.transition(to_grasp, self.Stage.GRASP_HANDLE)
                self.grasp_settle_steps[to_grasp] = 0

        grasp_rows = torch.where(self.mask(self.Stage.GRASP_HANDLE))[0]
        turn_rows = torch.empty(0, dtype=torch.int64, device=self.device)
        if len(grasp_rows):
            # Geometry rows correspond to the original combined stage rows.
            lookup = torch.searchsorted(rows, grasp_rows)
            panel_collision = geometry["panel_clearance"][lookup] < 0.005
            self.fail(grasp_rows[panel_collision], 11)
            present, norms, points = self._handle_contacts(state, grasp_rows)
            step = self.stage_steps[grasp_rows, None]
            self.handle_contact_last_seen[grasp_rows] = torch.where(
                present, step, self.handle_contact_last_seen[grasp_rows]
            )
            self.handle_contact_last_points[grasp_rows] = torch.where(
                present[..., None], points, self.handle_contact_last_points[grasp_rows]
            )
            recent = torch.all(step - self.handle_contact_last_seen[grasp_rows] <= 5, dim=-1)
            opposed = self._opposed_grasp(
                state, grasp_rows, self.handle_contact_last_points[grasp_rows]
            )
            newly_latched = (~self.grasp_latched[grasp_rows]) & recent & opposed
            self.grasp_latched[grasp_rows] |= newly_latched
            peak = norms.amax(dim=-1)
            joint_id = int(self.env.gripper_joint_ids[0].item())
            joint_speed = self.control_joint_speed(state, grasp_rows, joint_id)
            velocity_ok = joint_speed < 0.08
            force_ok = peak <= 50.0
            panel_clearance_ok = ~panel_collision
            grasp_stable = (
                self.grasp_latched[grasp_rows]
                & recent
                & opposed
                & velocity_ok
                & force_ok
                & panel_clearance_ok
            )
            self.record_acceptance(
                self.Stage.GRASP_HANDLE,
                grasp_rows,
                {
                    "grasp_latched": (
                        self.grasp_latched[grasp_rows],
                        self.grasp_latched[grasp_rows].to(torch.float32),
                        "==",
                        1.0,
                    ),
                    "bilateral_contact_recent": (recent, recent.to(torch.float32), "==", 1.0),
                    "opposed_contact_geometry": (opposed, opposed.to(torch.float32), "==", 1.0),
                    "gripper_speed_rad_s": (velocity_ok, joint_speed, "<", 0.08),
                    "peak_handle_contact_force_n": (force_ok, peak, "<=", 50.0),
                    "panel_clearance_m": (
                        panel_clearance_ok,
                        geometry["panel_clearance"][lookup],
                        ">=",
                        0.005,
                    ),
                },
                required_streak=10,
            )
            settled = consecutive_steps(grasp_stable, self.grasp_settle_steps[grasp_rows])
            self.grasp_settle_steps[grasp_rows] = settled
            turn_rows = grasp_rows[settled >= 10]
            self.grasp_bilateral_contact_seen[turn_rows] = True

        active = self.mask(self.Stage.INSERT_AROUND_HANDLE) | self.mask(self.Stage.GRASP_HANDLE)
        self.stage_steps[active] += 1
        return turn_rows

    def begin_press_handle(self, state: dict[str, torch.Tensor], rows: torch.Tensor) -> None:
        joints = state["door_joint_position"]
        self.turn_start_handle_angle[rows] = joints[rows, self.handle_joint_idx]
        self.turn_start_hinge_angle[rows] = joints[rows, self.hinge_idx]
        self.turn_target_handle_angle[rows] = torch.maximum(
            0.74 + self.turn_target_margin[rows],
            self.turn_start_handle_angle[rows],
        )
        self.turn_target_hinge_angle[rows] = self.hinge_axis_sign[rows] * (
            self.turn_gap_angle[rows] + math.radians(2.0) + self.turn_gap_margin[rows]
        )
        handle_pose = state["door_body_state_w"][rows, self.handle_body_idx, :7]
        panel_pose = state["door_body_state_w"][rows, self.panel_body_idx, :7]
        axis = _quat_rotate(panel_pose[:, 3:7], self.handle_joint_axis_panel_local[rows])
        axis = fixed_normalize(axis, eps=1e-8)
        self.turn_handle_axis_world[rows] = axis
        self.turn_axle_world[rows] = handle_pose[:, :3]
        current_goal = state.get("ee_goal_cart_world")
        start = state["ee_pos"][rows] if current_goal is None else current_goal[rows]
        current_goal_q = state.get("ee_goal_orn_quat_world")
        start_q = state["ee_orn"][rows] if current_goal_q is None else current_goal_q[rows]
        self.turn_grasp_offset_world[rows] = start - handle_pose[:, :3]
        self.turn_grasp_orientation[rows] = start_q
        self.turn_gap_gripper_angle[rows] = 0.0
        self.turn_release_slow_steps[rows] = 0
        self.transition(rows, self.Stage.PRESS_HANDLE)
        self.stage_steps[rows] = 1
        self._build_handle_rotation_path(rows)

    def _build_handle_rotation_path(self, rows: torch.Tensor) -> None:
        rotation_angle = (
            self.turn_target_handle_angle[rows] - self.turn_start_handle_angle[rows]
        ).clamp_min(0.0)
        endpoint_rotation = _axis_angle_quat(self.turn_handle_axis_world[rows], rotation_angle)
        endpoint = self.turn_axle_world[rows] + _quat_rotate(
            endpoint_rotation, self.turn_grasp_offset_world[rows]
        )
        distance = fixed_norm(
            endpoint - (self.turn_axle_world[rows] + self.turn_grasp_offset_world[rows])
        )
        counts = torch.maximum(
            torch.full_like(self.stage_steps[rows], 120),
            torch.maximum(
                torch.ceil(distance / 0.004).to(torch.int32) + 1,
                torch.ceil(rotation_angle / 0.03).to(torch.int32) + 1,
            ),
        )
        max_count = int(counts.max().item())
        if max_count > self.turn_path.capacity:
            raise RuntimeError(f"turn path length {max_count} exceeds {self.turn_path.capacity}")
        sample = torch.arange(max_count, device=self.device, dtype=torch.float32)
        fraction = (sample[None, :] / (counts[:, None] - 1).clamp_min(1)).clamp(max=1.0)
        angle = fraction * rotation_angle[:, None]
        axis = self.turn_handle_axis_world[rows, None, :].expand(-1, max_count, -1)
        rotation = _axis_angle_quat(axis, angle)
        axle = self.turn_axle_world[rows, None, :]
        offset = self.turn_grasp_offset_world[rows, None, :].expand(-1, max_count, -1)
        path = axle + _quat_rotate(rotation.reshape(-1, 4), offset.reshape(-1, 3)).reshape(
            len(rows), max_count, 3
        )
        orientation = _quat_multiply(
            rotation,
            self.turn_grasp_orientation[rows, None, :].expand(-1, max_count, -1),
        )
        self.turn_path.positions[rows, :max_count] = path
        self.turn_path.orientations[rows, :max_count] = orientation
        self.turn_path.length[rows] = counts
        self.turn_path.index[rows] = 1

    def _build_turn_gap_path(self, state: dict[str, torch.Tensor], rows: torch.Tensor) -> None:
        current_goal = state.get("ee_goal_cart_world")
        start = state["ee_pos"][rows] if current_goal is None else current_goal[rows]
        current_goal_q = state.get("ee_goal_orn_quat_world")
        start_q = state["ee_orn"][rows] if current_goal_q is None else current_goal_q[rows]
        self.turn_gap_gripper_angle[rows] = _signed_quat_rotation_about_axis(
            self.turn_grasp_orientation[rows],
            start_q,
            self.turn_handle_axis_world[rows],
        )
        current_hinge = state["door_joint_position"][rows, self.hinge_idx]
        current_hinge_delta = self.hinge_axis_sign[rows] * (
            current_hinge - self.turn_start_hinge_angle[rows]
        )
        remaining_hinge = self.turn_target_hinge_angle[rows] - current_hinge_delta
        endpoint = _rotate_about_z(start, self.hinge_world[rows], remaining_hinge)
        distance = fixed_norm(endpoint - start)
        angle = remaining_hinge.abs()
        counts = torch.maximum(
            torch.full_like(self.stage_steps[rows], 120),
            torch.maximum(
                torch.ceil(distance / 0.004).to(torch.int32) + 1,
                torch.ceil(angle / 0.03).to(torch.int32) + 1,
            ),
        )
        max_count = int(counts.max().item())
        if max_count > self.turn_path.capacity:
            raise RuntimeError(f"turn path length {max_count} exceeds {self.turn_path.capacity}")
        sample = torch.arange(max_count, device=self.device, dtype=torch.float32)
        fraction = sample[None, :] / (counts[:, None] - 1).clamp_min(1)
        fraction = fraction.clamp(max=1.0)
        hinge_rotation = fraction * remaining_hinge[:, None]
        path = _rotate_about_z(
            start[:, None, :].expand(-1, max_count, -1),
            self.hinge_world[rows, None, :],
            hinge_rotation,
        )
        orientation = _quat_multiply(
            _yaw_quat(hinge_rotation),
            start_q[:, None, :].expand(-1, max_count, -1),
        )
        self.turn_path.positions[rows, :max_count] = path
        self.turn_path.orientations[rows, :max_count] = orientation
        self.turn_path.length[rows] = counts
        self.turn_path.index[rows] = 1

    def command_press_handle(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.PRESS_HANDLE))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows, buffers)
        self.arm_trajectory(rows, self.turn_path, buffers)
        self.gripper_target(rows, self.q_closed[rows], buffers)

    def advance_press_handle(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.PRESS_HANDLE))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        latch_released = (
            state["door_joint_position"][rows, self.latch_joint_idx] >= PRESS_HANDLE_LATCH_RELEASE_M
        )
        latch_position = state["door_joint_position"][rows, self.latch_joint_idx]
        self.record_acceptance(
            self.Stage.PRESS_HANDLE,
            rows,
            {
                "latch_release_m": (
                    latch_released,
                    latch_position,
                    ">=",
                    PRESS_HANDLE_LATCH_RELEASE_M,
                )
            },
        )
        self.stage_steps[rows] += 1
        return rows[latch_released]

    def begin_open_door_with_handle(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor
    ) -> None:
        self._build_turn_gap_path(state, rows)
        self.turn_release_slow_steps[rows] = 0
        self.transition(rows, self.Stage.OPEN_DOOR_WITH_HANDLE)
        self.stage_steps[rows] = 1

    def command_open_door_with_handle(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.OPEN_DOOR_WITH_HANDLE))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows, buffers)
        self.arm_trajectory(rows, self.turn_path, buffers)
        self.gripper_target(rows, self.q_closed[rows], buffers)

    def advance_open_door_with_handle(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.OPEN_DOOR_WITH_HANDLE))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        joint = state["door_joint_position"]
        latch_released = joint[rows, self.latch_joint_idx] >= 0.02
        hinge = joint[rows, self.hinge_idx]
        gap_open = hinge.abs() >= self.turn_gap_angle[rows]
        hinge_speed = state["door_joint_velocity"][rows, self.hinge_idx].abs()
        slow = hinge_speed < self.turn_release_max_hinge_speed[rows]
        qualifying = latch_released & gap_open & slow
        settled = consecutive_steps(qualifying, self.turn_release_slow_steps[rows])
        self.turn_release_slow_steps[rows] = settled
        present, _, _ = self._handle_contacts(state, rows)
        retained = self.grasp_bilateral_contact_seen[rows] & torch.any(present, dim=-1)
        self.record_acceptance(
            self.Stage.OPEN_DOOR_WITH_HANDLE,
            rows,
            {
                "latch_released": (latch_released, joint[rows, self.latch_joint_idx], ">=", 0.02),
                "door_gap_angle_rad": (gap_open, hinge.abs(), ">=", self.turn_gap_angle[rows]),
                "door_hinge_speed_rad_s": (
                    slow,
                    hinge_speed,
                    "<",
                    self.turn_release_max_hinge_speed[rows],
                ),
                "release_slow_window": (
                    settled >= self.turn_release_slow_steps_required[rows],
                    settled,
                    ">=frames",
                    self.turn_release_slow_steps_required[rows],
                ),
                "handle_contact_retained": (retained, retained.to(torch.float32), "==", 1.0),
            },
        )
        complete = qualifying & (settled >= self.turn_release_slow_steps_required[rows]) & retained
        self.stage_steps[rows] += 1
        return rows[complete]

    def begin_reset_handle(self, state: dict[str, torch.Tensor], rows: torch.Tensor) -> None:
        handle_pose = state["door_body_state_w"][rows, self.handle_body_idx, :7]
        panel_pose = state["door_body_state_w"][rows, self.panel_body_idx, :7]
        axis = _quat_rotate(panel_pose[:, 3:7], self.handle_joint_axis_panel_local[rows])
        axis = fixed_normalize(axis, eps=1e-8)
        previous_index = (self.turn_path.index[rows] - 1).clamp_min(0).to(torch.int64)
        start = self.turn_path.positions[rows, previous_index]
        start_q = self.turn_path.orientations[rows, previous_index]
        self.release_return_axle_world[rows] = handle_pose[:, :3]
        self.release_return_axis_world[rows] = axis
        self.release_return_offset_world[rows] = start - handle_pose[:, :3]
        self.release_return_start_quat[rows] = start_q
        self.release_return_angle[rows] = -self.turn_gap_gripper_angle[rows]
        self.release_vertical_delta[rows] = -self.grasp_vertical_lift[rows]
        self.release_stable_steps[rows] = 0
        self.turn_path.clear(rows)
        counts = self.release_return_steps[rows].to(torch.int32) + 1
        max_count = int(counts.max().item())
        sample = torch.arange(max_count, device=self.device, dtype=torch.float32)
        fraction = (sample[None, :] / (counts[:, None] - 1).clamp_min(1)).clamp(max=1.0)
        delta = fraction * self.release_return_angle[rows, None]
        rotation = _axis_angle_quat(
            self.release_return_axis_world[rows, None, :].expand(-1, max_count, -1),
            delta,
        )
        self.turn_path.positions[rows, :max_count] = self.release_return_axle_world[
            rows, None, :
        ] + _quat_rotate(
            rotation.reshape(-1, 4),
            self.release_return_offset_world[rows, None, :]
            .expand(-1, max_count, -1)
            .reshape(-1, 3),
        ).reshape(len(rows), max_count, 3)
        self.turn_path.orientations[rows, :max_count] = _quat_multiply(
            rotation,
            self.release_return_start_quat[rows, None, :].expand(-1, max_count, -1),
        )
        self.turn_path.length[rows] = counts
        self.turn_path.index[rows] = 1
        self.transition(rows, self.Stage.RESET_HANDLE)
        self.stage_steps[rows] = 1

    def command_reset_handle(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.RESET_HANDLE))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows, buffers)
        self.arm_trajectory(rows, self.turn_path, buffers)
        self.gripper_target(rows, self.q_closed[rows], buffers)

    def advance_reset_handle(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.RESET_HANDLE))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        horizontal = state["door_joint_position"][rows, self.handle_joint_idx].abs() < 0.05
        path_complete = self.turn_path.complete(rows)
        handle_angle = state["door_joint_position"][rows, self.handle_joint_idx].abs()
        complete = path_complete & horizontal
        self.record_acceptance(
            self.Stage.RESET_HANDLE,
            rows,
            {
                "path_complete": (path_complete, path_complete.to(torch.float32), "==", 1.0),
                "handle_angle_abs_rad": (horizontal, handle_angle, "<", 0.05),
            },
            required_streak=10,
        )
        settled = consecutive_steps(complete, self.release_stable_steps[rows])
        self.release_stable_steps[rows] = settled
        self.stage_steps[rows] += 1
        return rows[settled >= 10]

    def command_release_handle(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.RELEASE_HANDLE))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows, buffers)
        self.arm_trajectory(rows, self.release_open_path, buffers)
        fraction = (self.stage_steps[rows].to(torch.float32) / 35.0).clamp(0.0, 1.0)
        self.gripper_target(
            rows, (1.0 - fraction) * self.q_closed[rows] + fraction * self.q_open[rows], buffers
        )

    def _release_blocking_handle_contacts(self, state, rows):
        present, _, points = self._handle_contacts(state, rows)
        blocking = present.clone()
        local_shaft = torch.zeros((len(rows), 3), device=self.device)
        local_shaft[:, 0] = 1.0
        shaft = _quat_rotate(state["ee_orn"][rows], local_shaft)
        retreat = 0.10 * self.door_normal[rows]
        retreat[:, 2] += 0.02
        for finger, body_index in enumerate(self.release_contact_body_indices):
            rear = separating_rear_contact(
                points[:, finger],
                state["robot_body_state_w"][rows, body_index, :7],
                self.release_contact_triangles[finger],
                shaft,
                retreat,
            )
            blocking[:, finger] &= ~rear
        return blocking

    def advance_release_handle(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.RELEASE_HANDLE))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        blocking = self._release_blocking_handle_contacts(state, rows)
        contact_clear = ~torch.any(blocking, dim=-1)
        gripper_open = (state["gripper_position"][rows, 0] - self.q_open[rows]).abs() < 0.08
        gripper_error = (state["gripper_position"][rows, 0] - self.q_open[rows]).abs()
        self.record_acceptance(
            self.Stage.RELEASE_HANDLE,
            rows,
            {
                "fixed_jaw_contact_clear": (
                    ~blocking[:, 0],
                    (~blocking[:, 0]).to(torch.float32),
                    "==",
                    1.0,
                ),
                "moving_jaw_contact_clear": (
                    ~blocking[:, 1],
                    (~blocking[:, 1]).to(torch.float32),
                    "==",
                    1.0,
                ),
                "gripper_open_error_rad": (gripper_open, gripper_error, "<", 0.08),
            },
            required_streak=10,
        )
        settled = consecutive_steps(contact_clear & gripper_open, self.release_stable_steps[rows])
        self.release_stable_steps[rows] = settled
        self.stage_steps[rows] += 1
        return rows[settled >= 10]

    def command_withdraw_from_handle(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.WITHDRAW_FROM_HANDLE))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows, buffers)
        self.arm_trajectory(rows, self.release_retreat_path, buffers)
        self.gripper_target(rows, self.q_open[rows], buffers)

    def advance_withdraw_from_handle(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.WITHDRAW_FROM_HANDLE))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        present, _, _ = self._handle_contacts(state, rows)
        path_complete = self.release_retreat_path.complete(rows)
        position_error = fixed_norm(state["ee_pos"][rows] - self.release_retreat_target_world[rows])
        contact_clear = ~torch.any(present, dim=-1)
        complete = path_complete & (position_error <= 0.010) & contact_clear
        self.record_acceptance(
            self.Stage.WITHDRAW_FROM_HANDLE,
            rows,
            {
                "path_complete": (path_complete, path_complete.to(torch.float32), "==", 1.0),
                "ee_position_error_m": (position_error <= 0.010, position_error, "<=", 0.010),
                "handle_contact_clear": (contact_clear, contact_clear.to(torch.float32), "==", 1.0),
            },
            required_streak=10,
        )
        settled = consecutive_steps(complete, self.release_stable_steps[rows])
        self.release_stable_steps[rows] = settled
        self.stage_steps[rows] += 1
        return rows[settled >= 10]

    def command_stow_arm_and_close_gripper(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(
            self.mask(self.Stage.STOW_ARM_AND_CLOSE_GRIPPER)
            | self.mask(self.Stage.STOW_ARM_AFTER_PULL)
        )[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows, buffers)
        self.arm_trajectory(rows, self.release_restore_path, buffers)
        fraction = (self.stage_steps[rows].to(torch.float32) / 50.0).clamp(0.0, 1.0)
        self.gripper_target(
            rows, (1.0 - fraction) * self.q_open[rows] + fraction * self.q_closed[rows], buffers
        )
        pulling = self.stage[rows] == int(self.Stage.STOW_ARM_AFTER_PULL)
        self.gripper_target(rows[pulling], self.q_closed[rows[pulling]], buffers)

    def advance_stow_arm_and_close_gripper(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(
            (
                self.mask(self.Stage.STOW_ARM_AND_CLOSE_GRIPPER)
                | self.mask(self.Stage.STOW_ARM_AFTER_PULL)
            )
            & (self.stage_steps > 0)
        )[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        stable, _, _ = self.arm_stability.update(state, rows)
        gripper_error = (state["gripper_position"][rows, 0] - self.q_closed[rows]).abs()
        orientation_error = _quat_distance(state["ee_orn"][rows], self.release_endpoint_quat[rows])
        position_error = fixed_norm(state["ee_pos"][rows] - self.release_endpoint_world[rows])
        restored = (
            (self.stage_steps[rows] >= 50)
            & (gripper_error < 0.03)
            & (orientation_error < 0.05)
            & (position_error <= 0.010)
            & stable
        )
        conditions = {
            "minimum_stage_steps": (self.stage_steps[rows] >= 50, self.stage_steps[rows], ">=", 50),
            "gripper_closed_error_rad": (gripper_error < 0.03, gripper_error, "<", 0.03),
            "ee_orientation_error_rad": (orientation_error < 0.05, orientation_error, "<", 0.05),
            "ee_position_error_m": (position_error <= 0.010, position_error, "<=", 0.010),
            "arm_stable": (stable, stable.to(torch.float32), "==", 1.0),
        }
        for stage in (self.Stage.STOW_ARM_AND_CLOSE_GRIPPER, self.Stage.STOW_ARM_AFTER_PULL):
            selected = self.stage[rows] == int(stage)
            stage_rows = rows[selected]
            if len(stage_rows):
                stage_conditions = {
                    key: (p[selected], v[selected], op, t)
                    for key, (p, v, op, t) in conditions.items()
                }
                self.record_acceptance(stage, stage_rows, stage_conditions, required_streak=10)
        settled = consecutive_steps(restored, self.release_stable_steps[rows])
        self.release_stable_steps[rows] = settled
        completed = rows[settled >= 10]
        pull_completed = completed[self.stage[completed] == int(self.Stage.STOW_ARM_AFTER_PULL)]
        if len(pull_completed):
            self.begin_follow_wall_to_panel(state, pull_completed)
        completed = completed[self.stage[completed] == int(self.Stage.STOW_ARM_AND_CLOSE_GRIPPER)]
        if len(completed):
            if self.terminal_stage == self.Stage.STOW_ARM_AND_CLOSE_GRIPPER:
                self.transition(completed, self.Stage.SUCCESS)
            elif getattr(self, "pull_door", False):
                self.begin_move_to_gap_station(state, completed)
            else:
                arm_ids = self.env.arm_joint_ids.to(device=self.device, dtype=torch.int64)
                self.align_panel_arm_joint_target[completed] = state["dof_pos"][
                    completed[:, None], arm_ids[None, :]
                ]
                self.panel_align_ee_goal_local[completed] = state["ee_goal_cart_local"][completed]
                reference = self.env.get_goal_reference_quat()[completed]
                self.panel_align_ee_orientation_local[completed] = _quat_multiply(
                    _quat_conjugate(reference), state["ee_orn"][completed]
                )
                self.reset_base_arrival(completed, ALIGN_PANEL_ARRIVAL)
                self.transition(completed, self.Stage.MOVE_TO_PANEL_STATION)
                self.stage_steps[completed] = 1
        active = self.mask(self.Stage.STOW_ARM_AND_CLOSE_GRIPPER) | self.mask(
            self.Stage.STOW_ARM_AFTER_PULL
        )
        self.stage_steps[active] += 1
        return completed

    def begin_move_to_gap_station(self, state, rows):
        # The stage-entry planning event selects a path-validated station before
        # the next command; hold the current station until it is installed.
        target = state["root_states"][rows, :2]
        yaw = _yaw_from_wxyz(state["root_states"][rows, 3:7])
        retreat = target - 0.20 * torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        self.pull_base_entry_pose[rows] = torch.cat((target, yaw[:, None]), dim=-1)
        self.pull_base_trajectory.clear(rows)
        self.pull_base_path[rows, 0] = torch.cat((retreat, yaw[:, None]), dim=-1)
        self.pull_base_path_length[rows] = 1
        self.pull_base_path_index[rows] = 0
        self.pull_station_xy[rows], self.pull_station_yaw[rows] = target, yaw
        self.pull_frame_clear_steps[rows] = 0
        arm_ids = self.env.arm_joint_ids.to(device=self.device, dtype=torch.int64)
        self.align_panel_arm_joint_target[rows] = state["dof_pos"][rows[:, None], arm_ids[None, :]]
        self.reset_base_arrival(rows, APPROACH_ARRIVAL)
        self.transition(rows, self.Stage.MOVE_TO_GAP_STATION)
        self.stage_steps[rows] = 1

    def install_initial_pull_station(self, rows, target, yaw, gap, clearance, wall_clearance):
        self.pull_initial_station_xy[rows], self.pull_initial_station_yaw[rows] = target, yaw
        self.pull_base_path[rows, 1] = torch.cat((target, yaw[:, None]), dim=-1)
        self.pull_base_path_length[rows] = 2
        self.pull_base_trajectory.install(
            rows, self.pull_base_entry_pose[rows], self.pull_base_path[rows], self.dt
        )
        self.pull_station_xy[rows], self.pull_station_yaw[rows] = target, yaw
        self.pull_gap_xy[rows], self.pull_planned_frame_clearance[rows] = gap, clearance
        self.pull_planned_wall_clearance[rows] = wall_clearance

    def sample_initial_pull_stations(self, rows):
        visits = self.stage_visit[rows, int(self.Stage.MOVE_TO_GAP_STATION)].to(torch.int64)
        return _pull_cycle_uniform(
            self.pull_cycle_seeds[rows],
            visits + 0x53544154,
            self.pull_station_config.first_candidate_count,
        )[..., :3]

    def command_gap_or_wall_station(self, state, buffers):
        if not self.pull_door:
            return
        following = torch.where(self.mask(self.Stage.FOLLOW_WALL_TO_PANEL))[0]
        if len(following):
            self.update_pull_panel_station(state, following)
        rows = torch.where(
            self.mask(self.Stage.MOVE_TO_GAP_STATION) | self.mask(self.Stage.FOLLOW_WALL_TO_PANEL)
        )[0]
        if not len(rows):
            return
        xy, yaw = self.pull_station_xy[rows], self.pull_station_yaw[rows]
        reference = torch.cat((xy, yaw[:, None]), dim=-1)
        velocity = torch.zeros_like(reference)
        path_complete = torch.ones(len(rows), dtype=torch.bool, device=self.device)
        initial = self.stage[rows] == int(self.Stage.MOVE_TO_GAP_STATION)
        initial_rows = rows[initial]
        if len(initial_rows):
            pose, feedforward, complete = self.pull_base_trajectory.sample(initial_rows)
            reference[initial], velocity[initial], path_complete[initial] = (
                pose,
                feedforward,
                complete,
            )
        follow_rows = rows[~initial]
        if len(follow_rows):
            velocity[~initial] = self.pull_follow_velocity[follow_rows]
            path_complete[~initial] = self.pull_follow_holding[follow_rows]
        stability_kwargs = {}
        if len(following):
            door_slow = (
                state["door_joint_velocity"][rows, self.hinge_idx].abs()
                < self.turn_release_max_hinge_speed[rows]
            )
            stability_kwargs["stability_condition"] = (
                self.stage[rows] != int(self.Stage.FOLLOW_WALL_TO_PANEL)
            ) | door_slow
        command, distance, yaw_error = self._track_station(
            state,
            rows,
            reference[:, :2],
            reference[:, 2],
            xy,
            yaw,
            reference_velocity_world=velocity,
            arrival_enabled=path_complete,
            **stability_kwargs,
            buffers=buffers,
        )
        self.pending_station_distance[rows] = distance
        self.pending_yaw_error[rows] = yaw_error
        self.base_hold_target_xy[rows], self.base_hold_nominal_xy[rows] = xy, xy
        self.base_hold_target_yaw[rows], self.base_hold_nominal_yaw[rows] = yaw, yaw
        self.arm_joint_target(rows, self.align_panel_arm_joint_target[rows], buffers)
        self.gripper_target(rows, self.q_closed[rows], buffers)

    def advance_move_to_gap_station(self, state):
        if not self.pull_door:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        rows = torch.where(self.mask(self.Stage.MOVE_TO_GAP_STATION))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        path_complete = (
            self.pull_base_trajectory.index[rows] >= self.pull_base_trajectory.length[rows] - 1
        )
        root = state["root_states"][rows]
        points = (
            _quat_rotate_points_torch(
                root[:, 3:7], self.pull_base_points[None, :, :].expand(len(rows), -1, -1)
            )
            + root[:, None, :3]
        )
        clearances = base_projection_clearances(
            self.hinge_world[rows, :2],
            torch.stack((self.pull_free_direction[rows], self.door_normal[rows, :2]), dim=1),
            torch.stack((self.pull_frame_edge[rows], self.pull_wall_front[rows]), dim=1),
            points[:, :, :2],
        )
        clearance, wall_clearance = clearances[:, 0], clearances[:, 1]
        frame_safe = clearance >= self.pull_station_config.clearance_m
        wall_safe = wall_clearance >= self.pull_station_config.wall_clearance_m
        safe = frame_safe & wall_safe & path_complete
        self.pull_frame_clear_steps[rows] = torch.where(
            safe & (self.base_arrival_settle_steps[rows] > 0),
            (self.pull_frame_clear_steps[rows] + 1).clamp(max=APPROACH_ARRIVAL.settle_frames),
            0,
        )
        clear_window = self.pull_frame_clear_steps[rows] >= APPROACH_ARRIVAL.settle_frames
        self.record_acceptance(
            self.Stage.MOVE_TO_GAP_STATION,
            rows,
            {
                "base_path_complete": (
                    path_complete,
                    self.pull_base_trajectory.index[rows],
                    ">=",
                    self.pull_base_trajectory.length[rows] - 1,
                ),
                "base_arrival_confirmed": (
                    self.base_arrival_confirmed[rows],
                    self.base_arrival_settle_steps[rows],
                    ">=frames",
                    self.base_arrival_required_frames[rows],
                ),
                "base_stability_confirmed": (
                    self.base_stability_confirmed[rows],
                    self.base_stability_steps[rows],
                    ">=frames",
                    self.base_arrival_required_frames[rows],
                ),
                "base_frame_projection_clearance_m": (
                    frame_safe,
                    clearance,
                    ">=",
                    self.pull_station_config.clearance_m,
                ),
                "base_wall_clearance_m": (
                    wall_safe,
                    wall_clearance,
                    ">=",
                    self.pull_station_config.wall_clearance_m,
                ),
                "frame_clear_stable_frames": (
                    clear_window,
                    self.pull_frame_clear_steps[rows],
                    ">=",
                    APPROACH_ARRIVAL.settle_frames,
                ),
            },
        )
        completed = rows[
            clear_window & self.base_arrival_confirmed[rows] & self.base_stability_confirmed[rows]
        ]
        if len(completed):
            if self.terminal_stage == self.Stage.MOVE_TO_GAP_STATION:
                self.transition(completed, self.Stage.SUCCESS)
            else:
                self.transition(completed, self.Stage.INSERT_INTO_DOOR_GAP)
                self.arm_stability.clear(completed)
                self.pull_insert_stable_steps[completed] = 0
        self.stage_steps[rows[self.stage[rows] == int(self.Stage.MOVE_TO_GAP_STATION)]] += 1
        return completed

    def begin_follow_wall_to_panel(self, state, rows):
        opened = (
            state["door_joint_position"][rows, self.hinge_idx]
            >= self.pull_panel_config.opening_angle_rad
        ) | self.pull_door_opened[rows]
        self.pull_door_opened[rows] |= opened
        if torch.any(opened):
            self.begin_pass_through_doorway(rows[opened], rolling=True)
        rows = rows[~opened]
        if not len(rows):
            return
        self.pending_station_distance[rows] = torch.inf
        self.pending_yaw_error[rows] = torch.inf
        root = state["root_states"][rows]
        shoulders = state["robot_body_state_w"][
            rows[:, None], self.front_shoulder_body_indices[None, :], :3
        ]
        distances = (
            fixed_dot(
                shoulders[:, :, :2] - self.hinge_world[rows, None, :2],
                self.door_normal[rows, None, :2],
            )
            - self.pull_wall_front[rows, None]
        )
        nearest = distances.abs().argmin(-1)
        owner = torch.arange(len(rows), device=self.device)
        shoulder = shoulders[owner, nearest]
        # Freeze the shoulder's wall-normal coordinate, not the trunk center.
        self.pull_shoulder_anchor[rows] = shoulder[:, :2]
        local = _quat_rotate_inverse(root[:, 3:7], shoulder - root[:, :3])
        self.pull_shoulder_local[rows] = local[:, :2]
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        normal = _quat_rotate(
            _yaw_quat(self.hinge_axis_sign[rows] * hinge), self.door_normal[rows]
        )[:, :2]
        panel_distance = fixed_dot(shoulder[:, :2] - self.hinge_world[rows, :2], normal)
        self.pull_panel_side[rows] = torch.where(panel_distance >= 0.0, 1.0, -1.0)
        yaw = _yaw_from_wxyz(root[:, 3:7])
        width = (self.pull_base_points[:, 1].amax() - self.pull_base_points[:, 1].amin()).expand(
            len(rows)
        )
        wall_normal = self.door_normal[rows, :2]
        free_direction = self.pull_free_direction[rows]
        hinge_left = (
            wall_normal[:, 0] * free_direction[:, 1] - wall_normal[:, 1] * free_direction[:, 0]
        ) > 0.0
        left_radius, right_radius = self.pull_station_config.shoulder_hinge_radius_m
        shoulder_radius = torch.where(hinge_left, left_radius, right_radius)
        route = shoulder_wall_curve(
            self.pull_shoulder_anchor[rows],
            self.pull_shoulder_local[rows],
            self.hinge_world[rows, :2],
            self.pull_free_direction[rows],
            self.door_normal[rows, :2],
            yaw,
            width,
            self.pull_station_config.panel_side_clearance_m,
            shoulder_hinge_radius_m=shoulder_radius,
        )
        start = torch.cat((root[:, :2], yaw[:, None]), dim=-1)
        self.pull_base_trajectory.install(rows, start, route, self.dt)
        self.pull_station_xy[rows], self.pull_station_yaw[rows] = root[:, :2], yaw
        self.pull_follow_holding[rows] = False
        self.pull_follow_velocity[rows] = 0.0
        arm_ids = self.env.arm_joint_ids.to(device=self.device, dtype=torch.int64)
        self.align_panel_arm_joint_target[rows] = state["dof_pos"][rows[:, None], arm_ids[None]]
        self.reset_base_arrival(rows, APPROACH_ARRIVAL)
        self.transition(rows, self.Stage.FOLLOW_WALL_TO_PANEL)
        self.stage_steps[rows] = 1

    def update_pull_panel_station(self, state, rows):
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        finished = (hinge >= self.pull_panel_config.opening_angle_rad) | self.pull_door_opened[rows]
        self.pull_door_opened[rows] = finished
        if torch.any(finished):
            self.begin_pass_through_doorway(rows[finished], rolling=True)
        rows = rows[~finished]
        if not len(rows):
            return
        index = self.pull_base_trajectory.index[rows]
        yaw = self.pull_base_trajectory.poses[rows, index, 2]
        beta = angle_to_wall(yaw, self.pull_free_direction[rows], self.door_normal[rows, :2])
        advance = beta < state["door_joint_position"][rows, self.hinge_idx]
        reference, velocity, complete = self.pull_base_trajectory.sample(rows, advance=advance)
        self.pull_station_xy[rows], self.pull_station_yaw[rows] = reference[:, :2], reference[:, 2]
        self.pull_follow_beta[rows] = beta
        self.pull_follow_holding[rows] = ~advance | complete
        self.pull_follow_velocity[rows] = velocity

    def advance_follow_wall_to_panel(self, state):
        empty = torch.empty(0, device=self.device, dtype=torch.int64)
        if not self.pull_door:
            return empty
        rows = torch.where(self.mask(self.Stage.FOLLOW_WALL_TO_PANEL))[0]
        if not len(rows):
            return empty
        # Opening can finish while tracking the moving panel.
        opened = (
            state["door_joint_position"][rows, self.hinge_idx]
            >= self.pull_panel_config.opening_angle_rad
        ) | self.pull_door_opened[rows]
        self.pull_door_opened[rows] |= opened
        if torch.any(opened):
            self.begin_pass_through_doorway(rows[opened], rolling=True)
        rows = rows[~opened]
        if not len(rows):
            return empty
        ready = (
            self.base_arrival_confirmed[rows]
            & self.base_stability_confirmed[rows]
            & self.pull_follow_holding[rows]
        )
        self.record_acceptance(
            self.Stage.FOLLOW_WALL_TO_PANEL,
            rows,
            {
                "door_speed_rad_s": (
                    state["door_joint_velocity"][rows, self.hinge_idx].abs()
                    < self.turn_release_max_hinge_speed[rows],
                    state["door_joint_velocity"][rows, self.hinge_idx].abs(),
                    "<",
                    self.turn_release_max_hinge_speed[rows],
                ),
                "base_arrival_confirmed": (
                    self.base_arrival_confirmed[rows],
                    self.base_arrival_settle_steps[rows],
                    ">=frames",
                    self.base_arrival_required_frames[rows],
                ),
                "base_stability_confirmed": (
                    self.base_stability_confirmed[rows],
                    self.base_stability_steps[rows],
                    ">=frames",
                    self.base_arrival_required_frames[rows],
                ),
                "trajectory_gate_holding": (
                    self.pull_follow_holding[rows],
                    self.pull_follow_beta[rows],
                    ">=",
                    state["door_joint_position"][rows, self.hinge_idx],
                ),
            },
        )
        insert = rows[ready & (self.stage[rows] == int(self.Stage.FOLLOW_WALL_TO_PANEL))]
        if len(insert):
            if self.terminal_stage == self.Stage.FOLLOW_WALL_TO_PANEL:
                self.transition(insert, self.Stage.SUCCESS)
            else:
                self.panel_contact_normal_sign[insert] = self.pull_panel_side[insert]
                self.ee_command_world[insert] = state["ee_pos"][insert]
                self.ee_orientation_command[insert] = state["ee_orn"][insert]
                self.ee_command_valid[insert] = True
                self.panel_contact_plan_valid[insert] = False
                self.panel_contact_settle_steps[insert] = 0
                self.transition(insert, self.Stage.MOVE_TO_PANEL_STANDOFF)
        self.stage_steps[rows[~ready]] += 1
        return insert

    def prepare_pull_cycle_samples(self, controllers):
        """Extract static episode seeds once; no per-visit cache or cycle bound."""
        if len(controllers) != self.num_envs:
            raise ValueError("pull seed batch length disagrees with environments")
        self.pull_cycle_seeds = torch.tensor(
            [controller.target_noise._seed & 0x7FFFFFFFFFFFFFFF for controller in controllers],
            dtype=torch.int64,
            device=self.device,
        )
        config = self.pull_panel_config
        height_offsets = torch.tensor(
            config.height_offsets_m, dtype=torch.float64, device=self.device
        )
        insertion = torch.tensor(
            config.insertion_lengths_m, dtype=torch.float64, device=self.device
        )
        grid = torch.cartesian_prod(height_offsets, insertion)
        center_insertion = (min(config.insertion_lengths_m) + max(config.insertion_lengths_m)) / 2.0
        distance = grid[:, 0].square() + (grid[:, 1] - center_insertion).square()
        grid = grid[torch.argsort(distance, stable=True)]
        grid[:, 0] += config.height_nominal_m
        self.pull_cycle_pose_grid = grid.to(torch.float32)
        self.pull_cycle_sample_lower = torch.tensor(
            (
                config.height_nominal_m + min(config.height_offsets_m),
                min(config.insertion_lengths_m),
                0.0,
                -config.station_angle_noise_rad,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.pull_cycle_sample_span = torch.tensor(
            (
                max(config.height_offsets_m) - min(config.height_offsets_m),
                max(config.insertion_lengths_m) - min(config.insertion_lengths_m),
                0.0,
                2.0 * config.station_angle_noise_rad,
            ),
            dtype=torch.float32,
            device=self.device,
        )

    def sample_pull_cycle_candidates(self, rows):
        """Return [B, candidate_count, 4] and increment each requested cycle once."""
        cycles = self.pull_cycle[rows].to(torch.int64)
        config = self.pull_panel_config
        uniform = _pull_cycle_uniform(self.pull_cycle_seeds[rows], cycles, config.candidate_count)
        samples = self.pull_cycle_sample_lower + uniform * self.pull_cycle_sample_span
        samples[..., :2] = self.pull_cycle_pose_grid
        self.pull_cycle[rows] += 1
        return samples

    def record_pull_cycle(self, rows, record):
        """Record batch-first diagnostic tensors, including candidate dimensions."""
        self.pull_cycle_history.append_batch(rows, record)

    def install_pull_insert_tensor(
        self,
        rows,
        positions,
        orientations,
        lengths,
        contact_points,
        orientation,
        face_offset_local,
        *,
        endpoint=None,
        tip_radius=None,
        ee_tangent_radius=None,
        **metadata,
    ):
        """Install selected inverse paths and desired geometry entirely on-device.

        ``endpoint`` is the desired endpoint, not the inverse command endpoint.
        Pass it when inverse commands differ from the geometric target. Extra
        metadata is diagnostic only and never changes the controller's gates.
        """
        rows = torch.as_tensor(rows, device=self.device, dtype=torch.int64)
        self.pull_insert_path.install_tensor(rows, positions, orientations, lengths)
        if endpoint is None:
            endpoint = positions[
                torch.arange(len(rows), device=self.device), lengths.to(torch.int64) - 1
            ]
        self.pull_insert_goal[rows] = endpoint
        self.pull_insert_orientation[rows] = orientation
        self.pull_face_offset_local[rows] = face_offset_local
        self.pull_panel_contact[rows] = contact_points

    def command_insert_into_door_gap(self, state, buffers):
        if not self.pull_door:
            return
        rows = torch.where(self.mask(self.Stage.INSERT_INTO_DOOR_GAP))[0]
        if not len(rows):
            return
        self._base_hold_command(state, rows, buffers)
        position, orientation = self.pull_insert_path.current(rows, advance=True)
        self.ee_command_world[rows] = position
        self.ee_orientation_command[rows] = orientation
        self.arm_pose_target(rows, position, orientation, buffers)
        self.gripper_target(rows, self.q_closed[rows], buffers)

    def advance_insert_into_door_gap(self, state):
        empty = torch.empty(0, device=self.device, dtype=torch.int64)
        if not self.pull_door:
            return empty
        rows = torch.where(self.mask(self.Stage.INSERT_INTO_DOOR_GAP))[0]
        if not len(rows):
            return empty
        position_error = fixed_norm(state["ee_pos"][rows] - self.pull_insert_goal[rows])
        orientation_error = _quat_distance(
            state["ee_orn"][rows], self.pull_insert_orientation[rows]
        )
        stable = self.arm_stability.update(state, rows)[0]
        face = state["ee_pos"][rows] + _quat_rotate(
            state["ee_orn"][rows], self.pull_face_offset_local[rows]
        )
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        normal = _quat_rotate(_yaw_quat(self.hinge_axis_sign[rows] * hinge), self.door_normal[rows])
        gap = fixed_dot(self.pull_panel_contact[rows] - face, normal)
        base_speed = fixed_norm(state["root_states"][rows, 7:10])
        gripper_error = (state["gripper_position"][rows, 0] - self.q_closed[rows]).abs()
        first_entry = self.pull_cycle[rows] == 1
        gap_ok = first_entry | ((gap >= -0.001) & (gap <= 0.010))
        ready = (
            (position_error <= 0.012)
            & (orientation_error < 0.05)
            & stable
            & self.pull_insert_path.complete(rows)
            & gap_ok
            & (base_speed < 0.02)
            & (gripper_error < 0.03)
        )
        conditions = {
            "path_complete": (
                self.pull_insert_path.complete(rows),
                self.pull_insert_path.index[rows],
                ">=",
                self.pull_insert_path.length[rows] - 1,
            ),
            "ee_position_error_m": (position_error <= 0.012, position_error, "<=", 0.012),
            "ee_orientation_error_rad": (orientation_error < 0.05, orientation_error, "<", 0.05),
            "arm_stable": (stable, stable.to(torch.float32), "==", 1.0),
            "base_linear_speed_m_s": (base_speed < 0.02, base_speed, "<", 0.02),
            "gripper_closed_error_rad": (gripper_error < 0.03, gripper_error, "<", 0.03),
        }
        for initial in (True, False):
            selected = torch.where(first_entry == initial)[0]
            if not len(selected):
                continue
            group_conditions = {
                name: (passed[selected], value[selected], relation, threshold)
                for name, (passed, value, relation, threshold) in conditions.items()
            }
            if not initial:
                group_conditions.update(
                    outside_face_gap_min_m=(gap[selected] >= -0.001, gap[selected], ">=", -0.001),
                    outside_face_gap_max_m=(gap[selected] <= 0.010, gap[selected], "<=", 0.010),
                )
            self.record_acceptance(
                self.Stage.INSERT_INTO_DOOR_GAP, rows[selected], group_conditions, required_streak=8
            )
        self.pull_insert_stable_steps[rows] = consecutive_steps(
            ready, self.pull_insert_stable_steps[rows]
        )
        completed = rows[self.pull_insert_stable_steps[rows] >= 8]
        if len(completed):
            if self.terminal_stage == self.Stage.INSERT_INTO_DOOR_GAP:
                self.transition(completed, self.Stage.SUCCESS)
                completed = empty
            else:
                opened = (
                    state["door_joint_position"][completed, self.hinge_idx]
                    >= self.pull_panel_config.opening_angle_rad
                )
                self.pull_door_opened[completed] |= opened
                # If the door coasted open during insertion, retract safely;
                # do not start a redundant arc or declare the task complete.
                self.transition(completed[opened], self.Stage.STOW_ARM_AFTER_PULL)
                pry = completed[~opened]
                if len(pry):
                    self.begin_pry_door_open(state, pry)
        active = self.mask(self.Stage.INSERT_INTO_DOOR_GAP)
        self.stage_steps[active] += 1
        return completed

    def begin_pry_door_open(self, state, rows):
        # Follow a hinge-centered positional arc with fixed world orientation.
        # The sparse planner maps all samples through the shared inverse once.
        start = state["door_joint_position"][rows, self.hinge_idx]
        self.push_start_hinge_angle[rows] = start
        self.push_last_progress_hinge[rows] = start
        self.push_no_progress_steps[rows] = 0
        self.push_speed_error_integral[rows] = 0.0
        self.push_overspeed_steps[rows] = 0
        self.push_command_hinge_angle[rows] = start
        self.pull_move_start_hinge[rows] = start
        fraction = torch.linspace(0.0, 1.0, 181, device=self.device)
        delta = (
            self.hinge_axis_sign[rows, None]
            * (self.push_goal_hinge_angle[rows] - start)[:, None]
            * fraction[None, :]
        )
        entry = state["ee_pos"][rows]
        center = self.hinge_world[rows]
        relative = entry - center
        c, s = torch.cos(delta), torch.sin(delta)
        points = center[:, None, :] + torch.stack(
            (
                c * relative[:, None, 0] - s * relative[:, None, 1],
                s * relative[:, None, 0] + c * relative[:, None, 1],
                relative[:, None, 2].expand_as(c),
            ),
            dim=-1,
        )
        q = state["ee_orn"][rows]
        orientations = q[:, None, :].expand(-1, 181, -1)
        self.pull_open_path.positions[rows, :181] = points
        self.pull_open_path.orientations[rows, :181] = orientations
        origin = self.env.get_goal_reference_origin()[rows]
        reference = self.env.get_goal_reference_quat()[rows]
        offset = self.env.get_goal_center_offset_local()[rows]
        local = (
            _quat_rotate_points_torch(_quat_conjugate(reference), points - origin[:, None, :])
            - offset[:, None, :]
        )
        outside = fixed_norm(local[:, :, :2]) > self.push_comfort_reach_radius[rows, None]
        first_outside = outside.to(torch.int64).argmax(dim=1)
        self.pull_open_path.length[rows] = torch.where(
            outside.any(dim=1), (first_outside + 1).clamp_min(2), 181
        ).to(torch.int32)
        self.pull_open_path.index[rows] = 0
        self.transition(rows, self.Stage.PRY_DOOR_OPEN)

    def command_pry_door_open(self, state, buffers):
        if not self.pull_door:
            return
        rows = torch.where(self.mask(self.Stage.PRY_DOOR_OPEN))[0]
        if not len(rows):
            return
        self._base_hold_command(state, rows, buffers)
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        target, integral = hinge_speed_target(
            hinge,
            state["door_joint_velocity"][rows, self.hinge_idx],
            self.push_speed_error_integral[rows],
            dt=self.dt,
            target_speed=self.push_target_angular_speed[rows],
            kp=self.push_speed_kp[rows],
            ki=self.push_speed_ki[rows],
            integral_limit=self.push_speed_integral_limit[rows],
            max_lead=self.push_max_angular_lead[rows],
            goal=self.push_goal_hinge_angle[rows],
            start=self.push_start_hinge_angle[rows],
        )
        self.push_speed_error_integral[rows] = integral
        self.push_command_hinge_angle[rows] = target
        index = (
            (target - self.push_start_hinge_angle[rows])
            / (self.push_goal_hinge_angle[rows] - self.push_start_hinge_angle[rows]).clamp_min(1e-6)
        ).clamp(0.0, 1.0) * 180.0
        position, orientation = self.pull_open_path.sample_at(rows, index)
        self.ee_command_world[rows] = _slew_position(self.ee_command_world[rows], position, 0.003)
        self.ee_orientation_command[rows] = _slew_quaternion(
            self.ee_orientation_command[rows], orientation, 0.025
        )
        self.arm_pose_target(
            rows, self.ee_command_world[rows], self.ee_orientation_command[rows], buffers
        )
        self.gripper_target(rows, self.q_closed[rows], buffers)

    def advance_pry_door_open(self, state):
        empty = torch.empty(0, device=self.device, dtype=torch.int64)
        if not self.pull_door:
            return empty
        rows = torch.where(self.mask(self.Stage.PRY_DOOR_OPEN))[0]
        if not len(rows):
            return empty
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        speed = state["door_joint_velocity"][rows, self.hinge_idx]
        self.push_overspeed_steps[rows] = consecutive_steps(
            speed > self.push_max_angular_speed[rows], self.push_overspeed_steps[rows]
        )
        self.fail(rows[self.push_overspeed_steps[rows] >= 3], 52)
        progressed = hinge >= self.push_last_progress_hinge[rows] + math.radians(0.25)
        self.push_last_progress_hinge[rows] = torch.where(
            progressed, hinge, self.push_last_progress_hinge[rows]
        )
        commanded = self.push_command_hinge_angle[rows] > hinge + math.radians(0.5)
        self.push_no_progress_steps[rows] = torch.where(
            progressed | ~commanded, 0, self.push_no_progress_steps[rows] + 1
        )
        self.fail(rows[self.push_no_progress_steps[rows] > 300], 51)
        fraction = (
            (self.push_command_hinge_angle[rows] - self.push_start_hinge_angle[rows])
            / (self.push_goal_hinge_angle[rows] - self.push_start_hinge_angle[rows]).clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        at_path_end = (self.pull_open_path.length[rows] < 181) & (
            fraction * 180.0 >= self.pull_open_path.length[rows] - 1
        )
        endpoint = self.pull_open_path.positions[
            rows, (self.pull_open_path.length[rows] - 1).to(torch.int64)
        ]
        endpoint_commanded = fixed_norm(self.ee_command_world[rows] - endpoint) <= 1e-4
        needs_reposition = outside_comfort_workspace(
            state["ee_goal_cart_local"][rows], self.push_comfort_reach_radius[rows]
        ) | (at_path_end & endpoint_commanded)
        opened = hinge >= self.pull_panel_config.opening_angle_rad
        minimum_open = hinge >= math.radians(30.0)
        complete = (opened | needs_reposition) & minimum_open
        reach = fixed_norm(state["ee_goal_cart_local"][rows, :2])
        self.record_acceptance(
            self.Stage.PRY_DOOR_OPEN,
            rows,
            {
                "door_angle_at_least_30deg": (minimum_open, hinge, ">=", math.radians(30.0)),
                "door_open_or_workspace_boundary": (
                    complete,
                    complete.to(torch.float32),
                    "==",
                    1.0,
                ),
                "arm_reach_m": (reach >= 0.0, reach, ">=", 0.0),
            },
        )
        completed_mask = complete & self.mask(self.Stage.PRY_DOOR_OPEN)[rows]
        completed = rows[completed_mask]
        if len(completed):
            self.pull_finished_for_reposition[completed] = (needs_reposition & ~opened)[
                completed_mask
            ]
            self.pull_door_opened[completed] |= opened[completed_mask]
            if self.terminal_stage == self.Stage.PRY_DOOR_OPEN:
                self.transition(completed, self.Stage.SUCCESS)
                completed = empty
            else:
                self.transition(completed, self.Stage.STOW_ARM_AFTER_PULL)
        self.stage_steps[self.mask(self.Stage.PRY_DOOR_OPEN)] += 1
        return completed

    def command_move_to_panel_station(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.MOVE_TO_PANEL_STATION))[0]
        if not len(rows):
            return
        command, distance, yaw_error = self._track_station(
            state,
            rows,
            self.panel_station_command_xy[rows],
            self.panel_station_command_yaw[rows],
            self.panel_station_xy[rows],
            self.panel_station_nominal_yaw[rows],
            buffers=buffers,
        )
        self.pending_station_distance[rows] = distance
        self.pending_yaw_error[rows] = yaw_error
        self.base_hold_target_xy[rows] = self.panel_station_command_xy[rows]
        self.base_hold_target_yaw[rows] = self.panel_station_command_yaw[rows]
        self.base_hold_nominal_xy[rows] = self.panel_station_xy[rows]
        self.base_hold_nominal_yaw[rows] = self.panel_station_nominal_yaw[rows]
        self.arm_joint_target(rows, self.align_panel_arm_joint_target[rows], buffers)
        self.gripper_target(rows, self.q_closed[rows], buffers)
        self.ee_command_world[rows] = state["ee_pos"][rows]
        self.ee_orientation_command[rows] = state["ee_orn"][rows]
        self.ee_command_valid[rows] = True

    def advance_move_to_panel_station(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.MOVE_TO_PANEL_STATION))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        self.record_acceptance(
            self.Stage.MOVE_TO_PANEL_STATION,
            rows,
            {
                "base_arrival_confirmed": (
                    self.base_arrival_confirmed[rows],
                    self.base_arrival_settle_steps[rows],
                    ">=frames",
                    self.base_arrival_required_frames[rows],
                ),
                "base_stability_confirmed": (
                    self.base_stability_confirmed[rows],
                    self.base_stability_steps[rows],
                    ">=frames",
                    self.base_arrival_required_frames[rows],
                ),
            },
        )
        completed = rows[self.base_arrival_confirmed[rows] & self.base_stability_confirmed[rows]]
        if len(completed):
            self.panel_contact_plan_valid[completed] = False
            self.panel_contact_settle_steps[completed] = 0
            self.transition(completed, self.Stage.MOVE_TO_PANEL_STANDOFF)
            self.stage_steps[completed] = 1
        active = self.mask(self.Stage.MOVE_TO_PANEL_STATION)
        self.stage_steps[active] += 1
        return completed

    def contact_plan_requests(self) -> torch.Tensor:
        return torch.where(
            self.mask(self.Stage.MOVE_TO_PANEL_STANDOFF) & ~self.panel_contact_plan_valid
        )[0]

    def command_move_to_panel_standoff(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(
            self.mask(self.Stage.MOVE_TO_PANEL_STANDOFF) & self.panel_contact_plan_valid
        )[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows, buffers)
        if not self.pull_door:
            position, orientation = self.panel_arrival_path.current(rows, advance=True)
            self.ee_command_world[rows] = position
            self.ee_orientation_command[rows] = orientation
        else:
            self.ee_command_world[rows] = _slew_position(
                self.ee_command_world[rows], self.panel_contact_command_target_world[rows], 0.003
            )
            self.ee_orientation_command[rows] = _slew_quaternion(
                self.ee_orientation_command[rows],
                self.panel_contact_command_orientation_world[rows],
                0.025,
            )
        self.arm_pose_target(
            rows,
            self.ee_command_world[rows],
            self.ee_orientation_command[rows],
            buffers,
        )
        self.gripper_target(rows, self.q_closed[rows], buffers)

    def advance_move_to_panel_standoff(
        self,
        state: dict[str, torch.Tensor],
        finger_clearance: torch.Tensor,
    ) -> torch.Tensor:
        rows = torch.where(
            self.mask(self.Stage.MOVE_TO_PANEL_STANDOFF) & self.panel_contact_plan_valid
        )[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        current_normal = (
            _quat_rotate(_yaw_quat(self.hinge_axis_sign[rows] * hinge), self.door_normal[rows])
            * self.panel_contact_normal_sign[rows, None]
        )
        ee_error = state["ee_pos"][rows] - self.panel_contact_target_world[rows]
        signed = fixed_dot(ee_error, current_normal)
        tangent_error = fixed_norm(ee_error - signed[:, None] * current_normal)
        insertion_axis = _quat_rotate(state["ee_orn"][rows], self.depth_axis_local[rows])
        insertion_axis = fixed_normalize(insertion_axis, eps=1e-8)
        normal_axis_error = torch.acos(fixed_dot(insertion_axis, -current_normal).clamp(-1.0, 1.0))
        orientation_error = _quat_distance(
            state["ee_orn"][rows], self.panel_contact_command_orientation_world[rows]
        )
        axis_allowed = (
            normal_axis_error <= math.radians(30.0) + self.grasp_orientation_tolerance[rows]
        )
        ee_speed = fixed_norm(state["ee_vel"][rows, :3])
        clearance = finger_clearance[rows]
        self.panel_last_finger_clearance[rows] = clearance
        self.panel_last_lateral_error[rows] = tangent_error
        self.panel_last_axis_error[rows] = orientation_error
        self.panel_last_ee_speed[rows] = ee_speed
        sample = (
            (clearance >= 0.0)
            & (clearance <= self.panel_standoff_acceptance_clearance[rows])
            & (tangent_error < 0.035)
            & (orientation_error <= self.grasp_orientation_tolerance[rows])
            & axis_allowed
            & (ee_speed < 0.06)
        )
        gripper_closed = state["gripper_position"][rows, 0].abs() < 0.20
        self.record_acceptance(
            self.Stage.MOVE_TO_PANEL_STANDOFF,
            rows,
            {
                "finger_clearance_nonnegative_m": (clearance >= 0.0, clearance, ">=", 0.0),
                "finger_clearance_max_m": (
                    clearance <= self.panel_standoff_acceptance_clearance[rows],
                    clearance,
                    "<=",
                    self.panel_standoff_acceptance_clearance[rows],
                ),
                "lateral_error_m": (tangent_error < 0.035, tangent_error, "<", 0.035),
                "insertion_axis_error_rad": (
                    orientation_error <= self.grasp_orientation_tolerance[rows],
                    orientation_error,
                    "<=",
                    self.grasp_orientation_tolerance[rows],
                ),
                "panel_normal_axis_error_rad": (
                    axis_allowed,
                    normal_axis_error,
                    "<=",
                    math.radians(30.0) + self.grasp_orientation_tolerance[rows],
                ),
                "ee_speed_m_s": (ee_speed < 0.06, ee_speed, "<", 0.06),
                "gripper_closed_abs_rad": (
                    gripper_closed,
                    state["gripper_position"][rows, 0].abs(),
                    "<",
                    0.20,
                ),
            },
            required_streak=8,
        )
        settled = consecutive_steps(sample, self.panel_contact_settle_steps[rows])
        self.panel_contact_settle_steps[rows] = settled
        ready = rows[(settled >= 8) & gripper_closed]
        self.stage_steps[rows] += 1
        return ready

    def begin_push_panel_open(
        self,
        state: dict[str, torch.Tensor],
        rows: torch.Tensor,
        finger_clearance: torch.Tensor,
    ) -> None:
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        self.push_start_hinge_angle[rows] = hinge
        self.push_last_progress_hinge[rows] = hinge
        self.push_no_progress_steps[rows] = 0
        self.push_orientation[rows] = state["ee_orn"][rows]
        self.push_tip_offset_world[rows] = _quat_rotate(
            self.push_orientation[rows], self.panel_push_tip_offset_local[rows]
        )
        surface = _rotate_about_z(
            self.panel_push_tip_surface_world[rows],
            self.hinge_world[rows],
            self.hinge_axis_sign[rows] * (hinge - self.panel_push_tip_hinge_angle[rows]),
        )
        self.push_contact_world[rows] = surface - self.push_tip_offset_world[rows]
        self.push_command_hinge_angle[rows] = hinge
        self.push_speed_error_integral[rows] = 0.0
        self.push_overspeed_steps[rows] = 0
        self.push_max_observed_angular_speed[rows] = state["door_joint_velocity"][
            rows, self.hinge_idx
        ].abs()
        self.push_arm_detached[rows] = False
        self.push_recovery_settle_steps[rows] = 0
        self.push_reengage_reachable[rows] = False
        self.push_previous_finger_clearance[rows] = finger_clearance[rows]
        self.push_base_start_xy[rows] = state["root_states"][rows, :2]
        self.push_entry_ee_world[rows] = state["ee_pos"][rows]
        self.push_entry_command_world[rows] = self.ee_command_world[rows]
        self.transition(rows, self.Stage.PUSH_PANEL_OPEN)
        self.stage_steps[rows] = 1

    def push_retract_requests(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.PUSH_PANEL_OPEN))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        retract = outside_comfort_workspace(
            state["ee_goal_cart_local"][rows], self.push_comfort_reach_radius[rows]
        )
        planned_stop = self.push_planned_stop_hinge[rows]
        retract |= (
            (planned_stop < math.radians(89.0) - 1.0e-5)
            & (self.push_command_hinge_angle[rows] >= planned_stop - 1.0e-4)
            & (
                state["door_joint_position"][rows, self.hinge_idx]
                >= planned_stop - math.radians(2.0)
            )
        )
        if self.pull_door:
            retract |= self.pull_door_opened[rows] | (
                state["door_joint_position"][rows, self.hinge_idx]
                >= self.pull_panel_config.opening_angle_rad
            )
        return rows[retract]

    def _world_orientation_from_local_rpy(
        self, rows: torch.Tensor, local_rpy: torch.Tensor
    ) -> torch.Tensor:
        return _quat_multiply(self.env.get_goal_reference_quat()[rows], _euler_xyz_quat(local_rpy))

    def command_push_panel_open(
        self,
        state: dict[str, torch.Tensor],
        finger_clearance: torch.Tensor,
        buffers: TensorCommandBuffers,
    ) -> None:
        rows = torch.where(self.mask(self.Stage.PUSH_PANEL_OPEN))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows, buffers)
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        hinge_velocity = state["door_joint_velocity"][rows, self.hinge_idx]
        target_hinge, integral = hinge_speed_target(
            hinge,
            hinge_velocity,
            self.push_speed_error_integral[rows],
            dt=self.dt,
            target_speed=self.push_target_angular_speed[rows],
            kp=self.push_speed_kp[rows],
            ki=self.push_speed_ki[rows],
            integral_limit=self.push_speed_integral_limit[rows],
            max_lead=self.push_tip_max_angular_lead[rows],
            goal=torch.minimum(
                self.push_goal_hinge_angle[rows], self.push_planned_stop_hinge[rows]
            ),
            start=self.push_start_hinge_angle[rows],
        )
        self.push_speed_error_integral[rows] = integral
        self.push_command_hinge_angle[rows] = target_hinge
        delta = self.hinge_axis_sign[rows] * (target_hinge - self.push_start_hinge_angle[rows])
        tip_offset = self.push_tip_offset_world[rows]
        target = fingertip_arc_position(
            self.push_contact_world[rows] + tip_offset, tip_offset, self.hinge_world[rows], delta
        )
        self.ee_command_world[rows] = _slew_position(self.ee_command_world[rows], target, 0.003)
        # Preserve the current baseline: PUSH_PANEL_OPEN ultimately holds the
        # accepted contact orientation instead of rotating it with the door.
        self.ee_orientation_command[rows] = self.push_orientation[rows]
        self.arm_pose_target(
            rows,
            self.ee_command_world[rows],
            self.ee_orientation_command[rows],
            buffers,
        )
        self.gripper_target(rows, self.q_closed[rows], buffers)
        self.push_previous_finger_clearance[rows] = finger_clearance[rows]

    def command_stow_arm_after_push(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.STOW_ARM_AFTER_PUSH))[0]
        if not len(rows):
            return
        command, _, _ = self._base_hold_command(state, rows, buffers)
        delta = (
            self.active_arm_retract_pose_local[rows, :3] - self.push_detached_ee_goal_local[rows]
        )
        distance = fixed_norm(delta)
        moving = self.stage_steps[rows] >= 5
        step = delta * torch.clamp(0.004 / distance.clamp_min(1e-8), max=1.0)[:, None]
        self.push_detached_ee_goal_local[rows] = torch.where(
            moving[:, None],
            self.push_detached_ee_goal_local[rows] + step,
            self.push_detached_ee_goal_local[rows],
        )
        orientation = self._world_orientation_from_local_rpy(
            rows, self.active_arm_retract_pose_local[rows, 3:]
        )
        self.ee_orientation_command[rows] = _slew_quaternion(
            self.ee_orientation_command[rows], orientation, 0.025
        )
        self.arm_pose_target(
            rows,
            self.push_detached_ee_goal_local[rows],
            self.ee_orientation_command[rows],
            buffers,
            frame="goal_reference_local",
        )
        self.gripper_target(rows, self.q_closed[rows], buffers)

    def advance_stow_arm_after_push(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.STOW_ARM_AFTER_PUSH))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        origin = self.env.get_goal_reference_origin()[rows]
        reference = self.env.get_goal_reference_quat()[rows]
        offset = self.env.get_goal_center_offset_local()[rows]
        actual_local = _quat_rotate_inverse(reference, state["ee_pos"][rows] - origin) - offset
        orientation_target = self._world_orientation_from_local_rpy(
            rows, self.active_arm_retract_pose_local[rows, 3:]
        )
        retracted = (
            (fixed_norm(actual_local - self.active_arm_retract_pose_local[rows, :3]) <= 0.010)
            & (_quat_distance(state["ee_orn"][rows], orientation_target) < 0.05)
            & (fixed_norm(state["ee_vel"][rows, :3]) < 0.04)
        )
        retract_position_error = fixed_norm(
            actual_local - self.active_arm_retract_pose_local[rows, :3]
        )
        retract_orientation_error = _quat_distance(state["ee_orn"][rows], orientation_target)
        retract_speed = fixed_norm(state["ee_vel"][rows, :3])
        self.record_acceptance(
            self.Stage.STOW_ARM_AFTER_PUSH,
            rows,
            {
                "ee_position_error_m": (
                    retract_position_error <= 0.010,
                    retract_position_error,
                    "<=",
                    0.010,
                ),
                "ee_orientation_error_rad": (
                    retract_orientation_error < 0.05,
                    retract_orientation_error,
                    "<",
                    0.05,
                ),
                "ee_speed_m_s": (retract_speed < 0.04, retract_speed, "<", 0.04),
            },
            required_streak=10,
        )
        settled = consecutive_steps(retracted, self.push_recovery_settle_steps[rows])
        self.push_recovery_settle_steps[rows] = settled
        completed = rows[settled >= 10]
        if len(completed):
            arm_ids = self.env.arm_joint_ids.to(device=self.device, dtype=torch.int64)
            self.base_advance_arm_joint_target[completed] = state["dof_pos"][
                completed[:, None], arm_ids[None, :]
            ]
            self.push_reengage_reachable[completed] = False
            self.base_advance_target_longitudinal[completed] = -torch.inf
            self.transition(completed, self.Stage.ADVANCE_TO_NEXT_PUSH_STATION)
        active = self.mask(self.Stage.STOW_ARM_AFTER_PUSH)
        self.stage_steps[active] += 1
        return completed

    def advance_push_panel_open(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = (
            self.mask(self.Stage.PUSH_PANEL_OPEN)
            | self.mask(self.Stage.STOW_ARM_AFTER_PUSH)
            | self.mask(self.Stage.ADVANCE_TO_NEXT_PUSH_STATION)
        )
        rows = torch.where(mask)[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        speed = state["door_joint_velocity"][rows, self.hinge_idx]
        self.push_max_observed_angular_speed[rows] = torch.maximum(
            self.push_max_observed_angular_speed[rows], speed.clamp_min(0.0)
        )
        overspeed = speed > self.push_max_angular_speed[rows]
        self.push_overspeed_steps[rows] = consecutive_steps(
            overspeed, self.push_overspeed_steps[rows]
        )
        overspeed_failure = self.push_overspeed_steps[rows] >= 3
        hinge_limit_failure = (~overspeed_failure) & (hinge > math.radians(94.0))
        failure = overspeed_failure | hinge_limit_failure
        self.fail(rows[overspeed_failure], 20)
        self.fail(rows[hinge_limit_failure], 22)
        through = rows[(~failure) & (hinge >= math.radians(89.0))]
        push = (self.stage[rows] == int(self.Stage.PUSH_PANEL_OPEN)) & ~failure
        push_rows = rows[push]
        if len(push_rows):
            push_hinge = state["door_joint_position"][push_rows, self.hinge_idx]
            self.record_acceptance(
                self.Stage.PUSH_PANEL_OPEN,
                push_rows,
                {
                    "door_hinge_angle_rad": (
                        push_hinge >= math.radians(89.0),
                        push_hinge,
                        ">=",
                        math.radians(89.0),
                    )
                },
            )
            progressed = push_hinge >= (
                self.push_last_progress_hinge[push_rows] + math.radians(0.25)
            )
            self.push_last_progress_hinge[push_rows] = torch.where(
                progressed, push_hinge, self.push_last_progress_hinge[push_rows]
            )
            commanded = self.push_command_hinge_angle[push_rows] > (push_hinge + math.radians(0.5))
            self.push_no_progress_steps[push_rows] = torch.where(
                progressed | ~commanded,
                0,
                self.push_no_progress_steps[push_rows] + 1,
            )
            stalled = self.push_no_progress_steps[push_rows] > 300
            self.fail(push_rows[stalled], 21)
        return through

    def _tracked_push_contact(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor
    ) -> torch.Tensor:
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        delta = self.hinge_axis_sign[rows] * (hinge - self.push_start_hinge_angle[rows])
        tip_offset = self.push_tip_offset_world[rows]
        return fingertip_arc_position(
            self.push_contact_world[rows] + tip_offset, tip_offset, self.hinge_world[rows], delta
        )

    def _base_advance_targets(
        self, state: dict[str, torch.Tensor], rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = state["root_states"][rows]
        forward = self.door_forward[rows, :2]
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        line_origin = self.panel_station_xy[rows]
        current_longitudinal = fixed_dot(base[:, :2] - line_origin, forward)
        line_position = line_origin + current_longitudinal[:, None] * forward
        contact = self._tracked_push_contact(state, rows)
        origin = self.env.get_goal_reference_origin()[rows]
        reference = self.env.get_goal_reference_quat()[rows]
        offset = self.env.get_goal_center_offset_local()[rows]
        contact_local = _quat_rotate_inverse(reference, contact - origin) - offset
        forward_world = torch.cat(
            (forward, torch.zeros((len(rows), 1), device=self.device)), dim=-1
        )
        forward_local = _quat_rotate_inverse(reference, forward_world)[:, :2]
        lateral_world_xy = line_position - base[:, :2]
        lateral_world = torch.cat(
            (lateral_world_xy, torch.zeros((len(rows), 1), device=self.device)), dim=-1
        )
        contact_xy = contact_local[:, :2] - _quat_rotate_inverse(reference, lateral_world)[:, :2]
        radius = self.push_reengage_reach_radius[rows]
        a = fixed_sum(forward_local.square())
        b = fixed_dot(contact_xy, forward_local)
        c = fixed_sum(contact_xy.square()) - radius.square()
        discriminant = b.square() - a * c
        root = torch.sqrt(discriminant.clamp_min(0.0))
        interval_start = (b - root) / a.clamp_min(1e-10)
        interval_end = (b + root) / a.clamp_min(1e-10)
        solvable = (c > 0.0) & (a > 1e-10) & (discriminant >= 0.0) & (interval_end >= 0.0)
        required = torch.where(
            c <= 0.0,
            torch.zeros_like(c),
            torch.where(solvable, interval_start.clamp_min(0.0), torch.zeros_like(c)),
        )
        required += torch.where(
            (c <= 0.0) | solvable,
            self.push_reengage_target_lead[rows],
            torch.zeros_like(required),
        )

        closed_free_end = 2.0 * self.panel_center_world[rows] - self.hinge_world[rows]
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        panel_free_end = _rotate_about_z(
            closed_free_end,
            self.hinge_world[rows],
            self.hinge_axis_sign[rows] * hinge,
        )
        panel_start = self.hinge_world[rows, :2]
        panel_end = panel_free_end[:, :2]
        shoulders = state["robot_body_state_w"][
            rows[:, None], self.front_shoulder_body_indices[None, :], :2
        ]
        count = torch.maximum(
            torch.full_like(self.stage_steps[rows], 2),
            torch.ceil(required / 0.005).to(torch.int32) + 1,
        )
        capacity = 256
        if torch.any(count > capacity):
            raise RuntimeError("base-advance shoulder search exceeds 256 candidates")
        candidate_id = torch.arange(capacity, device=self.device)
        valid = candidate_id[None, :] < count[:, None]
        fraction = candidate_id[None, :].to(torch.float32) / (count[:, None] - 1).clamp_min(1)
        advances = fraction * required[:, None]
        targets = line_position[:, None, :] + advances[..., None] * forward[:, None, :]
        displacement = targets - base[:, None, :2]
        points = shoulders[:, None, :, :] + displacement[:, :, None, :]
        segment = panel_end - panel_start
        segment_norm_sq = fixed_sum(segment.square())
        projection = (
            fixed_dot(
                points - panel_start[:, None, None, :],
                segment[:, None, None, :],
            )
            / segment_norm_sq[:, None, None].clamp_min(1e-10)
        ).clamp(0.0, 1.0)
        closest = panel_start[:, None, None, :] + projection[..., None] * segment[:, None, None, :]
        distances = fixed_norm(points - closest).amin(dim=-1)
        required_clearance = (
            self.push_shoulder_collision_radius[rows]
            + self.panel_half_thickness[rows]
            + self.push_shoulder_safety_margin[rows]
        )
        safe = (distances >= required_clearance[:, None]) & valid
        unsafe = (~safe) & valid
        first_unsafe = torch.argmax(unsafe.to(torch.int32), dim=-1)
        has_unsafe = torch.any(unsafe, dim=-1)
        target_index = torch.where(
            has_unsafe, (first_unsafe - 1).clamp_min(0), count.to(torch.int64) - 1
        )
        target_advance = advances[torch.arange(len(rows), device=self.device), target_index]
        proposed = current_longitudinal + target_advance
        previous = self.base_advance_target_longitudinal[rows]
        longitudinal = torch.maximum(current_longitudinal, torch.maximum(proposed, previous))
        longitudinal = torch.where(
            self.push_reengage_reachable[rows], current_longitudinal, longitudinal
        )
        self.base_advance_target_longitudinal[rows] = longitudinal
        return line_origin + longitudinal[:, None] * forward, yaw

    def command_advance_to_next_push_station(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(self.mask(self.Stage.ADVANCE_TO_NEXT_PUSH_STATION))[0]
        if not len(rows):
            return
        target_xy, target_yaw = self._base_advance_targets(state, rows)
        contact = self._tracked_push_contact(state, rows)
        origin = self.env.get_goal_reference_origin()[rows]
        reference = self.env.get_goal_reference_quat()[rows]
        offset = self.env.get_goal_center_offset_local()[rows]
        contact_local = _quat_rotate_inverse(reference, contact - origin) - offset
        reach = fixed_norm(contact_local[:, :2])
        reachable = self.push_reengage_reachable[rows]
        reachable = torch.where(
            reachable,
            reach <= self.push_reengage_release_radius[rows],
            reach <= self.push_reengage_reach_radius[rows],
        )
        self.push_reengage_reachable[rows] = reachable
        _, distance, yaw_error = self.base_target(
            state,
            rows,
            target_xy,
            target_yaw,
            buffers,
            task_stop=reachable,
            enforce_task_stop=False,
        )
        self.pending_station_distance[rows] = distance
        self.pending_yaw_error[rows] = yaw_error
        self.base_hold_target_xy[rows] = target_xy
        self.base_hold_target_yaw[rows] = target_yaw
        self.base_hold_nominal_xy[rows] = target_xy
        self.base_hold_nominal_yaw[rows] = target_yaw
        self.arm_joint_target(rows, self.base_advance_arm_joint_target[rows], buffers)
        self.gripper_target(rows, self.q_closed[rows], buffers)
        _, settled, stable = update_robot_stability(
            state["root_states"][rows],
            self.push_recovery_settle_steps[rows],
            required_frames=10,
            max_linear_speed_m_s=0.04,
            max_angular_speed_rad_s=TERMINAL_BASE_ANGULAR_SPEED_RAD_S,
            min_base_height_m=self.through_min_base_height[rows],
            min_upright_cos=THROUGH_ENTRY_UPRIGHT_COS,
            door_speed_rad_s=state["door_joint_velocity"][rows, self.hinge_idx],
            max_door_speed_rad_s=self.turn_release_max_hinge_speed[rows],
            additional_condition=reachable,
        )
        self.push_recovery_settle_steps[rows] = settled
        base = state["root_states"][rows]
        base_linear_speed = fixed_norm(base[:, 7:10])
        base_angular_speed = fixed_norm(base[:, 10:13])
        base_up = _quat_rotate(
            base[:, 3:7],
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(len(rows), -1),
        )[:, 2]
        door_speed = state["door_joint_velocity"][rows, self.hinge_idx].abs()
        self.record_acceptance(
            self.Stage.ADVANCE_TO_NEXT_PUSH_STATION,
            rows,
            {
                "contact_reachable": (
                    reachable,
                    reach,
                    "<=hysteresis_threshold",
                    self.push_reengage_release_radius[rows],
                ),
                "base_linear_speed_m_s": (base_linear_speed < 0.04, base_linear_speed, "<", 0.04),
                "base_angular_speed_rad_s": (
                    base_angular_speed < TERMINAL_BASE_ANGULAR_SPEED_RAD_S,
                    base_angular_speed,
                    "<",
                    TERMINAL_BASE_ANGULAR_SPEED_RAD_S,
                ),
                "base_height_m": (
                    base[:, 2] >= self.through_min_base_height[rows],
                    base[:, 2],
                    ">=",
                    self.through_min_base_height[rows],
                ),
                "base_upright_cos": (
                    base_up >= THROUGH_ENTRY_UPRIGHT_COS,
                    base_up,
                    ">=",
                    THROUGH_ENTRY_UPRIGHT_COS,
                ),
                "door_speed_rad_s": (
                    door_speed < self.turn_release_max_hinge_speed[rows],
                    door_speed,
                    "<",
                    self.turn_release_max_hinge_speed[rows],
                ),
                "robot_stable": (stable, stable.to(torch.float32), "==", 1.0),
            },
            required_streak=10,
        )

    def advance_advance_to_next_push_station(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.ADVANCE_TO_NEXT_PUSH_STATION))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        completed = rows[self.push_recovery_settle_steps[rows] >= 10]
        if len(completed):
            hinge = state["door_joint_position"][completed, self.hinge_idx]
            delta = self.hinge_axis_sign[completed] * (
                hinge - self.push_start_hinge_angle[completed]
            )
            self.panel_contact_orientation_seed[completed] = _quat_multiply(
                _yaw_quat(delta), self.push_orientation[completed]
            )
            self.panel_contact_orientation_seed_valid[completed] = True
            self.ee_command_world[completed] = state["ee_pos"][completed]
            self.panel_contact_plan_valid[completed] = False
            self.panel_contact_settle_steps[completed] = 0
            self.transition(completed, self.Stage.MOVE_TO_PANEL_STANDOFF)
            self.stage_steps[completed] = 1
        active = self.mask(self.Stage.ADVANCE_TO_NEXT_PUSH_STATION)
        self.stage_steps[active] += 1
        return completed

    def begin_pass_through_doorway(self, rows: torch.Tensor, *, rolling: bool = False) -> None:
        self.through_trajectory.clear(rows)
        self.through_path_previous_index[rows] = 0
        self.through_best_station_error[rows] = torch.inf
        self.through_no_progress_steps[rows] = 0
        self.through_settle_steps[rows] = 0
        self.through_stop_latched[rows] = False
        self.through_motion_ready[rows] = rolling
        self.through_entry_stable_steps[rows] = 0
        self.transition(rows, self.Stage.PASS_THROUGH_DOORWAY)
        self.stage_steps[rows] = 1

    def command_pass_through_doorway(
        self, state: dict[str, torch.Tensor], buffers: TensorCommandBuffers
    ) -> None:
        rows = torch.where(
            self.mask(self.Stage.PASS_THROUGH_DOORWAY) | self.mask(self.Stage.SUCCESS)
        )[0]
        if not len(rows):
            return
        entering = rows[self.through_trajectory.length[rows] == 0]
        if len(entering):
            root = state["root_states"][entering]
            start = torch.cat((root[:, :2], _yaw_from_wxyz(root[:, 3:7])[:, None]), dim=-1)
            envelope = robot_motion_envelope(
                root,
                state["robot_body_state_w"][entering],
                self.base_motion_points,
                self.base_motion_body_indices,
            )
            radius = fixed_norm(envelope[..., :2]).flatten(1).amax(-1)
            finish = torch.cat(
                (self.through_command_xy[entering], self.through_command_yaw[entering, None]),
                dim=-1,
            )
            route = frame_traversal_waypoints(
                start,
                self.initial_door_root[entering, :2],
                self.door_forward[entering, :2],
                finish,
                radius,
            )
            self.through_trajectory.install(entering, start, route, self.dt)
        reference, velocity, _ = self.through_trajectory.sample(rows)
        tail_clearance = self._tail_clearance(state, rows)
        latched = self.through_stop_latched[rows]
        latched = torch.where(
            latched,
            tail_clearance >= THROUGH_TAIL_RELEASE_DISTANCE,
            tail_clearance >= THROUGH_TAIL_TARGET_DISTANCE,
        )
        self.through_stop_latched[rows] = latched
        through = self.stage[rows] == int(self.Stage.PASS_THROUGH_DOORWAY)
        _, distance, yaw_error = self.base_target(
            state,
            rows,
            reference[:, :2],
            reference[:, 2],
            buffers,
            acceptance_xy=self.through_station_xy[rows],
            acceptance_yaw=self.through_station_yaw[rows],
            reference_velocity_world=velocity,
            task_stop=through & latched,
            motion_allowed=~through | self.through_motion_ready[rows],
            final_allowed=through,
        )
        self.pending_station_distance[rows] = distance
        self.pending_yaw_error[rows] = yaw_error
        self.arm_joint_target(rows, self.middle_arm_joint_target[rows], buffers)
        self.gripper_target(rows, self.q_closed[rows], buffers)

    def _tail_clearance(self, state: dict[str, torch.Tensor], rows: torch.Tensor) -> torch.Tensor:
        base = state["root_states"][rows]
        rear_local = torch.zeros((len(rows), 3), device=self.device)
        rear_local[:, 0] = ROBOT_TRUNK_REAR_X
        rear_world = _quat_rotate(base[:, 3:7], rear_local)
        tail_xy = base[:, :2] + rear_world[:, :2]
        return fixed_dot(
            tail_xy - self.initial_door_root[rows, :2],
            self.door_forward[rows, :2],
        )

    def advance_pass_through_doorway(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.where(self.mask(self.Stage.PASS_THROUGH_DOORWAY))[0]
        if not len(rows):
            return torch.empty(0, dtype=torch.int64, device=self.device)
        base = state["root_states"][rows]
        hinge = state["door_joint_position"][rows, self.hinge_idx]
        hinge_speed = state["door_joint_velocity"][rows, self.hinge_idx]
        up = _quat_rotate(
            base[:, 3:7],
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(len(rows), -1),
        )[:, 2]
        entry_confirmed, entry_steps, _ = update_robot_stability(
            base,
            self.through_entry_stable_steps[rows],
            required_frames=THROUGH_ENTRY_STABILITY_FRAMES,
            max_linear_speed_m_s=TERMINAL_BASE_LINEAR_SPEED_M_S,
            max_angular_speed_rad_s=TERMINAL_BASE_ANGULAR_SPEED_RAD_S,
            min_base_height_m=self.through_min_base_height[rows],
            min_upright_cos=THROUGH_ENTRY_UPRIGHT_COS,
            door_speed_rad_s=hinge_speed,
            max_door_speed_rad_s=TERMINAL_DOOR_SPEED_RAD_S,
        )
        self.through_entry_stable_steps[rows] = entry_steps
        self.through_motion_ready[rows] |= entry_confirmed
        distance = self.pending_station_distance[rows]
        improved = distance < self.through_best_station_error[rows] - 0.01
        path_progress = self.through_trajectory.index[rows] > self.through_path_previous_index[rows]
        self.through_path_previous_index[rows] = self.through_trajectory.index[rows]
        self.through_best_station_error[rows] = torch.where(
            improved, distance, self.through_best_station_error[rows]
        )
        no_progress = torch.where(
            ~self.through_motion_ready[rows]
            | improved
            | path_progress
            | (distance < BASE_STATION_TOLERANCE_M),
            0,
            self.through_no_progress_steps[rows] + 1,
        )
        self.through_no_progress_steps[rows] = no_progress
        arm_ids = self.env.arm_joint_ids.to(device=self.device, dtype=torch.int64)
        arm_position = state["dof_pos"][rows[:, None], arm_ids[None, :]]
        arm_velocity = self.control_joint_speed(state, rows[:, None], arm_ids[None, :])
        middle_error = (arm_position - self.middle_arm_joint_target[rows]).abs().amax(dim=-1)
        arm_speed = arm_velocity.abs().amax(dim=-1)
        gripper_id = int(self.env.gripper_joint_ids[0].item())
        gripper_speed = self.control_joint_speed(state, rows, gripper_id)
        door_speed = state["door_joint_velocity"][rows].abs().amax(dim=-1)
        terminal_extra = (
            self.through_stop_latched[rows]
            & (middle_error < 0.05)
            & (arm_speed < TERMINAL_ARM_JOINT_SPEED_RAD_S)
            & (gripper_speed < TERMINAL_GRIPPER_SPEED_RAD_S)
        )
        terminal_confirmed, settled, terminal_stable = update_robot_stability(
            base,
            self.through_settle_steps[rows],
            required_frames=TERMINAL_STABILITY_FRAMES + 1,
            max_linear_speed_m_s=TERMINAL_BASE_LINEAR_SPEED_M_S,
            max_angular_speed_rad_s=TERMINAL_BASE_ANGULAR_SPEED_RAD_S,
            min_base_height_m=self.through_min_base_height[rows],
            min_upright_cos=THROUGH_ENTRY_UPRIGHT_COS,
            door_speed_rad_s=door_speed,
            max_door_speed_rad_s=TERMINAL_DOOR_SPEED_RAD_S,
            additional_condition=terminal_extra,
        )
        self.through_settle_steps[rows] = settled
        base_linear_speed = fixed_norm(base[:, 7:10])
        base_angular_speed = fixed_norm(base[:, 10:13])
        self.record_acceptance(
            self.Stage.PASS_THROUGH_DOORWAY,
            rows,
            {
                "tail_clearance_reached": (
                    self.through_stop_latched[rows],
                    self._tail_clearance(state, rows),
                    ">=hysteresis_threshold",
                    THROUGH_TAIL_RELEASE_DISTANCE,
                ),
                "middle_joint_error_rad": (middle_error < 0.05, middle_error, "<", 0.05),
                "arm_speed_rad_s": (
                    arm_speed < TERMINAL_ARM_JOINT_SPEED_RAD_S,
                    arm_speed,
                    "<",
                    TERMINAL_ARM_JOINT_SPEED_RAD_S,
                ),
                "gripper_speed_rad_s": (
                    gripper_speed < TERMINAL_GRIPPER_SPEED_RAD_S,
                    gripper_speed,
                    "<",
                    TERMINAL_GRIPPER_SPEED_RAD_S,
                ),
                "base_linear_speed_m_s": (
                    base_linear_speed < TERMINAL_BASE_LINEAR_SPEED_M_S,
                    base_linear_speed,
                    "<",
                    TERMINAL_BASE_LINEAR_SPEED_M_S,
                ),
                "base_angular_speed_rad_s": (
                    base_angular_speed < TERMINAL_BASE_ANGULAR_SPEED_RAD_S,
                    base_angular_speed,
                    "<",
                    TERMINAL_BASE_ANGULAR_SPEED_RAD_S,
                ),
                "base_height_m": (
                    base[:, 2] >= self.through_min_base_height[rows],
                    base[:, 2],
                    ">=",
                    self.through_min_base_height[rows],
                ),
                "base_upright_cos": (
                    up >= THROUGH_ENTRY_UPRIGHT_COS,
                    up,
                    ">=",
                    THROUGH_ENTRY_UPRIGHT_COS,
                ),
                "door_speed_rad_s": (
                    door_speed < TERMINAL_DOOR_SPEED_RAD_S,
                    door_speed,
                    "<",
                    TERMINAL_DOOR_SPEED_RAD_S,
                ),
                "terminal_stable": (terminal_stable, terminal_stable.to(torch.float32), "==", 1.0),
            },
            required_streak=TERMINAL_STABILITY_FRAMES + 1,
        )
        tail = self._tail_clearance(state, rows)
        fallen = (up < 0.75) | (base[:, 2] < 0.30)
        swingback = (hinge < math.radians(65.0)) & (tail < 0.8)
        stalled = (distance >= BASE_STATION_TOLERANCE_M) & (no_progress > 150)
        swingback_only = (~fallen) & swingback
        stalled_only = (~fallen) & (~swingback) & stalled
        failure = fallen | swingback_only | stalled_only
        self.fail(rows[fallen], 30)
        self.fail(rows[swingback_only], 31)
        self.fail(rows[stalled_only], 32)
        success = rows[(~failure) & terminal_confirmed]
        self.stage[success] = int(self.Stage.SUCCESS)
        self.stage_steps[rows[(~failure) & ~terminal_confirmed]] += 1
        return success

    def apply_timeouts(self) -> torch.Tensor:
        active = (self.stage != int(self.Stage.SUCCESS)) & (self.stage != int(self.Stage.FAILURE))
        limits = self.stage_limits[self.stage]
        timed_out = active & (limits > 0) & (self.stage_steps > limits)
        timed_out_rows = torch.where(timed_out)[0]
        if len(timed_out_rows):
            timed_out_stages = self.stage[timed_out_rows].clone()
            for stage_id in torch.unique(timed_out_stages).tolist():
                stage_rows = timed_out_rows[timed_out_stages == int(stage_id)]
                self.fail(stage_rows, 100 + int(stage_id))
        return torch.where(timed_out)[0]

    def command(
        self,
        state: dict[str, torch.Tensor],
        buffers: TensorCommandBuffers,
        *,
        grasp_geometry: dict[str, torch.Tensor] | None,
        finger_clearance: torch.Tensor,
    ) -> None:
        """Issue one command for every active env using stage masks only."""
        terminal = self.mask(self.Stage.SUCCESS) | self.mask(self.Stage.FAILURE)
        self.stop_base(terminal, buffers)
        self.command_approach_handle(state, buffers)
        self.command_pregrasp_and_open_gripper(state, buffers)
        if grasp_geometry is not None:
            self.command_insert_and_grasp_handle(state, grasp_geometry, buffers)
        self.command_press_handle(state, buffers)
        self.command_open_door_with_handle(state, buffers)
        self.command_reset_handle(state, buffers)
        self.command_release_handle(state, buffers)
        self.command_withdraw_from_handle(state, buffers)
        self.command_stow_arm_and_close_gripper(state, buffers)
        self.command_gap_or_wall_station(state, buffers)
        self.command_insert_into_door_gap(state, buffers)
        self.command_pry_door_open(state, buffers)
        self.command_move_to_panel_station(state, buffers)
        self.command_move_to_panel_standoff(state, buffers)
        self.command_push_panel_open(state, finger_clearance, buffers)
        self.command_stow_arm_after_push(state, buffers)
        self.command_advance_to_next_push_station(state, buffers)
        self.command_pass_through_doorway(state, buffers)

    def advance(
        self,
        state: dict[str, torch.Tensor],
        *,
        grasp_geometry: dict[str, torch.Tensor] | None,
        finger_clearance: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Advance all envs once and return sparse planning event masks."""
        self.update_control_joint_velocity(state)
        events: dict[str, torch.Tensor] = {}
        # Reverse stage order prevents a forward transition from being
        # evaluated a second time in the same physics frame.
        events["success"] = self.advance_pass_through_doorway(state)
        push_to_through = self.advance_push_panel_open(state)
        if len(push_to_through):
            if self.pull_door:
                self.pull_door_opened[push_to_through] = True
            else:
                self.begin_pass_through_doorway(push_to_through)
        events["contact_replan"] = self.advance_advance_to_next_push_station(state)
        self.advance_stow_arm_after_push(state)
        push_rows = torch.where(self.mask(self.Stage.PUSH_PANEL_OPEN))[0]
        self.stage_steps[push_rows] += 1
        contact_to_push = self.advance_move_to_panel_standoff(state, finger_clearance)
        if len(contact_to_push):
            self.begin_push_panel_open(state, contact_to_push, finger_clearance)
        events["contact_plan"] = self.advance_move_to_panel_station(state)
        pull_retract = self.advance_pry_door_open(state)
        pull_insert_ready = self.advance_insert_into_door_gap(state)
        events["pull_open_plan"] = pull_insert_ready[
            self.stage[pull_insert_ready] == int(self.Stage.PRY_DOOR_OPEN)
        ]
        pull_retract = torch.cat(
            (
                pull_retract,
                pull_insert_ready[
                    self.stage[pull_insert_ready] == int(self.Stage.STOW_ARM_AFTER_PULL)
                ],
            )
        )
        pull_next_insert = self.advance_follow_wall_to_panel(state)
        events["contact_plan"] = torch.cat(
            (
                events["contact_plan"],
                pull_next_insert[
                    self.stage[pull_next_insert] == int(self.Stage.MOVE_TO_PANEL_STANDOFF)
                ],
            )
        )
        pull_station_complete = self.advance_move_to_gap_station(state)
        events["pull_insert_plan"] = pull_station_complete[
            self.stage[pull_station_complete] == int(self.Stage.INSERT_INTO_DOOR_GAP)
        ]
        restored = self.advance_stow_arm_and_close_gripper(state)
        events["pull_station_plan"] = restored[
            self.stage[restored] == int(self.Stage.MOVE_TO_GAP_STATION)
        ]
        events["restore_plan"] = torch.cat((self.advance_withdraw_from_handle(state), pull_retract))
        events["retreat_plan"] = self.advance_release_handle(state)
        events["release_open_plan"] = self.advance_reset_handle(state)
        push_pull_to_return = self.advance_open_door_with_handle(state)
        if len(push_pull_to_return):
            self.begin_reset_handle(state, push_pull_to_return)
        events["return_handle_plan"] = push_pull_to_return
        press_to_push_pull = self.advance_press_handle(state)
        if len(press_to_push_pull):
            self.begin_open_door_with_handle(state, press_to_push_pull)
        events["push_pull_plan"] = press_to_push_pull
        if grasp_geometry is not None:
            grasp_to_turn = self.advance_insert_and_grasp_handle(state, grasp_geometry)
            if len(grasp_to_turn):
                self.begin_press_handle(state, grasp_to_turn)
            events["press_handle_plan"] = grasp_to_turn
        else:
            events["press_handle_plan"] = torch.empty(0, dtype=torch.int64, device=self.device)
        events["center_plan"] = self.advance_pregrasp_and_open_gripper(state)
        events["pregrasp_plan"] = self.advance_approach_handle(state)
        events["timeout"] = self.apply_timeouts()
        return events
