"""GPU batch planning at FSM stage boundaries; host access is output-only."""

from __future__ import annotations

import math

import torch

from gr00t.rl.auto_open.control import APPROACH_ARRIVAL
from gr00t.rl.auto_open.first_pull_entry import insertion_endpoint_radius, plan_first_pull_entry
from gr00t.rl.auto_open.first_pull_station import (
    first_insertion_residuals,
    station_candidates,
    station_clearances,
)
from gr00t.rl.auto_open.motion_math import _yaw_from_wxyz
from gr00t.rl.auto_open.panel_motion import PullPanelConfig
from gr00t.rl.auto_open.pose_planning import PosePlanningMixin, best_feasible, first_feasible
from gr00t.rl.auto_open.pull_entry_gpu import PullEntryGPU, quat_rotate
from gr00t.rl.auto_open.tensor_stage_planning import GPUStagePlanningMixin


class TensorPlanningCoordinator(GPUStagePlanningMixin, PosePlanningMixin):
    """Generate, solve, and select whole path banks without per-env dispatch."""

    def __init__(self, controllers, stage_enum):
        self.controllers = controllers
        self.Stage = stage_enum
        self.num_envs = len(controllers)
        env = controllers[0].env
        self.arm_joint_ids = env.arm_joint_ids.to(dtype=torch.int64)
        self.joint_limits = env.dof_pos_limits[self.arm_joint_ids]
        self.stats = {"batches": 0, "requests": 0, "errors": 0, "labels": {}}
        self.path_diagnostics = []
        self.initialize_gpu_stage_constants()
        self.pull_geometry = PullEntryGPU.from_controllers(
            controllers, device=self.joint_limits.device, config=PullPanelConfig(), max_samples=1024
        )

    def _record_stats(self, label, requests, errors=None):
        self.stats["batches"] += 1
        self.stats["requests"] += requests
        if errors is not None:
            self.stats["errors"] = self.stats["errors"] + errors
        labels = self.stats["labels"]
        labels[label] = labels.get(label, 0) + requests

    def plan_pose_path(
        self,
        fsm,
        device_state,
        rows,
        *,
        bank_name="turn_path",
        label="turn_stage_path",
        prediction_tolerance=None,
        truncate_at_first_unreachable=False,
    ):
        if not len(rows):
            return
        bank = getattr(fsm, bank_name)
        lengths = bank.length[rows].clone()
        desired_positions = bank.positions[rows].clone()
        desired_orientations = bank.orientations[rows].clone()
        result, owner, sample = self._inverse_pose_bank(
            fsm, device_state, rows, desired_positions, desired_orientations, lengths
        )
        bank.positions[rows[owner], sample] = result.command_position_world
        bank.orientations[rows[owner], sample] = result.command_orientation_world_wxyz
        if prediction_tolerance is not None:
            error = torch.linalg.vector_norm(result.position_error, dim=-1)
            angle = 2.0 * torch.asin(
                torch.linalg.vector_norm(result.orientation_error, dim=-1).clamp(0.0, 1.0)
            )
            bad = (
                (error > prediction_tolerance)
                | (angle > math.radians(1.0))
                | ~torch.isfinite(error)
                | ~torch.isfinite(angle)
            )
            first_bad = torch.full_like(lengths, desired_positions.shape[1], dtype=torch.int64)
            first_bad.scatter_reduce_(0, owner[bad], sample[bad], reduce="amin")
            if truncate_at_first_unreachable:
                shortened = (first_bad >= 2) & (first_bad < lengths)
                lengths = torch.where(shortened, first_bad, lengths)
                bank.length[rows] = lengths.to(bank.length.dtype)
            retained = sample < lengths[owner]
            rejection_count = torch.zeros(len(rows), device=fsm.device, dtype=torch.int32)
            rejection_count.scatter_reduce_(
                0, owner, (bad & retained).to(torch.int32), reduce="amax"
            )
            rejected = rejection_count != 0
            diagnostic_error = torch.nan_to_num(error, nan=torch.inf, posinf=torch.inf)
            diagnostic_angle = torch.nan_to_num(angle, nan=torch.inf, posinf=torch.inf)
            max_position = torch.zeros(len(rows), device=fsm.device).scatter_reduce_(
                0, owner[retained], diagnostic_error[retained].to(torch.float32), reduce="amax"
            )
            max_orientation = torch.zeros_like(max_position).scatter_reduce_(
                0, owner[retained], diagnostic_angle[retained].to(torch.float32), reduce="amax"
            )
            worst = torch.full_like(lengths, len(sample), dtype=torch.int64)
            batch_index = torch.arange(len(sample), device=fsm.device)
            maxima = retained & (diagnostic_error == max_position[owner])
            worst.scatter_reduce_(0, owner[maxima], batch_index[maxima], reduce="amin")
            worst = worst.clamp_max(max(len(sample) - 1, 0))
            conditions = {
                "path_inverse_position_residual_m": (
                    max_position <= prediction_tolerance,
                    max_position,
                    "<=",
                    prediction_tolerance,
                ),
                "path_inverse_orientation_residual_rad": (
                    max_orientation <= math.radians(1.0),
                    max_orientation,
                    "<=",
                    math.radians(1.0),
                ),
            }
            for stage in (
                self.Stage.PRESS_HANDLE,
                self.Stage.OPEN_DOOR_WITH_HANDLE,
                self.Stage.RESET_HANDLE,
                self.Stage.PRY_DOOR_OPEN,
            ):
                selected = fsm.stage[rows] == int(stage)
                stage_conditions = {
                    name: (passed[selected], values[selected], relation, threshold)
                    for name, (passed, values, relation, threshold) in conditions.items()
                }
                fsm.record_acceptance(stage, rows[selected], stage_conditions)
            fsm.fail(rows[rejected], 49)
            self.path_diagnostics.append(
                (
                    label,
                    rows.clone(),
                    {
                        "sample_count": lengths.clone(),
                        "max_position_error_m": max_position,
                        "max_orientation_error_rad": max_orientation,
                        "worst_position_sample": sample[worst],
                        "desired_position_world_m": desired_positions[
                            torch.arange(len(rows), device=fsm.device), sample[worst]
                        ],
                        "predicted_position_world_m": desired_positions[
                            torch.arange(len(rows), device=fsm.device), sample[worst]
                        ]
                        + result.position_error[worst],
                        "inverse_command_position_world_m": result.command_position_world[worst],
                        "solved_arm_joints_rad": result.arm_joint_positions[worst],
                        "first_unreachable_sample": first_bad,
                    },
                )
            )
            self.stats["errors"] = self.stats["errors"] + rejected.sum()
        self._record_stats(label, len(rows))

    def plan_pull_open(self, fsm, device_state, rows):
        if not len(rows):
            return
        self.plan_pose_path(
            fsm,
            device_state,
            rows,
            bank_name="pull_open_path",
            label="pull_panel_arc",
            prediction_tolerance=fsm.pull_panel_config.position_tolerance_m,
            truncate_at_first_unreachable=True,
        )

    def plan_initial_pull_station(
        self, fsm, device_state, rows, *, arrival_xy_error=None, arrival_yaw_error=None
    ):
        if not len(rows):
            return
        config = fsm.pull_station_config
        fsm.pull_first_station_attempts[rows] += 1
        xy, yaw, local = station_candidates(
            fsm.hinge_world[rows],
            fsm.door_normal[rows],
            fsm.pull_free_direction[rows],
            fsm.pull_frame_edge[rows],
            fsm.pull_wall_front[rows],
            fsm.sample_initial_pull_stations(rows),
            config,
        )
        wall, frame = station_clearances(
            device_state["root_states"][rows],
            device_state["robot_body_state_w"][rows],
            fsm.pull_motion_points,
            fsm.pull_motion_body_indices,
            xy,
            yaw,
            fsm.door_normal[rows],
            fsm.pull_free_direction[rows],
            fsm.hinge_world[rows],
            fsm.pull_frame_edge[rows],
            fsm.pull_wall_front[rows],
        )
        clearance_safe = (wall >= config.wall_clearance_m + config.planning_margin_m) & (
            frame >= config.clearance_m + config.planning_margin_m
        )
        nominal_clearance_safe = clearance_safe.clone()
        candidate_count = xy.shape[1]
        evaluation_xy, evaluation_yaw, evaluation_safe = xy, yaw, clearance_safe
        if arrival_xy_error is not None:
            forecast_xy = xy + arrival_xy_error[:, None]
            forecast_yaw = yaw + arrival_yaw_error[:, None]
            forecast_wall, forecast_frame = station_clearances(
                device_state["root_states"][rows],
                device_state["robot_body_state_w"][rows],
                fsm.pull_motion_points,
                fsm.pull_motion_body_indices,
                forecast_xy,
                forecast_yaw,
                fsm.door_normal[rows],
                fsm.pull_free_direction[rows],
                fsm.hinge_world[rows],
                fsm.pull_frame_edge[rows],
                fsm.pull_wall_front[rows],
            )
            forecast_safe = (
                forecast_wall >= config.wall_clearance_m + config.planning_margin_m
            ) & (forecast_frame >= config.clearance_m + config.planning_margin_m)
            distance = torch.linalg.vector_norm(
                xy - device_state["root_states"][rows, None, :2], dim=-1
            )
            angle_delta = yaw - _yaw_from_wxyz(device_state["root_states"][rows, 3:7])[:, None]
            angle_distance = torch.atan2(angle_delta.sin(), angle_delta.cos()).abs()
            # A fresh arrival latch would otherwise immediately stop without
            # moving the rejected actual stance to the new nominal target.
            moves = (distance >= APPROACH_ARRIVAL.inner_distance_m) | (
                angle_distance >= APPROACH_ARRIVAL.inner_yaw_rad
            )
            clearance_safe &= forecast_safe & moves
            evaluation_xy = torch.cat((xy, forecast_xy), dim=1)
            evaluation_yaw = torch.cat((yaw, forecast_yaw), dim=1)
            evaluation_safe = torch.cat((clearance_safe, clearance_safe), dim=1)
        inverse_safe, error, angle = first_insertion_residuals(
            self.pull_geometry,
            device_state,
            rows,
            evaluation_xy,
            evaluation_yaw,
            evaluation_safe,
            fsm.pull_cycle_pose_grid,
            self.arm_joint_ids,
            self.joint_limits,
            fsm.pull_frame_edge[rows],
            fsm.pull_wall_front[rows],
            fsm.pull_hardware_z[rows],
            fsm.pull_panel_config,
            (
                fsm.env.get_goal_reference_origin(),
                fsm.env.get_goal_reference_quat(),
                fsm.env.get_goal_center_offset_local(),
            ),
        )
        forecast_feasible = torch.ones_like(clearance_safe)
        forecast_score = torch.zeros_like(wall)
        if arrival_xy_error is not None:
            forecast_feasible = inverse_safe[:, candidate_count:].any(-1)
            forecast_score = torch.maximum(
                error[:, candidate_count:] / fsm.pull_panel_config.position_tolerance_m,
                angle[:, candidate_count:] / math.radians(1.0),
            ).amin(-1)
            forecast_score = torch.nan_to_num(forecast_score, nan=torch.finfo(wall.dtype).max)
            inverse_safe = inverse_safe[:, :candidate_count]
        feasible = clearance_safe & inverse_safe.any(-1)
        accepted = feasible.any(-1)
        selected, _ = best_feasible(feasible, forecast_score)
        owners = torch.arange(len(rows), device=fsm.device)
        chosen_owner = owners[accepted]
        chosen = selected[accepted]
        rotation = (
            0.5
            * fsm.hinge_axis_sign[rows]
            * device_state["door_joint_position"][rows, fsm.hinge_idx]
        )
        tangent = fsm.pull_free_direction[rows]
        c, s = rotation.cos(), rotation.sin()
        # Gap midpoint is diagnostic only; station coordinates use the frame edge.
        edge = fsm.pull_free_edge[rows, None] * tangent
        rotated = torch.stack(
            (c * edge[:, 0] - s * edge[:, 1], s * edge[:, 0] + c * edge[:, 1]), -1
        )
        gap = fsm.hinge_world[rows, :2] + c[:, None] * rotated
        fsm.install_initial_pull_station(
            rows[accepted],
            xy[chosen_owner, chosen],
            yaw[chosen_owner, chosen],
            gap[accepted],
            frame[chosen_owner, chosen],
            wall[chosen_owner, chosen],
        )
        fsm.pull_initial_station_history.append_batch(
            rows,
            {
                "entry_root_state": device_state["root_states"][rows].clone(),
                "entry_arm_joint_position_rad": device_state["dof_pos"][rows][
                    :, self.arm_joint_ids
                ].clone(),
                "candidate_local_x_y_yaw_deg": local,
                "candidate_xy_world_m": xy,
                "candidate_yaw_world_rad": yaw,
                "wall_clearance_m": wall,
                "frame_clearance_m": frame,
                "nominal_clearance_safe": nominal_clearance_safe,
                "clearance_safe": clearance_safe,
                "insertion_candidate_feasible": inverse_safe,
                "path_max_position_error_m": error[:, :candidate_count],
                "path_max_orientation_error_rad": angle[:, :candidate_count],
                "forecast_path_max_position_error_m": error[:, candidate_count:],
                "forecast_path_max_orientation_error_rad": angle[:, candidate_count:],
                "forecast_is_ranking_only": True,
                "candidate_feasible": feasible,
                "selected_attempt": torch.where(accepted, selected, -1),
                "station_attempt": fsm.pull_first_station_attempts[rows].clone(),
                "arrival_xy_error_world_m": torch.zeros_like(xy[:, 0])
                if arrival_xy_error is None
                else arrival_xy_error,
                "arrival_yaw_error_rad": torch.zeros_like(yaw[:, 0])
                if arrival_yaw_error is None
                else arrival_yaw_error,
                "forecast_insertion_feasible": forecast_feasible,
            },
        )
        fsm.record_acceptance(
            self.Stage.MOVE_TO_GAP_STATION,
            rows,
            {
                "station_clearance_candidate_exists": (
                    clearance_safe.any(-1),
                    clearance_safe.sum(-1),
                    ">=",
                    1,
                ),
                "station_full_insertion_candidate_exists": (accepted, feasible.sum(-1), ">=", 1),
            },
        )
        fsm.fail(rows[~accepted], 55)
        self._record_stats("first_pull_station", len(rows), (~accepted).sum())

    def retry_first_pull_station(self, fsm, device_state, rows):
        """Re-enter the same station controller, with measured arrival error."""
        xy_error = device_state["root_states"][rows, :2] - fsm.pull_station_xy[rows]
        yaw_error = (
            _yaw_from_wxyz(device_state["root_states"][rows, 3:7]) - fsm.pull_station_yaw[rows]
        )
        yaw_error = torch.atan2(yaw_error.sin(), yaw_error.cos())
        fsm.pull_cycle[rows] = 0
        fsm.begin_move_to_gap_station(device_state, rows)
        self.plan_initial_pull_station(
            fsm, device_state, rows, arrival_xy_error=xy_error, arrival_yaw_error=yaw_error
        )

    def plan_pull_insert(self, fsm, device_state, rows):
        if not len(rows):
            return
        cycle = fsm.pull_cycle[rows].clone()
        draws = fsm.sample_pull_cycle_candidates(rows)
        sample_names = ("height_m", "insertion_m", "yaw_rad", "station_angle_rad")
        samples = dict(zip(sample_names, draws.unbind(-1)))
        first_entry = cycle == 0
        geometry = {}
        for initial in (True, False):
            selected = torch.where(first_entry == initial)[0]
            if not len(selected):
                continue
            group = rows[selected]
            group_samples = {key: value[selected] for key, value in samples.items()}
            kwargs = dict(
                frame_edge_m=fsm.pull_frame_edge[group],
                wall_front_m=fsm.pull_wall_front[group],
                hardware_height_bounds=fsm.pull_hardware_z[group],
            )
            reference = (
                fsm.env.get_goal_reference_origin()[group],
                fsm.env.get_goal_reference_quat()[group],
                fsm.env.get_goal_center_offset_local()[group],
            )
            if initial:
                result = plan_first_pull_entry(
                    self.pull_geometry,
                    device_state,
                    group,
                    group_samples,
                    config=fsm.pull_panel_config,
                    goal_reference=reference,
                    include_approach=True,
                    **kwargs,
                )
            else:
                result = self.pull_geometry.plan(device_state, group, group_samples, **kwargs)
            radius = insertion_endpoint_radius(
                result["endpoint"], device_state["root_states"][group], reference
            )
            safe = radius <= fsm.pull_panel_config.insertion_max_horizontal_radius_m
            result["endpoint_horizontal_radius_m"] = radius
            result["endpoint_radius_safe"] = safe
            result["feasible"] &= safe
            for key, value in result.items():
                if key not in geometry:
                    geometry[key] = value.new_empty((len(rows), *value.shape[1:]))
                geometry[key][selected] = value
        feasible_geometry = geometry["feasible"]
        candidate_owner, attempt = torch.where(feasible_geometry)
        positions = geometry["positions"][candidate_owner, attempt]
        orientations = geometry["orientations"][candidate_owner, attempt]
        lengths = geometry["lengths"][candidate_owner, attempt]
        max_position = torch.full(feasible_geometry.shape, torch.inf, device=fsm.device)
        max_orientation = torch.full_like(max_position, torch.inf)
        face_gap = torch.full_like(max_position, torch.inf)
        worst_sample = torch.full(feasible_geometry.shape, -1, device=fsm.device, dtype=torch.int64)
        worst_desired = torch.full((*feasible_geometry.shape, 3), torch.nan, device=fsm.device)
        worst_predicted = torch.full_like(worst_desired, torch.nan)
        worst_command = torch.full_like(worst_desired, torch.nan)
        worst_joints = torch.full((*feasible_geometry.shape, 6), torch.nan, device=fsm.device)
        solved_positions = positions.clone()
        solved_orientations = orientations.clone()
        solved_joints = positions.new_zeros((*positions.shape[:2], 6))
        wall_clearance = torch.full_like(max_position, torch.inf)
        if len(candidate_owner):
            result, owner, sample = self._inverse_pose_bank(
                fsm, device_state, rows[candidate_owner], positions, orientations, lengths
            )
            solved_positions[owner, sample] = result.command_position_world
            solved_orientations[owner, sample] = result.command_orientation_world_wxyz
            solved_joints[owner, sample] = result.arm_joint_positions
            initial_points = torch.where(first_entry[candidate_owner[owner]])[0]
            if len(initial_points):
                from gr00t.rl.auto_open.first_pull_wall import FirstPullArmWallGPU

                if not hasattr(self, "_first_pull_wall"):
                    self._first_pull_wall = FirstPullArmWallGPU(device=fsm.device)
                env_rows = rows[candidate_owner[owner[initial_points]]]
                gap = self._first_pull_wall.clearance(
                    device_state["root_states"][env_rows],
                    result.arm_joint_positions[initial_points],
                    fsm.q_closed[env_rows],
                    self.pull_geometry.hinge_world[env_rows],
                    self.pull_geometry.door_normal[env_rows],
                    self.pull_geometry.tangent[env_rows],
                    fsm.pull_frame_edge[env_rows],
                    fsm.pull_wall_front[env_rows],
                )
                per_candidate_gap = torch.full(
                    (len(candidate_owner),), torch.inf, device=fsm.device, dtype=gap.dtype
                )
                per_candidate_gap.scatter_reduce_(0, owner[initial_points], gap, reduce="amin")
                wall_clearance[candidate_owner, attempt] = per_candidate_gap.to(
                    wall_clearance.dtype
                )
            error = torch.linalg.vector_norm(result.position_error, dim=-1)
            angle = 2.0 * torch.asin(
                torch.linalg.vector_norm(result.orientation_error, dim=-1).clamp(0.0, 1.0)
            )
            per_candidate_position = torch.zeros(
                len(candidate_owner), device=fsm.device
            ).scatter_reduce_(0, owner, torch.nan_to_num(error, nan=torch.inf), reduce="amax")
            per_candidate_orientation = torch.zeros_like(per_candidate_position).scatter_reduce_(
                0, owner, torch.nan_to_num(angle, nan=torch.inf), reduce="amax"
            )
            max_position[candidate_owner, attempt] = per_candidate_position
            max_orientation[candidate_owner, attempt] = per_candidate_orientation
            worst = torch.full(
                (len(candidate_owner),), len(sample), dtype=torch.int64, device=fsm.device
            )
            diagnostic_error = torch.nan_to_num(error, nan=torch.inf, posinf=torch.inf)
            maxima = diagnostic_error == per_candidate_position[owner]
            indices = torch.arange(len(sample), device=fsm.device)
            worst.scatter_reduce_(0, owner[maxima], indices[maxima], reduce="amin")
            worst = worst.clamp_max(max(len(sample) - 1, 0))
            worst_sample[candidate_owner, attempt] = sample[worst]
            desired = positions[
                torch.arange(len(candidate_owner), device=fsm.device), sample[worst]
            ]
            worst_desired[candidate_owner, attempt] = desired
            worst_predicted[candidate_owner, attempt] = desired + result.position_error[worst]
            worst_command[candidate_owner, attempt] = result.command_position_world[worst]
            worst_joints[candidate_owner, attempt] = result.arm_joint_positions[worst]
            endpoint = lengths.to(torch.int64) - 1
            candidate_index = torch.arange(len(candidate_owner), device=fsm.device)
            face = solved_positions[candidate_index, endpoint] + quat_rotate(
                solved_orientations[candidate_index, endpoint],
                geometry["face_offset_local"][candidate_owner, attempt],
            )
            face_gap[candidate_owner, attempt] = (
                (geometry["contact_point"][candidate_owner, attempt] - face)
                * geometry["normal"][candidate_owner, attempt]
            ).sum(-1)
        tolerance = fsm.pull_panel_config.position_tolerance_m
        feasible = (
            feasible_geometry
            & (max_position <= tolerance)
            & (max_orientation <= math.radians(1.0))
            & (first_entry[:, None] | ((face_gap >= -0.001) & (face_gap <= 0.010)))
        )
        valid_segments = (
            torch.arange(solved_joints.shape[1] - 1, device=fsm.device)[None] < lengths[:, None] - 1
        )
        joint_delta = (solved_joints[:, 1:] - solved_joints[:, :-1]).abs().amax(-1)
        timed_lengths = 1 + torch.where(
            valid_segments, torch.ceil(joint_delta / 0.01).clamp_min(1), 0
        ).sum(1)
        path_smooth = torch.where(valid_segments, joint_delta, 0.0).amax(1) <= 0.15
        path_safe = (
            (wall_clearance[candidate_owner, attempt] >= 0.01)
            & path_smooth
            & (timed_lengths <= positions.shape[1])
        )
        feasible[candidate_owner, attempt] &= ~first_entry[candidate_owner] | path_safe
        selected, _ = first_feasible(feasible)
        accepted = feasible.any(1)
        best_score = torch.maximum(max_position / tolerance, max_orientation / math.radians(1.0))
        diagnostic_attempt = torch.where(accepted, selected, best_score.argmin(1))
        local_rows = torch.arange(len(rows), device=fsm.device)
        fsm.record_acceptance(
            self.Stage.INSERT_INTO_DOOR_GAP,
            rows,
            {
                "endpoint_horizontal_radius_m": (
                    geometry["endpoint_radius_safe"][local_rows, diagnostic_attempt],
                    geometry["endpoint_horizontal_radius_m"][local_rows, diagnostic_attempt],
                    "<=",
                    fsm.pull_panel_config.insertion_max_horizontal_radius_m,
                ),
                "path_inverse_position_residual_m": (
                    max_position[local_rows, diagnostic_attempt] <= tolerance,
                    max_position[local_rows, diagnostic_attempt],
                    "<=",
                    tolerance,
                ),
                "path_inverse_orientation_residual_rad": (
                    max_orientation[local_rows, diagnostic_attempt] <= math.radians(1.0),
                    max_orientation[local_rows, diagnostic_attempt],
                    "<=",
                    math.radians(1.0),
                ),
                "insertion_candidate_feasible": (accepted, accepted.to(torch.float32), "==", 1.0),
                "first_approach_wall_clearance_m": (
                    ~first_entry | (wall_clearance[local_rows, diagnostic_attempt] >= 0.01),
                    wall_clearance[local_rows, diagnostic_attempt],
                    ">=",
                    0.01,
                ),
            },
        )
        lookup = torch.full(feasible.shape, -1, dtype=torch.int64, device=fsm.device)
        lookup[candidate_owner, attempt] = torch.arange(len(candidate_owner), device=fsm.device)
        chosen = lookup[local_rows[accepted], selected[accepted]]
        accepted_rows = rows[accepted]
        chosen_attempt = selected[accepted]
        chosen_owner = local_rows[accepted]
        fsm.pull_cycle_samples[accepted_rows] = draws[chosen_owner, chosen_attempt]
        final_positions = solved_positions[chosen].clone()
        final_orientations = solved_orientations[chosen].clone()
        final_lengths = lengths[chosen].clone()
        initial_chosen = torch.where(first_entry[chosen_owner])[0]
        if len(initial_chosen):
            from gr00t.rl.auto_open.first_pull_wall import retime_first_pull_paths

            timed_p, timed_q, timed_n, _ = retime_first_pull_paths(
                final_positions[initial_chosen],
                final_orientations[initial_chosen],
                solved_joints[chosen[initial_chosen]],
                final_lengths[initial_chosen],
            )
            final_positions[initial_chosen], final_orientations[initial_chosen] = timed_p, timed_q
            final_lengths[initial_chosen] = timed_n
        fsm.install_pull_insert_tensor(
            accepted_rows,
            final_positions,
            final_orientations,
            final_lengths,
            geometry["contact_point"][chosen_owner, chosen_attempt],
            geometry["orientation"][chosen_owner, chosen_attempt],
            geometry["face_offset_local"][chosen_owner, chosen_attempt],
            endpoint=geometry["endpoint"][chosen_owner, chosen_attempt],
        )
        fsm.record_pull_cycle(
            rows,
            {
                "cycle": cycle,
                "candidate_samples": draws,
                "analytic_first_entry": first_entry,
                "first_approach_wall_clearance_m": wall_clearance,
                "first_entry_continuous_approach": first_entry,
                "pointwise_collision_checked": ~first_entry,
                "entry_ee_position_world_m": device_state["ee_pos"][rows].clone(),
                "entry_ee_orientation_world_wxyz": device_state["ee_orn"][rows].clone(),
                "entry_root_state": device_state["root_states"][rows].clone(),
                "entry_robot_body_poses_world_wxyz": device_state["robot_body_state_w"][
                    rows, :, :7
                ].clone(),
                "candidate_endpoint_world_m": geometry["endpoint"],
                "endpoint_horizontal_radius_m": geometry["endpoint_horizontal_radius_m"],
                "endpoint_radius_safe": geometry["endpoint_radius_safe"],
                "insertion_max_horizontal_radius_m": torch.full_like(
                    selected,
                    fsm.pull_panel_config.insertion_max_horizontal_radius_m,
                    dtype=torch.float32,
                ),
                "candidate_orientation_world_wxyz": geometry["orientation"],
                "candidate_face_offset_local_m": geometry["face_offset_local"],
                "geometry_feasible": feasible_geometry,
                "candidate_feasible": feasible,
                "geometry_constraints_pass": geometry.get("geometry_safe", feasible_geometry),
                "route_clear": geometry.get("route_safe", feasible_geometry),
                "route_start_free": geometry.get("route_start_free", feasible_geometry),
                "route_end_free": geometry.get("route_end_free", feasible_geometry),
                "route_direct": geometry.get("route_direct", feasible_geometry),
                "collision_clear": geometry.get("collision_safe", feasible_geometry),
                "path_buffer_overflow": geometry.get(
                    "overflow", torch.zeros_like(feasible_geometry)
                ),
                "max_position_error_m": max_position,
                "max_orientation_error_rad": max_orientation,
                "predicted_face_gap_m": face_gap,
                "worst_sample": worst_sample,
                "worst_desired_position_world_m": worst_desired,
                "worst_predicted_position_world_m": worst_predicted,
                "worst_command_position_world_m": worst_command,
                "worst_arm_joints_rad": worst_joints,
                "selected_attempt": torch.where(accepted, selected, -1),
                "tip_radius_m": geometry["tip_radius_m"][local_rows, diagnostic_attempt],
                "ee_tangent_radius_m": geometry["ee_tangent_radius_m"][
                    local_rows, diagnostic_attempt
                ],
                "gripper_height_bounds_m": geometry["gripper_height_bounds_m"][
                    local_rows, diagnostic_attempt
                ],
                "hardware_height_bounds_m": fsm.pull_hardware_z[rows].clone(),
            },
        )
        retry = torch.zeros_like(accepted)
        initial_rejected = torch.where(first_entry & ~accepted)[0]
        if len(initial_rejected):
            retry[initial_rejected] = (
                fsm.pull_first_station_attempts[rows[initial_rejected]]
                < fsm.pull_station_config.first_max_station_attempts
            )
        fsm.fail(rows[~feasible_geometry.any(1) & ~retry], 50)
        fsm.fail(rows[feasible_geometry.any(1) & ~accepted & ~retry], 49)
        if len(rows[retry]):
            self.retry_first_pull_station(fsm, device_state, rows[retry])
        self._record_stats("pull_panel_insertion", len(rows), (~accepted).sum())

    def finalize_diagnostics(self):
        """Materialize device diagnostics only after simulation has ended."""
        self.finalize_gpu_stage_diagnostics()
        for label, rows, quantities in self.path_diagnostics:
            values = {name: value.detach().cpu().tolist() for name, value in quantities.items()}
            for index, env_id in enumerate(rows.detach().cpu().tolist()):
                self.controllers[env_id].last_acceptance_diagnostics[f"{label}_inverse"] = {
                    name: value[index] for name, value in values.items()
                }
        errors = self.stats["errors"]
        if isinstance(errors, torch.Tensor):
            self.stats["errors"] = int(errors.item())
