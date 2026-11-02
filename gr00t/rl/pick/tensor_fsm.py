"""Pick stage definitions on the shared tensor execution/acceptance lifecycle."""

import math

import torch

from gr00t.rl.auto_open.acceptance import (
    StageAcceptanceMixin,
    TensorContactWindow,
    TensorJointStabilityWindow,
    TensorStabilityWindow,
    consecutive_steps,
)
from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory
from gr00t.rl.auto_open.control import APPROACH_ARRIVAL
from gr00t.rl.auto_open.execution import StageExecutionMixin
from gr00t.rl.auto_open.motion_math import _quat_distance
from gr00t.rl.auto_open.motion_targets import PaddedPathBank
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.auto_open.stages import TensorStageLifecycle
from gr00t.rl.pick.stages import BasketStage
from gr00t.rl.pick.command_compensation import LowTargetCommandCompensation
from gr00t.rl.pick.controller import PickStage, pose_window_speed
from gr00t.rl.pick.gaze import gaze_from_actual_camera
from gr00t.rl.pick.stages import stage_definitions

FAILURES = {
    1: "unexpected_nonfoot_scene_contact_over_20N_for_3_frames",
    2: "robot_base_height_below_0.18m",
    3: "object_lost_from_gripper",
    4: "stage_acceptance_timeout",
    5: "basket_reposition_stage_acceptance_timeout",
    6: "basket_reposition_object_or_robot_outside_safe_bounds",
    7: "approach_gaze_no_collision_free_aimed_trajectory",
    8: "episode_step_budget_exhausted",
    9: "stage_entry_planning_failed",
    11: "seated_carry_path_geometry_or_inverse_failed",
    12: "no_safe_grasp_candidate",
    13: "missing_verified_grasp_support_or_nonstandard_planner",
}


class TensorPickFSM(TensorStageLifecycle, StageExecutionMixin, StageAcceptanceMixin):
    Stage = BasketStage

    def __init__(self, env, controllers):
        self.env, self.device, self.num_envs = env, env.device, env.num_envs
        if len(controllers) != env.num_envs:
            raise ValueError("Each batch row requires its own task configuration")
        self.controllers = controllers
        self.Stage, self.stage_specs, self.interfaces = stage_definitions(
            basket=getattr(controllers[0], "basket", None) is not None
        )
        self.rows = torch.arange(self.num_envs, device=self.device)
        self.stage = torch.tensor([int(c.current_stage) for c in controllers], device=self.device)
        self.stage_entry_from = self.stage.clone()
        self.stage_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.stage_visit = torch.zeros(self.num_envs, 14, device=self.device, dtype=torch.int32)
        self.stage_visit[self.rows, self.stage] = 1
        self.stage_limits = torch.zeros(14, device=self.device, dtype=torch.int32)
        for stage, spec in self.stage_specs.items():
            self.stage_limits[int(stage)] = spec.budget_steps or 0
        self._acceptance = {}
        self.failure_code = torch.zeros_like(self.stage)
        self.failure_from_stage = torch.zeros_like(self.stage)
        self.settled = torch.zeros_like(self.stage_steps)
        self.collision_steps = torch.zeros_like(self.stage_steps)
        self.drop_steps = torch.zeros_like(self.stage_steps)
        self.grasp_retry_steps = torch.zeros_like(self.stage_steps)
        self.arm_stability = TensorStabilityWindow(
            self.num_envs, self.device, max_position_span_m=0.003, max_orientation_span_rad=0.015
        )
        self.object_stability = TensorStabilityWindow(
            self.num_envs, self.device, max_position_span_m=0.003, max_orientation_span_rad=0.03
        )
        self.gripper_stability = TensorJointStabilityWindow(self.num_envs, 1, self.device)
        self.contact_window = TensorContactWindow(self.num_envs, 2, self.device)
        self.arm_path = PaddedPathBank(self.num_envs, 4096, self.device)
        self.base_path = BasePoseTrajectory(self.num_envs, self.device)
        for name, value in vars(controllers[0]).items():
            if name.startswith("base_") or name in ("arm_induced_pitch", "through_min_base_height"):
                if torch.is_tensor(value) and value.ndim and value.shape[0] == 1:
                    setattr(self, name, torch.cat([getattr(c, name) for c in controllers]))
        self.hold_xy = torch.cat([c.hold_xy for c in controllers])
        self.hold_yaw = torch.cat([c.hold_yaw for c in controllers])
        self.target = torch.cat([c.target for c in controllers])
        self.gaze_offsets = torch.cat(
            [getattr(c, "gaze_offsets", self.hold_xy.new_zeros(1, 3)) for c in controllers]
        )
        self.approach_gaze_valid = torch.tensor(
            [c.approach_gaze_valid for c in controllers], device=self.device
        )
        self.initial_object_z = torch.cat([c.initial_object_z for c in controllers])
        self.min_lift_m = self.hold_yaw.new_full((self.num_envs,), 0.10)
        self.min_grip_force_n = self.hold_yaw.new_tensor(
            [float(c.min_grip_force_n) for c in controllers]
        )
        self.grasp_object_relative = self.hold_xy.new_zeros(self.num_envs, 3)
        self.withdraw_displacement = self.hold_xy.new_zeros(self.num_envs, 3)
        self.grasp_retries = torch.zeros_like(self.stage_steps)
        self.tried_grasp_orientations = self.target.new_zeros(self.num_envs, 8, 4)
        self.tried_grasp_mask = torch.zeros(self.num_envs, 8, device=self.device, dtype=torch.bool)
        self.awaiting_entry_refresh = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.basket_final_entry = torch.zeros_like(self.awaiting_entry_refresh)
        self.basket_entry_refresh_count = torch.zeros_like(self.stage_steps)
        self.basket_entry_orientation = self.target.new_zeros(self.num_envs, 4)
        self.basket_entry_orientation[:, 0] = 1
        self.basket_entry_object_orientation = self.basket_entry_orientation.clone()
        self.basket_entry_depth_fraction = self.target.new_full((self.num_envs,), torch.nan)
        self.nudge_attempts = torch.zeros_like(self.stage_steps)
        self.nudge_start = torch.zeros_like(self.grasp_object_relative)
        self.nudge_direction = torch.zeros_like(self.grasp_object_relative)
        self.nudge_space_before = torch.zeros_like(self.hold_yaw)
        self.nudge_current_path_unavailable = torch.zeros_like(self.awaiting_entry_refresh)
        self.nudge_performed = torch.zeros_like(self.awaiting_entry_refresh)
        self.contact_body_names = controllers[0].contact_body_names
        self.nonfoot_contact_ids = controllers[0].nonfoot_contact_ids
        self.jaw_contact_ids = controllers[0].jaw_contact_ids
        self.basket = getattr(controllers[0], "basket", None)
        self.command_compensation = (
            LowTargetCommandCompensation(self.num_envs, self.device)
            if self.basket is None
            else None
        )
        self.basket_payload_sizes = None
        if self.basket is not None:
            from gr00t.rl.pick.tensor_rows import pad_device_rows

            descriptors = [[c.episode, *c.episode.get("distractors", [])] for c in controllers]
            maximum = max(map(len, descriptors))
            self.basket_payload_sizes = self.target.new_tensor(
                [
                    [item["asset"]["size_m"] for item in row] + [[0, 0, 0]] * (maximum - len(row))
                    for row in descriptors
                ]
            )
            lengths = self.rows.new_tensor(list(map(len, descriptors)))
            self.payload_mask = torch.arange(maximum, device=self.device)[None] < lengths[:, None]
            self.basket_lift_margin = self.hold_yaw.new_tensor([c.lift_margin for c in controllers])
            self.object_support, self.object_support_mask = pad_device_rows(
                [c.object_support_points for c in controllers]
            )
        self.install_plans(self.rows, controllers, initial=True)
        self.fail(
            torch.where(self.mask(self.Stage.APPROACH_OBJECT) & ~self.approach_gaze_valid)[0], 7
        )

    def install_plans(self, rows, controllers, *, initial=False):
        """Batch configuration/path installation, never read an individual device scalar."""
        from gr00t.rl.pick.tensor_rows import pack_scalars, pad_device_rows

        old = self.stage[rows].clone()
        stages = self.stage.new_tensor([int(c.current_stage) for c in controllers])
        failed = stages == int(self.Stage.FAILURE)
        self.fail(rows[failed], 9)
        if not initial:
            for stage in self.Stage:
                if stage != self.Stage.FAILURE:
                    self.transition(rows[(stages == int(stage)) & (stages != old)], stage)
        self.stage_steps[rows] = 0
        self.settled[rows] = 0
        self.grasp_retry_steps[rows] = 0
        self.arm_stability.clear(rows)
        self.hold_xy[rows] = torch.cat([c.hold_xy for c in controllers])
        self.hold_yaw[rows] = torch.cat([c.hold_yaw for c in controllers])
        self.grasp_retries[rows] = self.grasp_retries.new_tensor(
            [c.grasp_retries for c in controllers]
        )
        histories = [c.tried_grasp_orientations for c in controllers]
        if max(map(len, histories)) > self.tried_grasp_orientations.shape[1]:
            raise ValueError("Grasp retry orientation history exceeds its fixed capacity")
        history, history_mask = pad_device_rows(histories)
        self.tried_grasp_mask[rows] = False
        self.tried_grasp_mask[rows, : history.shape[1]] = history_mask
        self.tried_grasp_orientations[rows, : history.shape[1]] = history
        for destination, source, default in (
            ("awaiting_entry_refresh", "awaiting_entry_refresh", False),
            ("basket_final_entry", "final_entry", False),
            ("basket_entry_refresh_count", "entry_refresh_count", 0),
            ("nudge_attempts", "nudge_attempts", 0),
            ("min_lift_m", "min_lift_m", 0.1),
        ):
            tensor = getattr(self, destination)
            tensor[rows] = pack_scalars([getattr(c, source, default) for c in controllers], tensor)
        for destination, source in (
            ("basket_entry_orientation", "entry_orientation"),
            ("basket_entry_object_orientation", "entry_object_orientation"),
            ("grasp_object_relative", "grasp_object_relative"),
            ("nudge_start", "nudge_start"),
            ("nudge_direction", "nudge_direction"),
        ):
            tensor = getattr(self, destination)
            defaults = tensor[rows]
            values = [getattr(c, source, None) for c in controllers]
            tensor[rows] = torch.stack(
                [
                    defaults[index] if value is None else value.reshape(tensor.shape[1:])
                    for index, value in enumerate(values)
                ]
            )
        self.nudge_space_before[rows] = pack_scalars(
            [getattr(c, "nudge_space_before", 0.0) for c in controllers], self.nudge_space_before
        )
        self.nudge_current_path_unavailable[rows] = self.nudge_current_path_unavailable.new_tensor(
            [
                getattr(c, "active_nudge_report", {}).get("trigger")
                not in (None, "explicit_debug_probe")
                for c in controllers
            ]
        )
        lengths = torch.cat([c.arm_path.length for c in controllers])
        hold = (stages == int(self.Stage.GRASP_OBJECT)) | (
            stages == int(self.Stage.VERIFY_RETENTION)
        )
        active = torch.where((lengths > 0) & (~hold | initial))[0]
        self.arm_path.install_tensor(
            rows[active],
            torch.cat([c.arm_path.positions for c in controllers])[active],
            torch.cat([c.arm_path.orientations for c in controllers])[active],
            lengths[active],
        )
        if initial:
            for name in ("poses", "velocities", "length", "index"):
                getattr(self.base_path, name)[rows] = torch.cat(
                    [getattr(c.base_path, name) for c in controllers]
                )
        self.failure_from_stage[rows[failed]] = old[failed]

    def command(self, state, buffers):
        compensation = (
            getattr(self, "command_compensation", None)
            if getattr(self, "basket", None) is None
            else None
        )
        terminal = (self.stage == 7) | (self.stage == 8)
        self.stop_base(torch.where(terminal)[0], buffers)
        approach = torch.where(self.stage == int(PickStage.APPROACH_OBJECT))[0]
        if len(approach):
            pose, velocity, complete = self.base_path.sample(approach)
            self.base_target(
                state,
                approach,
                pose[:, :2],
                pose[:, 2],
                buffers,
                acceptance_xy=self.hold_xy[approach],
                acceptance_yaw=self.hold_yaw[approach],
                reference_velocity_world=velocity,
                arrival=True,
                arrival_enabled=complete,
                toward_target_only=True,
            )
            p, q = self.arm_path.current(approach, advance=True)
            blend = (
                (self.stage_steps[approach] - self.base_path.length[approach]) * self.env.dt / 0.5
            ).clamp(0, 1)
            p, q = gaze_from_actual_camera(
                p,
                q,
                state["ee_pos"][approach],
                state["ee_orn"][approach],
                self.target[approach],
                self.gaze_offsets[approach],
                blend,
            )
            if compensation is not None:
                disabled = torch.zeros(len(approach), device=self.device, dtype=torch.bool)
                compensation.apply(
                    approach,
                    p,
                    state["ee_pos"][approach],
                    adapt=disabled,
                    retain=disabled,
                    reset=~disabled,
                    dt=float(self.env.dt),
                )
            self.arm_pose_target(approach, p, q, buffers)
        normal = torch.where(~terminal & (self.stage != 1) & (self.stage < 9))[0]
        if len(normal):
            if self.basket is not None:
                self.stop_base(normal, buffers)
            else:
                self.base_target(
                    state,
                    normal,
                    self.hold_xy[normal],
                    self.hold_yaw[normal],
                    buffers,
                    arrival=True,
                )
        for stage in range(2, 14):
            if stage in (7, 8):
                continue
            rows = torch.where(self.stage == stage)[0]
            if not len(rows):
                continue
            if stage >= 9:
                self.stop_base(rows, buffers)
            command_offset = None
            if compensation is not None:
                intended, orientation = self.arm_path.current(rows, advance=False)
                calibrate = (
                    self.arm_path.complete(rows)
                    & (_quat_distance(state["ee_orn"][rows], orientation) < 0.10)
                    & (stage == int(PickStage.MOVE_TO_GRASP))
                )
                commanded = compensation.apply(
                    rows,
                    intended,
                    state["ee_pos"][rows],
                    adapt=calibrate,
                    retain=torch.full_like(calibrate, stage in (2, 3, 4, 5, 6)),
                    reset=(self.stage_steps[rows] == 0) & (stage == 2),
                    dt=float(self.env.dt),
                )
                command_offset = commanded - intended
            self.arm_trajectory(
                rows,
                self.arm_path,
                buffers,
                advance=stage in (2, 4, 5, 10, 12),
                **({"command_offset_world": command_offset} if command_offset is not None else {}),
            )
        closed = ((self.stage >= 3) & (self.stage <= 7)) | (self.stage == 11) | (self.stage == 12)
        active_rows = self.rows[~terminal]
        self.gripper_target(
            active_rows, torch.where(closed[active_rows], 0.0, -math.pi / 3), buffers
        )

    def transition(self, rows, stage):
        destination = self.Stage(int(stage))
        allowed = torch.zeros(len(rows), device=self.device, dtype=torch.bool)
        for source, spec in self.stage_specs.items():
            if destination in spec.successors:
                allowed |= self.mask(source)[rows]
        if not bool(allowed.all()):
            raise ValueError(f"Invalid pick stage transition to {destination.name}")
        self.stage_entry_from[rows] = self.stage[rows]
        super().transition(rows, destination)
        self.settled[rows] = 0
        self.grasp_retry_steps[rows] = 0
        self.arm_stability.clear(rows)
        if destination == self.Stage.GRASP_OBJECT:
            self.gripper_stability.clear(rows)
        elif destination == self.Stage.VERIFY_RETENTION:
            self.object_stability.clear(rows)

    def unexpected_contact_forces(self, state):
        forces = state["contact_forces_w"].clone()
        for jaw, body in enumerate(self.jaw_contact_ids):
            if body is not None:
                forces[:, body] -= state["object_finger_forces"][:, jaw]
        if self.basket is not None:
            body = self.contact_body_names.index("base_link")
            forces[:, body] -= state["basket_payload_normal_reaction_w"]
        return forces

    def _basket_measures(self, state):
        poses = torch.cat((state["object_state"][:, None], state["distractor_states"]), 1)
        signs = torch.cartesian_prod(*[self.target.new_tensor([-1, 1])] * 3)
        corners = _quat_rotate(
            poses[:, :, None, 3:7], signs[None, None] * self.basket_payload_sizes[:, :, None] / 2
        )
        corners += poses[:, :, None, :3]
        root = state["root_states"]
        inverse = root[:, 3:7].clone()
        inverse[:, 1:] *= -1
        local = _quat_rotate(inverse[:, None, None], corners - root[:, None, None, :3])
        low, high = local.amin(2), local.amax(2)
        center, half = (low + high) / 2, (high - low) / 2
        margin = torch.minimum(
            center[:, 0, :2] - self.basket.floor_lower[:2],
            self.basket.floor_upper[:2] - center[:, 0, :2],
        ).amin(-1)
        space = torch.minimum(
            low[:, 0, :2] - self.basket.floor_lower[:2],
            self.basket.floor_upper[:2] - high[:, 0, :2],
        ).amin(-1)
        if poses.shape[1] > 1:
            separation = (
                (center[:, :1, :2] - center[:, 1:, :2]).abs() - half[:, :1, :2] - half[:, 1:, :2]
            ).amax(-1)
            space = torch.minimum(
                space, separation.masked_fill(~self.payload_mask[:, 1:], torch.inf).amin(-1)
            )
        world = (
            _quat_rotate(root[:, None, 3:7], self.basket.boxes.reshape(-1, 3)[None])
            + root[:, None, :3]
        )
        points = _quat_rotate(state["object_state"][:, None, 3:7], self.object_support)
        bottom = state["object_state"][:, 2] + points[..., 2].masked_fill(
            ~self.object_support_mask, torch.inf
        ).amin(-1)
        return margin, space, bottom - world[..., 2].amax(-1)

    def advance(self, state):
        events = {}
        completed = {}
        expired_by_stage = {}
        reposition_rows = self.rows[:0]
        active = (self.stage != 7) & (self.stage != 8)
        self.stage_steps[active] += 1
        self.begin_acceptance_frame()
        stable, _, _ = self.arm_stability.update(state, self.rows)
        end = (self.arm_path.length - 1).clamp_min(0).long()
        error = (state["ee_pos"] - self.arm_path.positions[self.rows, end]).norm(dim=-1)
        angle = _quat_distance(state["ee_orn"], self.arm_path.orientations[self.rows, end])
        recent, recent_force, contact_age = self.contact_window.update(
            state["object_finger_forces"], state["object_finger_contact_points"]
        )
        closing = _quat_rotate(state["ee_orn"], self.target.new_tensor([0, 0, 1]))
        signed = (
            (self.contact_window.last_points - state["object_state"][:, None, :3])
            * closing[:, None]
        ).sum(-1)
        opposed = torch.isfinite(signed).all(-1) & (signed[:, 0] * signed[:, 1] < 0)
        contact = recent.all(-1) & opposed
        grip_stable, span = self.gripper_stability.update(
            state["dof_pos"][:, self.env.gripper_joint_ids], self.rows
        )
        inverse = state["ee_orn"].clone()
        inverse[:, 1:] *= -1
        relative = _quat_rotate(inverse, state["object_state"][:, :3] - state["ee_pos"])
        drift = (relative - self.grasp_object_relative).norm(dim=-1)
        elevation = state["object_state"][:, 2] - self.initial_object_z
        object_stable, _, _ = self.object_stability.update(
            {"ee_pos": state["object_state"][:, :3], "ee_orn": state["object_state"][:, 3:7]},
            self.rows,
        )
        speed = pose_window_speed(self.object_stability.positions, float(self.env.dt))
        force = (
            self.unexpected_contact_forces(state)[:, self.nonfoot_contact_ids].norm(dim=-1).amax(-1)
            if len(self.nonfoot_contact_ids)
            else self.hold_yaw.new_zeros(self.num_envs)
        )
        normal = active & (self.stage < 9)
        self.collision_steps[normal] = consecutive_steps(
            force[normal] >= 20, self.collision_steps[normal]
        )
        self.drop_steps = consecutive_steps(
            (drift >= 0.015) & (self.stage >= 4) & (self.stage <= 6), self.drop_steps
        )
        basket = self._basket_measures(state) if self.basket is not None else None
        # Stage transitions are committed after all current-stage evaluations.
        # A row is never evaluated twice in one physics frame, including cycles.
        for stage_enum, spec in self.stage_specs.items():
            stage = int(stage_enum)
            if stage_enum in (self.Stage.SUCCESS, self.Stage.FAILURE):
                continue
            rows = torch.where(self.stage == stage)[0]
            if not len(rows):
                continue
            conditions = {
                "base_height_m": (
                    state["root_states"][:, 2] >= 0.18,
                    state["root_states"][:, 2],
                    ">=",
                    0.18,
                ),
                "unexpected_scene_contact_n": (force < 20, force, "<", 20),
            }
            if stage == 1:
                distance = (state["root_states"][:, :2] - self.hold_xy).norm(dim=-1)
                conditions.update(
                    base_distance_m=(
                        distance < APPROACH_ARRIVAL.outer_distance_m,
                        distance,
                        "<",
                        APPROACH_ARRIVAL.outer_distance_m,
                    ),
                    base_stable=(
                        self.base_stability_confirmed,
                        self.base_stability_confirmed.float(),
                        "==",
                        1,
                    ),
                )
            elif stage in (2, 4, 5, 9, 10, 11, 12, 13):
                complete = self.arm_path.complete(self.rows)
                conditions.update(
                    path_complete=(complete, complete.float(), "==", 1),
                    ee_position_error_m=(error < 0.012, error, "<", 0.012),
                    ee_orientation_error_rad=(angle < 0.06, angle, "<", 0.06),
                    arm_stable=(stable, stable.float(), "==", 1),
                )
            if stage == 3:
                conditions.update(
                    {
                        "fixed_jaw_contact_age_frames": (recent[:, 0], contact_age[:, 0], "<=", 5),
                        "moving_jaw_contact_age_frames": (recent[:, 1], contact_age[:, 1], "<=", 5),
                        "fixed_jaw_recent_force_n": (
                            recent_force[:, 0] >= self.min_grip_force_n,
                            recent_force[:, 0],
                            ">=",
                            self.min_grip_force_n,
                        ),
                        "moving_jaw_recent_force_n": (
                            recent_force[:, 1] >= self.min_grip_force_n,
                            recent_force[:, 1],
                            ">=",
                            self.min_grip_force_n,
                        ),
                        "opposed_contacts": (opposed, opposed.float(), "==", 1),
                        "jaw_peak_force_n": (
                            state["object_finger_forces"].norm(dim=-1).amax(-1) <= 50,
                            state["object_finger_forces"].norm(dim=-1).amax(-1),
                            "<=",
                            50,
                        ),
                        "arm_stable": (stable, stable.float(), "==", 1),
                        "gripper_window_frames": (
                            self.gripper_stability.count >= self.gripper_stability.window_size,
                            self.gripper_stability.count.float(),
                            ">=",
                            self.gripper_stability.window_size,
                        ),
                        "gripper_position_span_rad": (
                            grip_stable,
                            span,
                            "<",
                            self.gripper_stability.max_span_rad,
                        ),
                    }
                )
            if stage in (4, 5, 6):
                conditions.update(
                    object_lift_m=(elevation >= self.min_lift_m, elevation, ">=", self.min_lift_m),
                    object_height_world_m=(
                        state["object_state"][:, 2] >= 0.30,
                        state["object_state"][:, 2],
                        ">=",
                        0.30,
                    ),
                    carried_object_relative_error_m=(drift < 0.015, drift, "<", 0.015),
                )
                if basket is not None:
                    clearance = basket[2]
                    conditions["object_bottom_above_basket_rim_m"] = (
                        clearance >= self.basket_lift_margin,
                        clearance,
                        ">=",
                        self.basket_lift_margin,
                    )
            if stage == 6:
                conditions.update(
                    object_speed_m_s=(speed < 0.05, speed, "<", 0.05),
                    bilateral_contact=(contact, contact.float(), "==", 1),
                    object_pose_stable=(object_stable, object_stable.float(), "==", 1),
                    ee_height_world_m=(
                        state["ee_pos"][:, 2] >= 0.30,
                        state["ee_pos"][:, 2],
                        ">=",
                        0.30,
                    ),
                )
            if stage >= 9:
                conditions["object_center_inside_basket_margin_m"] = (
                    basket[0] >= 0,
                    basket[0],
                    ">=",
                    0,
                )
                if stage in (9, 11, 13):
                    jaw = state["dof_pos"][:, self.env.gripper_joint_ids]
                    jaw_error = (jaw - (0.0 if stage == 11 else -math.pi / 3)).abs().amax(-1)
                    conditions.update(
                        gripper_target_error_rad=(jaw_error < 0.05, jaw_error, "<", 0.05),
                        gripper_stable=(
                            grip_stable,
                            span,
                            "<",
                            self.gripper_stability.max_span_rad,
                        ),
                    )
                if stage == 12:
                    shift = (
                        (state["object_state"][:, :3] - self.nudge_start) * self.nudge_direction
                    ).sum(-1)
                    conditions.update(
                        measured_object_shift_m=(shift >= 0.005, shift, ">=", 0.005),
                        measured_nonoverlapping_space_m=(basket[1] >= 0, basket[1], ">=", 0),
                    )
            local = {
                name: (
                    value[0][rows],
                    value[1][rows],
                    value[2],
                    value[3][rows] if torch.is_tensor(value[3]) and value[3].ndim else value[3],
                )
                for name, value in conditions.items()
            }
            required = 50 if stage == 6 else 10
            passed = self.record_acceptance(
                self.Stage(stage), rows, local, required_streak=required
            )
            self.settled[rows] = consecutive_steps(passed, self.settled[rows])
            ready = rows[self.settled[rows] >= required]
            completed[stage_enum] = ready
            limit = (
                torch.maximum(self.arm_path.length[rows] + 150, self.stage_limits[stage])
                if stage >= 9
                else self.stage_limits[stage]
            )
            expired = rows[(self.stage_steps[rows] > limit) & (self.settled[rows] < required)]
            if self.basket is not None and stage == int(self.Stage.GRASP_OBJECT):
                reposition_rows = expired[self.nudge_attempts[expired] < 2]
                expired = expired[self.nudge_attempts[expired] >= 2]
            expired_by_stage[stage_enum] = expired
        bad = (
            normal
            & (self.stage == 2)
            & self.arm_path.complete(self.rows)
            & stable
            & ((error >= 0.012) | (angle >= 0.06))
        )
        self.grasp_retry_steps = consecutive_steps(bad, self.grasp_retry_steps)
        retry = bad & (self.grasp_retry_steps >= 25) & (self.grasp_retries < 2)
        self.fail(torch.where(self.active_mask() & normal & (self.collision_steps >= 3))[0], 1)
        self.fail(
            torch.where(self.active_mask() & normal & (state["root_states"][:, 2] < 0.18))[0], 2
        )
        self.fail(torch.where(self.active_mask() & (self.drop_steps >= 10))[0], 3)
        self.fail(
            torch.where(self.active_mask() & (self.stage == 1) & ~self.approach_gaze_valid)[0], 7
        )
        if basket is not None:
            unsafe = (basket[0] < 0) | (state["root_states"][:, 2] < 0.18)
            self.fail(torch.where(self.active_mask() & (self.stage >= 9) & unsafe)[0], 6)
        for stage, rows in expired_by_stage.items():
            rows = rows[self.mask(stage)[rows] & ~retry[rows]]
            self.fail(rows, 5 if int(stage) >= 9 else 4)
        for source, rows in completed.items():
            completed[source] = rows[self.mask(source)[rows]]
        if self.basket is not None:
            reposition_rows = reposition_rows[self.mask(self.Stage.GRASP_OBJECT)[reposition_rows]]
            self.transition(reposition_rows, self.Stage.OPEN_FOR_REPOSITION)
            events["open_for_reposition_plan"] = reposition_rows
            grasp_ready = completed.get(self.Stage.GRASP_OBJECT, self.rows[:0])
            completed[self.Stage.GRASP_OBJECT] = grasp_ready[
                self.mask(self.Stage.GRASP_OBJECT)[grasp_ready]
            ]
        retry_rows = torch.where(retry & self.mask(self.Stage.MOVE_TO_GRASP))[0]
        for source, rows in completed.items():
            destination = self.stage_specs[source].successors[0]
            self.transition(rows, destination)
            label = (
                "success"
                if destination == self.Stage.SUCCESS
                else destination.name.lower() + "_plan"
            )
            events[label] = torch.cat((events.get(label, rows[:0]), rows))
        self.transition(retry_rows, self.Stage.MOVE_TO_GRASP)
        label = "move_to_grasp_plan"
        events[label] = torch.cat((events.get(label, retry_rows[:0]), retry_rows))
        events["timeout"] = torch.where(
            self.mask(self.Stage.FAILURE) & ((self.failure_code == 4) | (self.failure_code == 5))
        )[0]
        return events

    def active_mask(self):
        return ~self.mask(self.Stage.SUCCESS) & ~self.mask(self.Stage.FAILURE)
