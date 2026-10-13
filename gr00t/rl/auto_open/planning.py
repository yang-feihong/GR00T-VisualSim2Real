"""IK, reachability, and path planning algorithms for AutoOpen."""

import math

import numpy as np
import torch

from gr00t.rl.envs.door.b2z1_reset_geometry import (
    quaternion_xyzw_to_matrix,
    solve_b2z1_arm_ik,
)
from gr00t.rl.auto_open.control import (
    _ee_world_orientation_target,
    _ee_world_target,
    _quat_distance,
    _quat_nlerp,
    _perturb_ee_target,
)
from gr00t.rl.auto_open.geometry import (
    _panel_tcp_for_orientation,
    _quat_align_vectors_np,
    _quat_axis_angle_np,
    _quat_mul_np,
    _quat_rotate_np,
    _rotate_about_z,
    _yaw_quat,
)
from gr00t.rl.auto_open.utils import as_numpy as _as_np

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
        _yaw_quat(current_hinge_world),
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


ARM_RETRACT_CANDIDATE_BATCH_SIZE = 64
ARM_RETRACT_MAX_CANDIDATES = 4096


def _sample_ball_uniform_volume(rng, count, radius):
    """Sample Cartesian offsets uniformly by volume inside a 3-D ball."""
    directions = rng.normal(size=(count, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    while np.any(norms <= 1.0e-12):
        invalid = norms[:, 0] <= 1.0e-12
        directions[invalid] = rng.normal(size=(int(np.sum(invalid)), 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
    radii = float(radius) * np.cbrt(rng.random(count))
    return directions / norms * radii[:, None]


def _sample_reachable_arm_retract_pose(self, state, stage_label):
    """Sample one per-entry retract pose, rejecting candidates by batched IK."""
    entry_index = int(self.arm_retract_entry_count)
    self.arm_retract_entry_count += 1
    seed_sequence = np.random.SeedSequence(
        [self.arm_retract_random_seed, 0xA2E7_2AC7, entry_index]
    )
    rng = np.random.default_rng(seed_sequence)
    base_pose = np.asarray(self.arm_retract_pose_local, dtype=np.float64)
    radius = float(self.arm_retract_position_noise_radius_m)
    rotation_bound = math.radians(float(self.arm_retract_rotation_noise_deg))

    dof_names = list(state["dof_names"])
    dof_positions = _as_np(state["dof_pos"])[0]
    dof_index = {name: index for index, name in enumerate(dof_names)}
    initial_arm = np.asarray(
        [dof_positions[dof_index[name]] for name in self.ee_steady_model.arm_joint_names],
        dtype=np.float64,
    )
    root = _as_np(state["root_states"])[0].astype(np.float64)
    root_xyzw = root[3:7][[1, 2, 3, 0]]
    device = torch.device(self.env.device)
    dtype = torch.float64
    joint_lower = torch.as_tensor(
        self.ee_steady_model.lower_limits, device=device, dtype=dtype
    )
    joint_upper = torch.as_tensor(
        self.ee_steady_model.upper_limits, device=device, dtype=dtype
    )
    rejected = []
    evaluated_count = 0

    while evaluated_count < ARM_RETRACT_MAX_CANDIDATES:
        count = min(
            ARM_RETRACT_CANDIDATE_BATCH_SIZE,
            ARM_RETRACT_MAX_CANDIDATES - evaluated_count,
        )
        position_delta = _sample_ball_uniform_volume(rng, count, radius)
        rpy_delta = rng.uniform(-rotation_bound, rotation_bound, size=(count, 3))
        poses = np.repeat(base_pose[None, :], count, axis=0)
        poses[:, :3] += position_delta
        poses[:, 3:] += rpy_delta
        positions_world = np.stack(
            [_ee_world_target(self, pose[:3]) for pose in poses]
        ).astype(np.float64)
        orientations_wxyz = np.stack(
            [_ee_world_orientation_target(self, pose[3:]) for pose in poses]
        ).astype(np.float64)
        orientations_xyzw = orientations_wxyz[:, [1, 2, 3, 0]]

        root_position = torch.as_tensor(
            root[:3], device=device, dtype=dtype
        ).expand(count, -1)
        root_quaternion = torch.as_tensor(
            root_xyzw, device=device, dtype=dtype
        ).expand(count, -1)
        target_position = torch.as_tensor(
            positions_world, device=device, dtype=dtype
        )
        target_rotation = quaternion_xyzw_to_matrix(
            torch.as_tensor(orientations_xyzw, device=device, dtype=dtype)
        )
        initial_joint_position = torch.as_tensor(
            initial_arm, device=device, dtype=dtype
        ).expand(count, -1)
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
        position_error_np = position_error.detach().cpu().numpy()
        orientation_error_np = orientation_error.detach().cpu().numpy()
        joint_margin_np = joint_margin.detach().cpu().numpy()
        rejected_indices = np.flatnonzero(~reachable_np)
        for index in rejected_indices:
            rejected.append(
                {
                    "pose_local": poses[index].tolist(),
                    "position_delta_m": position_delta[index].tolist(),
                    "rpy_delta_rad": rpy_delta[index].tolist(),
                    "position_error_m": float(position_error_np[index]),
                    "orientation_error_rad": float(orientation_error_np[index]),
                    "joint_margin_rad": float(joint_margin_np[index]),
                }
            )
        evaluated_count += count
        reachable_indices = np.flatnonzero(reachable_np)
        if reachable_indices.size:
            selected = int(reachable_indices[0])
            rejected_before_acceptance = evaluated_count - count + selected
            record = {
                "entry_index": entry_index,
                "stage": str(stage_label),
                "evaluated_candidates": evaluated_count,
                "unreachable_candidates": len(rejected),
                "rejected_before_acceptance": rejected_before_acceptance,
                "draws_until_acceptance": rejected_before_acceptance + 1,
                "accepted_pose_local": poses[selected].tolist(),
                "accepted_position_delta_m": position_delta[selected].tolist(),
                "accepted_rpy_delta_rad": rpy_delta[selected].tolist(),
                "accepted_position_error_m": float(position_error_np[selected]),
                "accepted_orientation_error_rad": float(
                    orientation_error_np[selected]
                ),
                "accepted_joint_margin_rad": float(joint_margin_np[selected]),
                "rejected": rejected,
            }
            self.arm_retract_randomization_records.append(record)
            self.active_arm_retract_pose_local = poses[selected].astype(np.float32)
            print(
                "[doorman][arm-retract-randomization] "
                f"stage={stage_label} entry={entry_index} "
                f"evaluated={evaluated_count} unreachable={len(rejected)} "
                f"pose_local={self.active_arm_retract_pose_local.round(5).tolist()}",
                flush=True,
            )
            return self.active_arm_retract_pose_local

    self.arm_retract_randomization_records.append(
        {
            "entry_index": entry_index,
            "stage": str(stage_label),
            "evaluated_candidates": evaluated_count,
            "unreachable_candidates": len(rejected),
            "rejected_before_acceptance": evaluated_count,
            "draws_until_acceptance": None,
            "accepted_pose_local": None,
            "accepted_position_delta_m": None,
            "accepted_rpy_delta_rad": None,
            "accepted_position_error_m": None,
            "accepted_orientation_error_rad": None,
            "accepted_joint_margin_rad": None,
            "rejected": rejected,
        }
    )
    raise RuntimeError(
        "arm retract randomization found no exact IK solution after "
        f"{ARM_RETRACT_MAX_CANDIDATES} candidates for {stage_label}"
    )


def _prepare_panel_contact_command(self, state):
    """Prepare CONTACT_PANEL's target before the stateful command runs."""
    if self.prepared_panel_contact_command is not None:
        return
    current_hinge_joint = float(
        _as_np(state["door_joint_position"])[0, self.hinge_idx]
    )
    current_hinge_world = self.hinge_axis_sign * current_hinge_joint
    current_normal = _quat_rotate_np(
        _yaw_quat(current_hinge_world), self.door_normal
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
    self.panel_contact_target_world = _panel_tcp_for_orientation(
        self, state, current_hinge_world, self.panel_contact_orientation
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
        _perturb_ee_target(
            self,
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
        _panel_tcp_for_orientation(
            self, state, current_hinge_world, command_orientation_world
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
    desired_position_world = _select_reachable_panel_contact_target(self,
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
    self.prepared_panel_contact_command = _panel_contact_inverse_command(self,
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
        or _quat_distance(
            desired_orientation,
            self.panel_contact_inverse_orientation_world,
        )
        > 1.0e-4
    )
    if target_changed:
        (
            self.panel_contact_inverse_command_world,
            self.panel_contact_inverse_command_orientation_world,
        ) = _steady_compensated_ee_command(self,
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
    self.turn_endpoint_world = _rotate_about_z(
        handle_endpoint, self.hinge_world, gap_delta
    )
    self.turn_endpoint_quat = _quat_mul_np(
        _yaw_quat(gap_delta),
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
    _ensure_turn_endpoint(self, state)
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
    angle = _quat_distance(start_q, self.turn_endpoint_quat)
    count = max(120, int(math.ceil(distance / 0.004)) + 1, int(math.ceil(angle / 0.03)) + 1)
    fractions = np.linspace(0.0, 1.0, count, dtype=np.float32)
    nominal_handle_current = self.turn_axle_world + _quat_rotate_np(
        _quat_axis_angle_np(
            self.turn_handle_axis_world,
            current_angle - self.turn_start_handle_angle,
        ),
        self.turn_grasp_offset_world,
    )
    nominal_current = _rotate_about_z(
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
        nominal = _rotate_about_z(
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
                _yaw_quat(planned_hinge),
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
        _reset_turn_path(self)
        print(
            f"[doorman][turn] latch released at {latch_position:.4f} m; "
            "starting door-gap arc",
            flush=True,
        )
    if (
        self.turn_path_stage != self.stage
        or self.turn_path_world is None
    ):
        _replan_turn_path(self, state)
    index = min(self.turn_path_index, len(self.turn_path_world) - 1)
    target = self.turn_path_world[index].copy()
    quat = self.turn_path_quat[index].copy()
    self.turn_path_index = min(self.turn_path_index + 1, len(self.turn_path_world) - 1)
    return target, quat

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
        command_target, command_orientation = _perturb_ee_target(self,
            "pregrasp",
            self.center_standoff_ee_world,
            self.grasp_orientation_quat,
        )
        self.arm_align_desired_target_world = command_target.copy()
        return _build_steady_compensated_path(self,
            state,
            start_position_world,
            command_target,
            command_orientation,
            start_orientation_world=_as_np(state["ee_orn"])[0],
            label="arm_align_path",
        )
    candidates = _pregrasp_pitch_candidates(self)
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
        command_position, command_orientation = _perturb_ee_target(self,
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
    _apply_pregrasp_candidate(self, selected)
    print(
        f"[doorman][pregrasp-pitch] selected={self.pregrasp_pitch_deg:.0f}deg "
        f"candidates={diagnostics}",
        flush=True,
    )
    command_target, command_orientation = _perturb_ee_target(self,
        "pregrasp",
        self.center_standoff_ee_world,
        self.grasp_orientation_quat,
    )
    self.arm_align_desired_target_world = command_target.copy()
    return _build_steady_compensated_path(self,
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
        desired_orientation = _quat_nlerp(
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
        desired_orientation = _quat_nlerp(
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
