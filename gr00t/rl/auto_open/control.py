"""Shared base, Cartesian, orientation, and stability control primitives."""

import math

import numpy as np
import torch

from legged_gym.utils.isaaclab_math import quat_rotate_inverse
from gr00t.rl.auto_open.geometry import (
    HANDLE_AXLE_TARGET_MARGIN_M,
    HANDLE_FREE_TARGET_MARGIN_M,
    _axial_target_correction,
    _grasp_semantic_axes_world,
    _quat_conjugate_np,
    _quat_mul_np,
    _quat_rotate_inverse_np,
    _quat_rotate_np,
    _quat_rotate_points_np,
    _rotate_about_z,
    _quat_to_euler_xyz_np,
    _quat_yaw,
    _task_axes_world,
)
from gr00t.rl.auto_open.utils import as_numpy as _as_np

MAX_BASE_PLANAR_SPEED = 0.4
MAX_BASE_YAW_RATE = 0.6
BASE_REVERSAL_CONFIRM_FRAMES = 3
BASE_REVERSAL_LOCK_FRAMES = 5
BASE_REVERSAL_JUMP = 0.15
BASE_REVERSAL_EPSILON = 1.0e-4
BASE_STATION_TOLERANCE_M = 0.12
BASE_YAW_TOLERANCE_RAD = 0.08
THROUGH_STOP_DISTANCE_M = 0.08
THROUGH_STOP_YAW_RAD = 0.05
ROBOT_TRUNK_REAR_X = -0.284
THROUGH_TAIL_TARGET_DISTANCE = 1.0


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


def _stabilize_base_command(self, command):
    """Reject brief sign reversals outside APPROACH."""
    target = np.asarray(command, dtype=np.float32).reshape(3)
    previous = np.asarray(self.base_command_previous, dtype=np.float32).reshape(3)
    pending_sign = np.asarray(
        self.base_reversal_pending_sign, dtype=np.int8
    ).reshape(3)
    pending_steps = np.asarray(
        self.base_reversal_pending_steps, dtype=np.int32
    ).reshape(3)
    lock_steps = np.asarray(self.base_reversal_lock_steps, dtype=np.int32).reshape(3)
    output = target.copy()
    for component in range(3):
        old = float(previous[component])
        new = float(target[component])
        if lock_steps[component] > 0:
            output[component] = old
            lock_steps[component] -= 1
            continue
        reversing = (
            abs(old) > BASE_REVERSAL_EPSILON
            and abs(new) > BASE_REVERSAL_EPSILON
            and old * new < 0.0
            and abs(new - old) >= BASE_REVERSAL_JUMP
        )
        if not reversing:
            pending_sign[component] = 0
            pending_steps[component] = 0
            continue
        new_sign = 1 if new > 0.0 else -1
        if pending_sign[component] == new_sign:
            pending_steps[component] += 1
        else:
            pending_sign[component] = new_sign
            pending_steps[component] = 1
        if pending_steps[component] < BASE_REVERSAL_CONFIRM_FRAMES:
            output[component] = old
        else:
            pending_sign[component] = 0
            pending_steps[component] = 0
            lock_steps[component] = BASE_REVERSAL_LOCK_FRAMES
    self.base_command_previous = output.astype(np.float32)
    self.base_reversal_pending_sign = pending_sign
    self.base_reversal_pending_steps = pending_steps
    self.base_reversal_lock_steps = lock_steps
    return self.base_command_previous.copy()


def _zero_cross_base_reversal(self, command):
    """Insert one zero command before a large APPROACH sign reversal."""
    target = np.asarray(command, dtype=np.float32).reshape(3)
    previous = np.asarray(self.base_command_previous, dtype=np.float32).reshape(3)
    crossing = (previous * target < 0.0) & (
        np.abs(target - previous) >= BASE_REVERSAL_JUMP
    )
    output = target.copy()
    output[crossing] = 0.0
    self.base_command_previous = output.astype(np.float32)
    return self.base_command_previous.copy()


def _base_commands_exact(
    state,
    station_xy,
    station_yaw,
    acceptance_xy=None,
    acceptance_yaw=None,
    minimum_speed_toward_target_only=False,
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
    if minimum_speed_toward_target_only:
        deadzone &= (command[:, 0] * local_x + command[:, 1] * local_y) > 0.0
    command[deadzone, :2] *= (0.15 / planar_norm[deadzone])[:, None]
    yaw_deadzone = (
        (np.abs(yaw_error) > BASE_YAW_TOLERANCE_RAD)
        & (np.abs(command[:, 2]) > 1.0e-6)
        & (np.abs(command[:, 2]) < 0.2)
    )
    if minimum_speed_toward_target_only:
        yaw_deadzone &= (command[:, 2] * yaw_error) > 0.0
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
        semantic_axes_world = _grasp_semantic_axes_world(
            self,
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
        _task_axes_world(self)
        if semantic_axes_world is None
        else semantic_axes_world,
        position_bounds=position_bounds,
        rotation_bounds=rotation_bounds,
        position_names=position_names,
        rotation_names=rotation_names,
        rotation_semantic_axes_world=rotation_semantic_axes_world,
    )
    if sample_label == "grasp_chain":
        target_position = np.asarray(perturbed[0], dtype=np.float32).copy()
        target_orientation = np.asarray(perturbed[1], dtype=np.float32)
        jaw_points_local = np.concatenate(
            (
                self.open_template["fixed_points"],
                self.open_template["moving_points"],
            ),
            axis=0,
        )
        jaw_offsets_world = _quat_rotate_points_np(
            target_orientation, jaw_points_local
        )
        free_axis = np.asarray(
            self.handle_free_axis_world, dtype=np.float32
        )
        jaw_axial = (
            target_position[None, :]
            + jaw_offsets_world
            - self.grasp_center_pos[None, :]
        ) @ free_axis
        handle_radius = max(
            float(self.handle_upper_z - self.handle_lower_z) * 0.5,
            1.0e-4,
        )
        axial_correction = _axial_target_correction(
            jaw_axial.min(),
            jaw_axial.max(),
            self.handle_half_length + handle_radius,
            HANDLE_AXLE_TARGET_MARGIN_M,
            HANDLE_FREE_TARGET_MARGIN_M,
        )
        target_position += axial_correction * free_axis
        perturbed = (target_position, target_orientation)
        self.grasp_semantic_position_delta_world = (
            target_position
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
        nominal_xy, nominal_yaw = _approach_base_target(self)
        target_xy, target_yaw = _perturb_base_target(self,
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
        minimum_speed_toward_target_only=default_target,
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


def _ee_world_orientation_target(self, local_rpy):
    """Map one goal-frame XYZ Euler orientation into world coordinates."""
    roll, pitch, yaw = np.asarray(local_rpy, dtype=np.float64).reshape(3)
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    local_quat = np.asarray(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dtype=np.float32,
    )
    goal_ref_quat = _as_np(self.env.get_goal_reference_quat())[0]
    return _quat_mul_np(goal_ref_quat, local_quat)

def _grasp_orientation_delta(self, ee_goal_local, grasp_quat_world):
    """Convert the configured B2Z1 grasp orientation into teleop RPY deltas."""
    desired_world = (
        self.active_grasp_quat.copy()
        if self.active_grasp_quat is not None
        else _desired_grasp_quat(self, grasp_quat_world)
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



def _quat_nlerp(start, end, fraction):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    if float(np.dot(start, end)) < 0.0:
        end = -end
    result = (1.0 - float(fraction)) * start + float(fraction) * end
    return result / max(float(np.linalg.norm(result)), 1.0e-8)

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
    current_contact = _rotate_about_z(
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

    contact_local = _as_np(_ee_local_target(self, current_contact))[0]
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
    panel_free_end = _rotate_about_z(
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

def _effective_grasp_target(self, grasp_center, ee_quat=None):
    target = np.asarray(grasp_center, dtype=np.float32).copy()
    if ee_quat is None:
        offset_world = getattr(self, "grasp_ee_offset_world", np.zeros(3, dtype=np.float32))
    else:
        offset_world = _quat_rotate_np(ee_quat, self.grasp_ee_offset_local)
    target -= offset_world
    return target

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
        _quat_distance(orientation, sample) for sample in orientations
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
