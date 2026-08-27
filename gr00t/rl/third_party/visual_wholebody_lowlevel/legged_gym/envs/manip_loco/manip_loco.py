from __future__ import annotations
import numpy as np
import time
import torch
from typing import Tuple
from legged_gym.utils.math import orientation_error
from legged_gym.utils.isaaclab_math import (
    quat_rotate_inverse,
    quat_apply,
    euler_from_quat,
    wrap_to_pi,
    torch_rand_float,
)
from legged_gym.envs.base.legged_robot import LeggedRobotIsaacLab
from .b2z1_config import B2Z1IsaacLabCfg

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for wxyz quaternions."""
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply for wxyz quaternions."""
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def sphere2cart(sphere: torch.Tensor) -> torch.Tensor:
    """ManipLoco sphere convention: [radius, pitch, yaw] -> [x, y, z]."""
    radius = sphere[..., 0]
    pitch = sphere[..., 1]
    yaw = sphere[..., 2]
    cp = torch.cos(pitch)
    return torch.stack(
        [
            radius * cp * torch.cos(yaw),
            radius * cp * torch.sin(yaw),
            radius * torch.sin(pitch),
        ],
        dim=-1,
    )


def cart2sphere(cart: torch.Tensor) -> torch.Tensor:
    """ManipLoco cartesian convention: [x, y, z] -> [radius, pitch, yaw]."""
    x, y, z = cart.unbind(-1)
    radius = torch.linalg.norm(cart, dim=-1).clamp_min(1e-8)
    pitch = torch.asin(torch.clamp(z / radius, -1.0, 1.0))
    yaw = torch.atan2(y, x)
    return torch.stack([radius, pitch, yaw], dim=-1)


_COMPUTE_EE_GOAL_POSE_FROM_REF = None


def _compute_ee_goal_pose_from_ref_impl(
    goal_ref_quat: torch.Tensor,
    goal_ref_origin: torch.Tensor,
    goal_center_offset: torch.Tensor,
    curr_ee_goal_cart: torch.Tensor,
    curr_ee_goal_sphere: torch.Tensor,
    ee_goal_orn_delta_rpy: torch.Tensor,
    arm_induced_pitch: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    goal_local = goal_center_offset + curr_ee_goal_cart

    qw = goal_ref_quat[:, 0]
    qx = goal_ref_quat[:, 1]
    qy = goal_ref_quat[:, 2]
    qz = goal_ref_quat[:, 3]
    vx = goal_local[:, 0]
    vy = goal_local[:, 1]
    vz = goal_local[:, 2]

    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    goal_world_x = goal_ref_origin[:, 0] + vx + qw * tx + qy * tz - qz * ty
    goal_world_y = goal_ref_origin[:, 1] + vy + qw * ty + qz * tx - qx * tz
    goal_world_z = goal_ref_origin[:, 2] + vz + qw * tz + qx * ty - qy * tx
    goal_world = torch.stack((goal_world_x, goal_world_y, goal_world_z), dim=-1)

    roll = ee_goal_orn_delta_rpy[:, 0] + 1.5707963267948966
    pitch = -curr_ee_goal_sphere[:, 1] + arm_induced_pitch + ee_goal_orn_delta_rpy[:, 1]
    yaw = ee_goal_orn_delta_rpy[:, 2] + curr_ee_goal_sphere[:, 2]
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    lw = cr * cp * cy + sr * sp * sy
    lx = sr * cp * cy - cr * sp * sy
    ly = cr * sp * cy + sr * cp * sy
    lz = cr * cp * sy - sr * sp * cy

    goal_quat = torch.stack(
        (
            qw * lw - qx * lx - qy * ly - qz * lz,
            qw * lx + qx * lw + qy * lz - qz * ly,
            qw * ly - qx * lz + qy * lw + qz * lx,
            qw * lz + qx * ly - qy * lx + qz * lw,
        ),
        dim=-1,
    )
    return goal_world, goal_quat


def _compute_ee_goal_pose_from_ref(*args):
    global _COMPUTE_EE_GOAL_POSE_FROM_REF
    if _COMPUTE_EE_GOAL_POSE_FROM_REF is None:
        _COMPUTE_EE_GOAL_POSE_FROM_REF = torch.jit.script(_compute_ee_goal_pose_from_ref_impl)
    return _COMPUTE_EE_GOAL_POSE_FROM_REF(*args)


class ManipLocoIsaacLab(LeggedRobotIsaacLab):
    """IsaacLab simulator-layer implementation for the ManipLoco task."""
    cfg: B2Z1IsaacLabCfg

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rgb_camera_debug_draw_enabled = False
        self._init_task_state()
        self._arm_pos_targets_for_step = None
        self._debug_draw = None
        if self.teleop_mode and self.teleop_debug:
            self._init_debug_draw()


    def _init_task_state(self):
        """Initialize ManipLoco task-local state.

        Goal/reference state is shared by all modes. Non-teleop goal trajectory
        state is used by normal play/train, while teleop state only buffers
        external keyboard/XR inputs.
        """
        self.teleop_mode = bool(self.cfg.env.teleop_mode)
        self._init_goal_reference_state()
        self._init_non_teleop_goal_trajectory_state()
        self._init_teleop_input_state()
        self._update_base_yaw_quat()

    def _init_goal_reference_state(self):
        """Initialize EE goal and reference-frame state shared by all modes."""
        self.arm_induced_pitch = float(self.cfg.goal_ee.arm_induced_pitch)
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)

        self.uses_goal_height_reference_mask = True
        self.observes_goal_height_reference_mask = (
            bool(self.cfg.goal_ee.sphere_center.mixed_height_reference)
            or self.cfg.goal_ee.ranges.ee_goal_sampling_mode in (
                "body_importance",
                "body_importance_single_dir",
            )
        )
        self.goal_height_follow_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_height_follow_override = None

        ranges_cfg = self.cfg.goal_ee.ranges
        self.goal_ee_ranges = {
            "pos_l": list(ranges_cfg.pos_l),
            "pos_p": list(ranges_cfg.pos_p),
            "pos_y": list(ranges_cfg.pos_y),
            "delta_orn_r": list(ranges_cfg.delta_orn_r),
            "delta_orn_p": list(ranges_cfg.delta_orn_p),
            "delta_orn_y": list(ranges_cfg.delta_orn_y),
        }

        self.ee_goal_sampling_mode = str(ranges_cfg.ee_goal_sampling_mode)
        valid_goal_sampling_modes = (
            "arm_front_sphere",
            "omnidirectional",
            "body_importance",
            "body_importance_single_dir",
        )
        if self.ee_goal_sampling_mode not in valid_goal_sampling_modes:
            raise ValueError(
                f"Unsupported ee_goal_sampling_mode={self.ee_goal_sampling_mode}. "
                f"Expected one of {valid_goal_sampling_modes}"
            )

        self.is_omnidirectional_sampling = self.ee_goal_sampling_mode == "omnidirectional"
        self.omnidirectional_rear_transition_pos_y_abs = float(
            ranges_cfg.omnidirectional_rear_transition_pos_y_abs
        )
        if not (0.0 <= self.omnidirectional_rear_transition_pos_y_abs < np.pi):
            raise ValueError("goal_ee.ranges.omnidirectional_rear_transition_pos_y_abs must be in [0, pi)")

        self.omnidirectional_pos_l = torch.tensor(
            ranges_cfg.omnidirectional_pos_l, device=self.device, dtype=torch.float32
        )
        self.omnidirectional_rear_pos_l = torch.tensor(
            ranges_cfg.omnidirectional_rear_pos_l, device=self.device, dtype=torch.float32
        )
        self.omnidirectional_rear_pos_p = torch.tensor(
            ranges_cfg.omnidirectional_rear_pos_p, device=self.device, dtype=torch.float32
        )

        sphere_center_cfg = self.cfg.goal_ee.sphere_center
        urdf_mount_cfg = self.cfg.goal_ee.urdf_mount

        self.ee_goal_center_offset = torch.tensor(
            [
                float(sphere_center_cfg.x_offset),
                float(sphere_center_cfg.y_offset),
                float(sphere_center_cfg.z_invariant_offset),
            ],
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.num_envs, 1)

        self.arm_base_offset = torch.tensor(
            list(urdf_mount_cfg.arm_base_offset),
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.num_envs, 1)
        self.arm_waist_offset = self.arm_base_offset.clone()
        self.arm_waist_offset[:, 2] += float(urdf_mount_cfg.arm_waist_offset_z)
        self.arm_shoulder_offset = self.arm_waist_offset.clone()
        self.arm_shoulder_offset[:, 2] += float(urdf_mount_cfg.arm_shoulder_offset_z)
        self.arm_mount_yaw_offset = float(urdf_mount_cfg.mount_yaw_offset)

        self._goal_ref_quat_buf = torch.zeros(self.num_envs, 4, device=self.device)
        self._goal_ref_origin_buf = torch.zeros(self.num_envs, 3, device=self.device)
        body_importance_cfg = self.cfg.goal_ee.body_importance_sampling
        z_invariant_offset = float(sphere_center_cfg.z_invariant_offset)
        self.body_importance_extension = float(body_importance_cfg.extension)
        self.body_importance_back_start_offset = float(body_importance_cfg.back_start_offset)
        self.body_importance_side_height = (
            torch.tensor(body_importance_cfg.side_height, device=self.device, dtype=torch.float32)
            - z_invariant_offset
        )
        self.body_importance_ground_height = (
            torch.tensor(body_importance_cfg.ground_height, device=self.device, dtype=torch.float32)
            - z_invariant_offset
        )
        self.body_importance_ground_height_ratio = float(body_importance_cfg.ground_height_ratio)
        self.body_importance_back_height = torch.tensor(
            body_importance_cfg.back_height,
            device=self.device,
            dtype=torch.float32,
        )

        self.collision_lower_limits = torch.tensor(
            self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float32
        )
        self.collision_upper_limits = torch.tensor(
            self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float32
        )
        self.underground_limit = float(self.cfg.goal_ee.underground_limit)
        self.num_collision_check_samples = int(self.cfg.goal_ee.num_collision_check_samples)
        self.collision_check_t = torch.linspace(
            0.0, 1.0, self.num_collision_check_samples, device=self.device
        )[None, None, :]

        self.init_start_ee_sphere = torch.tensor(
            ranges_cfg.init_pos_start,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        self.reset_init_ee_sphere = self.init_start_ee_sphere.repeat(self.num_envs, 1)

        self.curr_ee_goal_sphere = self.reset_init_ee_sphere.clone()
        self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)

        self.curr_ee_goal_cart_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.ee_goal_orn_quat[:, 0] = 1.0
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)

        self.body_importance_direction_ids = self._get_body_importance_direction_ids()
        self.body_importance_single_direction_ids = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )

    def _init_non_teleop_goal_trajectory_state(self):
        """Initialize EE goal trajectory buffers."""
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = torch.full(
            (self.num_envs,),
            float(self.cfg.goal_ee.traj_timesteps_default),
            device=self.device,
        )
        self.traj_total_timesteps = torch.full(
            (self.num_envs,),
            float(self.cfg.goal_ee.traj_total_timesteps_default),
            device=self.device,
        )

        self.ee_start_sphere = self.reset_init_ee_sphere.clone()
        self.ee_goal_sphere = self.reset_init_ee_sphere.clone()
        self.ee_goal_cart = sphere2cart(self.ee_goal_sphere)

    def _init_teleop_input_state(self):
        """Initialize keyboard/XR teleop input buffers only."""
        self.teleop_raw_commands = torch.zeros_like(self.commands)
        self.teleop_arm_control_mode = self.cfg.env.teleop_arm_control_mode

        self.teleop_raw_ee_goal_cart = self.curr_ee_goal_cart.clone()
        self.teleop_raw_ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.teleop_hold_actual_ee_target = torch.full(
            (self.num_envs,),
            bool(self.cfg.env.teleop_hold_actual_ee_target_on_init),
            dtype=torch.bool,
            device=self.device,
        )

        self.teleop_arm_joint_pos_targets = self.dof_pos[:, self.arm_joint_ids].clone()
        self.gripper_pos_targets = self.default_dof_pos[:, self.gripper_joint_ids].clone()
        self.teleop_debug = bool(self.cfg.env.teleop_debug)
        self._teleop_ee_synced_once = False
        self._teleop_inputs_dirty = True
        self.teleop_initialize_targets_on_next_reset = True

    def _init_debug_draw(self):
        self._debug_draw = None
        try:
            from isaacsim.util.debug_draw import _debug_draw
            self._debug_draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            try:
                from omni.isaac.debug_draw import _debug_draw
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            except Exception as e:
                print(f"[debug_draw][warn] debug draw unavailable: {e}")
                self._debug_draw = None

    def _pre_physics_step(self, actions: torch.Tensor):
        self._profile_step_start()
        profile_start = time.perf_counter()
        # Simulator replacement for refresh_rigid_body_state_tensor /
        # refresh_jacobian_tensors before IK target generation.
        section_start = time.perf_counter()
        self._refresh_ee_and_jacobian_for_ik()
        self._profile_record("ee_jacobian", time.perf_counter() - section_start)

        # Keep ManipLoco EE-goal math. This updates:
        #   curr_ee_goal_cart_world: world-frame IK position target
        #   ee_goal_orn_quat:       world-frame IK orientation target
        section_start = time.perf_counter()
        self._update_curr_ee_goal(refresh_base_yaw=False)
        self._profile_record("ee_goal", time.perf_counter() - section_start)

        # Parent only handles action clipping/delay/target_pos for legs.
        section_start = time.perf_counter()
        super()._pre_physics_step(actions)
        base_pre_elapsed = time.perf_counter() - section_start
        rgb_pose_elapsed = 0.0
        if getattr(self, "_step_profile_in_step", False):
            rgb_pose_elapsed = float(getattr(self, "_step_profile_current_sums", {}).get("rgb_pose", 0.0))
        self._profile_record("base_pre", max(base_pre_elapsed - rgb_pose_elapsed, 0.0))

        section_start = time.perf_counter()
        self._arm_pos_targets_for_step = None
        if self.teleop_mode and self.teleop_arm_control_mode == "joint":
            self._arm_pos_targets_for_step = self.teleop_arm_joint_pos_targets
        elif self.ee_pos is not None and self.ee_j_eef is not None:
            self._arm_pos_targets_for_step = self._get_arm_pos_targets()
        self._profile_record("arm_target", time.perf_counter() - section_start)
        self._profile_record("pre_total", max(time.perf_counter() - profile_start - rgb_pose_elapsed, 0.0))

    def _update_effective_teleop_inputs(self, force=True):
        if not self.teleop_mode:
            return
        if not force and not self._teleop_inputs_dirty:
            return

        self.commands[:] = self.teleop_raw_commands
        if bool(self.cfg.env.teleop_input_regularization):
            active_mask = (
                (torch.abs(self.commands[:, 0]) > float(self.cfg.env.teleop_zero_lin_vel_x_clip))
                | (torch.abs(self.commands[:, 1]) > float(self.cfg.env.teleop_zero_lin_vel_y_clip))
                | (torch.abs(self.commands[:, 2]) > float(self.cfg.env.teleop_zero_ang_vel_yaw_clip))
            ).unsqueeze(1)
            self.commands[:] *= active_mask

        # When joint-controlling the arm, keep EE target synced
        # to actual pose so switching back to EE mode does not jump.
        if self.teleop_arm_control_mode == "joint":
            if self.ee_pos is not None and self.ee_orn is not None:
                self._sync_teleop_ee_goal_to_current_pose()
            self._teleop_inputs_dirty = False
            return

        self.curr_ee_goal_cart[:] = self.teleop_raw_ee_goal_cart
        self.ee_goal_orn_delta_rpy[:] = self.teleop_raw_ee_goal_orn_delta_rpy
        self.curr_ee_goal_sphere[:] = cart2sphere(self.curr_ee_goal_cart)
        if (
            self.uses_goal_height_reference_mask
            and bool(self.cfg.env.teleop_auto_height_reference)
        ):
            yaw_from_mount = wrap_to_pi(self.curr_ee_goal_sphere[:, 2] - self.arm_mount_yaw_offset)
            follow_mask = torch.abs(yaw_from_mount) > (0.5 * np.pi)
            changed = follow_mask != self.goal_height_follow_mask
            if torch.any(changed):
                self.goal_height_follow_override = None
                self.goal_height_follow_mask[:] = follow_mask
                if self.num_envs == 1:
                    if bool(follow_mask[0].item()):
                        print("[teleop] auto height reference mode: trunk-follow")
                    else:
                        print("[teleop] auto height reference mode: z-invariant")
        self._teleop_inputs_dirty = False

    def _toggle_teleop_arm_control_mode(self):
        next_mode = "joint" if self.teleop_arm_control_mode == "ee" else "ee"
        self._set_teleop_arm_control_mode(next_mode)

    def apply_teleop_key(self, key: str):
        """Teleop key mapping used by run_teleop.sh."""
        k = key.lower()
        if k == "q":
            self.teleop_raw_commands[:, 0] = 0.0
            self.teleop_raw_commands[:, 1] = 0.0
        elif k == "w":
            self.teleop_raw_commands[:, 0] += float(self.cfg.env.teleop_base_lin_vel_step)
        elif k == "s":
            self.teleop_raw_commands[:, 0] -= float(self.cfg.env.teleop_base_lin_vel_step)
        elif k == ",":
            self.teleop_raw_commands[:, 1] += float(self.cfg.env.teleop_base_lin_vel_step)
        elif k == ".":
            self.teleop_raw_commands[:, 1] -= float(self.cfg.env.teleop_base_lin_vel_step)
        elif k == "e":
            self.teleop_raw_commands[:, 2] = 0.0
        elif k == "a":
            self.teleop_raw_commands[:, 2] += float(self.cfg.env.teleop_base_ang_vel_step)
        elif k == "d":
            self.teleop_raw_commands[:, 2] -= float(self.cfg.env.teleop_base_ang_vel_step)
        elif k == "g":
            self._toggle_teleop_arm_control_mode()

        if self.teleop_arm_control_mode == "joint":
            delta = float(self.cfg.env.teleop_arm_joint_step)
            if k in "yhujikzxcmbn":
                mapping = {"y": (0, +delta), "h": (0, -delta), "u": (1, +delta), "j": (1, -delta),
                           "i": (2, +delta), "k": (2, -delta), "z": (3, +delta), "x": (3, -delta),
                           "c": (4, +delta), "m": (4, -delta), "b": (5, +delta), "n": (5, -delta)}
                idx, dv = mapping[k]
                self.teleop_arm_joint_pos_targets[:, idx] += dv
            elif k == "l":
                self.teleop_arm_joint_pos_targets[:] = self.default_dof_pos[:, self.arm_joint_ids]
        else:
            if k == "y":
                self.teleop_raw_ee_goal_cart[:, 0] += float(self.cfg.env.teleop_ee_goal_pos_step)
            elif k == "h":
                self.teleop_raw_ee_goal_cart[:, 0] -= float(self.cfg.env.teleop_ee_goal_pos_step)
            elif k == "u":
                self.teleop_raw_ee_goal_cart[:, 1] += float(self.cfg.env.teleop_ee_goal_pos_step)
            elif k == "j":
                self.teleop_raw_ee_goal_cart[:, 1] -= float(self.cfg.env.teleop_ee_goal_pos_step)
            elif k == "i":
                self.teleop_raw_ee_goal_cart[:, 2] += float(self.cfg.env.teleop_ee_goal_pos_step)
            elif k == "k":
                self.teleop_raw_ee_goal_cart[:, 2] -= float(self.cfg.env.teleop_ee_goal_pos_step)
            elif k == "z":
                self.teleop_raw_ee_goal_orn_delta_rpy[:, 0] += float(self.cfg.env.teleop_ee_goal_orn_step)
            elif k == "x":
                self.teleop_raw_ee_goal_orn_delta_rpy[:, 0] -= float(self.cfg.env.teleop_ee_goal_orn_step)
            elif k == "c":
                self.teleop_raw_ee_goal_orn_delta_rpy[:, 1] += float(self.cfg.env.teleop_ee_goal_orn_step)
            elif k == "m":
                self.teleop_raw_ee_goal_orn_delta_rpy[:, 1] -= float(self.cfg.env.teleop_ee_goal_orn_step)
            elif k == "b":
                self.teleop_raw_ee_goal_orn_delta_rpy[:, 2] += float(self.cfg.env.teleop_ee_goal_orn_step)
            elif k == "n":
                self.teleop_raw_ee_goal_orn_delta_rpy[:, 2] -= float(self.cfg.env.teleop_ee_goal_orn_step)
            elif k == "l":
                self._reset_teleop_ee_goal_to_default()

        if k == "o":
            self.gripper_pos_targets += float(self.cfg.env.teleop_gripper_step)
        elif k == "p":
            self.gripper_pos_targets -= float(self.cfg.env.teleop_gripper_step)
        elif k == "r" and self.uses_goal_height_reference_mask:
            self.goal_height_follow_override = False
            self.goal_height_follow_mask[:] = False
            print("[teleop] height reference mode: z-invariant")
        elif k == "t" and self.uses_goal_height_reference_mask:
            self.goal_height_follow_override = True
            self.goal_height_follow_mask[:] = True
            print("[teleop] height reference mode: trunk-follow")

        self._teleop_inputs_dirty = True
        self._clip_teleop_targets()
        self._update_effective_teleop_inputs(force=True)
        # print(
        #     f"[teleop][env] key={key} cmd={self.teleop_raw_commands[0, :3].detach().cpu().tolist()} "
        #     f"arm_mode={self.teleop_arm_control_mode} ee_goal={self.curr_ee_goal_cart[0].detach().cpu().tolist()}"
        # )

    def _clip_teleop_targets(self):
        self.teleop_raw_commands[:, 0].clamp_(
            -float(self.cfg.env.teleop_lin_vel_x_limit),
            float(self.cfg.env.teleop_lin_vel_x_limit),
        )
        self.teleop_raw_commands[:, 1].clamp_(
            -float(self.cfg.env.teleop_lin_vel_y_limit),
            float(self.cfg.env.teleop_lin_vel_y_limit),
        )
        self.teleop_raw_commands[:, 2].clamp_(
            -float(self.cfg.env.teleop_ang_vel_yaw_limit),
            float(self.cfg.env.teleop_ang_vel_yaw_limit),
        )
        teleop_ee_goal_limits = (
            self.cfg.env.teleop_ee_goal_x_limit,
            self.cfg.env.teleop_ee_goal_y_limit,
            self.cfg.env.teleop_ee_goal_z_limit,
        )
        for dim, limits in enumerate(teleop_ee_goal_limits):
            lo, hi = limits
            self.teleop_raw_ee_goal_cart[:, dim].clamp_(float(lo), float(hi))
        self.teleop_raw_ee_goal_orn_delta_rpy[:] = wrap_to_pi(self.teleop_raw_ee_goal_orn_delta_rpy)

    def _refresh_ee_and_jacobian_for_ik(self):
        """Refresh IK input tensors at the policy step boundary."""
        if self.gripper_body_name not in self.body_names_to_idx:
            return

        self._update_base_yaw_quat()
        self.gripper_idx = self.body_names_to_idx[self.gripper_body_name]

        # EE pose from current rigid-body state (world frame).
        if hasattr(self.robot.data, "body_state_w"):
            body_state_w = self.robot.data.body_state_w
            self.ee_pos = body_state_w[:, self.gripper_idx, :3]
            self.ee_orn = body_state_w[:, self.gripper_idx, 3:7]
            if hasattr(self.robot.data, "body_vel_w"):
                self.ee_vel = self.robot.data.body_vel_w[:, self.gripper_idx, :]
            else:
                self.ee_vel = torch.zeros(self.num_envs, 6, device=self.device)

            # First teleop frame: sync the EE target to the current pose.
            # This should make dpos≈0 and drot≈0 before any key is pressed.
            if self.teleop_mode and not self._teleop_ee_synced_once:
                self._sync_teleop_ee_goal_to_current_pose()
                self._teleop_ee_synced_once = True
                print("[teleop][sync] initialized EE goal from current EE pose")
        else:
            self.ee_pos = None
            self.ee_orn = None

        # Arm DOFs are non-contiguous in IsaacLab sim order: [8, 13, 14, 15, 16, 17].
        # Never use arm_dof_start_idx:arm_dof_end_idx for IsaacLab arm tensors.
        if self.arm_joint_ids.numel() > 0:
            self.arm_dof_start_idx = int(self.arm_joint_ids.min().item())
            self.arm_dof_end_idx = int(self.arm_joint_ids.max().item()) + 1

        self.ee_j_eef = None
        root_view = getattr(self.robot, "root_physx_view", None)
        if root_view is not None and hasattr(root_view, "get_jacobians"):
            try:
                jac = root_view.get_jacobians()
                if jac is not None:
                    # Use explicit arm IDs because IsaacLab sim joint order differs from policy order.
                    # self.ee_j_eef = jac[:, self.gripper_idx, :6, self.arm_joint_ids]
                    floating_base_offset = jac.shape[-1] - self.num_dofs  # should be 6 for floating base
                    jac_arm_joint_ids = self.arm_joint_ids + floating_base_offset
                    self.ee_j_eef = jac[:, self.gripper_idx, :6, jac_arm_joint_ids]
            except Exception:
                self.ee_j_eef = None


    @property
    def base_pos(self):
        return self.root_states[:, :3]

    def _update_base_yaw_quat(self):
        q = self.base_quat
        w, x, y, z = q.unbind(-1)
        base_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        half_yaw = 0.5 * base_yaw
        if not hasattr(self, "base_yaw_quat"):
            self.base_yaw_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.base_yaw_quat[:, 0] = torch.cos(half_yaw)
        self.base_yaw_quat[:, 1:3] = 0.0
        self.base_yaw_quat[:, 3] = torch.sin(half_yaw)

    def _sync_teleop_arm_joint_targets_to_current_pose(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        self.teleop_arm_joint_pos_targets[env_ids] = self.dof_pos[env_ids][:, self.arm_joint_ids]

    def _sync_teleop_ee_goal_to_current_pose(self, env_ids=None):
        """Old Gym-compatible sync: make the EE target equal the actual EE pose."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        if self.ee_pos is None or self.ee_orn is None:
            return

        self._update_base_yaw_quat()
        goal_ref_origin = self.get_goal_reference_origin()[env_ids]
        goal_ref_quat = self.get_goal_reference_quat()[env_ids]
        goal_center_offset = self.get_goal_center_offset_local()[env_ids]
        ee_orn_normalized = self.ee_orn[env_ids] / torch.norm(
            self.ee_orn[env_ids], dim=-1, keepdim=True
        ).clamp_min(1e-8)

        ee_goal_local_with_center = quat_rotate_inverse(
            goal_ref_quat, self.ee_pos[env_ids] - goal_ref_origin
        )
        self.teleop_raw_ee_goal_cart[env_ids] = ee_goal_local_with_center - goal_center_offset
        self.curr_ee_goal_cart[env_ids] = self.teleop_raw_ee_goal_cart[env_ids]
        self.curr_ee_goal_sphere[env_ids] = cart2sphere(self.curr_ee_goal_cart[env_ids])

        local_ee_orn = quat_mul(quat_conjugate(goal_ref_quat), ee_orn_normalized)
        local_ee_orn_rpy = torch.stack(euler_from_quat(local_ee_orn), dim=-1)
        default_pitch = -self.curr_ee_goal_sphere[env_ids, 1] + self.arm_induced_pitch
        self.teleop_raw_ee_goal_orn_delta_rpy[env_ids, 0] = wrap_to_pi(local_ee_orn_rpy[:, 0] - np.pi / 2)
        self.teleop_raw_ee_goal_orn_delta_rpy[env_ids, 1] = wrap_to_pi(local_ee_orn_rpy[:, 1] - default_pitch)
        self.teleop_raw_ee_goal_orn_delta_rpy[env_ids, 2] = wrap_to_pi(
            local_ee_orn_rpy[:, 2] - self.curr_ee_goal_sphere[env_ids, 2]
        )
        self.ee_goal_orn_delta_rpy[env_ids] = self.teleop_raw_ee_goal_orn_delta_rpy[env_ids]
        self.teleop_hold_actual_ee_target[env_ids] = True

        self.curr_ee_goal_cart_world[env_ids] = self.ee_pos[env_ids]
        self.ee_goal_orn_quat[env_ids] = ee_orn_normalized
        self.ee_goal_orn_euler[env_ids] = torch.stack(
            euler_from_quat(self.ee_goal_orn_quat[env_ids]), dim=-1
        )

    def _set_teleop_arm_control_mode(self, mode):
        if mode == self.teleop_arm_control_mode:
            return
        if mode == "joint":
            self._sync_teleop_arm_joint_targets_to_current_pose()
        elif mode == "ee":
            self._sync_teleop_ee_goal_to_current_pose()
        else:
            raise ValueError(f"Unsupported teleop arm control mode: {mode}")
        self.teleop_arm_control_mode = mode
        print(f"[teleop] arm control mode: {mode}")

    def _reset_teleop_ee_goal_to_default(self):
        self.curr_ee_goal_sphere[:] = self.reset_init_ee_sphere[:]
        self.teleop_raw_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        self.teleop_raw_ee_goal_orn_delta_rpy[:] = 0.0
        self.ee_goal_orn_delta_rpy[:] = 0.0
        self.teleop_hold_actual_ee_target[:] = False
        self._update_effective_teleop_inputs()


    def _reset_task_specific_buffers_on_reset(self, env_ids: torch.Tensor):
        """ManipLoco-specific reset hook."""
        self._resample_goal_height_reference(env_ids, is_init=True)
        self._resample_ee_goal(env_ids, is_init=True)

    def _get_omnidirectional_goal_sampling_bounds(self, goal_yaw):
        rear_weight = self._get_omnidirectional_rear_weight(goal_yaw)

        pos_l_min_front = torch.full_like(goal_yaw, self.omnidirectional_pos_l[0].item())
        pos_l_min_back = torch.full_like(goal_yaw, self.omnidirectional_rear_pos_l[0].item())
        pos_l_min = torch.lerp(pos_l_min_front, pos_l_min_back, rear_weight)

        pos_l_max_front = torch.full_like(goal_yaw, self.omnidirectional_pos_l[1].item())
        pos_l_max_back = torch.full_like(goal_yaw, self.omnidirectional_rear_pos_l[1].item())
        pos_l_max = torch.lerp(pos_l_max_front, pos_l_max_back, rear_weight)

        pos_p_min_front = torch.full_like(goal_yaw, float(self.goal_ee_ranges["pos_p"][0]))
        pos_p_max_front = torch.full_like(goal_yaw, float(self.goal_ee_ranges["pos_p"][1]))
        pos_p_min_back = torch.full_like(goal_yaw, self.omnidirectional_rear_pos_p[0].item())
        pos_p_max_back = torch.full_like(goal_yaw, self.omnidirectional_rear_pos_p[1].item())
        pos_p_min = torch.lerp(pos_p_min_front, pos_p_min_back, rear_weight)
        pos_p_max = torch.lerp(pos_p_max_front, pos_p_max_back, rear_weight)

        return pos_l_min, pos_l_max, pos_p_min, pos_p_max

    def _get_omnidirectional_rear_weight(self, goal_yaw):
        abs_yaw = torch.abs(wrap_to_pi(goal_yaw))
        denom = np.pi - self.omnidirectional_rear_transition_pos_y_abs
        transition = (abs_yaw - self.omnidirectional_rear_transition_pos_y_abs) / denom
        transition = torch.clamp(transition, 0.0, 1.0)
        return transition * transition * (3.0 - 2.0 * transition)

    def _get_body_importance_direction_ids(self):
        mount_quadrant = int(round((self.arm_mount_yaw_offset % (2 * np.pi)) / (np.pi / 2))) % 4
        opposite_direction = (mount_quadrant + 2) % 4
        direction_ids = [direction_id for direction_id in range(4) if direction_id != opposite_direction]
        return torch.tensor(direction_ids, device=self.device, dtype=torch.long)

    def _sample_body_importance_region(
        self, mask, x_min, x_max, y_min, y_max, z_range, cart, ground_weighted_z=True
    ):
        num_samples = int(mask.sum().item())
        if num_samples == 0:
            return
        rand = torch.rand(num_samples, 3, device=self.device)
        cart[mask, 0] = x_min + (x_max - x_min) * rand[:, 0]
        cart[mask, 1] = y_min + (y_max - y_min) * rand[:, 1]

        if not ground_weighted_z:
            cart[mask, 2] = z_range[0] + (z_range[1] - z_range[0]) * rand[:, 2]
            return

        ground_ratio = min(1.0, max(0.0, self.body_importance_ground_height_ratio))
        ground_z_min = float(self.body_importance_ground_height[0].item())
        ground_z_max = float(self.body_importance_ground_height[1].item())
        ground_mask = torch.rand(num_samples, device=self.device) < ground_ratio
        sampled_z = torch.empty(num_samples, device=self.device)
        sampled_z[ground_mask] = ground_z_min + (ground_z_max - ground_z_min) * torch.rand(
            int(ground_mask.sum().item()), device=self.device
        )
        sampled_z[~ground_mask] = z_range[0] + (z_range[1] - z_range[0]) * torch.rand(
            int((~ground_mask).sum().item()), device=self.device
        )
        cart[mask, 2] = sampled_z

    def _resample_ee_goal_body_importance_once(self, env_ids, is_init=False):
        num_envs = len(env_ids)
        if num_envs == 0:
            return

        if self.ee_goal_sampling_mode == "body_importance_single_dir":
            sampled_directions = self.body_importance_single_direction_ids[env_ids]
        else:
            sampled_direction_indices = torch.randint(
                0,
                len(self.body_importance_direction_ids),
                (num_envs,),
                device=self.device,
            )
            sampled_directions = self.body_importance_direction_ids[sampled_direction_indices]

        back_mask = sampled_directions == 2
        start_goal_world = self._get_reset_init_goal_world(env_ids) if is_init else self.curr_ee_goal_cart_world[env_ids]
        if self.uses_goal_height_reference_mask:
            self.goal_height_follow_mask[env_ids] = back_mask
        self.ee_start_sphere[env_ids] = self._project_world_points_to_goal_sphere(env_ids, start_goal_world)

        extension = self.body_importance_extension
        body_x_min = float(self.collision_lower_limits[0].item())
        body_y_min = float(self.collision_lower_limits[1].item())
        body_x_max = float(self.collision_upper_limits[0].item())
        body_y_max = float(self.collision_upper_limits[1].item())
        back_x_max = min(max(-self.body_importance_back_start_offset, body_x_min), 0.0)

        side_z_range = (
            float(self.body_importance_side_height[0].item()),
            float(self.body_importance_side_height[1].item()),
        )
        back_z_range = (
            float(self.body_importance_back_height[0].item()),
            float(self.body_importance_back_height[1].item()),
        )
        cart_from_goal_center = torch.zeros(num_envs, 3, device=self.device)

        front_mask = sampled_directions == 0
        self._sample_body_importance_region(
            front_mask,
            body_x_max,
            body_x_max + extension,
            body_y_min - extension,
            body_y_max + extension,
            side_z_range,
            cart_from_goal_center,
        )

        left_mask = sampled_directions == 1
        self._sample_body_importance_region(
            left_mask,
            body_x_min,
            body_x_max,
            body_y_max,
            body_y_max + extension,
            side_z_range,
            cart_from_goal_center,
        )

        self._sample_body_importance_region(
            back_mask,
            body_x_min,
            back_x_max,
            body_y_min,
            body_y_max,
            back_z_range,
            cart_from_goal_center,
            ground_weighted_z=False,
        )

        right_mask = sampled_directions == 3
        self._sample_body_importance_region(
            right_mask,
            body_x_min,
            body_x_max,
            body_y_min - extension,
            body_y_min,
            side_z_range,
            cart_from_goal_center,
        )

        self.ee_goal_sphere[env_ids, :] = cart2sphere(cart_from_goal_center)

    def _interpolate_goal_sphere(self, start_sphere, goal_sphere, t):
        interp_sphere = torch.lerp(start_sphere, goal_sphere, t)
        yaw_t = t[..., 0] if t.ndim > 0 and t.shape[-1] == 1 else t
        start_yaw_arm_front = wrap_to_pi(start_sphere[..., 2] - self.arm_mount_yaw_offset)
        goal_yaw_arm_front = wrap_to_pi(goal_sphere[..., 2] - self.arm_mount_yaw_offset)
        interp_sphere[..., 2] = wrap_to_pi(
            torch.lerp(start_yaw_arm_front, goal_yaw_arm_front, yaw_t) + self.arm_mount_yaw_offset
        )
        return interp_sphere

    def _resample_goal_timing(self, env_ids):
        if len(env_ids) == 0:
            return

        base_traj_time = torch_rand_float(
            self.cfg.goal_ee.traj_time[0],
            self.cfg.goal_ee.traj_time[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        yaw_scale = torch.ones(len(env_ids), device=self.device)
        if self.ee_goal_sampling_mode in ("body_importance", "body_importance_single_dir"):
            start_yaw_arm_front = wrap_to_pi(self.ee_start_sphere[env_ids, 2] - self.arm_mount_yaw_offset)
            goal_yaw_arm_front = wrap_to_pi(self.ee_goal_sphere[env_ids, 2] - self.arm_mount_yaw_offset)
            yaw_delta_abs = torch.abs(goal_yaw_arm_front - start_yaw_arm_front)

            yaw_ref = max(float(self.cfg.goal_ee.yaw_adaptive_time_ref), 1e-6)
            min_scale = float(self.cfg.goal_ee.yaw_adaptive_time_min_scale)
            max_scale = float(self.cfg.goal_ee.yaw_adaptive_time_max_scale)
            if max_scale < min_scale:
                raise ValueError(
                    "cfg.goal_ee.yaw_adaptive_time_max_scale must be >= "
                    "cfg.goal_ee.yaw_adaptive_time_min_scale"
                )

            yaw_scale = torch.clamp(yaw_delta_abs / yaw_ref, min=min_scale, max=max_scale)

        traj_time = base_traj_time * yaw_scale
        hold_time = torch_rand_float(
            self.cfg.goal_ee.hold_time[0],
            self.cfg.goal_ee.hold_time[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.traj_timesteps[env_ids] = traj_time / self.dt
        self.traj_total_timesteps[env_ids] = self.traj_timesteps[env_ids] + hold_time / self.dt

    def _resample_goal_height_reference(self, env_ids, is_init=False):
        if not self.uses_goal_height_reference_mask or len(env_ids) == 0:
            return

        if self.teleop_mode and is_init:
            if self.goal_height_follow_override is None:
                self.goal_height_follow_mask[env_ids] = False
            else:
                self.goal_height_follow_mask[env_ids] = self.goal_height_follow_override
            return
        elif self.teleop_mode:
            return

        if self.ee_goal_sampling_mode in ("body_importance", "body_importance_single_dir"):
            return

        trunk_follow_ratio = (
            self.cfg.goal_ee.sphere_center.trunk_follow_ratio
            if self.cfg.goal_ee.sphere_center.mixed_height_reference
            else 0.0
        )
        self.goal_height_follow_mask[env_ids] = torch.rand(len(env_ids), device=self.device) < trunk_follow_ratio

    def _resample_ee_goal(self, env_ids, is_init=False):
        if self.teleop_mode and is_init:
            if self.teleop_initialize_targets_on_next_reset:
                self._reset_teleop_ee_goal_to_default()
                self.teleop_initialize_targets_on_next_reset = False
            return
        elif self.teleop_mode:
            return

        if len(env_ids) == 0:
            return

        init_env_ids = env_ids.clone()

        if is_init:
            self.ee_goal_orn_delta_rpy[env_ids, :] = 0.0
            self.ee_start_sphere[env_ids] = self.reset_init_ee_sphere[env_ids].clone()
        else:
            self._resample_ee_goal_orn_once(env_ids)
            self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()

        for _ in range(10):
            self._resample_ee_goal_position_once(env_ids, is_init=is_init)
            collision_mask = self._collision_check(env_ids)
            env_ids = env_ids[collision_mask]
            if len(env_ids) == 0:
                break

        self.ee_goal_cart[init_env_ids, :] = sphere2cart(self.ee_goal_sphere[init_env_ids, :])
        self._resample_goal_timing(init_env_ids)
        if self.ee_goal_sampling_mode == "body_importance_single_dir":
            self.traj_total_timesteps[init_env_ids] = self.max_episode_length + 1
        self.goal_timer[init_env_ids] = 0.0

    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_delta_orn_r = torch_rand_float(
            self.goal_ee_ranges["delta_orn_r"][0],
            self.goal_ee_ranges["delta_orn_r"][1],
            (len(env_ids), 1),
            device=self.device,
        )
        ee_goal_delta_orn_p = torch_rand_float(
            self.goal_ee_ranges["delta_orn_p"][0],
            self.goal_ee_ranges["delta_orn_p"][1],
            (len(env_ids), 1),
            device=self.device,
        )
        ee_goal_delta_orn_y = torch_rand_float(
            self.goal_ee_ranges["delta_orn_y"][0],
            self.goal_ee_ranges["delta_orn_y"][1],
            (len(env_ids), 1),
            device=self.device,
        )
        self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat(
            [ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y],
            dim=-1,
        )

    def _resample_ee_goal_position_once(self, env_ids, is_init=False):
        if self.ee_goal_sampling_mode in ("body_importance", "body_importance_single_dir"):
            self._resample_ee_goal_body_importance_once(env_ids, is_init=is_init)
            return

        sampled_yaw = torch_rand_float(
            self.goal_ee_ranges["pos_y"][0],
            self.goal_ee_ranges["pos_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        if self.is_omnidirectional_sampling:
            start_yaw = wrap_to_pi(self.ee_start_sphere[env_ids, 2] - self.arm_mount_yaw_offset)
            goal_yaw = torch.clamp(sampled_yaw + start_yaw, min=-np.pi + 1e-6, max=np.pi - 1e-6)
            self.ee_goal_sphere[env_ids, 2] = wrap_to_pi(goal_yaw + self.arm_mount_yaw_offset)
            pos_l_min, pos_l_max, pos_p_min, pos_p_max = self._get_omnidirectional_goal_sampling_bounds(goal_yaw)
            self.ee_goal_sphere[env_ids, 0] = pos_l_min + (pos_l_max - pos_l_min) * torch.rand(
                len(env_ids), device=self.device
            )
            self.ee_goal_sphere[env_ids, 1] = pos_p_min + (pos_p_max - pos_p_min) * torch.rand(
                len(env_ids), device=self.device
            )
            return

        self.ee_goal_sphere[env_ids, 2] = wrap_to_pi(sampled_yaw + self.arm_mount_yaw_offset)
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(
            self.goal_ee_ranges["pos_l"][0],
            self.goal_ee_ranges["pos_l"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(
            self.goal_ee_ranges["pos_p"][0],
            self.goal_ee_ranges["pos_p"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

    def _collision_check(self, env_ids):
        ee_target_all_sphere = self._interpolate_goal_sphere(
            self.ee_start_sphere[env_ids][None, ...],
            self.ee_goal_sphere[env_ids][None, ...],
            self.collision_check_t.squeeze(0).squeeze(0)[:, None, None],
        )
        ee_target_cart = sphere2cart(ee_target_all_sphere.reshape(-1, 3)).reshape(
            self.num_collision_check_samples,
            -1,
            3,
        )
        collision_mask = torch.any(
            torch.logical_and(
                torch.all(ee_target_cart < self.collision_upper_limits, dim=-1),
                torch.all(ee_target_cart > self.collision_lower_limits, dim=-1),
            ),
            dim=0,
        )
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def _update_curr_ee_goal(self, refresh_base_yaw: bool = True):
        """Update the EE target using the configured goal sampling mode."""
        if not self.teleop_mode:
            t = torch.clip(self.goal_timer / self.traj_timesteps, 0.0, 1.0)
            self.curr_ee_goal_sphere[:] = self._interpolate_goal_sphere(
                self.ee_start_sphere,
                self.ee_goal_sphere,
                t[:, None],
            )
            self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        else:
            self._update_effective_teleop_inputs(force=False)

        if refresh_base_yaw:
            self._update_base_yaw_quat()
        invariant_origin = self.get_invariant_goal_reference_origin()
        if self.uses_goal_height_reference_mask:
            goal_ref_quat = self._goal_ref_quat_buf
            goal_ref_origin = self._goal_ref_origin_buf
            torch.where(
                self.goal_height_follow_mask.unsqueeze(1),
                self.base_quat,
                self.base_yaw_quat,
                out=goal_ref_quat,
            )
            torch.where(
                self.goal_height_follow_mask.unsqueeze(1),
                self.robot.data.root_pos_w,
                invariant_origin,
                out=goal_ref_origin,
            )
        else:
            goal_ref_quat = self.base_yaw_quat
            goal_ref_origin = invariant_origin

        goal_world, goal_quat = _compute_ee_goal_pose_from_ref(
            goal_ref_quat,
            goal_ref_origin,
            self.get_goal_center_offset_local(),
            self.curr_ee_goal_cart,
            self.curr_ee_goal_sphere,
            self.ee_goal_orn_delta_rpy,
            self.arm_induced_pitch,
        )
        self.curr_ee_goal_cart_world.copy_(goal_world)
        self.ee_goal_orn_quat.copy_(goal_quat)
        if bool(self.cfg.compute_rewards):
            self.ee_goal_orn_euler = torch.stack(euler_from_quat(self.ee_goal_orn_quat), dim=-1)

        if self.teleop_mode:
            return

        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()

        if len(resample_id) > 0 and self.cfg.env.stop_update_goal:
            self.commands[resample_id, 0] = 0.0
            self.commands[resample_id, 1] = 0.0
            self.commands[resample_id, 2] = 0.0
            if self.teleop_mode:
                self.teleop_raw_commands[resample_id, 0] = 0.0
                self.teleop_raw_commands[resample_id, 1] = 0.0
                self.teleop_raw_commands[resample_id, 2] = 0.0

        self._resample_ee_goal(resample_id)

    def _control_ik(self, dpose: torch.Tensor) -> torch.Tensor:
        """Damped least-squares IK for the arm target path."""
        j_eef_T = torch.transpose(self.ee_j_eef, 1, 2)
        damping_sq = float(self.cfg.goal_ee.ik_damping) ** 2
        damping_matrix = getattr(self, "_ik_damping_matrix", None)
        if (
            damping_matrix is None
            or getattr(self, "_ik_damping_sq", None) != damping_sq
            or damping_matrix.device != self.ee_j_eef.device
            or damping_matrix.dtype != self.ee_j_eef.dtype
        ):
            damping_matrix = torch.eye(6, device=self.ee_j_eef.device, dtype=self.ee_j_eef.dtype).mul_(damping_sq)
            self._ik_damping_matrix = damping_matrix.unsqueeze(0)
            self._ik_damping_sq = damping_sq
            damping_matrix = self._ik_damping_matrix
        A = torch.bmm(self.ee_j_eef, j_eef_T)
        A.add_(damping_matrix)
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))#.view(self.num_envs, 6)
        return u.squeeze(-1)

    @staticmethod
    def _quat_mul_np_wxyz(q, r):
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = r
        return np.asarray(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dtype=np.float32,
        )

    def _get_arm_pos_targets_single_env_cpu(self) -> torch.Tensor:
        ee_pos = self.ee_pos[0].detach().cpu().numpy().astype(np.float32, copy=False)
        ee_orn = self.ee_orn[0].detach().cpu().numpy().astype(np.float32, copy=False)
        goal_pos = self.curr_ee_goal_cart_world[0].detach().cpu().numpy().astype(np.float32, copy=False)
        goal_orn = self.ee_goal_orn_quat[0].detach().cpu().numpy().astype(np.float32, copy=False)
        jac = self.ee_j_eef[0].detach().cpu().numpy().astype(np.float32, copy=False)
        arm_q = self.dof_pos[0, self.arm_joint_ids].detach().cpu().numpy().astype(np.float32, copy=False)

        ee_orn_norm = ee_orn / max(float(np.linalg.norm(ee_orn)), 1.0e-8)
        current_conj = ee_orn_norm.copy()
        current_conj[1:] *= -1.0
        q_r = self._quat_mul_np_wxyz(goal_orn, current_conj)
        sign = 1.0 if q_r[0] >= 0.0 else -1.0

        dpose = np.empty((6, 1), dtype=np.float32)
        dpose[:3, 0] = goal_pos - ee_pos
        dpose[3:, 0] = q_r[1:4] * sign

        damping_sq = np.float32(float(self.cfg.goal_ee.ik_damping) ** 2)
        j_t = jac.T
        a = jac @ j_t
        a.flat[::7] += damping_sq
        delta_q = (j_t @ np.linalg.solve(a, dpose)).reshape(-1).astype(np.float32, copy=False)
        target_np = (arm_q + delta_q).reshape(1, -1)

        target_buf = getattr(self, "_arm_pos_target_cpu_fast_buf", None)
        if (
            target_buf is None
            or target_buf.shape != (1, self.arm_joint_ids.numel())
            or target_buf.device != self.device
            or target_buf.dtype != self.dof_pos.dtype
        ):
            target_buf = torch.empty(1, self.arm_joint_ids.numel(), device=self.device, dtype=self.dof_pos.dtype)
            self._arm_pos_target_cpu_fast_buf = target_buf
        target_buf.copy_(torch.from_numpy(target_np).to(device=self.device, dtype=self.dof_pos.dtype))
        return target_buf

    def _get_arm_pos_targets(self) -> torch.Tensor:
        """Arm target generation.

        Gym formula:
            dpose = [goal_pos_world - ee_pos, orientation_error(goal_quat, ee_quat)]
            arm_q_target = q_arm_now + damped_least_squares_IK(dpose)

        IsaacLab difference: arm joints are non-contiguous in sim order, so use arm_joint_ids.
        """
        if self.ee_pos is None or self.ee_orn is None or self.ee_j_eef is None:
            return self.target_pos[:, self.arm_joint_ids]

        if self.teleop_mode and self.num_envs == 1:
            return self._get_arm_pos_targets_single_env_cpu()

        dpos = self.curr_ee_goal_cart_world - self.ee_pos
        ee_orn_norm = self.ee_orn / torch.norm(self.ee_orn, dim=-1, keepdim=True).clamp_min(1e-8)
        drot = orientation_error(self.ee_goal_orn_quat, ee_orn_norm)
        dpose = getattr(self, "_ik_dpose_buf", None)
        if dpose is None or dpose.shape != (self.num_envs, 6, 1) or dpose.device != dpos.device or dpose.dtype != dpos.dtype:
            dpose = torch.zeros(self.num_envs, 6, 1, device=dpos.device, dtype=dpos.dtype)
            self._ik_dpose_buf = dpose
        dpose[:, :3, 0] = dpos
        dpose[:, 3:, 0] = drot
        return self._control_ik(dpose) + self.dof_pos[:, self.arm_joint_ids]

    def get_goal_reference_quat(self):
        """Returns the goal-reference orientation in world coordinates."""
        self._update_base_yaw_quat()
        if not self.uses_goal_height_reference_mask:
            return self.base_yaw_quat
        return torch.where(self.goal_height_follow_mask.unsqueeze(1), self.base_quat, self.base_yaw_quat)

    def get_goal_reference_origin(self):
        """Returns the goal-reference origin in world coordinates."""
        invariant_origin = self.get_invariant_goal_reference_origin()
        if not self.uses_goal_height_reference_mask:
            return invariant_origin
        return torch.where(self.goal_height_follow_mask.unsqueeze(1), self.base_pos, invariant_origin)

    def get_invariant_goal_reference_origin(self, env_ids=None):
        if self.cfg.terrain.terrain_type == "plane":
            if env_ids is None:
                root_pos = self.robot.data.root_pos_w
                origin = getattr(self, "_invariant_goal_origin_buf", None)
                if (
                    origin is None
                    or origin.shape != (self.num_envs, 3)
                    or origin.device != root_pos.device
                    or origin.dtype != root_pos.dtype
                ):
                    origin = torch.zeros(self.num_envs, 3, device=root_pos.device, dtype=root_pos.dtype)
                    self._invariant_goal_origin_buf = origin
                origin[:, :2] = root_pos[:, :2]
                origin[:, 2].zero_()
                return origin
            root_xy = self.robot.data.root_pos_w[env_ids, :2]
            origin = torch.zeros(root_xy.shape[0], 3, device=root_xy.device, dtype=root_xy.dtype)
            origin[:, :2] = root_xy
            return origin

        if env_ids is None:
            root_xy = self.robot.data.root_pos_w[:, :2]
        else:
            root_xy = self.robot.data.root_pos_w[env_ids, :2]
        ground_z = self.get_ground_heights(root_xy, env_ids=env_ids).view(-1, 1)
        return torch.cat([root_xy, ground_z], dim=1)

    def get_goal_center_offset_local(self):
        """Returns the target-center offset in the goal local frame."""
        if not self.uses_goal_height_reference_mask:
            return self.ee_goal_center_offset
        trunk_follow_anchor = self.cfg.goal_ee.sphere_center.trunk_follow_anchor
        if trunk_follow_anchor == "arm_base":
            trunk_follow_center_offset = self.arm_base_offset
        elif trunk_follow_anchor == "arm_waist":
            trunk_follow_center_offset = self.arm_waist_offset
        elif trunk_follow_anchor == "arm_shoulder":
            trunk_follow_center_offset = self.arm_shoulder_offset
        else:
            raise ValueError(f"Unsupported trunk_follow_anchor: {trunk_follow_anchor}")
        return torch.where(
            self.goal_height_follow_mask.unsqueeze(1),
            trunk_follow_center_offset,
            self.ee_goal_center_offset,
        )

    def transform_goal_local_to_world(self, local_points):
        """Maps points from the goal local frame to world coordinates."""
        return self.get_goal_reference_origin() + quat_apply(self.get_goal_reference_quat(), local_points)

    def get_ee_goal_spherical_center(self):
        """Returns the cyan-sphere center in world coordinates."""
        return self.transform_goal_local_to_world(self.get_goal_center_offset_local())

    def _project_world_points_to_goal_sphere(self, env_ids, world_points):
        goal_ref_quat = self.get_goal_reference_quat()[env_ids]
        goal_ref_origin = self.get_goal_reference_origin()[env_ids]
        goal_center_offset = self.get_goal_center_offset_local()[env_ids]
        local_with_center = quat_rotate_inverse(goal_ref_quat, world_points - goal_ref_origin)
        return cart2sphere(local_with_center - goal_center_offset)

    def _get_reset_init_goal_world(self, env_ids):
        reset_init_cart = sphere2cart(self.reset_init_ee_sphere[env_ids])
        invariant_origin = self.get_invariant_goal_reference_origin(env_ids)
        invariant_center = self.ee_goal_center_offset[env_ids]
        return invariant_origin + quat_apply(self.base_yaw_quat[env_ids], invariant_center + reset_init_cart)

    def _get_arm_base_world_pos(self):
        """Returns the arm-base origin in world coordinates."""
        self._update_base_yaw_quat()
        arm_base_quat = self.base_yaw_quat
        if self.uses_goal_height_reference_mask:
            arm_base_quat = torch.where(self.goal_height_follow_mask.unsqueeze(1), self.base_quat, self.base_yaw_quat)
        return self.base_pos + quat_apply(arm_base_quat, self.arm_base_offset)

    def _apply_action(self):
        """Apply arm/gripper position targets in the ManipLoco control path."""
        profile_start = time.perf_counter()
        section_start = time.perf_counter()
        self.torques = self._compute_torques(self.actions)
        self._profile_record("torques", time.perf_counter() - section_start)

        section_start = time.perf_counter()
        self.robot.set_joint_effort_target(self.torques)
        self._profile_record("set_effort", time.perf_counter() - section_start)

        section_start = time.perf_counter()
        pos_targets = self._joint_position_target_buf
        pos_targets.copy_(self.dof_pos)
        pos_targets[:, self.gripper_joint_ids] = self.gripper_pos_targets

        if self.teleop_mode and self.teleop_arm_control_mode == "joint":
            pos_targets[:, self.arm_joint_ids] = self.teleop_arm_joint_pos_targets
        elif self._arm_pos_targets_for_step is not None:
            pos_targets[:, self.arm_joint_ids] = self._arm_pos_targets_for_step
        else:
            pos_targets[:, self.arm_joint_ids] = self.target_pos[:, self.arm_joint_ids]
            pos_targets[:, self.gripper_joint_ids] = self.target_pos[:, self.gripper_joint_ids]
        self._profile_record("pos_target", time.perf_counter() - section_start)

        section_start = time.perf_counter()
        self.robot.set_joint_position_target(pos_targets)
        self._profile_record("set_position", time.perf_counter() - section_start)
        
        if self.teleop_mode and self.teleop_debug and int(self.global_steps) % 5 == 0:
            self._draw_ee_goal_curr()        
        self._profile_record("apply_total", time.perf_counter() - profile_start)

    def _draw_points(self, points_w: torch.Tensor, colors, size: float = 12.0):
        """Draw world-frame points with IsaacSim debug draw."""
        if not self._debug_draw:
            return
        if points_w.numel() == 0:
            return

        pts = points_w.detach().cpu().float().tolist()
        cols = [colors for _ in pts]
        sizes = [size for _ in pts]
        self._debug_draw.draw_points(pts, cols, sizes)

    def _get_debug_draw(self):
        """Lazy acquire IsaacSim debug draw interface."""
        if not (self.teleop_mode and self.teleop_debug):
            return None

        if self._debug_draw is not None:
            return self._debug_draw

        self._debug_draw = None
        try:
            from isaacsim.util.debug_draw import _debug_draw
            self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            print("[debug_draw] acquired from isaacsim.util.debug_draw")
        except Exception as e1:
            try:
                from omni.isaac.debug_draw import _debug_draw
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
                print("[debug_draw] acquired from omni.isaac.debug_draw")
            except Exception as e2:
                print(f"[debug_draw][warn] unavailable: {e1} / {e2}")
                self._debug_draw = False

        return self._debug_draw

    def _debug_draw_clear(self, clear_points=True, clear_lines=True):
        draw = self._get_debug_draw()
        if draw is None or draw is False:
            return

        try:
            if clear_points and hasattr(draw, "clear_points"):
                draw.clear_points()
        except Exception as e:
            print(f"[debug_draw][warn] clear_points failed: {e}")

        try:
            if clear_lines and hasattr(draw, "clear_lines"):
                draw.clear_lines()
        except Exception as e:
            print(f"[debug_draw][warn] clear_lines failed: {e}")

    def _draw_ee_goal_curr(self, env_ids=None):
        """Safe IsaacSim debug draw version.

        Yellow: curr_ee_goal_cart_world
        Blue:   ee_pos
        Cyan:   EE goal sphere center
        White:  robot root
        Green:  world origin

        Important:
        - no clear_points / clear_lines
        - no _refresh_ee_and_jacobian_for_ik inside draw
        - no axes/lines first
        - draw only low frequency from caller
        """
        draw = self._get_debug_draw()
        if draw is None or draw is False:
            return

        step = int(self.global_steps)

        # clear less frequently to avoid viewer/debug-draw stall
        if step % 5 == 0:
            self._debug_draw_clear(clear_points=True, clear_lines=True)
            
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

        points = []
        colors = []
        sizes = []

        def add_points(tensor_points, color, size):
            if tensor_points is None:
                return
            if tensor_points.numel() == 0:
                return
            pts = tensor_points.detach().cpu().float().tolist()
            points.extend(pts)
            colors.extend([color] * len(pts))
            sizes.extend([float(size)] * len(pts))

        # Yellow: target EE goal in world.
        add_points(self.curr_ee_goal_cart_world[env_ids], (1.0, 1.0, 0.0, 1.0), 12.0)

        # Blue: measured EE position. Do NOT refresh here; draw should be read-only.
        if self.ee_pos is not None:
            add_points(self.ee_pos[env_ids], (0.0, 0.0, 1.0, 1.0), 10.0)

        # Cyan: get_ee_goal_spherical_center().
        try:
            add_points(self.get_ee_goal_spherical_center()[env_ids], (0.0, 1.0, 1.0, 1.0), 10.0)
        except Exception as e:
            if int(self.global_steps) % 200 == 0:
                print(f"[debug_draw][warn] goal center draw skipped: {e}")

        # White: robot root.
        try:
            if hasattr(self.robot.data, "root_pos_w"):
                add_points(self.robot.data.root_pos_w[env_ids], (1.0, 1.0, 1.0, 1.0), 10.0)
        except Exception:
            pass

        # Green: world origin.
        add_points(torch.zeros(1, 3, device=self.device), (0.0, 1.0, 0.0, 1.0), 14.0)

        if len(points) == 0:
            return

        try:
            draw.draw_points(points, colors, sizes)
        except Exception as e:
            print(f"[debug_draw][error] draw_points failed: {e}")

    def enable_stereo_rgb_camera_debug_draw(self, enabled=True):
        self._rgb_camera_debug_draw_enabled = bool(enabled)

    def rgb_camera_backend_available(self):
        return True

    def _camera_rgb_to_numpy(self, camera, env_id, update=False):
        if update:
            camera.update(self.dt)
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            return None
        image = rgb[int(env_id)].detach().cpu()
        if image.shape[-1] > 3:
            image = image[..., :3]
        return image.numpy().astype(np.uint8)

    def _tiled_camera_rgb_to_numpy(self, camera, sensor_index):
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            raise RuntimeError("IsaacLab tiled camera did not produce an 'rgb' output.")
        image = rgb[int(sensor_index)].detach().cpu()
        if image.shape[-1] > 3:
            image = image[..., :3]
        return image.numpy().astype(np.uint8)

    def render_stereo_rgb_camera_rigs(self, env_id=0, rig_names=None, update=False):
        images_by_name = {}
        selected_names = set(rig_names) if rig_names is not None else None

        if update:
            for rig_name, rig in self._rgb_camera_rigs.items():
                if selected_names is not None and rig_name not in selected_names:
                    continue
                if bool(rig.get("pose_update", True)):
                    self._update_rgb_camera_world_poses(rig, env_id)

        updated_tiled_cameras = set()
        for rig_name, rig in self._rgb_camera_rigs.items():
            if selected_names is not None and rig_name not in selected_names:
                continue
            cameras = rig["cameras"]
            if rig.get("backend") == "tiled":
                tiled_camera = cameras["tiled"]
                camera_key = id(tiled_camera)
                if update and camera_key not in updated_tiled_cameras:
                    tiled_camera.update(self.dt)
                    updated_tiled_cameras.add(camera_key)
                eye_images = {
                    eye["name"]: self._tiled_camera_rgb_to_numpy(tiled_camera, eye["sensor_index"])
                    for eye in rig.get("eyes", [])
                }
                if rig["mode"] == "mono":
                    images_by_name[rig_name] = [eye_images["mono"]]
                else:
                    images_by_name[rig_name] = [eye_images["left"], eye_images["right"]]
            elif rig["mode"] == "mono":
                mono_rgb = self._camera_rgb_to_numpy(cameras["mono"], env_id, update=update)
                if mono_rgb is not None:
                    images_by_name[rig_name] = [mono_rgb]
            else:
                left_rgb = self._camera_rgb_to_numpy(cameras["left"], env_id, update=update)
                right_rgb = self._camera_rgb_to_numpy(cameras["right"], env_id, update=update)
                if left_rgb is not None and right_rgb is not None:
                    images_by_name[rig_name] = [left_rgb, right_rgb]
        return images_by_name
