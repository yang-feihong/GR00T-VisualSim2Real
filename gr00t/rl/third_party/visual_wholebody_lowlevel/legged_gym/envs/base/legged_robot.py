from __future__ import annotations
import math
import time
import torch
from typing import Tuple

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_matrix, quat_inv

from legged_gym.utils.isaaclab_math import quat_rotate_inverse, quat_mul, quat_apply, quat_from_euler_xyz, euler_from_quat, torch_rand_float, wrap_to_pi
from .legged_robot_config import LeggedRobotIsaacLabCfg
import numpy as np


def resolve_schedule_value(schedule, counter=0.0, default_end_iter=None):
    """Resolve the four-value linear schedules stored in ManipLoco metadata."""

    values = [float(value) for value in schedule]
    if len(values) != 4:
        raise ValueError(f"Expected [start, end, start_iter, end_iter], got {schedule!r}")
    start_value, end_value, start_iter, end_iter = values
    if end_iter <= start_iter and default_end_iter is not None:
        end_iter = float(default_end_iter)
    if end_iter <= start_iter:
        return end_value if counter >= end_iter else start_value
    alpha = min(max((float(counter) - start_iter) / (end_iter - start_iter), 0.0), 1.0)
    return start_value + alpha * (end_value - start_value)


_FAST_GAIT_UPDATE = None


def _fast_gait_update_impl(
    gait_indices: torch.Tensor,
    commands: torch.Tensor,
    phase_offsets: torch.Tensor,
    dt: float,
    min_frequency: float,
    max_frequency: float,
    lin_vel_ref: float,
    ang_vel_ref: float,
    ang_vel_weight: float,
    lin_vel_x_clip: float,
    ang_vel_yaw_clip: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cmd_x = commands[:, 0]
    cmd_y = commands[:, 1]
    cmd_yaw = commands[:, 2]
    lin_cmd_level = torch.sqrt(cmd_x * cmd_x + cmd_y * cmd_y) / lin_vel_ref
    yaw_cmd_level = torch.abs(cmd_yaw) / ang_vel_ref
    gait_level = torch.clamp(lin_cmd_level + ang_vel_weight * yaw_cmd_level, 0.0, 1.0)
    frequencies = min_frequency + (max_frequency - min_frequency) * gait_level
    walking_mask = (
        (torch.abs(cmd_x) > lin_vel_x_clip)
        | (torch.abs(cmd_y) > lin_vel_x_clip)
        | (torch.abs(cmd_yaw) > ang_vel_yaw_clip)
    )
    frequencies = torch.where(walking_mask, frequencies, torch.zeros_like(frequencies))
    next_gait_indices = torch.remainder(gait_indices + dt * frequencies, 1.0)
    next_gait_indices = torch.where(walking_mask, next_gait_indices, torch.zeros_like(next_gait_indices))
    foot_phase = torch.remainder(next_gait_indices.unsqueeze(1) + phase_offsets.unsqueeze(0), 1.0)
    clock_inputs = torch.sin(foot_phase * 6.283185307179586)
    return next_gait_indices, clock_inputs, frequencies


def _fast_gait_update(*args):
    global _FAST_GAIT_UPDATE
    if _FAST_GAIT_UPDATE is None:
        _FAST_GAIT_UPDATE = torch.jit.script(_fast_gait_update_impl)
    return _FAST_GAIT_UPDATE(*args)


class LeggedRobotIsaacLab(DirectRLEnv):
    """IsaacLab DirectRLEnv implementation of the legged robot simulator layer."""
    cfg: LeggedRobotIsaacLabCfg

    def __init__(self, cfg: LeggedRobotIsaacLabCfg, render_mode: str | None = None, **kwargs):
        self._rgb_camera_rigs = {}
        self._rgb_camera_requires_reset_sync = False
        self._rgb_camera_reset_sync_count = 0
        self._step_profile_enabled = bool(getattr(cfg, "profile_env_step", False))
        self._step_profile_in_step = False
        self._step_profile_wrap_warnings = set()
        self._reset_step_profile_stats()
        observes_goal_height_reference_mask = bool(
            cfg.goal_ee.sphere_center.mixed_height_reference
            or cfg.goal_ee.ranges.ee_goal_sampling_mode in ("body_importance", "body_importance_single_dir")
        )
        self.uses_goal_height_reference_mask = observes_goal_height_reference_mask
        self.observes_goal_height_reference_mask = observes_goal_height_reference_mask
        if not cfg.robot_usd_path:
            raise ValueError("cfg.robot_usd_path must point to the converted B2Z1 USD asset.")
        cfg.robot.spawn.usd_path = cfg.robot_usd_path
        self.body_names_to_idx = {}
        self.reset_init_ee_sphere = None
        super().__init__(cfg, render_mode, **kwargs)

        from legged_gym.utils.isaaclab_app import apply_renderer_feature_settings
        apply_renderer_feature_settings()
        self._rgb_camera_requires_reset_sync = bool(self._rgb_camera_rigs)

        self.num_actions = int(cfg.action_space)
        self.num_obs = int(cfg.observation_space)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.episode_length_s = float(cfg.episode_length_s)
        self.episode_length_steps = int(math.ceil(self.episode_length_s / self.step_dt))
        self.legacy_max_episode_length_s = self.episode_length_s
        self.legacy_max_episode_length = self.episode_length_steps

        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.arm_rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._never_done_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._never_timeout_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.extras = {}

        num_feet = len(self.cfg.foot_body_names)
        self.desired_contact_states = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.gait_frequencies = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                            requires_grad=False)
        self.gait_stance_durations = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_pair_phases = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.gait_pair_phases[:, 0] = 0.25
        self.gait_transition_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gait_transition_elapsed_s = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_transition_start_frequency = torch.zeros_like(self.gait_frequencies)
        self.gait_transition_target_frequency = torch.zeros_like(self.gait_frequencies)
        self.gait_transition_start_stance_duration = torch.ones_like(self.gait_stance_durations)
        self.gait_transition_target_stance_duration = torch.ones_like(self.gait_stance_durations)
        self.gait_frequency_commands = torch.full_like(self.commands[:, :3], float("nan"))
        self.gait_requested_frequencies = torch.zeros_like(self.gait_frequencies)
        self.velocity_command_changed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_velocity_command = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.foot_indices = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device)


        schedule_counter = float(self.cfg.commands.curriculum_playback_counter)
        schedule_total_iterations = self.cfg.commands.curriculum_playback_total_iterations
        lin_vel_x_min = resolve_schedule_value(
            self.cfg.commands.lin_vel_x_min_schedule,
            counter=schedule_counter,
            default_end_iter=schedule_total_iterations,
        )
        lin_vel_x_max = resolve_schedule_value(
            self.cfg.commands.lin_vel_x_max_schedule,
            counter=schedule_counter,
            default_end_iter=schedule_total_iterations,
        )
        ang_vel_yaw_max = resolve_schedule_value(
            self.cfg.commands.ang_vel_yaw_schedule,
            counter=schedule_counter,
            default_end_iter=schedule_total_iterations,
        )
        self.command_ranges = {
            "lin_vel_x": [lin_vel_x_min, lin_vel_x_max],
            "ang_vel_yaw": [-ang_vel_yaw_max, ang_vel_yaw_max],
        }

        ######################################################################
        scale = self.cfg.control.action_scale
        if isinstance(scale, (float, int)):
            self.action_scale_tensor = torch.full((self.num_actions,), float(scale), device=self.device)
        else:
            self.action_scale_tensor = torch.tensor(scale, dtype=torch.float32, device=self.device)

        assert self.action_scale_tensor.numel() == self.num_actions
        #########################################################################

        self._build_joint_maps()
        hip_body_names = [name.replace("_foot", "_hip") for name in self.cfg.asset.policy_foot_names]
        missing_hip_body_names = [name for name in hip_body_names if name not in self.body_names_to_idx]
        if missing_hip_body_names:
            raise RuntimeError(f"No hip body found for policy feet: {missing_hip_body_names}")
        self.hip_body_indices = torch.tensor(
            [self.body_names_to_idx[name] for name in hip_body_names],
            dtype=torch.long,
            device=self.device,
        )
        self._init_policy_compat_buffers()
        self._init_reward_compat()
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        self._install_internal_step_profile_wrappers()

    def reset(self, seed=None, options=None):
        observations, extras = super().reset(seed=seed, options=options)
        self.synchronize_rgb_cameras_after_reset()
        return observations, extras

    def synchronize_rgb_cameras_after_reset(self, env_id=0):
        rigs = self._rgb_camera_rigs
        if not rigs:
            self._rgb_camera_requires_reset_sync = False
            return
        if not self._rgb_camera_requires_reset_sync:
            return
        if not self.body_names_to_idx:
            raise RuntimeError("RGB camera reset synchronization requires initialized robot body maps")

        self.sim.forward()
        self._update_rgb_camera_poses_for_env(env_id=int(env_id))
        self.sim.render()
        self.sim.render()

        updated_sensors = set()
        for rig in rigs.values():
            cameras = rig.get("cameras", {})
            sensors = [cameras.get("tiled")] if rig.get("backend") == "tiled" else list(cameras.values())
            for sensor in sensors:
                if sensor is None or id(sensor) in updated_sensors:
                    continue
                sensor.update(0.0, force_recompute=True)
                updated_sensors.add(id(sensor))

        self._rgb_camera_requires_reset_sync = False
        self._rgb_camera_reset_sync_count += 1

    def _reset_step_profile_stats(self):
        self._step_profile_step_count = 0
        self._step_profile_in_step = False
        self._step_profile_sums = {}
        self._step_profile_maxes = {}
        self._step_profile_count_sums = {}
        self._step_profile_count_maxes = {}
        self._step_profile_current_sums = {}
        self._step_profile_current_counts = {}

    def _profile_step_start(self):
        if self._step_profile_enabled and not self._step_profile_in_step:
            self._install_internal_step_profile_wrappers()
            self._step_profile_step_count += 1
            self._step_profile_in_step = True
            self._step_profile_current_sums = {}
            self._step_profile_current_counts = {}

    def _profile_step_end(self):
        if not self._step_profile_enabled or not self._step_profile_in_step:
            return
        for name, total in self._step_profile_current_sums.items():
            self._step_profile_sums[name] = self._step_profile_sums.get(name, 0.0) + total
            self._step_profile_maxes[name] = max(self._step_profile_maxes.get(name, 0.0), total)
        for name, total in self._step_profile_current_counts.items():
            self._step_profile_count_sums[name] = self._step_profile_count_sums.get(name, 0) + total
            self._step_profile_count_maxes[name] = max(self._step_profile_count_maxes.get(name, 0), total)
        self._step_profile_in_step = False

    def _profile_record(self, name: str, duration_s: float):
        if not self._step_profile_enabled or not self._step_profile_in_step:
            return
        duration_s = float(duration_s)
        self._step_profile_current_sums[name] = self._step_profile_current_sums.get(name, 0.0) + duration_s

    def _profile_count(self, name: str, count: int = 1):
        if not self._step_profile_enabled or not self._step_profile_in_step:
            return
        count = int(count)
        self._step_profile_current_counts[name] = self._step_profile_current_counts.get(name, 0) + count

    def pop_step_profile_stats(self):
        if not self._step_profile_enabled:
            return {}
        self._profile_step_end()
        step_count = max(int(self._step_profile_step_count), 1)
        stats = {
            name: {
                "avg_ms": total * 1000.0 / step_count,
                "max_ms": self._step_profile_maxes.get(name, 0.0) * 1000.0,
            }
            for name, total in self._step_profile_sums.items()
        }
        stats["_counts"] = {
            name: {
                "avg": total / step_count,
                "max": self._step_profile_count_maxes.get(name, 0),
            }
            for name, total in self._step_profile_count_sums.items()
        }
        stats["_steps"] = step_count
        self._reset_step_profile_stats()
        return stats

    def _wrap_profile_method(self, owner, method_name: str, timing_key: str, count_key: str | None = None):
        if owner is None or not hasattr(owner, method_name):
            return
        timing_keys = tuple(timing_key) if isinstance(timing_key, (list, tuple, set)) else (timing_key,)
        try:
            original_method = getattr(owner, method_name)
        except Exception:
            return
        if getattr(original_method, "_vw_profile_wrapper", False):
            return

        def timed_method(*args, **kwargs):
            if not self._step_profile_enabled or not self._step_profile_in_step:
                return original_method(*args, **kwargs)
            start_time = time.perf_counter()
            try:
                return original_method(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start_time
                for key in timing_keys:
                    self._profile_record(key, elapsed)
                if count_key is not None:
                    self._profile_count(count_key)

        timed_method._vw_profile_wrapper = True
        try:
            setattr(owner, method_name, timed_method)
        except Exception as exc:
            warning_key = f"{type(owner).__name__}.{method_name}"
            if warning_key not in self._step_profile_wrap_warnings:
                self._step_profile_wrap_warnings.add(warning_key)
                print(f"[env_profile] unable to wrap {warning_key}: {exc}")

    @staticmethod
    def _profile_key_token(name) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_").lower() or "unnamed"

    def _wrap_scene_entity_write_methods(self):
        scene = getattr(self, "scene", None)
        if scene is None:
            return
        groups = (
            ("articulations", "scene_articulations_write", "scene_articulation"),
            ("rigid_objects", "scene_rigid_objects_write", "scene_rigid_object"),
            ("rigid_object_collections", "scene_rigid_object_collections_write", "scene_rigid_object_collection"),
            ("deformable_objects", "scene_deformable_objects_write", "scene_deformable_object"),
            ("sensors", "scene_sensors_write", "scene_sensor"),
            ("surface_grippers", "scene_surface_grippers_write", "scene_surface_gripper"),
        )
        for public_name, aggregate_key, detail_prefix in groups:
            entities = getattr(scene, public_name, None)
            if entities is None:
                entities = getattr(scene, f"_{public_name}", None)
            if not hasattr(entities, "items"):
                continue
            for entity_name, entity in list(entities.items()):
                if entity is None or not hasattr(entity, "write_data_to_sim"):
                    continue
                if public_name == "articulations" and entity is getattr(self, "robot", None):
                    continue
                token = self._profile_key_token(entity_name)
                detail_key = f"{detail_prefix}_{token}_write"
                self._wrap_profile_method(entity, "write_data_to_sim", (detail_key, aggregate_key))

    def _install_internal_step_profile_wrappers(self):
        if not self._step_profile_enabled:
            return

        self._wrap_profile_method(getattr(self, "scene", None), "write_data_to_sim", "write_data_to_sim")
        self._wrap_profile_method(getattr(self, "sim", None), "step", "physics_elapsed", "sim_steps")
        self._wrap_profile_method(getattr(self, "sim", None), "render", "render_elapsed", "render_calls")
        self._wrap_profile_method(getattr(self, "scene", None), "update", "scene_update_elapsed", "scene_updates")

        robot = getattr(self, "robot", None)
        self._wrap_profile_method(robot, "write_data_to_sim", "robot_write_data_to_sim")
        self._wrap_profile_method(robot, "_apply_actuator_model", "robot_apply_actuator_model")

        root_physx_view = getattr(robot, "root_physx_view", None)
        self._wrap_profile_method(
            root_physx_view,
            "set_dof_actuation_forces",
            "robot_set_dof_actuation_forces",
        )
        self._wrap_profile_method(
            root_physx_view,
            "set_dof_position_targets",
            "robot_set_dof_position_targets",
        )
        self._wrap_profile_method(
            root_physx_view,
            "set_dof_velocity_targets",
            "robot_set_dof_velocity_targets",
        )
        self._wrap_scene_entity_write_methods()

    def _setup_scene(self):
        self._setup_imported_scene()

        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        if self.cfg.enable_doorman_scene:
            if self.cfg.doorman_door is None:
                raise RuntimeError("cfg.doorman_door must be set when cfg.enable_doorman_scene is True.")
            self.door = Articulation(self.cfg.doorman_door)
            self.scene.articulations["door"] = self.door
            dome_light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.98, 0.95, 0.88))
            self.scene.extras["dome_light"] = dome_light_cfg.func("/World/DomeLight", dome_light_cfg)
        else:
            self.door = None

        if bool(self.cfg.enable_contact_sensor):
            self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self.contact_sensor
        else:
            self.contact_sensor = None
        self._setup_rgb_cameras()

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._init_terrain_height_samples()
        self.scene.clone_environments(copy_from_source=False)
        global_prim_paths = [self.cfg.terrain.prim_path]
        if self.cfg.scene_usd_path:
            global_prim_paths.append(self.cfg.scene_prim_path)
        self.scene.filter_collisions(global_prim_paths=global_prim_paths)
        self.sim.set_camera_view((4.0, 0.0, 2.0), (0.0, 0.0, 0.5))

    def _setup_imported_scene(self):
        scene_usd_path = str(self.cfg.scene_usd_path or "").strip()
        if not scene_usd_path:
            return
        scene_position = list(self.cfg.scene_position or [])
        if len(scene_position) != 3:
            raise ValueError(
                "cfg.scene_position must contain exactly three values when cfg.scene_usd_path is set."
            )
        scene_prim_path = str(self.cfg.scene_prim_path or "").strip()
        if not scene_prim_path:
            raise ValueError("cfg.scene_prim_path must be set when cfg.scene_usd_path is set.")

        scene_cfg = sim_utils.UsdFileCfg(usd_path=scene_usd_path)
        scene_cfg.func(
            scene_prim_path,
            scene_cfg,
            translation=tuple(float(value) for value in scene_position),
        )
        print(f"[scene] imported USD scene: {scene_usd_path} at {scene_position}")

    @staticmethod
    def _euler_xyz_to_quat_tuple(roll: float, pitch: float, yaw: float):
        half_roll = 0.5 * float(roll)
        half_pitch = 0.5 * float(pitch)
        half_yaw = 0.5 * float(yaw)
        cr, sr = math.cos(half_roll), math.sin(half_roll)
        cp, sp = math.cos(half_pitch), math.sin(half_pitch)
        cy, sy = math.cos(half_yaw), math.sin(half_yaw)
        return (
            float(cr * cp * cy + sr * sp * sy),
            float(sr * cp * cy - cr * sp * sy),
            float(cr * sp * cy + sr * cp * sy),
            float(cr * cp * sy - sr * sp * cy),
        )


    @staticmethod
    def _quat_rotate_np_wxyz(quat_wxyz, vec_xyz):
        # Rotate a 3D vector by a wxyz quaternion.
        quat = np.asarray(quat_wxyz, dtype=np.float64)
        vec = np.asarray(vec_xyz, dtype=np.float64)

        norm = np.linalg.norm(quat)
        if norm <= 1.0e-12:
            raise ValueError(f"Invalid zero quaternion for RGB camera offset rotation: {quat_wxyz}")
        quat = quat / norm

        w = quat[0]
        q_vec = quat[1:4]

        # q * v * q^-1, written in vector form.
        return vec + 2.0 * w * np.cross(q_vec, vec) + 2.0 * np.cross(
            q_vec, np.cross(q_vec, vec)
        )

    @staticmethod
    def _quat_mul_np_wxyz(q_wxyz, r_wxyz):
        q = np.asarray(q_wxyz, dtype=np.float64)
        r = np.asarray(r_wxyz, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        r_norm = np.linalg.norm(r)
        if q_norm > 1.0e-12:
            q = q / q_norm
        if r_norm > 1.0e-12:
            r = r / r_norm
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = r
        return np.asarray(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _spawn_rgb_camera_prim(prim_path: str, spawn_cfg, position, orientation):
        if not hasattr(spawn_cfg, "func"):
            raise RuntimeError("RGB camera spawn config does not expose a spawn function.")
        kwargs = {
            "translation": tuple(float(v) for v in position),
            "orientation": tuple(float(v) for v in orientation),
        }
        try:
            spawn_cfg.func(prim_path, spawn_cfg, **kwargs)
        except TypeError:
            spawn_cfg.func(prim_path, spawn_cfg)

    def _rgb_camera_spec_to_eye_defs(self, spec, horizontal_aperture: float, sensor_prefix: str = ""):
        mount_name = str(spec["name"])
        camera_mode = str(spec["mode"])
        body_name = str(spec["body_name"])
        base_pos = np.asarray(spec["local_position"], dtype=np.float64)
        base_rot = self._euler_xyz_to_quat_tuple(*spec["local_rotation"])
        projection_type = str(spec.get("projection_type", "pinhole"))
        if projection_type == "pinhole":
            hfov_rad = np.deg2rad(float(spec["horizontal_fov"]))
            focal_length = 0.5 * horizontal_aperture / max(float(np.tan(0.5 * hfov_rad)), 1.0e-6)
        else:
            focal_length = None

        if camera_mode == "mono":
            offsets = [("mono", np.asarray(spec["mono_offset"], dtype=np.float64))]
        elif camera_mode == "stereo":
            offsets = [
                ("left", np.asarray(spec["left_offset"], dtype=np.float64)),
                ("right", np.asarray(spec["right_offset"], dtype=np.float64)),
            ]
        else:
            raise ValueError(f"Unsupported RGB camera mode: {camera_mode}")

        eye_defs = []
        for eye_name, eye_offset in offsets:
            sensor_name = f"{sensor_prefix}{mount_name}_{eye_name}"
            rotated_eye_offset = self._quat_rotate_np_wxyz(base_rot, eye_offset)
            eye_pos = base_pos + rotated_eye_offset
            eye_defs.append(
                {
                    "eye_name": eye_name,
                    "sensor_name": sensor_name,
                    "local_pos": tuple(float(v) for v in eye_pos.tolist()),
                    "local_quat": base_rot,
                }
            )
        return mount_name, camera_mode, body_name, focal_length, eye_defs

    @staticmethod
    def _rgb_camera_spawn_config(spec, horizontal_aperture: float, focal_length):
        projection_type = str(spec.get("projection_type", "pinhole"))
        if projection_type == "pinhole":
            return sim_utils.PinholeCameraCfg(
                focal_length=float(focal_length),
                horizontal_aperture=float(horizontal_aperture),
                clipping_range=(0.01, 20.0),
            )

        if projection_type == "opencvFisheye":
            return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=[
                    float(spec["fx"]),
                    float(spec.get("skew", 0.0)),
                    float(spec["cx"]),
                    0.0,
                    float(spec["fy"]),
                    float(spec["cy"]),
                    0.0,
                    0.0,
                    1.0,
                ],
                width=int(spec["width"]),
                height=int(spec["height"]),
                clipping_range=(0.01, 20.0),
            )

        return sim_utils.FisheyeCameraCfg(
            projection_type=projection_type,
            clipping_range=(0.01, 20.0),
            focal_length=float(spec["focal_length_mm"]),
            fisheye_nominal_width=float(spec["fisheye_nominal_width"]),
            fisheye_nominal_height=float(spec["fisheye_nominal_height"]),
            fisheye_optical_centre_x=float(spec["fisheye_optical_centre_x"]),
            fisheye_optical_centre_y=float(spec["fisheye_optical_centre_y"]),
            fisheye_max_fov=float(spec["fisheye_max_fov"]),
            fisheye_polynomial_a=float(spec["fisheye_polynomial_a"]),
            fisheye_polynomial_b=float(spec["fisheye_polynomial_b"]),
            fisheye_polynomial_c=float(spec["fisheye_polynomial_c"]),
            fisheye_polynomial_d=float(spec["fisheye_polynomial_d"]),
            fisheye_polynomial_e=float(spec["fisheye_polynomial_e"]),
            fisheye_polynomial_f=float(spec["fisheye_polynomial_f"]),
        )

    @staticmethod
    def _apply_rgb_camera_lens_calibration(camera_prim_path: str, spec):
        if str(spec.get("projection_type", "pinhole")) != "opencvFisheye":
            return

        from pxr import Gf

        camera_prim = sim_utils.get_current_stage().GetPrimAtPath(camera_prim_path)
        if not camera_prim.IsValid():
            raise RuntimeError(f"RGB camera prim does not exist: {camera_prim_path}")
        camera_prim.ApplyAPI("OmniLensDistortionOpenCvFisheyeAPI")
        attributes = {
            "omni:lensdistortion:model": "opencvFisheye",
            "omni:lensdistortion:opencvFisheye:imageSize": Gf.Vec2i(int(spec["width"]), int(spec["height"])),
            "omni:lensdistortion:opencvFisheye:cx": float(spec["cx"]),
            "omni:lensdistortion:opencvFisheye:cy": float(spec["cy"]),
            "omni:lensdistortion:opencvFisheye:fx": float(spec["fx"]),
            "omni:lensdistortion:opencvFisheye:fy": float(spec["fy"]),
        }
        for index, value in enumerate(spec["distortion_coefficients"], start=1):
            attributes[f"omni:lensdistortion:opencvFisheye:k{index}"] = float(value)
        for attribute_name, value in attributes.items():
            attribute = camera_prim.GetAttribute(attribute_name)
            if not attribute.IsValid():
                raise RuntimeError(
                    f"Isaac Sim did not create OpenCV fisheye attribute {attribute_name!r} "
                    f"for camera {camera_prim_path!r}."
                )
            attribute.Set(value)
        skew = float(spec.get("skew", 0.0))
        if abs(skew) > 1.0e-9:
            print(
                f"[rgb_camera][warn] OpenCV fisheye skew={skew:g} px is retained as calibration metadata "
                "but is not supported by the Isaac Sim lens-distortion schema."
            )

    def _setup_tiled_rgb_cameras(self, specs, tiled_camera_cls, tiled_camera_cfg_cls, horizontal_aperture: float):
        all_eye_defs = []
        rig_defs = []
        common_key = None
        for spec in specs:
            if str(spec.get("projection_type", "pinhole")) != "pinhole":
                raise ValueError(
                    "rgb_camera_backend='tiled' does not support the calibrated fisheye configuration; "
                    "use --rgb_camera_backend camera (or the default auto backend)."
                )
            mount_name, camera_mode, body_name, focal_length, eye_defs = self._rgb_camera_spec_to_eye_defs(
                spec,
                horizontal_aperture,
            )
            key = (
                int(spec["height"]),
                int(spec["width"]),
                round(float(focal_length), 8),
                round(float(horizontal_aperture), 8),
            )
            if common_key is None:
                common_key = key
            elif key != common_key:
                raise ValueError(
                    "rgb_camera_backend='tiled' requires all enabled RGB cameras in the config to share "
                    "height, width, and horizontal_fov. Use --rgb_camera_backend camera for mixed camera intrinsics."
                )
            rig_defs.append((mount_name, camera_mode, body_name, eye_defs, spec.get("valid_circle_mask")))
            all_eye_defs.extend(eye_defs)

        if not all_eye_defs:
            return
        for index, eye_def in enumerate(all_eye_defs):
            eye_def["sensor_name"] = f"tiled_{index:03d}_{eye_def['sensor_name']}"

        camera_spawn_cfg = sim_utils.PinholeCameraCfg(
            focal_length=float(common_key[2]),
            horizontal_aperture=float(horizontal_aperture),
            clipping_range=(0.01, 20.0),
        )
        for eye_def in all_eye_defs:
            self._spawn_rgb_camera_prim(
                f"/World/RGBCameras/{eye_def['sensor_name']}/Camera",
                camera_spawn_cfg,
                eye_def["local_pos"],
                eye_def["local_quat"],
            )

        tiled_camera_cfg = tiled_camera_cfg_cls(
            prim_path="/World/RGBCameras/tiled_.*/Camera",
            update_period=float(self.cfg.rgb_camera_update_period_s),
            height=int(common_key[0]),
            width=int(common_key[1]),
            data_types=["rgb"],
            update_latest_camera_pose=True,
            spawn=None,
            offset=tiled_camera_cfg_cls.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
        )
        tiled_camera = tiled_camera_cls(tiled_camera_cfg)
        self.scene.sensors["rgb_cameras_tiled"] = tiled_camera

        sensor_index = 0
        for mount_name, camera_mode, body_name, eye_defs, valid_circle_mask in rig_defs:
            eyes = []
            for eye_def in eye_defs:
                eyes.append(
                    {
                        "name": eye_def["eye_name"],
                        "local_pos": eye_def["local_pos"],
                        "local_quat": eye_def["local_quat"],
                        "local_pos_np": np.asarray(eye_def["local_pos"], dtype=np.float32),
                        "local_quat_np": np.asarray(eye_def["local_quat"], dtype=np.float32),
                        "sensor_index": sensor_index,
                    }
                )
                sensor_index += 1
            self._rgb_camera_rigs[mount_name] = {
                "mode": camera_mode,
                "body_name": body_name,
                "backend": "tiled",
                "pose_update": True,
                "valid_circle_mask": valid_circle_mask,
                "cameras": {"tiled": tiled_camera},
                "eyes": eyes,
            }
            print(
                f"[rgb_camera] registered {camera_mode} rig {mount_name!r} "
                f"on body {body_name!r} backend=tiled sensors={len(eyes)} shared_tiled=True"
            )


    def _setup_rgb_cameras(self):
        specs = list(self.cfg.rgb_camera_specs or [])
        self._rgb_camera_rigs = {}
        if not specs:
            print("[rgb_camera] no RGB camera sensors configured")
            return

        camera_backend = str(getattr(self.cfg, "rgb_camera_backend", "auto") or "auto").strip().lower()
        if camera_backend not in ("auto", "camera", "tiled"):
            raise ValueError(f"Unsupported rgb_camera_backend={camera_backend!r}; expected auto/camera/tiled.")

        if camera_backend == "tiled":
            try:
                from isaaclab.sensors import TiledCamera, TiledCameraCfg
            except Exception as exc:
                raise RuntimeError("IsaacLab TiledCamera sensor support is required for rgb_camera_backend='tiled'.") from exc
        else:
            try:
                from isaaclab.sensors import Camera, CameraCfg
            except Exception as exc:
                raise RuntimeError("IsaacLab Camera sensor support is required for RGB camera rigs.") from exc

        horizontal_aperture = 20.955
        if camera_backend == "tiled":
            self._setup_tiled_rgb_cameras(specs, TiledCamera, TiledCameraCfg, horizontal_aperture)
            return

        for spec in specs:
            mount_name, camera_mode, body_name, focal_length, eye_defs = self._rgb_camera_spec_to_eye_defs(
                spec,
                horizontal_aperture,
            )

            camera_spawn_cfg = self._rgb_camera_spawn_config(spec, horizontal_aperture, focal_length)

            rig_cameras = {}
            for eye_def in eye_defs:
                camera_cfg = CameraCfg(
                    prim_path=f"/World/RGBCameras/{eye_def['sensor_name']}",
                    update_period=float(self.cfg.rgb_camera_update_period_s),
                    height=int(spec["height"]),
                    width=int(spec["width"]),
                    data_types=["rgb"],
                    update_latest_camera_pose=True,
                    spawn=camera_spawn_cfg,
                    offset=CameraCfg.OffsetCfg(
                        pos=eye_def["local_pos"],
                        rot=eye_def["local_quat"],
                        convention="world",
                    ),
                )
                camera = Camera(camera_cfg)
                self._apply_rgb_camera_lens_calibration(camera_cfg.prim_path, spec)
                diagnostic = spec.get("door_diagnostic")
                if diagnostic:
                    light_cfg = sim_utils.SphereLightCfg(
                        intensity=float(diagnostic["fill_light_intensity"]),
                        radius=float(diagnostic["fill_light_radius_m"]),
                        color=(1.0, 0.96, 0.90),
                    )
                    light_cfg.func(
                        f"{camera_cfg.prim_path}/FillLight",
                        light_cfg,
                        translation=(
                            0.0,
                            float(diagnostic["fill_light_right_offset_m"]),
                            float(diagnostic["fill_light_up_offset_m"]),
                        ),
                    )
                self.scene.sensors[eye_def["sensor_name"]] = camera
                rig_cameras[eye_def["eye_name"]] = camera
            resolved_backend = "camera"

            self._rgb_camera_rigs[mount_name] = {
                "mode": camera_mode,
                "body_name": body_name,
                "backend": resolved_backend,
                "pose_update": True,
                "door_diagnostic": spec.get("door_diagnostic"),
                "valid_circle_mask": spec.get("valid_circle_mask"),
                "cameras": rig_cameras,
                "eyes": [
                    {
                        "name": eye_def["eye_name"],
                        "local_pos": eye_def["local_pos"],
                        "local_quat": eye_def["local_quat"],
                        "local_pos_np": np.asarray(eye_def["local_pos"], dtype=np.float32),
                        "local_quat_np": np.asarray(eye_def["local_quat"], dtype=np.float32),
                        "sensor_index": index,
                    }
                    for index, eye_def in enumerate(eye_defs)
                ],
            }
            print(
                f"[rgb_camera] registered {camera_mode} rig {mount_name!r} "
                f"on body {body_name!r} backend={resolved_backend} sensors={len(eye_defs)}"
            )


    def _get_rgb_camera_body_state_w(self, body_name: str, env_ids: torch.Tensor):
        if not self.body_names_to_idx:
            raise RuntimeError("RGB camera body pose update requires body_names_to_idx; _build_joint_maps() has not run.")
        if body_name not in self.body_names_to_idx:
            raise RuntimeError(f"RGB camera body {body_name!r} is not in robot body_names: {self.body_names}")

        body_idx = int(self.body_names_to_idx[body_name])
        data = self.robot.data

        if hasattr(data, "body_pos_w") and hasattr(data, "body_quat_w"):
            return data.body_pos_w[env_ids, body_idx], data.body_quat_w[env_ids, body_idx]

        if body_idx == 0 and hasattr(data, "root_pos_w") and hasattr(data, "root_quat_w"):
            return data.root_pos_w[env_ids], data.root_quat_w[env_ids]

        raise RuntimeError(
            "RGB camera body pose update could not find ArticulationData.body_pos_w/body_quat_w. "
            "This IsaacLab version does not expose per-body poses through the expected API."
        )

    def _fill_rgb_camera_pose_arrays(self, rig, body_pos_w_np, body_quat_w_np):
        eyes = list(rig.get("eyes", []))
        num_eyes = len(eyes)
        positions = rig.get("_pose_positions_np")
        orientations = rig.get("_pose_orientations_np")
        if positions is None or positions.shape != (num_eyes, 3) or positions.dtype != np.float32:
            positions = np.zeros((num_eyes, 3), dtype=np.float32)
            rig["_pose_positions_np"] = positions
        if orientations is None or orientations.shape != (num_eyes, 4) or orientations.dtype != np.float32:
            orientations = np.zeros((num_eyes, 4), dtype=np.float32)
            rig["_pose_orientations_np"] = orientations

        for index, eye in enumerate(eyes):
            local_pos = eye.get("local_pos_np")
            if local_pos is None:
                local_pos = np.asarray(eye["local_pos"], dtype=np.float32)
                eye["local_pos_np"] = local_pos
            local_quat = eye.get("local_quat_np")
            if local_quat is None:
                local_quat = np.asarray(eye["local_quat"], dtype=np.float32)
                eye["local_quat_np"] = local_quat
            positions[index, :] = body_pos_w_np + self._quat_rotate_np_wxyz(body_quat_w_np, local_pos)
            orientations[index, :] = self._quat_mul_np_wxyz(body_quat_w_np, local_quat)
        return positions, orientations

    def _set_rgb_camera_world_poses_from_arrays(self, rig, positions, orientations):
        if rig.get("backend") == "tiled":
            sensor_ids = [int(eye["sensor_index"]) for eye in rig.get("eyes", [])]
            if sensor_ids:
                rig["cameras"]["tiled"].set_world_poses(
                    positions=positions,
                    orientations=orientations,
                    env_ids=sensor_ids,
                    convention="world",
                )
            return

        for index, eye in enumerate(rig.get("eyes", [])):
            rig["cameras"][eye["name"]].set_world_poses(
                positions=positions[index : index + 1],
                orientations=orientations[index : index + 1],
                env_ids=[0],
                convention="world",
            )

    def _update_rgb_camera_world_poses(self, rig, env_id=0):
        if rig.get("door_diagnostic"):
            self._update_door_rgb_camera_world_pose(rig, env_id)
            return
        env_id = int(env_id)
        env_ids_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
        body_pos_w, body_quat_w = self._get_rgb_camera_body_state_w(str(rig["body_name"]), env_ids_t)
        body_pos_w_np = body_pos_w[0].detach().cpu().numpy()
        body_quat_w_np = body_quat_w[0].detach().cpu().numpy()
        positions, orientations = self._fill_rgb_camera_pose_arrays(rig, body_pos_w_np, body_quat_w_np)
        self._set_rgb_camera_world_poses_from_arrays(rig, positions, orientations)

    def _initialize_door_rgb_camera_local_pose(self, rig, env_id):
        if self.door is None:
            raise RuntimeError("Door diagnostic RGB cameras require the doorman door articulation.")
        panel_ids, _ = self.door.find_bodies("door_panel", preserve_order=True)
        handle_ids, _ = self.door.find_bodies("door_handle", preserve_order=True)
        if len(panel_ids) != 1 or len(handle_ids) != 1:
            raise RuntimeError(
                "Door diagnostic RGB camera expected exactly one door_panel and door_handle body, "
                f"got {panel_ids} and {handle_ids}."
            )

        panel_id = int(panel_ids[0])
        handle_id = int(handle_ids[0])
        panel_pos = self.door.data.body_pos_w[env_id, panel_id]
        panel_quat = self.door.data.body_quat_w[env_id, panel_id]
        handle_pos = self.door.data.body_pos_w[env_id, handle_id]
        handle_quat = self.door.data.body_quat_w[env_id, handle_id]
        diagnostic = rig["door_diagnostic"]

        offset = torch.tensor(
            diagnostic["target_offset_leaf_m"], device=self.device, dtype=panel_pos.dtype
        )
        target = handle_pos + quat_apply(panel_quat.unsqueeze(0), offset.unsqueeze(0))[0]
        axis_local = torch.tensor(
            diagnostic["view_axis_local"], device=self.device, dtype=panel_pos.dtype
        )
        axis_quat = handle_quat if diagnostic["view_axis_reference"] == "handle_body" else panel_quat
        axis = quat_apply(axis_quat.unsqueeze(0), axis_local.unsqueeze(0))[0]
        axis = torch.nn.functional.normalize(axis, dim=0) * float(
            diagnostic["view_axis_direction_sign"]
        )
        camera_pos = target + float(diagnostic["axis_distance_m"]) * axis
        forward = -axis
        up_hint = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=panel_pos.dtype)
        if torch.abs(torch.dot(up_hint, forward)) > 0.95:
            up_hint = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=panel_pos.dtype)
        right = torch.nn.functional.normalize(torch.cross(up_hint, forward, dim=0), dim=0)
        up = torch.cross(forward, right, dim=0)
        camera_quat = quat_from_matrix(torch.stack((forward, right, up), dim=-1).unsqueeze(0))[0]
        panel_quat_inverse = quat_inv(panel_quat.unsqueeze(0))[0]
        local_pos = quat_apply(
            panel_quat_inverse.unsqueeze(0), (camera_pos - panel_pos).unsqueeze(0)
        )[0]
        local_quat = quat_mul(panel_quat_inverse.unsqueeze(0), camera_quat.unsqueeze(0))[0]
        rig["_door_local_pose"] = (local_pos, local_quat, panel_id)

    def _update_door_rgb_camera_world_pose(self, rig, env_id):
        env_id = int(env_id)
        if rig.get("_door_local_pose") is None:
            self._initialize_door_rgb_camera_local_pose(rig, env_id)
        local_pos, local_quat, panel_id = rig["_door_local_pose"]
        panel_pos = self.door.data.body_pos_w[env_id, panel_id]
        panel_quat = self.door.data.body_quat_w[env_id, panel_id]
        camera_pos = panel_pos + quat_apply(panel_quat.unsqueeze(0), local_pos.unsqueeze(0))[0]
        camera_quat = quat_mul(panel_quat.unsqueeze(0), local_quat.unsqueeze(0))[0]
        positions = camera_pos.detach().cpu().numpy().reshape(1, 3)
        orientations = camera_quat.detach().cpu().numpy().reshape(1, 4)
        self._set_rgb_camera_world_poses_from_arrays(rig, positions, orientations)

    def _update_rgb_camera_poses_for_env(self, env_id=0):
        rigs = [
            rig
            for rig in getattr(self, "_rgb_camera_rigs", {}).values()
            if bool(rig.get("pose_update", True))
        ]
        if not rigs:
            return
        env_id = int(env_id)

        read_start = time.perf_counter()
        body_names = []
        body_indices = []
        for rig in rigs:
            if rig.get("door_diagnostic"):
                continue
            body_name = str(rig["body_name"])
            if body_name in body_names:
                continue
            if body_name not in self.body_names_to_idx:
                raise RuntimeError(f"RGB camera body {body_name!r} is not in robot body_names: {self.body_names}")
            body_names.append(body_name)
            body_indices.append(int(self.body_names_to_idx[body_name]))

        data = self.robot.data
        if hasattr(data, "body_pos_w") and hasattr(data, "body_quat_w"):
            body_pos_np = data.body_pos_w[env_id, body_indices].detach().cpu().numpy()
            body_quat_np = data.body_quat_w[env_id, body_indices].detach().cpu().numpy()
        else:
            env_ids_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
            body_pos_list = []
            body_quat_list = []
            for body_name in body_names:
                body_pos_w, body_quat_w = self._get_rgb_camera_body_state_w(body_name, env_ids_t)
                body_pos_list.append(body_pos_w[0].detach().cpu().numpy())
                body_quat_list.append(body_quat_w[0].detach().cpu().numpy())
            body_pos_np = np.asarray(body_pos_list, dtype=np.float32)
            body_quat_np = np.asarray(body_quat_list, dtype=np.float32)
        body_name_to_row = {body_name: index for index, body_name in enumerate(body_names)}
        self._profile_record("rgb_pose_read", time.perf_counter() - read_start)

        math_total = 0.0
        set_total = 0.0
        for rig in rigs:
            if rig.get("door_diagnostic"):
                set_start = time.perf_counter()
                self._update_door_rgb_camera_world_pose(rig, env_id)
                set_total += time.perf_counter() - set_start
                continue
            body_row = body_name_to_row[str(rig["body_name"])]
            math_start = time.perf_counter()
            positions, orientations = self._fill_rgb_camera_pose_arrays(
                rig,
                body_pos_np[body_row],
                body_quat_np[body_row],
            )
            math_total += time.perf_counter() - math_start

            set_start = time.perf_counter()
            self._set_rgb_camera_world_poses_from_arrays(rig, positions, orientations)
            set_total += time.perf_counter() - set_start
        self._profile_record("rgb_pose_math", math_total)
        self._profile_record("rgb_pose_set", set_total)

    def _should_update_rgb_camera_poses_this_step(self):
        if not any(bool(rig.get("pose_update", True)) for rig in getattr(self, "_rgb_camera_rigs", {}).values()):
            return False
        update_period = float(getattr(self.cfg, "rgb_camera_update_period_s", 0.0) or 0.0)
        interval = 1 if update_period <= 0.0 else max(1, int(round(update_period / float(self.dt))))
        return self.global_steps <= 1 or self.global_steps % interval == 0

    def _build_joint_maps(self):
        self.dof_names = list(self.robot.data.joint_names)
        self.body_names = list(self.robot.data.body_names)
        self.dof_names_to_idx = {name: i for i, name in enumerate(self.dof_names)}
        self.body_names_to_idx = {name: i for i, name in enumerate(self.body_names)}
        self.num_dofs = len(self.dof_names)
        self.num_bodies = len(self.body_names)

        self.policy_joint_ids = torch.tensor(
            [self.dof_names_to_idx[name] for name in self.cfg.policy_joint_names if name in self.dof_names_to_idx],
            dtype=torch.long, device=self.device)
        if self.policy_joint_ids.numel() != len(self.cfg.policy_joint_names):
            missing = [n for n in self.cfg.policy_joint_names if n not in self.dof_names_to_idx]
            raise RuntimeError(f"Missing policy joints in USD articulation: {missing}. Available={self.dof_names}")

        self.feet_indices = torch.tensor(
            [self.body_names_to_idx[name] for name in self.cfg.foot_body_names if name in self.body_names_to_idx],
            dtype=torch.long, device=self.device)
        self.termination_contact_indices = torch.tensor(
            [self.body_names_to_idx[name] for name in self.cfg.terminate_body_names if name in self.body_names_to_idx],
            dtype=torch.long, device=self.device)
        penalized_contact_patterns = list(self.cfg.asset.penalize_contacts_on or [])
        self.penalized_contact_indices = torch.tensor(
            [
                i for i, name in enumerate(self.body_names)
                if any(pattern in name for pattern in penalized_contact_patterns)
            ],
            dtype=torch.long,
            device=self.device,
        )

        self.default_dof_pos = self.robot.data.default_joint_pos[0].clone()
        for name, value in self.cfg.default_joint_angles.items():
            if name in self.dof_names_to_idx:
                self.default_dof_pos[self.dof_names_to_idx[name]] = float(value)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)

        joint_pos_limits = getattr(self.robot.data, "soft_joint_pos_limits", None)
        if joint_pos_limits is None:
            joint_pos_limits = getattr(self.robot.data, "joint_pos_limits", None)
        if joint_pos_limits is None:
            joint_pos_limits = torch.stack(
                [
                    torch.full((self.num_dofs,), -torch.inf, device=self.device),
                    torch.full((self.num_dofs,), torch.inf, device=self.device),
                ],
                dim=-1,
            )
        elif joint_pos_limits.dim() == 3:
            joint_pos_limits = joint_pos_limits[0]
        self.dof_pos_limits = joint_pos_limits.to(device=self.device, dtype=torch.float32)
        joint_effort_limits = self.robot.data.joint_effort_limits
        if joint_effort_limits.dim() == 2:
            joint_effort_limits = joint_effort_limits[0]
        self.torque_limits = joint_effort_limits.to(device=self.device, dtype=torch.float32)
        if not torch.all(torch.isfinite(self.torque_limits)) or torch.any(self.torque_limits <= 0.0):
            raise RuntimeError(f"Invalid joint effort limits: {self.torque_limits.detach().cpu().tolist()}")

        self.p_gains = torch.zeros(self.num_dofs, device=self.device)
        self.d_gains = torch.zeros(self.num_dofs, device=self.device)
        for i, name in enumerate(self.dof_names):
            for key, val in self.cfg.control.stiffness.items():
                if key == name or key in name:
                    self.p_gains[i] = float(val)
            for key, val in self.cfg.control.damping.items():
                if key == name or key in name:
                    self.d_gains[i] = float(val)
        ################################################################################
        self.arm_joint_names = list(self.cfg.arm_joint_names)
        self.arm_joint_ids = torch.tensor(
            [self.dof_names_to_idx[n] for n in self.arm_joint_names if n in self.dof_names_to_idx],
            dtype=torch.long,
            device=self.device,
        )
        hip_joint_names = list(self.cfg.asset.hip_joint_names or [])
        self.hip_indices = torch.tensor(
            [self.dof_names_to_idx[n] for n in hip_joint_names if n in self.dof_names_to_idx],
            dtype=torch.long,
            device=self.device,
        )

        self.gripper_joint_names = list(self.cfg.gripper_joint_names)
        self.gripper_joint_ids = torch.tensor(
            [self.dof_names_to_idx[n] for n in self.gripper_joint_names if n in self.dof_names_to_idx],
            dtype=torch.long,
            device=self.device,
        )        
        print("[arm_joint_ids]", self.arm_joint_ids.detach().cpu().tolist())
        print("[gripper_joint_ids]", self.gripper_joint_ids.detach().cpu().tolist())        
        ############################################        
        self.policy_all_joint_names = list(self.cfg.policy_joint_names)
        for gripper_joint_name in self.cfg.gripper_joint_names:
            if gripper_joint_name in self.dof_names_to_idx and gripper_joint_name not in self.policy_all_joint_names:
                self.policy_all_joint_names.append(gripper_joint_name)

        self.policy_all_joint_ids = torch.tensor(
            [self.dof_names_to_idx[n] for n in self.policy_all_joint_names],
            dtype=torch.long,
            device=self.device,
        )#  

    def _env_to_policy_all(self, vec: torch.Tensor) -> torch.Tensor:
        """Convert a sim-order 19-DOF tensor to policy-all order.

        Only use this for tensors whose last dim is num_dofs=19:
        - dof_pos
        - dof_vel
        - default_dof_pos
        - torques if needed

        Do NOT use this for action tensors, because actions are 18-dim.
        """
        if vec.shape[-1] != self.num_dofs:
            raise RuntimeError(
                f"_env_to_policy_all expects last dim == num_dofs ({self.num_dofs}), "
                f"but got shape {tuple(vec.shape)}. "
                "Do not call _env_to_policy_all() on 18-dim action tensors."
            )
        return vec[:, self.policy_all_joint_ids]

    def _env_to_policy_dog(self, vec: torch.Tensor) -> torch.Tensor:
        return vec[:, self.policy_joint_ids[:12]]

    def _policy_to_env_all(self, actions: torch.Tensor) -> torch.Tensor:
        # Input is 18 policy actions. Return same 18 policy order here.
        # Actual sim-order mapping happens via self.policy_joint_ids in _compute_torques.
        return actions               

    def _init_policy_compat_buffers(self):
        # Policy-side buffers used by the ManipLoco observation path.
        num_proprio = int(self.cfg.env.num_proprio)
        num_priv = int(self.cfg.env.num_priv)
        history_len = int(self.cfg.env.history_len)
        num_gripper_joints = int(self.cfg.env.num_gripper_joints)

        self._policy_obs_buf = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)
        self._proprio_buf = torch.zeros(self.num_envs, num_proprio, device=self.device)
        self.privileged_obs_buf = None

        self.obs_history_buf = torch.zeros(
            self.num_envs, history_len, num_proprio, device=self.device
        )

        # action_history_buf stores env-order full policy actions.
        # action_delay=3 is in metadata, but auto mode uses undelayed actions early in training/play.
        self.action_delay = int(self.cfg.action_delay)
        self.action_delay_mode = self.cfg.action_delay_mode
        self.action_history_buf = torch.zeros(
            self.num_envs,
            max(self.action_delay + 1, int(self.cfg.action_delay_history_length_min)),
            self.num_actions,
            device=self.device,
        )

        self.global_steps = 0
        self.last_torques = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.target_pos = self.default_dof_pos.clone()
        self._scaled_actions_buf = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self._actions_work_buf = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self._policy_target_pos_buf = torch.zeros(self.num_envs, self.policy_joint_ids.numel(), device=self.device)
        self._torques_unclipped_buf = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self._joint_position_target_buf = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self._default_policy_joint_pos = self.default_dof_pos[:, self.policy_joint_ids].clone()
        self._leg_policy_joint_ids = self.policy_joint_ids[:12]
        self._default_leg_joint_pos = self.default_dof_pos[:, self._leg_policy_joint_ids].clone()
        self._leg_scaled_actions_buf = torch.zeros(self.num_envs, self._leg_policy_joint_ids.numel(), device=self.device)
        self._leg_target_pos_buf = torch.zeros_like(self._default_leg_joint_pos)
        self._leg_torques_unclipped_buf = torch.zeros_like(self._default_leg_joint_pos)
        self._leg_joint_slice = None
        if self._leg_policy_joint_ids.numel() == 12:
            leg_start = int(self._leg_policy_joint_ids[0].item())
            expected_leg_ids = torch.arange(
                leg_start,
                leg_start + self._leg_policy_joint_ids.numel(),
                device=self.device,
                dtype=self._leg_policy_joint_ids.dtype,
            )
            if bool(torch.equal(self._leg_policy_joint_ids, expected_leg_ids)):
                self._leg_joint_slice = slice(leg_start, leg_start + self._leg_policy_joint_ids.numel())

        self._policy_obs_joint_ids = (
            self.policy_all_joint_ids[:-num_gripper_joints]
            if num_gripper_joints > 0
            else self.policy_all_joint_ids
        )
        self._default_policy_obs_joint_pos = self.default_dof_pos[:, self._policy_obs_joint_ids].clone()
        self._dof_pos_obs_buf = torch.zeros(self.num_envs, self._policy_obs_joint_ids.numel(), device=self.device)
        self._dof_vel_obs_buf = torch.zeros_like(self._dof_pos_obs_buf)
        self._body_orientation_obs_buf = torch.zeros(self.num_envs, 2, device=self.device)
        self._commands_obs_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self._foot_contacts_obs_buf = torch.zeros(
            self.num_envs, len(self.cfg.foot_body_names), device=self.device
        )
        self._goal_height_mask_obs_buf = torch.zeros(self.num_envs, 1, device=self.device)

        self._p_gains_view = self.p_gains.unsqueeze(0)
        self._d_gains_view = self.d_gains.unsqueeze(0)
        self.last_dof_vel = torch.zeros(self.num_envs, self.num_dofs, device=self.device)

        # Old obs scales used by ManipLoco.compute_observations().
        self.obs_scales = type("ObsScales", (), {})()
        self.obs_scales.ang_vel = float(self.cfg.obs_scales["ang_vel"])
        self.obs_scales.dof_pos = float(self.cfg.obs_scales["dof_pos"])
        self.obs_scales.dof_vel = float(self.cfg.obs_scales["dof_vel"])

        # Old command scaling for [lin_x, lin_y, yaw].
        self.commands_scale = torch.tensor(self.cfg.commands_scale, dtype=torch.float32, device=self.device)

        self.mass_params_tensor = torch.tensor(
            [self.cfg.priv_mass_params], dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self.friction_coeffs_tensor = torch.tensor(
            [self.cfg.priv_friction_coeffs], dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        motor_strength_minus_1 = torch.tensor(
            [self.cfg.priv_motor_strength_minus_1], dtype=torch.float32, device=self.device
        )
        self.motor_strength = 1.0 + motor_strength_minus_1
        self._priv_obs_const = torch.cat(
            [
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[:, :12] - 1,
            ],
            dim=-1,
        )
        self._policy_priv_start = num_proprio
        self._policy_priv_end = self._policy_priv_start + num_priv
        self._policy_history_start = self._policy_priv_end
        self._policy_obs_buf[:, self._policy_priv_start : self._policy_priv_end] = self._priv_obs_const
        self._ee_goal_orientation_dummy = torch.zeros(self.num_envs, 3, device=self.device)

        # EE goal obs mode "command": checkpoint expects curr_ee_goal_cart.
        self.curr_ee_goal_cart = torch.tensor(
            self.cfg.initial_ee_goal_cart,
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)        
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart_world = None
        self.ee_goal_orn_quat = None
        self.ee_pos = None
        self.ee_orn = None
        self.ee_j_eef = None
        self.gripper_body_name = self.cfg.gripper_body_name

        # Height-reference modes always exist; mixed_height_reference only controls the policy mode bit.
        self.goal_height_follow_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # gait obs: gait_indices + 4 clock inputs.
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)
        self._gait_phase_buf = torch.zeros(self.num_envs, 4, device=self.device)
        self._gait_phase_scaled_buf = torch.zeros_like(self._gait_phase_buf)

        # Foot contacts come from the contact sensor when it is available.
        self.foot_contacts_from_sensor = torch.zeros(
            self.num_envs, len(self.cfg.foot_body_names), dtype=torch.bool, device=self.device
        )
        self.feet_air_time = torch.zeros(
            self.num_envs, len(self.cfg.foot_body_names), dtype=torch.float, device=self.device
        )
        self._foot_contact_forces_w = torch.zeros(
            self.num_envs, len(self.cfg.foot_body_names), 3, dtype=torch.float, device=self.device
        )
        self._contact_forces_w = torch.zeros(
            self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device
        )
        self.last_contact_forces = torch.zeros_like(self._foot_contact_forces_w)
        self._printed_contact_shape = False

        # Noise disabled for play.
        self.add_noise = False

    @staticmethod
    def _class_or_dict_to_dict(obj):
        if obj is None:
            raise ValueError("Expected config object or dict, got None.")
        if isinstance(obj, dict):
            return dict(obj)
        out = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue
            value = getattr(obj, key)
            if callable(value) or isinstance(value, type):
                continue
            out[key] = value
        return out

    def _init_reward_compat(self):
        if not bool(self.cfg.compute_rewards):
            self.reward_scales = {}
            self.arm_reward_scales = {}
            self.reward_container = None
            self.reward_functions = []
            self.reward_names = []
            self.arm_reward_functions = []
            self.arm_reward_names = []
            self.episode_sums = {}
            self.episode_metric_sums = {}
            return

        raise RuntimeError(
            "This vendored visual_wholebody low-level snapshot omits training rewards. "
            "Use it with cfg.compute_rewards=False for frozen low-level control."
        )

    @property
    def root_states(self):
        # Gym-style [pos(3), quat wxyz(4), lin_vel_w(3), ang_vel_w(3)]
        return torch.cat((
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.robot.data.root_lin_vel_w,
            self.robot.data.root_ang_vel_w,
        ), dim=-1)

    @property
    def dof_pos(self):
        return self.robot.data.joint_pos

    @property
    def dof_vel(self):
        return self.robot.data.joint_vel

    @property
    def base_quat(self):
        return self.robot.data.root_quat_w

    @property
    def base_lin_vel(self):
        return self.robot.data.root_lin_vel_b

    @property
    def base_ang_vel(self):
        return self.robot.data.root_ang_vel_b

    @property
    def projected_gravity(self):
        return self.robot.data.projected_gravity_b

    @property
    def rigid_body_state(self):
        return self.robot.data.body_state_w

    @property
    def foot_velocities(self):
        return self.rigid_body_state[:, self.feet_indices, 7:10]

    @property
    def force_sensor_tensor(self):
        return self._foot_contact_forces_w

    @property
    def contact_forces(self):
        if self.contact_sensor is None:
            return self._contact_forces_w
        return self.contact_sensor.data.net_forces_w

    def _pre_physics_step(self, actions: torch.Tensor):
        self._profile_step_start()

        if actions.shape[-1] != self.num_actions:
            raise RuntimeError(f"Expected action dim {self.num_actions}, got {actions.shape[-1]}")

        clip = float(self.cfg.control.clip_actions)

        section_start = time.perf_counter()
        actions_work = self._actions_work_buf
        actions_work.copy_(actions.to(self.device))
        # ManipLoco step path zeroes arm action columns.
        actions_work[:, 12:] = 0.0
        torch.clamp(actions_work, -clip, clip, out=actions_work)
        self._profile_record("base_action_clip", time.perf_counter() - section_start)

        mode = self.action_delay_mode

        section_start = time.perf_counter()
        if mode == "undelayed":
            self.action_history_buf[:, -1].copy_(actions_work)
            effective_actions = actions_work
        elif mode == "delayed":
            self.action_history_buf[:, :-1] = self.action_history_buf[:, 1:].clone()
            self.action_history_buf[:, -1] = actions_work
            effective_actions = self.action_history_buf[:, -2]
        else:
            self.action_history_buf[:, :-1] = self.action_history_buf[:, 1:].clone()
            self.action_history_buf[:, -1] = actions_work
            # Play/training early phase uses undelayed actions.
            if self.global_steps < int(self.cfg.action_delay_auto_switch_steps):
                effective_actions = self.action_history_buf[:, -1]
            else:
                effective_actions = self.action_history_buf[:, -2]
        self._profile_record("base_action_history", time.perf_counter() - section_start)

        section_start = time.perf_counter()
        if bool(self.cfg.compute_rewards):
            self.last_actions.copy_(self.actions)
            self.last_dof_vel.copy_(self.dof_vel)
            self.last_torques.copy_(self.torques)
        self.actions.copy_(effective_actions)
        self.global_steps += 1
        self._profile_record("base_state_bookkeeping", time.perf_counter() - section_start)

        if self._should_update_rgb_camera_poses_this_step():
            section_start = time.perf_counter()
            self._update_rgb_camera_poses_for_env(env_id=0)
            self._profile_record("rgb_pose", time.perf_counter() - section_start)

    def _apply_action(self):
        # Full sim-order joint position target.
        # Shape: [num_envs, num_dofs].
        # It is generated in _compute_torques(actions).

        self.torques = self._compute_torques(self.actions)
        # Optional effort target. The ManipLoco control path
        # arm/gripper torque is zeroed in _compute_torques().
        self.robot.set_joint_effort_target(self.torques)
                    
        # arm/gripper: only position target
        pos_targets = self._joint_position_target_buf
        pos_targets.copy_(self.dof_pos)
        pos_targets[:, self.arm_joint_ids] = self.target_pos[:, self.arm_joint_ids]
        pos_targets[:, self.gripper_joint_ids] = self.target_pos[:, self.gripper_joint_ids]

        self.robot.set_joint_position_target(pos_targets)

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute sim-order effort targets from policy-order actions.

        actions:
            [num_envs, 18], policy joint order:
            12 leg joints + 6 arm joints.

        target_pos:
            [num_envs, 19], IsaacLab sim joint order:
            12 legs + 6 arm + jointGripper.
        """

        if actions.shape[-1] != self.num_actions:
            raise RuntimeError(f"Expected action dim {self.num_actions}, got {actions.shape[-1]}")

        if self._leg_joint_slice is not None and self.cfg.control.control_type == "P":
            scaled_legs = self._leg_scaled_actions_buf
            torch.mul(actions[:, :12], self.action_scale_tensor[:12], out=scaled_legs)

            leg_targets = self._leg_target_pos_buf
            torch.add(self._default_leg_joint_pos, scaled_legs, out=leg_targets)

            leg_slice = self._leg_joint_slice
            self.target_pos[:, leg_slice] = leg_targets

            leg_torques = self._leg_torques_unclipped_buf
            torch.sub(leg_targets, self.dof_pos[:, leg_slice], out=leg_torques)
            leg_torques.mul_(self._p_gains_view[:, leg_slice])
            leg_torques.addcmul_(self.dof_vel[:, leg_slice], self._d_gains_view[:, leg_slice], value=-1.0)

            leg_torque_limits = self.torque_limits[leg_slice]
            torch.maximum(leg_torques, -leg_torque_limits, out=self.torques[:, leg_slice])
            torch.minimum(self.torques[:, leg_slice], leg_torque_limits, out=self.torques[:, leg_slice])
            return self.torques

        # Per-action scale from checkpoint metadata.
        # Shape: [num_envs, 18]
        scaled = self._scaled_actions_buf
        torch.mul(actions, self.action_scale_tensor, out=scaled)

        # Start from default pose in sim joint order.
        # Shape: [num_envs, num_dofs]
        target_pos = self.target_pos
        target_pos.copy_(self.default_dof_pos)

        # Write policy actions into corresponding sim joint ids.
        joint_ids = self.policy_joint_ids
        torch.add(
            self._default_policy_joint_pos,
            scaled[:, : joint_ids.numel()],
            out=self._policy_target_pos_buf,
        )
        target_pos[:, joint_ids] = self._policy_target_pos_buf

        # Keep for IsaacLab position drive.
        self.target_pos = target_pos

        if self.cfg.control.control_type == "P":
            torques_unclipped = self._torques_unclipped_buf
            torch.sub(target_pos, self.dof_pos, out=torques_unclipped)
            torques_unclipped.mul_(self._p_gains_view)
            torques_unclipped.addcmul_(self.dof_vel, self._d_gains_view, value=-1.0)
        elif self.cfg.control.control_type == "T":
            torques_unclipped = self._torques_unclipped_buf
            torques_unclipped.zero_()
            torques_unclipped[:, joint_ids] = scaled[:, : joint_ids.numel()]
        else:
            raise NotImplementedError(f"Unsupported control type: {self.cfg.control.control_type}")

        # ManipLoco zeroes arm torque and uses position targets for arm/gripper.
        if self.arm_joint_ids.numel() > 0:
            torques_unclipped[:, self.arm_joint_ids] = 0.0
        if self.gripper_joint_ids.numel() > 0:
            torques_unclipped[:, self.gripper_joint_ids] = 0.0

        torques = self.torques
        torch.maximum(torques_unclipped, -self.torque_limits, out=torques)
        torch.minimum(torques, self.torque_limits, out=torques)

        return torques 

    def _get_body_orientation(self, return_yaw: bool = False):
        r, p, y = euler_from_quat(self.base_quat)
        body_angles = torch.stack([r, p, y], dim=-1)
        return body_angles if return_yaw else body_angles[:, :-1]

    def _get_body_orientation_obs(self):
        q = self.base_quat
        w, x, y, z = q.unbind(-1)
        out = self._body_orientation_obs_buf
        out[:, 0] = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        out[:, 1] = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        return out

    def get_ground_heights(self, xy: torch.Tensor | None = None, env_ids: torch.Tensor | None = None):
        """Return terrain ground height at world xy positions.

        Converts world xy into terrain height-grid indices and uses the minimum
        of neighboring samples.
        """
        if xy is None:
            if env_ids is None:
                xy = self.root_states[:, :2]
            else:
                xy = self.root_states[env_ids, :2]
        else:
            xy = xy[..., :2]
        if xy.shape[-1] != 2:
            raise RuntimeError(f"get_ground_heights expects xy coordinates with final dim 2, got {tuple(xy.shape)}")

        xy_flat = xy.reshape(-1, 2)
        if self.cfg.terrain.terrain_type == "plane":
            heights = torch.zeros(xy_flat.shape[0], device=xy.device, dtype=xy.dtype)
            return heights.view(xy.shape[:-1])
        if self._terrain_height_samples is None:
            raise RuntimeError(
                f"Terrain type {self.cfg.terrain.terrain_type!r} does not have regular height samples."
            )

        origin_xy = self._terrain_height_origin_xy.to(device=xy.device, dtype=xy.dtype)
        horizontal_scale = self._terrain_height_horizontal_scale.to(device=xy.device, dtype=xy.dtype)
        vertical_scale = self._terrain_height_vertical_scale.to(device=xy.device, dtype=xy.dtype)
        height_samples = self._terrain_height_samples.to(device=xy.device, dtype=xy.dtype)

        points = ((xy_flat - origin_xy) / horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, height_samples.shape[0] - 2)
        py = torch.clip(py, 0, height_samples.shape[1] - 2)

        heights1 = height_samples[px, py]
        heights2 = height_samples[px + 1, py]
        heights3 = height_samples[px, py + 1]
        heights = torch.minimum(torch.minimum(heights1, heights2), heights3)
        heights = heights * vertical_scale
        return heights.view(xy.shape[:-1])

    def _init_terrain_height_samples(self):
        self._terrain_height_samples = None
        self._terrain_height_origin_xy = None
        self._terrain_height_horizontal_scale = None
        self._terrain_height_vertical_scale = None
        if self.cfg.terrain.terrain_type == "plane":
            return

        mesh = self._terrain.meshes["terrain"]
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise RuntimeError(f"Expected terrain vertices with shape [N, 3], got {vertices.shape}")

        xs, x_inverse = np.unique(vertices[:, 0], return_inverse=True)
        ys, y_inverse = np.unique(vertices[:, 1], return_inverse=True)
        if len(xs) < 2 or len(ys) < 2:
            raise RuntimeError("Terrain mesh does not contain a 2D height grid.")
        x_steps = np.diff(xs)
        y_steps = np.diff(ys)
        horizontal_scale = float(np.median(x_steps))
        if not np.allclose(x_steps, horizontal_scale) or not np.allclose(y_steps, horizontal_scale):
            raise RuntimeError("Terrain mesh vertices do not form a regular height grid.")

        height_samples = np.full((len(xs), len(ys)), -np.inf, dtype=np.float32)
        np.maximum.at(height_samples, (x_inverse, y_inverse), vertices[:, 2])
        if not np.all(np.isfinite(height_samples)):
            raise RuntimeError("Terrain mesh vertices do not form a complete regular height grid.")

        self._terrain_height_origin_xy = torch.tensor((xs[0], ys[0]), device=self.device, dtype=torch.float32)
        self._terrain_height_horizontal_scale = torch.tensor(horizontal_scale, device=self.device, dtype=torch.float32)
        self._terrain_height_vertical_scale = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self._terrain_height_samples = torch.tensor(height_samples, device=self.device, dtype=torch.float32)

    def get_base_height_above_ground(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        ground_z = self.get_ground_heights(self.root_states[:, :2], env_ids=env_ids)
        return self.root_states[:, 2] - ground_z

    def _use_goal_height_reference_mask(self) -> bool:
        return bool(self.uses_goal_height_reference_mask)

    def _observe_goal_height_reference_mask(self) -> bool:
        return bool(self.observes_goal_height_reference_mask)

    def _use_default_trunk_follow_policy_obs(self) -> bool:
        return (
            self._use_goal_height_reference_mask()
            and self.cfg.env.trunk_follow_arm_obs_mode == "default"
        )

    def _get_policy_trunk_follow_mask(self):
        if not self._use_goal_height_reference_mask():
            return None
        if self._use_default_trunk_follow_policy_obs():
            return torch.zeros_like(self.goal_height_follow_mask)
        return self.goal_height_follow_mask

    @staticmethod
    def _sphere_to_cart(sphere: torch.Tensor) -> torch.Tensor:
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

    def _get_policy_dof_observations(self):
        dof_pos_for_obs = self.dof_pos
        dof_vel_for_obs = self.dof_vel
        if (
            self._use_default_trunk_follow_policy_obs()
            and torch.any(self.goal_height_follow_mask)
            and self.arm_joint_ids.numel() > 0
        ):
            dof_pos_for_obs = self.dof_pos.clone()
            dof_vel_for_obs = self.dof_vel.clone()
            trunk_follow_mask = self.goal_height_follow_mask.unsqueeze(1)
            default_arm_pos = self.default_dof_pos[:, self.arm_joint_ids]
            dof_pos_for_obs[:, self.arm_joint_ids] = torch.where(
                trunk_follow_mask,
                default_arm_pos,
                dof_pos_for_obs[:, self.arm_joint_ids],
            )
            dof_vel_for_obs[:, self.arm_joint_ids] = torch.where(
                trunk_follow_mask,
                torch.zeros_like(dof_vel_for_obs[:, self.arm_joint_ids]),
                dof_vel_for_obs[:, self.arm_joint_ids],
            )

            dof_pos_obs = self._env_to_policy_all(
                (dof_pos_for_obs - self.default_dof_pos) * self.obs_scales.dof_pos
            )[:, :-self.num_gripper_joints]
            dof_vel_obs = self._env_to_policy_all(
                dof_vel_for_obs * self.obs_scales.dof_vel
            )[:, :-self.num_gripper_joints]
            return dof_pos_obs, dof_vel_obs

        obs_joint_ids = self._policy_obs_joint_ids
        dof_pos_obs = self._dof_pos_obs_buf
        torch.sub(self.dof_pos[:, obs_joint_ids], self._default_policy_obs_joint_pos, out=dof_pos_obs)
        dof_pos_obs.mul_(self.obs_scales.dof_pos)

        dof_vel_obs = self._dof_vel_obs_buf
        torch.mul(self.dof_vel[:, obs_joint_ids], self.obs_scales.dof_vel, out=dof_vel_obs)
        return dof_pos_obs, dof_vel_obs

    def _get_policy_ee_goal_local_cart(self, ee_goal_local_cart):
        if (
            not self._use_default_trunk_follow_policy_obs()
            or not torch.any(self.goal_height_follow_mask)
            or self.reset_init_ee_sphere is None
        ):
            return ee_goal_local_cart
        default_ee_goal_cart = self._sphere_to_cart(self.reset_init_ee_sphere)
        return torch.where(
            self.goal_height_follow_mask.unsqueeze(1),
            default_ee_goal_cart,
            ee_goal_local_cart,
        )

    def _refresh_contact_compat_buffers(self):
        if self.contact_sensor is None:
            self._contact_forces_w.zero_()
            self._foot_contact_forces_w.zero_()
            self.foot_contacts_from_sensor.zero_()
            return self._foot_contact_forces_w

        net_forces = self.contact_sensor.data.net_forces_w
        self._contact_forces_w = net_forces

        if not self._printed_contact_shape:
            self._printed_contact_shape = True
            print("[contact net_forces_w shape]", tuple(net_forces.shape))
            if hasattr(self.contact_sensor, "body_names"):
                print("[contact sensor body_names]", self.contact_sensor.body_names)

        if net_forces.shape[1] == len(self.cfg.foot_body_names):
            forces = net_forces
        elif net_forces.shape[1] == len(self.body_names):
            forces = net_forces[:, self.feet_indices, :]
        else:
            print("[WARN] unexpected contact force shape:", tuple(net_forces.shape))
            print("[WARN] robot feet_indices:", self.feet_indices.detach().cpu().tolist())
            forces = torch.zeros(self.num_envs, len(self.cfg.foot_body_names), 3, device=self.device)

        self._foot_contact_forces_w = forces
        force_norm = torch.norm(forces, dim=-1)
        self.foot_contacts_from_sensor = force_norm > float(self.cfg.contact_force_threshold)
        return forces

    def _update_policy_aux_obs(self):
        if self.contact_sensor is not None:
            self._refresh_contact_compat_buffers()
        
    def _get_observations(self):
        profile_start = time.perf_counter()
        section_start = time.perf_counter()
        self._update_policy_aux_obs()
        self._profile_record("obs_aux", time.perf_counter() - section_start)

        section_start = time.perf_counter()
        ee_goal_obs_mode = self.cfg.env.ee_goal_obs_mode

        if ee_goal_obs_mode == "command":
            ee_goal_local_cart = self.curr_ee_goal_cart
        elif ee_goal_obs_mode == "arm_base_target":
            arm_base_name = self.cfg.asset.arm_waist_name

            if (
                arm_base_name in self.body_names_to_idx
                and hasattr(self.robot.data, "body_state_w")
                and self.curr_ee_goal_cart_world is not None
            ):
                arm_base_idx = self.body_names_to_idx[arm_base_name]
                arm_base_pos = self.robot.data.body_state_w[:, arm_base_idx, :3]
                ee_goal_local_cart = quat_rotate_inverse(
                    self.base_quat,
                    self.curr_ee_goal_cart_world - arm_base_pos,
                )
            else:
                ee_goal_local_cart = self.curr_ee_goal_cart
        else:
            raise ValueError(f"Unsupported ee_goal_obs_mode: {ee_goal_obs_mode}")

        obs_body_orientation = self._get_body_orientation_obs()
        # dim 2

        obs_base_ang_vel = self.base_ang_vel * self.obs_scales.ang_vel
        # dim 3

        obs_dof_pos, obs_dof_vel = self._get_policy_dof_observations()
        # dim 18

        obs_ee_goal_local_cart = self._get_policy_ee_goal_local_cart(ee_goal_local_cart)
        # dim 3

        obs_last_leg_actions = self.action_history_buf[:, -1, :12]
        # dim 12

        obs_commands = self._commands_obs_buf
        torch.mul(self.commands[:, :3], self.commands_scale, out=obs_commands)
        # dim 3
        self._profile_record("obs_terms", time.perf_counter() - section_start)

        section_start = time.perf_counter()
        obs_buf = self._proprio_buf
        obs_parts = [
            obs_body_orientation,
            obs_base_ang_vel,
            obs_dof_pos,
            obs_dof_vel,
            obs_last_leg_actions,
        ]
        offset = 2 + 3 + 18 + 18 + 12
        if self.cfg.env.observe_foot_contacts:
            obs_contacts = self._foot_contacts_obs_buf
            if self.cfg.env.zero_observed_foot_contacts:
                obs_contacts.zero_()
            else:
                obs_contacts.copy_(self.foot_contacts_from_sensor)
            obs_parts.append(obs_contacts)
            offset += 4
        obs_parts.append(obs_commands)
        offset += 3
        obs_parts.append(obs_ee_goal_local_cart)
        offset += 3
        obs_parts.append(self._ee_goal_orientation_dummy)
        offset += 3
        if self._observe_goal_height_reference_mask():
            obs_goal_height_mask = self._goal_height_mask_obs_buf
            obs_goal_height_mask.copy_(self._get_policy_trunk_follow_mask().unsqueeze(1))
            obs_parts.append(obs_goal_height_mask)
            offset += 1
        if self.cfg.env.observe_gait_commands:
            obs_parts.append(self.gait_indices.unsqueeze(1))
            offset += 1
            obs_parts.append(self.clock_inputs)
            offset += 4

        # Sanity: policy proprio dim must match the config/metadata.
        num_proprio = int(self.cfg.env.num_proprio)
        num_priv = int(self.cfg.env.num_priv)
        if offset != num_proprio:
            raise RuntimeError(
                f"Expected proprio dim {num_proprio}, got {offset}. "
                f"Check obs layout."
            )
        torch.cat(obs_parts, dim=1, out=obs_buf)

        priv_buf = self._priv_obs_const
        if priv_buf.shape[-1] != num_priv:
            raise RuntimeError(f"Expected priv dim {num_priv}, got {priv_buf.shape[-1]}")

        policy_obs_buf = self._policy_obs_buf
        policy_obs_buf[:, :num_proprio] = obs_buf
        policy_obs_buf[:, self._policy_history_start :] = self.obs_history_buf.reshape(self.num_envs, -1)

        if policy_obs_buf.shape[-1] != self.num_obs:
            raise RuntimeError(
                f"Expected obs dim {self.num_obs}, got {policy_obs_buf.shape[-1]}"
            )
        self._profile_record("obs_pack", time.perf_counter() - section_start)

        # Update history after constructing obs, matching ManipLoco observation ordering.
        section_start = time.perf_counter()
        self.obs_history_buf[:, :-1] = self.obs_history_buf[:, 1:].clone()
        self.obs_history_buf[:, -1] = obs_buf
        reset_mask = self.episode_length_buf <= 1
        self.obs_history_buf[reset_mask] = obs_buf[reset_mask].unsqueeze(1)
        self._profile_record("obs_history", time.perf_counter() - section_start)

        section_start = time.perf_counter()
        torch.clamp(
            policy_obs_buf,
            -self.cfg.control.clip_observations,
            self.cfg.control.clip_observations,
            out=policy_obs_buf,
        )
        self._profile_record("obs_clamp", time.perf_counter() - section_start)

        self._profile_record("obs", time.perf_counter() - profile_start)
        self._profile_step_end()
        return {"policy": policy_obs_buf}

    def get_gait_phase_ratios(self):
        stance_ratio = float(self.cfg.env.trot_stance_ratio)
        swing_ratio = float(self.cfg.env.trot_swing_ratio)
        return max(stance_ratio, 1e-6), max(swing_ratio, 1e-6)

    def get_gait_swing_duration_max_s(self):
        return float(self.cfg.env.trot_swing_duration_max_s)

    def get_gait_stance_duration(self, frequencies=None):
        stance_ratio, swing_ratio = self.get_gait_phase_ratios()
        relative_swing_duration = swing_ratio / (stance_ratio + swing_ratio)
        max_swing_duration_s = max(self.get_gait_swing_duration_max_s(), 0.0)
        if max_swing_duration_s <= 0.0:
            swing_duration = relative_swing_duration
        else:
            if frequencies is None:
                frequencies = self.gait_frequencies
            max_swing_duration = torch.clamp(frequencies * max_swing_duration_s, min=0.0, max=1.0)
            swing_duration = torch.minimum(
                torch.full_like(max_swing_duration, relative_swing_duration),
                max_swing_duration,
            )
        return 1.0 - swing_duration

    def _reset_gait_state(self, env_ids):
        self.gait_frequencies[env_ids] = 0.0
        self.gait_transition_active[env_ids] = False
        self.gait_transition_elapsed_s[env_ids] = 0.0
        self.gait_transition_start_frequency[env_ids] = 0.0
        self.gait_transition_target_frequency[env_ids] = 0.0
        self.gait_frequency_commands[env_ids] = float("nan")
        self.gait_requested_frequencies[env_ids] = 0.0

        gait_pattern = str(self.cfg.env.gait_pattern).lower()
        if gait_pattern == "adaptive_trot":
            reset_gait_index = 0.25
            reset_stance_duration = torch.ones_like(self.gait_frequencies[env_ids])
            reset_pair_phases = (0.375, 0.125)
            reset_foot_phases = {
                "FL_foot": 0.375,
                "FR_foot": 0.125,
                "RL_foot": 0.125,
                "RR_foot": 0.375,
            }
            zero_clock_inputs = False
        elif gait_pattern == "fixed_trot":
            fixed_frequency = torch.full_like(
                self.gait_frequencies[env_ids],
                float(self.cfg.env.fixed_trot_frequency),
            )
            reset_gait_index = 0.0
            reset_stance_duration = self.get_gait_stance_duration(fixed_frequency)
            reset_pair_phases = (0.5, 0.0)
            reset_foot_phases = {
                "FL_foot": 0.5,
                "FR_foot": 0.0,
                "RL_foot": 0.0,
                "RR_foot": 0.5,
            }
            zero_clock_inputs = True
        else:
            raise ValueError(f"Unsupported gait_pattern: {self.cfg.env.gait_pattern}")

        policy_foot_names = list(self.cfg.asset.policy_foot_names)
        reset_foot_phases = torch.tensor(
            [reset_foot_phases[name] for name in policy_foot_names],
            dtype=self.clock_inputs.dtype,
            device=self.device,
        )
        self.gait_indices[env_ids] = reset_gait_index
        self.gait_stance_durations[env_ids] = reset_stance_duration
        self.gait_pair_phases[env_ids, 0] = reset_pair_phases[0]
        self.gait_pair_phases[env_ids, 1] = reset_pair_phases[1]
        self.gait_transition_start_stance_duration[env_ids] = reset_stance_duration
        self.gait_transition_target_stance_duration[env_ids] = reset_stance_duration
        if hasattr(self, "foot_indices"):
            self.foot_indices[env_ids] = reset_foot_phases

        if zero_clock_inputs:
            self.clock_inputs[env_ids] = 0.0
            self.doubletime_clock_inputs[env_ids] = 0.0
            self.halftime_clock_inputs[env_ids] = 0.0
        else:
            self.clock_inputs[env_ids] = torch.sin(2.0 * np.pi * reset_foot_phases)
            self.doubletime_clock_inputs[env_ids] = torch.sin(4.0 * np.pi * reset_foot_phases)
            self.halftime_clock_inputs[env_ids] = torch.sin(np.pi * reset_foot_phases)
        self.desired_contact_states[env_ids] = 1.0

    def _set_fixed_trot_stopped_state(self, env_ids):
        if len(env_ids) == 0:
            return
        fixed_frequency = torch.full_like(
            self.gait_frequencies[env_ids],
            float(self.cfg.env.fixed_trot_frequency),
        )
        fixed_stance_duration = self.get_gait_stance_duration(fixed_frequency)
        self.gait_indices[env_ids] = 0.0
        self.gait_stance_durations[env_ids] = fixed_stance_duration
        self.gait_transition_start_stance_duration[env_ids] = fixed_stance_duration
        self.gait_transition_target_stance_duration[env_ids] = fixed_stance_duration
        self.gait_pair_phases[env_ids, 0] = 0.5
        self.gait_pair_phases[env_ids, 1] = 0.0
        fixed_foot_phases = {
            "FL_foot": 0.5,
            "FR_foot": 0.0,
            "RL_foot": 0.0,
            "RR_foot": 0.5,
        }
        for index, foot_name in enumerate(self.cfg.asset.policy_foot_names):
            self.foot_indices[env_ids, index] = fixed_foot_phases[foot_name]
        self.clock_inputs[env_ids] = 0.0
        self.doubletime_clock_inputs[env_ids] = 0.0
        self.halftime_clock_inputs[env_ids] = 0.0
        self.desired_contact_states[env_ids] = 1.0

    @staticmethod
    def _uniform_to_gait_pair_phases(uniform_phases, stance_durations):
        stance_durations = stance_durations.unsqueeze(1)
        swing_durations = (1.0 - stance_durations).clamp(min=1e-6)
        return torch.where(
            uniform_phases <= stance_durations,
            uniform_phases / (2.0 * stance_durations.clamp(min=1e-6)),
            0.5 + (uniform_phases - stance_durations) / (2.0 * swing_durations),
        )

    def _start_gait_transitions(self, env_mask, target_frequencies):
        if not torch.any(env_mask):
            return
        self.gait_transition_active[env_mask] = True
        self.gait_transition_elapsed_s[env_mask] = 0.0
        self.gait_transition_start_frequency[env_mask] = self.gait_frequencies[env_mask]
        self.gait_transition_target_frequency[env_mask] = target_frequencies[env_mask]
        self.gait_transition_start_stance_duration[env_mask] = self.gait_stance_durations[env_mask]
        self.gait_transition_target_stance_duration[env_mask] = self.get_gait_stance_duration(
            target_frequencies[env_mask]
        )

    def _advance_gait_transitions(self, env_mask, duration_s):
        if not torch.any(env_mask):
            return
        transition_duration = max(float(self.cfg.env.gait_transition_duration_s), self.dt)
        if not torch.is_tensor(duration_s):
            duration_s = torch.full(
                (int(torch.sum(env_mask).item()),),
                float(duration_s),
                dtype=torch.float,
                device=self.device,
            )
        previous_frequency = self.gait_frequencies[env_mask]
        elapsed = self.gait_transition_elapsed_s[env_mask] + duration_s
        progress = torch.clamp(elapsed / transition_duration, 0.0, 1.0)
        start_frequency = self.gait_transition_start_frequency[env_mask]
        target_frequency = self.gait_transition_target_frequency[env_mask]
        frequencies = torch.lerp(start_frequency, target_frequency, progress)
        mean_frequency = 0.5 * (previous_frequency + frequencies)
        self.gait_indices[env_mask] = torch.remainder(
            self.gait_indices[env_mask] + mean_frequency * duration_s,
            1.0,
        )
        self.gait_frequencies[env_mask] = frequencies
        self.gait_stance_durations[env_mask] = torch.lerp(
            self.gait_transition_start_stance_duration[env_mask],
            self.gait_transition_target_stance_duration[env_mask],
            progress,
        )
        self.gait_transition_elapsed_s[env_mask] = elapsed
        completed_local = progress >= 1.0
        if torch.any(completed_local):
            completed_ids = env_mask.nonzero(as_tuple=False).flatten()[completed_local]
            self.gait_transition_active[completed_ids] = False
            self.gait_transition_elapsed_s[completed_ids] = transition_duration
            self.gait_frequencies[completed_ids] = self.gait_transition_target_frequency[completed_ids]
            self.gait_stance_durations[completed_ids] = self.gait_transition_target_stance_duration[
                completed_ids
            ]

    def _stop_gait_at_shared_stance(self, env_mask):
        if not torch.any(env_mask):
            return
        self.gait_frequencies[env_mask] = 0.0
        self.gait_transition_active[env_mask] = False
        self.gait_transition_elapsed_s[env_mask] = 0.0
        self.gait_transition_start_frequency[env_mask] = 0.0
        self.gait_transition_target_frequency[env_mask] = 0.0
        self.gait_transition_start_stance_duration[env_mask] = self.gait_stance_durations[env_mask]
        self.gait_transition_target_stance_duration[env_mask] = self.gait_stance_durations[env_mask]

    def _advance_constant_gait_frequency(self, env_mask, duration_s):
        if not torch.any(env_mask):
            return
        if not torch.is_tensor(duration_s):
            duration_s = torch.full(
                (int(torch.sum(env_mask).item()),),
                float(duration_s),
                dtype=torch.float,
                device=self.device,
            )
        self.gait_indices[env_mask] = torch.remainder(
            self.gait_indices[env_mask] + self.gait_frequencies[env_mask] * duration_s,
            1.0,
        )

    def _refresh_gait_pair_phases(self):
        uniform_pair_phases = torch.stack(
            (torch.remainder(self.gait_indices + 0.5, 1.0), self.gait_indices),
            dim=1,
        )
        self.gait_pair_phases[:] = self._uniform_to_gait_pair_phases(
            uniform_pair_phases, self.gait_stance_durations
        )

    def get_gait_swing_permission_mask(self):
        if not self.cfg.env.observe_gait_commands:
            return torch.zeros_like(self.foot_contacts_from_sensor)
        return self.foot_indices > 0.5

    def _step_contact_targets(self):
        profile_start = time.perf_counter()
        self.velocity_command_changed[:] = torch.any(
            torch.abs(self.commands[:, :3] - self.last_velocity_command) > 1e-6,
            dim=1,
        )
        self.last_velocity_command[:] = self.commands[:, :3]
        if not self.cfg.env.observe_gait_commands:
            self._profile_record("gait", time.perf_counter() - profile_start)
            return

        gait_pattern = str(self.cfg.env.gait_pattern).lower()
        if gait_pattern not in ("adaptive_trot", "fixed_trot"):
            raise ValueError(f"Unsupported gait_pattern: {self.cfg.env.gait_pattern}")
        command_changed = torch.any(
            torch.isnan(self.gait_frequency_commands)
            | (torch.abs(self.commands[:, :3] - self.gait_frequency_commands) > 1e-6),
            dim=1,
        )
        if torch.any(command_changed):
            requested_frequencies = self._get_gait_frequency_targets()
            self.gait_requested_frequencies[command_changed] = requested_frequencies[command_changed]
            self.gait_frequency_commands[command_changed] = self.commands[command_changed, :3]
        target_frequencies = self.gait_requested_frequencies
        target_changed_during_transition = self.gait_transition_active & (
            torch.abs(target_frequencies - self.gait_transition_target_frequency) > 1e-6
        )
        changed_to_stop = target_changed_during_transition & (target_frequencies <= 1e-6)
        self.gait_transition_active[changed_to_stop] = False
        self.gait_transition_elapsed_s[changed_to_stop] = 0.0
        self._start_gait_transitions(
            target_changed_during_transition & (~changed_to_stop), target_frequencies
        )

        handled = self.gait_transition_active.clone()
        self._advance_gait_transitions(handled, self.dt)
        pending = (~handled) & (torch.abs(target_frequencies - self.gait_frequencies) > 1e-6)
        stopping = pending & (target_frequencies <= 1e-6)
        non_stopping = pending & (~stopping)
        self._start_gait_transitions(non_stopping, target_frequencies)
        self._advance_gait_transitions(non_stopping, self.dt)
        handled |= non_stopping

        self._refresh_gait_pair_phases()
        both_stance = torch.all(self.gait_pair_phases <= 0.5, dim=1)
        start_now = stopping & both_stance
        self._stop_gait_at_shared_stance(start_now)
        handled |= start_now
        waiting = stopping & (~both_stance)
        if torch.any(waiting):
            waiting_durations = self.gait_stance_durations[waiting]
            waiting_indices = self.gait_indices[waiting]
            frequencies = self.gait_frequencies[waiting].clamp(min=1e-6)
            first_pair_swing = (
                (waiting_indices > (waiting_durations - 0.5)) & (waiting_indices < 0.5)
            )
            time_to_shared_stance = torch.where(
                first_pair_swing,
                (0.5 - waiting_indices) / frequencies,
                (1.0 - waiting_indices) / frequencies,
            )
            crosses_shared_stance_local = time_to_shared_stance <= self.dt
            if torch.any(crosses_shared_stance_local):
                waiting_ids = waiting.nonzero(as_tuple=False).flatten()
                crossing_ids = waiting_ids[crosses_shared_stance_local]
                crossing_mask = torch.zeros_like(waiting)
                crossing_mask[crossing_ids] = True
                crossing_times = time_to_shared_stance[crosses_shared_stance_local]
                self._advance_constant_gait_frequency(crossing_mask, crossing_times)
                self._stop_gait_at_shared_stance(crossing_mask)
                handled |= crossing_mask

        self._advance_constant_gait_frequency(~handled, self.dt)
        self._refresh_gait_pair_phases()
        shaped_foot_indices = {
            "FL_foot": self.gait_pair_phases[:, 0],
            "FR_foot": self.gait_pair_phases[:, 1],
            "RL_foot": self.gait_pair_phases[:, 1],
            "RR_foot": self.gait_pair_phases[:, 0],
        }
        self.foot_indices = torch.cat(
            [shaped_foot_indices[name].unsqueeze(1) for name in self.cfg.asset.policy_foot_names],
            dim=1,
        )
        for index, foot_name in enumerate(self.cfg.asset.policy_foot_names):
            phases = shaped_foot_indices[foot_name]
            self.clock_inputs[:, index] = torch.sin(2 * np.pi * phases)
            self.doubletime_clock_inputs[:, index] = torch.sin(4 * np.pi * phases)
            self.halftime_clock_inputs[:, index] = torch.sin(np.pi * phases)

        def _compute_smoothing_multiplier(phases):
            phases = torch.remainder(phases, 1.0)
            return (
                smoothing_cdf_start(phases) * (1 - smoothing_cdf_start(phases - 0.5))
                + smoothing_cdf_start(phases - 1) * (1 - smoothing_cdf_start(phases - 1.5))
            )

        smoothing_cdf_start = torch.distributions.normal.Normal(
            0, self.cfg.rewards.kappa_gait_probs
        ).cdf
        for index, foot_name in enumerate(self.cfg.asset.policy_foot_names):
            self.desired_contact_states[:, index] = _compute_smoothing_multiplier(
                shaped_foot_indices[foot_name]
            )

        fixed_trot_stopped = (
            (gait_pattern == "fixed_trot")
            & (target_frequencies <= 1e-6)
            & (self.gait_frequencies <= 1e-6)
        )
        if torch.any(fixed_trot_stopped):
            self._set_fixed_trot_stopped_state(
                torch.nonzero(fixed_trot_stopped, as_tuple=False).flatten()
            )
        self._profile_record("gait", time.perf_counter() - profile_start)

    def _get_gait_frequency_targets(self):
        gait_pattern = str(self.cfg.env.gait_pattern).lower()
        walking_mask = self._get_walking_cmd_mask()
        if gait_pattern == "fixed_trot":
            fixed_frequency = float(self.cfg.env.fixed_trot_frequency)
            if fixed_frequency <= 0.0:
                raise ValueError(
                    f"cfg.env.fixed_trot_frequency must be positive, got {fixed_frequency}"
                )
            return torch.where(
                walking_mask,
                torch.full_like(self.gait_frequencies, fixed_frequency),
                torch.zeros_like(self.gait_frequencies),
            )
        if gait_pattern != "adaptive_trot":
            raise ValueError(f"Unsupported gait_pattern: {self.cfg.env.gait_pattern}")
        max_stride_x = max(float(self.cfg.env.gait_max_stride_x), 1e-6)
        max_stride_y = max(float(self.cfg.env.gait_max_stride_y), 1e-6)
        commanded_hip_velocities = self._get_commanded_hip_velocities()
        hip_cycle_frequencies = torch.linalg.norm(
            torch.stack(
                (
                    commanded_hip_velocities[:, :, 0] / max_stride_x,
                    commanded_hip_velocities[:, :, 1] / max_stride_y,
                ),
                dim=-1,
            ),
            dim=-1,
        )
        frequencies = torch.amax(hip_cycle_frequencies, dim=1)
        return torch.where(walking_mask, frequencies, torch.zeros_like(frequencies))

    def _get_commanded_hip_velocities(self):
        hip_xy_world = self.rigid_body_state[:, self.hip_body_indices, :2]
        yaw = euler_from_quat(self.base_quat)[2]
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        base_xy_world = self.root_states[:, :2].unsqueeze(1)
        hip_from_base_world = hip_xy_world - base_xy_world
        hip_from_base_xy = torch.stack(
            (
                cos_yaw * hip_from_base_world[:, :, 0] + sin_yaw * hip_from_base_world[:, :, 1],
                -sin_yaw * hip_from_base_world[:, :, 0] + cos_yaw * hip_from_base_world[:, :, 1],
            ),
            dim=-1,
        )
        yaw_rate = self.commands[:, 2].view(-1, 1)
        return torch.stack(
            (
                self.commands[:, 0].unsqueeze(1) - yaw_rate * hip_from_base_xy[:, :, 1],
                self.commands[:, 1].unsqueeze(1) + yaw_rate * hip_from_base_xy[:, :, 0],
            ),
            dim=-1,
        )
    
    def _get_walking_cmd_mask(self, env_ids=None, return_all=False):
        if env_ids is None:
            commands = self.commands
        else:
            commands = self.commands[env_ids]
        walking_mask0 = torch.abs(commands[:, 0]) > self.cfg.commands.lin_vel_x_clip
        walking_mask1 = torch.abs(commands[:, 1]) > self.cfg.commands.lin_vel_x_clip
        walking_mask2 = torch.abs(commands[:, 2]) > self.cfg.commands.ang_vel_yaw_clip
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2
        if return_all:
            return walking_mask0, walking_mask1, walking_mask2, walking_mask
        return walking_mask
        
    def _reward_tracking_lin_vel(self):
        lin_err = torch.sum((self.commands[:, :2] - self.base_lin_vel[:, :2]) ** 2, dim=-1)
        return torch.exp(-lin_err / float(self.cfg.rewards.tracking_lin_vel_sigma))

    def _reward_tracking_ang_vel(self):
        yaw_err = (self.commands[:, 2] - self.base_ang_vel[:, 2]) ** 2
        return torch.exp(-yaw_err / float(self.cfg.rewards.tracking_ang_vel_sigma))

    def _reward_lin_vel_z(self):
        return self.robot.data.root_lin_vel_b[:, 2] ** 2

    def _reward_ang_vel_xy(self):
        return torch.sum(self.base_ang_vel[:, :2] ** 2, dim=-1)

    def _reward_orientation(self):
        return torch.sum(self.projected_gravity[:, :2] ** 2, dim=-1)

    def _reward_torques(self):
        return torch.sum(self.torques[:, self.policy_joint_ids[:12]] ** 2, dim=-1)

    def _reward_dof_vel(self):
        return torch.sum(self.dof_vel[:, self.policy_joint_ids] ** 2, dim=-1)

    def _reward_action_rate(self):
        return torch.sum((self.actions - self.last_actions) ** 2, dim=-1)

    def _arm_reward_tracking_ee_pos(self):
        if self.ee_pos is None or self.curr_ee_goal_cart_world is None:
            return torch.zeros(self.num_envs, device=self.device)
        pos_err = torch.sum((self.ee_pos - self.curr_ee_goal_cart_world) ** 2, dim=-1)
        return torch.exp(-pos_err / float(self.cfg.rewards.tracking_ee_pos_sigma))

    def _arm_reward_tracking_ee_orn(self):
        if self.ee_orn is None or self.ee_goal_orn_quat is None:
            return torch.zeros(self.num_envs, device=self.device)
        dot = torch.abs(torch.sum(self.ee_orn * self.ee_goal_orn_quat, dim=-1)).clamp(0.0, 1.0)
        orn_err = 2.0 * torch.acos(dot)
        return torch.exp(-(orn_err ** 2) / float(self.cfg.rewards.tracking_ee_orn_sigma))

    def _arm_reward_arm_action_rate(self):
        return torch.sum((self.actions[:, 12:] - self.last_actions[:, 12:]) ** 2, dim=-1)

    def _get_rewards(self):
        profile_start = time.perf_counter()
        if not bool(self.cfg.compute_rewards):
            self.rew_buf.zero_()
            self.arm_rew_buf.zero_()
            self._profile_record("rewards", time.perf_counter() - profile_start)
            return self.rew_buf

        self._refresh_contact_compat_buffers()
        self.rew_buf.zero_()
        self.arm_rew_buf.zero_()
        for name, rew_fn in zip(self.reward_names, self.reward_functions):
            rew, metric = rew_fn()
            rew = rew * self.reward_scales[name] * self.dt
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_metric_sums[name] += metric * self.dt
        if bool(self.cfg.rewards.only_positive_rewards):
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        if "termination" in self.reward_scales:
            rew, metric = self.reward_container._reward_termination()
            rew = rew * self.reward_scales["termination"] * self.dt
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.episode_metric_sums["termination"] += metric * self.dt
        self.rew_buf /= 100.0

        for name, rew_fn in zip(self.arm_reward_names, self.arm_reward_functions):
            rew, metric = rew_fn()
            rew = rew * self.arm_reward_scales[name] * self.dt
            self.arm_rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_metric_sums[name] += metric * self.dt
        if bool(self.cfg.rewards.only_positive_rewards):
            self.arm_rew_buf[:] = torch.clip(self.arm_rew_buf[:], min=0.0)
        if "arm_termination" in self.arm_reward_scales:
            rew, metric = self.reward_container._reward_termination()
            rew = rew * self.arm_reward_scales["arm_termination"] * self.dt
            self.arm_rew_buf += rew
            self.episode_sums["arm_termination"] += rew
            self.episode_metric_sums["arm_termination"] += metric * self.dt
        self.arm_rew_buf /= 100.0
        self._profile_record("rewards", time.perf_counter() - profile_start)
        return self.rew_buf

    def _post_physics_step_callback(self):
        if self.cfg.env.teleop_mode:
            self._step_contact_targets()
            return
        interval = int(float(self.cfg.commands.resampling_time) / self.dt)
        if interval <= 0:
            return
        env_ids = (self.episode_length_buf % interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._step_contact_targets()

    def _get_dones(self):
        callback_start = time.perf_counter()
        self._post_physics_step_callback()
        self._profile_record("post_callback", time.perf_counter() - callback_start)

        profile_start = time.perf_counter()
        if not bool(self.cfg.check_terminations):
            self._profile_record("dones", time.perf_counter() - profile_start)
            return self._never_done_buf, self._never_timeout_buf
        if self.contact_sensor is not None and self.termination_contact_indices.numel() > 0:
            bad_contact = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices], dim=-1) > 1.0, dim=1)
        else:
            bad_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        roll, pitch, _ = euler_from_quat(self.base_quat)
        low_base = self.root_states[:, 2] < 0.1
        bad_orientation = (torch.abs(roll) > 0.8) | (torch.abs(pitch) > 0.8)
        if self.cfg.env.teleop_mode:
            self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            self.time_out_buf = self.episode_length_buf > self.episode_length_steps
        terminated = bad_contact | bad_orientation | low_base
        self._profile_record("dones", time.perf_counter() - profile_start)
        return terminated, self.time_out_buf

    def _reset_task_specific_buffers_on_reset(self, env_ids: torch.Tensor):
        """Task-specific reset hook.

        Base LeggedRobot has no extra task buffers to reset. Subclasses that
        maintain task-local state override this method explicitly.
        """
        return

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() > 0:
            self.extras["episode"] = {}
            denom = max(float(self.episode_length_s), self.dt)
            for name, values in self.episode_sums.items():
                self.extras["episode"][f"rew_{name}"] = torch.mean(values[env_ids]) / denom
                values[env_ids] = 0.0
            for name, values in self.episode_metric_sums.items():
                self.extras["episode"][f"metric_{name}"] = torch.mean(values[env_ids]) / denom
                values[env_ids] = 0.0
        super()._reset_idx(env_ids)
        joint_pos = self.default_dof_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.last_contact_forces[env_ids] = 0.0
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        if self.cfg.enable_doorman_scene and self.cfg.doorman_randomize_robot_init:
            root_state[:, 0:1] = (
                self.scene.env_origins[env_ids, 0:1]
                + torch_rand_float(-1.5, -0.6, (len(env_ids), 1), self.device)
            )
            root_state[:, 1:2] = (
                self.scene.env_origins[env_ids, 1:2]
                + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), self.device)
            )
            random_yaw = torch_rand_float(
                -torch.pi / 4, torch.pi / 4, (len(env_ids), 1), self.device
            ).squeeze(-1)
            zero_angle = torch.zeros_like(random_yaw)
            root_state[:, 3:7] = quat_from_euler_xyz(zero_angle, zero_angle, random_yaw)
        elif not self.cfg.enable_doorman_scene:
            root_state[:, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), self.device)
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
        self._reset_doorman_door_states(env_ids)
        self._reset_task_specific_buffers_on_reset(env_ids)
        self._resample_commands(env_ids)
        if self.cfg.env.teleop_mode:
            self.commands[env_ids] = self.teleop_raw_commands[env_ids]
        else:
            reset_command = torch.tensor(
                self.cfg.reset_command, dtype=self.commands.dtype, device=self.device
            )
            self.commands[env_ids, : reset_command.numel()] = reset_command.unsqueeze(0)
        self.last_velocity_command[env_ids] = self.commands[env_ids, :3]
        self.velocity_command_changed[env_ids] = False
        self._reset_gait_state(env_ids)
        if self._rgb_camera_rigs:
            self._rgb_camera_requires_reset_sync = True

    def _reset_doorman_door_states(self, env_ids: torch.Tensor):
        if not self.cfg.enable_doorman_scene:
            return
        if self.door is None:
            raise RuntimeError("self.door is not initialized while cfg.enable_doorman_scene is True.")
        for rig in self._rgb_camera_rigs.values():
            if rig.get("door_diagnostic"):
                rig.pop("_door_local_pose", None)

        root_state = self.door.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.door.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.door.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

        joint_pos = self.door.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        if self.cfg.doorman_randomize_door_init_state:
            rand_count = int(env_ids.numel() // 3)
            if rand_count > 0 and joint_pos.shape[1] > 0:
                selected = torch.randperm(env_ids.numel(), device=self.device)[:rand_count]
                joint_pos[selected, 0] = torch_rand_float(
                    0.261799, 1.74533, (rand_count, 1), self.device
                ).squeeze(-1)

        self.door.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.door.set_joint_position_target(joint_pos, env_ids=env_ids)


    def _resample_commands(self, env_ids):
        if env_ids.numel() == 0:
            return
        if self.cfg.env.teleop_mode:
            return

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = 0
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # set small commands to zero
        self.commands[env_ids, :] *= (torch.logical_or(torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip, torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip)).unsqueeze(1)
        if self._use_goal_height_reference_mask():
            trunk_follow_ratio = (
                float(self.cfg.goal_ee.sphere_center.trunk_follow_ratio)
                if bool(self.cfg.goal_ee.sphere_center.mixed_height_reference)
                else 0.0
            )
            self.goal_height_follow_mask[env_ids] = (
                torch.rand(len(env_ids), device=self.device) < trunk_follow_ratio
            )

    # Script API conveniences used by entrypoints.
    def step(self, actions: torch.Tensor):
        try:
            obs_out, rewards, terminated, truncated, extras = super().step(actions)
        finally:
            self._profile_step_end()
        obs = obs_out["policy"] if isinstance(obs_out, dict) else obs_out
        dones = torch.logical_or(terminated, truncated)
        infos = dict(extras or {})
        infos["time_outs"] = self.time_out_buf.clone()
        return obs, self.privileged_obs_buf, rewards, self.arm_rew_buf.clone(), dones, infos

    def get_observations(self):
        obs = self._get_observations()
        return obs["policy"] if isinstance(obs, dict) else obs

    def get_privileged_observations(self):
        return None
