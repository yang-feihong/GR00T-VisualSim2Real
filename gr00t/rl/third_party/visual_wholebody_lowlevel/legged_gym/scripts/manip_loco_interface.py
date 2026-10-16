from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
for p in [str(LOW_LEVEL_ROOT)]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from legged_gym.utils.helpers import build_common_arg_parser, build_env_cfg, format_env_vec, format_float_sequence, load_policy_from_checkpoint
from legged_gym.utils.isaaclab_app import add_app_launcher_args, launch_app
from legged_gym.utils.run_metadata import checkpoint_model_path
from legged_gym.utils.episode_rng import (
    AUTO_OPEN_CANONICAL_ENV_SEED_ANCHOR,
    AUTO_OPEN_POLICY_BLOCK_SIZE,
)
from legged_gym.scripts.isaaclab_viewer import IsaacLabViewerController


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_common_arg_parser()
    parser.add_argument("--stdin_teleop", action="store_true")
    parser.add_argument("--rgb_camera_dump_dir", type=str, default=None)
    return parser


def load_policy(ckpt_path: str, device):
    return load_policy_from_checkpoint(ckpt_path, device)


class ManipLoco_Policy:
    def __init__(self, args) -> None:
        self.args = args
        self.env = None
        self.policy = None
        self.obs = None
        self.env_cfg = None
        self.rgb_camera_manager = None
        self._rgb_camera_dumped = False
        self.timestamp = 0
        self._last_status_wall_time = time.perf_counter()
        self._last_status_sim_time = 0.0
        self.print_timing_breakdown = bool(args.print_timing_breakdown)
        self._status_tree_prefix = None
        self._reset_timing_stats()
        self.init_env()

    def configure_env_cfg(self, env_cfg):
        pass

    def init_env(self):
        from legged_gym.envs.manip_loco.manip_loco import ManipLocoIsaacLab

        self.env_cfg = build_env_cfg(self.args, for_play=True)
        self.configure_env_cfg(self.env_cfg)
        # Isaac Lab reads the seed from the environment config.
        if getattr(self.args, "seed", None) is not None and hasattr(self.env_cfg, "seed"):
            self.env_cfg.seed = int(self.args.seed)
        if bool(self.env_cfg.env.teleop_mode):
            self.env_cfg.episode_length_s = 10000.0
        self.env = ManipLocoIsaacLab(self.env_cfg)
        self.env.reset()
        self.obs = self.env.get_observations()

        self.policy = load_policy(str(checkpoint_model_path(self.args)), self.env.device)

        try:
            from legged_gym.scripts.rgb_camera_debug import SimRgbCameraManager

            self.rgb_camera_manager = SimRgbCameraManager(self.env, self.args.rgb_camera_config)
        except Exception as exc:
            if self.args.rgb_camera_config:
                raise
            self.rgb_camera_manager = None
            print(f"[rgb_camera][warn] RGB camera manager unavailable: {exc}")

    def _policy_actions(self, observations):
        """Run DoorMan inference with one batch shape across all workflows."""
        observations = observations.detach()
        real_batch = int(observations.shape[0])
        if not bool(self.env_cfg.enable_doorman_scene):
            return self.policy(observations, hist_encoding=True)

        explicit_seeds = getattr(self.args, "episode_seeds", None)
        if explicit_seeds is None:
            episode_seeds = torch.arange(
                int(self.args.seed),
                int(self.args.seed) + real_batch,
                device=observations.device,
                dtype=torch.long,
            )
        else:
            if len(explicit_seeds) != real_batch:
                raise ValueError("episode_seeds must match the policy batch size")
            episode_seeds = torch.as_tensor(
                explicit_seeds, device=observations.device, dtype=torch.long
            )
        relative = episode_seeds - AUTO_OPEN_CANONICAL_ENV_SEED_ANCHOR
        block_ids = torch.div(
            relative, AUTO_OPEN_POLICY_BLOCK_SIZE, rounding_mode="floor"
        )
        rows = torch.remainder(relative, AUTO_OPEN_POLICY_BLOCK_SIZE)
        actions = None
        for block_id in torch.unique(block_ids, sorted=True):
            env_ids = torch.nonzero(block_ids == block_id, as_tuple=False).flatten()
            block = observations[env_ids[0]].expand(
                AUTO_OPEN_POLICY_BLOCK_SIZE, *observations.shape[1:]
            ).clone()
            block[rows[env_ids]] = observations[env_ids]
            block_actions = self.policy(block, hist_encoding=True)
            if actions is None:
                actions = torch.empty(
                    (real_batch, *block_actions.shape[1:]),
                    device=block_actions.device,
                    dtype=block_actions.dtype,
                )
            actions[env_ids] = block_actions[rows[env_ids]]
        return actions

    def close(self):
        """Release scene and camera references before rebuilding in this app."""
        env = self.env
        camera_manager = self.rgb_camera_manager
        self.rgb_camera_manager = None
        self.obs = None
        self.policy = None
        self.env = None
        if env is not None:
            # DirectRLEnv.close() deletes the scene before stopping the
            # timeline, but the compatibility rig dictionary also owns every
            # Camera object.  Drop both owners while the stage is still valid
            # so Camera.__del__ can detach its Replicator annotators first.
            rgb_rigs = getattr(env, "_rgb_camera_rigs", None)
            had_rgb_cameras = bool(rgb_rigs)
            if rgb_rigs is not None:
                rgb_rigs.clear()
            scene = getattr(env, "scene", None)
            sensors = getattr(scene, "sensors", None)
            if sensors is not None:
                for name in list(sensors):
                    sensor = sensors[name]
                    if type(sensor).__name__ in ("Camera", "TiledCamera"):
                        del sensors[name]
                    del sensor
            del camera_manager
            gc.collect()
            # Camera.__del__ detaches annotators but Isaac Sim's global
            # ViewportManager still owns the HydraTexture/render-product
            # objects.  Destroy them explicitly so they cannot recreate the
            # old camera paths during the next stage initialization.
            if had_rgb_cameras:
                from omni.replicator.core.scripts.utils.viewport_manager import destroy_hydra_textures

                destroy_hydra_textures()

            stage = env.sim.stage
            # Isaac Lab 5.1 only executes its complete PhysX/stage cleanup
            # branch when this flag is set.  Set it at teardown rather than at
            # construction: DoorSpawner needs the regular app-owned USD stage,
            # while subsequent environments need the old /World cleared.
            if bool(getattr(self.args, "clear_stage_on_close", False)):
                env.cfg.sim.create_stage_in_memory = True
                # SimulationContext's STOP callback otherwise loops on
                # render() while the timeline is stopped.  With active
                # SyntheticData render products this never returns in
                # headless mode.  DirectRLEnv.close() clears the context right
                # after stop(), so suppressing that callback during teardown
                # is the intended one-shot behavior here.
                env.sim._disable_app_control_on_stop_handle = True
            env.close()
            # The camera prims live outside /World/envs.  Delete the
            # project-owned mount root after stop/clear; deleting it earlier
            # lets Fabric's final synchronization author it back into USD.
            camera_root = stage.GetPrimAtPath("/World/RGBCameras")
            if camera_root.IsValid():
                stage.RemovePrim("/World/RGBCameras")
            if stage.GetPrimAtPath("/World/RGBCameras").IsValid():
                raise RuntimeError("Failed to remove stale /World/RGBCameras during environment teardown")
        del env
        gc.collect()

    def _format_vec(self, tensor, env_id=0, precision=3):
        return format_env_vec(tensor, env_id=env_id, precision=precision)

    def _format_pose6(self, pos_tensor, rpy_tensor, env_id=0):
        pos = pos_tensor[env_id].detach().cpu().tolist()
        rpy = rpy_tensor[env_id].detach().cpu().tolist()
        return format_float_sequence(pos + rpy)

    def _reset_timing_stats(self):
        self._timing_count = 0
        self._policy_time_sum = 0.0
        self._policy_time_max = 0.0
        self._env_step_time_sum = 0.0
        self._env_step_time_max = 0.0
        self._rgb_preview_time_sum = 0.0
        self._rgb_preview_time_max = 0.0
        self._loop_time_sum = 0.0
        self._loop_time_max = 0.0

    def _record_timing(self, policy_dt, env_step_dt, rgb_preview_dt, loop_dt):
        self._timing_count += 1
        self._policy_time_sum += policy_dt
        self._policy_time_max = max(self._policy_time_max, policy_dt)
        self._env_step_time_sum += env_step_dt
        self._env_step_time_max = max(self._env_step_time_max, env_step_dt)
        self._rgb_preview_time_sum += rgb_preview_dt
        self._rgb_preview_time_max = max(self._rgb_preview_time_max, rgb_preview_dt)
        self._loop_time_sum += loop_dt
        self._loop_time_max = max(self._loop_time_max, loop_dt)

    @staticmethod
    def _avg_max_ms(total, maximum, count):
        if count <= 0:
            return 0.0, 0.0
        return total * 1000.0 / count, maximum * 1000.0

    @staticmethod
    def _format_env_profile_tree_v2(profile, env_step_avg_ms, root="2"):
        if not profile:
            return ""

        def indent_for(number):
            return "    " * (str(number).count(".") + 1)

        def format_line(number, title, value):
            return f"{indent_for(number)}{number} {title}，{value}"

        def value_text(label):
            data = profile.get(label)
            if data is None:
                return None
            return f"{float(data['avg_ms']):.2f}/{float(data['max_ms']):.2f}ms"

        def add(lines, number, title, label, indent):
            text = value_text(label)
            if text is not None:
                lines.append(format_line(number, title, text))

        lines = []
        add(lines, f"{root}.1", "算法侧：步进前处理(pre_total)", "pre_total", 2)
        add(lines, f"{root}.1.1", "基础动作裁剪、延迟和历史更新(base_pre)", "base_pre", 3)
        add(lines, f"{root}.1.1.1", "动作拷贝、清零和裁剪(base_action_clip)", "base_action_clip", 4)
        add(lines, f"{root}.1.1.2", "动作历史更新(base_action_history)", "base_action_history", 4)
        add(lines, f"{root}.1.1.3", "动作状态记账(base_state_bookkeeping)", "base_state_bookkeeping", 4)
        add(lines, f"{root}.1.2", "末端位姿和雅可比刷新(ee_jacobian)", "ee_jacobian", 3)
        add(lines, f"{root}.1.3", "末端目标位置和姿态更新(ee_goal)", "ee_goal", 3)
        add(lines, f"{root}.1.4", "手臂关节目标计算(arm_target)", "arm_target", 3)

        add(lines, f"{root}.2", "相机侧：RGB相机位姿更新(rgb_pose)", "rgb_pose", 2)
        add(lines, f"{root}.2.1", "读取机器人挂载体位姿(rgb_pose_read)", "rgb_pose_read", 3)
        add(lines, f"{root}.2.2", "计算相机世界位姿(rgb_pose_math)", "rgb_pose_math", 3)
        add(lines, f"{root}.2.3", "写入相机世界位姿到Isaac(rgb_pose_set)", "rgb_pose_set", 3)

        add(lines, f"{root}.3", "算法侧：动作应用(apply_total_sum)", "apply_total", 2)
        add(lines, f"{root}.3.1", "腿部PD力矩计算(torques)", "torques", 3)
        add(lines, f"{root}.3.2", "写入力矩目标到IsaacLab(set_effort)", "set_effort", 3)
        add(lines, f"{root}.3.3", "手臂和夹爪位置目标组装(pos_target)", "pos_target", 3)
        add(lines, f"{root}.3.4", "写入位置目标到IsaacLab(set_position)", "set_position", 3)

        add(lines, f"{root}.4", "仿真器内部：写入仿真数据(write_sim)", "write_data_to_sim", 2)
        add(lines, f"{root}.4.1", "机器人写入仿真数据(robot_write)", "robot_write_data_to_sim", 3)
        add(lines, f"{root}.4.1.1", "执行器模型计算(act_model)", "robot_apply_actuator_model", 4)
        add(lines, f"{root}.4.1.2", "写入力矩目标到物理引擎(sim_effort)", "robot_set_dof_actuation_forces", 4)
        add(lines, f"{root}.4.1.3", "写入关节位置目标到物理引擎(sim_pos)", "robot_set_dof_position_targets", 4)
        add(lines, f"{root}.4.1.4", "写入关节速度目标到物理引擎(sim_vel)", "robot_set_dof_velocity_targets", 4)
        add(lines, f"{root}.4.2", "其他关节体写入(other_articulations_write)", "scene_articulations_write", 3)

        dynamic_index = 1
        for label in sorted(profile):
            if label.startswith("scene_articulation_") and label.endswith("_write") and label != "scene_articulations_write":
                text = value_text(label)
                if text is not None:
                    object_name = label[len("scene_articulation_") : -len("_write")]
                    number = f"{root}.4.2.{dynamic_index}"
                    lines.append(format_line(number, f"关节体{object_name}写入({label})", text))
                    dynamic_index += 1

        scene_write_self = profile.get("write_data_to_sim", {}).get("avg_ms", 0.0) - sum(
            profile.get(label, {}).get("avg_ms", 0.0)
            for label in (
                "robot_write_data_to_sim",
                "scene_articulations_write",
                "scene_rigid_objects_write",
                "scene_rigid_object_collections_write",
                "scene_deformable_objects_write",
                "scene_sensors_write",
                "scene_surface_grippers_write",
            )
        )
        if "write_data_to_sim" in profile:
            lines.append(
                format_line(f"{root}.4.3", "场景自身写入(scene_write_self)", f"{max(scene_write_self, 0.0):.2f}ms")
            )

        add(lines, f"{root}.5", "仿真器内部：物理仿真(physics)", "physics_elapsed", 2)
        add(lines, f"{root}.6", "仿真器内部：内部渲染(render)", "render_elapsed", 2)
        add(lines, f"{root}.7", "仿真器内部：场景状态更新(scene_update)", "scene_update_elapsed", 2)
        add(lines, f"{root}.8", "算法侧：后物理回调(post_callback)", "post_callback", 2)
        add(lines, f"{root}.8.1", "步态相位和时钟输入计算(gait)", "gait", 3)
        add(lines, f"{root}.9", "算法侧：终止判断(dones)", "dones", 2)
        add(lines, f"{root}.10", "算法侧：奖励计算(rewards)", "rewards", 2)
        add(lines, f"{root}.11", "算法侧：观测构建(obs)", "obs", 2)
        add(lines, f"{root}.11.1", "观测辅助刷新(obs_aux)", "obs_aux", 3)
        add(lines, f"{root}.11.2", "观测项计算(obs_terms)", "obs_terms", 3)
        add(lines, f"{root}.11.3", "观测拼装(obs_pack)", "obs_pack", 3)
        add(lines, f"{root}.11.4", "观测历史更新(obs_history)", "obs_history", 3)
        add(lines, f"{root}.11.5", "观测裁剪(obs_clamp)", "obs_clamp", 3)

        tracked = sum(
            profile.get(label, {}).get("avg_ms", 0.0)
            for label in (
                "pre_total",
                "rgb_pose",
                "apply_total",
                "write_data_to_sim",
                "physics_elapsed",
                "render_elapsed",
                "scene_update_elapsed",
                "post_callback",
                "dones",
                "rewards",
                "obs",
            )
        )
        lines.append(format_line(f"{root}.12", "未归类(blackbox)", f"{max(env_step_avg_ms - tracked, 0.0):.2f}ms"))
        return "\n" + "\n".join(lines)

    def _emit_status_report(self, report_text):
        print(report_text)

    def step(self):
        loop_start = time.perf_counter()
        policy_start = time.perf_counter()
        with torch.no_grad():
            actions = self._policy_actions(self.obs)
        policy_dt = time.perf_counter() - policy_start

        env_step_start = time.perf_counter()
        self.obs, *_ = self.env.step(actions.detach())
        env_step_dt = time.perf_counter() - env_step_start

        rgb_preview_dt = 0.0
        if self.rgb_camera_manager is not None:
            rgb_preview_start = time.perf_counter()
            self.rgb_camera_manager.maybe_update_preview(env_id=0)
            self._dump_rgb_cameras_once()
            rgb_preview_dt = time.perf_counter() - rgb_preview_start

        loop_dt = time.perf_counter() - loop_start
        self._record_timing(policy_dt, env_step_dt, rgb_preview_dt, loop_dt)

        if self.timestamp % 50 == 0 and self.timestamp > 0:
            sim_time = (self.timestamp + 1) * self.env.dt
            now = time.perf_counter()
            wall_dt = now - self._last_status_wall_time
            sim_dt = sim_time - self._last_status_sim_time
            rtf = sim_dt / wall_dt if wall_dt > 0.0 else 0.0
            timing_count = self._timing_count
            policy_avg_ms, policy_max_ms = self._avg_max_ms(
                self._policy_time_sum, self._policy_time_max, timing_count
            )
            env_step_avg_ms, env_step_max_ms = self._avg_max_ms(
                self._env_step_time_sum, self._env_step_time_max, timing_count
            )
            rgb_avg_ms, rgb_max_ms = self._avg_max_ms(
                self._rgb_preview_time_sum, self._rgb_preview_time_max, timing_count
            )
            loop_avg_ms, loop_max_ms = self._avg_max_ms(
                self._loop_time_sum, self._loop_time_max, timing_count
            )
            env_profile = {}
            if self.print_timing_breakdown and hasattr(self.env, "pop_step_profile_stats"):
                env_profile = self.env.pop_step_profile_stats()
            status_tree_prefix = self._status_tree_prefix
            env_profile_root = f"{status_tree_prefix}.2" if status_tree_prefix else "2"
            env_profile_text = (
                self._format_env_profile_tree_v2(env_profile, env_step_avg_ms, root=env_profile_root)
                if self.print_timing_breakdown
                else ""
            )
            cmd_text = self._format_vec(self.env.commands[:, :3])
            vel_text = self._format_vec(self.env.robot.data.root_lin_vel_w)
            ee_goal_text = self._format_vec(self.env.curr_ee_goal_cart)
            if status_tree_prefix and self.print_timing_breakdown:
                def status_indent(number):
                    return "    " * (str(number).count(".") + 1)

                report_text = (
                    f"{status_indent(f'{status_tree_prefix}.1')}{status_tree_prefix}.1 策略推理(policy)，{policy_avg_ms:.2f}/{policy_max_ms:.2f}ms"
                    f"\n{status_indent(f'{status_tree_prefix}.2')}{status_tree_prefix}.2 环境步进(env_step)，{env_step_avg_ms:.2f}/{env_step_max_ms:.2f}ms"
                    f"{env_profile_text}"
                    f"\n{status_indent(f'{status_tree_prefix}.3')}{status_tree_prefix}.3 RGB预览(rgb_preview)，{rgb_avg_ms:.2f}/{rgb_max_ms:.2f}ms"
                    f"\n状态"
                    f"\n    sim={sim_time:.2f}s wall_dt={wall_dt:.2f}s sim_dt={sim_dt:.2f}s RTF={rtf:.3f}"
                    f"\n    step={self.timestamp}"
                    f"\n    cmd={cmd_text}"
                    f"\n    vel={vel_text}"
                    f"\n    arm_mode={self.env.teleop_arm_control_mode}"
                    f"\n    ee_goal={ee_goal_text}"
                )
            elif self.print_timing_breakdown:
                report_text = (
                    f"[sim={sim_time:.2f}s wall_dt={wall_dt:.2f}s sim_dt={sim_dt:.2f}s RTF={rtf:.3f}]"
                    f"\n耗时统计(avg/max)：内部策略步总计(manip_step_loop)，{loop_avg_ms:.2f}/{loop_max_ms:.2f}ms"
                    f"\n    1 策略推理(policy)，{policy_avg_ms:.2f}/{policy_max_ms:.2f}ms"
                    f"\n    2 环境步进(env_step)，{env_step_avg_ms:.2f}/{env_step_max_ms:.2f}ms"
                    f"{env_profile_text}"
                    f"\n    3 RGB预览(rgb_preview)，{rgb_avg_ms:.2f}/{rgb_max_ms:.2f}ms"
                    f"\n状态"
                    f"\n    step={self.timestamp}"
                    f"\n    cmd={cmd_text}"
                    f"\n    vel={vel_text}"
                    f"\n    arm_mode={self.env.teleop_arm_control_mode}"
                    f"\n    ee_goal={ee_goal_text}"
                )
            else:
                report_text = (
                    f"[sim={sim_time:.2f}s] RTF={rtf:.3f} step={self.timestamp} "
                    f"cmd={cmd_text} vel={vel_text} "
                    f"arm_mode={self.env.teleop_arm_control_mode} "
                    f"ee_goal={ee_goal_text}"
                )
            self._emit_status_report(report_text)
            self._last_status_wall_time = now
            self._last_status_sim_time = sim_time
            self._reset_timing_stats()

        self.timestamp += 1
        duration = time.perf_counter() - loop_start
        time.sleep(max(float(self.env.dt) - duration, 0.0))

    def _dump_rgb_cameras_once(self):
        dump_dir = str(self.args.rgb_camera_dump_dir or "").strip()
        if self._rgb_camera_dumped or not dump_dir or self.rgb_camera_manager is None:
            return
        import cv2

        output_dir = Path(dump_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        images_by_name = self.rgb_camera_manager.render_images(env_id=0, update=True)
        for camera_name, images in images_by_name.items():
            for index, rgb in enumerate(images):
                suffix = "" if len(images) == 1 else f"_{index}"
                output_path = output_dir / f"{camera_name}{suffix}.png"
                if not cv2.imwrite(str(output_path), rgb[:, :, ::-1]):
                    raise RuntimeError(f"Failed to write RGB camera image: {output_path}")
                print(
                    f"[rgb_camera] wrote {output_path} shape={rgb.shape} "
                    f"range=[{int(rgb.min())},{int(rgb.max())}] mean={float(rgb.mean()):.2f}"
                )
        self._rgb_camera_dumped = True
    def set_teleop_targets(
        self,
        base_command,
        ee_goal_cart=None,
        ee_orn_delta_rpy=None,
        gripper_target=None,
        ee_goal_orn_quat_world=None,
        ee_goal_orn_quat_world_enabled=None,
    ):
        """Forward scripted targets to the environment teleop buffers."""
        return self.env.set_teleop_targets(
            base_command,
            ee_goal_cart=ee_goal_cart,
            ee_orn_delta_rpy=ee_orn_delta_rpy,
            gripper_target=gripper_target,
            ee_goal_orn_quat_world=ee_goal_orn_quat_world,
            ee_goal_orn_quat_world_enabled=ee_goal_orn_quat_world_enabled,
        )

    def get_control_state(self):
        return self.env.get_control_state()

    def step_once(self, sleep=False):
        """Run one low-level policy/simulation step and return its action/state."""
        if self.timestamp == 0:
            print("[doorman] first low-level policy inference...", flush=True)
        with torch.no_grad():
            actions = self._policy_actions(self.obs)
        if self.timestamp == 0:
            print(
                f"[doorman] first policy action shape={tuple(actions.shape)} "
                f"finite={bool(torch.isfinite(actions).all().item())}",
                flush=True,
            )
            print("[doorman] first Isaac Lab simulation step...", flush=True)
        self.obs, *_ = self.env.step(actions.detach())
        if self.timestamp == 0:
            print("[doorman] first simulation step complete", flush=True)
        if self.rgb_camera_manager is not None:
            self.rgb_camera_manager.maybe_update_preview(env_id=0)
        rgb_frames = self._read_rgb_frames()
        result = {
            "low_level_action": actions.detach().clone(),
            "state": self.get_control_state(),
            "rgb_frames": rgb_frames,
            "rgb": next(iter(rgb_frames.values()), None),
        }
        self.timestamp += 1
        if sleep:
            time.sleep(float(self.env.dt))
        return result

    def _read_rgb_frame(self):
        """Best-effort extraction of the first configured RGB sensor frame."""
        sensors = getattr(getattr(self.env, "scene", None), "sensors", {})
        for sensor in sensors.values() if hasattr(sensors, "values") else []:
            data = getattr(sensor, "data", None)
            output = getattr(data, "output", None)
            if isinstance(output, dict) and "rgb" in output:
                frame = output["rgb"]
                if hasattr(frame, "shape") and frame.ndim >= 4:
                    frame = frame[0]
                return frame.detach().cpu() if torch.is_tensor(frame) else frame
        return None

    def _read_rgb_frames(self):
        """Read every configured camera when its simulation-time FPS is due."""
        manager = self.rgb_camera_manager
        if manager is None or not manager.enabled:
            frame = self._read_rgb_frame()
            return {} if frame is None else {"rgb": frame}
        sim_time = float(self.timestamp) * float(self.env.dt)
        due_rigs = manager.due_rigs_at_sim_time(sim_time)
        if not due_rigs:
            return {}
        rig_names = [rig["name"] for rig in due_rigs]
        images_by_name = manager.render_images(env_id=0, rig_names=rig_names, update=False)
        frames = {}
        for rig in due_rigs:
            images = images_by_name.get(rig["name"], [])
            for index, frame in enumerate(images):
                name = rig["name"] if len(images) == 1 else f"{rig['name']}_{index}"
                frames[name] = frame.detach().cpu() if torch.is_tensor(frame) else frame
            if images:
                manager.mark_published_at_sim_time(rig, sim_time)
        return frames


def main():
    parser = build_arg_parser()
    add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = launch_app(args)

    runner = ManipLoco_Policy(args)
    viewer_controller = IsaacLabViewerController(
        runner.env,
        simulation_app=simulation_app,
        robot_key_handler=runner.env.apply_teleop_key if args.teleop_mode else None,
        enabled=not bool(getattr(args, "headless", False)),
    ).install_keyboard()
    if args.teleop_mode:
        print("[teleop] Click viewport, then use WASD/YUHJ.../OP/G plus viewer hotkeys.")

    while simulation_app.is_running() and not viewer_controller.stop_requested:
        if viewer_controller.consume_reset_request():
            runner.env.reset()
            runner.obs = runner.env.get_observations()
        if viewer_controller.paused:
            viewer_controller.tick()
            time.sleep(float(runner.env.dt))
            continue
        runner.step()
        viewer_controller.tick()


if __name__ == "__main__":
    main()
