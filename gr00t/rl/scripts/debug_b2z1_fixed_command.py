# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the B2Z1 low-level policy with a fixed physical velocity command.

Example:

    ./isaac-sim/python.sh gr00t/rl/scripts/debug_b2z1_fixed_command.py \
        +exp=b2z1_flat_lowlevel \
        num_envs=1 headless=False \
        +velocity_command='[0.2,0.0,0.0]' \
        +num_debug_steps=1000
"""

import argparse
import os
import site
import sys
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import OmegaConf, open_dict

from gr00t.rl.utils.config_utils import register_rl_resolvers

register_rl_resolvers()

from gr00t.rl.scripts.b2z1_physics_fixture import (
    JOINT_POS,
    POLICY_ACTION,
    ROOT_POS,
    EE_GOAL_CART,
    VELOCITY_COMMAND,
    joint_state_by_name,
    snapshot,
    tensor,
)


def run_open_loop(env, trace_path: str, num_substeps: int, root_height: float):
    root_pos = (ROOT_POS[0], ROOT_POS[1], root_height)
    root_state = tensor((*root_pos, 0.0, 0.0, 0.0, 1.0, *([0.0] * 6)), env.device)
    joint_pos = joint_state_by_name(env.dof_names, JOINT_POS, env.device)
    joint_vel = torch.zeros_like(joint_pos)
    env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    env.simulator.write_root_state_to_sim(root_state, env_ids)
    env.simulator.write_joint_state_to_sim(joint_pos, joint_vel, env_ids)
    env.simulator.clear_external_force_and_torque()
    env.simulator.scene.write_data_to_sim()
    env.simulator.scene.update(dt=env.sim_dt)
    env._pre_compute_observations_callback()

    # Match both physical state and retained actuator command state.
    env.actions_after_delay.zero_()
    env._b2z1_arm_pos_targets.copy_(
        env.default_dof_pos[:, env._b2z1_arm_joint_indices]
    )
    gripper_target_range = (
        env._b2z1_gripper_closed_target - env._b2z1_gripper_open_target
    )
    env.actions_after_delay[:, env._b2z1_gripper_joint_index] = (
        env._b2z1_gripper_open_command
        + (
            joint_pos[:, env._b2z1_gripper_joint_index]
            - env._b2z1_gripper_open_target
        )
        / gripper_target_range
        * (env._b2z1_gripper_closed_command - env._b2z1_gripper_open_command)
    )
    env._apply_force_in_physics_step()
    env.simulator.scene.write_data_to_sim()

    data = env.simulator._robot.data
    metadata = {
        "body_names": list(data.body_names),
        "articulation_dof_names": list(data.joint_names),
        "initial_root_state": snapshot(env.simulator.robot_root_states),
        "initial_joint_pos": snapshot(env.simulator.dof_pos),
        "initial_joint_vel": snapshot(env.simulator.dof_vel),
        "initial_body_pos": snapshot(data.body_pos_w),
        "initial_body_quat": snapshot(data.body_quat_w),
        "initial_body_lin_vel": snapshot(data.body_lin_vel_w),
        "initial_body_ang_vel": snapshot(data.body_ang_vel_w),
        "body_mass": snapshot(data.default_mass),
        "body_inertia": snapshot(data.default_inertia),
        "joint_armature": snapshot(data.default_joint_armature),
        "joint_friction": snapshot(data.default_joint_friction_coeff),
        "default_joint_pos": snapshot(env.default_dof_pos),
        "hard_dof_pos_limits": snapshot(env.simulator.hard_dof_pos_limits),
        "dof_vel_limits": snapshot(env.dof_vel_limits),
        "torque_limits": snapshot(env.torque_limits),
        "articulation_joint_pos_limits": snapshot(data.joint_pos_limits),
        "articulation_joint_vel_limits": snapshot(data.joint_vel_limits),
        "articulation_joint_effort_limits": snapshot(data.joint_effort_limits),
    }

    env._b2z1_commands.zero_()
    env._b2z1_commands[:, 3:6] = (
        tensor(EE_GOAL_CART, env.device) - env._b2z1_default_ee_goal_cart
    )
    env._refresh_b2z1_arm_target()
    goal_position, goal_orientation_xyzw = env._compute_b2z1_ee_goal_world()
    ik_alignment = {
        "goal_position": snapshot(goal_position),
        "goal_orientation_wxyz": snapshot(goal_orientation_xyzw[:, [3, 0, 1, 2]]),
        "arm_target": snapshot(env._b2z1_arm_pos_targets),
    }

    env._b2z1_commands.zero_()
    env._b2z1_commands[:, :3] = tensor(VELOCITY_COMMAND, env.device)
    env._b2z1_commands[:, 3:6] = (
        tensor(EE_GOAL_CART, env.device) - env._b2z1_default_ee_goal_cart
    )
    env._lowlevel_prev_policy_actions.zero_()
    env._lowlevel_obs_history.zero_()
    env.episode_length_buf.zero_()
    gait_reset_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    env._set_lowlevel_fixed_trot_stopped(gait_reset_mask)
    policy_obs = env._build_lowlevel_obs().clone()
    with torch.no_grad():
        policy_output = env._lowlevel_policy(policy_obs, hist_encoding=True)

    env.actions_after_delay.zero_()
    env.actions_after_delay[:, env._lowlevel_policy_joint_indices] = tensor(
        POLICY_ACTION, env.device
    )
    env.actions_after_delay[:, env._b2z1_gripper_joint_index] = (
        env._b2z1_gripper_open_command
        + (
            joint_pos[:, env._b2z1_gripper_joint_index]
            - env._b2z1_gripper_open_target
        )
        / gripper_target_range
        * (env._b2z1_gripper_closed_command - env._b2z1_gripper_open_command)
    )
    env._b2z1_arm_pos_targets.copy_(
        env.default_dof_pos[:, env._b2z1_arm_joint_indices]
    )
    records = []
    for _ in range(num_substeps):
        env._apply_force_in_physics_step()
        requested = env.torques.clone()
        position_target = env.simulator._robot.data.joint_pos_target.clone()
        env.simulator.simulate_at_each_physics_step()
        records.append(
            {
                "root_state": snapshot(env.simulator.robot_root_states),
                "joint_pos": snapshot(env.simulator.dof_pos),
                "joint_vel": snapshot(env.simulator.dof_vel),
                "requested_torque": snapshot(requested),
                "position_target": snapshot(position_target),
                "control_target": snapshot(env._b2z1_arm_pos_targets),
                "applied_torque": snapshot(
                    env.simulator._robot.data.applied_torque[:, env.simulator.dof_ids]
                ),
            }
        )
    torch.save(
        {
            "dof_names": list(env.dof_names),
            "metadata": metadata,
            "policy_alignment": {
                "proprio": snapshot(policy_obs[:, : env._lowlevel_num_prop]),
                "observation": snapshot(policy_obs),
                "action": snapshot(policy_output),
            },
            "ik_alignment": ik_alignment,
            "records": records,
        },
        trace_path,
    )
    logger.info(f"Wrote open-loop trace to {trace_path}")


def isaacsim_ext_folders() -> list[str]:
    folders = []
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        isaacsim_dir = Path(site_dir) / "isaacsim"
        for child in ("exts", "extscache", "apps"):
            ext_dir = isaacsim_dir / child
            if ext_dir.is_dir():
                folders.append(str(ext_dir))

    ov_ext_cache = Path.home() / ".local/share/ov/data/exts/v2"
    if ov_ext_cache.is_dir():
        folders.append(str(ov_ext_cache))
    isaaclab_source = Path("/workspace/isaaclab/source")
    if not isaaclab_source.exists():
        isaaclab_source = Path("/workspace/IsaacLab/source")
    if isaaclab_source.is_dir():
        folders.append(str(isaaclab_source))
    return folders


def isaacsim_kit_args(use_gpu: bool, enable_renderer: bool = False) -> str:
    args = []
    for folder in isaacsim_ext_folders():
        args.extend(["--ext-folder", folder])
    if not enable_renderer:
        args.extend(
            [
                "--/app/vulkan=false",
                "--/renderer/enabled=none",
                "--/renderer/active=none",
                "--/renderer/multiGpu/enabled=false",
                "--/renderer/multiGpu/autoEnable=false",
            ]
        )
    if not use_gpu:
        args.extend(
            [
                "--/physics/fabricUseGPUInterop=false",
                "--/physics/cudaDevice=-1",
            ]
        )
    return " ".join(args)


def write_skeleton_video(video_path: str, body_names: list[str], frames: list[torch.Tensor], fps: int):
    import imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path = Path(video_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    body_index = {name: i for i, name in enumerate(body_names)}
    edge_names = []
    for leg in ("FR", "FL", "RR", "RL"):
        edge_names.extend(
            [
                ("base_link", f"{leg}_hip"),
                (f"{leg}_hip", f"{leg}_thigh"),
                (f"{leg}_thigh", f"{leg}_calf"),
                (f"{leg}_calf", f"{leg}_foot"),
            ]
        )
    edge_names.extend(
        [
            ("base_link", "link00"),
            ("link00", "link01"),
            ("link01", "link02"),
            ("link02", "link03"),
            ("link03", "link04"),
            ("link04", "link05"),
            ("link05", "link06"),
            ("link06", "gripper_link"),
        ]
    )
    edges = [(body_index[a], body_index[b]) for a, b in edge_names if a in body_index and b in body_index]

    data = torch.stack(frames).cpu().numpy()
    mins = data.min(axis=(0, 1))
    maxs = data.max(axis=(0, 1))
    center = (mins + maxs) * 0.5
    span = float(max(maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2], 1.0))
    radius = span * 0.65
    root_path = data[:, body_index.get("base_link", 0), :]

    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    writer = imageio.get_writer(str(path), fps=fps)
    try:
        for frame_id, points in enumerate(data):
            ax.clear()
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(max(0.0, center[2] - radius * 0.5), center[2] + radius)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=22, azim=-55)
            ax.plot(root_path[: frame_id + 1, 0], root_path[: frame_id + 1, 1], root_path[: frame_id + 1, 2], color="0.3", linewidth=1.0)
            for a, b in edges:
                color = "tab:blue" if body_names[a].startswith(("F", "R")) else "tab:green"
                if body_names[a].startswith("link") or body_names[b].startswith("link") or body_names[b] == "gripper_link":
                    color = "tab:red"
                ax.plot(
                    [points[a, 0], points[b, 0]],
                    [points[a, 1], points[b, 1]],
                    [points[a, 2], points[b, 2]],
                    color=color,
                    linewidth=2.5,
                )
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=12, color="black")
            ax.text2D(0.03, 0.95, f"B2Z1 walk rollout  frame {frame_id + 1}/{len(data)}", transform=ax.transAxes)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
    finally:
        writer.close()
        plt.close(fig)

    logger.info(f"Wrote B2Z1 skeleton video to {path}")


@hydra.main(config_path="../config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    simulator_type = config.simulator["_target_"].split(".")[-1]
    simulation_app = None
    if simulator_type == "IsaacSim":
        try:
            with open("./rl/simulator/isaacsim/.isaacsim_version", "r", encoding="utf-8") as f:
                isaacsim_version = f.read().strip()
        except FileNotFoundError:
            isaacsim_version = "4.5"

        if isaacsim_version == "4.5":
            import isaaclab
            from isaaclab.app import AppLauncher
        elif isaacsim_version == "4.2":
            import omni.isaac.lab as isaaclab
            from omni.isaac.lab.app import AppLauncher
        else:
            raise ValueError(f"Unsupported IsaacSim version: {isaacsim_version}")

        parser = argparse.ArgumentParser(description="Debug B2Z1 fixed command.")
        AppLauncher.add_app_launcher_args(parser)
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.enable_cameras = (
            config.simulator.config.cameras.enable_cameras
            or config.simulator.config.render_results
        )
        args_cli.headless = config.headless
        args_cli.device = device
        args_cli.kit_args = " ".join(
            part
            for part in (
                args_cli.kit_args,
                isaacsim_kit_args(
                    torch.cuda.is_available(), enable_renderer=args_cli.enable_cameras
                ),
                "--enable isaacsim.sensors.camera" if args_cli.enable_cameras else "",
            )
            if part
        )

        if args_cli.enable_cameras and config.headless:
            os.environ.pop("DISPLAY", None)
            isaaclab_apps_path = Path(isaaclab.__file__).resolve().parents[3] / "apps"
            source_file = isaaclab_apps_path / "isaaclab.python.headless.rendering.kit"
            if not source_file.exists():
                source_file = (
                    Path(__file__).resolve().parents[1]
                    / "apps/phc.isaaclab.python.headless.rendering.kit"
                )
            args_cli.experience = source_file
        elif config.headless and not args_cli.enable_cameras:
            args_cli.experience = (
                Path(__file__).resolve().parents[1]
                / "apps/b2z1.isaaclab.python.headless.no_render.kit"
            )

        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

    from gr00t.rl.trl.utils.common import custom_instantiate
    from gr00t.rl.utils.helpers import pre_process_config

    os.chdir(hydra.utils.get_original_cwd())
    pre_process_config(config)
    no_render_headless = (
        config.headless
        and not config.simulator.config.render_results
        and not config.simulator.config.cameras.enable_cameras
    )
    with open_dict(config.simulator.config):
        config.simulator.config.disable_visual_markers = no_render_headless
        config.simulator.config.disable_visual_materials = no_render_headless
    config.env.config.save_rendering_dir = str(Path(config.experiment_dir) / "debug_renderings")
    config.env.config.experiment_dir = str(Path(config.experiment_dir))
    env = custom_instantiate(config.env, device=device, _resolve=False)
    env.reset_all()

    open_loop_trace_path = config.get("open_loop_trace_path", None)
    if open_loop_trace_path:
        run_open_loop(
            env,
            str(open_loop_trace_path),
            int(config.get("open_loop_substeps", 4)),
            float(config.get("open_loop_root_height", ROOT_POS[2])),
        )
        if simulation_app is not None:
            simulation_app.close()
        return

    if config.get("fixed_command", None) is not None:
        raise ValueError(
            "fixed_command used normalized high-level actions and is no longer supported; "
            "use velocity_command=[vx,vy,wz] in m/s, m/s, rad/s."
        )
    velocity_command = torch.tensor(
        config.get("velocity_command", [0.0, 0.0, 0.0]),
        device=env.device,
        dtype=torch.float32,
    ).view(1, 3)
    physical_commands = torch.zeros(
        env.num_envs, env.COMMAND_DIM, device=env.device, dtype=torch.float32
    )
    physical_commands[:, :3] = velocity_command
    physical_commands[:, 9] = float(config.get("gripper_command", -1.0))
    num_steps = int(config.get("num_debug_steps", 1000))

    logger.info(
        "Running B2Z1 low-level policy for {} steps with physical velocity "
        "[vx_mps, vy_mps, yaw_rate_radps]={}".format(
            num_steps, velocity_command[0].tolist()
        )
    )
    initial_root_w = None
    final_root_w = None
    reset_count = 0
    skeleton_video_path = config.get("skeleton_video_path", None)
    skeleton_stride = int(config.get("skeleton_video_stride", 2))
    skeleton_frames = []
    trace_path = config.get("trace_path", None)
    trace_steps = set(int(x) for x in config.get("trace_steps", [0, 1, 2, 3, 4, 5, 10, 20, 50]))
    trace = {"meta": {}, "steps": {}} if trace_path is not None else None
    debug_stage_cycle = bool(config.get("debug_stage_cycle", False))
    camera_dump_dir = config.get("camera_dump_dir", None)
    if trace is not None:
        trace["meta"]["dof_names"] = list(env.dof_names)
        trace["meta"]["dof_ids"] = list(env.simulator.dof_ids)
        trace["meta"]["isaaclab_joint_names"] = list(env.simulator._robot.data.joint_names)
        trace["meta"]["body_names"] = list(env.simulator.body_names)
        trace["meta"]["default_dof_pos"] = env.default_dof_pos[0].detach().cpu()
        if hasattr(env, "_lowlevel_policy_joint_indices"):
            policy_joint_indices = env._lowlevel_policy_joint_indices.detach().cpu().tolist()
            trace["meta"]["policy_joint_indices"] = policy_joint_indices
            trace["meta"]["policy_joint_names"] = [env.dof_names[i] for i in policy_joint_indices]
    for step in range(num_steps):
        if debug_stage_cycle:
            if not hasattr(env, "stage_buf"):
                raise ValueError("debug_stage_cycle requires an environment with stage_buf")
            env.stage_buf.fill_(step % env.num_stages)
        obs, rew, reset, extras = env.step_physical_b2z1_commands(
            physical_commands.clone()
        )
        env.render_results()
        if step == 0 and camera_dump_dir:
            import cv2

            output_dir = Path(str(camera_dump_dir))
            output_dir.mkdir(parents=True, exist_ok=True)
            images = env.simulator.get_rgb_images_uint8()
            for camera_name, image_batch in images.items():
                rgb = image_batch[0].detach().cpu().numpy().astype("uint8")
                output_path = output_dir / f"{camera_name}.png"
                if not cv2.imwrite(str(output_path), rgb[:, :, ::-1]):
                    raise RuntimeError(f"Failed to write camera image: {output_path}")
                logger.info(
                    "camera={} shape={} range=[{},{}] mean={:.2f} path={}".format(
                        camera_name,
                        rgb.shape,
                        int(rgb.min()),
                        int(rgb.max()),
                        float(rgb.mean()),
                        output_path,
                    )
                )
        if skeleton_video_path is not None and step % skeleton_stride == 0:
            skeleton_frames.append(env.simulator._rigid_body_pos[0].detach().cpu().clone())
        if trace is not None and step in trace_steps:
            step_trace = {
                "reward": rew[0].detach().cpu(),
                "commands": env.get_physical_b2z1_commands()[0].detach().cpu(),
                "dof_pos": env.simulator.dof_pos[0].detach().cpu(),
                "dof_vel": env.simulator.dof_vel[0].detach().cpu(),
                "base_lin_vel": env.base_lin_vel[0].detach().cpu(),
                "base_ang_vel": env.base_ang_vel[0].detach().cpu(),
                "base_quat": env.base_quat[0].detach().cpu(),
                "root_state": env.simulator.robot_root_states[0].detach().cpu(),
                "rigid_body_pos": env.simulator._rigid_body_pos[0].detach().cpu(),
                "rigid_body_rot": env.simulator._rigid_body_rot[0].detach().cpu(),
                "torques": env.torques[0].detach().cpu(),
                "actions_after_delay": env.actions_after_delay[0].detach().cpu(),
                "arm_pos_targets": env._b2z1_arm_pos_targets[0].detach().cpu(),
                "joint_pos_targets": (
                    env.simulator._robot.data.joint_pos_target[0].detach().cpu()
                ),
            }
            if hasattr(env, "stage_buf"):
                step_trace["stage"] = env.stage_buf[0].detach().cpu()
            for key, attr_name in (
                ("task_contact_prim_rot_wxyz", "task_contact_prim_rot_wxyz"),
                ("hand_target_pos_source", "hand_transform_pos"),
                ("hand_target_quat_source_wxyz", "hand_transform_rot"),
            ):
                if hasattr(env.simulator, attr_name):
                    step_trace[key] = getattr(env.simulator, attr_name)[0].detach().cpu()
            hand_sensor = env.simulator.scene.sensors.get("hand_frame_transformer")
            if hand_sensor is not None:
                step_trace.update(
                    {
                        "hand_source_pos_w": hand_sensor.data.source_pos_w[0].detach().cpu(),
                        "hand_source_quat_wxyz": hand_sensor.data.source_quat_w[0].detach().cpu(),
                        "grasp_target_pos_w": hand_sensor.data.target_pos_w[0].detach().cpu(),
                        "grasp_target_quat_wxyz": hand_sensor.data.target_quat_w[0].detach().cpu(),
                    }
                )
            if hasattr(env, "_lowlevel_policy_joint_indices"):
                policy_ids = env._lowlevel_policy_joint_indices
                step_trace["policy_dof_pos"] = env.simulator.dof_pos[0, policy_ids].detach().cpu()
                step_trace["policy_dof_vel"] = env.simulator.dof_vel[0, policy_ids].detach().cpu()
                step_trace["policy_torques"] = env.torques[0, policy_ids].detach().cpu()
                step_trace["policy_actions_after_delay"] = (
                    env.actions_after_delay[0, policy_ids].detach().cpu()
                )
            for attr_name in (
                "_debug_lowlevel_prop_obs",
                "_debug_lowlevel_full_obs",
                "_debug_lowlevel_actions",
                "_debug_lowlevel_env_actions",
                "_debug_lowlevel_env_torques",
                "_lowlevel_prev_policy_actions",
            ):
                if hasattr(env, attr_name):
                    step_trace[attr_name] = getattr(env, attr_name)[0].detach().cpu()
            trace["steps"][step] = step_trace
        reset_count += int(reset[0].item())
        root_w = env.simulator._robot.data.root_pos_w[0].detach().cpu()
        base_body_w = env.simulator._rigid_body_pos[0, 0].detach().cpu()
        final_root_w = root_w.clone()
        if initial_root_w is None:
            initial_root_w = root_w.clone()
        if step % 50 == 0:
            cmd = env.get_physical_b2z1_commands()[0].detach().cpu().tolist()
            logger.info(
                "step={} root_w={} base_body_w={} reset={} command={} reward={}".format(
                    step,
                    root_w.tolist(),
                    base_body_w.tolist(),
                    bool(reset[0].item()),
                    cmd,
                    float(rew[0]),
                )
            )
        if bool(reset[0].item()):
            cmd = env.get_physical_b2z1_commands()[0].detach().cpu().tolist()
            reset_details = {
                "step": step,
                "root_w": root_w.tolist(),
                "base_body_w": base_body_w.tolist(),
                "command": cmd,
                "reward": float(rew[0]),
            }
            for attr_name in (
                "episode_length_buf",
                "time_out_buf",
                "stage_buf",
                "time_in_stage_buf",
                "relative_door_pos_buf",
            ):
                if hasattr(env, attr_name):
                    value = getattr(env, attr_name)[0].detach().cpu()
                    reset_details[attr_name] = value.tolist() if value.ndim > 0 else value.item()
            logger.info(f"reset_details={reset_details}")
        if simulation_app is not None and not simulation_app.is_running():
            break

    if initial_root_w is not None and final_root_w is not None:
        delta = final_root_w - initial_root_w
        logger.info(
            "B2Z1 low-level summary: initial_root_w={} final_root_w={} delta_root_w={} reset_count={}".format(
                initial_root_w.tolist(),
                final_root_w.tolist(),
                delta.tolist(),
                reset_count,
            )
        )

    env.end_render_results()
    if skeleton_video_path is not None and skeleton_frames:
        video_fps = max(1, int(1 / env.dt / skeleton_stride))
        write_skeleton_video(
            str(skeleton_video_path),
            list(env.simulator.body_names),
            skeleton_frames,
            video_fps,
        )
    if trace_path is not None:
        trace_file = Path(str(trace_path))
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trace, trace_file)
        logger.info(f"Wrote B2Z1 debug trace to {trace_file}")

    if no_render_headless:
        logger.info("Completed B2Z1 low-level run; skipping Kit shutdown in no-render mode.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
