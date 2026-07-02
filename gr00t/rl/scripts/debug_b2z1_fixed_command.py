# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run B2Z1 door env with a fixed 10D high-level command.

Example:

    ./isaac-sim/python.sh gr00t/rl/scripts/debug_b2z1_fixed_command.py \
        +exp=wbmanip/door_open_b2z1_lstm \
        num_envs=1 headless=False \
        +fixed_command='[0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0]' \
        +num_debug_steps=1000
"""

import argparse
import os
import site
import shutil
import sys
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import OmegaConf, open_dict

from gr00t.rl.utils.config_utils import register_rl_resolvers

register_rl_resolvers()


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
    isaaclab_source = Path("/workspace/IsaacLab/source")
    if isaaclab_source.is_dir():
        folders.append(str(isaaclab_source))
    return folders


def isaacsim_kit_args() -> str:
    args = []
    for folder in isaacsim_ext_folders():
        args.extend(["--ext-folder", folder])
    args.extend(
        [
            "--/app/vulkan=false",
            "--/renderer/enabled=none",
            "--/renderer/active=none",
            "--/renderer/multiGpu/enabled=false",
            "--/renderer/multiGpu/autoEnable=false",
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
        args_cli.kit_args = " ".join(part for part in (args_cli.kit_args, isaacsim_kit_args()) if part)

        if args_cli.enable_cameras and config.headless:
            dest_path = Path(isaaclab.__file__).resolve().parent.parent.parent.parent / "apps"
            source_file = (
                Path(__file__).resolve().parents[1]
                / "apps/phc.isaaclab.python.headless.rendering.kit"
            )
            shutil.copy(source_file, dest_path)
            args_cli.experience = dest_path / "phc.isaaclab.python.headless.rendering.kit"
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

    fixed_command = torch.tensor(
        config.get("fixed_command", [0.0] * 10),
        device=env.device,
        dtype=torch.float32,
    ).view(1, 10)
    actions = fixed_command.repeat(env.num_envs, 1)
    num_steps = int(config.get("num_debug_steps", 1000))

    logger.info(f"Running B2Z1 fixed command for {num_steps} steps: {fixed_command[0].tolist()}")
    initial_root_w = None
    final_root_w = None
    reset_count = 0
    skeleton_video_path = config.get("skeleton_video_path", None)
    skeleton_stride = int(config.get("skeleton_video_stride", 2))
    skeleton_frames = []
    trace_path = config.get("trace_path", None)
    trace_steps = set(int(x) for x in config.get("trace_steps", [0, 1, 2, 3, 4, 5, 10, 20, 50]))
    trace = {"meta": {}, "steps": {}} if trace_path is not None else None
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
        obs, rew, reset, extras = env.step({"actions": actions.clone()})
        env.render_results()
        if skeleton_video_path is not None and step % skeleton_stride == 0:
            skeleton_frames.append(env.simulator._rigid_body_pos[0].detach().cpu().clone())
        if trace is not None and step in trace_steps:
            step_trace = {
                "commands": env.get_physical_b2z1_commands()[0].detach().cpu(),
                "dof_pos": env.simulator.dof_pos[0].detach().cpu(),
                "dof_vel": env.simulator.dof_vel[0].detach().cpu(),
                "base_lin_vel": env.base_lin_vel[0].detach().cpu(),
                "base_ang_vel": env.base_ang_vel[0].detach().cpu(),
                "base_quat": env.base_quat[0].detach().cpu(),
                "root_state": env.simulator.robot_root_states[0].detach().cpu(),
                "rigid_body_pos": env.simulator._rigid_body_pos[0].detach().cpu(),
                "torques": env.torques[0].detach().cpu(),
                "actions_after_delay": env.actions_after_delay[0].detach().cpu(),
            }
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
            "B2Z1 fixed-command summary: initial_root_w={} final_root_w={} delta_root_w={} reset_count={}".format(
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
        logger.info("Completed B2Z1 fixed-command debug run; skipping Kit shutdown in no-render mode.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
