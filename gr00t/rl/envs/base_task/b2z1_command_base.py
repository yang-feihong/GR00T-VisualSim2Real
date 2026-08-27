# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn
from loguru import logger
from typing_extensions import override

from gr00t.rl.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from gr00t.rl.isaac_utils.rotations import get_euler_xyz_in_tensor
from gr00t.rl.utils.torch_utils import normalize, quat_conjugate, quat_mul, quat_rotate


def _activation(name):
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported B2Z1 low-level activation: {name}")


class _StateHistoryEncoder(nn.Module):
    def __init__(self, activation, input_size, tsteps, output_size):
        super().__init__()
        if tsteps != 10:
            raise ValueError("The bundled B2Z1 checkpoint uses history_len=10.")
        channel_size = 10
        self.tsteps = tsteps
        self.encoder = nn.Sequential(nn.Linear(input_size, 3 * channel_size), activation)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=4, stride=2),
            activation,
            nn.Conv1d(2 * channel_size, channel_size, kernel_size=2, stride=1),
            activation,
            nn.Flatten(),
        )
        self.linear_output = nn.Sequential(nn.Linear(channel_size * 3, output_size), activation)

    def forward(self, obs):
        nd = obs.shape[0]
        projection = self.encoder(obs.reshape(nd * self.tsteps, -1))
        output = self.conv_layers(projection.reshape(nd, self.tsteps, -1).permute(0, 2, 1))
        return self.linear_output(output)


class _B2Z1LowLevelActor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        act = _activation(str(cfg.get("activation", "elu")))
        num_prop = int(cfg.get("num_proprio", 66))
        num_priv = int(cfg.get("num_priv", 18))
        num_hist = int(cfg.get("history_len", 10))
        priv_dims = list(cfg.get("priv_encoder_dims", [64, 20]))
        actor_dims = list(cfg.get("actor_hidden_dims", [512, 256, 128]))
        leg_head_dims = list(cfg.get("leg_control_head_hidden_dims", [128, 128]))
        arm_head_dims = list(cfg.get("arm_control_head_hidden_dims", [128, 128]))
        num_leg_actions = int(cfg.get("num_leg_actions", 12))
        num_arm_actions = int(cfg.get("num_arm_actions", 6))
        output_tanh = bool(cfg.get("output_tanh", False))

        self.num_prop = num_prop
        self.num_priv = num_priv
        self.num_hist = num_hist
        self.latent_mode = str(cfg.get("latent_mode", "history"))
        if self.latent_mode not in ("privileged", "history"):
            raise ValueError(
                f"Unsupported B2Z1 low-level latent_mode: {self.latent_mode}"
            )

        priv_layers = []
        in_dim = num_priv
        for out_dim in priv_dims:
            priv_layers += [nn.Linear(in_dim, out_dim), act]
            in_dim = out_dim
        self.priv_encoder = nn.Sequential(*priv_layers)
        latent_dim = priv_dims[-1]
        self.history_encoder = _StateHistoryEncoder(act, num_prop, num_hist, latent_dim)

        backbone_layers = []
        in_dim = num_prop + latent_dim
        for out_dim in actor_dims:
            backbone_layers += [nn.Linear(in_dim, out_dim), act]
            in_dim = out_dim
        self.actor_backbone = nn.Sequential(*backbone_layers)

        self.actor_leg_control_head = self._make_head(
            actor_dims[-1], leg_head_dims, num_leg_actions, act, output_tanh
        )
        self.actor_arm_control_head = self._make_head(
            actor_dims[-1], arm_head_dims, num_arm_actions, act, output_tanh
        )

    @staticmethod
    def _make_head(input_dim, hidden_dims, output_dim, activation, output_tanh):
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers += [nn.Linear(in_dim, out_dim), activation]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))
        if output_tanh:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist * self.num_prop :]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))

    def forward(self, obs, hist_encoding=None):
        obs_prop = obs[:, : self.num_prop]
        use_history = self.latent_mode == "history" if hist_encoding is None else bool(hist_encoding)
        if use_history:
            latent = self.infer_hist_latent(obs)
        else:
            obs_priv = obs[:, self.num_prop : self.num_prop + self.num_priv]
            latent = self.priv_encoder(obs_priv)
        backbone_input = torch.cat([obs_prop, latent], dim=1)
        backbone_output = self.actor_backbone(backbone_input)
        return torch.cat(
            [
                self.actor_leg_control_head(backbone_output),
                self.actor_arm_control_head(backbone_output),
            ],
            dim=-1,
        )


class _B2Z1LowLevelPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.actor = _B2Z1LowLevelActor(cfg)

    def forward(self, obs, hist_encoding=None):
        return self.actor(obs, hist_encoding=hist_encoding)


class B2Z1CommandBase(LeggedRobotBase):
    """Bridge the high-level door policy to the B2Z1 whole-body controller.

    The high-level policy owns a compact 10D command:
      [vx, vy, yaw_rate, ee_dx, ee_dy, ee_dz, ee_droll, ee_dpitch, ee_dyaw, gripper]

    The simulator still receives full robot joint targets, so this class is the
    only place where the 10D command is expanded through the B2Z1 low-level
    whole-body checkpoint and IK gripper/arm targets.
    """

    COMMAND_DIM = 10

    @property
    def ground_height(self):
        return 0.0

    def __init__(self, config, device):
        self._lowlevel_policy = None
        self._num_lowlevel_actions = int(config.robot.get("lowlevel_actions_dim", 18))
        self._num_joint_actions = int(config.robot.get("joint_actions_dim", config.robot.dof_obs_size))
        super().__init__(config, device)

        self._command_scale = torch.tensor(
            self.config.robot.b2z1_command.command_scale,
            device=self.device,
            dtype=torch.float32,
        ).view(1, self.COMMAND_DIM)
        self._command_clip = torch.tensor(
            self.config.robot.b2z1_command.command_clip,
            device=self.device,
            dtype=torch.float32,
        ).view(1, self.COMMAND_DIM)
        self._command_lower = torch.tensor(
            self.config.robot.b2z1_command.get(
                "command_lower", (-self._command_clip).squeeze(0).tolist()
            ),
            device=self.device,
            dtype=torch.float32,
        ).view(1, self.COMMAND_DIM)
        self._command_upper = torch.tensor(
            self.config.robot.b2z1_command.get(
                "command_upper", self._command_clip.squeeze(0).tolist()
            ),
            device=self.device,
            dtype=torch.float32,
        ).view(1, self.COMMAND_DIM)
        if torch.any(self._command_lower >= self._command_upper):
            raise ValueError("B2Z1 command_lower must be smaller than command_upper.")
        self._b2z1_gripper_open_target = float(
            self.config.robot.b2z1_command.gripper_open_pos
        )
        self._b2z1_gripper_closed_target = float(
            self.config.robot.b2z1_command.gripper_closed_pos
        )
        self._b2z1_gripper_open_command = float(
            self.config.robot.b2z1_command.get("gripper_open_command", -1.0)
        )
        self._b2z1_gripper_closed_command = float(
            self.config.robot.b2z1_command.get("gripper_closed_command", 1.0)
        )
        if self._b2z1_gripper_open_command >= self._b2z1_gripper_closed_command:
            raise ValueError("B2Z1 gripper open command must be smaller than closed command.")
        if self._b2z1_gripper_open_target == self._b2z1_gripper_closed_target:
            raise ValueError("B2Z1 gripper open and closed targets must differ.")
        if not (
            self._command_lower[0, 9] <= self._b2z1_gripper_open_command
            and self._b2z1_gripper_closed_command <= self._command_upper[0, 9]
        ):
            raise ValueError("B2Z1 command bounds must contain both gripper commands.")
        self._load_lowlevel_policy()

    @override
    def _init_buffers(self, reinit_sim=False):
        super()._init_buffers(reinit_sim=reinit_sim)
        self.actions = torch.zeros(
            self.num_envs,
            self._num_joint_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions_after_delay = torch.zeros_like(self.actions)
        self.last_actions = torch.zeros_like(self.actions)
        self._b2z1_commands = torch.zeros(
            self.num_envs, self.COMMAND_DIM, device=self.device, requires_grad=False
        )
        self._b2z1_commands_unclipped = torch.zeros_like(self._b2z1_commands)
        self._last_b2z1_commands = torch.zeros_like(self._b2z1_commands)
        self._b2z1_joint_actions = torch.zeros(
            self.num_envs, self._num_joint_actions, device=self.device, requires_grad=False
        )
        cmd_cfg = self.config.robot.b2z1_command
        self._lowlevel_num_prop = int(cmd_cfg.get("lowlevel_num_proprio", 66))
        self._lowlevel_num_priv = int(cmd_cfg.get("lowlevel_num_priv", 0))
        self._lowlevel_history_len = int(cmd_cfg.get("lowlevel_history_len", 10))
        self._lowlevel_obs_clip = float(cmd_cfg.get("lowlevel_obs_clip", 100.0))
        self._lowlevel_action_clip = float(cmd_cfg.get("lowlevel_action_clip", 100.0))
        self._lowlevel_obs_history = torch.zeros(
            self.num_envs,
            self._lowlevel_history_len,
            self._lowlevel_num_prop,
            device=self.device,
            requires_grad=False,
        )
        self._lowlevel_prev_policy_actions = torch.zeros(
            self.num_envs, self._num_lowlevel_actions, device=self.device, requires_grad=False
        )
        self._lowlevel_privileged_obs = torch.tensor(
            cmd_cfg.get("lowlevel_privileged_obs", [0.0] * self._lowlevel_num_priv),
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        if self._lowlevel_privileged_obs.shape[-1] != self._lowlevel_num_priv:
            raise ValueError(
                "B2Z1 lowlevel_privileged_obs must have "
                f"{self._lowlevel_num_priv} values, got "
                f"{self._lowlevel_privileged_obs.shape[-1]}."
            )
        self._lowlevel_gait_indices = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self._lowlevel_clock_inputs = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self._lowlevel_gait_frequencies = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self._lowlevel_gait_transition_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self._lowlevel_gait_transition_elapsed = torch.zeros_like(
            self._lowlevel_gait_frequencies
        )
        self._lowlevel_gait_transition_start_frequency = torch.zeros_like(
            self._lowlevel_gait_frequencies
        )
        self._lowlevel_gait_transition_target_frequency = torch.zeros_like(
            self._lowlevel_gait_frequencies
        )
        fixed_frequency = torch.full_like(
            self._lowlevel_gait_frequencies,
            float(cmd_cfg.get("lowlevel_fixed_trot_frequency", 1.5)),
        )
        self._lowlevel_gait_stance_durations = self._get_lowlevel_gait_stance_duration(
            fixed_frequency
        )
        self._lowlevel_gait_transition_start_stance = (
            self._lowlevel_gait_stance_durations.clone()
        )
        self._lowlevel_gait_transition_target_stance = (
            self._lowlevel_gait_stance_durations.clone()
        )
        self._lowlevel_gait_pair_phases = torch.zeros(
            self.num_envs, 2, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self._lowlevel_command_obs_scale = torch.tensor(
            cmd_cfg.get("lowlevel_command_obs_scale", [2.0, 2.0, 0.25]),
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        self._lowlevel_velocity_command_lower = torch.tensor(
            cmd_cfg.get("lowlevel_velocity_command_lower", [-0.8, -0.8, -1.0]),
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        self._lowlevel_velocity_command_upper = torch.tensor(
            cmd_cfg.get("lowlevel_velocity_command_upper", [0.8, 0.8, 1.0]),
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        self._lowlevel_ang_vel_scale = float(cmd_cfg.get("lowlevel_ang_vel_scale", 0.25))
        self._lowlevel_dof_pos_scale = float(cmd_cfg.get("lowlevel_dof_pos_scale", 1.0))
        self._lowlevel_dof_vel_scale = float(cmd_cfg.get("lowlevel_dof_vel_scale", 0.05))
        policy_joint_names = list(
            cmd_cfg.get("lowlevel_policy_joint_names", self.config.robot.lower_dof_names)
        )
        self._lowlevel_policy_joint_indices = torch.tensor(
            [self.dof_names.index(name) for name in policy_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        env_joint_names = list(cmd_cfg.get("lowlevel_env_joint_names", policy_joint_names))
        self._lowlevel_env_joint_indices = torch.tensor(
            [self.dof_names.index(name) for name in env_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        policy_name_to_pos = {name: i for i, name in enumerate(policy_joint_names)}
        self._lowlevel_env_indices_in_policy = torch.tensor(
            [policy_name_to_pos[name] for name in env_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self._lowlevel_env_action_scale = torch.tensor(
            cmd_cfg.get("lowlevel_action_scale", self.config.robot.control.action_scale[: len(env_joint_names)]),
            device=self.device,
            dtype=torch.float32,
        ).view(1, len(env_joint_names))
        self._lowlevel_policy_action_scale = torch.tensor(
            cmd_cfg.get(
                "lowlevel_action_scale",
                self.config.robot.control.action_scale[: len(policy_joint_names)],
            ),
            device=self.device,
            dtype=torch.float32,
        ).view(1, len(policy_joint_names))
        leg_joint_names = [
            name for name in self.dof_names if name.endswith(("hip_joint", "thigh_joint", "calf_joint"))
        ]
        self._b2z1_leg_joint_indices = torch.tensor(
            [self.dof_names.index(name) for name in leg_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self._b2z1_leg_sim_joint_ids = torch.tensor(
            [self.simulator.dof_ids[int(idx)] for idx in self._b2z1_leg_joint_indices],
            device=self.device,
            dtype=torch.long,
        )
        arm_joint_names = list(cmd_cfg.get("lowlevel_arm_joint_names", self.config.robot.arm_dof_names))
        self._b2z1_arm_joint_indices = torch.tensor(
            [self.dof_names.index(name) for name in arm_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self._b2z1_arm_sim_joint_ids = torch.tensor(
            [self.simulator.dof_ids[int(idx)] for idx in self._b2z1_arm_joint_indices],
            device=self.device,
            dtype=torch.long,
        )
        self._b2z1_gripper_joint_index = self.dof_names.index(cmd_cfg.gripper_joint_name)
        self._b2z1_gripper_sim_joint_id = torch.tensor(
            [self.simulator.dof_ids[int(self._b2z1_gripper_joint_index)]],
            device=self.device,
            dtype=torch.long,
        )
        self._b2z1_arm_gripper_joint_indices = torch.cat(
            [
                self._b2z1_arm_joint_indices,
                torch.tensor(
                    [self._b2z1_gripper_joint_index],
                    device=self.device,
                    dtype=torch.long,
                ),
            ]
        )
        self._b2z1_arm_gripper_sim_joint_ids = torch.cat(
            [self._b2z1_arm_sim_joint_ids, self._b2z1_gripper_sim_joint_id]
        )
        self._b2z1_gripper_body_index = self.body_names.index(self.config.robot.gripper_body_name)
        self._b2z1_gripper_sim_body_id = int(
            self.simulator.body_ids[self._b2z1_gripper_body_index]
        )
        self._b2z1_arm_pos_targets = torch.zeros(
            self.num_envs, len(arm_joint_names), device=self.device, requires_grad=False
        )
        self._b2z1_arm_pos_targets.copy_(
            self.default_dof_pos[:, self._b2z1_arm_joint_indices]
        )
        self._b2z1_ik_damping = float(cmd_cfg.get("lowlevel_ik_damping", 0.05))
        self._b2z1_disable_arm_ik = bool(cmd_cfg.get("lowlevel_disable_arm_ik", False))
        self._b2z1_leg_control_mode = str(cmd_cfg.get("lowlevel_leg_control_mode", "effort"))
        self._b2z1_goal_center_offset = torch.tensor(
            cmd_cfg.get("lowlevel_goal_center_offset", [0.2, 0.0, 0.7]),
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        self._b2z1_default_ee_goal_cart = torch.tensor(
            cmd_cfg.get("lowlevel_default_ee_goal_cart", [0.4619397663, 0.0, 0.1913417162]),
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        self._b2z1_arm_induced_pitch = float(cmd_cfg.get("lowlevel_arm_induced_pitch", 0.38))
        self._b2z1_ee_goal_sphere_lower = torch.tensor(
            cmd_cfg.get(
                "lowlevel_ee_goal_sphere_lower",
                [0.4, -1.4959965017, -1.5707963268],
            ),
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        self._b2z1_ee_goal_sphere_upper = torch.tensor(
            cmd_cfg.get(
                "lowlevel_ee_goal_sphere_upper",
                [0.95, 1.0471975512, 1.5707963268],
            ),
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        if torch.any(self._b2z1_ee_goal_sphere_lower >= self._b2z1_ee_goal_sphere_upper):
            raise ValueError(
                "B2Z1 lowlevel_ee_goal_sphere_lower must be smaller than the upper bound."
            )
        self._b2z1_ee_pos = None
        self._b2z1_ee_orn = None
        self._b2z1_ee_j_eef = None

    def _load_lowlevel_policy(self):
        policy_path = self.config.robot.b2z1_command.get("lowlevel_policy_path", None)
        if not policy_path:
            raise RuntimeError("B2Z1 low-level policy path must point to model_*.pt.")
        if not os.path.isfile(policy_path):
            raise FileNotFoundError(f"B2Z1 low-level policy not found: {policy_path}")

        checkpoint = torch.load(policy_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise RuntimeError(
                f"Unsupported B2Z1 low-level checkpoint format: {policy_path}. "
                "Expected an RSL-RL training checkpoint with model_state_dict."
            )

        state_dict = checkpoint["model_state_dict"]
        actor_state_dict = {
            key: value for key, value in state_dict.items() if key.startswith("actor.")
        }
        policy_cfg = self.config.robot.b2z1_command.get("lowlevel_policy_model", {})
        self._lowlevel_policy = _B2Z1LowLevelPolicy(policy_cfg).to(self.device)
        self._lowlevel_policy.load_state_dict(actor_state_dict, strict=True)
        self._lowlevel_policy.eval()
        logger.info(f"Loaded B2Z1 low-level actor checkpoint: {policy_path}")

    @override
    def step(self, actor_state):
        highlevel_actions = actor_state["actions"]
        if highlevel_actions.shape[-1] != self.COMMAND_DIM:
            raise ValueError(
                f"B2Z1 high-level action must be {self.COMMAND_DIM}D, got {highlevel_actions.shape[-1]}"
            )

        highlevel_action_clip = float(self.config.robot.control.action_clip_value)
        highlevel_actions = torch.clamp(
            highlevel_actions, -highlevel_action_clip, highlevel_action_clip
        )
        self.log_dict["highlevel_action_clip_frac"] = (
            highlevel_actions.abs() == highlevel_action_clip
        ).sum() / highlevel_actions.numel()
        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = highlevel_actions
            highlevel_actions = self.action_queue[
                torch.arange(self.num_envs, device=self.device), self.action_delay_idx
            ]

        return self._step_b2z1_commands(
            highlevel_actions * self._command_scale,
            self._command_lower,
            self._command_upper,
        )

    def step_physical_b2z1_commands(self, physical_commands):
        """Step with B2Z1 deployment commands in physical units."""
        if physical_commands.shape[-1] != self.COMMAND_DIM:
            raise ValueError(
                f"B2Z1 physical command must be {self.COMMAND_DIM}D, "
                f"got {physical_commands.shape[-1]}"
            )

        command_lower = self._command_lower.clone()
        command_upper = self._command_upper.clone()
        command_lower[:, :3] = self._lowlevel_velocity_command_lower
        command_upper[:, :3] = self._lowlevel_velocity_command_upper
        return self._step_b2z1_commands(physical_commands, command_lower, command_upper)

    def _step_b2z1_commands(self, physical_commands, command_lower, command_upper):
        self._last_b2z1_commands[:] = self._b2z1_commands
        self._b2z1_commands_unclipped[:] = physical_commands
        self._b2z1_commands[:] = torch.maximum(
            torch.minimum(self._b2z1_commands_unclipped, command_upper),
            command_lower,
        )
        self._project_b2z1_ee_command_to_workspace()
        self._b2z1_joint_actions[:] = self._compute_b2z1_joint_actions()

        self._pre_physics_step(self._b2z1_joint_actions)
        self._physics_step()
        self._post_physics_step()
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _compute_b2z1_joint_actions(self):
        if self._lowlevel_policy is None:
            raise RuntimeError("B2Z1 low-level policy was not loaded.")

        lowlevel_obs = self._build_lowlevel_obs()
        with torch.no_grad():
            lowlevel_actions = self._lowlevel_policy(lowlevel_obs, hist_encoding=True)
        lowlevel_actions = torch.clamp(
            lowlevel_actions[:, : self._num_lowlevel_actions],
            -self._lowlevel_action_clip,
            self._lowlevel_action_clip,
        )
        lowlevel_actions[:, 12:] = 0.0
        self._debug_lowlevel_actions = lowlevel_actions.detach().clone()
        self._lowlevel_prev_policy_actions[:] = lowlevel_actions
        joint_actions = torch.zeros(
            self.num_envs, self._num_joint_actions, device=self.device, requires_grad=False
        )
        joint_actions[:, self._lowlevel_policy_joint_indices] = lowlevel_actions

        joint_actions[:, self._b2z1_gripper_joint_index] = (
            self._b2z1_commands[:, 9]
        )
        return joint_actions

    def _build_lowlevel_obs(self):
        """Build the ManipLoco deployment observation for the current checkpoint."""

        obs_dim = int(self.config.robot.b2z1_command.get("lowlevel_policy_obs_dim", 0))
        cmd_cfg = self.config.robot.b2z1_command
        self._step_lowlevel_gait_clock()
        rpy = get_euler_xyz_in_tensor(self.base_quat)
        policy_joint_ids = self._lowlevel_policy_joint_indices
        dof_pos_obs = (
            self.simulator.dof_pos[:, policy_joint_ids] - self.default_dof_pos[:, policy_joint_ids]
        ) * self._lowlevel_dof_pos_scale
        dof_vel_obs = self.simulator.dof_vel[:, policy_joint_ids] * self._lowlevel_dof_vel_scale
        if bool(cmd_cfg.get("lowlevel_observe_foot_contacts", False)):
            if bool(cmd_cfg.get("lowlevel_zero_observed_foot_contacts", False)):
                foot_contacts = torch.zeros(self.num_envs, 4, device=self.device)
            else:
                contact_threshold = float(
                    cmd_cfg.get("lowlevel_foot_contact_force_threshold", 1.5)
                )
                foot_contacts = (
                    torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1)
                    > contact_threshold
                ).to(dtype=torch.float32)
                # ContactSensor retains its previous force cache across an articulation reset.
                foot_contacts[self.episode_length_buf <= 1] = 0.0
        else:
            foot_contacts = torch.zeros(self.num_envs, 4, device=self.device)
        obs_buf = torch.cat(
            [
                rpy[:, :2],
                self.base_ang_vel * self._lowlevel_ang_vel_scale,
                dof_pos_obs,
                dof_vel_obs,
                self._lowlevel_prev_policy_actions[:, :12],
                foot_contacts,
                self._b2z1_commands[:, :3] * self._lowlevel_command_obs_scale,
                self._get_b2z1_ee_goal_cart(),
                torch.zeros(self.num_envs, 3, device=self.device),
            ],
            dim=-1,
        )
        if bool(cmd_cfg.get("lowlevel_observe_gait_commands", False)):
            obs_buf = torch.cat(
                [
                    obs_buf,
                    self._lowlevel_gait_indices.unsqueeze(1),
                    self._lowlevel_clock_inputs,
                ],
                dim=-1,
            )
        if obs_buf.shape[-1] != self._lowlevel_num_prop:
            raise RuntimeError(
                f"B2Z1 low-level proprio dim mismatch: got {obs_buf.shape[-1]}, "
                f"expected {self._lowlevel_num_prop}"
            )

        full_obs_terms = [obs_buf]
        if self._lowlevel_num_priv > 0:
            full_obs_terms.append(self._lowlevel_privileged_obs.expand(self.num_envs, -1))
        full_obs_terms.append(self._lowlevel_obs_history.view(self.num_envs, -1))
        full_obs = torch.cat(full_obs_terms, dim=-1)
        self._debug_lowlevel_prop_obs = obs_buf.detach().clone()
        self._debug_lowlevel_full_obs = full_obs.detach().clone()
        if obs_dim > 0 and full_obs.shape[-1] != obs_dim:
            raise RuntimeError(
                f"B2Z1 low-level obs dim mismatch: got {full_obs.shape[-1]}, expected {obs_dim}"
            )

        reset_mask = self.episode_length_buf <= 1
        if torch.any(~reset_mask):
            self._lowlevel_obs_history[~reset_mask] = torch.cat(
                [
                    self._lowlevel_obs_history[~reset_mask, 1:],
                    obs_buf[~reset_mask].unsqueeze(1),
                ],
                dim=1,
            )
        if torch.any(reset_mask):
            self._lowlevel_obs_history[reset_mask] = obs_buf[reset_mask].unsqueeze(1).repeat(
                1, self._lowlevel_history_len, 1
            )
        return torch.clamp(full_obs, -self._lowlevel_obs_clip, self._lowlevel_obs_clip)

    def _step_lowlevel_gait_clock(self):
        cmd_cfg = self.config.robot.b2z1_command
        if not bool(cmd_cfg.get("lowlevel_observe_gait_commands", False)):
            return

        frequency = float(cmd_cfg.get("lowlevel_fixed_trot_frequency", 2.0))
        lin_x_clip = float(cmd_cfg.get("lowlevel_lin_vel_x_clip", 0.2))
        lin_y_clip = float(cmd_cfg.get("lowlevel_lin_vel_y_clip", 0.2))
        yaw_clip = float(cmd_cfg.get("lowlevel_ang_vel_yaw_clip", 0.5))
        is_walking = (
            (torch.abs(self._b2z1_commands[:, 0]) > lin_x_clip)
            | (torch.abs(self._b2z1_commands[:, 1]) > lin_y_clip)
            | (torch.abs(self._b2z1_commands[:, 2]) > yaw_clip)
        )
        target_frequencies = torch.where(
            is_walking,
            torch.full_like(self._lowlevel_gait_frequencies, frequency),
            torch.zeros_like(self._lowlevel_gait_frequencies),
        )

        active = self._lowlevel_gait_transition_active
        target_changed = active & (
            torch.abs(target_frequencies - self._lowlevel_gait_transition_target_frequency)
            > 1.0e-6
        )
        changed_to_stop = target_changed & (target_frequencies <= 1.0e-6)
        active[changed_to_stop] = False
        self._lowlevel_gait_transition_elapsed[changed_to_stop] = 0.0
        self._start_lowlevel_gait_transition(target_changed & ~changed_to_stop, target_frequencies)

        handled = active.clone()
        self._advance_lowlevel_gait_transition(handled)
        pending = (~handled) & (
            torch.abs(target_frequencies - self._lowlevel_gait_frequencies) > 1.0e-6
        )
        stopping = pending & (target_frequencies <= 1.0e-6)
        starting = pending & ~stopping
        self._start_lowlevel_gait_transition(starting, target_frequencies)
        self._advance_lowlevel_gait_transition(starting)
        handled |= starting

        self._refresh_lowlevel_gait_pair_phases()
        both_stance = torch.all(self._lowlevel_gait_pair_phases <= 0.5, dim=1)
        stop_now = stopping & both_stance
        self._set_lowlevel_fixed_trot_stopped(stop_now)
        handled |= stop_now

        waiting = stopping & ~both_stance
        if torch.any(waiting):
            waiting_indices = self._lowlevel_gait_indices[waiting]
            waiting_stance = self._lowlevel_gait_stance_durations[waiting]
            waiting_frequency = self._lowlevel_gait_frequencies[waiting].clamp(min=1.0e-6)
            first_pair_swing = (waiting_indices > waiting_stance - 0.5) & (
                waiting_indices < 0.5
            )
            time_to_stance = torch.where(
                first_pair_swing,
                (0.5 - waiting_indices) / waiting_frequency,
                (1.0 - waiting_indices) / waiting_frequency,
            )
            crossing_local = time_to_stance <= self.dt
            if torch.any(crossing_local):
                waiting_ids = torch.where(waiting)[0]
                crossing_ids = waiting_ids[crossing_local]
                crossing = torch.zeros_like(waiting)
                crossing[crossing_ids] = True
                self._advance_lowlevel_gait_constant(
                    crossing, time_to_stance[crossing_local]
                )
                self._set_lowlevel_fixed_trot_stopped(crossing)
                handled |= crossing

        self._advance_lowlevel_gait_constant(~handled, self.dt)
        self._refresh_lowlevel_gait_pair_phases()
        phases = torch.stack(
            [
                self._lowlevel_gait_pair_phases[:, 0],
                self._lowlevel_gait_pair_phases[:, 1],
                self._lowlevel_gait_pair_phases[:, 1],
                self._lowlevel_gait_pair_phases[:, 0],
            ],
            dim=1,
        )
        self._lowlevel_clock_inputs[:] = torch.sin(2.0 * torch.pi * phases)
        stopped = target_frequencies <= 1.0e-6
        stopped &= self._lowlevel_gait_frequencies <= 1.0e-6
        self._set_lowlevel_fixed_trot_stopped(stopped)

    def _get_lowlevel_gait_stance_duration(self, frequencies):
        cmd_cfg = self.config.robot.b2z1_command
        stance_ratio = max(float(cmd_cfg.get("lowlevel_trot_stance_ratio", 0.5)), 1.0e-6)
        swing_ratio = max(float(cmd_cfg.get("lowlevel_trot_swing_ratio", 0.5)), 1.0e-6)
        relative_swing = swing_ratio / (stance_ratio + swing_ratio)
        max_swing_s = max(
            float(cmd_cfg.get("lowlevel_trot_swing_duration_max_s", 0.4)), 0.0
        )
        if max_swing_s <= 0.0:
            swing_duration = torch.full_like(frequencies, relative_swing)
        else:
            swing_duration = torch.minimum(
                torch.full_like(frequencies, relative_swing),
                torch.clamp(frequencies * max_swing_s, min=0.0, max=1.0),
            )
        return 1.0 - swing_duration

    def _start_lowlevel_gait_transition(self, mask, target_frequencies):
        if not torch.any(mask):
            return
        self._lowlevel_gait_transition_active[mask] = True
        self._lowlevel_gait_transition_elapsed[mask] = 0.0
        self._lowlevel_gait_transition_start_frequency[mask] = (
            self._lowlevel_gait_frequencies[mask]
        )
        self._lowlevel_gait_transition_target_frequency[mask] = target_frequencies[mask]
        self._lowlevel_gait_transition_start_stance[mask] = (
            self._lowlevel_gait_stance_durations[mask]
        )
        self._lowlevel_gait_transition_target_stance[mask] = (
            self._get_lowlevel_gait_stance_duration(target_frequencies[mask])
        )

    def _advance_lowlevel_gait_transition(self, mask):
        if not torch.any(mask):
            return
        duration = max(
            float(
                self.config.robot.b2z1_command.get(
                    "lowlevel_gait_transition_duration_s", 0.5
                )
            ),
            self.dt,
        )
        previous_frequency = self._lowlevel_gait_frequencies[mask]
        elapsed = self._lowlevel_gait_transition_elapsed[mask] + self.dt
        progress = torch.clamp(elapsed / duration, 0.0, 1.0)
        frequencies = torch.lerp(
            self._lowlevel_gait_transition_start_frequency[mask],
            self._lowlevel_gait_transition_target_frequency[mask],
            progress,
        )
        mean_frequency = 0.5 * (previous_frequency + frequencies)
        self._lowlevel_gait_indices[mask] = torch.remainder(
            self._lowlevel_gait_indices[mask] + mean_frequency * self.dt, 1.0
        )
        self._lowlevel_gait_frequencies[mask] = frequencies
        self._lowlevel_gait_stance_durations[mask] = torch.lerp(
            self._lowlevel_gait_transition_start_stance[mask],
            self._lowlevel_gait_transition_target_stance[mask],
            progress,
        )
        self._lowlevel_gait_transition_elapsed[mask] = elapsed
        completed_local = progress >= 1.0
        if torch.any(completed_local):
            completed_ids = torch.where(mask)[0][completed_local]
            self._lowlevel_gait_transition_active[completed_ids] = False
            self._lowlevel_gait_transition_elapsed[completed_ids] = duration
            self._lowlevel_gait_frequencies[completed_ids] = (
                self._lowlevel_gait_transition_target_frequency[completed_ids]
            )
            self._lowlevel_gait_stance_durations[completed_ids] = (
                self._lowlevel_gait_transition_target_stance[completed_ids]
            )

    def _advance_lowlevel_gait_constant(self, mask, duration):
        if not torch.any(mask):
            return
        if not torch.is_tensor(duration):
            duration = torch.full(
                (int(mask.sum().item()),),
                float(duration),
                dtype=torch.float32,
                device=self.device,
            )
        self._lowlevel_gait_indices[mask] = torch.remainder(
            self._lowlevel_gait_indices[mask]
            + self._lowlevel_gait_frequencies[mask] * duration,
            1.0,
        )

    def _refresh_lowlevel_gait_pair_phases(self):
        uniform = torch.stack(
            [
                torch.remainder(self._lowlevel_gait_indices + 0.5, 1.0),
                self._lowlevel_gait_indices,
            ],
            dim=1,
        )
        stance = self._lowlevel_gait_stance_durations.unsqueeze(1)
        swing = (1.0 - stance).clamp(min=1.0e-6)
        self._lowlevel_gait_pair_phases[:] = torch.where(
            uniform <= stance,
            uniform / (2.0 * stance.clamp(min=1.0e-6)),
            0.5 + (uniform - stance) / (2.0 * swing),
        )

    def _set_lowlevel_fixed_trot_stopped(self, mask):
        if not torch.any(mask):
            return
        frequency = torch.full_like(
            self._lowlevel_gait_frequencies[mask],
            float(
                self.config.robot.b2z1_command.get("lowlevel_fixed_trot_frequency", 1.5)
            ),
        )
        self._lowlevel_gait_indices[mask] = 0.0
        self._lowlevel_gait_frequencies[mask] = 0.0
        self._lowlevel_gait_transition_active[mask] = False
        self._lowlevel_gait_transition_elapsed[mask] = 0.0
        self._lowlevel_gait_stance_durations[mask] = (
            self._get_lowlevel_gait_stance_duration(frequency)
        )
        self._lowlevel_gait_transition_start_stance[mask] = (
            self._lowlevel_gait_stance_durations[mask]
        )
        self._lowlevel_gait_transition_target_stance[mask] = (
            self._lowlevel_gait_stance_durations[mask]
        )
        self._lowlevel_gait_pair_phases[mask, 0] = 0.5
        self._lowlevel_gait_pair_phases[mask, 1] = 0.0
        self._lowlevel_clock_inputs[mask] = 0.0

    @override
    def _pre_physics_step(self, actions):
        self._refresh_b2z1_arm_target()
        clip_action_limit = float(self.config.robot.control.action_clip_value)
        self.actions[:] = torch.clamp(actions, -clip_action_limit, clip_action_limit)
        self.actions_after_delay[:] = self.actions
        self.log_dict["joint_action_clip_frac"] = (
            self.actions.abs() == clip_action_limit
        ).sum() / self.actions.numel()

    @override
    def _apply_force_in_physics_step(self):
        if self.config.simulator.config.name != "isaacsim":
            return super()._apply_force_in_physics_step()

        gripper_command = self.actions_after_delay[:, self._b2z1_gripper_joint_index]
        gripper_close_fraction = (
            gripper_command - self._b2z1_gripper_open_command
        ) / (self._b2z1_gripper_closed_command - self._b2z1_gripper_open_command)
        gripper_close_fraction = gripper_close_fraction.clamp(0.0, 1.0)
        gripper_target = torch.lerp(
            torch.full_like(gripper_close_fraction, self._b2z1_gripper_open_target),
            torch.full_like(gripper_close_fraction, self._b2z1_gripper_closed_target),
            gripper_close_fraction,
        ).unsqueeze(-1)
        if self._b2z1_leg_control_mode == "position":
            self.torques.zero_()
            self.simulator._robot.set_joint_effort_target(
                self.torques, joint_ids=self.simulator.dof_ids
            )
            jpos_target = self.actions_after_delay * self.action_scale + self.default_dof_pos
            jpos_target[:, self._b2z1_arm_joint_indices] = self._b2z1_arm_pos_targets
            jpos_target[:, self._b2z1_gripper_joint_index] = gripper_target.squeeze(-1)
            self.simulator._robot.set_joint_position_target(
                jpos_target, joint_ids=self.simulator.dof_ids
            )
        elif self._b2z1_leg_control_mode == "effort":
            self.torques = self._compute_b2z1_lowlevel_torques()
            self.simulator._robot.set_joint_effort_target(
                self.torques,
                joint_ids=self.simulator.dof_ids,
            )
            jpos_target = torch.empty(
                self.num_envs,
                self._b2z1_arm_gripper_joint_indices.numel(),
                device=self.device,
            )
            jpos_target[:, : self._b2z1_arm_joint_indices.numel()] = self._b2z1_arm_pos_targets
            jpos_target[:, -1] = gripper_target.squeeze(-1)
            self.simulator._robot.set_joint_position_target(
                jpos_target,
                joint_ids=self._b2z1_arm_gripper_sim_joint_ids,
            )
        else:
            raise ValueError(
                f"Unsupported B2Z1 low-level leg control mode: {self._b2z1_leg_control_mode}"
            )
        self._set_perturbation_forces()
        self.simulator.apply_rigid_body_force_at_pos_tensor(self.push_force, self.push_force_pos)

    def _compute_b2z1_lowlevel_torques(self):
        """Compute B2Z1 low-level PD efforts from policy-order actions to named sim joints."""

        policy_actions = self.actions_after_delay[:, self._lowlevel_policy_joint_indices]
        joint_ids = self._lowlevel_policy_joint_indices
        actions_scaled = policy_actions * self._lowlevel_policy_action_scale
        target_pos = self.default_dof_pos[:, joint_ids] + actions_scaled
        policy_torques = (
            self.p_gains[joint_ids] * (target_pos - self.simulator.dof_pos[:, joint_ids])
            - self.d_gains[joint_ids] * self.simulator.dof_vel[:, joint_ids]
        )
        policy_torques[:, 12:] = 0.0
        torque_limits = self.torque_limits[joint_ids].view(1, -1)
        policy_torques = torch.maximum(
            torch.minimum(policy_torques, torque_limits), -torque_limits
        )

        torques = torch.zeros_like(self.torques)
        torques[:, joint_ids] = policy_torques
        torques[:, self._b2z1_gripper_joint_index] = 0.0
        self._debug_lowlevel_env_actions = policy_actions.detach().clone()
        self._debug_lowlevel_env_torques = policy_torques.detach().clone()
        return torques

    def _refresh_b2z1_arm_target(self):
        self._b2z1_arm_pos_targets.copy_(
            self.default_dof_pos[:, self._b2z1_arm_joint_indices]
        )
        if self._b2z1_disable_arm_ik:
            return
        robot = getattr(self.simulator, "_robot", None)
        root_view = getattr(robot, "root_physx_view", None)
        if robot is None or root_view is None or not hasattr(root_view, "get_jacobians"):
            return
        try:
            self._b2z1_ee_pos = self.simulator._rigid_body_pos[:, self._b2z1_gripper_body_index]
            self._b2z1_ee_orn = self.simulator._rigid_body_rot[:, self._b2z1_gripper_body_index]
            jac = root_view.get_jacobians()
            floating_base_offset = jac.shape[-1] - self.num_dof
            jac_arm_joint_ids = self._b2z1_arm_sim_joint_ids + floating_base_offset
            self._b2z1_ee_j_eef = jac[
                :, self._b2z1_gripper_sim_body_id, :6, jac_arm_joint_ids
            ]
        except Exception as exc:
            logger.debug(f"B2Z1 arm IK target unavailable this step: {exc}")
            return

        goal_pos, goal_quat = self._compute_b2z1_ee_goal_world()
        dpos = goal_pos - self._b2z1_ee_pos
        ee_orn = normalize(self._b2z1_ee_orn)
        drot = self._orientation_error_xyzw(goal_quat, ee_orn)
        dpose = torch.cat([dpos, drot], dim=-1).unsqueeze(-1)
        j_eef_t = torch.transpose(self._b2z1_ee_j_eef, 1, 2)
        damping = torch.eye(6, device=self.device, dtype=self._b2z1_ee_j_eef.dtype).mul_(
            self._b2z1_ik_damping**2
        )
        delta_q = torch.bmm(
            j_eef_t,
            torch.linalg.solve(torch.bmm(self._b2z1_ee_j_eef, j_eef_t) + damping[None], dpose),
        ).squeeze(-1)
        self._b2z1_arm_pos_targets.copy_(
            self.simulator.dof_pos[:, self._b2z1_arm_joint_indices] + delta_q
        )

    def _compute_b2z1_ee_goal_world(self):
        base_yaw_quat = self._get_b2z1_base_yaw_quat()
        origin = torch.zeros(self.num_envs, 3, device=self.device)
        origin[:, :2] = self.simulator.robot_root_states[:, :2]
        ee_goal_cart = self._get_b2z1_ee_goal_cart()
        goal_local = self._b2z1_goal_center_offset + ee_goal_cart
        goal_pos = origin + quat_rotate(base_yaw_quat, goal_local)
        goal_quat = self._compute_b2z1_ee_quat_world_from_cart(
            ee_goal_cart, self._b2z1_commands[:, 6:9], base_yaw_quat
        )
        return goal_pos, goal_quat

    def _get_b2z1_base_yaw_quat(self):
        qx, qy, qz, qw = self.base_quat.unbind(-1)
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        half_yaw = 0.5 * yaw
        base_yaw_quat = torch.zeros(self.num_envs, 4, device=self.device)
        base_yaw_quat[:, 2] = torch.sin(half_yaw)
        base_yaw_quat[:, 3] = torch.cos(half_yaw)
        return base_yaw_quat

    def _compute_b2z1_ee_quat_world_from_cart(
        self, ee_goal_cart, orn_delta=None, base_yaw_quat=None
    ):
        if base_yaw_quat is None:
            base_yaw_quat = self._get_b2z1_base_yaw_quat()
        if orn_delta is None:
            orn_delta = torch.zeros_like(ee_goal_cart)
        sphere = self._cart_to_sphere(ee_goal_cart)
        # ManipLoco/lab30 defines the gripper's nominal roll as +pi/2.
        roll = orn_delta[:, 0] + 1.5707963267948966
        pitch = -sphere[:, 1] + self._b2z1_arm_induced_pitch + orn_delta[:, 1]
        goal_yaw = sphere[:, 2] + orn_delta[:, 2]
        goal_quat_local = self._quat_from_euler_xyz_xyzw(roll, pitch, goal_yaw)
        return quat_mul(base_yaw_quat, goal_quat_local)

    def _get_b2z1_ee_goal_cart(self):
        return self._b2z1_default_ee_goal_cart + self._b2z1_commands[:, 3:6]

    def _project_b2z1_ee_command_to_workspace(self):
        goal_cart = self._get_b2z1_ee_goal_cart()
        goal_sphere = self._cart_to_sphere(goal_cart)
        goal_sphere = torch.maximum(
            torch.minimum(goal_sphere, self._b2z1_ee_goal_sphere_upper),
            self._b2z1_ee_goal_sphere_lower,
        )
        effective_goal_cart = self._sphere_to_cart(goal_sphere)
        self._b2z1_commands[:, 3:6] = effective_goal_cart - self._b2z1_default_ee_goal_cart

    @staticmethod
    def _cart_to_sphere(cart):
        radius = torch.linalg.norm(cart, dim=-1).clamp_min(1e-8)
        pitch = torch.asin(torch.clamp(cart[:, 2] / radius, -1.0, 1.0))
        yaw = torch.atan2(cart[:, 1], cart[:, 0])
        return torch.stack([radius, pitch, yaw], dim=-1)

    @staticmethod
    def _sphere_to_cart(sphere):
        radius, pitch, yaw = sphere.unbind(dim=-1)
        horizontal_radius = radius * torch.cos(pitch)
        return torch.stack(
            [
                horizontal_radius * torch.cos(yaw),
                horizontal_radius * torch.sin(yaw),
                radius * torch.sin(pitch),
            ],
            dim=-1,
        )

    @staticmethod
    def _quat_from_euler_xyz_xyzw(roll, pitch, yaw):
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        return torch.stack(
            [
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ],
            dim=-1,
        )

    @staticmethod
    def _orientation_error_xyzw(desired, current):
        q_r = quat_mul(desired, quat_conjugate(current))
        return q_r[:, :3] * torch.sign(q_r[:, 3:4])

    @override
    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf=target_buf)
        if hasattr(self, "_lowlevel_obs_history"):
            self._lowlevel_obs_history[env_ids] = 0.0
        if hasattr(self, "_lowlevel_prev_policy_actions"):
            self._lowlevel_prev_policy_actions[env_ids] = 0.0
        if hasattr(self, "_lowlevel_gait_indices"):
            self._lowlevel_gait_indices[env_ids] = 0.0
            self._lowlevel_clock_inputs[env_ids] = 0.0
            gait_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            gait_reset_mask[env_ids] = True
            self._set_lowlevel_fixed_trot_stopped(gait_reset_mask)
            self._lowlevel_gait_transition_start_frequency[env_ids] = 0.0
            self._lowlevel_gait_transition_target_frequency[env_ids] = 0.0
        for buffer_name in (
            "_b2z1_commands",
            "_b2z1_commands_unclipped",
            "_last_b2z1_commands",
            "_b2z1_joint_actions",
        ):
            if hasattr(self, buffer_name):
                getattr(self, buffer_name)[env_ids] = 0.0
        if hasattr(self, "_b2z1_commands"):
            self._b2z1_commands[env_ids, 9] = self._b2z1_gripper_open_command
            self._b2z1_commands_unclipped[env_ids, 9] = self._b2z1_gripper_open_command
            self._last_b2z1_commands[env_ids, 9] = self._b2z1_gripper_open_command
        if hasattr(self, "_delta_actions"):
            gripper_delta_index = (self._delta_action_indices == 9).nonzero(as_tuple=False)
            if gripper_delta_index.numel() != 1:
                raise RuntimeError("B2Z1 gripper action index 9 must appear exactly once.")
            self._delta_actions[env_ids, gripper_delta_index.item()] = (
                self._b2z1_gripper_open_command
            )
        if hasattr(self, "_b2z1_arm_pos_targets"):
            self._b2z1_arm_pos_targets[env_ids] = self.default_dof_pos[
                0, self._b2z1_arm_joint_indices
            ]

    @override
    def _action_backmap(self):
        return torch.zeros(self.num_envs, self.COMMAND_DIM, device=self.device)

    def get_physical_b2z1_commands(self):
        return self._b2z1_commands

    def get_physical_homie_commands(self):
        return self.get_physical_b2z1_commands()

    def _reward_penalty_b2z1_command_rate(self):
        return torch.sum(torch.square(self._last_b2z1_commands - self._b2z1_commands), dim=1)

    def _reward_penalty_b2z1_command_limit(self):
        return torch.sum(torch.square(self._b2z1_commands_unclipped - self._b2z1_commands), dim=1)

    def _get_obs_b2z1_commands(self):
        return self._b2z1_commands

    def _get_obs_b_homie_commands(self):
        return self._get_obs_b2z1_commands()

    def _get_obs_actions(self):
        return self._b2z1_commands
