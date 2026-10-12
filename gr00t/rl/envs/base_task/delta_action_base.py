# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
from typing_extensions import override

from gr00t.rl.envs.legged_base_task.legged_robot_base import LeggedRobotBase


class DeltaActionBase(LeggedRobotBase):
    def __init__(self, config, device):
        self._delta_action_indices = torch.tensor(
            config.delta_action_indices, dtype=torch.long, device=device
        )
        self._delta_action_scale = config.delta_action_scale
        self._delta_action_step_clip = float(config.get("delta_action_step_clip", 1.0))
        super().__init__(config, device)

    @override
    def _init_buffers(self):
        super()._init_buffers()
        self._delta_actions = torch.zeros(
            self.num_envs, len(self._delta_action_indices), device=self.device, requires_grad=False
        )
        self._last_delta_actions = torch.zeros(
            self.num_envs, len(self._delta_action_indices), device=self.device, requires_grad=False
        )

    @override
    def step(self, actor_state):
        actions = actor_state["actions"]
        delta_actions_clip = self.config.get("delta_action_clip", 100.0)
        raw_delta_actions = actions[:, self._delta_action_indices]
        bounded_delta_actions = raw_delta_actions.clamp(
            -self._delta_action_step_clip, self._delta_action_step_clip
        )
        self.log_dict["delta_action_step_clip_frac"] = (
            raw_delta_actions.abs() > self._delta_action_step_clip
        ).float().mean()
        self._last_delta_actions[:] = bounded_delta_actions
        self._integrate_policy_delta_actions(
            bounded_delta_actions, delta_actions_clip
        )

        actor_state["actions"][:, self._delta_action_indices] = self._delta_actions

        if self.config.get("zero_vel", False):
            zero_env_ids = (
                ((self.stage_buf >= 1) & (self.stage_buf < 4)).nonzero(as_tuple=False).squeeze(-1)
            )

            # For teacher training mode: zero out policy actions
            if "gt_actions" not in actor_state:
                # Teacher training mode: zero out policy actions
                actor_state["actions"][zero_env_ids, :3] = 0.0
                logger.warning(
                    "overwriting policy actions with zero vel in delta action base STEP"
                )
            # For distillation mode: actions need to be zeroed out if enforce_teacher_rollout is True
            # else:
            #     # Check if enforce_teacher_rollout is True (set by trainer from algo config)
            #     if getattr(self, "enforce_teacher_rollout", False):
            #         actor_state["actions"][zero_env_ids, :3] = 0.0

        if self.config.get("zero_finger", False):
            if "gt_actions" not in actor_state:
                zero_env_ids = (
                    ((self.stage_buf == 0) | (self.stage_buf == 1))
                    .nonzero(as_tuple=False)
                    .squeeze(-1)
                )
                actor_state["actions"][zero_env_ids, 10] = 0.5

        if self.config.get("print_delta_actions", False):
            logger.info(f"policy actions: {self._last_delta_actions}")
            logger.info(f"cumulative_actions: {self._delta_actions}")
            logger.info(f"effective actions on envs: {actor_state['actions']}")
            logger.info(f"episode_length: {self.episode_length_buf}")
            logger.warning(f"stage: {self.stage_buf}")
        return super().step(actor_state)

    def _integrate_policy_delta_actions(
        self, bounded_delta_actions, delta_actions_clip
    ):
        """Map bounded policy deltas into the downstream action accumulator."""
        self._delta_actions += bounded_delta_actions * self._delta_action_scale
        self._normalize_delta_action_accumulator()
        if self.config.get("no_delta_vel", False):
            # Base velocity is an instantaneous physical command. Integrating it
            # like an EE pose delta creates an unintended random walk to saturation.
            self._delta_actions[:, :3] = bounded_delta_actions[:, :3]
        self._delta_actions = torch.clamp(
            self._delta_actions, -delta_actions_clip, delta_actions_clip
        )

    def _normalize_delta_action_accumulator(self):
        """Normalize accumulator coordinates with non-Euclidean topology."""
        normalizer = getattr(super(), "_normalize_delta_action_accumulator", None)
        if normalizer is not None:
            normalizer()

    def _reward_penalty_delta_action_rate(self):
        return torch.sum(torch.square(self._last_delta_actions), dim=-1)

    def _get_obs_delta_actions(self):
        return self._last_delta_actions

    def _store_delta_actions_buffer(self, env_ids):
        return self._delta_actions[env_ids]

    def _load_delta_actions_buffer(self, env_ids, data):
        self._delta_actions[env_ids] = data

    def _get_delta_actions_buffer_shape(self):
        return self._delta_actions.shape

    def _store_last_delta_actions_buffer(self, env_ids):
        return self._last_delta_actions[env_ids]

    def _load_last_delta_actions_buffer(self, env_ids, data):
        self._last_delta_actions[env_ids] = data

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        self._delta_actions[env_ids] = 0.0
        self._last_delta_actions[env_ids] = 0.0
        return super()._reset_buffers_callback(env_ids, target_buf)

    def _post_reset_callback(self, env_ids):
        if self.config.get("reset_delta_actions_with_backmap", False):
            # BUG BUG BUG
            pass

            # xx, yy = torch.meshgrid(env_ids, self._delta_action_indices)
            # self._delta_actions[env_ids] = self._action_backmap()[xx, yy]

            # logger.warning(f"reset_delta_actions_with_backmap: {self._delta_actions[env_ids]}")

            # import ipdb; ipdb.set_trace()

            # logger.warning(f"done")
