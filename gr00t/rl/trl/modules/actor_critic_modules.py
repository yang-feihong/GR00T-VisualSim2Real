# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from gr00t.rl.trl.utils.common import custom_instantiate
from gr00t.rl.trl.utils.rl import compute_episode_attnmask
from gr00t.rl.trl.modules.temporal_action_noise import (
    advance_standardized_ar1_context,
    conditional_gaussian_parameters,
    validate_temporal_noise_parameters,
)
from gr00t.rl.utils.running_mean_std import RunningMeanStd


class Actor(nn.Module):

    is_recurrent = False

    def __init__(
        self,
        env_config,
        algo_config,
        backbone,
        module_dim_dict={},
        running_mean_std=False,
        max_rollout_history=1,
        input_key="actor_obs",
    ):
        super(Actor, self).__init__()

        self.algo_config = algo_config
        obs_dim_dict = env_config.robot.algo_obs_dim_dict
        self.input_key = input_key
        residual_slice = algo_config.get("action_residual_obs_slice", None)
        residual_key = algo_config.get("action_residual_obs_key", None)
        if residual_slice is not None and residual_key is not None:
            raise ValueError(
                "Configure only one of action_residual_obs_slice and "
                "action_residual_obs_key"
            )
        self.action_residual_obs_slice = None
        self.zero_initialize_action_output = bool(
            algo_config.get("zero_initialize_action_output", False)
        )
        if residual_slice is not None:
            if len(residual_slice) != 2:
                raise ValueError(
                    "action_residual_obs_slice must contain [start, stop]"
                )
            start, stop = (int(value) for value in residual_slice)
            if start < 0 or stop <= start:
                raise ValueError(
                    "action_residual_obs_slice must define a non-empty positive slice"
                )
            self.action_residual_obs_slice = slice(start, stop)

        noise_stage_slice = algo_config.get("action_noise_stage_obs_slice", None)
        noise_stage_key = algo_config.get("action_noise_stage_obs_key", None)
        self.action_noise_stage_obs_slice = None
        if noise_stage_key is not None:
            obs_names = sorted(env_config.obs.obs_dict[self.input_key])
            if noise_stage_key not in obs_names:
                raise ValueError(
                    f"action_noise_stage_obs_key {noise_stage_key!r} is not in "
                    f"{self.input_key}"
                )
            obs_dims = env_config.obs.obs_dims
            start = sum(int(obs_dims[name]) for name in obs_names[: obs_names.index(noise_stage_key)])
            stop = start + int(obs_dims[noise_stage_key])
            noise_stage_slice = (start, stop)
        if noise_stage_slice is not None:
            if len(noise_stage_slice) != 2:
                raise ValueError(
                    "action_noise_stage_obs_slice must contain [start, stop]"
                )
            start, stop = (int(value) for value in noise_stage_slice)
            if start < 0 or stop <= start:
                raise ValueError(
                    "action_noise_stage_obs_slice must define a non-empty positive slice"
                )
            stage_scales = torch.as_tensor(
                algo_config.action_noise_stage_scales, dtype=torch.float32
            )
            num_stages = stop - start
            if stage_scales.ndim == 1:
                if stage_scales.numel() != num_stages:
                    raise ValueError(
                        "action_noise_stage_scales must contain one scalar per stage"
                    )
                stage_scales = stage_scales[:, None]
            elif stage_scales.ndim != 2 or stage_scales.shape[0] != num_stages:
                raise ValueError(
                    "action_noise_stage_scales must be [num_stages] or "
                    "[num_stages, num_actions]"
                )
            if torch.any(stage_scales <= 0.0):
                raise ValueError("action_noise_stage_scales must be positive")
            self.action_noise_stage_obs_slice = slice(start, stop)
            self.register_buffer(
                "action_noise_stage_scales", stage_scales, persistent=False
            )
        else:
            self.action_noise_stage_scales = None

        init_noise_std = algo_config.init_noise_std

        self.max_rollout_history = max_rollout_history
        self.actor_module = custom_instantiate(
            backbone,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _resolve=False,
        )

        self.running_mean_std = None
        if running_mean_std:
            self.running_mean_std = RunningMeanStd(
                (obs_dim_dict[self.input_key],), per_channel=True
            )

        # Action noise
        self.num_actions = self.actor_module.output_dim
        self.state_dependent_action_noise = bool(
            algo_config.get("state_dependent_action_noise", False)
        )
        self.action_noise_scale_module = None
        if self.state_dependent_action_noise:
            self.action_noise_scale_module = custom_instantiate(
                deepcopy(backbone),
                env_config=env_config,
                algo_config=algo_config,
                obs_dim_dict=obs_dim_dict,
                module_dim_dict=module_dim_dict,
                _resolve=False,
            )
            if self.action_noise_scale_module.output_dim != self.num_actions:
                raise ValueError(
                    "The action-noise attenuation network must output one value "
                    f"per action, got {self.action_noise_scale_module.output_dim}"
                )
            min_scale = torch.as_tensor(
                algo_config.get("state_dependent_action_noise_min_scale", 1.0e-4),
                dtype=torch.float32,
            )
            if min_scale.numel() not in (1, self.num_actions):
                raise ValueError(
                    "state_dependent_action_noise_min_scale must be scalar or "
                    "contain one value per action"
                )
            min_scale = min_scale.expand(self.num_actions).clone()
            if torch.any((min_scale < 0.0) | (min_scale >= 1.0)):
                raise ValueError(
                    "state_dependent_action_noise_min_scale must be in [0, 1)"
                )
            self.register_buffer(
                "state_dependent_action_noise_min_scale", min_scale, persistent=True
            )
            self.register_buffer(
                "exploration_budget_dual_weight",
                torch.zeros(self.num_actions, dtype=torch.float32),
                persistent=True,
            )
            self._initialize_action_noise_attenuation()
        if residual_key is not None:
            obs_names = sorted(env_config.obs.obs_dict[self.input_key])
            if residual_key not in obs_names:
                raise ValueError(
                    f"action_residual_obs_key {residual_key!r} is not in "
                    f"{self.input_key}"
                )
            obs_dims = env_config.obs.obs_dims
            key_dim = int(obs_dims[residual_key])
            if key_dim < self.num_actions:
                raise ValueError(
                    f"action_residual_obs_key {residual_key!r} has {key_dim} values, "
                    f"fewer than num_actions={self.num_actions}"
                )
            key_stop = sum(
                int(obs_dims[name])
                for name in obs_names[: obs_names.index(residual_key) + 1]
            )
            # The task geometry contract keeps its action reference in the
            # final num_actions values. Resolve the global slice from the key
            # layout so adding another observation cannot silently shift it.
            self.action_residual_obs_slice = slice(
                key_stop - self.num_actions, key_stop
            )
        self._zero_initialize_action_output()
        if (
            self.action_noise_stage_scales is not None
            and self.action_noise_stage_scales.shape[-1] not in (1, self.num_actions)
        ):
            raise ValueError(
                "action_noise_stage_scales must contain either one value or "
                "num_actions values per stage"
            )
        residual_stage_masks = algo_config.get("action_residual_stage_masks", None)
        if residual_stage_masks is not None:
            if self.action_noise_stage_obs_slice is None:
                raise ValueError(
                    "action_residual_stage_masks requires a configured stage observation"
                )
            residual_stage_masks = torch.as_tensor(
                residual_stage_masks, dtype=torch.float32
            )
            expected_shape = (
                self.action_noise_stage_obs_slice.stop
                - self.action_noise_stage_obs_slice.start,
                self.num_actions,
            )
            if tuple(residual_stage_masks.shape) != expected_shape:
                raise ValueError(
                    "action_residual_stage_masks must be [num_stages, num_actions], "
                    f"expected {expected_shape}, got {tuple(residual_stage_masks.shape)}"
                )
            if torch.any((residual_stage_masks < 0.0) | (residual_stage_masks > 1.0)):
                raise ValueError("action_residual_stage_masks values must be in [0, 1]")
            self.register_buffer(
                "action_residual_stage_masks",
                residual_stage_masks,
                persistent=False,
            )
        else:
            self.action_residual_stage_masks = None
        self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))

        temporal_rho = algo_config.get("temporal_action_noise_rho", None)
        temporal_fraction = algo_config.get(
            "temporal_action_noise_correlated_fraction", None
        )
        if (temporal_rho is None) != (temporal_fraction is None):
            raise ValueError(
                "temporal_action_noise_rho and "
                "temporal_action_noise_correlated_fraction must be configured together"
            )
        self.temporal_action_noise_enabled = temporal_rho is not None
        if self.temporal_action_noise_enabled:
            self.temporal_action_noise_rho = float(temporal_rho)
            temporal_fraction = torch.as_tensor(
                temporal_fraction, dtype=torch.float32
            )
            if temporal_fraction.numel() not in (1, self.num_actions):
                raise ValueError(
                    "temporal_action_noise_correlated_fraction must be a scalar "
                    f"or contain {self.num_actions} action values"
                )
            validate_temporal_noise_parameters(
                self.temporal_action_noise_rho,
                temporal_fraction,
            )
            self.register_buffer(
                "temporal_action_noise_correlated_fraction",
                temporal_fraction,
                persistent=False,
            )
        else:
            self.temporal_action_noise_rho = 0.0
            self.temporal_action_noise_correlated_fraction = 0.0
        self.state_dependent_temporal_action_noise = bool(
            algo_config.get("state_dependent_temporal_action_noise", False)
        )
        self.temporal_action_noise_fraction_module = None
        if self.state_dependent_temporal_action_noise:
            if not self.temporal_action_noise_enabled:
                raise ValueError(
                    "state-dependent temporal action noise requires temporal action noise"
                )
            self.temporal_action_noise_fraction_module = custom_instantiate(
                deepcopy(backbone),
                env_config=env_config,
                algo_config=algo_config,
                obs_dim_dict=obs_dim_dict,
                module_dim_dict=module_dim_dict,
                _resolve=False,
            )
            if self.temporal_action_noise_fraction_module.output_dim != self.num_actions:
                raise ValueError(
                    "The temporal-fraction network must output one value per action, "
                    f"got {self.temporal_action_noise_fraction_module.output_dim}"
                )
            min_fraction = torch.as_tensor(
                algo_config.get(
                    "state_dependent_temporal_action_noise_min_fraction", 0.0
                ),
                dtype=torch.float32,
            )
            max_fraction = torch.as_tensor(
                algo_config.get(
                    "state_dependent_temporal_action_noise_max_fraction", 0.99
                ),
                dtype=torch.float32,
            )
            initial_fraction = torch.as_tensor(
                algo_config.get(
                    "state_dependent_temporal_action_noise_initial_fraction",
                    temporal_fraction,
                ),
                dtype=torch.float32,
            )
            for name, value in (
                ("min_fraction", min_fraction),
                ("max_fraction", max_fraction),
                ("initial_fraction", initial_fraction),
            ):
                if value.numel() not in (1, self.num_actions):
                    raise ValueError(
                        f"state-dependent temporal {name} must be scalar or "
                        f"contain {self.num_actions} action values"
                    )
            min_fraction = min_fraction.expand(self.num_actions).clone()
            max_fraction = max_fraction.expand(self.num_actions).clone()
            initial_fraction = initial_fraction.expand(self.num_actions).clone()
            if torch.any(min_fraction < 0.0) or torch.any(max_fraction >= 1.0):
                raise ValueError(
                    "state-dependent temporal fraction bounds must satisfy "
                    "0 <= min_fraction and max_fraction < 1"
                )
            if torch.any(min_fraction > max_fraction):
                raise ValueError(
                    "state-dependent temporal min_fraction must not exceed max_fraction"
                )
            variable_fraction = max_fraction > min_fraction
            invalid_variable_initial = variable_fraction & (
                (initial_fraction <= min_fraction)
                | (initial_fraction >= max_fraction)
            )
            invalid_constant_initial = (~variable_fraction) & (
                initial_fraction != min_fraction
            )
            if torch.any(invalid_variable_initial | invalid_constant_initial):
                raise ValueError(
                    "state-dependent temporal initial_fraction must lie strictly "
                    "inside variable bounds and equal constant bounds"
                )
            self.register_buffer(
                "state_dependent_temporal_action_noise_min_fraction",
                min_fraction,
                persistent=True,
            )
            self.register_buffer(
                "state_dependent_temporal_action_noise_max_fraction",
                max_fraction,
                persistent=True,
            )
            self._initialize_temporal_action_noise_fraction(initial_fraction)
        self._temporal_action_noise_context = None
        self._base_action_mean = None
        self._marginal_action_std = None
        self._global_action_std = None
        self._action_noise_attenuation = None
        self._action_noise_context = None
        self._effective_temporal_action_noise_fraction = None

        if algo_config.get("freeze_noise_std", False):
            self.std.requires_grad = False

        self.clamp_noise_std = algo_config.get("clamp_noise_std", False)
        if self.clamp_noise_std:
            self.max_noise_std = algo_config.get("max_noise_std", 1.0)

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # Initialize observation buffer for rollout
        self.obs_dict_buffer = TensorDict()
        self.dones_buffer = None
        self.steps = 0
        self.is_eval_mode = False

    def _zero_initialize_action_output(self):
        """Start residual or explicitly configured delta policies at zero mean."""
        if (
            self.action_residual_obs_slice is None
            and not self.zero_initialize_action_output
        ):
            return

        output_layers = [
            module
            for module in self.actor_module.modules()
            if isinstance(module, nn.Linear) and module.out_features == self.num_actions
        ]
        if not output_layers:
            raise ValueError(
                "A zero-initialized actor requires a linear action output projection with "
                f"out_features={self.num_actions}"
            )
        output_layer = output_layers[-1]
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    def _initialize_action_noise_attenuation(self):
        initial_scale = float(
            self.algo_config.get("state_dependent_action_noise_initial_scale", 0.95)
        )
        min_scale = self.state_dependent_action_noise_min_scale
        if not 0.0 < initial_scale < 1.0 or torch.any(min_scale >= initial_scale):
            raise ValueError(
                "state-dependent noise initial scale must be in (min_scale, 1)"
            )
        output_layers = [
            module
            for module in self.action_noise_scale_module.modules()
            if isinstance(module, nn.Linear) and module.out_features == self.num_actions
        ]
        if not output_layers:
            raise ValueError(
                "The action-noise attenuation network needs a final linear action head"
            )
        output_layer = output_layers[-1]
        target = torch.full_like(min_scale, initial_scale)
        normalized = (target - min_scale) / (1.0 - min_scale)
        nn.init.zeros_(output_layer.weight)
        with torch.no_grad():
            output_layer.bias.copy_(torch.logit(normalized))

    def _initialize_temporal_action_noise_fraction(self, initial_fraction):
        output_layers = [
            module
            for module in self.temporal_action_noise_fraction_module.modules()
            if isinstance(module, nn.Linear) and module.out_features == self.num_actions
        ]
        if not output_layers:
            raise ValueError(
                "The temporal-fraction network needs a final linear action head"
            )
        output_layer = output_layers[-1]
        min_fraction = self.state_dependent_temporal_action_noise_min_fraction
        max_fraction = self.state_dependent_temporal_action_noise_max_fraction
        span = max_fraction - min_fraction
        normalized = torch.where(
            span > 0.0,
            (initial_fraction - min_fraction) / span.clamp_min(
                torch.finfo(span.dtype).eps
            ),
            torch.full_like(span, 0.5),
        )
        nn.init.zeros_(output_layer.weight)
        with torch.no_grad():
            output_layer.bias.copy_(torch.logit(normalized))

    def _clamp_action_noise_std(self):
        max_std = torch.as_tensor(
            self.max_noise_std, dtype=self.std.dtype, device=self.std.device
        )
        if max_std.numel() not in (1, self.num_actions):
            raise ValueError(
                "max_noise_std must be a scalar or contain one value per action; "
                f"got {max_std.numel()} values for {self.num_actions} actions"
            )
        min_std = torch.as_tensor(
            self.algo_config.get("min_noise_std", 1.0e-6),
            dtype=self.std.dtype,
            device=self.std.device,
        )
        if min_std.numel() not in (1, self.num_actions):
            raise ValueError(
                "min_noise_std must be a scalar or contain one value per action; "
                f"got {min_std.numel()} values for {self.num_actions} actions"
            )
        min_std = min_std.expand_as(self.std)
        max_std = max_std.expand_as(self.std)
        if torch.any(max_std < min_std):
            raise ValueError("max_noise_std must be greater than or equal to min_noise_std")
        with torch.no_grad():
            self.std.copy_(torch.maximum(torch.minimum(self.std, max_std), min_std))

    def reset(self, dones=None):
        pass

    def _get_action_residual_mask(self, obs_dict, last_step_only=False):
        if self.action_residual_stage_masks is None:
            return None
        stage_obs = obs_dict[self.input_key][
            ..., self.action_noise_stage_obs_slice
        ]
        if last_step_only:
            stage_obs = stage_obs[:, -1]
        stage_index = torch.argmax(stage_obs, dim=-1)
        return self.action_residual_stage_masks[stage_index]

    def _compute_action_std(self, mean, obs_dict, last_step_only=False):
        global_std = mean * 0.0 + self.std
        action_std = global_std
        attenuation = torch.ones_like(action_std)
        if self.state_dependent_action_noise:
            noise_obs = obs_dict[self.input_key]
            if self.running_mean_std is not None:
                noise_obs = self.running_mean_std(noise_obs, update_stats=False)
            if last_step_only:
                noise_obs = noise_obs[:, -1]
            logits = self.action_noise_scale_module(noise_obs)
            min_scale = self.state_dependent_action_noise_min_scale.to(
                dtype=logits.dtype, device=logits.device
            )
            attenuation = min_scale + (1.0 - min_scale) * torch.sigmoid(logits)
            action_std = action_std * attenuation
        if self.action_noise_stage_obs_slice is not None:
            stage_obs = obs_dict[self.input_key][
                ..., self.action_noise_stage_obs_slice
            ]
            if last_step_only:
                stage_obs = stage_obs[:, -1]
            stage_index = torch.argmax(stage_obs, dim=-1)
            stage_scale = self.action_noise_stage_scales[stage_index]
            if stage_scale.shape[-1] == 1:
                stage_scale = stage_scale.expand_as(action_std)
            action_std = action_std * stage_scale
        residual_mask = self._get_action_residual_mask(obs_dict, last_step_only)
        if residual_mask is not None:
            min_std = torch.as_tensor(
                self.algo_config.get("min_noise_std", 1.0e-6),
                dtype=action_std.dtype,
                device=action_std.device,
            )
            action_std = action_std * residual_mask + min_std * (1.0 - residual_mask)
        self._global_action_std = global_std
        self._action_noise_attenuation = attenuation
        return action_std

    def _compute_temporal_action_noise_fraction(
        self, mean, obs_dict, last_step_only=False
    ):
        if not self.temporal_action_noise_enabled:
            return None
        fraction = torch.ones_like(mean) * torch.as_tensor(
            self.temporal_action_noise_correlated_fraction,
            dtype=mean.dtype,
            device=mean.device,
        )
        if getattr(self, "state_dependent_temporal_action_noise", False):
            fraction_obs = obs_dict[self.input_key]
            if self.running_mean_std is not None:
                fraction_obs = self.running_mean_std(
                    fraction_obs, update_stats=False
                )
            if last_step_only:
                fraction_obs = fraction_obs[:, -1]
            logits = self.temporal_action_noise_fraction_module(fraction_obs)
            min_fraction = (
                self.state_dependent_temporal_action_noise_min_fraction.to(
                    dtype=logits.dtype, device=logits.device
                )
            )
            max_fraction = (
                self.state_dependent_temporal_action_noise_max_fraction.to(
                    dtype=logits.dtype, device=logits.device
                )
            )
            fraction = min_fraction + (max_fraction - min_fraction) * torch.sigmoid(
                logits
            )
        residual_mask = self._get_action_residual_mask(obs_dict, last_step_only)
        if residual_mask is not None:
            fraction = fraction * residual_mask
        return fraction

    def forward(self, obs_dict, **kwargs):
        obs_dict = obs_dict.copy()
        action_residual_mask = self._get_action_residual_mask(obs_dict)
        action_residual = None
        if self.action_residual_obs_slice is not None:
            action_residual = obs_dict[self.input_key][
                ..., self.action_residual_obs_slice
            ]
            if action_residual.shape[-1] != self.num_actions:
                raise RuntimeError(
                    "action_residual_obs_slice must select one reference value per "
                    f"action, got {action_residual.shape[-1]} values for "
                    f"{self.num_actions} actions"
                )
        if self.running_mean_std is not None:
            with torch.no_grad():
                obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        if self.algo_config.get("use_clampped_std", False):
            self.std.clamp_(min=self.algo_config.std_clamp_min, max=self.algo_config.std_clamp_max)

        action_mean = self.actor_module(obs_dict[self.input_key], **kwargs)
        if action_residual_mask is not None:
            action_mean = action_mean * action_residual_mask
        if action_residual is not None:
            action_mean = action_mean + action_residual
        return action_mean

    @property
    def has_normalized_actions(self):
        return False

    @property
    def action_mean(self):
        return self._base_action_mean

    @property
    def action_distribution_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def marginal_action_std(self):
        return self._marginal_action_std

    @property
    def global_action_std(self):
        return self._global_action_std

    @property
    def action_noise_attenuation(self):
        return self._action_noise_attenuation

    @property
    def action_noise_context(self):
        return self._action_noise_context

    @property
    def temporal_action_noise_fraction(self):
        return self._effective_temporal_action_noise_fraction

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _set_action_distribution(
        self,
        base_mean,
        marginal_std,
        action_noise_context=None,
        marginalize_temporal=False,
        temporal_action_noise_fraction=None,
    ):
        if self.temporal_action_noise_enabled and temporal_action_noise_fraction is None:
            temporal_action_noise_fraction = torch.ones_like(
                marginal_std
            ) * torch.as_tensor(
                self.temporal_action_noise_correlated_fraction,
                dtype=marginal_std.dtype,
                device=marginal_std.device,
            )
        self._base_action_mean = base_mean
        self._marginal_action_std = marginal_std
        self._action_noise_context = action_noise_context
        self._effective_temporal_action_noise_fraction = (
            temporal_action_noise_fraction
        )
        if self.temporal_action_noise_enabled and not marginalize_temporal:
            if action_noise_context is None:
                raise ValueError(
                    "Temporal action noise requires a stored action_noise_context"
                )
            distribution_mean, distribution_std = conditional_gaussian_parameters(
                base_mean,
                marginal_std,
                action_noise_context,
                temporal_action_noise_fraction,
            )
        else:
            if action_noise_context is not None:
                raise ValueError(
                    "action_noise_context was provided while temporal action noise is disabled"
                )
            distribution_mean, distribution_std = base_mean, marginal_std
        self.distribution = Normal(distribution_mean, distribution_std)

    def _advance_temporal_action_noise_context(self, base_mean, cur_dones):
        if not self.temporal_action_noise_enabled:
            return None
        if cur_dones is None:
            cur_dones = torch.zeros(
                base_mean.shape[:-1], dtype=torch.bool, device=base_mean.device
            )
        innovation = torch.randn_like(base_mean)
        self._temporal_action_noise_context = advance_standardized_ar1_context(
            self._temporal_action_noise_context,
            cur_dones,
            innovation,
            self.temporal_action_noise_rho,
        ).detach()
        return self._temporal_action_noise_context

    def _action_state_dict(self, actions):
        state = {
            "actions": actions,
            "action_mean": self.action_mean,
            # Storage and diagnostics always use the marginal policy scale.
            # Temporal conditional scales are reconstructed from the sequence.
            "action_sigma": self.marginal_action_std,
        }
        if self.temporal_action_noise_enabled:
            state["action_temporal_fraction"] = (
                self._effective_temporal_action_noise_fraction
            )
        return TensorDict(state)

    def update_distribution(
        self,
        obs_dict,
        episode_attnmask=None,
        last_step_only=False,
        action_noise_context=None,
        marginalize_temporal=False,
        **kwargs,
    ):
        # obs_dict[self.input_key] = torch.zeros_like(obs_dict[self.input_key]) # DEBUG
        # logger.warning(f"obs_dict[self.input_key]: {obs_dict[self.input_key]}")
        mean = self.forward(obs_dict, episode_attnmask=episode_attnmask, **kwargs)
        if last_step_only:
            mean = mean[:, -1]

        if self.clamp_noise_std:
            self._clamp_action_noise_std()
        action_std = self._compute_action_std(mean, obs_dict, last_step_only)
        temporal_fraction = self._compute_temporal_action_noise_fraction(
            mean, obs_dict, last_step_only
        )
        self._set_action_distribution(
            mean,
            action_std,
            action_noise_context,
            marginalize_temporal=marginalize_temporal,
            temporal_action_noise_fraction=temporal_fraction,
        )

    def act(
        self,
        obs_dict,
        episode_attnmask=None,
        action_noise_context=None,
        **kwargs,
    ):
        self.update_distribution(
            obs_dict,
            episode_attnmask=episode_attnmask,
            action_noise_context=action_noise_context,
            **kwargs,
        )
        actions = self.distribution.sample()
        return self._action_state_dict(actions)

    def deterministic_action(self, obs_dict, episode_attnmask=None, **kwargs):
        """Return the network base mean without constructing a sampling distribution."""
        return self.forward(
            obs_dict,
            episode_attnmask=episode_attnmask,
            **kwargs,
        )

    def update_dones_buffer_and_compute_episode_attnmask(self, cur_dones):
        if self.steps > 0 and self.max_rollout_history > 1:
            if self.dones_buffer is None:
                self.dones_buffer = cur_dones.clone().unsqueeze(1)
            else:
                self.dones_buffer = torch.cat(
                    [self.dones_buffer, cur_dones.clone().unsqueeze(1)], dim=1
                )
            if self.dones_buffer.shape[1] > self.max_rollout_history - 1:
                self.dones_buffer = self.dones_buffer[:, -self.max_rollout_history + 1 :]
            dones = torch.cat(
                [self.dones_buffer, torch.zeros_like(self.dones_buffer[:, :1])], dim=1
            )
        else:
            dones = torch.zeros_like(cur_dones.unsqueeze(1))
        episode_attnmask_from_dones = compute_episode_attnmask(dones)
        return episode_attnmask_from_dones

    def _update_obs_buffer(self, obs_dict, episode_attnmask=None, cur_dones=None):
        update_episode_attnmask = False

        for key in obs_dict.keys():
            if key not in self.obs_dict_buffer:
                self.obs_dict_buffer[key] = obs_dict[key].unsqueeze(1)
            else:
                self.obs_dict_buffer[key] = torch.cat(
                    [self.obs_dict_buffer[key], obs_dict[key].unsqueeze(1)], dim=1
                )
            if self.obs_dict_buffer[key].shape[1] > self.max_rollout_history:
                update_episode_attnmask = True
                self.obs_dict_buffer[key] = self.obs_dict_buffer[key][
                    :, -self.max_rollout_history :
                ]

        if episode_attnmask is not None and update_episode_attnmask:
            episode_attnmask = episode_attnmask[
                :, -self.max_rollout_history :, -self.max_rollout_history :
            ]

        if cur_dones is not None:
            episode_attnmask_from_dones = self.update_dones_buffer_and_compute_episode_attnmask(
                cur_dones
            )
            if episode_attnmask is not None:
                assert (episode_attnmask == episode_attnmask_from_dones).all()
            if episode_attnmask is None:
                episode_attnmask = episode_attnmask_from_dones

        return episode_attnmask

    def rollout(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)
        mean = self.forward(
            obs_dict=self.obs_dict_buffer,
            episode_attnmask=episode_attnmask,
            **kwargs,
        )[:, -1]
        if self.clamp_noise_std:
            self._clamp_action_noise_std()
        marginal_std = self._compute_action_std(
            mean, self.obs_dict_buffer, last_step_only=True
        )
        temporal_fraction = self._compute_temporal_action_noise_fraction(
            mean, self.obs_dict_buffer, last_step_only=True
        )
        action_noise_context = self._advance_temporal_action_noise_context(
            mean, cur_dones
        )
        self._set_action_distribution(
            mean,
            marginal_std,
            action_noise_context,
            temporal_action_noise_fraction=temporal_fraction,
        )
        self.steps += 1
        return self._action_state_dict(self.distribution.sample())

    def rollout_with_obj_pred(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        """
        Rollout method compatible with object prediction trainer.
        This base Actor class doesn't predict object positions, so it doesn't return predicted_obj_pos.
        """
        return self.rollout(
            obs_dict,
            episode_attnmask=episode_attnmask,
            cur_dones=cur_dones,
            **kwargs,
        )

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)
        actions_mean = self.forward(
            obs_dict=self.obs_dict_buffer, episode_attnmask=episode_attnmask, **kwargs
        )
        # last step only
        actions_mean = actions_mean[:, -1]
        self.steps += 1
        return actions_mean

    def act_pure_inference(self, obs_dict, episode_attnmask=None, **kwargs):
        """
        Pure inference mode for temporal models like transformer.
        Need to construct obs_dict and episode_attnmask outside the model.
        """
        actions_mean = self.forward(obs_dict=obs_dict, episode_attnmask=episode_attnmask, **kwargs)
        # last step only
        actions_mean = actions_mean[:, -1]
        return actions_mean

    def to_cpu(self):
        if self.running_mean_std is not None:
            self.running_mean_std.to("cpu")
        self.actor = deepcopy(self.actor).to("cpu")
        self.std.to("cpu")

    def init_rollout(self):
        """Initialize the observation buffer for rollout phase."""
        self.obs_dict_buffer = TensorDict()
        self.dones_buffer = None
        self.steps = 0
        # The sequence likelihood starts from the same stationary prior as the
        # physical sampler at every rollout boundary.
        self._temporal_action_noise_context = None

    def clear_rollout(self):
        """Clear the observation buffer after rollout phase."""
        self.obs_dict_buffer = TensorDict()
        self.dones_buffer = None
        self.steps = 0
        del self.distribution

    def eval_mode(self):
        self.is_eval_mode = True

    def train_mode(self):
        self.is_eval_mode = False


class Critic(nn.Module):
    def __init__(
        self, env_config, algo_config, backbone, module_dim_dict={}, running_mean_std=False
    ):
        super(Critic, self).__init__()

        obs_dim_dict = env_config.robot.algo_obs_dim_dict
        self.critic_module = custom_instantiate(
            backbone,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _resolve=False,
        )

        self.running_mean_std = None
        if running_mean_std:
            self.running_mean_std = RunningMeanStd((obs_dim_dict["critic_obs"],), per_channel=True)

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, obs_dict, **kwargs):
        obs_dict = obs_dict
        if self.running_mean_std is not None:
            with torch.no_grad():
                obs_dict["critic_obs"] = self.running_mean_std(obs_dict["critic_obs"])
        value = self.critic(obs_dict, **kwargs)
        return value
