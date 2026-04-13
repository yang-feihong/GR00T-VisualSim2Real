# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.distributions import Normal

from gr00t.rl.utils.running_mean_std import RunningMeanStd

from .encoder_modules import Encoder
from .modules import BaseModule


class PPOActor(nn.Module):
    def _init_actor_module(self):
        pass

    @property
    def actor(self):
        return self.actor_module

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def has_normalized_actions(self):
        return False

    def update_distribution(self, obs_dict):
        pass

    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict)
        return {
            "actions": self.distribution.sample(),
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
        }

    def rollout(self, obs_dict, **kwargs):
        return self.act(obs_dict, **kwargs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        NotImplementedError

    def to_cpu(self):
        self.actor = deepcopy(self.actor).to("cpu")
        self.std.to("cpu")

    def init_rollout(self):
        pass

    def clear_rollout(self):
        pass

    def eval_mode(self):
        pass


class PPOStateActor(PPOActor):
    def __init__(
        self,
        obs_dim_dict,
        module_config_dict,
        num_actions,
        init_noise_std,
        input_key,
        module_dim_dict={},
    ):
        super().__init__()
        module_config_dict = self._process_module_config(module_config_dict, num_actions)

        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict
        self.module_dim_dict = module_dim_dict
        self.input_key = input_key
        self.actor_module = self._init_actor_module()
        self.num_actions = num_actions
        # Action noise
        if module_config_dict.get("freeze_std", False):
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            self.std.requires_grad = False
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _init_actor_module(self):
        """
        kwargs:
            obs_dim_dict, module_config_dict, module_dim_dict
        """

        return BaseModule(self.obs_dim_dict, self.module_config_dict, self.module_dim_dict)

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict["output_dim"]):
            if output_dim == "robot_action_dim":
                module_config_dict["output_dim"][idx] = num_actions
        return module_config_dict

    def update_distribution(self, obs_dict):
        mean = self.actor(obs_dict[self.input_key])
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act_inference(self, obs_dict):
        actions_mean = self.actor(obs_dict[self.input_key])
        return actions_mean


class VisionStateModule(nn.Module):
    def __init__(
        self,
        obs_dim_dict,
        mlp_module_config_dict,
        vision_module_config_dict,
        module_dim_dict,
        input_key,
    ):
        super().__init__()
        self.mlp_module = BaseModule(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=mlp_module_config_dict,
            module_dim_dict=module_dim_dict,
        )
        self.encoder = Encoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=vision_module_config_dict,
            module_dim_dict=module_dim_dict,
        )
        self.vision_module_config_dict = vision_module_config_dict
        self.input_key = input_key

    def forward(self, obs_dict):
        rgb_image = obs_dict["vision_obs"].clone()

        batch_size = rgb_image.shape[0]

        # Check if the encoder is using CNN or MLP architecture
        encoder_type = self.vision_module_config_dict.layer_config.type

        if encoder_type == "CNN":
            # For CNN, reshape and permute the image
            # [B, H, W, C] -> [B, C, H, W]
            # logger.info("Using CNN encoder")
            rgb_image = rgb_image.permute(0, 3, 1, 2)
            latent = self.encoder(rgb_image).reshape(batch_size, -1)
        elif encoder_type == "MLP":
            # logger.info("Using MLP encoder")
            if len(rgb_image.shape) > 2:
                rgb_image = rgb_image.reshape(batch_size, -1)
            latent = self.encoder(rgb_image).reshape(batch_size, -1)
        else:
            raise ValueError(f"Invalid encoder type: {encoder_type}")

        concated_obs = torch.cat([obs_dict[self.input_key], latent], dim=1)
        return self.mlp_module(concated_obs)


class VisionStateWithTransformModule(VisionStateModule):
    def __init__(
        self,
        obs_dim_dict,
        mlp_module_config_dict,
        vision_module_config_dict,
        module_dim_dict,
        transforms_cfg,
        use_data_augmentation=False,
    ):
        super().__init__(
            obs_dim_dict,
            mlp_module_config_dict,
            vision_module_config_dict,
            module_dim_dict,
        )

        import json
        from pathlib import Path

        from gr00t.data.schema import TrainableDatasetMetadata_V1_2

        metadata_path = Path(
            "/mnt/amlfs-03/shared/datasets/lerobot/trl_g1_fix_lower_right_hand/g1_fix_lower_right_hand.bar_20250522/meta/metadata.json"
        )
        with open(metadata_path, "r") as f:
            metadata = TrainableDatasetMetadata_V1_2.model_validate(json.load(f))

        embodiment_tag = "g1_fix_lower_right_hand"

        transform_cfg = deepcopy(transforms_cfg[embodiment_tag])
        transform_cfg.transforms = transform_cfg.transforms[:-1]  # remove model_specific_transform

        self.transform = instantiate(transform_cfg)
        self.transform.set_metadata(metadata)
        if use_data_augmentation:
            self.transform.train()
        else:
            self.transform.eval()

        """ TEMP """
        # We need a unnormalize_transform to unnormalize the action
        unnormalize_transform_cfg = deepcopy(transforms_cfg[embodiment_tag])
        if (
            unnormalize_transform_cfg.transforms[-3]._target_
            == "gr00t.data.transform.StateActionTransform"
        ):
            unnormalize_transform_cfg.transforms = unnormalize_transform_cfg.transforms[-3:]
            self.unnormalize_transform = instantiate(unnormalize_transform_cfg)
            self.unnormalize_transform.set_metadata(metadata)
        else:
            assert (
                unnormalize_transform_cfg.transforms[-3]._target_
                == "gr00t.data.transform.StateActionToTensor"
            ), f"{unnormalize_transform_cfg.transforms[-3]._target_=}"
            for i, transform in enumerate(unnormalize_transform_cfg.transforms):
                assert (
                    transform._target_ != "gr00t.data.transform.StateActionTransform"
                ), f"{i=}: {transform._target_=}"
            self.unnormalize_transform = None
        """ /TEMP """

    def forward(self, obs_dict):
        rgb_image = obs_dict["vision_obs"].clone()  # [B, H, W, C], float32
        device = rgb_image.device
        rgb_image = (rgb_image * 255).to(torch.uint8)  # [B, H, W, C], uint8
        assert 255 >= rgb_image.max() > 200, f"{rgb_image.max()=}"
        data_dict = {
            "video.ego_view_res256": rgb_image.cpu().numpy(),
        }
        if "gt_actions" in obs_dict:
            # Normalize the teacher action
            data_dict["action.action"] = obs_dict["gt_actions"].cpu().numpy()
        data_dict = self.transform(data_dict)
        rgb_image = data_dict["video"]  # [B, 1, H, W, C]
        assert rgb_image.shape[1] == 1, f"{rgb_image.shape=}"
        rgb_image = torch.from_numpy(rgb_image).to(device)
        rgb_image = rgb_image.squeeze(dim=1)  # [B, H, W, C], uint8
        rgb_image = rgb_image.to(torch.float32) / 255.0  # [B, H, W, C], float32

        obs_dict = deepcopy(obs_dict)
        obs_dict["vision_obs"] = rgb_image

        # forward the model
        result = super().forward(obs_dict)

        return_dict = {}
        """
        This is for managing normalized/not normalized actions/gt_actions
        
        There are two cases:
        1. Only has "actions": then the action is not normalized, and we learn the unnormalized action
        2. Has 2 or 3 keys:
            "actions": unnormalized action
            "normalized_actions": normalized action
            (optional) "normalized_gt_actions": normalized gt_action, useful for distillation
        """
        if self.unnormalize_transform is not None:
            # Case 2: Normalizing the gt_action and unnormalizing the action
            if "gt_actions" in obs_dict:
                assert "action" in data_dict, f"{data_dict.keys()=}, {obs_dict.keys()=}"
                # In theory it should be -1.0 to 1.0, but after all we're not using the strict bounds, so it's possible that the action is out of bounds by a bit
                # We'll just check that it's not too far off
                assert (
                    data_dict["action"].min() > -2.0 and data_dict["action"].max() < 2.0
                ), f"{data_dict['action'].min()=}, {data_dict['action'].max()=}"
                return_dict["normalized_gt_actions"] = data_dict["action"]

            # Unnormalize the action
            unnormalized_pred = self.unnormalize_transform.unapply(dict(action=result))
            unnormalized_actions = unnormalized_pred["action.action"]

            # Make the return dict
            return_dict["actions"] = unnormalized_actions
            return_dict["normalized_actions"] = result
        else:
            # Case 1: The action is not normalized, and we learn the unnormalized action
            return_dict["actions"] = result

        return return_dict


class PPOVisionStateActor(PPOStateActor):
    def __init__(
        self,
        obs_dim_dict,
        mlp_module_config_dict,
        vision_module_config_dict,
        num_actions,
        init_noise_std,
        input_key,
        module_dim_dict={},
    ):
        self.vision_module_config_dict = vision_module_config_dict
        super().__init__(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=mlp_module_config_dict,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            input_key=input_key,
            module_dim_dict=module_dim_dict,
        )

    def _init_actor_module(self):
        return VisionStateModule(
            obs_dim_dict=self.obs_dim_dict,
            mlp_module_config_dict=self.module_config_dict,
            vision_module_config_dict=self.vision_module_config_dict,
            module_dim_dict=self.module_dim_dict,
            input_key=self.input_key,
        )

    def update_distribution(self, obs_dict):
        mean = self.actor(obs_dict)
        self.distribution = Normal(mean, mean * 0.0 + self.std)


class PPOCritic(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict):
        super().__init__()

        self.critic_module = BaseModule(obs_dim_dict, module_config_dict)
        if module_config_dict.get("running_mean_std", False):
            self.running_mean_std = RunningMeanStd((obs_dim_dict["critic_obs"],), per_channel=True)
            self.critic_module = nn.Sequential(self.running_mean_std, self.critic_module)

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, obs_dict, **kwargs):
        value = self.critic(obs_dict["critic_obs"])
        return value


#


class PPOStateActorFixSigma(PPOStateActor):
    def __init__(
        self, obs_dim_dict, module_config_dict, num_actions, input_key, module_dim_dict={}
    ):
        super().__init__(
            obs_dim_dict, module_config_dict, num_actions, 0.0, input_key, module_dim_dict
        )

    def update_distribution(self, obs_dict):
        mean = self.actor(obs_dict[self.input_key])
        self.distribution = mean

    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict)
        return {
            "actions": self.distribution,
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
        }

    @property
    def action_mean(self):
        return self.distribution

    @property
    def action_std(self):
        return self.distribution * 0.0 + self.std

    def get_actions_log_prob(self, actions):
        return torch.ones(actions.shape[0], dtype=actions.dtype, device=actions.device)


class PPOVisionStateActorFixSigma(PPOStateActorFixSigma):
    def __init__(
        self,
        obs_dim_dict,
        mlp_module_config_dict,
        vision_module_config_dict,
        num_actions,
        input_key,
        module_dim_dict={},
    ):
        self.vision_module_config_dict = vision_module_config_dict
        super().__init__(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=mlp_module_config_dict,
            num_actions=num_actions,
            input_key=input_key,
            module_dim_dict=module_dim_dict,
        )

    def _init_actor_module(self):
        return VisionStateModule(
            obs_dim_dict=self.obs_dim_dict,
            mlp_module_config_dict=self.module_config_dict,
            vision_module_config_dict=self.vision_module_config_dict,
            module_dim_dict=self.module_dim_dict,
            input_key=self.input_key,
        )

    def update_distribution(self, obs_dict):
        mean = self.actor(obs_dict)
        self.distribution = mean


class PPOVisionStateActorWithTransformFixSigma(PPOVisionStateActorFixSigma):
    def __init__(
        self,
        obs_dim_dict,
        mlp_module_config_dict,
        vision_module_config_dict,
        num_actions,
        input_key,
        transforms_cfg,
        use_data_augmentation,
        image_resolution,
        module_dim_dict={},
    ):
        self.transforms_cfg = transforms_cfg
        self.use_data_augmentation = use_data_augmentation

        obs_dim_dict_copy = deepcopy(obs_dim_dict)
        obs_dim_dict_copy["vision_obs"] = image_resolution * image_resolution * 3
        super().__init__(
            obs_dim_dict=obs_dim_dict_copy,
            mlp_module_config_dict=mlp_module_config_dict,
            vision_module_config_dict=vision_module_config_dict,
            num_actions=num_actions,
            input_key=input_key,
            module_dim_dict=module_dim_dict,
        )

    def _init_actor_module(self):
        return VisionStateWithTransformModule(
            obs_dim_dict=self.obs_dim_dict,
            mlp_module_config_dict=self.module_config_dict,
            vision_module_config_dict=self.vision_module_config_dict,
            module_dim_dict=self.module_dim_dict,
            transforms_cfg=self.transforms_cfg,
            use_data_augmentation=self.use_data_augmentation,
        )

    @property
    def has_normalized_actions(self):
        return self.actor.unnormalize_transform is not None

    def update_distribution(self, obs_dict):
        return_dict = self.actor(obs_dict)
        self.step_result_dict = return_dict
        self.distribution = return_dict["actions"]

    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict)
        step_result_dict = self.step_result_dict
        return {
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
            **step_result_dict,
        }
