import unittest

import torch
import torch.nn as nn
from tensordict import TensorDict

from gr00t.rl.agents.modules.data_utils import RolloutStorage
from gr00t.rl.trl.modules.actor_critic_modules import Actor
from gr00t.rl.trl.modules.actor_critic_modules_recurrent import RecurrentActor
from gr00t.rl.trl.modules.memory import Memory
from gr00t.rl.trl.modules.temporal_action_noise import (
    conditional_gaussian_parameters,
    marginal_temporal_gaussian_log_prob,
)
from gr00t.rl.trl.utils.rl import split_and_pad_trajectories


def make_actor(temporal_enabled, rho=0.9, fraction=0.8):
    actor = Actor.__new__(Actor)
    nn.Module.__init__(actor)
    actor.temporal_action_noise_enabled = temporal_enabled
    actor.temporal_action_noise_rho = rho
    actor.temporal_action_noise_correlated_fraction = fraction
    actor._temporal_action_noise_context = None
    actor._base_action_mean = None
    actor._marginal_action_std = None
    actor._global_action_std = None
    actor._action_noise_attenuation = None
    actor._action_noise_context = None
    actor._effective_temporal_action_noise_fraction = None
    actor.distribution = None
    return actor


def make_recurrent_actor(rho=0.9, fraction=0.8):
    actor = RecurrentActor.__new__(RecurrentActor)
    nn.Module.__init__(actor)
    actor.input_key = "actor_obs"
    actor.running_mean_std = None
    actor.algo_config = {}
    actor.action_residual_obs_slice = None
    actor.action_residual_stage_masks = None
    actor.action_noise_stage_obs_slice = None
    actor.action_noise_stage_scales = None
    actor.state_dependent_action_noise = False
    actor.num_actions = 2
    actor.actor_module = nn.Linear(5, actor.num_actions)
    actor.memory = Memory(input_size=3, type="lstm", hidden_size=5)
    actor.std = nn.Parameter(torch.tensor([0.25, 0.4]))
    actor.clamp_noise_std = False
    actor.temporal_action_noise_enabled = True
    actor.temporal_action_noise_rho = rho
    actor.temporal_action_noise_correlated_fraction = fraction
    actor._temporal_action_noise_context = None
    actor._base_action_mean = None
    actor._marginal_action_std = None
    actor._global_action_std = None
    actor._action_noise_attenuation = None
    actor._action_noise_context = None
    actor._effective_temporal_action_noise_fraction = None
    actor.distribution = None
    actor.obs_dict_buffer = TensorDict()
    actor.dones_buffer = None
    actor.max_rollout_history = 1
    actor.steps = 0
    return actor


class ActorTemporalNoiseInterfaceTest(unittest.TestCase):
    def test_state_dependent_fraction_is_bounded_and_trainable(self):
        actor = make_actor(True, fraction=0.4)
        actor.input_key = "actor_obs"
        actor.running_mean_std = None
        actor.action_residual_stage_masks = None
        actor.state_dependent_temporal_action_noise = True
        actor.temporal_action_noise_fraction_module = nn.Linear(3, 2)
        actor.register_buffer(
            "state_dependent_temporal_action_noise_min_fraction",
            torch.tensor([0.1, 0.2]),
        )
        actor.register_buffer(
            "state_dependent_temporal_action_noise_max_fraction",
            torch.tensor([0.8, 0.9]),
        )
        obs = {"actor_obs": torch.randn(7, 3)}
        fraction = actor._compute_temporal_action_noise_fraction(
            torch.zeros(7, 2), obs
        )

        self.assertTrue(torch.all(fraction > torch.tensor([0.1, 0.2])))
        self.assertTrue(torch.all(fraction < torch.tensor([0.8, 0.9])))
        gradient = torch.autograd.grad(
            fraction.square().sum(),
            actor.temporal_action_noise_fraction_module.weight,
        )[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_fraction_head_initialization_reproduces_requested_constant(self):
        actor = make_actor(True, fraction=0.4)
        actor.input_key = "actor_obs"
        actor.running_mean_std = None
        actor.action_residual_stage_masks = None
        actor.num_actions = 2
        actor.state_dependent_temporal_action_noise = True
        actor.temporal_action_noise_fraction_module = nn.Sequential(
            nn.Linear(3, 5), nn.SiLU(), nn.Linear(5, 2)
        )
        actor.register_buffer(
            "state_dependent_temporal_action_noise_min_fraction",
            torch.tensor([0.05, 0.10]),
        )
        actor.register_buffer(
            "state_dependent_temporal_action_noise_max_fraction",
            torch.tensor([0.90, 0.95]),
        )
        requested = torch.tensor([0.30, 0.80])
        actor._initialize_temporal_action_noise_fraction(requested)

        fraction = actor._compute_temporal_action_noise_fraction(
            torch.zeros(11, 2), {"actor_obs": torch.randn(11, 3)}
        )
        self.assertTrue(
            torch.allclose(
                fraction,
                requested.expand_as(fraction),
                atol=1.0e-7,
                rtol=0.0,
            )
        )

    def test_constant_fraction_dimension_remains_exact(self):
        actor = make_actor(True, fraction=0.4)
        actor.input_key = "actor_obs"
        actor.running_mean_std = None
        actor.action_residual_stage_masks = None
        actor.num_actions = 2
        actor.state_dependent_temporal_action_noise = True
        actor.temporal_action_noise_fraction_module = nn.Linear(3, 2)
        actor.register_buffer(
            "state_dependent_temporal_action_noise_min_fraction",
            torch.tensor([0.0, 0.1]),
        )
        actor.register_buffer(
            "state_dependent_temporal_action_noise_max_fraction",
            torch.tensor([0.0, 0.9]),
        )
        actor._initialize_temporal_action_noise_fraction(torch.tensor([0.0, 0.6]))

        fraction = actor._compute_temporal_action_noise_fraction(
            torch.zeros(9, 2), {"actor_obs": torch.randn(9, 3)}
        )
        self.assertTrue(torch.equal(fraction[:, 0], torch.zeros(9)))
        self.assertTrue(torch.allclose(fraction[:, 1], torch.full((9,), 0.6)))

    def test_global_capacity_entropy_does_not_train_state_attenuation(self):
        actor = Actor.__new__(Actor)
        nn.Module.__init__(actor)
        actor.input_key = "actor_obs"
        actor.running_mean_std = None
        actor.state_dependent_action_noise = True
        actor.action_noise_stage_obs_slice = None
        actor.action_noise_stage_scales = None
        actor.action_residual_stage_masks = None
        actor.std = nn.Parameter(torch.tensor([0.2, 0.05]))
        actor.register_buffer(
            "state_dependent_action_noise_min_scale", torch.tensor([0.001, 0.001])
        )
        actor.action_noise_scale_module = nn.Linear(3, 2)
        mean = torch.zeros(8, 2)
        obs = {"actor_obs": torch.randn(8, 3)}

        actual_std = actor._compute_action_std(mean, obs)
        capacity_entropy = torch.distributions.Normal(
            torch.zeros_like(actor.global_action_std), actor.global_action_std
        ).entropy().sum()
        actual_entropy = torch.distributions.Normal(
            torch.zeros_like(actual_std), actual_std
        ).entropy().sum()
        capacity_scale_grad = torch.autograd.grad(
            capacity_entropy,
            actor.action_noise_scale_module.weight,
            retain_graph=True,
            allow_unused=True,
        )[0]
        capacity_std_grad = torch.autograd.grad(
            capacity_entropy, actor.std, retain_graph=True
        )[0]
        actual_scale_grad = torch.autograd.grad(
            actual_entropy, actor.action_noise_scale_module.weight
        )[0]

        self.assertIsNone(capacity_scale_grad)
        self.assertTrue(torch.all(capacity_std_grad > 0.0))
        self.assertGreater(float(actual_scale_grad.abs().sum()), 0.0)
        self.assertTrue(torch.all(actual_std <= actor.global_action_std))

    def test_explicit_zero_output_initialization_does_not_require_residual(self):
        actor = Actor.__new__(Actor)
        nn.Module.__init__(actor)
        actor.num_actions = 3
        actor.action_residual_obs_slice = None
        actor.zero_initialize_action_output = True
        actor.actor_module = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 3))

        actor._zero_initialize_action_output()

        output_layer = actor.actor_module[-1]
        self.assertTrue(torch.equal(output_layer.weight, torch.zeros_like(output_layer.weight)))
        self.assertTrue(torch.equal(output_layer.bias, torch.zeros_like(output_layer.bias)))

    def test_default_non_residual_output_initialization_is_unchanged(self):
        actor = Actor.__new__(Actor)
        nn.Module.__init__(actor)
        actor.num_actions = 3
        actor.action_residual_obs_slice = None
        actor.zero_initialize_action_output = False
        actor.actor_module = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 3))
        original_weight = actor.actor_module[-1].weight.detach().clone()

        actor._zero_initialize_action_output()

        self.assertTrue(torch.equal(actor.actor_module[-1].weight, original_weight))

    def test_disabled_interface_is_exactly_the_original_gaussian(self):
        actor = make_actor(False)
        base_mean = torch.tensor([[0.1, -0.2]])
        marginal_std = torch.tensor([[0.3, 0.4]])
        actor._set_action_distribution(base_mean, marginal_std)

        self.assertTrue(torch.equal(actor.action_mean, base_mean))
        self.assertTrue(torch.equal(actor.action_distribution_mean, base_mean))
        self.assertTrue(torch.equal(actor.action_std, marginal_std))
        state = actor._action_state_dict(base_mean.clone())
        self.assertEqual(
            set(state.keys()), {"actions", "action_mean", "action_sigma"}
        )

    def test_enabled_interface_separates_base_and_distribution_mean(self):
        actor = make_actor(True, fraction=0.75)
        base_mean = torch.tensor([[0.1, -0.2]])
        marginal_std = torch.tensor([[0.3, 0.4]])
        context = torch.tensor([[1.0, -0.5]])
        expected_mean, expected_std = conditional_gaussian_parameters(
            base_mean, marginal_std, context, 0.75
        )
        actor._set_action_distribution(base_mean, marginal_std, context)

        self.assertTrue(torch.equal(actor.action_mean, base_mean))
        self.assertTrue(torch.allclose(actor.action_distribution_mean, expected_mean))
        self.assertTrue(torch.allclose(actor.action_std, expected_std))
        state = actor._action_state_dict(expected_mean.clone())
        self.assertEqual(
            set(state.keys()),
            {
                "actions",
                "action_mean",
                "action_sigma",
                "action_temporal_fraction",
            },
        )
        self.assertTrue(torch.equal(state["action_mean"], base_mean))
        self.assertTrue(torch.equal(state["action_sigma"], marginal_std))
        self.assertTrue(
            torch.equal(
                state["action_temporal_fraction"],
                torch.full_like(marginal_std, 0.75),
            )
        )

    def test_enabled_interface_rejects_missing_context(self):
        actor = make_actor(True)
        with self.assertRaises(ValueError):
            actor._set_action_distribution(
                torch.zeros(2, 3), torch.ones(2, 3), None
            )

    def test_marginalized_update_distribution_uses_base_parameters(self):
        actor = make_actor(True)
        base_mean = torch.tensor([[0.1, -0.2]])
        marginal_std = torch.tensor([[0.3, 0.4]])
        actor._set_action_distribution(
            base_mean,
            marginal_std,
            marginalize_temporal=True,
        )
        self.assertTrue(torch.equal(actor.action_distribution_mean, base_mean))
        self.assertTrue(torch.equal(actor.action_std, marginal_std))

    def test_rollout_boundary_resets_temporal_sampler(self):
        actor = make_actor(True)
        actor._temporal_action_noise_context = torch.ones(2, 3)
        actor.init_rollout()
        self.assertIsNone(actor._temporal_action_noise_context)

    def test_recurrent_storage_minibatch_reconstructs_rollout_log_prob(self):
        torch.manual_seed(23)
        num_envs = 7
        num_steps = 6
        actor = make_recurrent_actor()
        storage = RolloutStorage(num_envs, num_steps)
        storage.register_key("actor_obs", shape=(3,))
        storage.register_key("actions", shape=(actor.num_actions,))
        storage.register_key("action_mean", shape=(actor.num_actions,))
        storage.register_key("action_sigma", shape=(actor.num_actions,))
        storage.register_key(
            "action_temporal_fraction", shape=(actor.num_actions,)
        )

        dones = torch.zeros(num_envs, dtype=torch.bool)
        for _ in range(num_steps):
            obs = torch.randn(num_envs, 3)
            state = actor.rollout(
                TensorDict({"actor_obs": obs}, batch_size=[num_envs]),
                cur_dones=dones,
            )
            storage.update_key("actor_obs", obs)
            storage.update_key("actions", state["actions"].detach())
            storage.update_key("action_mean", state["action_mean"].detach())
            storage.update_key("action_sigma", state["action_sigma"].detach())
            storage.update_key(
                "action_temporal_fraction",
                state["action_temporal_fraction"].detach(),
            )
            storage.increment_step()

        env_order = torch.randperm(num_envs)
        update_dones = torch.zeros(num_envs, num_steps, dtype=torch.bool)[env_order]
        observations = storage.actor_obs.transpose(0, 1)[env_order]
        actions = storage.actions.transpose(0, 1)[env_order]
        old_mean = storage.action_mean.transpose(0, 1)[env_order]
        old_sigma = storage.action_sigma.transpose(0, 1)[env_order]
        temporal_fraction = storage.action_temporal_fraction.transpose(0, 1)[
            env_order
        ]
        old_log_prob = marginal_temporal_gaussian_log_prob(
            actions,
            old_mean,
            old_sigma,
            update_dones,
            actor.temporal_action_noise_rho,
            temporal_fraction,
        )

        padded_obs, masks = split_and_pad_trajectories(
            observations.transpose(0, 1), update_dones.transpose(0, 1)
        )
        actor.memory.reset()
        actor.act(
            TensorDict(
                {"actor_obs": padded_obs.transpose(0, 1)},
                batch_size=list(padded_obs.transpose(0, 1).shape[:2]),
            ),
            masks=masks.transpose(0, 1),
            original_dones=update_dones,
            marginalize_temporal=True,
        )
        repeated_log_prob = marginal_temporal_gaussian_log_prob(
            actions,
            actor.action_mean,
            actor.marginal_action_std,
            update_dones,
            actor.temporal_action_noise_rho,
            temporal_fraction,
        )

        log_prob_error = (repeated_log_prob - old_log_prob).abs()
        self.assertLess(float(log_prob_error.max()), 2.0e-6)
        self.assertTrue(
            torch.allclose(
                torch.exp(repeated_log_prob - old_log_prob),
                torch.ones_like(old_log_prob),
                atol=2.0e-6,
                rtol=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
