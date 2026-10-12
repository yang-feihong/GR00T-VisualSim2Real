import math
import unittest

import torch

from gr00t.rl.trl.modules.temporal_action_noise import (
    advance_standardized_ar1_context,
    conditional_gaussian_log_prob,
    conditional_gaussian_parameters,
    marginal_temporal_gaussian_log_prob,
    marginal_temporal_gaussian_parameters,
    paired_temporal_gaussian_parameters,
    validate_temporal_noise_parameters,
)


class TemporalActionNoiseTest(unittest.TestCase):
    def test_parameter_validation(self):
        validate_temporal_noise_parameters(0.95, 0.9)
        for rho in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                validate_temporal_noise_parameters(rho, 0.5)
        for fraction in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                validate_temporal_noise_parameters(0.9, fraction)

    def test_done_replaces_context_with_fresh_stationary_sample(self):
        context = torch.tensor([[5.0, -5.0], [1.0, 2.0]])
        innovation = torch.tensor([[0.25, -0.5], [0.5, -0.25]])
        dones = torch.tensor([True, False])
        result = advance_standardized_ar1_context(
            context, dones, innovation, rho=0.8
        )
        expected_second = 0.8 * context[1] + 0.6 * innovation[1]
        self.assertTrue(torch.equal(result[0], innovation[0]))
        self.assertTrue(torch.allclose(result[1], expected_second))

    def test_conditional_components_preserve_marginal_variance(self):
        marginal_std = torch.tensor([[0.2, 0.5]], dtype=torch.float64)
        base_mean = torch.zeros_like(marginal_std)
        context = torch.zeros_like(marginal_std)
        fraction = 0.84
        conditional_mean, conditional_std = conditional_gaussian_parameters(
            base_mean, marginal_std, context, fraction
        )
        context_variance = marginal_std.square() * fraction
        independent_variance = conditional_std.square()
        self.assertTrue(torch.equal(conditional_mean, base_mean))
        self.assertTrue(
            torch.allclose(
                context_variance + independent_variance,
                marginal_std.square(),
                atol=1.0e-14,
            )
        )

    def test_per_action_fraction_preserves_each_marginal_variance(self):
        marginal_std = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float64)
        base_mean = torch.zeros_like(marginal_std)
        context = torch.ones_like(marginal_std)
        fraction = torch.tensor([0.0, 0.5, 0.9], dtype=torch.float64)
        conditional_mean, conditional_std = conditional_gaussian_parameters(
            base_mean, marginal_std, context, fraction
        )
        context_variance = marginal_std.square() * fraction
        self.assertTrue(
            torch.allclose(
                context_variance + conditional_std.square(),
                marginal_std.square(),
                atol=1.0e-14,
            )
        )
        self.assertEqual(float(conditional_mean[0, 0]), 0.0)
        self.assertGreater(float(conditional_mean[0, 2]), 0.0)

    def test_sampled_action_noise_has_expected_temporal_correlation(self):
        generator = torch.Generator().manual_seed(7)
        num_envs = 20000
        rho = 0.92
        fraction = 0.85
        context = torch.randn(num_envs, 1, generator=generator)
        previous_noise = None
        pairs = []
        for _ in range(12):
            innovation = torch.randn(num_envs, 1, generator=generator)
            context = advance_standardized_ar1_context(
                context,
                torch.zeros(num_envs, dtype=torch.bool),
                innovation,
                rho,
            )
            white_noise = torch.randn(num_envs, 1, generator=generator)
            action_noise = (
                math.sqrt(fraction) * context
                + math.sqrt(1.0 - fraction) * white_noise
            )
            if previous_noise is not None:
                pairs.append(torch.cat((previous_noise, action_noise), dim=-1))
            previous_noise = action_noise
        pairs = torch.cat(pairs, dim=0)
        variance = pairs[:, 1].var(unbiased=False)
        covariance = (pairs[:, 0] * pairs[:, 1]).mean()
        expected_lag_one = fraction * rho
        self.assertAlmostEqual(float(variance), 1.0, delta=0.02)
        self.assertAlmostEqual(float(covariance / variance), expected_lag_one, delta=0.02)

    def test_log_prob_ratio_is_exact_when_conditioning_context_is_stored(self):
        actions = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
        old_mean = torch.tensor([[0.1, -0.1]], dtype=torch.float64)
        new_mean = torch.tensor([[0.12, -0.08]], dtype=torch.float64)
        old_std = torch.tensor([[0.2, 0.3]], dtype=torch.float64)
        new_std = torch.tensor([[0.22, 0.27]], dtype=torch.float64)
        context = torch.tensor([[0.5, -1.0]], dtype=torch.float64)
        fraction = 0.8

        old_log_prob = conditional_gaussian_log_prob(
            actions, old_mean, old_std, context, fraction
        )
        repeated_log_prob = conditional_gaussian_log_prob(
            actions, old_mean, old_std, context, fraction
        )
        self.assertTrue(torch.equal(torch.exp(repeated_log_prob - old_log_prob), torch.ones(1)))

        new_log_prob = conditional_gaussian_log_prob(
            actions, new_mean, new_std, context, fraction
        )
        new_cond_mean, new_cond_std = conditional_gaussian_parameters(
            new_mean, new_std, context, fraction
        )
        old_cond_mean, old_cond_std = conditional_gaussian_parameters(
            old_mean, old_std, context, fraction
        )

        def manual_log_prob(mean, std):
            normalized = (actions - mean) / std
            return (
                -0.5 * normalized.square()
                - torch.log(std)
                - 0.5 * math.log(2.0 * math.pi)
            ).sum(dim=-1)

        expected_ratio = torch.exp(
            manual_log_prob(new_cond_mean, new_cond_std)
            - manual_log_prob(old_cond_mean, old_cond_std)
        )
        self.assertTrue(
            torch.allclose(
                torch.exp(new_log_prob - old_log_prob),
                expected_ratio,
                atol=1.0e-12,
            )
        )

    def test_dense_recurrent_rollout_shape_reconstructs_identical_log_prob(self):
        generator = torch.Generator().manual_seed(11)
        shape = (3, 7, 4)
        base_mean = torch.randn(shape, generator=generator)
        marginal_std = torch.rand(shape, generator=generator) + 0.1
        context = torch.randn(shape, generator=generator)
        conditional_mean, conditional_std = conditional_gaussian_parameters(
            base_mean, marginal_std, context, 0.85
        )
        actions = conditional_mean + conditional_std * torch.randn(
            shape, generator=generator
        )
        rollout_log_prob = conditional_gaussian_log_prob(
            actions, base_mean, marginal_std, context, 0.85
        )
        update_log_prob = conditional_gaussian_log_prob(
            actions.clone(),
            base_mean.clone(),
            marginal_std.clone(),
            context.clone(),
            0.85,
        )
        self.assertEqual(tuple(rollout_log_prob.shape), shape[:-1])
        self.assertTrue(torch.equal(rollout_log_prob, update_log_prob))
        self.assertTrue(
            torch.equal(
                torch.exp(update_log_prob - rollout_log_prob),
                torch.ones(shape[:-1]),
            )
        )

    def test_marginal_sequence_log_prob_matches_dense_gaussian(self):
        dtype = torch.float64
        rho = 0.8
        fraction = 0.7
        actions = torch.tensor(
            [[[0.2], [-0.1], [0.3], [0.15]]], dtype=dtype
        )
        mean = torch.tensor(
            [[[0.05], [0.02], [-0.03], [0.01]]], dtype=dtype
        )
        std = torch.full_like(actions, 0.4)
        dones = torch.zeros(1, 4, dtype=torch.bool)
        actual = marginal_temporal_gaussian_log_prob(
            actions, mean, std, dones, rho, fraction
        ).sum(dim=1)

        time = torch.arange(4, dtype=dtype)
        correlation = fraction * rho ** (time[:, None] - time[None, :]).abs()
        correlation.diagonal().add_(1.0 - fraction)
        covariance = std[0, :, 0, None] * correlation * std[0, :, 0, None].T
        expected = torch.distributions.MultivariateNormal(
            mean[0, :, 0], covariance_matrix=covariance
        ).log_prob(actions[0, :, 0])
        self.assertTrue(torch.allclose(actual, expected[None], atol=1.0e-12))

    def test_marginal_sequence_gradient_matches_dense_gaussian(self):
        dtype = torch.float64
        rho = 0.9
        fraction = 0.85
        actions = torch.tensor(
            [[[0.2], [-0.1], [0.3]]], dtype=dtype
        )
        mean = torch.tensor(
            [[[0.05], [0.02], [-0.03]]], dtype=dtype, requires_grad=True
        )
        std = torch.full_like(actions, 0.3)
        dones = torch.zeros(1, 3, dtype=torch.bool)
        log_prob = marginal_temporal_gaussian_log_prob(
            actions, mean, std, dones, rho, fraction
        ).sum()
        actual_gradient = torch.autograd.grad(log_prob, mean)[0]

        time = torch.arange(3, dtype=dtype)
        correlation = fraction * rho ** (time[:, None] - time[None, :]).abs()
        correlation.diagonal().add_(1.0 - fraction)
        covariance = 0.3**2 * correlation
        expected_gradient = torch.linalg.solve(
            covariance, actions[0, :, 0] - mean.detach()[0, :, 0]
        )
        self.assertTrue(
            torch.allclose(
                actual_gradient[0, :, 0], expected_gradient, atol=1.0e-11
            )
        )

    def test_marginal_filter_resets_after_done(self):
        actions = torch.tensor([[[1.0], [1.0], [1.0]]], dtype=torch.float64)
        mean = torch.zeros_like(actions)
        std = torch.ones_like(actions)
        dones = torch.tensor([[False, True, False]])
        conditional_mean, conditional_std = marginal_temporal_gaussian_parameters(
            actions, mean, std, dones, rho=0.9, correlated_fraction=0.8
        )
        self.assertGreater(float(conditional_mean[0, 1, 0]), 0.0)
        self.assertEqual(float(conditional_mean[0, 2, 0]), 0.0)
        self.assertEqual(float(conditional_std[0, 2, 0]), 1.0)

    def test_zero_fraction_steps_do_not_receive_future_sequence_credit(self):
        actions = torch.tensor(
            [[[0.1], [-0.2], [0.4], [0.5]]], dtype=torch.float64
        )
        mean = torch.zeros_like(actions, requires_grad=True)
        std = torch.full_like(actions, 0.2)
        dones = torch.zeros(1, 4, dtype=torch.bool)
        fraction = torch.tensor(
            [[[0.0], [0.0], [0.9], [0.9]]], dtype=torch.float64
        )
        log_prob = marginal_temporal_gaussian_log_prob(
            actions, mean, std, dones, rho=0.95, correlated_fraction=fraction
        )
        future_gradient = torch.autograd.grad(log_prob[:, 2:].sum(), mean)[0]

        self.assertTrue(torch.equal(future_gradient[:, :2], torch.zeros_like(mean[:, :2])))
        self.assertGreater(float(future_gradient[:, 2:].abs().sum()), 0.0)

    def test_marginal_ratio_is_one_for_identical_parameters(self):
        generator = torch.Generator().manual_seed(19)
        actions = torch.randn(3, 7, 4, generator=generator)
        mean = torch.randn(3, 7, 4, generator=generator)
        std = torch.rand(3, 7, 4, generator=generator) + 0.2
        dones = torch.zeros(3, 7, dtype=torch.bool)
        dones[0, 3] = True
        old = marginal_temporal_gaussian_log_prob(
            actions, mean, std, dones, 0.95, 0.9
        )
        new = marginal_temporal_gaussian_log_prob(
            actions.clone(), mean.clone(), std.clone(), dones.clone(), 0.95, 0.9
        )
        self.assertTrue(torch.equal(torch.exp(new - old), torch.ones_like(old)))

    def test_paired_filter_uses_distinct_old_and_new_fractions(self):
        generator = torch.Generator().manual_seed(29)
        actions = torch.randn(2, 6, 3, generator=generator, dtype=torch.float64)
        mean = torch.randn(2, 6, 3, generator=generator, dtype=torch.float64)
        std = torch.rand(2, 6, 3, generator=generator, dtype=torch.float64) + 0.2
        dones = torch.zeros(2, 6, dtype=torch.bool)
        dones[1, 2] = True
        old_fraction = torch.full_like(actions, 0.2)
        new_fraction = torch.full_like(actions, 0.85, requires_grad=True)

        old_mean, old_std, new_mean, new_std = paired_temporal_gaussian_parameters(
            actions,
            mean,
            std,
            old_fraction,
            mean,
            std,
            new_fraction,
            dones,
            rho=0.95,
        )
        old_log_prob = torch.distributions.Normal(old_mean, old_std).log_prob(
            actions
        ).sum(dim=-1)
        new_log_prob = torch.distributions.Normal(new_mean, new_std).log_prob(
            actions
        ).sum(dim=-1)
        direct_old = marginal_temporal_gaussian_log_prob(
            actions, mean, std, dones, 0.95, old_fraction
        )
        direct_new = marginal_temporal_gaussian_log_prob(
            actions, mean, std, dones, 0.95, new_fraction
        )

        self.assertTrue(torch.allclose(old_log_prob, direct_old, atol=1.0e-12))
        self.assertTrue(torch.allclose(new_log_prob, direct_new, atol=1.0e-12))
        self.assertGreater(float((new_log_prob - old_log_prob).abs().max()), 1.0e-4)
        fraction_gradient = torch.autograd.grad(new_log_prob.sum(), new_fraction)[0]
        self.assertTrue(torch.isfinite(fraction_gradient).all())
        self.assertGreater(float(fraction_gradient.abs().sum()), 0.0)

    def test_paired_filter_ratio_is_one_when_all_parameters_match(self):
        generator = torch.Generator().manual_seed(31)
        actions = torch.randn(3, 5, 2, generator=generator, dtype=torch.float64)
        mean = torch.randn(3, 5, 2, generator=generator, dtype=torch.float64)
        std = torch.rand(3, 5, 2, generator=generator, dtype=torch.float64) + 0.1
        fraction = torch.rand(3, 5, 2, generator=generator, dtype=torch.float64) * 0.9
        dones = torch.zeros(3, 5, dtype=torch.bool)
        dones[0, 1] = True

        old_mean, old_std, new_mean, new_std = paired_temporal_gaussian_parameters(
            actions,
            mean,
            std,
            fraction,
            mean.clone(),
            std.clone(),
            fraction.clone(),
            dones,
            rho=0.9,
        )
        old_log_prob = torch.distributions.Normal(old_mean, old_std).log_prob(
            actions
        ).sum(dim=-1)
        new_log_prob = torch.distributions.Normal(new_mean, new_std).log_prob(
            actions
        ).sum(dim=-1)
        self.assertTrue(
            torch.allclose(
                torch.exp(new_log_prob - old_log_prob),
                torch.ones_like(old_log_prob),
                atol=1.0e-12,
                rtol=0.0,
            )
        )

    def test_zero_fraction_has_finite_zero_boundary_gradient(self):
        fraction = torch.tensor([0.0, 0.8], requires_grad=True)
        sequence_fraction = fraction.reshape(1, 1, 2).expand(1, 2, 2)
        actions = torch.tensor([[[0.2, 0.1], [0.3, -0.2]]])
        mean = torch.zeros_like(actions)
        std = torch.ones_like(actions)
        dones = torch.zeros(1, 2, dtype=torch.bool)

        log_prob = marginal_temporal_gaussian_log_prob(
            actions,
            mean,
            std,
            dones,
            rho=0.95,
            correlated_fraction=sequence_fraction,
        ).sum()
        gradient = torch.autograd.grad(log_prob, fraction)[0]

        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(float(gradient[0]), 0.0)
        self.assertNotEqual(float(gradient[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
