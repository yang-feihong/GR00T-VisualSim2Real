import unittest

import torch

from gr00t.rl.trl.modules.exploration_budget import (
    exploration_budget_dual_loss,
    exploration_variance_budget_loss,
    normalized_action_variance,
)


class ExplorationVarianceBudgetTest(unittest.TestCase):
    def test_quiet_minority_has_no_regularizer_gradient_when_budget_is_met(self):
        scale = torch.ones(100, 2)
        scale[-5:] = 0.1
        std = (scale * torch.tensor([0.2, 0.01])).requires_grad_()

        loss, mean_variance, shortfall = exploration_variance_budget_loss(
            std,
            max_std=torch.tensor([0.2, 0.01]),
            minimum_variance_fraction=0.9,
            dual_weight=1.0,
        )
        gradient = torch.autograd.grad(loss, std)[0]

        self.assertTrue(torch.all(mean_variance > 0.95))
        self.assertTrue(torch.equal(shortfall, torch.zeros_like(shortfall)))
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_local_surrogate_can_reduce_quiet_states_after_budget_is_met(self):
        scale = torch.ones(100, 1)
        scale[-5:] = 0.2
        std = scale.requires_grad_()
        budget_loss, _, _ = exploration_variance_budget_loss(
            std, 1.0, minimum_variance_fraction=0.8, dual_weight=10.0
        )
        local_surrogate = std[-5:].square().mean()
        gradient = torch.autograd.grad(budget_loss + local_surrogate, std)[0]

        self.assertTrue(torch.equal(gradient[:-5], torch.zeros_like(gradient[:-5])))
        self.assertTrue(torch.all(gradient[-5:] > 0.0))

    def test_budget_activation_matches_quiet_occupancy_bound(self):
        quiet_scale = 0.1
        target = 0.8
        maximum_quiet_fraction = (1.0 - target) / (1.0 - quiet_scale**2)
        self.assertAlmostEqual(maximum_quiet_fraction, 0.2020202, places=6)

        for quiet_count, should_violate in ((20, False), (21, True)):
            scale = torch.ones(100, 1)
            scale[-quiet_count:] = quiet_scale
            _, _, shortfall = exploration_variance_budget_loss(
                scale, 1.0, minimum_variance_fraction=target, dual_weight=1.0
            )
            self.assertEqual(bool(shortfall.item() > 0.0), should_violate)

    def test_global_collapse_receives_only_variance_increasing_gradient(self):
        max_std = torch.tensor([0.2, 0.01])
        std = (torch.ones(32, 2) * 0.5 * max_std).requires_grad_()

        loss, mean_variance, shortfall = exploration_variance_budget_loss(
            std,
            max_std=max_std,
            minimum_variance_fraction=0.8,
            dual_weight=0.3,
        )
        gradient = torch.autograd.grad(loss, std)[0]

        self.assertTrue(torch.allclose(mean_variance, torch.full((2,), 0.25)))
        self.assertTrue(torch.allclose(shortfall, torch.full((2,), 0.55)))
        self.assertGreater(float(loss), 0.0)
        self.assertTrue(torch.all(gradient < 0.0))

    def test_action_dimensions_have_independent_budgets(self):
        max_std = torch.tensor([0.2, 0.1])
        scale = torch.tensor([[1.0, 0.2], [1.0, 0.2]])
        std = (scale * max_std).requires_grad_()

        loss, _, shortfall = exploration_variance_budget_loss(
            std,
            max_std=max_std,
            minimum_variance_fraction=torch.tensor([0.8, 0.8]),
            dual_weight=torch.tensor([1.0, 2.0]),
        )
        gradient = torch.autograd.grad(loss, std)[0]

        self.assertEqual(float(shortfall[0]), 0.0)
        self.assertGreater(float(shortfall[1]), 0.0)
        self.assertTrue(torch.equal(gradient[:, 0], torch.zeros(2)))
        self.assertTrue(torch.all(gradient[:, 1] < 0.0))

    def test_sequence_and_batch_dimensions_are_both_aggregated(self):
        std = torch.tensor(
            [
                [[0.2, 0.05], [0.1, 0.05]],
                [[0.2, 0.025], [0.1, 0.025]],
            ]
        )
        actual = normalized_action_variance(std, torch.tensor([0.2, 0.05]))
        expected = torch.tensor([(1.0 + 0.25 + 1.0 + 0.25) / 4.0, 0.625])
        self.assertTrue(torch.allclose(actual, expected))

    def test_dual_multiplier_increases_only_for_violated_budget(self):
        log_dual = torch.zeros(2, requires_grad=True)
        loss, dual = exploration_budget_dual_loss(
            log_dual,
            mean_variance=torch.tensor([0.4, 0.9]),
            minimum_variance_fraction=0.8,
        )
        gradient = torch.autograd.grad(loss, log_dual)[0]

        self.assertTrue(torch.all(dual > 0.0))
        self.assertLess(float(gradient[0]), 0.0)
        self.assertGreater(float(gradient[1]), 0.0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            normalized_action_variance(torch.ones(2), 1.0)
        with self.assertRaises(ValueError):
            normalized_action_variance(torch.ones(2, 3), torch.ones(2))
        with self.assertRaises(ValueError):
            normalized_action_variance(torch.full((2, 3), 1.1), 1.0)
        with self.assertRaises(ValueError):
            exploration_variance_budget_loss(
                torch.ones(2, 3), 1.0, minimum_variance_fraction=1.1, dual_weight=1.0
            )
        with self.assertRaises(ValueError):
            exploration_variance_budget_loss(
                torch.ones(2, 3), 1.0, minimum_variance_fraction=0.5, dual_weight=-1.0
            )
        with self.assertRaises(ValueError):
            exploration_budget_dual_loss(
                torch.zeros(2), torch.zeros(3), minimum_variance_fraction=0.5
            )


if __name__ == "__main__":
    unittest.main()
