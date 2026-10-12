import unittest

import torch

from gr00t.rl.envs.door.stage_conditions import (
    bilateral_contact_force_progress,
    bottleneck_balanced_progress,
    excessive_relative_speed_penalty,
    grasp_collision_aware_position_progress,
    grasp_completion_condition,
    grasp_enclosure_progress,
    open_latch_completion_condition,
    opposed_handle_contact,
    walk_to_door_completion_condition,
)


class DoorOpenStageConditionTest(unittest.TestCase):
    def test_relative_speed_penalty_is_zero_inside_safe_band(self):
        penalty = excessive_relative_speed_penalty(
            torch.tensor([[0.12, 0.16, 0.0], [0.0, 0.0, 0.0]]),
            safe_speed=0.2,
        )
        self.assertTrue(torch.allclose(penalty, torch.zeros(2)))

    def test_relative_speed_penalty_grows_smoothly_above_safe_band(self):
        velocity = torch.tensor([[0.3, 0.0, 0.0]], requires_grad=True)
        penalty = excessive_relative_speed_penalty(velocity, safe_speed=0.2)
        self.assertAlmostEqual(penalty.item(), 0.25, places=6)
        penalty.backward()
        self.assertGreater(velocity.grad[0, 0].item(), 0.0)

    def test_walk_completion_requires_base_and_middle_arm_stability(self):
        result = walk_to_door_completion_condition(
            operation_distance=torch.tensor([0.15, 0.17, 0.15, 0.15, 0.15]),
            base_yaw_error=torch.tensor([0.07, 0.07, 0.09, 0.07, 0.07]),
            handle_contact_free=torch.tensor([True] * 5),
            panel_contact_free=torch.tensor([True] * 5),
            base_motion=torch.tensor([0.04] * 5),
            middle_arm_error=torch.tensor([0.04, 0.04, 0.04, 0.06, 0.04]),
            middle_arm_speed=torch.tensor([0.07, 0.07, 0.07, 0.07, 0.09]),
        )
        self.assertTrue(
            torch.equal(result, torch.tensor([True, False, False, False, False]))
        )

    def test_grasp_requires_base_range_and_bilateral_grasp(self):
        result = grasp_completion_condition(
            operation_distance=torch.tensor([0.15, 0.17, 0.15]),
            base_yaw_error=torch.tensor([0.07, 0.07, 0.09]),
            precise_bilateral_grasp=torch.tensor([True, True, True]),
        )
        self.assertTrue(
            torch.equal(result, torch.tensor([True, False, False]))
        )

    def test_grasp_does_not_gate_on_base_motion(self):
        parameters = grasp_completion_condition.__code__.co_varnames[
            : grasp_completion_condition.__code__.co_argcount
        ]
        self.assertNotIn("base_motion", parameters)

    def test_bottleneck_balanced_progress_credits_every_component(self):
        baseline = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        baseline_score = bottleneck_balanced_progress(baseline)
        gains = []
        for component_idx in range(baseline.shape[-1]):
            improved = baseline.clone()
            improved[:, component_idx] += 0.05
            gains.append(
                (bottleneck_balanced_progress(improved) - baseline_score).item()
            )
        self.assertTrue(all(gain > 0.0 for gain in gains))
        self.assertGreater(gains[0], gains[-1])

    def test_bottleneck_balanced_progress_requires_all_components_for_one(self):
        complete = torch.ones(2, 5)
        incomplete = complete.clone()
        incomplete[0, 2] = 0.5
        incomplete[1, 4] = 0.0
        self.assertTrue(
            torch.equal(bottleneck_balanced_progress(complete), torch.ones(2))
        )
        self.assertTrue(torch.all(bottleneck_balanced_progress(incomplete) < 1.0))

    def test_grasp_enclosure_progress_rewards_open_align_then_close(self):
        radius = torch.tensor([0.02])
        no_contact = torch.zeros(1)

        closed = grasp_enclosure_progress(
            torch.tensor([[0.0, 0.0]]), radius, torch.zeros(1), no_contact
        )[0]
        open_off_center = grasp_enclosure_progress(
            torch.tensor([[0.01, 0.05]]), radius, torch.zeros(1), no_contact
        )[0]
        open_aligned = grasp_enclosure_progress(
            torch.tensor([[-0.04, 0.04]]), radius, torch.ones(1), no_contact
        )[0]
        closed_around_handle = grasp_enclosure_progress(
            torch.tensor([[-0.02, 0.02]]), radius, torch.ones(1), torch.ones(1)
        )[0]

        self.assertLess(closed.item(), open_off_center.item())
        self.assertLess(open_off_center.item(), open_aligned.item())
        self.assertLess(open_aligned.item(), closed_around_handle.item())
        self.assertAlmostEqual(closed_around_handle.item(), 1.0, places=6)

    def test_grasp_navigation_retracts_before_crossing_handle(self):
        radius = torch.tensor([0.02])
        for vertical, local_jaws, expected_vertical_sign in (
            (0.08, (-0.04, 0.00), -1.0),
            (-0.08, (0.00, 0.04), 1.0),
        ):
            normal = torch.tensor([0.0], requires_grad=True)
            vertical_error = torch.tensor([vertical], requires_grad=True)
            pocket_error = torch.stack(
                [normal, torch.zeros_like(normal), vertical_error], dim=-1
            )
            local = torch.tensor([local_jaws])
            jaws = local + vertical_error[:, None]
            progress = grasp_collision_aware_position_progress(
                pocket_error, jaws, radius
            )[0]
            progress.backward()
            self.assertLess(normal.grad.item(), 0.0)
            self.assertEqual(
                torch.sign(vertical_error.grad).item(), expected_vertical_sign
            )
            self.assertGreater(
                abs(normal.grad.item()), 5.0 * abs(vertical_error.grad.item())
            )

    def test_grasp_navigation_aligns_at_safe_standoff(self):
        radius = torch.tensor([0.02])
        normal = torch.tensor([-0.11], requires_grad=True)
        vertical = torch.tensor([0.08], requires_grad=True)
        pocket_error = torch.stack(
            [normal, torch.zeros_like(normal), vertical], dim=-1
        )
        jaws = torch.tensor([[-0.04, 0.00]]) + vertical[:, None]
        progress = grasp_collision_aware_position_progress(
            pocket_error, jaws, radius
        )[0]
        progress.backward()
        self.assertLess(vertical.grad.item(), 0.0)
        self.assertGreater(normal.grad.item(), 0.0)

    def test_grasp_navigation_rewards_partial_enclosure_at_fixed_standoff(self):
        radius = torch.tensor([0.02])
        scores = []
        for between_quality in (0.1, 0.5, 0.9):
            jaws = torch.tensor(
                [[-between_quality * radius.item(), between_quality * radius.item()]]
            )
            progress = grasp_collision_aware_position_progress(
                torch.tensor([[-0.10, 0.0, 0.0]]), jaws, radius
            )[0]
            scores.append(progress.item())
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])

    def test_grasp_navigation_retains_route_credit_during_final_approach(self):
        radius = torch.tensor([0.02])
        scores = []
        for clearance in (0.10, 0.08, 0.04, 0.0):
            progress = grasp_collision_aware_position_progress(
                torch.tensor([[-clearance, 0.0, 0.0]]),
                torch.tensor([[-0.02, 0.02]]),
                radius,
            )[0]
            scores.append(progress.item())
        self.assertTrue(all(left < right for left, right in zip(scores, scores[1:])))

    def test_grasp_navigation_route_does_not_move_with_current_jaw_axis(self):
        pocket_error = torch.tensor([[-0.10, 0.0, 0.01]])
        radius = torch.tensor([0.02])
        centered = grasp_collision_aware_position_progress(
            pocket_error, torch.tensor([[-0.04, 0.04]]), radius
        )
        shifted_projection = grasp_collision_aware_position_progress(
            pocket_error, torch.tensor([[0.46, 0.54]]), radius
        )
        self.assertTrue(torch.allclose(centered[0], shifted_projection[0]))
        self.assertTrue(torch.allclose(centered[4], shifted_projection[4]))

    def test_grasp_navigation_approaches_only_after_enclosure(self):
        radius = torch.tensor([0.02])
        normal = torch.tensor([-0.08], requires_grad=True)
        pocket_error = torch.stack(
            [normal, torch.zeros_like(normal), torch.zeros_like(normal)], dim=-1
        )
        progress = grasp_collision_aware_position_progress(
            pocket_error, torch.tensor([[-0.03, 0.03]]), radius
        )[0]
        progress.backward()
        self.assertGreater(normal.grad.item(), 0.0)

    def test_grasp_navigation_is_complete_at_physical_goal(self):
        progress = grasp_collision_aware_position_progress(
            torch.zeros(1, 3),
            torch.tensor([[-0.02, 0.02]]),
            torch.tensor([0.02]),
        )[0]
        self.assertTrue(torch.allclose(progress, torch.ones_like(progress)))

    def test_grasp_enclosure_progress_has_opening_and_closing_gradients(self):
        radius = torch.tensor([0.02])
        half_aperture = torch.tensor([0.005], requires_grad=True)
        opening_score = grasp_enclosure_progress(
            torch.stack([-half_aperture, half_aperture], dim=-1),
            radius,
            torch.zeros(1),
            torch.zeros(1),
        )[0]
        opening_score.backward()
        self.assertGreater(half_aperture.grad.item(), 0.0)

        spread = torch.tensor([2.0], requires_grad=True)
        closing_score = grasp_enclosure_progress(
            torch.stack([-spread * radius, spread * radius], dim=-1),
            radius,
            torch.ones(1),
            torch.zeros(1),
        )[0]
        closing_score.backward()
        self.assertLess(spread.grad.item(), 0.0)

    def test_grasp_enclosure_progress_scales_with_randomized_handle_radius(self):
        radii = torch.tensor([0.01, 0.02, 0.03])
        jaw_coordinates = torch.stack([-radii, radii], dim=-1)
        progress = grasp_enclosure_progress(
            jaw_coordinates,
            radii,
            torch.ones(3),
            torch.ones(3),
        )[0]
        self.assertTrue(torch.allclose(progress, torch.ones_like(progress)))

    def test_grasp_enclosure_progress_rewards_each_contact_step(self):
        radius = torch.tensor([0.02])
        jaw_coordinates = torch.tensor([[-0.02, 0.02]])
        scores = []
        for physical_quality in (0.0, 0.25, 1.0):
            scores.append(
                grasp_enclosure_progress(
                    jaw_coordinates,
                    radius,
                    torch.ones(1),
                    torch.tensor([physical_quality]),
                )[0].item()
            )
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])

    def test_bilateral_contact_force_progress_rewards_balance(self):
        valid = torch.ones(3, 2, dtype=torch.bool)
        progress, per_jaw = bilateral_contact_force_progress(
            torch.tensor([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]),
            valid,
        )
        self.assertTrue(torch.allclose(progress, torch.tensor([0.0, 0.25, 1.0])))
        self.assertTrue(torch.allclose(per_jaw[-1], torch.ones(2)))

    def test_bilateral_contact_force_progress_penalizes_overload(self):
        valid = torch.ones(2, 2, dtype=torch.bool)
        progress, _ = bilateral_contact_force_progress(
            torch.tensor([[30.0, 30.0], [50.0, 50.0]]),
            valid,
        )
        self.assertGreater(progress[0].item(), progress[1].item())

    def test_open_latch_requires_base_latch_gap_and_bilateral_grasp(self):
        result = open_latch_completion_condition(
            operation_distance=torch.tensor([0.15, 0.17, 0.15, 0.15, 0.15]),
            base_yaw_error=torch.tensor([0.07, 0.07, 0.09, 0.07, 0.07]),
            latch_position=torch.tensor([0.02, 0.02, 0.02, 0.019, 0.02]),
            hinge_angle=torch.tensor([0.20, 0.20, 0.20, 0.20, 0.05]),
            gap_angle=torch.tensor([0.10] * 5),
            precise_bilateral_grasp=torch.tensor([True, True, True, True, True]),
        )

        self.assertTrue(
            torch.equal(result, torch.tensor([True, False, False, False, False]))
        )

    def test_opposed_contact_allows_different_axial_positions(self):
        result = opposed_handle_contact(
            recent=torch.tensor([[True, True]]),
            stored_points_finite=torch.tensor([True]),
            stored_axial=torch.tensor([[0.045, -0.030]]),
            usable_half_length=torch.tensor([0.060]),
            stored_closing=torch.tensor([[0.010, -0.012]]),
            handle_radius=torch.tensor([0.020]),
        )
        self.assertTrue(result.item())

    def test_opposed_contact_rejects_invalid_geometry(self):
        common = {
            "recent": torch.tensor([[True, True]]),
            "stored_points_finite": torch.tensor([True]),
            "stored_axial": torch.tensor([[0.020, -0.020]]),
            "usable_half_length": torch.tensor([0.060]),
            "stored_closing": torch.tensor([[0.010, -0.012]]),
            "handle_radius": torch.tensor([0.020]),
        }
        cases = (
            {"recent": torch.tensor([[True, False]])},
            {"stored_points_finite": torch.tensor([False])},
            {"stored_axial": torch.tensor([[0.061, -0.020]])},
            {"stored_closing": torch.tensor([[0.010, 0.012]])},
            {"stored_closing": torch.tensor([[0.004, -0.012]])},
        )
        for override in cases:
            with self.subTest(override=override):
                self.assertFalse(
                    opposed_handle_contact(**(common | override)).item()
                )


if __name__ == "__main__":
    unittest.main()
