import unittest

import torch

from gr00t.rl.envs.base_task.staged_task_events import (
    advance_stage_for_next_observation,
    compute_stage_advance_event,
    fixed_stage_for_mode,
    mask_stage_advance_event,
    stage_completion_event,
)


class StagedTaskEventTest(unittest.TestCase):
    def test_fixed_stage_selection_keeps_training_and_evaluation_independent(self):
        self.assertEqual(fixed_stage_for_mode(1, 5, False), 1)
        self.assertEqual(fixed_stage_for_mode(1, 5, True), 5)
        self.assertIsNone(fixed_stage_for_mode(1, None, True))

    def test_event_uses_only_the_active_stage_condition(self):
        event = compute_stage_advance_event(
            torch.tensor([0, 1, 1, 2]),
            [
                torch.tensor([True, True, True, True]),
                torch.tensor([False, True, True, True]),
            ],
        )
        self.assertTrue(torch.equal(event, torch.tensor([True, True, True, False])))

    def test_reset_masks_event_for_reward_and_transition(self):
        event = torch.tensor([True, True, True, False])
        self.assertTrue(
            torch.equal(
                mask_stage_advance_event(
                    event,
                    torch.tensor([False, False, True, False]),
                    torch.tensor([False, False, False, False]),
                ),
                torch.tensor([True, True, False, False]),
            )
        )

    def test_completion_reward_belongs_to_completing_action(self):
        event = stage_completion_event(
            torch.tensor([True, False]), torch.tensor([False, False])
        )
        self.assertTrue(torch.equal(event.float() / 0.02, torch.tensor([50.0, 0.0])))

        # Entering the next stage with a zero stage timer must not award the
        # previous action's completion event a second time.
        next_step_event = stage_completion_event(
            torch.tensor([False, False]), torch.tensor([False, False])
        )
        self.assertTrue(torch.equal(next_step_event, torch.tensor([False, False])))

    def test_transition_is_committed_before_next_policy_observation(self):
        stage = torch.tensor([0, 1, 1, 2])
        event = torch.tensor([True, True, True, False])
        reset = torch.tensor([False, False, True, False])
        just_reset = torch.tensor([False, False, False, False])

        # The completing action is still credited against the stage that
        # produced it.
        old_stage = stage.clone()
        old_stage_credit = event & (old_stage == torch.tensor([0, 1, 1, 2]))

        advance, next_stage = advance_stage_for_next_observation(
            stage, event, reset, just_reset
        )
        next_stage_observation = torch.nn.functional.one_hot(next_stage, 4)

        self.assertTrue(torch.equal(old_stage_credit, event))
        self.assertTrue(torch.equal(advance, torch.tensor([True, True, False, False])))
        self.assertTrue(torch.equal(next_stage, torch.tensor([1, 2, 1, 2])))
        self.assertEqual(next_stage_observation[0].argmax().item(), 1)
        self.assertEqual(next_stage_observation[1].argmax().item(), 2)

    def test_just_reset_environment_does_not_advance(self):
        advance, next_stage = advance_stage_for_next_observation(
            torch.tensor([0, 2]),
            torch.tensor([True, True]),
            torch.tensor([False, False]),
            torch.tensor([True, False]),
        )
        self.assertTrue(torch.equal(advance, torch.tensor([False, True])))
        self.assertTrue(torch.equal(next_stage, torch.tensor([0, 3])))


if __name__ == "__main__":
    unittest.main()
