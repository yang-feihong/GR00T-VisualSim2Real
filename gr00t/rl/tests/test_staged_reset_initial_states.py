import unittest

import torch

from gr00t.rl.envs.base_task.staged_reset_library import (
    pack_stage_initial_state_library,
    restore_stage_initial_state_library,
)


def make_staged_task_stub():
    num_samples = torch.tensor(
        [[4, 4], [3, 1], [0, 2]], dtype=torch.long
    )
    data = torch.zeros(3, 4, 2, 1)
    for stage in range(3):
        for sample in range(4):
            for env in range(2):
                data[stage, sample, env, 0] = 100 * stage + 10 * sample + env
    state_cases = {"value": {"type": "buffer", "data": data}}
    return state_cases, num_samples


class StagedResetInitialStateTest(unittest.TestCase):
    def test_checkpoint_preserves_multiple_recent_states_per_environment(self):
        state_cases, num_samples = make_staged_task_stub()
        checkpoint = pack_stage_initial_state_library(
            state_cases, num_samples, 4, 2
        )

        self.assertEqual(
            checkpoint["format"], "per_environment_stage_initial_states_v2"
        )
        self.assertTrue(
            torch.equal(
                checkpoint["sample_counts"],
                torch.tensor([[0, 0], [2, 1], [0, 2]]),
            )
        )
        packed = checkpoint["cases"]["value"]["data"]
        self.assertEqual(tuple(packed.shape), (3, 2, 2, 1))
        self.assertEqual(packed[1, 0, 0, 0].item(), 120.0)
        self.assertEqual(packed[1, 1, 0, 0].item(), 110.0)
        self.assertEqual(packed[1, 0, 1, 0].item(), 101.0)

    def test_checkpoint_reload_restores_random_sampling_pool(self):
        source_cases, source_counts = make_staged_task_stub()
        checkpoint = pack_stage_initial_state_library(
            source_cases, source_counts, 4, 2
        )
        target_cases, _ = make_staged_task_stub()
        target_cases["value"]["data"].zero_()

        restored_counts = restore_stage_initial_state_library(
            checkpoint, target_cases, 3, 2, 4, torch.device("cpu")
        )
        restored_counts[0] = 4

        self.assertTrue(
            torch.equal(
                restored_counts,
                torch.tensor([[4, 4], [2, 1], [0, 2]]),
            )
        )
        restored = target_cases["value"]["data"]
        self.assertEqual(restored[1, 0, 0, 0].item(), 120.0)
        self.assertEqual(restored[1, 1, 0, 0].item(), 110.0)

    def test_restore_rejects_a_different_environment_topology(self):
        state_cases, num_samples = make_staged_task_stub()
        checkpoint = pack_stage_initial_state_library(
            state_cases, num_samples, 4, 2
        )
        with self.assertRaisesRegex(ValueError, "environment count"):
            restore_stage_initial_state_library(
                checkpoint, state_cases, 3, 3, 4, torch.device("cpu")
            )


if __name__ == "__main__":
    unittest.main()
