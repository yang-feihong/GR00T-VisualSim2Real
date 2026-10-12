import unittest

import torch

from gr00t.rl.envs.door.eval_diagnostics import (
    closest_event_mask,
    max_acceptance_event_mask,
    update_event_snapshot,
)


class GraspEvalDiagnosticsTest(unittest.TestCase):
    def test_event_selection_uses_strict_distance_and_count_then_distance(self):
        self.assertTrue(
            torch.equal(
                closest_event_mask(
                    torch.tensor([0.02, 0.03]), torch.tensor([0.03, 0.03])
                ),
                torch.tensor([True, False]),
            )
        )
        selected = max_acceptance_event_mask(
            torch.tensor([9.0, 8.0, 9.0, 9.0]),
            torch.tensor([0.08, 0.01, 0.04, 0.06]),
            torch.tensor([8.0, 9.0, 9.0, 9.0]),
            torch.tensor([0.02, 0.02, 0.05, 0.05]),
        )
        self.assertTrue(
            torch.equal(selected, torch.tensor([True, False, True, False]))
        )

    def test_snapshot_copies_sources_and_records_the_following_frame(self):
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        error = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        source_before = source.clone()
        error_before = error.clone()
        snapshot = {
            "pending_next": torch.zeros(2, dtype=torch.bool),
            "error_next": torch.full((2, 3), float("nan")),
            "value": torch.full((2, 2), float("nan")),
            "error_current": torch.full((2, 3), float("nan")),
        }

        update_event_snapshot(
            snapshot,
            torch.tensor([True, False]),
            {"value": source, "error_current": error},
            error,
        )
        source.add_(10.0)
        next_error = error + 1.0
        update_event_snapshot(
            snapshot,
            torch.zeros(2, dtype=torch.bool),
            {"value": source, "error_current": next_error},
            next_error,
        )

        self.assertTrue(torch.equal(source_before, torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(torch.equal(error, error_before))
        self.assertTrue(torch.equal(snapshot["value"][0], source_before[0]))
        self.assertTrue(torch.equal(snapshot["error_current"][0], error_before[0]))
        self.assertTrue(torch.equal(snapshot["error_next"][0], next_error[0]))
        self.assertTrue(torch.isnan(snapshot["value"][1]).all())

    def test_replacement_does_not_mix_old_event_next_frame(self):
        snapshot = {
            "pending_next": torch.tensor([True]),
            "error_next": torch.full((1, 3), float("nan")),
            "value": torch.tensor([[1.0]]),
            "error_current": torch.tensor([[1.0, 1.0, 1.0]]),
        }
        current = torch.tensor([[2.0, 2.0, 2.0]])
        update_event_snapshot(
            snapshot,
            torch.tensor([True]),
            {"value": torch.tensor([[2.0]]), "error_current": current},
            current,
        )

        self.assertEqual(snapshot["value"].item(), 2.0)
        self.assertTrue(torch.isnan(snapshot["error_next"]).all())
        self.assertTrue(snapshot["pending_next"].item())


if __name__ == "__main__":
    unittest.main()
