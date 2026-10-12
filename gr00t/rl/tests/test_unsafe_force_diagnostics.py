import tempfile
import unittest
from pathlib import Path

import torch

from gr00t.rl.envs.door.unsafe_force_diagnostics import UnsafeForceEventRecorder


class UnsafeForceEventRecorderTest(unittest.TestCase):
    def test_records_previous_and_trigger_frames_without_aliasing(self):
        recorder = UnsafeForceEventRecorder(2, "cpu")
        first = {"force": torch.tensor([[10.0], [20.0]])}
        recorder.record(
            torch.tensor([False, False]), torch.zeros(2, dtype=torch.long), first
        )
        second = {"force": torch.tensor([[700.0], [30.0]])}
        recorder.record(
            torch.tensor([True, False]), torch.tensor([1, 0]), second
        )
        second["force"][0] = -1.0

        event = recorder.events[0]
        self.assertTrue(event["previous_valid"])
        self.assertEqual(event["trigger_object"], "panel")
        self.assertEqual(event["previous"]["force"].item(), 10.0)
        self.assertEqual(event["trigger"]["force"].item(), 700.0)

    def test_invalidate_marks_first_post_reset_event_without_previous_frame(self):
        recorder = UnsafeForceEventRecorder(1, "cpu")
        feature = {"force": torch.tensor([[10.0]])}
        recorder.record(
            torch.tensor([False]), torch.zeros(1, dtype=torch.long), feature
        )
        recorder.invalidate(torch.tensor([0]))
        recorder.record(
            torch.tensor([True]), torch.zeros(1, dtype=torch.long), feature
        )
        self.assertFalse(recorder.events[0]["previous_valid"])

    def test_save_marks_optimizer_as_not_executed(self):
        recorder = UnsafeForceEventRecorder(1, "cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.pt"
            recorder.save(path, {"optimizer_executed": False})
            saved = torch.load(path, weights_only=False)
        self.assertFalse(saved["metadata"]["optimizer_executed"])
        self.assertEqual(saved["trigger_object_names"], ("handle", "panel", "frame", "other"))

if __name__ == "__main__":
    unittest.main()
