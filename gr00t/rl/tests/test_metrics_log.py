import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from gr00t.rl.trl.utils.metrics_log import (
    append_training_metrics_jsonl,
    take_remaining_episode_ids,
)


class TrainingMetricsLogTest(unittest.TestCase):
    def test_episode_ids_are_truncated_to_exact_remaining_count(self):
        completed = [7, 2, 9, 4]
        self.assertEqual(
            take_remaining_episode_ids(completed, collected_count=8, target_count=10),
            [7, 2],
        )
        self.assertEqual(
            take_remaining_episode_ids(completed, collected_count=10, target_count=10),
            [],
        )

    def test_rank_zero_appends_complete_json_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = append_training_metrics_jsonl(
                directory,
                {
                    "step": 1,
                    "Policy/temporal_fraction/0/mean": torch.tensor(0.75),
                    "Env/stable_reached_10": 0.25,
                },
                is_world_process_zero=True,
            )
            append_training_metrics_jsonl(
                directory,
                {"step": 2, "train/stage_count/grasp": 3},
                is_world_process_zero=True,
            )

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([record["step"] for record in records], [1, 2])
            self.assertAlmostEqual(
                records[0]["Policy/temporal_fraction/0/mean"], 0.75
            )
            self.assertEqual(records[1]["train/stage_count/grasp"], 3)
            self.assertEqual(records[0]["_non_finite_metrics"], [])

    def test_non_main_rank_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as directory:
            result = append_training_metrics_jsonl(
                directory,
                {"step": 1},
                is_world_process_zero=False,
            )
            self.assertIsNone(result)
            self.assertFalse(Path(directory, "training_metrics.jsonl").exists())

    def test_non_finite_values_are_explicit_standard_json_nulls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = append_training_metrics_jsonl(
                directory,
                {"finite": 1.0, "nan": math.nan, "positive_inf": math.inf},
                is_world_process_zero=True,
            )

            record = json.loads(path.read_text())
            self.assertEqual(record["finite"], 1.0)
            self.assertIsNone(record["nan"])
            self.assertIsNone(record["positive_inf"])
            self.assertEqual(
                record["_non_finite_metrics"], ["nan", "positive_inf"]
            )

    def test_non_scalar_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TypeError, "must be a scalar"):
                append_training_metrics_jsonl(
                    directory,
                    {"not_scalar": torch.ones(2)},
                    is_world_process_zero=True,
                )
