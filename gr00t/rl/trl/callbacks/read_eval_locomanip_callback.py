# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
from pathlib import Path

import wandb
from loguru import logger
from transformers import TrainerCallback

from gr00t.rl.trl.utils.common import wandb_run_exists


class ReadEvalLocomanipCallback(TrainerCallback):
    """Callback to read evaluation metrics and log them to wandb."""

    def __init__(self, eval_dir: str, check_interval: int = 1):
        """
        Initialize the callback.

        Args:
            experiment_dir: Path to the experiment directory where eval results are stored
            check_interval: How often to check for new evaluation results (in log steps)
        """
        super().__init__()
        self.eval_dir = Path(eval_dir)
        self.check_interval = check_interval
        self.last_check_step = 0
        self.state = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Check for new evaluation results and log them to wandb."""
        logger.info("checking for new evaluation results")
        # Only check every check_interval steps to avoid excessive file I/O
        self.accelerator = kwargs.get("accelerator")
        self.env = kwargs.get("env")
        if state.global_step - self.last_check_step < self.check_interval:
            return

        next_eval_step, next_eval_step_dir = self.find_next_unread_checkpoint(
            state.eval_step, mode="metrics"
        )
        if next_eval_step is not None:
            self._check_and_log_eval_results(next_eval_step, next_eval_step_dir)
            state.eval_step = next_eval_step

        next_eval_render_step, next_eval_render_step_dir = self.find_next_unread_checkpoint(
            state.eval_render_step, mode="render"
        )
        if next_eval_render_step is not None:
            self._check_and_log_eval_render_results(
                next_eval_render_step, next_eval_render_step_dir
            )
            state.eval_render_step = next_eval_render_step

        self.last_check_step = state.global_step
        logger.info(f"last checked step: {self.last_check_step}")

    def find_next_unread_checkpoint(self, current_eval_step: int = 0, mode: str = "metrics"):
        """
        Find the next unread checkpoint folder that has finished evaluation.

        Args:
            current_eval_step: Current evaluation step to start searching from

        Returns:
            tuple: (eval_step, eval_step_dir) or (None, None) if not found
        """
        if not self.eval_dir.exists():
            return None, None

        # Find all evaluation directories with 6-digit zero-padded names
        eval_step_dirs = []
        for d in self.eval_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                eval_step = int(d.name)
                if eval_step > current_eval_step:
                    eval_step_dirs.append((eval_step, d))

        if not eval_step_dirs:
            return None, None

        # Sort by step number to get the next one in sequence
        eval_step_dirs.sort(key=lambda x: x[0])

        # Find the first one that has finished evaluation
        for eval_step, eval_step_dir in eval_step_dirs:
            metrics_finish_file = eval_step_dir / f"{mode}_finish.txt"
            if metrics_finish_file.exists():
                return eval_step, eval_step_dir

        return None, None

    def _check_and_log_eval_results(self, eval_step, eval_step_dir):
        """Check for new evaluation results and log them to wandb."""

        metrics_file = eval_step_dir / "metrics_eval.json"
        # Read and log the metrics
        with open(metrics_file, "r") as f:
            metrics_eval = json.load(f)

        # Add eval_step if not already present
        if "eval_step" not in metrics_eval:
            metrics_eval["eval_step"] = eval_step

        # Log to wandb
        if self.accelerator.is_main_process and wandb_run_exists():
            wandb.log(metrics_eval)

        print(f"Logged evaluation metrics for step {eval_step}")

    def _check_and_log_eval_render_results(self, eval_step, eval_step_dir):
        """Check for new evaluation results and log them to wandb."""

        video_dir = eval_step_dir / "render_results"
        video_files = []
        for i, video_file in enumerate(sorted(video_dir.iterdir())):
            if video_file.is_file() and video_file.name.endswith(".mp4"):
                video_files.append((i, video_file))

        wandb_videos = {
            f"videos_hard/{i: 04d}": wandb.Video(str(video_file), format="mp4")
            for i, video_file in reversed(video_files)
        }
        wandb_videos["eval_step"] = eval_step

        # Log to wandb
        if self.accelerator.is_main_process and wandb_run_exists():
            wandb.log(wandb_videos)

        print(f"Logged rendered video for step {eval_step}")
