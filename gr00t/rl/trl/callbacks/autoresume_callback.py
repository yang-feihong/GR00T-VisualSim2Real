# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import sys
from pathlib import Path

import wandb
from transformers import TrainerCallback

from gr00t.rl.trl.callbacks.model_save_callback import ModelSaveCallback
from gr00t.rl.trl.utils.common import wandb_run_exists

try:
    sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
    from userlib.auto_resume import AutoResume
except ModuleNotFoundError:
    AutoResume = None


class AutoResumeCallback(TrainerCallback):
    """Callback to save model state_dict during training."""

    def __init__(self, save_dir, every_n_steps=10):
        """
        Args:
            every_n_steps (int): Check for resume every N steps
        """
        self.save_dir = Path(save_dir)
        self.every_n_steps = every_n_steps
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        """Save model state_dict at the end of each step if frequency matches."""
        if state.global_step % self.every_n_steps == 0:
            model = kwargs.get("model")
            optimizer = kwargs.get("optimizer")
            lr_scheduler = kwargs.get("lr_scheduler")
            env = kwargs.get("env")
            env_state_dict = env.get_env_state_dict()
            if AutoResume is not None and AutoResume.termination_requested():
                if state.is_world_process_zero:
                    resume_file = str(self.save_dir / "last.pt")
                    ModelSaveCallback.save_checkpoint(
                        model, optimizer, lr_scheduler, state, env_state_dict, args, resume_file
                    )
                    message = f"[Auto Resume] Terminateing. Resume_file: {resume_file}"
                    print(message, flush=True)
                    AutoResume.request_resume(
                        user_dict={
                            "resume_file": resume_file,
                            "wandb_id": wandb.run.id if wandb_run_exists() else None,
                            "save_dir": str(self.save_dir),
                        },
                        message=message,
                    )
                    print(
                        f"[rank {args.global_rank}] [Auto Resume] Requesting resume. Resume_file: {resume_file}",
                        flush=True,
                    )
                control.should_training_stop = True
                print(f"[rank {args.global_rank}] [Auto Resume] Stopping training", flush=True)

        return
