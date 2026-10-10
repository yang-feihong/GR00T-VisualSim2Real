"""Probe the standard-data contact audit against a real AutoOpen scene."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
LOWLEVEL_ROOT = SCRIPT_ROOT / "third_party" / "visual_wholebody_lowlevel"
sys.path.insert(0, str(LOWLEVEL_ROOT))

from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy  # noqa: E402
from legged_gym.utils.isaaclab_app import launch_app  # noqa: E402
from gr00t.rl.data_collection.contact_audit import SubstepContactAudit  # noqa: E402
from gr00t.rl.scripts.doorman_auto_open import _default_paths, _parser  # noqa: E402


def main():
    args = _parser().parse_args()
    args.num_envs = 1
    args.batched_envs = 0
    args.disable_rgb_cameras = True
    _default_paths(args)
    app = launch_app(args)
    runner = None
    audit = None
    try:
        runner = ManipLoco_Policy(args)
        audit = SubstepContactAudit.create(runner.env)
        print("[contact-audit][view] " + json.dumps(audit.description), flush=True)
        audit.attach()
        audit.begin_tick()
        result = runner.step_once()
        rows = audit.end_tick()
        print(
            "[contact-audit][tick] "
            + json.dumps(
                {
                    "pairs": len(rows),
                    "rows": [row.__dict__ for row in rows],
                    "state_finite": bool(
                        torch.isfinite(result["state"]["root_states"]).all().item()
                    ),
                }
            ),
            flush=True,
        )
    finally:
        if audit is not None:
            audit.detach()
        if runner is not None:
            runner.close()
        app.close()


if __name__ == "__main__":
    main()
