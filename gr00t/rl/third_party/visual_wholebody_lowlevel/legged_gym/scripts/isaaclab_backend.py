from __future__ import annotations

import sys
from pathlib import Path

LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
for p in [str(LOW_LEVEL_ROOT)]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

def make_env(args, for_play=False, env_cfg=None):
    from legged_gym.envs.manip_loco.manip_loco import ManipLocoIsaacLab
    from legged_gym.utils.helpers import build_env_cfg

    env = ManipLocoIsaacLab(env_cfg if env_cfg is not None else build_env_cfg(args, for_play=for_play))
    env.reset()
    return env
