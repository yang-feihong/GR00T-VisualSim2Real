from __future__ import annotations

import copy


def build_training_doorman_door_cfg(num_envs: int):
    """Reuse the exact door asset configuration used by high-level training."""
    from gr00t.rl.data.tasks.door.scenario_cfg.isaacsim import TaskObjCfgDict

    door_cfg = copy.deepcopy(TaskObjCfgDict["door"])
    door_cfg = door_cfg.replace(prim_path="/World/envs/env_.*/door")
    door_cfg.spawn.assets_cfg = door_cfg.spawn.assets_cfg[: int(num_envs)]
    return door_cfg
