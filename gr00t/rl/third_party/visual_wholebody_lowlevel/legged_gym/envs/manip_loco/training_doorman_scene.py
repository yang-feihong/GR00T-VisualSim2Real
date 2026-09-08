from __future__ import annotations

import copy


def build_training_doorman_door_cfg(num_envs: int, door_open_lr: str | None = None):
    """Reuse the exact door asset configuration used by high-level training."""
    from gr00t.rl.data.tasks.door.scenario_cfg.isaacsim import TaskObjCfgDict

    door_cfg = copy.deepcopy(TaskObjCfgDict["door"])
    door_cfg = door_cfg.replace(prim_path="/World/envs/env_.*/door")
    door_cfg.spawn.assets_cfg = door_cfg.spawn.assets_cfg[: int(num_envs)]
    if door_open_lr is not None:
        if door_open_lr not in ("left", "right"):
            raise ValueError(f"Unsupported door_open_lr={door_open_lr!r}")
        door_cfg.spawn.assets_cfg = [
            asset_cfg.replace(door_open_lr=[door_open_lr])
            for asset_cfg in door_cfg.spawn.assets_cfg
        ]
    return door_cfg
