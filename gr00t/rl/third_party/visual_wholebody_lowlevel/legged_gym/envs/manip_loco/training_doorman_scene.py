from __future__ import annotations

import copy

from legged_gym.utils.episode_rng import episode_choice, stream_seed


def episode_door_side(episode_seed: int) -> str:
    return episode_choice(episode_seed, "door_side", ("left", "right"))


def build_training_doorman_door_cfg(
    num_envs: int,
    door_open_lr: str | None = None,
    physics_only: bool = False,
):
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
    if physics_only:
        door_cfg.spawn.assets_cfg = [
            asset_cfg.replace(
                randomize_material=False,
                use_preloaded_materials=False,
                add_lights=False,
            )
            for asset_cfg in door_cfg.spawn.assets_cfg
        ]
    return door_cfg


def build_grouped_training_doorman_door_cfgs(
    num_envs: int,
    seed: int,
    physics_only: bool = False,
    door_open_lr: str | None = None,
    episode_seeds=None,
):
    """Build separate homogeneous articulation views for mirrored doors."""
    if num_envs < 1:
        raise ValueError("Grouped door configuration requires an environment")
    if door_open_lr is not None and door_open_lr not in ("left", "right"):
        raise ValueError(f"Unsupported door_open_lr={door_open_lr!r}")
    episode_seeds = (
        [int(seed) + env_id for env_id in range(num_envs)]
        if episode_seeds is None
        else [int(value) for value in episode_seeds]
    )
    if len(episode_seeds) != int(num_envs):
        raise ValueError("episode_seeds must contain exactly num_envs values")
    sides = [
        door_open_lr
        if door_open_lr is not None
        else episode_door_side(episode_seed)
        for episode_seed in episode_seeds
    ]
    result = []
    for side in ("left", "right"):
        env_ids = [index for index, value in enumerate(sides) if value == side]
        if not env_ids:
            continue
        env_pattern = "|".join(str(index) for index in env_ids)
        door_cfg = build_training_doorman_door_cfg(
            len(env_ids), side, physics_only=physics_only
        )
        door_cfg.spawn.assets_cfg = [
            asset_cfg.replace(
                articulation_root_name=f"root_{side}",
                random_seed=stream_seed(
                    episode_seeds[env_id], "door_physics"
                ),
                visual_random_seed=stream_seed(
                    episode_seeds[env_id], "door_visual"
                ),
            )
            for asset_cfg, env_id in zip(
                door_cfg.spawn.assets_cfg, env_ids, strict=True
            )
        ]
        door_cfg = door_cfg.replace(
            prim_path=f"/World/envs/env_({env_pattern})/door_{side}"
        )
        result.append((door_cfg, env_ids))
    return result
