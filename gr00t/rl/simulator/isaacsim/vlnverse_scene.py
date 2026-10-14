"""VLNVerse static-scene discovery, sampling, and Isaac Sim installation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VlnverseSceneRecord:
    scene_id: str
    scene_dir: Path
    usd_path: Path
    occupancy_path: Path | None
    occupancy_metadata_path: Path | None
    rooms_path: Path | None

    @property
    def has_navigation_metadata(self) -> bool:
        return (
            self.occupancy_path is not None
            and self.occupancy_metadata_path is not None
            and self.rooms_path is not None
        )


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def discover_vlnverse_scenes(
    root_dir: str | Path,
    *,
    require_navigation_metadata: bool = True,
    scene_ids: Sequence[str] | None = None,
) -> list[VlnverseSceneRecord]:
    """Return sorted, loadable scene records under a VLNVerse asset root."""
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"VLNVerse scene root does not exist: {root}")

    allowed = set(scene_ids or ())
    records: list[VlnverseSceneRecord] = []
    for scene_dir in sorted(root.glob("kujiale_*")):
        if not scene_dir.is_dir() or (allowed and scene_dir.name not in allowed):
            continue
        usd_path = scene_dir / "start_result_navigation.usd"
        if not usd_path.is_file():
            continue
        occupancy_path = scene_dir / "occupancy.png"
        occupancy_metadata_path = scene_dir / "occupancy.json"
        rooms_path = scene_dir / "rooms.json"
        record = VlnverseSceneRecord(
            scene_id=scene_dir.name,
            scene_dir=scene_dir,
            usd_path=usd_path,
            occupancy_path=occupancy_path if occupancy_path.is_file() else None,
            occupancy_metadata_path=(
                occupancy_metadata_path if occupancy_metadata_path.is_file() else None
            ),
            rooms_path=rooms_path if rooms_path.is_file() else None,
        )
        if require_navigation_metadata and not record.has_navigation_metadata:
            continue
        records.append(record)

    if not records:
        qualifier = " with navigation metadata" if require_navigation_metadata else ""
        raise RuntimeError(f"No VLNVerse scenes{qualifier} found under {root}")
    missing = allowed.difference(record.scene_id for record in records)
    if missing:
        raise ValueError(f"Requested VLNVerse scenes are unavailable: {sorted(missing)}")
    return records


def select_vlnverse_scenes(
    records: Sequence[VlnverseSceneRecord],
    num_envs: int,
    *,
    seed: int,
    mode: str = "random",
    fixed_scene_id: str | None = None,
) -> list[VlnverseSceneRecord]:
    """Select one deterministic scene per environment."""
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    records = list(records)
    by_id = {record.scene_id: record for record in records}
    if mode == "fixed":
        if fixed_scene_id not in by_id:
            raise ValueError(f"Fixed VLNVerse scene is unavailable: {fixed_scene_id}")
        return [by_id[fixed_scene_id]] * num_envs
    if mode == "cycle":
        offset = int(seed) % len(records)
        return [records[(offset + env_id) % len(records)] for env_id in range(num_envs)]
    if mode != "random":
        raise ValueError(f"Unknown VLNVerse scene selection mode: {mode}")

    # Shuffle complete cycles instead of sampling with replacement. This gives
    # broad scene coverage while remaining independent of worker batch size.
    selected: list[VlnverseSceneRecord] = []
    cycle = 0
    while len(selected) < num_envs:
        generator = np.random.default_rng(np.random.SeedSequence([int(seed), cycle, 0x564C4E]))
        selected.extend(records[index] for index in generator.permutation(len(records)))
        cycle += 1
    return selected[:num_envs]


def _clearance_mask(free_mask: np.ndarray, radius_pixels: int) -> np.ndarray:
    if radius_pixels <= 0:
        return free_mask.copy()
    blocked = (~free_mask).astype(np.int32)
    padded = np.pad(blocked, radius_pixels, constant_values=1)
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    width = 2 * radius_pixels + 1
    window_sum = (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )
    return free_mask & (window_sum == 0)


def sample_scene_spawn_candidates(
    record: VlnverseSceneRecord,
    *,
    count: int,
    seed: int,
    clearance_radius_m: float,
    free_pixel_value: int = 255,
) -> np.ndarray:
    """Sample local ``(x, y, yaw)`` poses from a scene occupancy map."""
    if not record.has_navigation_metadata:
        raise ValueError(f"Scene has no navigation metadata: {record.scene_id}")
    if count < 1:
        raise ValueError("candidate count must be positive")

    with record.occupancy_metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    scale = float(metadata["scale"])
    lower = np.asarray(metadata["lower"], dtype=np.float64)
    upper = np.asarray(metadata["upper"], dtype=np.float64)
    image = np.asarray(Image.open(record.occupancy_path).convert("L"))
    free_mask = image == int(free_pixel_value)
    radius_pixels = int(math.ceil(float(clearance_radius_m) / scale))
    safe_mask = _clearance_mask(free_mask, radius_pixels)
    rows, columns = np.nonzero(safe_mask)
    if rows.size == 0:
        raise RuntimeError(
            f"Scene {record.scene_id} has no free spawn pixels with "
            f"{clearance_radius_m:.3f} m clearance"
        )

    generator = np.random.default_rng(np.random.SeedSequence([int(seed), 0x535041574E]))
    indices = generator.choice(rows.size, size=count, replace=rows.size < count)
    selected_rows = rows[indices].astype(np.float64)
    selected_columns = columns[indices].astype(np.float64)
    x = upper[0] - (selected_columns + 0.5) * scale
    y = lower[1] + (selected_rows + 0.5) * scale
    yaw = generator.uniform(-math.pi, math.pi, size=count)
    return np.stack((x, y, yaw), axis=-1).astype(np.float32)


def install_vlnverse_scenes(simulator: Any, config: Any, sim_utils: Any) -> None:
    """Install deterministic per-environment scene references and spawn banks."""
    root_dir = _cfg_get(config, "root_dir")
    selection = _cfg_get(config, "selection", {})
    spawn = _cfg_get(config, "robot_spawn", {})
    seed = int(_cfg_get(selection, "seed", 0))
    scene_ids = list(_cfg_get(selection, "scene_ids", []) or [])
    records = discover_vlnverse_scenes(
        root_dir,
        require_navigation_metadata=bool(
            _cfg_get(selection, "require_navigation_metadata", True)
        ),
        scene_ids=scene_ids,
    )
    selected = select_vlnverse_scenes(
        records,
        simulator.num_envs,
        seed=seed,
        mode=str(_cfg_get(selection, "mode", "random")),
        fixed_scene_id=_cfg_get(selection, "fixed_scene_id"),
    )

    usd_cfg = sim_utils.MultiUsdFileCfg(
        usd_path=[str(record.usd_path) for record in selected],
        random_choice=False,
    )
    prim_name = str(_cfg_get(config, "prim_name", "VlnverseScene"))
    usd_cfg.func(f"/World/envs/env_.*/{prim_name}", usd_cfg)
    disabled_physics_scenes = 0
    if bool(_cfg_get(config, "disable_embedded_physics_scenes", True)):
        from pxr import Usd, UsdPhysics

        stage = sim_utils.get_current_stage()
        physics_scene_prims = []
        for env_id in range(simulator.num_envs):
            scene_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/{prim_name}")
            physics_scene_prims.extend(
                prim for prim in Usd.PrimRange(scene_prim) if prim.IsA(UsdPhysics.Scene)
            )
        for prim in physics_scene_prims:
            prim.SetActive(False)
        disabled_physics_scenes = len(physics_scene_prims)

    if bool(_cfg_get(spawn, "enabled", True)):
        candidate_count = int(_cfg_get(spawn, "candidate_count", 256))
        candidate_banks = []
        for env_id, record in enumerate(selected):
            candidate_banks.append(
                sample_scene_spawn_candidates(
                    record,
                    count=candidate_count,
                    seed=int(np.random.SeedSequence([seed, env_id]).generate_state(1)[0]),
                    clearance_radius_m=float(_cfg_get(spawn, "clearance_radius_m", 0.55)),
                    free_pixel_value=int(_cfg_get(spawn, "free_pixel_value", 255)),
                )
            )
        simulator.static_scene_spawn_candidates = np.stack(candidate_banks)
    simulator.static_scene_ids = [record.scene_id for record in selected]
    simulator.static_scene_usd_paths = [str(record.usd_path) for record in selected]
    simulator.static_scene_seed = seed
    simulator.static_scene_metadata = {
        "provider": "vlnverse",
        "seed": seed,
        "selection_mode": str(_cfg_get(selection, "mode", "random")),
        "scene_ids": list(simulator.static_scene_ids),
        "usd_paths": list(simulator.static_scene_usd_paths),
        "disabled_embedded_physics_scenes": disabled_physics_scenes,
        "robot_spawn_enabled": bool(_cfg_get(spawn, "enabled", True)),
        "robot_spawn_clearance_radius_m": float(
            _cfg_get(spawn, "clearance_radius_m", 0.55)
        ),
    }


def static_scene_enabled(config: Any) -> bool:
    return bool(_cfg_get(config, "enabled", False))
