# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from isaaclab.sim.utils import bind_visual_material
from pxr import Usd

_DYNAMIC_MATERIAL_RANDOMIZERS = []


def _resolve_material_randomization_prim_paths(stage: Usd.Stage, prim_path: str) -> list[str]:
    if stage.GetPrimAtPath(prim_path).IsValid():
        return [prim_path]

    suffix = prim_path
    if "/root/" in prim_path:
        suffix = "/root/" + prim_path.split("/root/", 1)[1]

    return [str(prim.GetPath()) for prim in stage.Traverse() if str(prim.GetPath()).endswith(suffix)]


# Build the class lazily so that the module can be imported before SimulationApp init.
# The actual class is created on first access via get_material_randomization_class().
_MaterialRandomization = None


def get_material_randomization_class():
    """Return the MaterialRandomization class, creating it on first call."""
    global _MaterialRandomization
    if _MaterialRandomization is not None:
        return _MaterialRandomization

    class MaterialRandomization:
        @staticmethod
        def add_to_prim(
            stage: Usd.Stage,
            prim_path: str,
            randomization_interval: float,
            randomization_material_list: list[str],
        ):
            _DYNAMIC_MATERIAL_RANDOMIZERS.append(
                {
                    "prim_path": str(prim_path),
                    "randomization_interval": float(randomization_interval),
                    "randomization_material_list": list(randomization_material_list),
                    "last_randomization_time": 0.0,
                    "randomization_interval_perturbation": np.random.uniform(
                        randomization_interval * -0.1, randomization_interval * 0.1
                    ),
                    "resolved_prim_paths": None,
                }
            )

    _MaterialRandomization = MaterialRandomization
    return _MaterialRandomization


def step_dynamic_material_randomization(stage: Usd.Stage, current_time: float):
    for randomizer in _DYNAMIC_MATERIAL_RANDOMIZERS:
        if (
            current_time - randomizer["last_randomization_time"]
            < randomizer["randomization_interval"]
            + randomizer["randomization_interval_perturbation"]
        ):
            continue

        material_list = randomizer["randomization_material_list"]
        if not material_list:
            continue

        if randomizer["resolved_prim_paths"] is None:
            randomizer["resolved_prim_paths"] = _resolve_material_randomization_prim_paths(
                stage, randomizer["prim_path"]
            )
        if not randomizer["resolved_prim_paths"]:
            continue

        randomizer["last_randomization_time"] = current_time
        randomizer["randomization_interval_perturbation"] = np.random.uniform(
            randomizer["randomization_interval"] * -0.1,
            randomizer["randomization_interval"] * 0.1,
        )
        for resolved_prim_path in randomizer["resolved_prim_paths"]:
            bind_visual_material(
                resolved_prim_path,
                str(np.random.choice(material_list)),
                stage,
            )
