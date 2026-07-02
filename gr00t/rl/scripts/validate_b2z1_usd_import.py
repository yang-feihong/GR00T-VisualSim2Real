# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate that the converted B2Z1 USD can be referenced by Isaac Sim."""

from __future__ import annotations

from pathlib import Path

from gr00t.rl.scripts.convert_b2z1_urdf_to_usd import (
    ensure_isaacsim_library_path,
    isaacsim_extra_args,
)


def main():
    repo_root = Path(__file__).resolve().parents[3]
    usd_path = repo_root / "gr00t/rl/data/robots/b2z1/b2z1.usd"
    if not usd_path.is_file():
        raise FileNotFoundError(f"B2Z1 USD does not exist: {usd_path}")

    ensure_isaacsim_library_path()

    from isaacsim import SimulationApp

    import isaacsim

    experience = Path(isaacsim.__file__).resolve().parent / "kit/apps/omni.app.empty.kit"
    simulation_app = SimulationApp(
        {
            "headless": True,
            "extra_args": isaacsim_extra_args(),
            "create_new_stage": False,
        },
        experience=str(experience),
    )

    try:
        from pxr import Usd, UsdGeom

        source_stage = Usd.Stage.Open(str(usd_path))
        if source_stage is None:
            raise RuntimeError(f"Failed to open B2Z1 USD: {usd_path}")

        default_prim = source_stage.GetDefaultPrim()
        if not default_prim:
            raise RuntimeError(f"B2Z1 USD has no default prim: {usd_path}")

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        robot_prim = UsdGeom.Xform.Define(stage, "/World/B2Z1").GetPrim()
        robot_prim.GetReferences().AddReference(str(usd_path), str(default_prim.GetPath()))

        for _ in range(5):
            simulation_app.update()

        prim_paths = {str(prim.GetPath()) for prim in stage.Traverse()}
        required_names = {
            "base_link",
            "FR_hip_joint",
            "FL_hip_joint",
            "RR_hip_joint",
            "RL_hip_joint",
            "joint1",
            "joint6",
            "jointGripper",
            "gripper_link",
        }
        missing = [
            name
            for name in required_names
            if not any(path.endswith(f"/{name}") for path in prim_paths)
        ]
        if missing:
            raise RuntimeError(f"B2Z1 USD imported, but required prims are missing: {missing}")

        print(f"Imported B2Z1 USD into Isaac Sim stage: {usd_path}")
        print(f"Source default prim: {default_prim.GetPath()}")
        print(f"Referenced prim: /World/B2Z1")
        print(f"Imported prim count: {len(prim_paths)}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
