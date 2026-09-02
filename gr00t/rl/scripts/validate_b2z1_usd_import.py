# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate that the converted B2Z1 USD can be referenced by Isaac Sim."""

from __future__ import annotations

from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[3]
    usd_path = (
        repo_root
        / "gr00t/rl/data/robots/b2z1_lab30/b2z1_mount_0_x0p22_y0_z0p17.usd"
    )
    if not usd_path.is_file():
        raise FileNotFoundError(f"B2Z1 USD does not exist: {usd_path}")

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(
        {
            "headless": True,
            "enable_cameras": False,
            "experience": str(
                repo_root / "gr00t/rl/apps/b2z1.isaaclab.python.headless.no_render.kit"
            ),
        }
    ).app

    try:
        from pxr import Usd, UsdGeom, UsdPhysics

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

        source_prims = list(
            Usd.PrimRange.Stage(source_stage, Usd.TraverseInstanceProxies())
        )
        collision_roots = {
            "fixed": "/b2_description/link06/collisions/",
            "moving": "/b2_description/gripperMover/collisions/",
        }
        collision_counts = {
            name: sum(
                str(prim.GetPath()).startswith(root)
                and prim.HasAPI(UsdPhysics.CollisionAPI)
                for prim in source_prims
            )
            for name, root in collision_roots.items()
        }
        # link06 contains its wrist capsule in addition to the 25 fixed gripper hulls.
        expected_collision_counts = {"fixed": 26, "moving": 24}
        if collision_counts != expected_collision_counts:
            raise RuntimeError(
                "B2Z1 gripper collision decomposition is incomplete: "
                f"expected {expected_collision_counts}, got {collision_counts}"
            )

        print(f"Imported B2Z1 USD into Isaac Sim stage: {usd_path}")
        print(f"Source default prim: {default_prim.GetPath()}")
        print(f"Referenced prim: /World/B2Z1")
        print(f"Imported prim count: {len(prim_paths)}")
        print(f"Gripper collision counts: {collision_counts}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
