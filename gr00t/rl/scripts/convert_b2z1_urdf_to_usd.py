# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline converter for the B2Z1 URDF asset.

Run this with Isaac Sim / Isaac Lab's Python environment, not with plain system
Python:

    ./isaac-sim/python.sh gr00t/rl/scripts/convert_b2z1_urdf_to_usd.py

The training runtime continues to load the generated USD through the existing
IsaacSim._setup_scene() path.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import site
import sys
import types
from pathlib import Path


def isaacsim_roots():
    roots = []
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        isaacsim_dir = Path(site_dir) / "isaacsim"
        if isaacsim_dir.is_dir():
            roots.append(isaacsim_dir)
    return roots


def isaacsim_extension_roots():
    roots = []

    def add(path):
        path = Path(path)
        if path.is_dir():
            resolved = path.resolve()
            if resolved not in roots:
                roots.append(resolved)

    for isaacsim_dir in isaacsim_roots():
        for ext_dir_name in ("exts", "extscache", "extsPhysics", "extsDeprecated"):
            add(isaacsim_dir / ext_dir_name)

        kit_dir = isaacsim_dir / "kit/data/Kit"
        if kit_dir.is_dir():
            for ext_root in kit_dir.glob("**/exts*"):
                add(ext_root)
                for numbered_root in ext_root.glob("*"):
                    add(numbered_root)

    add(Path.home() / ".local/share/ov/data/exts/v2")
    return roots


def isaacsim_extra_args():
    extra_args = []
    for ext_dir in isaacsim_extension_roots():
        extra_args.extend(["--ext-folder", str(ext_dir)])

    extra_args.extend(
        [
            "--enable",
            "omni.kit.loop-isaac",
            "--enable",
            "omni.usd",
            "--enable",
            "omni.kit.usd.layers",
            "--/app/runLoops/main/rateLimitEnabled=true",
            "--enable",
            "omni.usd.schema.physx",
            "--/app/runLoops/main/rateLimitFrequency=60",
            "--/app/runLoops/main/rateLimitUseBusyLoop=false",
        ]
    )
    return extra_args


def isaacsim_library_paths():
    paths = []

    def add(path):
        path = Path(path)
        if path.is_dir():
            resolved = str(path.resolve())
            if resolved not in paths:
                paths.append(resolved)

    for isaacsim_dir in isaacsim_roots():
        add(isaacsim_dir / "kit/kernel/plugins")

    for root in isaacsim_extension_roots():
        for lib_dir in root.rglob("*"):
            if lib_dir.name in {"bin", "lib", "libs"}:
                add(lib_dir)

    add(Path(sys.executable).resolve().parents[1] / "lib")
    return paths


def ensure_isaacsim_library_path():
    if os.environ.get("B2Z1_ISAACSIM_LD_READY") == "1":
        return

    lib_paths = isaacsim_library_paths()
    if not lib_paths:
        return

    env = os.environ.copy()
    old_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths + ([old_ld_path] if old_ld_path else []))
    env["B2Z1_ISAACSIM_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def parse_args():
    repo_root = Path(__file__).resolve().parents[3]
    default_urdf = (
        repo_root
        / "gr00t/rl/third_party/visual_wholebody_lowlevel/resources/robots/b2z1/urdf"
        / "generated/b2z1_mount_0_x0p22_y0_z0p17.urdf"
    )
    default_usd = repo_root / "gr00t/rl/data/robots/b2z1/b2z1.usd"

    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=default_urdf)
    parser.add_argument("--usd", type=Path, default=default_usd)
    parser.add_argument("--merge-fixed-joints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend", choices=("isaaclab", "importer"), default="isaaclab")
    return parser.parse_args()


def convert_with_isaacsim_urdf_importer(args):
    import carb
    from pxr import Usd

    print("Finding Isaac Sim URDF importer extension...", flush=True)
    urdf_ext_dir = next(
        (
            ext_dir
            for root in isaacsim_extension_roots()
            for ext_dir in root.glob("isaacsim.asset.importer.urdf*")
            if (ext_dir / "bin").is_dir()
        ),
        None,
    )
    if urdf_ext_dir is None:
        raise RuntimeError("Could not find isaacsim.asset.importer.urdf extension.")

    print(f"Loading URDF importer plugin from: {urdf_ext_dir}", flush=True)
    carb.get_framework().load_plugins(
        loaded_file_wildcards=["*.plugin"], search_paths=[str(urdf_ext_dir / "bin")]
    )
    print("URDF importer plugin loaded.", flush=True)

    package_name = "isaacsim.asset.importer.urdf"
    python_dir = urdf_ext_dir / package_name.replace(".", "/")
    shared_lib = next(python_dir.glob("_urdf*.so"))

    package = types.ModuleType(package_name)
    package.__path__ = [str(python_dir)]
    sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(f"{package_name}._urdf", shared_lib)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load URDF importer module: {shared_lib}")
    urdf_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = urdf_module
    spec.loader.exec_module(urdf_module)
    print("URDF importer Python bindings loaded.", flush=True)

    print("Acquiring URDF importer interface...", flush=True)
    importer = urdf_module.acquire_urdf_interface()
    print(f"URDF importer interface acquired: {importer}", flush=True)
    print("Creating URDF import config...", flush=True)
    import_config = urdf_module.ImportConfig()
    print("URDF import config created.", flush=True)

    config_values = (
        ("set_distance_scale", 1.0),
        ("set_make_default_prim", True),
        ("set_create_physics_scene", False),
        ("set_density", 0.0),
        ("set_convex_decomp", False),
        ("set_collision_from_visuals", False),
        ("set_merge_fixed_joints", bool(args.merge_fixed_joints)),
        ("set_fix_base", False),
        ("set_self_collision", False),
        ("set_parse_mimic", False),
        ("set_replace_cylinders_with_capsules", False),
    )
    for setter_name, value in config_values:
        print(f"Configuring {setter_name}={value!r}", flush=True)
        getattr(import_config, setter_name)(value)

    urdf_dir = str(args.urdf.resolve().parent)
    urdf_name = args.urdf.name
    print(f"Parsing URDF: {args.urdf}", flush=True)
    robot_model = importer.parse_urdf(urdf_dir, urdf_name, import_config)
    if robot_model is None:
        raise RuntimeError(f"Failed to parse URDF file: {args.urdf}")

    args.usd.unlink(missing_ok=True)
    Usd.Stage.CreateNew(str(args.usd)).Save()
    print(f"Importing robot to USD: {args.usd}", flush=True)
    result = importer.import_robot(
        urdf_dir, urdf_name, robot_model, import_config, str(args.usd.resolve()), False
    )
    if not result:
        raise RuntimeError(f"Failed to import URDF as USD: {args.usd}")


def main():
    args = parse_args()
    args.usd.parent.mkdir(parents=True, exist_ok=True)
    ensure_isaacsim_library_path()

    simulation_app = None
    try:
        if args.backend == "importer":
            from isaacsim import SimulationApp

            empty_experience = (
                Path(__import__("isaacsim").__file__).resolve().parent
                / "kit/apps/omni.app.empty.kit"
            )
            simulation_app = SimulationApp(
                {
                    "headless": args.headless,
                    "extra_args": isaacsim_extra_args(),
                    "create_new_stage": False,
                },
                experience=str(empty_experience),
            )
            print("Isaac Sim app started; converting URDF to USD...", flush=True)
            try:
                convert_with_isaacsim_urdf_importer(args)
                print(f"Wrote USD: {args.usd}")
            finally:
                simulation_app.close()
            return

        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(
            {
                "headless": args.headless,
                "enable_cameras": False,
                "experience": str(
                    Path(__file__).resolve().parents[1]
                    / "apps/b2z1.isaaclab.python.headless.no_render.kit"
                ),
            }
        ).app

        import isaaclab.sim as sim_utils
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    except ImportError:
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=str(args.urdf),
        usd_dir=str(args.usd.parent),
        usd_file_name=args.usd.name,
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=False,
        merge_fixed_joints=bool(args.merge_fixed_joints),
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        self_collision=False,
        collision_from_visuals=False,
        replace_cylinders_with_capsules=True,
    )
    converter = UrdfConverter(cfg)
    print(f"Wrote USD: {converter.usd_path}")
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
