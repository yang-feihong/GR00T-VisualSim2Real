"""Measure the passive latch behavior of the procedural DoorMan door."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
LOWLEVEL_ROOT = SCRIPT_ROOT / "third_party" / "visual_wholebody_lowlevel"
sys.path.insert(0, str(LOWLEVEL_ROOT))

from legged_gym.utils.helpers import build_common_arg_parser  # noqa: E402
from legged_gym.utils.isaaclab_app import add_app_launcher_args, launch_app  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = build_common_arg_parser()
    add_app_launcher_args(parser)
    parser.set_defaults(
        num_envs=1,
        enable_doorman_scene=True,
        doorman_randomize_robot_init=False,
        doorman_randomize_door_init_state=False,
        action_delay_mode="undelayed",
        teleop_mode=True,
        mount_deg=90,
        headless=True,
    )
    parser.add_argument("--settle_steps", type=int, default=50)
    parser.add_argument("--push_steps", type=int, default=150)
    parser.add_argument("--torques", type=float, nargs="+", default=(1.0, 5.0, 10.0, 20.0))
    parser.add_argument(
        "--latch_positions",
        type=float,
        nargs="+",
        default=(0.0, 0.019, 0.020, 0.025),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/door_latch_diagnostic.json"))
    return parser


def _default_paths(args) -> None:
    repo = SCRIPT_ROOT.parent.parent
    run_dir = (
        repo
        / "ckpt"
        / "b2z1-low"
        / "20260918_102500_is-dc7zasdnsm4oadn4-devmachine-0_gpu2_agent1_uhryvo2cn"
    )
    if not args.ckpt_path:
        args.ckpt_path = str(run_dir / "model_54000.pt")
    if not args.run_metadata_path:
        args.run_metadata_path = str(run_dir / "run_metadata.json")
    if not args.robot_usd_path:
        args.robot_usd_path = str(
            repo
            / "gr00t"
            / "rl"
            / "data"
            / "robots"
            / "b2z1_lab30"
            / "b2z1_mount_90_x0p22_y0_z0p17.usd"
        )


def _world_bounds(stage, path: str) -> dict:
    from pxr import Gf, Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=False,
    )
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"missing runtime prim: {path}")
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    return {
        "min": [float(value) for value in aligned.GetMin()],
        "max": [float(value) for value in aligned.GetMax()],
    }


def _authored_geometry(stage, path: str) -> dict:
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    result = {
        "local_to_world": [[float(transform[row][col]) for col in range(4)] for row in range(4)]
    }
    if prim.IsA(UsdGeom.Cone):
        cone = UsdGeom.Cone(prim)
        radius = float(cone.GetRadiusAttr().Get())
        height = float(cone.GetHeightAttr().Get())
        axis = str(cone.GetAxisAttr().Get())
        points = []
        # USD cones are centered on their axis, with the tip at +height/2.
        for index in range(128):
            angle = 2.0 * 3.141592653589793 * index / 128.0
            radial_a = radius * __import__("math").cos(angle)
            radial_b = radius * __import__("math").sin(angle)
            if axis == "X":
                local = Gf.Vec3d(-height / 2.0, radial_a, radial_b)
            elif axis == "Y":
                local = Gf.Vec3d(radial_a, -height / 2.0, radial_b)
            else:
                local = Gf.Vec3d(radial_a, radial_b, -height / 2.0)
            points.append(transform.Transform(local))
        tip_local = {
            "X": Gf.Vec3d(height / 2.0, 0.0, 0.0),
            "Y": Gf.Vec3d(0.0, height / 2.0, 0.0),
            "Z": Gf.Vec3d(0.0, 0.0, height / 2.0),
        }[axis]
        points.append(transform.Transform(tip_local))
        result.update(
            {
                "type": "Cone",
                "radius": radius,
                "height": height,
                "axis": axis,
                "sampled_world_bounds": {
                    "min": [min(float(point[i]) for point in points) for i in range(3)],
                    "max": [max(float(point[i]) for point in points) for i in range(3)],
                },
            }
        )
    elif prim.IsA(UsdGeom.Cube):
        result.update({"type": "Cube", "size": float(UsdGeom.Cube(prim).GetSizeAttr().Get())})
    return result


def _step_physics(env, steps: int) -> None:
    for _ in range(steps):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)


def _reset_door(
    env,
    joint_count: int,
    hinge_idx: int,
    handle_idx: int,
    latch_idx: int,
    latch_position: float,
) -> None:
    joint_pos = torch.zeros((1, joint_count), device=env.device)
    joint_pos[:, handle_idx] = latch_position / 0.03 * (45.0 * 3.141592653589793 / 180.0)
    joint_pos[:, latch_idx] = latch_position
    joint_vel = torch.zeros_like(joint_pos)
    env.door.write_joint_state_to_sim(joint_pos, joint_vel)
    env.door.set_joint_position_target(joint_pos)
    env.door.set_joint_effort_target(
        torch.zeros((1, 1), device=env.device),
        joint_ids=[hinge_idx],
    )


def main() -> None:
    args = _parser().parse_args()
    _default_paths(args)
    simulation_app = launch_app(args)
    try:
        from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        runner = ManipLoco_Policy(args)
        env = runner.env
        door = env.door
        if door is None:
            raise RuntimeError("DoorMan door was not created")

        # Keep the robot out of the experiment while retaining the same scene.
        robot_pose = env.robot.data.root_state_w[:, :7].clone()
        robot_pose[:, 0] = 5.0
        robot_pose[:, 1] = 5.0
        env.robot.write_root_pose_to_sim(robot_pose)
        env.robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))

        joint_names = list(door.data.joint_names)
        body_names = list(door.data.body_names)
        hinge_idx = joint_names.index("hinge_joint")
        handle_idx = joint_names.index("handle_joint")
        latch_idx = joint_names.index("latch_joint")
        stage = omni.usd.get_context().get_stage()
        base = "/World/envs/env_0/door"
        paths = {
            "latch": f"{base}/latch_link/latch_geom",
            "left_frame": f"{base}/root/left_frame",
            "right_frame": f"{base}/root/right_frame",
            "panel": f"{base}/door_panel/panel",
        }
        geometry = {
            name: {
                "usd_extent_world_aabb": _world_bounds(stage, path),
                "authored_geometry": _authored_geometry(stage, path),
            }
            for name, path in paths.items()
        }

        collision = {}
        for name, path in paths.items():
            prim = stage.GetPrimAtPath(path)
            collision_api = UsdPhysics.CollisionAPI(prim)
            physx_api = PhysxSchema.PhysxCollisionAPI(prim)
            collision[name] = {
                "path": path,
                "collision_enabled": (
                    bool(collision_api.GetCollisionEnabledAttr().Get())
                    if collision_api
                    else False
                ),
                "contact_offset": (
                    physx_api.GetContactOffsetAttr().Get() if physx_api else None
                ),
                "rest_offset": physx_api.GetRestOffsetAttr().Get() if physx_api else None,
            }

        articulation_attributes = {}
        for path in (base, f"{base}/root"):
            prim = stage.GetPrimAtPath(path)
            articulation_attributes[path] = {
                name: prim.GetAttribute(name).Get()
                for name in (
                    "physxArticulation:enabledSelfCollisions",
                    "physxArticulation:solverPositionIterationCount",
                    "physxArticulation:solverVelocityIterationCount",
                )
                if prim.GetAttribute(name).IsValid()
            }
        hinge_prim = stage.GetPrimAtPath(f"{base}/root/hinge_joint")
        hinge_physx = PhysxSchema.PhysxJointAPI(hinge_prim)
        hinge_drive = UsdPhysics.DriveAPI.Get(hinge_prim, "angular")
        hinge_properties = {
            "joint_friction": hinge_physx.GetJointFrictionAttr().Get(),
            "drive_stiffness": hinge_drive.GetStiffnessAttr().Get(),
            "drive_damping": hinge_drive.GetDampingAttr().Get(),
            "drive_max_force": hinge_drive.GetMaxForceAttr().Get(),
            "drive_target_position": hinge_drive.GetTargetPositionAttr().Get(),
            "drive_target_velocity": hinge_drive.GetTargetVelocityAttr().Get(),
        }

        trials = []
        for latch_position in args.latch_positions:
            for torque_value in args.torques:
                _reset_door(
                    env,
                    len(joint_names),
                    hinge_idx,
                    handle_idx,
                    latch_idx,
                    latch_position,
                )
                _step_physics(env, args.settle_steps)
                initial = door.data.joint_pos[0].detach().cpu().tolist()
                torque = torch.tensor([[float(torque_value)]], device=env.device)
                door.set_joint_effort_target(
                    torque,
                    joint_ids=[hinge_idx],
                )
                samples = []
                for step in range(args.push_steps):
                    _step_physics(env, 1)
                    if step < 10 or step % 10 == 9 or step == args.push_steps - 1:
                        samples.append(
                            {
                                "step": step + 1,
                                "position": door.data.joint_pos[0].detach().cpu().tolist(),
                                "velocity": door.data.joint_vel[0].detach().cpu().tolist(),
                            }
                        )
                final = door.data.joint_pos[0].detach().cpu().tolist()
                trials.append(
                    {
                        "requested_latch_position_m": float(latch_position),
                        "hinge_effort_nm": float(torque_value),
                        "initial": initial,
                        "final": final,
                        "hinge_peak_rad": max(abs(row["position"][hinge_idx]) for row in samples),
                        "handle_peak_rad": max(abs(row["position"][handle_idx]) for row in samples),
                        "latch_peak_m": max(abs(row["position"][latch_idx]) for row in samples),
                        "samples": samples,
                    }
                )

        result = {
            "seed": int(args.seed),
            "joint_names": joint_names,
            "body_names": body_names,
            "geometry_world_aabb": geometry,
            "collision": collision,
            "articulation_attributes": articulation_attributes,
            "hinge_properties": hinge_properties,
            "trials": trials,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        if hasattr(env, "close"):
            env.close()
    finally:
        if not args.headless and simulation_app is not None:
            simulation_app.close()
        if args.headless:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
