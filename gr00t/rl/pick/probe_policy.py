"""Isolated batched whole-body reach measurements, not collection acceptance."""

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import torch

from gr00t.rl.pick.grasp_clearance import GRASP_CLEARANCE_M


def command_bank(trace, path, stations, frames):
    """Cross every station with recorded reference goals and a proposed goal."""
    positions = torch.cat(
        [trace[frame]["ee_goal_cart_world"] for frame in frames] + [path[0][:, -1]]
    )
    quats = torch.cat(
        [trace[frame]["ee_goal_orn_quat_world"] for frame in frames] + [path[1][:, -1]]
    )
    return (
        stations.repeat_interleave(len(positions), 0),
        positions.repeat(len(stations), 1),
        quats.repeat(len(stations), 1),
    )


def offset_command_goals(intended, height_offsets):
    if height_offsets.shape != intended.shape[:1]:
        raise ValueError("Each diagnostic command needs its own height offset")
    commanded = intended.clone()
    commanded[:, 2] += height_offsets.to(commanded)
    return commanded


def fk_descent_pose(start_root, end_root, start_joints, end_joints, fraction):
    from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

    from gr00t.rl.auto_open.planning_geometry import _slerp
    from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian

    t = fraction**3 * (10 + fraction * (-15 + 6 * fraction))
    roots = torch.lerp(start_root, end_root, t)
    roots[:, 3:7] = _slerp(start_root[:, 3:7], end_root[:, 3:7], roots.new_full((len(roots),), t))
    joints = torch.lerp(start_joints, end_joints, t)
    positions, rotations, _ = b2z1_gripper_pose_and_jacobian(
        roots[:, :3], roots[:, [4, 5, 6, 3]], joints
    )
    return positions, _matrix_to_quaternion_wxyz(rotations)


def synthetic_reach_bank(count, *, offset=0, seed=7919, topdown=False):
    """Asset-independent low workspace coverage, with disjoint reproducible shards."""
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat

    if count < 1 or offset < 0:
        raise ValueError(
            "Synthetic reach coverage requires a positive count and nonnegative offset"
        )
    sampler = torch.quasirandom.SobolEngine(12, scramble=True, seed=seed)
    if offset:
        sampler.fast_forward(offset)
    samples = sampler.draw(count)
    low = samples.new_tensor([0.35, -0.30, 0.08])
    high = samples.new_tensor([0.85, 0.30, 0.30])
    if topdown:
        low[2], high[2] = 0.09, 0.19
    goals = low + samples[:, :3] * (high - low)
    pitch_range = math.pi / 12 if topdown else math.pi / 4
    angles = (samples[:, 3:6] * 2 - 1) * samples.new_tensor([math.pi, pitch_range, math.pi])
    angles[:, 1] += math.pi / 2
    quats = _euler_xyz_quat(angles)
    arm_jitter = (samples[:, 6:] * 2 - 1) * 0.05
    return torch.zeros(count, 3), goals, quats, arm_jitter


def synthetic_fk_reach_bank(count, limits, open_jaw_cloud, *, offset=0, seed=7919):
    """Asset-independent low poses conditioned on unsaturated arm geometry, on GPU."""
    from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
    from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian

    if count < 1 or offset < 0:
        raise ValueError("FK reach sampling requires positive count and nonnegative offset")
    generator = torch.Generator(device=limits.device).manual_seed(seed)
    samples = torch.rand(1048576, 14, device=limits.device, dtype=limits.dtype, generator=generator)
    joints = limits[:, 0] + 0.08 + samples[:, :6] * (limits[:, 1] - limits[:, 0] - 0.16)
    roots = samples.new_zeros(len(samples), 3)
    roots[:, 2] = 0.34 + 0.04 * samples[:, 6]
    angles = torch.zeros_like(roots)
    angles[:, 1] = (6 + 4 * samples[:, 7]) * (math.pi / 180)
    body = _euler_xyz_quat(angles)
    position, rotation, _ = b2z1_gripper_pose_and_jacobian(roots, body[:, [1, 2, 3, 0]], joints)
    eligible = (
        (position[:, 0] >= 0.35)
        & (position[:, 0] <= 0.85)
        & (position[:, 1].abs() <= 0.30)
        & (position[:, 2] >= 0.075)
        & (position[:, 2] <= 0.20)
        & (rotation[:, 2, 0] <= -math.cos(math.pi / 4))
    )
    indices = eligible.nonzero().flatten()
    quaternion = _matrix_to_quaternion_wxyz(rotation[indices])
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate

    bottom = _quat_rotate(quaternion[:, None], open_jaw_cloud[None])[..., 2].amin(-1)
    floor_clear = position[indices, 2] + bottom >= 0.002
    indices = indices[floor_clear]
    quaternion = quaternion[floor_clear]
    if len(indices) < offset + count:
        raise ValueError(
            f"Only {len(indices)} FK-conditioned low commands; requested {offset}+{count}"
        )
    chosen = indices[offset : offset + count]
    return (
        samples.new_zeros(count, 3),
        position[chosen],
        quaternion[offset : offset + count],
        (samples[chosen, 8:] * 2 - 1) * 0.05,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--recorded-grasp", action="store_true")
    parser.add_argument("--replay-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--ik-orientation-gain", type=float)
    parser.add_argument("--unconstrained-arm-ik", action="store_true")
    parser.add_argument("--reference-frames", type=int, nargs="+", required=True)
    parser.add_argument("--initial-frame", type=int, required=True)
    parser.add_argument("--station-yaw-offsets-deg", type=float, nargs="+", default=[0])
    parser.add_argument("--station-indices", type=int, nargs="+")
    parser.add_argument("--candidate-replay-json", type=Path)
    parser.add_argument("--candidate-indices", type=int, nargs="+")
    parser.add_argument("--station-forward-offsets-m", type=float, nargs="+", default=[0])
    parser.add_argument("--station-side-offsets-m", type=float, nargs="+", default=[0])
    parser.add_argument("--station-joint-samples", type=int, default=0)
    parser.add_argument("--local-orientation-offsets-deg", type=float, nargs="+", default=[0])
    parser.add_argument("--pocket-depth-preferences", type=float, nargs="+", default=[0])
    parser.add_argument("--fk-body-heights", type=float, nargs="+")
    parser.add_argument("--fk-body-pitches-deg", type=float, nargs="+", default=[0])
    parser.add_argument("--blend-to-candidate-index", type=int)
    parser.add_argument("--blend-steps", type=int, default=41)
    parser.add_argument("--synthetic-count", type=int, default=0)
    parser.add_argument("--synthetic-offset", type=int, default=0)
    parser.add_argument("--synthetic-seed", type=int, default=7919)
    parser.add_argument("--synthetic-topdown", action="store_true")
    parser.add_argument("--synthetic-fk", action="store_true")
    parser.add_argument("--synthetic-catalog", type=Path)
    parser.add_argument("--prewalk-steps", type=int, default=0)
    parser.add_argument("--restore-stance", action="store_true")
    parser.add_argument("--command-height-offsets-m", type=float, nargs="+", default=[0.0])
    parser.add_argument("--gripper-angle-rad", type=float, default=0.0)
    args = parser.parse_args()
    if args.synthetic_count < 0 or args.synthetic_offset < 0:
        parser.error("Synthetic count and offset must be nonnegative")
    if args.prewalk_steps < 0 or (args.prewalk_steps and not args.synthetic_count):
        parser.error("Prewalk measurements require independent synthetic commands")
    if args.restore_stance and args.synthetic_count:
        parser.error("Recorded stance replay cannot be exported as independent calibration")
    if any(not math.isfinite(value) for value in args.command_height_offsets_m):
        parser.error("Command offsets must be finite")
    if args.command_height_offsets_m != [0.0] and (args.synthetic_count or args.fk_body_heights):
        parser.error("Command compensation probes require recorded targets, without FK paths")
    if args.synthetic_catalog and (not args.synthetic_count or args.synthetic_fk):
        parser.error("Catalog measurements require a synthetic count, without FK sampling")
    if args.synthetic_catalog:
        metadata = json.loads(args.synthetic_catalog.read_text())
        if args.synthetic_count != 5 * len(metadata["goals_world"]):
            parser.error("Catalog probe count must equal commands times five offsets")
    if args.synthetic_fk and (not args.synthetic_count or args.gripper_angle_rad != -math.pi / 3):
        parser.error("FK-conditioned measurements require a synthetic count and fully open jaws")
    if not math.isfinite(args.gripper_angle_rad) or not -math.pi / 3 <= args.gripper_angle_rad <= 0:
        parser.error("Gripper angle must be within the physical open/closed range")
    if args.synthetic_count and (
        args.recorded_grasp
        or args.replay_path is not None
        or args.candidate_replay_json is not None
        or args.proposal is not None
        or args.station_indices is not None
        or args.fk_body_heights is not None
    ):
        parser.error("Synthetic measurements cannot be combined with recorded pose proposals")
    if args.recorded_grasp:
        if args.replay_path is not None or args.candidate_replay_json is not None:
            parser.error("Recorded grasp cannot be combined with another candidate source")
        args.candidate_replay_json = args.run / "summary.json"
    if args.ik_orientation_gain is not None and (
        not math.isfinite(args.ik_orientation_gain) or args.ik_orientation_gain <= 0
    ):
        parser.error("ik-orientation-gain must be finite and positive")
    if args.replay_path is None and args.candidate_replay_json is None and not args.synthetic_count:
        parser.error("Provide --replay-path or --candidate-replay-json")
    if args.steps < 150:
        parser.error("At least 150 measurement steps are required")
    if not all(0 <= value <= 1 for value in args.pocket_depth_preferences):
        parser.error("Pocket depth preferences must be in [0, 1]")
    if (args.output / "summary.json").exists():
        parser.error("Refusing to overwrite completed measurements")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "request.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    from gr00t.rl.pick.recorded_states import load_recorded_states

    trace = load_recorded_states(args.run)
    proposal = (
        {"station_poses": [[0, 0, 0]]}
        if args.synthetic_count
        else json.loads(args.proposal.read_text())
        if args.proposal is not None
        else json.loads((args.run / "summary.json").read_text())["planning"]["initial_station"]
    )
    station_poses = torch.tensor(proposal["station_poses"], dtype=torch.float32)
    if args.station_indices is not None:
        station_poses = station_poses[args.station_indices]
    forward, side = torch.meshgrid(
        station_poses.new_tensor(
            [0] if args.station_joint_samples else args.station_forward_offsets_m
        ),
        station_poses.new_tensor(
            [0] if args.station_joint_samples else args.station_side_offsets_m
        ),
        indexing="ij",
    )
    forward, side = forward.flatten(), side.flatten()
    heading = station_poses[:, 2]
    direction = torch.stack((heading.cos(), heading.sin()), -1)
    lateral = torch.stack((-heading.sin(), heading.cos()), -1)
    station_poses = station_poses[:, None].expand(-1, len(forward), -1).clone()
    station_poses[..., :2] += forward[None, :, None] * direction[:, None]
    station_poses[..., :2] += side[None, :, None] * lateral[:, None]
    station_poses = station_poses.flatten(0, 1)
    yaw_offsets = torch.tensor(
        [0] if args.station_joint_samples else args.station_yaw_offsets_deg
    ) * (math.pi / 180)
    station_poses = station_poses[:, None].expand(-1, len(yaw_offsets), -1).clone()
    station_poses[..., 2] += yaw_offsets
    if args.station_joint_samples:
        if args.station_joint_samples < 1 or args.synthetic_count:
            parser.error("Joint station sampling requires recorded poses and a positive count")
        station_poses = station_poses.repeat_interleave(args.station_joint_samples, dim=0)
    if args.synthetic_count:
        stations, goals, quats, synthetic_jitter = synthetic_reach_bank(
            args.synthetic_count,
            offset=args.synthetic_offset,
            seed=args.synthetic_seed,
            topdown=args.synthetic_topdown,
        )
    elif args.candidate_replay_json is None:
        path = torch.load(args.replay_path, weights_only=False, map_location="cpu")["approach"]
        stations, goals, quats = command_bank(
            trace, path, station_poses.flatten(0, 1), args.reference_frames
        )
    if args.candidate_replay_json:
        replay = json.loads(args.candidate_replay_json.read_text())
        if args.recorded_grasp:
            replay = replay["planning"]["grasp_attempts"][0]
            if args.candidate_indices is None:
                quality = replay["convex_contact_model"]
                args.candidate_indices = [
                    index
                    for index, fit in enumerate(quality["pocket_fit"][0])
                    if fit
                    and quality["antipodal_margin"][0][index] >= 0
                    and quality["environment_clearance_m"][0][index] >= GRASP_CLEARANCE_M
                ]
                if not args.candidate_indices:
                    parser.error("Recorded grasp has no geometrically admitted candidate")
        bank = torch.tensor(
            replay["candidate_orientations_world_wxyz"]
            if "candidate_orientations_world_wxyz" in replay
            else replay["selected_grasp_orientation"]
        ).reshape(-1, 4)
        if args.candidate_indices is not None:
            bank = bank[args.candidate_indices]
        if args.blend_to_candidate_index is not None:
            if len(bank) != 1 or args.blend_steps < 2:
                parser.error("Blend measurements require one source pose and at least two steps")
            bank = bank.repeat_interleave(args.blend_steps, 0)
        perturbations = len(args.local_orientation_offsets_deg) ** 3
        bank = bank.repeat_interleave(perturbations, 0)
        bank = bank.repeat_interleave(len(args.pocket_depth_preferences), 0)
        poses = station_poses.flatten(0, 1)
        if args.station_joint_samples:
            if perturbations != 1 or len(args.pocket_depth_preferences) != 1:
                parser.error("Joint pose sampling requires singleton local tilt/depth options")
            generator = torch.Generator().manual_seed(args.synthetic_seed)
            indices = torch.randint(len(bank), (len(poses),), generator=generator)
            stations, quats = poses, bank[indices]
        else:
            stations = poses.repeat_interleave(len(bank), 0)
            quats = bank.repeat(len(poses), 1)
        goals = torch.zeros(len(quats), 3)
    command_height_offsets = goals.new_tensor(args.command_height_offsets_m).repeat(len(goals))
    stations, goals, quats = (
        value.repeat_interleave(len(args.command_height_offsets_m), 0)
        for value in (stations, goals, quats)
    )
    base_count = len(stations)
    if args.fk_body_heights is not None:
        count = len(args.fk_body_heights) * len(args.fk_body_pitches_deg)
        stations, goals, quats = (value.repeat(count, 1) for value in (stations, goals, quats))
    from legged_gym.utils.isaaclab_app import launch_app

    from gr00t.rl.auto_open.cli import _default_paths, _parser

    options = _parser().parse_args(
        [
            "--output_dir",
            str(args.output),
            "--headless",
            "--device",
            "cuda:0",
            "--disable_rgb_cameras",
            "--summary_only",
            "--num_envs",
            str(len(stations)),
            "--env_spacing",
            "8",
            "--seed",
            "1700",
        ]
    )
    options.enable_doorman_scene = False
    _default_paths(options)
    app = launch_app(options)
    runner = None
    try:
        from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy

        from gr00t.rl.auto_open.execution import StageExecutionMixin, TensorCommandBuffers
        from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_distance

        class ReachRunner(ManipLoco_Policy):
            def configure_env_cfg(self, cfg):
                cfg.check_terminations = False
                cfg.enable_contact_sensor = True
                cfg.robot.spawn.activate_contact_sensors = True
                cfg.env.teleop_ee_goal_z_limit = [-0.75, 0.6]
                cfg.robot.spawn.rigid_props.solver_position_iteration_count = 16
                cfg.robot.spawn.rigid_props.solver_velocity_iteration_count = 2
                if args.ik_orientation_gain is not None:
                    cfg.goal_ee.teleop_ik_orientation_gain = args.ik_orientation_gain

        runner = ReachRunner(options)
        env = runner.env
        if args.unconstrained_arm_ik:
            from types import MethodType

            from legged_gym.utils.math import orientation_error

            def original_training_arm_targets(instance):
                goal_position, goal_orientation = instance._get_ik_control_goal()
                orientation = instance.ee_orn
                orientation = orientation / orientation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                rotation_error = orientation_error(goal_orientation, orientation)
                rotation_error *= float(instance.cfg.goal_ee.teleop_ik_orientation_gain)
                error = torch.cat((goal_position - instance.ee_pos, rotation_error), -1)
                return instance.dof_pos[:, instance.arm_joint_ids] + instance._control_ik(
                    error[..., None]
                )

            env._get_arm_pos_targets = MethodType(original_training_arm_targets, env)
        device = env.device
        torch.set_grad_enabled(False)
        stations, goals, quats = (value.to(device) for value in (stations, goals, quats))
        if args.station_joint_samples:
            generator = torch.Generator(device=device).manual_seed(args.synthetic_seed)
            draws = torch.rand((len(stations), 3), device=device, generator=generator)
            ranges = stations.new_tensor(
                [
                    [min(values), max(values)]
                    for values in (
                        args.station_forward_offsets_m,
                        args.station_side_offsets_m,
                        [value * math.pi / 180 for value in args.station_yaw_offsets_deg],
                    )
                ]
            )
            offsets = ranges[:, 0] + draws * (ranges[:, 1] - ranges[:, 0])
            c, s = stations[:, 2].cos(), stations[:, 2].sin()
            stations[:, 0] += offsets[:, 0] * c - offsets[:, 1] * s
            stations[:, 1] += offsets[:, 0] * s + offsets[:, 1] * c
            stations[:, 2] += offsets[:, 2]
        if args.synthetic_fk:
            from gr00t.rl.pick.collision import ArmCollisionSamples

            model = torch.load(
                args.run / "collision_model.pt", weights_only=False, map_location=device
            )
            collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
            collision.gripper_surface = model["gripper_surface"]
            cloud = torch.cat(collision._gripper_local_surfaces(goals))
            stations, goals, quats, synthetic_jitter = synthetic_fk_reach_bank(
                args.synthetic_count,
                env.dof_pos_limits[env.arm_joint_ids],
                cloud,
                offset=args.synthetic_offset,
                seed=args.synthetic_seed,
            )
        if args.synthetic_catalog:
            from gr00t.rl.pick.measured_commands import load_measured_commands

            local_goal, local_quat, _, _ = load_measured_commands([args.synthetic_catalog], goals)
            offsets = goals.new_tensor([-0.06, -0.03, 0.0, 0.03, 0.06])
            if len(local_goal) * len(offsets) != args.synthetic_count:
                raise ValueError("Catalog probe count must equal commands times five offsets")
            goals = local_goal.repeat_interleave(len(offsets), 0)
            goals[:, 0] += offsets.repeat(len(local_goal))
            quats = local_quat.repeat_interleave(len(offsets), 0)
            stations = goals.new_zeros(len(goals), 3)
        if args.candidate_replay_json:
            from gr00t.rl.auto_open.motion_math import _quat_multiply
            from gr00t.rl.auto_open.planning_geometry import _slerp

            if args.blend_to_candidate_index is not None:
                destination = quats.new_tensor(replay["candidate_orientations_world_wxyz"]).reshape(
                    -1, 4
                )[args.blend_to_candidate_index]
                fractions = torch.linspace(0, 1, args.blend_steps, device=device)
                fractions = fractions.repeat_interleave(
                    perturbations * len(args.pocket_depth_preferences)
                )
                quats = _slerp(
                    quats,
                    destination.expand_as(quats),
                    fractions.repeat(len(quats) // len(fractions)),
                )

            offsets = quats.new_tensor(args.local_orientation_offsets_deg) * (math.pi / 180)
            rotations = torch.stack(torch.meshgrid(offsets, offsets, offsets, indexing="ij"), -1)
            delta = _euler_xyz_quat(rotations.reshape(-1, 3))
            delta = delta.repeat_interleave(len(args.pocket_depth_preferences), 0)
            quats = _quat_multiply(quats, delta.repeat(len(quats) // len(delta), 1))
        physical = torch.ones(len(goals), dtype=torch.bool, device=device)
        pocket_diagnostics = {}
        if args.candidate_replay_json:
            from gr00t.rl.pick.collision import ArmCollisionSamples
            from gr00t.rl.pick.grasp_quality import load_grasp_support
            from gr00t.rl.pick.planning import pocket_pose_targets
            from gr00t.rl.pick.scene_geometry import read_scene_triangles

            episode = json.loads((args.run / "episode.json").read_text())
            model = torch.load(
                args.run / "collision_model.pt", weights_only=False, map_location="cpu"
            )
            collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
            collision.parts = [(i, p.to(device)) for i, p in model["parts"]]
            collision.arm_joint_ids = model["arm_joint_ids"].to(device)
            collision.gripper_surface = [p.to(device) for p in model["gripper_surface"]]
            points, faces = read_scene_triangles(
                Path(episode["scene_usd_path"]), Path(episode["geometry_cache_root"])
            )
            from gr00t.rl.pick.scene_geometry import static_scene_geometry

            geometry = static_scene_geometry(
                points, faces, device, backend="native", ground_plane_z_m=0.0
            )
            supports = load_grasp_support(episode["asset"]["usd_path"], device)
            from gr00t.rl.pick.geometry_views import TranslatedGeometry
            from gr00t.rl.pick.support_features import uniform_surface_support

            geometry = TranslatedGeometry(
                geometry, supports.new_tensor(episode.get("batch_scene_origin_m", [0, 0, 0]))
            )
            if len(supports) > 9984:
                supports, support_mask = uniform_surface_support(
                    supports[None],
                    torch.ones((1, len(supports)), device=device, dtype=torch.bool),
                    count=9984,
                )
                supports = supports[0, support_mask[0]]
            object_state = trace[args.initial_frame]["object_state"].to(device)
            depth_preference = quats.new_tensor(args.pocket_depth_preferences).repeat(
                len(quats) // len(args.pocket_depth_preferences)
            )
            unique, inverse_quats = torch.unique(
                torch.cat((quats, depth_preference[:, None]), -1), dim=0, return_inverse=True
            )
            unique_quats = unique[:, :4]
            unique_goals, quality = pocket_pose_targets(
                unique_quats[None],
                object_state[:, :3],
                object_state[:, 3:7],
                supports,
                collision,
                geometry,
                depth_preference=unique[None, :, 4],
            )
            goals = unique_goals[0, inverse_quats]
            physical = (
                quality["pocket_fit"]
                & (quality["antipodal_margin"] >= 0)
                & (quality["environment_clearance_m"] >= GRASP_CLEARANCE_M)
            )[0, inverse_quats]
            pocket_diagnostics = {
                key: quality[key][0, inverse_quats].cpu().tolist()
                for key in (
                    "center_local",
                    "contact_depth_margin_m",
                    "center_depth_margin_m",
                    "antipodal_margin",
                )
            }
        ids = torch.arange(len(stations), device=device)
        root = env.robot.data.default_root_state.clone()
        root[:, :3] += env.scene.env_origins
        root[:, 0] += 4
        root[:, 7:] = 0
        root[:, 3:7] = _euler_xyz_quat(
            torch.stack((stations[:, 2] * 0, stations[:, 2] * 0, stations[:, 2]), -1)
        )
        if args.restore_stance:
            from gr00t.rl.auto_open.motion_math import _quat_multiply

            recorded_root = trace[args.initial_frame]["root_states"].to(device)
            w, x, y, z = recorded_root[:, 3:7].unbind(-1)
            recorded_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            angle = stations[:, 2] - recorded_yaw
            turn = torch.stack(((angle / 2).cos(), angle * 0, angle * 0, (angle / 2).sin()), -1)
            root[:, 3:7] = _quat_multiply(turn, recorded_root[:, 3:7])
            root[:, 2] = recorded_root[:, 2] + env.scene.env_origins[:, 2]
        goals[:, :2] += root[:, :2] - stations[:, :2]
        goals[:, 2] += env.scene.env_origins[:, 2]
        model = torch.load(args.run / "collision_model.pt", weights_only=False, map_location="cpu")
        synthetic_geometry = {}
        if args.synthetic_count:
            from gr00t.rl.auto_open.motion_math import _quat_multiply
            from gr00t.rl.auto_open.planning_geometry import _quat_rotate
            from gr00t.rl.pick.collision import ArmCollisionSamples

            collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
            collision.gripper_surface = [part.to(device) for part in model["gripper_surface"]]
            fixed, moving = collision._gripper_local_surfaces(goals)
            delta = math.pi / 3 + args.gripper_angle_rad
            rotation = goals.new_tensor([math.cos(delta / 2), 0, math.sin(delta / 2), 0])
            fixed_bottom = _quat_rotate(quats[:, None], fixed[None])[..., 2].amin(-1)
            moving_bottom = _quat_rotate(_quat_multiply(quats, rotation)[:, None], moving[None])[
                ..., 2
            ].amin(-1)
            synthetic_geometry["goal_gripper_floor_clearance_m"] = (
                (
                    goals[:, 2]
                    - env.scene.env_origins[:, 2]
                    + torch.minimum(fixed_bottom, moving_bottom)
                )
                .cpu()
                .tolist()
            )
        joints = env.default_dof_pos.clone()
        if args.restore_stance:
            joints[:] = trace[args.initial_frame]["dof_pos"].to(device)
        arms = trace[args.initial_frame]["dof_pos"][:, model["arm_joint_ids"]].to(device)
        if args.synthetic_count:
            episode = json.loads((args.run / "episode.json").read_text())
            arms = goals.new_tensor(episode["arm_preset"])[None] + synthetic_jitter.to(device)
        joints[:, env.arm_joint_ids] = arms
        joints[:, env.gripper_joint_ids] = args.gripper_angle_rad
        env.robot.write_root_pose_to_sim(root[:, :7])
        env.robot.write_root_velocity_to_sim(root[:, 7:])
        env.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        env.robot.set_joint_position_target(joints)
        runner.obs = env.get_observations()
        buffers = TensorCommandBuffers.from_env(env)
        buffers.base.zero_()
        buffers.arm_joint_mode.fill_(True)
        buffers.arm_joint[:] = arms
        buffers.gripper.fill_(args.gripper_angle_rad)
        executor = StageExecutionMixin()
        executor.env = env
        executor.arm_induced_pitch = torch.full((len(ids),), env.arm_induced_pitch, device=device)

        def step():
            lease = os.environ.get("AUTO_OPEN_LAUNCH_LEASE")
            if lease and (
                not Path(lease).exists() or time.time() - Path(lease).stat().st_mtime > 10
            ):
                raise RuntimeError("Launcher lease expired")
            buffers.flush(env)
            env._clip_teleop_targets()
            env._update_effective_teleop_inputs(force=True)
            return runner.step_once()["state"]

        for _ in range(100):
            state = step()
        if args.prewalk_steps:
            from gr00t.rl.auto_open.motion_math import _quat_multiply
            from gr00t.rl.auto_open.planning_geometry import _quat_rotate

            before_walk = state["root_states"].clone()
            phases = torch.arange(len(ids), device=device).remainder(min(50, args.prewalk_steps))
            duration = args.prewalk_steps - phases
            for index in range(args.prewalk_steps):
                buffers.base[:, 0] = torch.where(index < duration, 0.3, 0.0)
                state = step()
            buffers.base.zero_()
            for _ in range(100):
                state = step()

            def yaw(roots):
                w, x, y, z = roots[:, 3:7].unbind(-1)
                return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

            angle = yaw(state["root_states"]) - yaw(before_walk)
            turn = torch.stack(((angle / 2).cos(), angle * 0, angle * 0, (angle / 2).sin()), -1)
            relative = goals - before_walk[:, :3]
            goals[:, :2] = state["root_states"][:, :2] + _quat_rotate(turn, relative)[:, :2]
            quats = _quat_multiply(turn, quats)
        initial = state["ee_pos"].clone()
        initial_q = state["ee_orn"].clone()
        initial_root = state["root_states"].clone()
        endpoint_root = endpoint_joints = None
        endpoint_diagnostics = {}
        if args.fk_body_heights is not None:
            from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
            from gr00t.rl.pick.planning import continuous_inverse_seed, inverse_bank
            from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

            heights, pitches = torch.meshgrid(
                goals.new_tensor(args.fk_body_heights),
                goals.new_tensor(args.fk_body_pitches_deg) * (math.pi / 180),
                indexing="ij",
            )
            heights = heights.flatten().repeat_interleave(base_count)
            pitches = pitches.flatten().repeat_interleave(base_count)
            endpoint_root = initial_root.clone()
            endpoint_root[:, 2] = env.scene.env_origins[:, 2] + heights
            endpoint_root[:, 3:7] = _euler_xyz_quat(
                torch.stack((pitches * 0, pitches, stations[:, 2]), -1)
            )
            limits = goals.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
            initial_joints = state["dof_pos"][:, env.arm_joint_ids].clone()
            seeds = torch.stack(
                (
                    initial_joints,
                    limits.mean(-1).expand_as(initial_joints),
                    limits.sum(-1) - initial_joints,
                ),
                1,
            )
            solved = inverse_bank(
                goals[:, None, None].expand(-1, 3, 1, -1).flatten(0, 1),
                quats[:, None, None].expand(-1, 3, 1, -1).flatten(0, 1),
                endpoint_root[:, None].expand(-1, 3, -1).flatten(0, 1),
                seeds.flatten(0, 1),
                limits,
            )
            position_error = solved.position_error.norm(dim=-1).reshape(-1, 3)
            orientation_error = (
                2 * solved.orientation_error.norm(dim=-1).clamp_max(1).asin()
            ).reshape(-1, 3)
            joint_bank = solved.arm_joint_positions.reshape(-1, 3, 6)
            chosen = continuous_inverse_seed(
                position_error, orientation_error, joint_bank, initial_joints[:, None]
            )
            endpoint_joints = joint_bank.gather(1, chosen[..., None].expand(-1, 1, 6)).squeeze(1)
            endpoint_diagnostics = {
                "fk_body_heights": args.fk_body_heights,
                "fk_body_pitches_deg": args.fk_body_pitches_deg,
                "fk_endpoint_position_error_m": position_error.gather(1, chosen)
                .flatten()
                .cpu()
                .tolist(),
                "fk_endpoint_orientation_error_rad": orientation_error.gather(1, chosen)
                .flatten()
                .cpu()
                .tolist(),
            }
        aligned_q = torch.where((initial_q * quats).sum(-1, keepdim=True) < 0, -quats, quats)
        command_goals = goals.clone()
        if args.command_height_offsets_m != [0.0]:
            command_goals = offset_command_goals(goals, command_height_offsets)
        history = []
        contact_names = list(env.contact_sensor.body_names)
        nonfoot = [i for i, name in enumerate(contact_names) if not name.lower().endswith("foot")]
        contact_history = []
        started = time.perf_counter()
        for index in range(args.steps):
            fraction = min((index + 1) / 150, 1)
            reference = torch.lerp(initial, command_goals, fraction)
            reference_q = torch.lerp(initial_q, aligned_q, fraction)
            reference_q /= reference_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            if endpoint_root is not None and index < 150:
                reference, reference_q = fk_descent_pose(
                    initial_root, endpoint_root, initial_joints, endpoint_joints, fraction
                )
            executor.arm_pose_target(ids, reference, reference_q, buffers)
            state = step()
            contact_history.append(state["contact_forces_w"].norm(dim=-1))
            history.append(
                torch.cat(
                    (
                        state["root_states"],
                        state["ee_pos"],
                        state["ee_orn"],
                        state["dof_pos"][:, env.arm_joint_ids],
                        (state["ee_pos"] - goals).norm(dim=-1, keepdim=True),
                        _quat_distance(state["ee_orn"], quats)[:, None],
                    ),
                    -1,
                )
            )
            if index % 100 == 99:
                print(f"[pick:policy-probe] step={index + 1}/{args.steps}", flush=True)
        history = torch.stack(history)
        torch.cuda.synchronize()
        result = {
            **synthetic_geometry,
            "checkpoint_path": str(options.ckpt_path),
            "checkpoint_sha256": hashlib.sha256(Path(options.ckpt_path).read_bytes()).hexdigest(),
            **endpoint_diagnostics,
            "pocket_quality": pocket_diagnostics,
            "note": "No object or furniture: reachability measurement, not grasp acceptance",
            "candidate_order": "station-major; reference_frames, then proposed",
            "reference_frames": args.reference_frames,
            "initial_frame": args.initial_frame,
            "station_yaw_offsets_deg": args.station_yaw_offsets_deg,
            "station_indices": args.station_indices,
            "candidate_replay_json": str(args.candidate_replay_json),
            "candidate_indices": args.candidate_indices,
            "station_forward_offsets_m": args.station_forward_offsets_m,
            "station_side_offsets_m": args.station_side_offsets_m,
            "local_orientation_offsets_deg": args.local_orientation_offsets_deg,
            "pocket_depth_preferences": args.pocket_depth_preferences,
            "blend_to_candidate_index": args.blend_to_candidate_index,
            "blend_steps": args.blend_steps,
            "physical_pocket_valid": physical.cpu().tolist(),
            "initial_root_states": initial_root.cpu().tolist(),
            "source_run": str(args.run),
            "steps": args.steps,
            "synthetic_count": args.synthetic_count,
            "synthetic_offset": args.synthetic_offset,
            "synthetic_seed": args.synthetic_seed,
            "synthetic_topdown": args.synthetic_topdown,
            "synthetic_fk_conditioned": args.synthetic_fk,
            "synthetic_catalog": str(args.synthetic_catalog) if args.synthetic_catalog else None,
            "prewalk_steps": args.prewalk_steps,
            "restored_recorded_stance": args.restore_stance,
            "command_height_offsets_m": args.command_height_offsets_m,
            "raw_command_goals_world": command_goals.cpu().tolist(),
            **(
                {
                    "source_sample_ids": (
                        torch.arange(args.synthetic_count)
                        + (2000000 if args.synthetic_catalog else 1000000)
                        + args.synthetic_offset
                    ).tolist()
                }
                if args.synthetic_fk or args.synthetic_catalog
                else {}
            ),
            "gripper_angle_rad": args.gripper_angle_rad,
            "initial_arm_joints_rad": arms.cpu().tolist(),
            "ik_orientation_gain": float(env.cfg.goal_ee.teleop_ik_orientation_gain),
            "arm_ik_mode": "training_unconstrained" if args.unconstrained_arm_ik else "constrained",
            "dt": float(env.dt),
            "elapsed_s": time.perf_counter() - started,
            "stations": stations.cpu().tolist(),
            "goals_world": goals.cpu().tolist(),
            "goal_quaternions_wxyz": quats.cpu().tolist(),
            "final_position_error_m": history[-25:, :, -2].amax(0).cpu().tolist(),
            "final_orientation_error_rad": history[-25:, :, -1].amax(0).cpu().tolist(),
            "q6_span_rad": (history[:, :, 25].amax(0) - history[:, :, 25].amin(0)).cpu().tolist(),
            "final_root_states": history[-1, :, :13].cpu().tolist(),
            "contact_body_names": contact_names,
            "final_body_contact_peak_n": torch.stack(contact_history)[-25:].amax(0).cpu().tolist(),
            "final_nonfoot_contact_peak_n": torch.stack(contact_history)[-25:, :, nonfoot]
            .amax((0, 2))
            .cpu()
            .tolist(),
        }
        torch.save(history.cpu(), args.output / "trace.pt")
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[pick:policy-probe] {args.output / 'summary.json'}", flush=True)
    except BaseException:
        (args.output / "runtime_error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        if runner is not None:
            runner.close()
        app.close()


if __name__ == "__main__":
    main()
