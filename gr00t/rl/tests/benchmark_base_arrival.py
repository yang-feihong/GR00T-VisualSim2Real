"""Isolated station-controller experiment using production physics/policy/control.

Run inside doorman_train. This is a measurement harness, not a collection entry.
No production configuration is modified. Results include GPU-resident traces.
"""

import argparse
import itertools
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[3]
LOWLEVEL = REPO / "gr00t/rl/third_party/visual_wholebody_lowlevel"


def profiles(phase, position=None, yaw=None):
    if phase == "recovery":
        return [dict(inner_m=0.015, outer_m=0.08, inner_yaw=0.015, outer_yaw=0.10)] + [
            dict(inner_m=i, outer_m=o, inner_yaw=a, outer_yaw=b)
            for i, o in ((0.02, 0.10), (0.03, 0.12), (0.05, 0.15))
            for a, b in ((0.05, 0.15), (0.075, 0.15), (0.10, 0.20))
        ]
    if phase == "position":
        return [
            dict(inner_m=i, outer_m=o, inner_yaw=0.05, outer_yaw=0.10)
            for i, o in itertools.product((0.01, 0.02, 0.03, 0.05), (0.05, 0.08, 0.12, 0.15))
        ]
    if phase == "validation":
        a, b = yaw
        return (
            [dict(inner_m=0.05, outer_m=0.15, inner_yaw=0.05, outer_yaw=0.10)]
            + [
                dict(inner_m=i, outer_m=o, inner_yaw=a, outer_yaw=b)
                for i, o in itertools.product((0.005, 0.01, 0.015, 0.02), (0.08, 0.12))
            ]
            + [
                dict(inner_m=0.01, outer_m=0.15, inner_yaw=a, outer_yaw=b),
                dict(inner_m=0.0025, outer_m=0.12, inner_yaw=a, outer_yaw=b),
                dict(inner_m=0.015, outer_m=0.12, inner_yaw=0.005, outer_yaw=b),
            ]
        )
    i, o = position
    return [
        dict(inner_m=i, outer_m=o, inner_yaw=a, outer_yaw=b)
        for a, b in itertools.product((0.015, 0.025, 0.05), (0.05, 0.10, 0.15))
    ]


def scenarios(expanded=False, recovery=False):
    if recovery:
        return [
            dict(
                arm=arm,
                bearing_rad=bearing * math.pi / 2,
                distance_m=distance,
                turn_rad=turn,
                initial_yaw_rad=0.0,
                controller=controller,
            )
            for arm, controller, bearing, distance, turn in itertools.product(
                ("middle", "restored"),
                ("track", "hold"),
                range(4),
                (0.03, 0.07),
                (-0.10, -0.06, 0.06, 0.10),
            )
        ]
    result = []
    for arm in ("middle", "restored"):
        distances = (0.05, 0.10, 0.30, 1.0) if expanded else (0.30, 1.0)
        for bearing, distance, turn in itertools.product(range(8), distances, (0.0, 0.8)):
            result.append(
                dict(
                    arm=arm,
                    bearing_rad=bearing * math.pi / 4,
                    distance_m=distance,
                    turn_rad=turn * (1 if bearing % 2 else -1),
                    initial_yaw_rad=(bearing % 4) * math.pi / 2,
                )
            )
        turns = (
            (-2.5, -math.pi / 2, -1.2, -0.4, 0.4, 1.2, math.pi / 2, 2.5)
            if expanded
            else (-1.2, -0.4, 0.4, 1.2)
        )
        for turn in turns:
            result.append(
                dict(arm=arm, bearing_rad=0.0, distance_m=0.0, turn_rad=turn, initial_yaw_rad=0.0)
            )
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--phase", choices=("position", "yaw", "validation", "recovery"), default="position"
    )
    p.add_argument("--position", type=float, nargs=2, default=(0.01, 0.08))
    p.add_argument("--yaw", type=float, nargs=2, default=(0.025, 0.10))
    p.add_argument("--expanded-cases", action="store_true")
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=150)
    p.add_argument("--hold-steps", type=int, default=100)
    p.add_argument("--smoke", action="store_true")
    return p


def summarize(records):
    import numpy as np

    output = []
    for profile in sorted({r["profile_index"] for r in records}):
        rows = [r for r in records if r["profile_index"] == profile]
        successful = [r for r in rows if r["success"]]
        first = [r for r in rows if r["one_pass"]]

        def percentile(key, q, subset=successful):
            return float(np.quantile([r[key] for r in subset], q)) if subset else None

        output.append(
            dict(
                profile_index=profile,
                profile=rows[0]["profile"],
                trials=len(rows),
                success_count=len(successful),
                one_pass_count=len(first),
                success_rate=len(successful) / len(rows),
                one_pass_rate=len(first) / len(rows),
                within_3cm_2deg_count=sum(
                    r["one_pass"]
                    and r["final_distance_m"] <= 0.03
                    and r["final_yaw_rad"] <= math.radians(2)
                    for r in rows
                ),
                within_5cm_3deg_count=sum(
                    r["one_pass"]
                    and r["final_distance_m"] <= 0.05
                    and r["final_yaw_rad"] <= math.radians(3)
                    for r in rows
                ),
                distance_median_m=percentile("final_distance_m", 0.5),
                distance_p90_m=percentile("final_distance_m", 0.9),
                yaw_median_deg=None
                if not successful
                else math.degrees(percentile("final_yaw_rad", 0.5)),
                yaw_p90_deg=None
                if not successful
                else math.degrees(percentile("final_yaw_rad", 0.9)),
                elapsed_median_s=percentile("elapsed_s", 0.5),
                elapsed_p90_s=percentile("elapsed_s", 0.9),
                mean_restarts=float(np.mean([r["restarts"] for r in rows])),
                failures={
                    reason: sum(r["failure"] == reason for r in rows)
                    for reason in ("timeout", "reset", "unstable", "nonfoot_contact")
                },
            )
        )
    return output


def supervise(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "benchmark_source.py").write_bytes(Path(__file__).read_bytes())
    production_sources = [
        REPO / "gr00t/rl/auto_open/tensor_fsm.py",
        REPO / "gr00t/rl/auto_open/control.py",
    ]
    (args.output / "config.json").write_text(
        json.dumps(
            dict(
                arguments=vars(args),
                production_sha256={
                    str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in production_sources
                },
            ),
            default=str,
            indent=2,
        )
        + "\n"
    )
    children, handles = [], []
    start = time.perf_counter()

    def cleanup(signum=None, frame=None):
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if signum is not None:
            raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cleanup)
    try:
        for rank, gpu in enumerate(args.gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            handle = open(args.output / f"rank{rank}.log", "w")
            handles.append(handle)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(args.output),
                "--rank",
                str(rank),
                "--world-size",
                str(len(args.gpus)),
                "--phase",
                args.phase,
                "--position",
                *map(str, args.position),
                "--max-steps",
                str(args.max_steps),
                "--yaw",
                *map(str, args.yaw),
                "--warmup-steps",
                str(args.warmup_steps),
                "--hold-steps",
                str(args.hold_steps),
            ]
            if args.smoke:
                command.append("--smoke")
            if args.expanded_cases:
                command.append("--expanded-cases")
            children.append(
                subprocess.Popen(
                    command,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        while any(c.poll() is None for c in children):
            failed = [i for i, c in enumerate(children) if c.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"benchmark ranks failed: {failed}")
            print(
                f"[base-benchmark] elapsed_s={time.perf_counter() - start:.1f} "
                f"workers={[c.poll() for c in children]}",
                flush=True,
            )
            time.sleep(10)
        if any(c.returncode for c in children):
            raise RuntimeError("benchmark worker failure")
        records = []
        for rank in range(len(args.gpus)):
            records.extend(json.loads((args.output / f"rank{rank}.json").read_text())["trials"])
        result = dict(
            phase=args.phase,
            elapsed_s=time.perf_counter() - start,
            scenarios=scenarios(args.expanded_cases, args.phase == "recovery"),
            definitions=dict(
                one_pass="Success with one zero-command braking window and no resumption",
                success="Production arrival/stability plus 2 seconds of continuous confirmation",
                independent_accuracy="Counts use all trials as denominator, not only successful trials",
            ),
            profiles=summarize(records),
            trials=records,
        )
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["profiles"], indent=2), flush=True)
    finally:
        cleanup()
        for handle in handles:
            handle.close()


def worker(args):
    import torch

    sys.path.insert(0, str(LOWLEVEL))
    from gr00t.rl.auto_open.cli import _parser, _default_paths
    from legged_gym.utils.isaaclab_app import launch_app
    from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy
    from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM
    from gr00t.rl.auto_open.execution import TensorCommandBuffers
    from gr00t.rl.auto_open.motion_math import _yaw_from_wxyz

    all_profiles = profiles(args.phase, args.position, args.yaw)
    cases = scenarios(args.expanded_cases, args.phase == "recovery")
    if args.smoke:
        all_profiles = [all_profiles[0], all_profiles[-1]]
        cases = [cases[i] for i in (0, 3, 8, 15, 36, 39, 44, 51)]
    combinations = [
        (i, j)
        for i in range(len(all_profiles))
        if i % args.world_size == args.rank
        for j in range(len(cases))
    ]
    if not combinations:
        (args.output / f"rank{args.rank}.json").write_text(json.dumps(dict(trials=[])))
        return
    n = len(combinations)
    options = _parser().parse_args(
        [
            "--output_dir",
            str(args.output / f"rank{args.rank}"),
            "--headless",
            "--device",
            "cuda:0",
            "--disable_rgb_cameras",
            "--summary_only",
            "--num_envs",
            str(n),
            "--env_spacing",
            "8",
            "--seed",
            str(1700 + args.rank * 10000),
            "--doorman_randomize_robot_init",
            "false",
        ]
    )
    options.episode_seeds = list(range(1700 + args.rank * 10000, 1700 + args.rank * 10000 + n))
    options.output_dir.mkdir(parents=True, exist_ok=True)
    _default_paths(options)
    app = launch_app(options)
    runner = ManipLoco_Policy(options)
    env, device = runner.env, runner.env.device
    torch.set_grad_enabled(False)
    ids = torch.arange(n, device=device)
    root = env.robot.data.default_root_state.clone()
    root[:, :3] += env.scene.env_origins
    root[:, 0] += 4.0
    root[:, 7:] = 0.0
    initial_yaw = torch.tensor(
        [cases[j]["initial_yaw_rad"] for _, j in combinations], device=device
    )
    root[:, 3:7] = 0.0
    root[:, 3], root[:, 6] = torch.cos(initial_yaw / 2), torch.sin(initial_yaw / 2)
    joints = env.default_dof_pos.clone()
    middle = [-1.57, 1.75, -0.785, -0.93, 0.0, 0.0]
    # Actual held posture before the first sidestep, seed1700 / 20261022T172826.
    restored = [-1.46238208, 1.31942749, -0.48900816, -0.93008339, -0.12042879, 0.03562522]
    arms = torch.tensor(
        [middle if cases[j]["arm"] == "middle" else restored for _, j in combinations],
        device=device,
    )
    joints[:, env.arm_joint_ids] = arms
    joints[:, env.gripper_joint_ids] = 0.0
    env.robot.write_root_pose_to_sim(root[:, :7])
    env.robot.write_root_velocity_to_sim(root[:, 7:])
    env.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
    env.robot.set_joint_position_target(joints)
    runner.obs = env.get_observations()
    buffers = TensorCommandBuffers.from_env(env)
    buffers.arm_joint_mode[:] = True
    buffers.arm_joint[:] = arms
    buffers.base[:] = 0.0
    buffers.gripper[:] = 0.0

    def step():
        buffers.flush(env)
        env._clip_teleop_targets()
        env._update_effective_teleop_inputs(force=True)
        return runner.step_once()["state"]

    for _ in range(args.warmup_steps):
        state = step()
    start_root = state["root_states"].clone()
    start_yaw = _yaw_from_wxyz(start_root[:, 3:7])
    bearing = (
        torch.tensor([cases[j]["bearing_rad"] for _, j in combinations], device=device) + start_yaw
    )
    distance = torch.tensor([cases[j]["distance_m"] for _, j in combinations], device=device)
    target = start_root[:, :2] + distance[:, None] * torch.stack((bearing.cos(), bearing.sin()), -1)
    yaw = start_yaw + torch.tensor([cases[j]["turn_rad"] for _, j in combinations], device=device)
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device, fsm.num_envs = device, n
    for name in ("base_arrival_confirmed", "base_stability_confirmed"):
        setattr(fsm, name, torch.zeros(n, dtype=torch.bool, device=device))
    for name in (
        "base_arrival_settle_steps",
        "base_arrival_max_settle_steps",
        "base_stability_steps",
    ):
        setattr(fsm, name, torch.zeros(n, dtype=torch.int32, device=device))
    for name, key in [
        ("base_arrival_inner_distance", "inner_m"),
        ("base_arrival_outer_distance", "outer_m"),
        ("base_arrival_inner_yaw", "inner_yaw"),
        ("base_arrival_outer_yaw", "outer_yaw"),
    ]:
        setattr(
            fsm, name, torch.tensor([all_profiles[i][key] for i, _ in combinations], device=device)
        )
    fsm.base_arrival_required_frames = torch.full((n,), 25, dtype=torch.int32, device=device)
    fsm.base_arrival_max_linear_speed = torch.full((n,), 0.02, device=device)
    fsm.base_arrival_max_angular_speed = torch.full((n,), 0.05, device=device)
    fsm.through_min_base_height = torch.full(
        (n,), float(env.cfg.rewards.base_height_target_min), device=device
    )
    fsm.base_command_previous = torch.zeros(n, 3, device=device)
    fsm.base_reversal_pending_sign = torch.zeros(n, 3, dtype=torch.int8, device=device)
    fsm.base_reversal_pending_steps = torch.zeros(n, 3, dtype=torch.int32, device=device)
    fsm.base_reversal_lock_steps = torch.zeros(n, 3, dtype=torch.int32, device=device)
    fsm.base_hold_target_xy = fsm.base_hold_nominal_xy = target
    fsm.base_hold_target_yaw = fsm.base_hold_nominal_yaw = yaw
    fsm.pending_station_distance = torch.zeros(n, device=device)
    fsm.pending_yaw_error = torch.zeros(n, device=device)
    hold_controller = torch.tensor(
        [cases[j].get("controller") == "hold" for _, j in combinations], device=device
    )
    success = torch.zeros(n, dtype=torch.bool, device=device)
    failure = torch.zeros(n, dtype=torch.int32, device=device)
    finished = torch.zeros_like(success)
    previous_zero = torch.zeros_like(success)
    stops = torch.zeros(n, dtype=torch.int32, device=device)
    restarts = torch.zeros_like(stops)
    hold = torch.zeros_like(stops)
    unsafe_frames = torch.zeros_like(stops)
    contact_frames = torch.zeros_like(stops)
    terminal = torch.zeros(n, 3, device=device)
    terminal_step = torch.full((n,), args.max_steps, dtype=torch.int32, device=device)
    min_height = start_root[:, 2].clone()
    sensor_names = list(env.contact_sensor.body_names)
    if not set(env.cfg.foot_body_names).issubset(sensor_names):
        raise RuntimeError("Contact sensor does not identify all four feet")
    nonfoot = torch.tensor(
        [i for i, name in enumerate(sensor_names) if name not in env.cfg.foot_body_names],
        device=device,
    )
    trace = torch.zeros(args.max_steps, n, 27, device=device)
    started = time.perf_counter()
    for index in range(args.max_steps):
        active = ~finished
        rows = ids[active]
        buffers.base[:] = 0.0
        track_rows, hold_rows = rows[~hold_controller[rows]], rows[hold_controller[rows]]
        if len(track_rows):
            buffers.base[track_rows] = fsm._track_station(
                state,
                track_rows,
                target[track_rows],
                yaw[track_rows],
                target[track_rows],
                yaw[track_rows],
            )[0]
        if len(hold_rows):
            buffers.base[hold_rows] = fsm._base_hold_command(state, hold_rows)[0]
        zero = fsm.base_arrival_settle_steps > 0
        stops += (active & zero & ~previous_zero).int()
        restarts += (active & previous_zero & ~zero).int()
        previous_zero = zero.clone()
        state = step()
        r = state["root_states"]
        error = torch.linalg.vector_norm(target - r[:, :2], dim=-1)
        delta = yaw - _yaw_from_wxyz(r[:, 3:7])
        angle = torch.atan2(delta.sin(), delta.cos()).abs()
        speed = torch.linalg.vector_norm(r[:, 7:10], dim=-1)
        angular = torch.linalg.vector_norm(r[:, 10:13], dim=-1)
        upright = 1.0 - 2.0 * (r[:, 4].square() + r[:, 5].square())
        stable = (
            (speed < 0.02)
            & (angular < 0.05)
            & (r[:, 2] >= fsm.through_min_base_height)
            & (upright >= math.cos(math.radians(10)))
        )
        inside = (error < fsm.base_arrival_outer_distance) & (angle < fsm.base_arrival_outer_yaw)
        confirmed = (
            fsm.base_arrival_confirmed & fsm.base_stability_confirmed & stable & inside & zero
        )
        hold = torch.where(confirmed & active, hold + 1, 0)
        unsafe = (r[:, 2] < fsm.through_min_base_height) | (upright < math.cos(math.radians(10)))
        unsafe_frames = torch.where(unsafe & active, unsafe_frames + 1, 0)
        forces, force_body = torch.linalg.vector_norm(
            env.contact_sensor.data.net_forces_w[:, nonfoot], dim=-1
        ).max(-1)
        contact_frames = torch.where((forces > 5.0) & active, contact_frames + 1, 0)
        reset = env.reset_terminated | env.reset_time_outs
        failure[active & reset] = 2
        failure[active & ~reset & (unsafe_frames >= 10)] = 3
        failure[active & ~reset & (unsafe_frames < 10) & (contact_frames >= 3)] = 4
        completed = active & (hold >= args.hold_steps) & (failure == 0)
        success[completed] = True
        newly_finished = active & (completed | (failure != 0))
        terminal[newly_finished] = torch.stack((error, angle, speed), -1)[newly_finished]
        terminal_step[newly_finished] = index + 1
        finished |= newly_finished
        min_height = torch.where(active, torch.minimum(min_height, r[:, 2]), min_height)
        trace[index] = torch.cat(
            (
                r[:, :3] - env.scene.env_origins,
                r[:, 3:13],
                buffers.base,
                state["base_command"],
                torch.stack(
                    (
                        error,
                        angle,
                        zero.float(),
                        confirmed.float(),
                        forces,
                        active.float(),
                        upright,
                        nonfoot[force_body].float(),
                    ),
                    -1,
                ),
            ),
            -1,
        )
        if index % 100 == 0:
            print(
                f"[base-benchmark] rank={args.rank} step={index} wall_s={time.perf_counter() - started:.1f} "
                f"active={int((~finished).sum())} success={int(success.sum())}",
                flush=True,
            )
            if bool(finished.all()):
                break
    remaining = ~finished
    failure[remaining] = 1
    terminal[remaining] = torch.stack((error, angle, speed), -1)[remaining]
    data = (
        torch.cat(
            (
                success[:, None],
                failure[:, None],
                stops[:, None],
                restarts[:, None],
                terminal,
                terminal_step[:, None],
                min_height[:, None],
            ),
            -1,
        )
        .cpu()
        .tolist()
    )
    records = []
    for k, (profile, scenario) in enumerate(combinations):
        a = data[k]
        records.append(
            dict(
                profile_index=profile,
                profile=all_profiles[profile],
                scenario_index=scenario,
                scenario=cases[scenario],
                success=bool(a[0]),
                failure={0: None, 1: "timeout", 2: "reset", 3: "unstable", 4: "nonfoot_contact"}[
                    int(a[1])
                ],
                stops=int(a[2]),
                restarts=int(a[3]),
                one_pass=bool(a[0]) and a[2] == 1 and a[3] == 0,
                final_distance_m=a[4],
                final_yaw_rad=a[5],
                final_speed_m_s=a[6],
                elapsed_s=a[7] * env.dt,
                min_base_height_m=a[8],
            )
        )
    torch.save(
        dict(
            trace=trace[: index + 1].cpu(),
            target_xy=target.cpu(),
            target_yaw=yaw.cpu(),
            start_root=start_root.cpu(),
            env_origins=env.scene.env_origins.cpu(),
            dt=env.dt,
            contact_body_names=sensor_names,
            columns=[
                "x",
                "y",
                "z",
                "qw",
                "qx",
                "qy",
                "qz",
                "vx",
                "vy",
                "vz",
                "wx",
                "wy",
                "wz",
                "cmd_vx",
                "cmd_vy",
                "cmd_wz",
                "effective_vx",
                "effective_vy",
                "effective_wz",
                "distance",
                "yaw_error",
                "zero_window",
                "confirmed",
                "max_nonfoot_force",
                "active",
                "upright",
                "peak_contact_body",
            ],
        ),
        args.output / f"rank{args.rank}_trace.pt",
    )
    (args.output / f"rank{args.rank}.json").write_text(
        json.dumps(
            dict(
                trials=records,
                elapsed_s=time.perf_counter() - started,
                checkpoint=options.ckpt_path,
                solver=[
                    env.cfg.robot.spawn.articulation_props.solver_position_iteration_count,
                    env.cfg.robot.spawn.articulation_props.solver_velocity_iteration_count,
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"[base-benchmark] rank={args.rank} done {int(success.sum())}/{n}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.rank is None:
        supervise(arguments)
    else:
        try:
            worker(arguments)
        except BaseException:
            import traceback

            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
