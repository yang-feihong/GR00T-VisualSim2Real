"""Seed-scoped scene metadata to GPU reset tensors for the shared batch runtime."""

import math

import torch


def basket_payload_reset_tensors(descriptor_rows, roots, origins):
    """Place new payload layouts in each measured basket frame, without moving its robot."""
    from gr00t.rl.auto_open.motion_math import _quat_multiply
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate

    result = []
    for descriptors in descriptor_rows:
        active = torch.tensor(
            ["basket_local_position" in item for item in descriptors], device=roots.device
        )
        local = roots.new_tensor(
            [item.get("basket_local_position", [0, 0, 0]) for item in descriptors]
        )
        yaw = roots.new_tensor([item.get("basket_local_yaw", 0.0) for item in descriptors]) / 2
        orientation = torch.stack((yaw.cos(), yaw * 0, yaw * 0, yaw.sin()), -1)
        values = roots.new_zeros(len(roots), 13)
        values[:, :3] = roots[:, :3] + _quat_rotate(roots[:, 3:7], local)
        values[:, 3:7] = _quat_multiply(roots[:, 3:7], orientation)
        parked = roots.new_tensor([item["object_position"] for item in descriptors]) + origins
        values[:, :3] = torch.where(active[:, None], values[:, :3], parked)
        values[:, 3:7] = torch.where(active[:, None], values[:, 3:7], orientation)
        result.append(values)
    return result


def batch_reset_tensors(episodes, origins, default_joints, arm_ids, gripper_ids, distractors=()):
    if len(episodes) != len(origins) or len(default_joints) != len(episodes):
        raise ValueError("Each environment requires its own scene/reset descriptor")
    root = origins.new_zeros(len(episodes), 13)
    root[:, :2] = root.new_tensor([item["base_xy"] for item in episodes])
    root[:, 2] = 0.55
    root[:, :3] += origins
    yaw = root.new_tensor([item["base_yaw"] for item in episodes]) / 2
    root[:, 3], root[:, 6] = yaw.cos(), yaw.sin()
    joints = default_joints.clone()
    joints[:, arm_ids] = joints.new_tensor([item["initial_arm_joints"] for item in episodes])
    joints[:, gripper_ids] = -math.pi / 3
    objects = []
    for descriptors in (episodes, *distractors):
        values = root.new_zeros(len(episodes), 13)
        values[:, :3] = (
            values.new_tensor([item["object_position"] for item in descriptors]) + origins
        )
        yaw = values.new_tensor([item["object_yaw"] for item in descriptors]) / 2
        values[:, 3], values[:, 6] = yaw.cos(), yaw.sin()
        objects.append(values)
    return root, joints, objects


def create_pick_environment(args, episodes):
    from legged_gym.utils.helpers import build_env_cfg

    from gr00t.rl.pick.environment import PickEnvironment

    args.num_envs = len(episodes)
    cfg = build_env_cfg(args, for_play=True)
    cfg.episode_length_s = 10000.0
    cfg.sim.create_stage_in_memory = True
    cfg.scene.replicate_physics = False
    cfg.enable_contact_sensor = True
    cfg.check_terminations = False
    cfg.seed = int(episodes[0]["seed"])
    cfg.doorman_episode_seeds = tuple(int(item["seed"]) for item in episodes)
    cfg.scene_usd_path = ""
    cfg.env.teleop_ee_goal_z_limit = [-0.75, 0.6]
    cfg.robot.spawn.rigid_props.solver_position_iteration_count = 16
    cfg.robot.spawn.rigid_props.solver_velocity_iteration_count = 2
    cfg.robot.spawn.activate_contact_sensors = True
    return PickEnvironment(cfg, episodes), cfg


def load_probe_episodes(paths):
    import json

    from gr00t.rl.pick.prepare_batch import local_episode_descriptor

    episodes = []
    for path in paths:
        raw = json.loads(path.read_text())
        records = raw if isinstance(raw, list) else [raw]
        episodes.extend(local_episode_descriptor(item.get("episode", item)) for item in records)
    if not episodes or len({item["seed"] for item in episodes}) != len(episodes):
        raise ValueError("Batch scene probe requires nonempty distinct seeds")
    return episodes


def stability_profile_metrics(ready):
    """GPU-only timing facts; pose stability is not full grasp admission."""
    if ready.ndim != 2 or not all(ready.shape) or ready.dtype != torch.bool:
        raise ValueError("Stability history must be a nonempty [steps, rows] boolean tensor")

    def first(mask):
        return torch.where(mask.any(), mask.long().argmax() + 1, -1)

    ever = ready.any(0)
    first_by_row = torch.where(ever, ready.long().argmax(0) + 1, -1)
    counts = ready.long().sum(-1)
    polled = (torch.arange(len(ready), device=ready.device) + 1) % 8 == 0
    return {
        "pose_stable_first_step": first_by_row,
        "pose_stable_frames_total": ready.long().sum(0),
        "pose_stable_rows_by_step": counts,
        "rows_ever_pose_stable": ever.long().sum(),
        "first_all_pose_stable_step": first(ready.all(-1)),
        "first_poll_all_pose_stable_step": first(ready.all(-1) & polled),
        "first_poll_80pct_pose_stable_step": first(
            (counts >= (4 * ready.shape[1] + 4) // 5) & polled
        ),
    }


def main():
    """Initialize several real scenes together; no claim of grasp acceptance."""
    import json
    from pathlib import Path

    from legged_gym.utils.isaaclab_app import launch_app

    from gr00t.rl.auto_open.cli import _default_paths, _parser

    parser = _parser()
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[])
    parser.add_argument("--reset-rows", type=int, nargs="+", default=[])
    parser.add_argument("--preserve-robot", action="store_true")
    parser.add_argument("--profile-stability", action="store_true")
    args = parser.parse_args()
    episodes = load_probe_episodes(args.episodes)
    if args.probe_seeds:
        by_seed = {episode["seed"]: episode for episode in episodes}
        if (
            len(set(args.probe_seeds)) != len(args.probe_seeds)
            or not set(args.probe_seeds) <= by_seed.keys()
        ):
            parser.error("Probe seeds must be unique members of the supplied episode descriptors")
        episodes = [by_seed[seed] for seed in args.probe_seeds]
    if args.profile_stability and (args.reset_rows or not 1 <= args.max_steps <= 300):
        parser.error("Stability profiling requires 1..300 steps without a reset probe")
    if args.reset_rows and (
        len(set(args.reset_rows)) != len(args.reset_rows)
        or min(args.reset_rows) < 0
        or max(args.reset_rows) >= len(episodes)
        or len(args.reset_rows) >= len(episodes)
        or args.max_steps < 2
    ):
        raise ValueError("Reset probe requires distinct valid rows and an unchanged partner")
    args.num_envs = len(episodes)
    args.seed = episodes[0]["seed"]
    args.enable_doorman_scene = False
    args.disable_rgb_cameras = True
    _default_paths(args)
    args.env_spacing = 32.0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    app = launch_app(args)
    runner = None
    try:
        from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy, load_policy
        from legged_gym.utils.run_metadata import checkpoint_model_path

        from gr00t.rl.auto_open.execution import TensorCommandBuffers
        from gr00t.rl.task_runtime import BatchedTaskLoop, reset_task_rows

        class PickSceneProbe(ManipLoco_Policy):
            def init_env(self):
                self.env, self.env_cfg = create_pick_environment(self.args, episodes)
                self.obs = self.env.get_observations()
                self.policy = load_policy(str(checkpoint_model_path(self.args)), self.env.device)

        runner = PickSceneProbe(args)
        env = runner.env
        if args.profile_stability:
            reset_task_rows(runner, torch.arange(env.num_envs, device=env.device))
        else:
            env.reset()
            runner.obs = env.get_observations()
        initial = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in env.get_control_state().items()
        }
        if getattr(args, "print_timing_breakdown", False):
            env.pop_step_profile_stats()
        stage = torch.ones(len(episodes), device=env.device, dtype=torch.long)
        loop = BatchedTaskLoop(
            runner,
            [item["seed"] for item in episodes],
            max_steps=args.max_steps,
            success_stage=2,
            failure_stage=3,
        )
        buffers = TensorCommandBuffers.from_env(env)
        buffers.base.zero_()
        buffers.arm_joint_mode.fill_(True)
        buffers.arm_joint.copy_(initial["dof_pos"][:, env.arm_joint_ids])
        if args.profile_stability:
            from gr00t.rl.auto_open.acceptance import TensorStabilityWindow
            from gr00t.rl.pick.task_initialization import update_pick_pose_stability

            buffers.arm_joint.copy_(
                initial["dof_pos"].new_tensor([e["initial_arm_joints"] for e in episodes])
            )
            base_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
            object_steps, other_steps = torch.zeros_like(base_steps), torch.zeros_like(base_steps)
            slots = initial["distractor_states"].shape[1]
            mask = (
                torch.arange(slots, device=env.device)[None]
                < torch.tensor(
                    [len(e.get("distractors", [])) for e in episodes], device=env.device
                )[:, None]
            )
            window = (
                TensorStabilityWindow(env.num_envs * (slots + 1), env.device)
                if episodes[0].get("task_type") == "pick_basket"
                else None
            )
            ready_history = torch.zeros(
                args.max_steps, env.num_envs, device=env.device, dtype=torch.bool
            )
            physical_history = {
                key: value.new_empty((args.max_steps, *value.shape))
                for key, value in initial.items()
                if key in ("root_states", "object_state", "distractor_states", "dof_pos")
            }
        buffers.gripper.fill_(-math.pi / 3)
        state = initial
        reset_report = None
        for step, _ in loop.ticks(stage):
            loop.publish(buffers)
            state = loop.step()["state"]
            if args.profile_stability:
                for key in (
                    "dof_vel",
                    "contact_forces_w",
                    "lowlevel_policy_observation",
                    "lowlevel_policy_action",
                    "physics_substep_dof_pos",
                    "physics_substep_dof_vel",
                    "physics_substep_torque",
                    "physics_substep_position_target",
                ):
                    if key in state and key not in physical_history:
                        value = state[key]
                        physical_history[key] = value.new_empty((args.max_steps, *value.shape))
                stable, base_steps, object_steps, other_steps, _, _ = update_pick_pose_stability(
                    state, base_steps, object_steps, other_steps, mask, window
                )
                ready_history[step] = stable & (object_steps >= 15) & (other_steps >= 15)
                for key, values in physical_history.items():
                    values[step] = state[key]
            if args.reset_rows and step + 1 == args.max_steps // 2:
                rows = torch.tensor(args.reset_rows, device=env.device, dtype=torch.long)
                keep_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
                keep_mask[rows] = False
                keep = torch.where(keep_mask)[0]
                before = {
                    key: state[key].clone()
                    for key in (
                        "root_states",
                        "object_state",
                        "distractor_states",
                        "dof_pos",
                        "dof_vel",
                    )
                }
                history = env.obs_history_buf.clone()
                observation = runner.obs.clone()
                actions = env.action_history_buf.clone()
                expected_payloads = None
                if args.preserve_robot:
                    descriptors = [env.pick_episode] + [
                        [
                            item["distractors"][slot]
                            if slot < len(item.get("distractors", []))
                            else {"object_position": [0, 0, -100.0 - slot]}
                            for item in env.pick_episode
                        ]
                        for slot in range(len(env.pick_distractors))
                    ]
                    expected_payloads = basket_payload_reset_tensors(
                        descriptors, before["root_states"], env.scene.env_origins
                    )
                    reset_task_rows(runner, rows, preserve_robot=torch.ones_like(keep_mask))
                else:
                    reset_task_rows(runner, rows)
                after = env.get_control_state()
                partner_unchanged = all(
                    torch.equal(before[key][keep], after[key][keep]) for key in before
                )
                reset_pose_equal = all(
                    torch.allclose(initial[key][rows], after[key][rows], atol=2e-7, rtol=0)
                    for key in before
                )
                reset_report = {
                    "rows": args.reset_rows,
                    "partner_physical_state_bitwise_equal": partner_unchanged,
                    "partner_policy_history_bitwise_equal": torch.equal(
                        history[keep], env.obs_history_buf[keep]
                    ),
                    "partner_action_history_bitwise_equal": torch.equal(
                        actions[keep], env.action_history_buf[keep]
                    ),
                    "partner_policy_observation_bitwise_equal": torch.equal(
                        observation[keep], runner.obs[keep]
                    ),
                    **(
                        {"selected_initial_state_restored": reset_pose_equal}
                        if expected_payloads is None
                        else {
                            "all_robot_states_bitwise_equal": all(
                                torch.equal(before[key], after[key])
                                for key in ("root_states", "dof_pos", "dof_vel")
                            ),
                            "all_policy_inputs_bitwise_equal": torch.equal(observation, runner.obs),
                            "all_policy_history_bitwise_equal": torch.equal(
                                history, env.obs_history_buf
                            ),
                            "all_action_history_bitwise_equal": torch.equal(
                                actions, env.action_history_buf
                            ),
                            "selected_payload_target_restored": torch.allclose(
                                after["object_state"][rows],
                                expected_payloads[0][rows],
                                atol=2e-7,
                                rtol=0,
                            ),
                        }
                    ),
                }
                (args.output_dir / "partial_reset_probe.json").write_text(
                    json.dumps(reset_report, indent=2) + "\n"
                )
                torch.save(
                    {"before": before, "after": after, "rows": rows, "initial": initial},
                    args.output_dir / "partial_reset_state.pt",
                )
                assert all(value for key, value in reset_report.items() if key != "rows"), (
                    reset_report
                )
        stability_report = None
        if args.profile_stability:
            from gr00t.rl.auto_open.tensor_metadata import to_host

            stability_report = {
                "full_grasp_admission_evaluated": False,
                "predicate_source": "shared update_pick_pose_stability",
                "arm_command_source": "episode initial_arm_joints; base zero; gripper -pi/3",
                "checkpoint": str(checkpoint_model_path(args)),
                "seeds": [e["seed"] for e in episodes],
                "physics_steps": loop.physics_steps,
                "policy_physics_s": loop.policy_physics_s,
                **to_host(stability_profile_metrics(ready_history[: loop.physics_steps])),
            }
            (args.output_dir / "stability_probe.json").write_text(
                json.dumps(stability_report, indent=2) + "\n"
            )
            torch.save(
                {
                    "initial": {
                        key: value for key, value in initial.items() if torch.is_tensor(value)
                    },
                    "ready": ready_history[: loop.physics_steps].cpu(),
                    **{
                        key: values[: loop.physics_steps].cpu()
                        for key, values in physical_history.items()
                    },
                },
                args.output_dir / "stability_history.pt",
            )
        report = {
            "probe": "shared_batch_scene_initialization_not_grasp_acceptance",
            "seeds": [item["seed"] for item in episodes],
            "scene_ids": [item["scene_id"] for item in episodes],
            "asset_names": [item["asset"]["name"] for item in episodes],
            "num_envs": env.num_envs,
            "physics_steps": loop.physics_steps,
            "policy_physics_s": loop.policy_physics_s,
            "command_publish_s": loop.publish_s,
            "environment_step_profile": env.pop_step_profile_stats()
            if getattr(args, "print_timing_breakdown", False)
            else None,
            "env_origins": env.scene.env_origins.cpu().tolist(),
            "initial_root_states": initial["root_states"].cpu().tolist(),
            "initial_object_states": initial["object_state"].cpu().tolist(),
            "final_root_states": state["root_states"].cpu().tolist(),
            "final_object_states": state["object_state"].cpu().tolist(),
            "finite": bool(
                torch.isfinite(state["root_states"]).all()
                & torch.isfinite(state["object_state"]).all()
            ),
            "partial_reset": reset_report,
            "stability_profile": stability_report,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if runner is not None:
            runner.close()
        if not args.headless:
            app.close()


if __name__ == "__main__":
    main()
