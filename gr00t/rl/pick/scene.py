"""VLNVerse scene setup and recording for the common task entry and GPU ranks."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")


def _planning_failure_reason(controller, stage):
    detail = controller.planning_diagnostics.get(stage.lower(), {})
    return detail.get("failure") or controller.reason or "stage_entry_planning_failed"


def prepare_shared_rank(args):
    """Prepare scene assets before Kit loads USD plugins, once per GPU rank."""
    args.pick_initialization_started = time.monotonic()
    args.pick_initialization_timing = {}
    seeds = args.episode_seeds or [args.seed]
    manifest = args.output_dir / "prepared_scenes.json"
    command = [
        sys.executable,
        "-m",
        "gr00t.rl.pick.prepare_batch",
        "--seeds",
        *map(str, seeds),
        "--task",
        args.task_kind.removeprefix("pick_"),
        "--output",
        str(manifest),
        "--scenes-root",
        str(args.pick_scenes_root),
        "--support",
        args.pick_support,
    ]
    for flag, value in (
        ("--episodes-manifest", args.pick_episode_manifest),
        ("--scene-id", args.pick_scene_id),
        ("--asset", args.pick_asset),
    ):
        if value is not None:
            command += [flag, str(value)]
    if args.replay_source_dataset_dir is not None:
        supplied = {}
        for seed in seeds:
            matches = list(
                args.replay_source_dataset_dir.glob(f"logs/rank*-control/seed_{seed}/episode.json")
            )
            if len(matches) != 1:
                raise ValueError(f"No unique canonical scene descriptor for seed {seed}")
            supplied[str(seed)] = str(matches[0])
        source_manifest = args.output_dir / "replay_scenes.json"
        _write(source_manifest, supplied)
        command += ["--episodes-manifest", str(source_manifest)]
    usd_libraries = next(Path("/workspace/isaaclab/_isaac_sim/extscache").glob("omni.usd.libs-*"))
    environment = dict(os.environ)
    environment["AUTO_OPEN_MANAGED_RANK_PARENT_PID"] = str(os.getpid())
    environment["PYTHONPATH"] = f"{usd_libraries}:{environment.get('PYTHONPATH', '')}"
    environment["LD_LIBRARY_PATH"] = f"{usd_libraries}/bin:{environment.get('LD_LIBRARY_PATH', '')}"
    subprocess.run(command, check=True, env=environment)
    args.prepared_scenes = json.loads(manifest.read_text())
    args.pick_initialization_timing["preparation_s"] = (
        time.monotonic() - args.pick_initialization_started
    )
    from gr00t.rl.pick.dataset import write_initialization_rejection

    for row in args.prepared_scenes:
        if "rejection" in row:
            reason = row["rejection"]["reason"]
            write_initialization_rejection(
                args.standard_dataset_dir / f"episode-seed{row['seed']}",
                seed=row["seed"],
                task_type=args.task_kind,
                reason=reason,
                details=row["rejection"]["details"],
            )
            _write(
                args.output_dir / f"seed_{row['seed']}/summary.json",
                {
                    "seed": row["seed"],
                    "success": False,
                    "failure": reason,
                    "failed_stage": "INITIALIZATION",
                    "task_type": args.task_kind,
                },
            )
    if not any("episode" in row for row in args.prepared_scenes):
        _write(
            args.output_dir / "summary.json",
            {
                "episode_seeds": seeds,
                "successes": 0,
                "num_envs": 0,
                "preparation_rejected": len(seeds),
            },
        )
        return False
    return True


def run_shared_rank(args):
    from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy, load_policy
    from legged_gym.scripts.rgb_camera_debug import SimRgbCameraManager
    from legged_gym.utils.run_metadata import checkpoint_model_path

    from gr00t.rl.auto_open.execution import TensorCommandBuffers
    from gr00t.rl.auto_open.runtime import _write_stage_progress
    from gr00t.rl.auto_open.tensor_metadata import to_host
    from gr00t.rl.data_collection.auto_open_standard import (
        camera_streams_for_mode,
        transform_collection_images,
    )
    from gr00t.rl.data_collection.auto_open_writer import StreamingVideoEncoder
    from gr00t.rl.pick.collection import (
        InitializationDebugRecorder,
        PickEpisodeCollector,
        create_pick_contact_audit,
    )
    from gr00t.rl.pick.controller import PickController
    from gr00t.rl.pick.dataset import write_initialization_rejection, write_pick_dataset
    from gr00t.rl.pick.distractors import target_semantics
    from gr00t.rl.pick.gaze import sample_approach_offsets_batch
    from gr00t.rl.pick.geometry_views import TranslatedGeometry
    from gr00t.rl.pick.grasp_quality import load_grasp_support
    from gr00t.rl.pick.scene_adapter import create_pick_environment
    from gr00t.rl.pick.scene_geometry import read_scene_triangles, static_scene_geometry
    from gr00t.rl.pick.task_initialization import admit_pick_batch
    from gr00t.rl.pick.tensor_fsm import FAILURES, TensorPickFSM
    from gr00t.rl.pick.tensor_planning import TensorPickPlanningCoordinator, episode_in_world
    from gr00t.rl.task_runtime import BatchedTaskLoop

    source_modules = ("scene", "grasp_quality", "station_batch", "measured_station_batch")
    sources = {name: sys.modules[f"gr00t.rl.pick.{name}"].__file__ for name in source_modules}
    _write(args.output_dir / "runtime_sources.json", sources)
    print(f"[pick:runtime] sources={json.dumps(sources)}", flush=True)
    episodes = [row["episode"] for row in args.prepared_scenes if "episode" in row]
    seeds = [item["seed"] for item in episodes]
    args.num_envs = len(episodes)
    args.env_spacing = 32.0
    if args.replay_source_dataset_dir is not None:
        if args.replay_render_batch_size < 1:
            raise ValueError("Replay render batch size must be positive")
        args.rgb_camera_sidecar_pool_size = min(args.replay_render_batch_size, len(episodes))

    class SceneRunner(ManipLoco_Policy):
        def init_env(self):
            self.env, self.env_cfg = create_pick_environment(self.args, episodes)
            self.env.reset()
            self.obs = self.env.get_observations()
            self.policy = load_policy(str(checkpoint_model_path(self.args)), self.env.device)
            self.env.pick_policy_checkpoint_sha256 = hashlib.sha256(
                checkpoint_model_path(self.args).read_bytes()
            ).hexdigest()
            self.rgb_camera_manager = SimRgbCameraManager(self.env, self.args.rgb_camera_config)
            if self.args.collection_camera_mode != "none":
                self.rgb_camera_manager.set_episode_camera_configs(
                    {row: self.args.episode_camera_configs[seed] for row, seed in enumerate(seeds)}
                )

        def _read_rgb_frames(self):
            # The shared collector captures synchronized images before publication.
            return {}

    runner = SceneRunner(args)
    args.pick_initialization_timing["physical_environment_ready_s"] = (
        time.monotonic() - args.pick_initialization_started
    )
    env = runner.env
    env.pick_geometry_backend = getattr(args, "pick_geometry_backend", "native")
    audit, encoders = None, {}
    initialization_recorder = None
    try:
        if args.replay_source_dataset_dir is not None:
            from gr00t.rl.pick.replay import render_shared_rank

            return render_shared_rank(args, runner, episodes)
        cached, geometries = {}, []
        for row, episode in enumerate(episodes):
            path = episode["scene_usd_path"]
            if path not in cached:
                points, faces, labels = read_scene_triangles(
                    Path(path), Path(episode["geometry_cache_root"]), return_labels=True
                )
                cached[path] = static_scene_geometry(
                    points,
                    faces,
                    env.device,
                    labels=labels,
                    backend=env.pick_geometry_backend,
                    ground_plane_z_m=0.0,
                )
            geometries.append(TranslatedGeometry(cached[path], env.scene.env_origins[row]))
        started = time.monotonic()
        args.pick_initialization_timing["geometry_ready_s"] = (
            started - args.pick_initialization_started
        )
        if args.debug_trace and args.collection_camera_mode == "six":
            initialization_recorder = InitializationDebugRecorder(
                runner, seeds, args.output_dir, args.task_kind
            )
        try:
            state, views, collision, checks, planning = admit_pick_batch(
                runner,
                episodes,
                geometries,
                args.output_dir,
                observer=(initialization_recorder.capture if initialization_recorder else None),
            )
        finally:
            if initialization_recorder is not None:
                initialization_recorder.close()
        args.pick_initialization_timing["admission_ready_s"] = (
            time.monotonic() - args.pick_initialization_started
        )
        model_path = args.output_dir / "collision_model.pt"
        torch.save(
            {
                "parts": [(index, points.cpu()) for index, points in collision.parts],
                "gripper_surface": [points.cpu() for points in collision.gripper_surface],
                "arm_joint_ids": collision.arm_joint_ids.cpu(),
                "gripper_joint_ids": env.gripper_joint_ids.cpu(),
            },
            model_path,
        )
        factory = PickController
        if args.task_kind == "pick_basket":
            from gr00t.rl.pick.basket_controller import BasketPickController

            factory = BasketPickController
        controllers = []
        origins_host = to_host(env.scene.env_origins)
        env.pick_approach_offsets = sample_approach_offsets_batch(seeds, env.device)
        for row, episode in enumerate(episodes):
            world = episode_in_world(episode, origins_host[row])
            world["target_semantics"] = target_semantics(world)
            _write(args.output_dir / f"seed_{episode['seed']}/episode.json", world)
            os.link(model_path, args.output_dir / f"seed_{episode['seed']}/collision_model.pt")
            support = getattr(planning[row], "object_support_points", None)
            if support is None:
                support = load_grasp_support(episode["asset"]["usd_path"], env.device)
            control = factory(
                views[row],
                world,
                planning[row],
                collision=collision,
                object_support_points=support,
                initialize=checks[row]["accepted"] and args.task_kind != "pick_basket",
            )
            if args.task_kind == "pick_basket" and checks[row]["accepted"]:
                control.current_stage = (
                    control.Stage.OPEN_FOR_REPOSITION
                    if world.get("basket_initial_reposition_required", False)
                    else control.Stage.MOVE_TO_GRASP
                )
                control.stage.fill_(int(control.current_stage))
            controllers.append(control)
        fsm = TensorPickFSM(env, controllers)
        fsm.fail(fsm.rows[~env.pick_initial_admitted], 10)
        planner = TensorPickPlanningCoordinator(controllers)
        if args.task_kind == "pick_basket":
            planner.install_initial_basket_plans(fsm, state)
        else:
            planner.install_initial_external_stations(fsm, state)
            args.pick_initialization_timing["stations_ready_s"] = (
                time.monotonic() - args.pick_initialization_started
            )
            planner.install_initial_external_gaze(fsm)
        args.pick_initialization_timing["planning_ready_s"] = (
            time.monotonic() - args.pick_initialization_started
        )
        _write(args.output_dir / "initialization_timing.json", args.pick_initialization_timing)
        collector = PickEpisodeCollector(env, debug=args.debug_trace)
        buffers = TensorCommandBuffers.from_env(env)
        loop = BatchedTaskLoop(
            runner,
            seeds,
            max_steps=args.max_steps,
            success_stage=fsm.Stage.SUCCESS,
            failure_stage=fsm.Stage.FAILURE,
        )
        layout = camera_streams_for_mode(args.collection_camera_mode, task_kind=args.task_kind)
        if args.debug_trace:
            audit = create_pick_contact_audit(env)
            audit.attach()
        for row, seed in enumerate(seeds):
            encoders[row] = {
                stream["key"]: StreamingVideoEncoder(
                    args.output_dir / f"seed_{seed}/.video_staging/{name}.mp4", stream["shape"]
                )
                for name, stream in layout.items()
            }
        _write_stage_progress(args.output_dir, fsm.Stage, fsm, 0, 0.0)
        traces = [[] for _ in episodes]
        transitions = [[] for _ in episodes]
        transition_frames = []
        timing = {
            name: 0.0
            for name in (
                "command_s",
                "teleop_prepare_s",
                "policy_physics_s",
                "advance_s",
                "planning_s",
            )
        }
        before, stages = state, fsm.stage.clone()

        def prepare_tick(state, active):
            if args.debug_trace:
                fsm.begin_acceptance_frame()

        def capture_before(step, active, state):
            nonlocal before, stages
            before, stages = state, fsm.stage.clone()
            if layout:
                rows = torch.where(active)[0].cpu().tolist()
                frames = runner.rgb_camera_manager.render_images_batch(rows, rig_names=list(layout))
                for row in rows:
                    rendered = frames[row]
                    images = {name: rendered[name][0] for name in layout}
                    images = transform_collection_images(
                        images, args.collection_camera_mode, task_kind=args.task_kind
                    )
                    for key, image in images.items():
                        encoders[row][key].append(image)
            if audit is not None:
                audit.begin_tick()

        def capture_after(step, active, result):
            recorded_after = result["state"]
            if args.debug_trace and fsm.command_compensation is not None:
                recorded_after = dict(recorded_after)
                recorded_after["ee_intended_goal_world"] = fsm.command_compensation.intended.clone()
                recorded_after["ee_command_height_offset_m"] = (
                    fsm.command_compensation.applied.clone()
                )
            collector.append(
                before,
                recorded_after,
                stages,
                joint_mode=env.teleop_arm_joint_control_mask,
                contacts={} if audit is None else audit.end_tick_by_env(),
            )

        def report_tick(step, stage_before, events):
            if step == 0:
                args.pick_initialization_timing["first_formal_step_s"] = (
                    time.monotonic() - args.pick_initialization_started
                )
                entry_started = os.environ.get("AUTO_OPEN_ENTRY_STARTED_NS")
                if entry_started is not None:
                    args.pick_initialization_timing["launcher_to_first_formal_step_s"] = (
                        time.time_ns() - int(entry_started)
                    ) / 1e9
                _write(
                    args.output_dir / "initialization_timing.json", args.pick_initialization_timing
                )
            if args.debug_trace:
                for row, values in enumerate(fsm.acceptance_frame_rows(step)):
                    traces[row].extend(values)
            transition_frames.append((step + 1, stage_before.clone(), fsm.stage.clone()))
            if step % 50 == 0:
                _write_stage_progress(
                    args.output_dir, fsm.Stage, fsm, step + 1, time.monotonic() - started
                )

        for _, state, _ in loop.run_stages(
            fsm,
            planner,
            state,
            buffers,
            timing=timing,
            prepare=prepare_tick,
            capture_before=capture_before,
            capture_after=capture_after,
            report=report_tick,
        ):
            pass
        fsm.fail(torch.where(fsm.active_mask())[0], 8)
        planner.finish_diagnostics()
        for step, before_stage, after_stage in to_host(transition_frames):
            for row, (before_value, after_value) in enumerate(
                zip(before_stage, after_stage, strict=True)
            ):
                if before_value != after_value:
                    transitions[row].append(
                        {
                            "step": step,
                            "from": fsm.Stage(before_value).name,
                            "to": fsm.Stage(after_value).name,
                        }
                    )
        for group in encoders.values():
            for encoder in group.values():
                encoder.close()
        final_host = to_host(
            {
                "failure_code": fsm.failure_code,
                "failure_stage": fsm.failure_from_stage,
                "stage": fsm.stage,
                "steps": loop.step_counts,
                "command_compensation_peak_m": (
                    fsm.command_compensation.peak
                    if fsm.command_compensation is not None
                    else torch.zeros_like(fsm.hold_yaw)
                ),
            }
        )
        if collector.states:
            collector.prepare_finish()
        acceptance_reports = fsm.acceptance_diagnostics_batch(fsm.Stage)
        for row, controller in enumerate(controllers):
            seed, output = seeds[row], args.output_dir / f"seed_{seeds[row]}"
            code = final_host["failure_code"][row]
            failed_stage = fsm.Stage(final_host["failure_stage"][row]).name if code else None
            success = final_host["stage"][row] == int(fsm.Stage.SUCCESS)
            reason = (
                _planning_failure_reason(controller, failed_stage)
                if code == 9
                else FAILURES.get(code)
            )
            if code == 10:
                reason, failed_stage = "initial_state_admission_failed", "INITIALIZATION"
            summary = {
                "seed": seed,
                "task_type": args.task_kind,
                "scene_id": controller.episode["scene_id"],
                "asset": controller.episode["asset"]["name"],
                "success": success,
                "failure": reason,
                "failed_stage": failed_stage,
                "initial_accepted": checks[row]["accepted"],
                "mode": "debug" if args.debug_trace else "canonical",
                "camera_mode": args.collection_camera_mode,
                "elapsed_s": time.monotonic() - started,
                "planning": controller.planning_diagnostics,
                "transitions": transitions[row],
                "acceptance_diagnostics": acceptance_reports[row],
                "target_semantics": controller.episode["target_semantics"],
                "command_compensation": {
                    "enabled": fsm.command_compensation is not None,
                    "maximum_downward_offset_m": 0.02,
                    "peak_applied_abs_m": final_host["command_compensation_peak_m"][row],
                    "physical_reference_unchanged": True,
                },
            }
            count = final_host["steps"][row]
            shard = args.standard_dataset_dir / f"episode-seed{seed}"
            if count:
                record = collector.finish(env_id=row, frame_count=count)
                record["runtime_config"] = {
                    "ckpt_path": args.ckpt_path,
                    "run_metadata_path": args.run_metadata_path,
                }
                record["contact_body_names"] = list(env.contact_sensor.body_names)
                record["robot_body_names"] = list(env.body_names)
                torch.save(record, output / "control_record.pt")
                report = write_pick_dataset(
                    shard,
                    controller.episode,
                    record,
                    summary,
                    checks[row],
                    staged_videos={key: encoder.path for key, encoder in encoders[row].items()},
                    camera_calibration=getattr(args, "camera_mount_randomizations", {}).get(seed),
                    contact_audit={} if audit is None else audit.description,
                )
                summary["data_collection"] = {
                    "dataset": os.path.relpath(shard, output),
                    "accepted_episodes": report["accepted_episodes"],
                    "quarantined_episodes": report["quarantined_episodes"],
                }
            else:
                write_initialization_rejection(
                    shard, seed=seed, task_type=args.task_kind, reason=reason, details=checks[row]
                )
            _write(output / "summary.json", summary)
            if args.debug_trace:
                (output / "acceptance_trace.jsonl").write_text(
                    "".join(json.dumps(item, allow_nan=True) + "\n" for item in traces[row])
                )
        _write(
            args.output_dir / "summary.json",
            {
                "episode_seeds": args.episode_seeds,
                "num_envs": env.num_envs,
                "physics_steps": loop.physics_steps,
                "policy_physics_s": loop.policy_physics_s,
                "successes": int((fsm.stage == int(fsm.Stage.SUCCESS)).sum()),
                "elapsed_s": time.monotonic() - started,
                "execution_model": "persistent_gpu_rank_batched_environments",
            },
        )
    finally:
        if audit is not None:
            audit.detach()
        for group in encoders.values():
            for encoder in group.values():
                encoder.close()
        runner.close()
