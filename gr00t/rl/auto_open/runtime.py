"""Vectorized Isaac Sim execution runtime for the AutoOpen stage controller."""

import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy

from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
from gr00t.rl.auto_open.cli import LOWLEVEL_ROOT
from gr00t.rl.auto_open.control import (
    _approach_base_target,
    _perturb_base_target,
)
from gr00t.rl.auto_open.execution import TensorCommandBuffers
from gr00t.rl.auto_open.geometry import (
    _batched_geometry_metrics_gpu,
    _batched_panel_finger_clearance_gpu,
    _make_batched_geometry_cache,
    _resolved_door_prim_path,
    initial_approach_clearance_gpu,
)
from gr00t.rl.auto_open.io import (
    door_asset_stage_snapshot as _door_asset_stage_snapshot,
)
from gr00t.rl.auto_open.io import (
    save_rgb_mosaic_video as _save_rgb_mosaic_video,
)
from gr00t.rl.auto_open.io import (
    save_rgb_video as _save_rgb_video,
)
from gr00t.rl.auto_open.io import (
    standard_simulation_provenance as _standard_simulation_provenance,
)
from gr00t.rl.auto_open.lifecycle import reset_controller
from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM
from gr00t.rl.auto_open.tensor_planning import TensorPlanningCoordinator
from gr00t.rl.auto_open.utils import as_numpy as _as_np
from gr00t.rl.envs.door.b2z1_reset_geometry import (
    b2z1_gripper_pose_and_jacobian,
)
from gr00t.rl.task_runtime import BatchedTaskLoop
from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_norm,
    fixed_normalize,
)


def _slice_control_state(state, env_id, num_envs):
    """Return a one-environment view while preserving shared name metadata."""
    sliced = {}
    for key, value in state.items():
        if (
            (torch.is_tensor(value) or isinstance(value, np.ndarray))
            and value.ndim > 0
            and value.shape[0] == num_envs
        ):
            sliced[key] = value[env_id : env_id + 1]
        else:
            sliced[key] = value
    return sliced


def _slice_all_control_states(state, num_envs):
    """Materialize rows for initialization and audit serialization."""
    return [_slice_control_state(state, env_id, num_envs) for env_id in range(num_envs)]


def _control_state_to_host(state, num_envs):
    """Copy observations for initialization or output, never online planning."""
    hosted = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == num_envs:
            hosted[key] = value.detach().cpu().numpy()
        else:
            hosted[key] = value
    return hosted


def _target_height_invariant_poses_gpu(root_states, arm_targets):
    """Dataset poses with GPU FK and frame transforms before serialization."""
    roots = root_states.to(dtype=torch.float64)
    position, rotation, _ = b2z1_gripper_pose_and_jacobian(
        roots[:, :3], roots[:, [4, 5, 6, 3]], arm_targets.to(dtype=torch.float64)
    )
    gripper_to_dataset = roots.new_tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    w, x, y, z = fixed_normalize(roots[:, 3:7], eps=1.0e-12).unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    world_to_yaw = roots.new_zeros((roots.shape[0], 3, 3))
    world_to_yaw[:, 0, 0] = cosine
    world_to_yaw[:, 0, 1] = sine
    world_to_yaw[:, 1, 0] = -sine
    world_to_yaw[:, 1, 1] = cosine
    world_to_yaw[:, 2, 2] = 1.0
    translated = position.clone()
    translated[:, :2] -= roots[:, :2]
    local_position = (world_to_yaw @ translated.unsqueeze(-1)).squeeze(-1)
    local_position -= roots.new_tensor((0.22, 0.0, 0.7))
    local_rotation = world_to_yaw @ (rotation @ gripper_to_dataset)
    return torch.cat((local_rotation[:, :, 0], local_rotation[:, :, 1], local_position), dim=1).to(
        dtype=torch.float32
    )


def _write_stage_progress(output_dir, stage_enum, fsm, episode_step, elapsed_s):
    stages = fsm.stage.detach().cpu().numpy()
    stage_steps = fsm.stage_steps.detach().cpu().numpy()
    lines = [f"episode_step={episode_step} elapsed_s={elapsed_s:.1f}"]
    for value in sorted(set(int(item) for item in stages)):
        mask = stages == value
        name = stage_enum(value).name
        steps = stage_steps[mask]
        limit = int(fsm.stage_limits[value].item())
        budget = "terminal" if limit <= 0 else str(limit)
        lines.append(
            f"stage={name} count={int(mask.sum())} "
            f"stage_steps={int(steps.min())}-{int(steps.max())}/{budget}"
        )
    path = output_dir / "progress.txt"
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    return fsm.stage.detach().clone()


def _write_batched_episode_result(
    args,
    controller,
    runner,
    state,
    env_id,
    seed,
    step_count,
    max_finger_force,
    max_abs_hinge_speed,
    initial_root_state,
    initial_dof_position,
    initial_policy_observation,
    first_low_level_action,
):
    episode_dir = args.output_dir / f"seed_{seed}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    door_path = _resolved_door_prim_path(runner.env, env_id)
    door_prim = runner.env.sim.stage.GetPrimAtPath(door_path)
    door_randomization = dict(door_prim.GetMetadata("customData") or {})
    arm_joint_ids = _as_np(runner.env.arm_joint_ids).astype(int)
    env_origin = _as_np(runner.env.scene.env_origins)[env_id]
    initial_root_local = initial_root_state.copy()
    initial_root_local[:3] -= env_origin
    arm_retract_records = controller.arm_retract_randomization_records
    arm_retract_evaluated = sum(
        int(record["evaluated_candidates"]) for record in arm_retract_records
    )
    arm_retract_unreachable = sum(
        int(record["unreachable_candidates"]) for record in arm_retract_records
    )
    arm_retract_resampled = sum(
        int(record["rejected_before_acceptance"]) for record in arm_retract_records
    )
    metadata = {
        "schema_version": "doorman_boundary_v1",
        "seed": seed,
        "batch_env_id": env_id,
        "batch_seed": int(args.seed),
        "success": controller.stage.name == "SUCCESS",
        "failure_reason": controller.failure_reason,
        "failure_stage": controller.failure_stage,
        "acceptance_diagnostics": controller.last_acceptance_diagnostics,
        "stage_exit_diagnostics": controller.stage_exit_diagnostics,
        "target_noise": controller.target_noise.metadata(),
        "arm_retract_randomization": {
            "base_pose_local": controller.arm_retract_pose_local.tolist(),
            "position_radius_m": float(controller.arm_retract_position_noise_radius_m),
            "rotation_bound_deg": float(controller.arm_retract_rotation_noise_deg),
            "entries": arm_retract_records,
            "evaluated_candidates": arm_retract_evaluated,
            "unreachable_candidates": arm_retract_unreachable,
            "rejected_before_acceptance": arm_retract_resampled,
            "unreachable_frequency": (
                arm_retract_unreachable / arm_retract_evaluated if arm_retract_evaluated else 0.0
            ),
        },
        "camera": {"enabled": False, "names": [], "frame_count": {}},
        "scene": {
            "door_prim_path": door_path,
            "robot_side": controller.robot_side,
            "door_randomization": door_randomization,
        },
        "robot_state": {
            "initial_approach_clearance": controller.initial_approach_clearance,
            "initial_base_pose_w": initial_root_state.tolist(),
            "initial_base_pose_env": initial_root_local.tolist(),
            "initial_arm_joint_position": initial_dof_position[arm_joint_ids].tolist(),
            "initial_gripper_position": float(
                initial_dof_position[int(runner.env.gripper_joint_ids[0].item())]
            ),
        },
        "consistency_signature": {
            "initial_policy_observation": initial_policy_observation.tolist(),
            "first_low_level_action": first_low_level_action.tolist(),
        },
        "result": {
            "final_stage": controller.stage.name,
            "requested_terminal_stage": args.terminal_stage,
            "door_open_io": args.doorman_door_open_io,
            "steps": int(step_count),
            "max_finger_force_n": float(max_finger_force),
            "max_abs_hinge_speed_deg_s": math.degrees(max_abs_hinge_speed),
            "summary_only": True,
        },
    }
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _finish_standard_collection(
    args,
    *,
    collector,
    controller,
    runner,
    episode_seed,
    dataset_dir,
    initial_root_state,
    initial_dof_position,
    initial_gripper_position,
    door_randomization,
    door_asset_snapshot,
):
    success = controller.stage.name == "SUCCESS"
    reasons = (
        ()
        if success
        else (
            f"autoopen_failure:{controller.failure_stage or controller.stage.name}:"
            f"{controller.failure_reason or 'unknown'}",
        )
    )
    episode = collector.finish(
        acceptance_reasons=reasons,
        provenance={
            "seed": episode_seed,
            "initial_root_pose_w": initial_root_state.tolist(),
            "initial_dof_position": initial_dof_position.tolist(),
            "initial_gripper_position": float(initial_gripper_position),
            "initial_approach_clearance": controller.initial_approach_clearance,
            "door_randomization": door_randomization,
            "door_asset": door_asset_snapshot,
            "camera_mount_randomization": dict(
                getattr(args, "camera_mount_randomizations", {}).get(
                    int(episode_seed), args.camera_mount_randomization
                )
                or {}
            ),
            "contact_audit": collector.contact_view_description,
            "robot_joint_names": list(runner.env.dof_names),
            "gripper_joint_index": int(runner.env.gripper_joint_ids[0].item()),
            "images_enabled": args.collection_camera_mode != "none",
            "camera_mode": args.collection_camera_mode,
        },
    )
    collector.close()
    from gr00t.rl.data_collection.auto_open_writer import AutoOpenDatasetWriter

    camera_calibration = getattr(args, "episode_camera_configs", {}).get(
        int(episode_seed), {"audit_only": True}
    )
    writer = AutoOpenDatasetWriter(
        dataset_dir,
        camera_calibration=camera_calibration,
        simulation_provenance=_standard_simulation_provenance(args, runner, episode_seed),
    )
    writer.write_episode(episode)
    writer.finalize()


def _run_tensor_fsm_loop(
    args,
    *,
    stage_enum,
    runner,
    controllers,
    geometry_cache,
    device_state,
    approach_command_targets,
    approach_nominal_targets,
    initial_root_states,
    initial_dof_positions,
    initial_policy_observations,
    initial_gripper_positions,
    standard_collectors,
    standard_door_randomizations,
    standard_door_asset_snapshots,
    episode_seeds,
    video_frames,
    video_timestamps,
):
    """Run the GPU-resident FSM without per-env command/advance calls."""
    Stage = stage_enum
    env = runner.env
    num_envs = len(controllers)
    fsm = TensorDoorFSM(
        stage_enum=Stage,
        controllers=controllers,
        env=env,
        approach_command_xy=np.stack([target[0] for target in approach_command_targets]),
        approach_command_yaw=np.asarray([target[1] for target in approach_command_targets]),
        approach_nominal_xy=np.stack([target[0] for target in approach_nominal_targets]),
        approach_nominal_yaw=np.asarray([target[1] for target in approach_nominal_targets]),
        terminal_stage=Stage[args.terminal_stage],
        pull_door=args.doorman_door_open_io == "out",
        pull_panel_contact_baseline=getattr(
            args, "pull_panel_contact_baseline", "trunk_front_minus10"
        ),
    )
    buffers = TensorCommandBuffers.from_env(env)
    planner = TensorPlanningCoordinator(controllers, Stage)
    timing = {
        "command_s": 0.0,
        "teleop_prepare_s": 0.0,
        "policy_physics_s": 0.0,
        "state_to_host_s": 0.0,
        "advance_s": 0.0,
        "planning_s": 0.0,
        "geometry_gpu_s": 0.0,
        "panel_clearance_backend_s": 0.0,
        "standard_capture_s": 0.0,
        "standard_complete_s": 0.0,
        "standard_finalize_s": 0.0,
    }
    batch_loop = BatchedTaskLoop(
        runner,
        episode_seeds,
        max_steps=args.max_steps,
        success_stage=Stage.SUCCESS,
        failure_stage=Stage.FAILURE,
    )
    step_counts = batch_loop.step_counts
    max_finger_forces = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    max_hinge_speeds = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    first_actions = batch_loop.first_actions
    loop_started = time.monotonic()
    last_progress_stages = _write_stage_progress(args.output_dir, Stage, fsm, 0, 0.0)
    debug_acceptance_rows = [[] for _ in range(num_envs)]
    debug_event_rows = [[] for _ in range(num_envs)]

    capture_env_ids, capture_root_states, inverse_input_world = [], None, None

    def prepare_tick(device_state, active):
        nonlocal max_finger_forces
        planning_start = time.perf_counter()
        contact_rows = fsm.contact_plan_requests()
        retract_rows = fsm.push_retract_requests(device_state)
        if len(contact_rows):
            planner.plan_contact(fsm, device_state, contact_rows)
        if len(retract_rows):
            planner.plan_push_retract(fsm, device_state, retract_rows)
        timing["planning_s"] += time.perf_counter() - planning_start
        forces = device_state.get("finger_contact_forces_w")
        if forces is not None:
            force = fixed_norm(forces).amax(dim=-1)
            max_finger_forces = torch.maximum(max_finger_forces, force)
            fsm.fail(torch.where(active & (force > 600.0))[0], 2)
        if args.debug_trace:
            fsm.begin_acceptance_frame()

    def scene_geometry(device_state):
        grasp_rows = torch.where(
            (fsm.stage == int(Stage.INSERT_AROUND_HANDLE)) | (fsm.stage == int(Stage.GRASP_HANDLE))
        )[0]
        geometry_start = time.perf_counter()
        grasp_geometry = (
            _batched_geometry_metrics_gpu(
                device_state,
                geometry_cache,
                grasp_rows,
                as_numpy=False,
            )
            if len(grasp_rows)
            else None
        )
        timing["geometry_gpu_s"] += time.perf_counter() - geometry_start
        panel_rows = torch.where(
            (fsm.stage == int(Stage.MOVE_TO_PANEL_STANDOFF))
            | (fsm.stage == int(Stage.PUSH_PANEL_OPEN))
            | (fsm.stage == int(Stage.STOW_ARM_AFTER_PUSH))
            | (fsm.stage == int(Stage.ADVANCE_TO_NEXT_PUSH_STATION))
        )[0]
        clearance = torch.full((num_envs,), torch.nan, dtype=torch.float32, device=env.device)
        if len(panel_rows):
            panel_start = time.perf_counter()
            clearance[panel_rows] = _batched_panel_finger_clearance_gpu(
                device_state,
                geometry_cache,
                panel_rows,
                normal_sign=fsm.panel_contact_normal_sign[panel_rows],
            )
            timing["panel_clearance_backend_s"] += time.perf_counter() - panel_start

        return {"grasp_geometry": grasp_geometry, "finger_clearance": clearance}

    def capture_before(step, active, device_state):
        nonlocal capture_env_ids, capture_root_states, inverse_input_world
        capture_env_ids = []
        capture_state_host = None
        capture_root_states = None
        inverse_input_world = None
        if standard_collectors:
            standard_capture_start = time.perf_counter()
            capture_env_ids = torch.where(active)[0].detach().cpu().tolist()
            capture_root_states = device_state["root_states"].detach().clone()
            capture_state_host = _control_state_to_host(device_state, num_envs)
            capture_states = _slice_all_control_states(capture_state_host, num_envs)
            high_batch = (
                torch.cat(
                    (buffers.base, buffers.ee_cart, buffers.ee_rpy, buffers.gripper),
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            inverse_input_world = env.curr_ee_goal_cart_world.detach().clone()
            stages_before_step = fsm.stage.detach().cpu().numpy()
            standard_collectors[0].contact_audit.begin_tick()
            for env_id in capture_env_ids:
                standard_collectors[env_id].capture_before_step(
                    capture_states[env_id],
                    high_batch[env_id],
                    stage_name=Stage(int(stages_before_step[env_id])).name,
                    begin_contact_tick=False,
                )
            timing["standard_capture_s"] += time.perf_counter() - standard_capture_start

    def capture_after(step, active, result):
        nonlocal max_hinge_speeds
        if standard_collectors:
            standard_complete_start = time.perf_counter()
            contacts_by_env = standard_collectors[0].contact_audit.end_tick_by_env()
            post_state_host = _control_state_to_host(result["state"], num_envs)
            post_states = _slice_all_control_states(post_state_host, num_envs)
            target_poses = (
                _target_height_invariant_poses_gpu(
                    capture_root_states, result["state"]["arm_position_target"]
                )
                .detach()
                .cpu()
                .numpy()
            )
            inverse_output_world, _, _ = b2z1_gripper_pose_and_jacobian(
                capture_root_states[:, :3],
                capture_root_states[:, [4, 5, 6, 3]],
                result["state"]["arm_position_target"],
            )
            inverse_input_host = inverse_input_world.detach().cpu().numpy()
            inverse_output_host = inverse_output_world.detach().cpu().numpy()
            joint_modes = env.teleop_arm_joint_control_mask.detach().cpu().numpy()
            for env_id in capture_env_ids:
                standard_collectors[env_id].complete_after_step(
                    {"state": post_states[env_id]},
                    contacts=contacts_by_env.get(env_id, ()),
                    target_height_invariant_pose=target_poses[env_id],
                    joint_mode=joint_modes[env_id],
                    ee_inverse_input_world=inverse_input_host[env_id],
                    ee_inverse_output_world=inverse_output_host[env_id],
                )
            timing["standard_complete_s"] += time.perf_counter() - standard_complete_start
        elif args.record_video:
            manager = runner.rgb_camera_manager
            if manager is None or not manager.enabled:
                raise RuntimeError("video regression requires enabled RGB cameras")
            timestamp = step * float(env.dt)
            for env_id in torch.where(active)[0].detach().cpu().tolist():
                images_by_name = manager.render_images(env_id=env_id, update=False)
                for name, images in images_by_name.items():
                    if len(images) != 1:
                        raise RuntimeError(
                            f"camera {name} returned {len(images)} frames for environment {env_id}"
                        )
                    video_frames.setdefault(env_id, {}).setdefault(name, []).append(images[0])
                    video_timestamps.setdefault(env_id, {}).setdefault(name, []).append(timestamp)
        device_state = result["state"]
        hinge_speed = device_state["door_joint_velocity"][:, fsm.hinge_idx].abs()
        max_hinge_speeds = torch.maximum(max_hinge_speeds, hinge_speed)

    def report_tick(step, debug_stage_before, events):
        nonlocal last_progress_stages
        if args.debug_trace:
            frame_rows = fsm.acceptance_frame_rows(step)
            for env_id, rows_for_env in enumerate(frame_rows):
                debug_acceptance_rows[env_id].extend(rows_for_env)
            debug_stage_after = fsm.stage.detach().cpu().tolist()
            debug_stage_before_host = debug_stage_before.detach().cpu().tolist()
            for env_id, (before, after) in enumerate(
                zip(debug_stage_before_host, debug_stage_after)
            ):
                if before != after:
                    debug_event_rows[env_id].append(
                        {
                            "frame_index": int(step) + 1,
                            "state_frame_index": int(step) + 1,
                            "action_frame_index": int(step),
                            "event": "stage_transition",
                            "from_stage": Stage(int(before)).name,
                            "to_stage": Stage(int(after)).name,
                        }
                    )
            for event_name, event_rows in events.items():
                for env_id in event_rows.detach().cpu().tolist():
                    debug_event_rows[env_id].append(
                        {
                            "frame_index": int(step) + 1,
                            "state_frame_index": int(step) + 1,
                            "action_frame_index": int(step),
                            "event": event_name,
                            "stage": Stage(int(debug_stage_after[env_id])).name,
                        }
                    )
        if step % 50 == 0 or bool(torch.any(fsm.stage != last_progress_stages).item()):
            last_progress_stages = _write_stage_progress(
                args.output_dir,
                Stage,
                fsm,
                step + 1,
                time.monotonic() - loop_started,
            )

    for step, device_state, events in batch_loop.run_stages(
        fsm,
        planner,
        device_state,
        buffers,
        timing=timing,
        prepare=prepare_tick,
        geometry=scene_geometry,
        capture_before=capture_before,
        capture_after=capture_after,
        report=report_tick,
    ):
        if args.pitch_probe_output:
            planned = fsm.arm_align_path.length > 0
            failed = fsm.stage == int(Stage.FAILURE)
            if bool(torch.all(planned | failed).item()):
                break

    planner.finalize_diagnostics()

    if args.pitch_probe_output:
        planned = (fsm.arm_align_path.length > 0).detach().cpu().tolist()
        failed = (fsm.stage == int(Stage.FAILURE)).detach().cpu().tolist()
        probe = {
            "seeds": [
                {
                    "seed": int(seed),
                    "status": "selected"
                    if planned[index]
                    else "failed"
                    if failed[index]
                    else "timeout",
                    "pitch_deg": float(controllers[index].pregrasp_pitch_deg)
                    if planned[index]
                    else None,
                    "candidates": getattr(controllers[index], "pregrasp_pitch_diagnostics", None),
                }
                for index, seed in enumerate(episode_seeds)
            ]
        }
        output = Path(args.pitch_probe_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
        print(f"[doorman][pitch-probe] output={output}", flush=True)

    final_state = _control_state_to_host(device_state, num_envs)
    state_rows = _slice_all_control_states(final_state, num_envs)
    stages = fsm.stage.detach().cpu().numpy()
    stage_steps = fsm.stage_steps.detach().cpu().numpy()
    failure_codes = fsm.failure_code.detach().cpu().numpy()
    failure_from_stages = fsm.failure_from_stage.detach().cpu().numpy()
    failure_reason_by_code = {
        2: "finger_contact_force_exceeded_600_n",
        10: "center_handle_panel_clearance_below_5_mm",
        11: "grasp_handle_panel_clearance_below_5_mm",
        20: "push_door_overspeed_for_3_frames",
        21: "push_door_no_progress_for_300_frames",
        22: "door_hinge_angle_exceeded_94_deg",
        30: "through_robot_fallen",
        31: "through_door_swung_back_before_tail_clearance",
        32: "through_base_made_no_progress_for_150_frames",
        40: "pregrasp_path_planning_failed",
        41: "center_handle_geometry_invalid",
        42: "center_handle_vertical_alignment_infeasible",
        43: "center_handle_path_planning_failed",
        44: "arm_path_planning_failed",
        45: "panel_contact_inverse_mapping_failed",
        46: "arm_retract_pose_sampling_failed",
        47: "pregrasp_no_reachable_pitch",
        48: "pull_no_frame_and_wall_safe_target_at_requested_gap_distance",
        49: "panel_path_inverse_residual_exceeded",
        50: "pull_panel_entry_geometry_invalid",
        51: "pull_panel_arc_no_progress",
        52: "pull_panel_door_overspeed",
        53: "pull_base_station_sweep_intersects_wall_or_panel",
        54: "pull_no_distinct_next_station_within_approved_angle_noise",
        55: "pull_initial_station_no_clearance_and_full_insertion_feasible_candidate",
    }

    def failure_reason(code, origin):
        if code == 0:
            return "max_steps_before_terminal_stage"
        if code >= 100:
            return f"{Stage(origin).name.lower()}_acceptance_timeout"
        return failure_reason_by_code.get(code, f"unknown_failure_code_{code}")

    step_counts_np = step_counts.detach().cpu().numpy()
    max_forces_np = max_finger_forces.detach().cpu().numpy()
    max_hinge_np = max_hinge_speeds.detach().cpu().numpy()
    first_actions_np = first_actions.detach().cpu().numpy()
    arrival_settle_np = fsm.base_arrival_settle_steps.detach().cpu().numpy()
    arrival_max_settle_np = fsm.base_arrival_max_settle_steps.detach().cpu().numpy()
    arrival_distance_np = fsm.pending_station_distance.detach().cpu().numpy()
    arrival_yaw_np = fsm.pending_yaw_error.detach().cpu().numpy()
    successes = 0
    base_seed = int(args.seed if args.seed is not None else 0)
    failure_acceptance = {}
    debug_trace_summaries = [None] * num_envs
    if args.debug_trace:
        from gr00t.rl.auto_open.debug_trace import summarize_acceptance_trace

        debug_trace_summaries = [summarize_acceptance_trace(rows) for rows in debug_acceptance_rows]

    def acceptance_unmet(condition):
        return (not condition["current"]) or (condition.get("window_satisfied") is False)

    for env_id, controller in enumerate(controllers):
        controller.stage = Stage(int(stages[env_id]))
        controller.stage_steps = int(stage_steps[env_id])
        if controller.stage not in (Stage.SUCCESS, Stage.FAILURE):
            controller.failure_stage = controller.stage.name
            controller.stage = Stage.FAILURE
            controller.failure_reason = f"max_steps:{args.max_steps}"
        elif controller.stage == Stage.FAILURE and controller.failure_reason is None:
            origin = int(failure_from_stages[env_id])
            code = int(failure_codes[env_id])
            controller.failure_reason = failure_reason(code, origin)
            controller.failure_stage = Stage(origin).name if origin else Stage.FAILURE.name
        root = final_state["root_states"][env_id]
        if fsm.pull_door and int(fsm.stage_visit[env_id, int(Stage.MOVE_TO_GAP_STATION)].item()):
            controller.last_acceptance_diagnostics["pull_clear_sweep_plan"] = {
                "target_xy_world_m": fsm.pull_initial_station_xy[env_id].detach().cpu().tolist(),
                "target_yaw_rad": float(fsm.pull_initial_station_yaw[env_id].item()),
                "gap_center_xy_world_m": fsm.pull_gap_xy[env_id].detach().cpu().tolist(),
                "frame_free_boundary_from_hinge_m": float(fsm.pull_frame_edge[env_id].item()),
                "planned_base_frame_clearance_m": float(
                    fsm.pull_planned_frame_clearance[env_id].item()
                ),
                "wall_front_from_hinge_m": float(fsm.pull_wall_front[env_id].item()),
                "planned_base_wall_clearance_m": float(
                    fsm.pull_planned_wall_clearance[env_id].item()
                ),
                "required_wall_clearance_m": fsm.pull_station_config.wall_clearance_m,
                "extra_planning_margin_m": fsm.pull_station_config.planning_margin_m,
                "sampling_local_x_range_m": fsm.pull_station_config.first_x_range_m,
                "sampling_local_y_range_m": fsm.pull_station_config.first_y_range_m,
                "sampling_local_yaw_range_deg": fsm.pull_station_config.first_yaw_range_deg,
                "candidate_planning": fsm.pull_initial_station_history[env_id],
                "required_clearance_m": fsm.pull_station_config.clearance_m,
            }
            if int(fsm.stage_visit[env_id, int(Stage.INSERT_INTO_DOOR_GAP)].item()):
                controller.last_acceptance_diagnostics["pull_panel_result"] = {
                    "cycles": fsm.pull_cycle_history[env_id],
                    "base_stations": fsm.pull_station_history[env_id],
                    "needs_base_reposition": bool(fsm.pull_finished_for_reposition[env_id].item()),
                    "comfort_radius_m": float(fsm.push_comfort_reach_radius[env_id].item()),
                    "max_angular_lead_rad": float(fsm.push_max_angular_lead[env_id].item()),
                    "target_door_speed_rad_s": float(fsm.push_target_angular_speed[env_id].item()),
                }
        diagnostic_stage = int(failure_from_stages[env_id]) or int(stages[env_id])
        stage_acceptance = fsm.acceptance_diagnostics(env_id, diagnostic_stage)
        failure_acceptance[env_id] = stage_acceptance
        controller.last_acceptance_diagnostics.update(
            {
                "stage": Stage(diagnostic_stage).name,
                "conditions": stage_acceptance,
                "unmet_conditions": [
                    name
                    for name, condition in stage_acceptance.items()
                    if acceptance_unmet(condition)
                ],
                "debug_trace": (
                    {
                        "schema_version": debug_trace_summaries[env_id]["schema_version"],
                        "rows": debug_trace_summaries[env_id]["rows"],
                        "path": "debug/acceptance_trace.jsonl",
                        "summary_path": "debug/acceptance_summary.json",
                    }
                    if args.debug_trace
                    else None
                ),
                "base_arrival_settle_steps": int(arrival_settle_np[env_id]),
                "base_arrival_max_settle_steps": int(arrival_max_settle_np[env_id]),
                "base_arrival_distance_m": float(arrival_distance_np[env_id]),
                "base_arrival_yaw_error_rad": float(arrival_yaw_np[env_id]),
                "base_arrival_linear_speed_m_s": float(np.linalg.norm(root[7:10])),
                "base_arrival_angular_speed_rad_s": float(np.linalg.norm(root[10:13])),
                "base_arrival_planar_speed_m_s": float(np.linalg.norm(root[7:9])),
                "base_arrival_yaw_rate_rad_s": float(abs(root[12])),
                "base_arrival_velocity_xyz_m_s": root[7:10].astype(float).tolist(),
                "base_arrival_angular_velocity_xyz_rad_s": root[10:13].astype(float).tolist(),
            }
        )
        successes += int(controller.stage == Stage.SUCCESS)
        _write_batched_episode_result(
            args,
            controller,
            runner,
            state_rows[env_id],
            env_id,
            int(episode_seeds[env_id]),
            step_counts_np[env_id],
            max_forces_np[env_id],
            max_hinge_np[env_id],
            initial_root_states[env_id],
            initial_dof_positions[env_id],
            initial_policy_observations[env_id],
            first_actions_np[env_id],
        )
    if args.record_video:
        for env_id, episode_seed in enumerate(episode_seeds):
            video_dir = args.output_dir / f"seed_{int(episode_seed)}" / "video"
            video_dir.mkdir(exist_ok=True)
            env_frames = video_frames.get(env_id, {})
            env_timestamps = video_timestamps.get(env_id, {})
            for name, frames in env_frames.items():
                _save_rgb_video(
                    frames,
                    env_timestamps[name],
                    video_dir / f"{name}.mp4",
                )
            _save_rgb_mosaic_video(
                env_frames,
                env_timestamps,
                video_dir / "all_cameras_mosaic.mp4",
            )
    if standard_collectors:
        standard_finalize_start = time.perf_counter()
        for env_id, collector in enumerate(standard_collectors):
            episode_seed = int(episode_seeds[env_id])
            _finish_standard_collection(
                args,
                collector=collector,
                controller=controllers[env_id],
                runner=runner,
                episode_seed=episode_seed,
                dataset_dir=(Path(args.standard_dataset_dir) / f"episode-seed{episode_seed}"),
                initial_root_state=initial_root_states[env_id],
                initial_dof_position=initial_dof_positions[env_id],
                initial_gripper_position=initial_gripper_positions[env_id],
                door_randomization=standard_door_randomizations[env_id],
                door_asset_snapshot=standard_door_asset_snapshots[env_id],
            )
            if args.debug_trace:
                from gr00t.rl.auto_open.debug_trace import write_debug_trace

                trace_root = (
                    Path(args.standard_dataset_dir) / f"episode-seed{episode_seed}" / "debug"
                )
                write_debug_trace(
                    trace_root,
                    debug_acceptance_rows[env_id],
                    debug_event_rows[env_id],
                )
        standard_collectors[0].contact_audit.detach()
        timing["standard_finalize_s"] += time.perf_counter() - standard_finalize_start
        timing.update(
            {
                f"collector_{name}": float(value)
                for name, value in {
                    key: sum(float(c.timing.get(key, 0.0)) for c in standard_collectors)
                    for key in {key for c in standard_collectors for key in c.timing}
                }.items()
            }
        )
    summary = {
        "batch_seed": base_seed,
        "episode_seeds": [int(seed) for seed in episode_seeds],
        "num_envs": num_envs,
        "active_env_ids": list(range(num_envs)),
        "successes": successes,
        "success_rate": successes / num_envs,
        "engine": "tensor_fsm",
        "requested_terminal_stage": args.terminal_stage,
        "door_open_io": args.doorman_door_open_io,
        "batch_timing": timing,
        "planning": planner.stats,
        "failure_codes": {
            str(code): int(np.sum(failure_codes == code))
            for code in np.unique(failure_codes)
            if code
        },
        "failure_from_stages": {
            Stage(int(stage)).name: int(np.sum(failure_from_stages == stage))
            for stage in np.unique(failure_from_stages)
            if stage
        },
        "failures": [
            {
                "seed": int(episode_seeds[env_id]),
                "code": int(failure_codes[env_id]),
                "stage": Stage(int(failure_from_stages[env_id]) or int(stages[env_id])).name,
                "reason": failure_reason(
                    int(failure_codes[env_id]), int(failure_from_stages[env_id])
                ),
                "stage_steps": int(stage_steps[env_id]),
                "acceptance_conditions": failure_acceptance[env_id],
                "unmet_conditions": [
                    name
                    for name, condition in failure_acceptance[env_id].items()
                    if acceptance_unmet(condition)
                ],
            }
            for env_id in range(num_envs)
            if int(stages[env_id]) != int(Stage.SUCCESS)
        ],
        "panel_terminal_diagnostics": {
            "finger_clearance_m": fsm.panel_last_finger_clearance.detach().cpu().tolist(),
            "lateral_error_m": fsm.panel_last_lateral_error.detach().cpu().tolist(),
            "axis_error_rad": fsm.panel_last_axis_error.detach().cpu().tolist(),
            "ee_speed_m_s": fsm.panel_last_ee_speed.detach().cpu().tolist(),
            "settle_steps": fsm.panel_contact_settle_steps.detach().cpu().tolist(),
        },
        "push_terminal_diagnostics": {
            "hinge_rad": device_state["door_joint_position"][:, fsm.hinge_idx]
            .detach()
            .cpu()
            .tolist(),
            "hinge_speed_rad_s": device_state["door_joint_velocity"][:, fsm.hinge_idx]
            .detach()
            .cpu()
            .tolist(),
            "command_hinge_rad": fsm.push_command_hinge_angle.detach().cpu().tolist(),
            "start_hinge_rad": fsm.push_start_hinge_angle.detach().cpu().tolist(),
            "ee_actual_world_m": device_state["ee_pos"].detach().cpu().tolist(),
            "ee_command_world_m": fsm.ee_command_world.detach().cpu().tolist(),
            "ee_goal_local_m": device_state["ee_goal_cart_local"].detach().cpu().tolist(),
            "no_progress_steps": fsm.push_no_progress_steps.detach().cpu().tolist(),
            "arm_detached": fsm.push_arm_detached.detach().cpu().tolist(),
            "entry_ee_world_m": fsm.push_entry_ee_world.detach().cpu().tolist(),
            "entry_command_world_m": fsm.push_entry_command_world.detach().cpu().tolist(),
            "contact_world_m": fsm.push_contact_world.detach().cpu().tolist(),
        },
        "debug_traces": {
            str(int(episode_seeds[env_id])): {
                "rows": int(debug_trace_summaries[env_id]["rows"]),
                "event_rows": len(debug_event_rows[env_id]),
                "path": f"episode-seed{int(episode_seeds[env_id])}/debug/acceptance_trace.jsonl",
            }
            for env_id in range(num_envs)
        }
        if args.debug_trace
        else {},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "[doorman][batch-timing] "
        + " ".join(f"{name}={value:.3f}" for name, value in timing.items()),
        flush=True,
    )
    print(f"[doorman][batch] complete: {successes}/{num_envs} active", flush=True)
    if args.headless:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    runner.close()
    return successes


def _run_batched(args, *, controller_cls, stage_enum):
    DoorController = controller_cls
    Stage = stage_enum
    num_envs = int(args.batched_envs)
    if num_envs < 1:
        raise ValueError("--batched_envs must be positive")
    args.num_envs = num_envs
    if args.env_spacing is None:
        args.env_spacing = 8.0
    base_seed = int(args.seed if args.seed is not None else 0)
    episode_seeds = (
        list(range(base_seed, base_seed + num_envs))
        if args.episode_seeds is None
        else [int(seed) for seed in args.episode_seeds]
    )
    if len(episode_seeds) != num_envs:
        raise ValueError("--episode_seeds must contain exactly --batched_envs values")
    if len(set(episode_seeds)) != num_envs:
        raise ValueError("--episode_seeds must not contain duplicates")
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    runner = ManipLoco_Policy(args)
    env = runner.env
    if args.collection_camera_mode != "none":
        manager = runner.rgb_camera_manager
        if manager is None or not manager.enabled:
            raise RuntimeError("configured collection cameras were not initialized")
        configs = getattr(args, "episode_camera_configs", {})
        manager.set_episode_camera_configs(
            {env_id: configs[int(seed)] for env_id, seed in enumerate(episode_seeds)}
        )
    door_handle_paths = [
        str(prim.GetPath())
        for prim in env.sim.stage.Traverse()
        if prim.GetName() == "door_handle" and str(prim.GetPath()).startswith("/World/envs/")
    ]
    print(
        f"[doorman][batch-init] usd_door_handle_count={len(door_handle_paths)} "
        f"paths={door_handle_paths if num_envs <= 32 else 'omitted'}",
        flush=True,
    )
    # Generate one current PhysX contact frame from the collision-free reset
    # pose before policy inference. ContactSensor.reset() marks its buffers
    # outdated, so reading immediately would reload the previous solve.
    env.reset()
    neutral_actions = torch.zeros(
        (num_envs, env.num_actions), dtype=torch.float32, device=env.device
    )
    runner.obs, *_ = env.step(neutral_actions)
    device_state = runner.get_control_state()
    full_state = _control_state_to_host(device_state, num_envs)
    state_rows = _slice_all_control_states(full_state, num_envs)
    shared_arm_joint_limits = read_arm_joint_limits(
        LOWLEVEL_ROOT / "resources/robots/b2z1/urdf" / "b2z1.urdf",
    )
    controllers = []
    shared_robot_geometry = None
    controller_stdout = open(os.devnull, "w") if args.summary_only else None
    with (
        contextlib.redirect_stdout(controller_stdout)
        if controller_stdout is not None
        else contextlib.nullcontext()
    ):
        for env_id in range(num_envs):
            sliced_state = state_rows[env_id]
            controller = DoorController(
                runner,
                robot_side=args.robot_side,
                position_only_center=args.position_only_center,
                disable_push_base_assist=args.disable_push_base_assist,
                env_id=env_id,
                shared_arm_joint_limits=shared_arm_joint_limits,
                shared_robot_geometry=shared_robot_geometry,
                target_noise_seed=episode_seeds[env_id],
                target_noise_config=args.target_noise_config,
                arm_retract_pose_local=args.arm_retract_pose_local,
                arm_retract_position_noise_radius_m=(args.arm_retract_position_noise_radius_m),
                arm_retract_rotation_noise_deg=args.arm_retract_rotation_noise_deg,
            )
            reset_controller(
                controller,
                state=sliced_state,
                reset_env=False,
                close_gripper=False,
            )
            controllers.append(controller)
            if shared_robot_geometry is None:
                shared_robot_geometry = controller.center_geometry
    geometry_cache = _make_batched_geometry_cache(controllers, env.device)

    # Initialization measures open-jaw geometry. Close the complete batch
    # before starting control so all environments share the same semantics.
    gripper_joint_id = int(env.gripper_joint_ids[0].item())
    joint_pos = env.robot.data.joint_pos.clone()
    joint_vel = env.robot.data.joint_vel.clone()
    joint_pos[:, gripper_joint_id] = torch.as_tensor(
        [controller.q_closed for controller in controllers],
        device=env.device,
        dtype=joint_pos.dtype,
    )
    joint_vel[:, gripper_joint_id] = 0.0
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    env.robot.set_joint_position_target(joint_pos)
    runner.obs = env.get_observations()
    device_state = runner.get_control_state()
    full_state = _control_state_to_host(device_state, num_envs)
    initial_root_states = _as_np(full_state["root_states"])[:, :7].copy()
    initial_dof_positions = _as_np(full_state["dof_pos"]).copy()
    initial_gripper_positions = _as_np(full_state["gripper_position"]).reshape(-1).copy()
    initial_policy_observations = _as_np(runner.obs).copy()
    approach_clearance = initial_approach_clearance_gpu(device_state, geometry_cache)
    station_xy = torch.as_tensor(np.stack([c.station_xy for c in controllers]), device=env.device)
    station_xy = station_xy + (
        (approach_clearance["approach_distance_m"] - 0.85)[:, None]
        * geometry_cache["door_normal"][:, :2]
    ).to(station_xy.dtype)
    # Materialize initialization metadata once; geometry and target math stay on GPU.
    clearance_host = {
        name: value.detach().cpu().tolist() for name, value in approach_clearance.items()
    }
    stations_host = station_xy.detach().cpu().numpy()
    for env_id, controller in enumerate(controllers):
        controller.initial_approach_clearance = {
            name: value[env_id] for name, value in clearance_host.items()
        }
        controller.approach_distance = clearance_host["approach_distance_m"][env_id]
        controller.station_xy = stations_host[env_id].copy()
    # panel_station_xy is a later-stage target; this rule is APPROACH_HANDLE-only.
    standard_collectors = []
    standard_door_randomizations = []
    standard_door_asset_snapshots = []
    video_frames = {}
    video_timestamps = {}
    if args.standard_dataset_dir is not None:
        from gr00t.rl.data_collection.auto_open_collector import (
            AutoOpenStandardEpisodeCollector,
        )
        from gr00t.rl.data_collection.contact_audit import SubstepContactAudit

        contact_audit = SubstepContactAudit.create(env)
        contact_audit.attach()
        for env_id, episode_seed in enumerate(episode_seeds):
            shard_dir = Path(args.standard_dataset_dir) / f"episode-seed{episode_seed}"
            standard_collectors.append(
                AutoOpenStandardEpisodeCollector(
                    runner,
                    controllers[env_id],
                    episode_index=0,
                    seed=episode_seed,
                    video_staging_root=(shard_dir / ".staging" / "episode_000000"),
                    capture_images=args.collection_camera_mode != "none",
                    camera_mode=args.collection_camera_mode,
                    env_id=env_id,
                    contact_audit=contact_audit,
                    debug_diagnostics=args.debug_trace,
                )
            )
            door_prim = env.sim.stage.GetPrimAtPath(_resolved_door_prim_path(env, env_id))
            standard_door_randomizations.append(dict(door_prim.GetMetadata("customData") or {}))
            standard_door_asset_snapshots.append(
                _door_asset_stage_snapshot(env.sim.stage, door_prim)
            )
    print(
        "[doorman][batch-init] door_joint_position="
        f"{_as_np(full_state['door_joint_position']).round(5).tolist()} "
        "gripper_position="
        f"{_as_np(full_state['gripper_position']).round(5).reshape(-1).tolist()}",
        flush=True,
    )

    approach_nominal_targets = [_approach_base_target(controller) for controller in controllers]
    approach_command_targets = [
        _perturb_base_target(controller, "approach", *target)
        for controller, target in zip(controllers, approach_nominal_targets, strict=True)
    ]
    if controller_stdout is not None:
        controller_stdout.close()
    return _run_tensor_fsm_loop(
        args,
        stage_enum=Stage,
        runner=runner,
        controllers=controllers,
        geometry_cache=geometry_cache,
        device_state=device_state,
        approach_command_targets=approach_command_targets,
        approach_nominal_targets=approach_nominal_targets,
        initial_root_states=initial_root_states,
        initial_dof_positions=initial_dof_positions,
        initial_policy_observations=initial_policy_observations,
        initial_gripper_positions=initial_gripper_positions,
        standard_collectors=standard_collectors,
        standard_door_randomizations=standard_door_randomizations,
        standard_door_asset_snapshots=standard_door_asset_snapshots,
        episode_seeds=episode_seeds,
        video_frames=video_frames,
        video_timestamps=video_timestamps,
    )
