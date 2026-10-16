"""Vectorized Isaac Sim execution runtime for the AutoOpen stage controller."""

import concurrent.futures
import contextlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_norm,
    fixed_normalize,
)

from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy
from legged_gym.utils.ee_steady_state import B2Z1SteadyEeForwardModel
from legged_gym.utils.steady_path_worker import (
    initialize_steady_path_worker,
)
from gr00t.rl.auto_open.cli import LOWLEVEL_ROOT
from gr00t.rl.auto_open.geometry import (
    _batched_geometry_metrics_gpu,
    _batched_handle_centers_gpu,
    _batched_panel_finger_clearance_gpu,
    _make_batched_geometry_cache,
    _resolved_door_prim_path,
)
from gr00t.rl.auto_open.lifecycle import reset_controller
from gr00t.rl.auto_open.io import (
    door_asset_stage_snapshot as _door_asset_stage_snapshot,
    save_rgb_mosaic_video as _save_rgb_mosaic_video,
    save_rgb_video as _save_rgb_video,
    standard_simulation_provenance as _standard_simulation_provenance,
)
from gr00t.rl.auto_open.control import (
    _approach_base_target,
    _arm_drive_stiffness,
    _limit_base_commands,
    _perturb_base_target,
)
from gr00t.rl.data_collection.auto_open_collector import (
    batched_target_height_invariant_poses,
)
from gr00t.rl.auto_open.utils import as_numpy as _as_np
from gr00t.rl.auto_open.tensor_fsm import TensorCommandBuffers, TensorDoorFSM
from gr00t.rl.auto_open.tensor_planning import TensorPlanningCoordinator


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
    """Materialize row views once for command and post-step advance."""
    return [
        _slice_control_state(state, env_id, num_envs)
        for env_id in range(num_envs)
    ]

def _control_state_to_host(state, num_envs):
    """Copy vectorized controller observations to host once per simulation step."""
    hosted = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == num_envs:
            hosted[key] = value.detach().cpu().numpy()
        else:
            hosted[key] = value
    return hosted

def _update_batched_reference_buffers(env, buffers):
    buffers["get_goal_reference_origin"] = _as_np(env.get_goal_reference_origin())
    buffers["get_goal_reference_quat"] = _as_np(env.get_goal_reference_quat())
    buffers["get_goal_center_offset_local"] = _as_np(
        env.get_goal_center_offset_local()
    )

class _BatchedEnvView:
    """Expose one row of a vectorized ManipLoco environment to a scalar controller."""

    _ROW_METHODS = {
        "get_goal_reference_origin",
        "get_goal_reference_quat",
        "get_goal_center_offset_local",
    }

    def __init__(self, env, env_id, reference_buffers=None, arm_mode_buffer=None):
        self._env = env
        self._env_id = int(env_id)
        self._reference_buffers = reference_buffers
        self._arm_mode_buffer = arm_mode_buffer

    def __getattr__(self, name):
        if name == "teleop_arm_joint_pos_targets":
            return self._env.teleop_arm_joint_pos_targets[
                self._env_id : self._env_id + 1
            ]
        attribute = getattr(self._env, name)
        if name in self._ROW_METHODS:
            if self._reference_buffers is not None:
                return lambda: self._reference_buffers[name][
                    self._env_id : self._env_id + 1
                ]
            return lambda: attribute()[self._env_id : self._env_id + 1]
        return attribute

    def _set_teleop_arm_control_mode(self, mode):
        if self._arm_mode_buffer is not None:
            joint_mode = mode == "joint"
            if bool(self._arm_mode_buffer[self._env_id]) == joint_mode:
                return
            self._arm_mode_buffer[self._env_id] = joint_mode
        self._env._set_teleop_arm_control_mode(mode, [self._env_id])

class _BatchedRunnerView:
    """Collect one controller's commands into shared vectorized teleop buffers."""

    def __init__(
        self,
        runner,
        env_id,
        state,
        reference_buffers=None,
        arm_mode_buffer=None,
        command_buffers=None,
    ):
        self._runner = runner
        self.env_id = int(env_id)
        self.env = _BatchedEnvView(
            runner.env, env_id, reference_buffers, arm_mode_buffer
        )
        self.state = state
        self.command_buffers = command_buffers

    @property
    def obs(self):
        return self._runner.obs

    @obs.setter
    def obs(self, value):
        self._runner.obs = value

    def set_state(self, state):
        self.state = state

    def get_control_state(self):
        return self.state

    def set_teleop_targets(
        self,
        base_command,
        ee_goal_cart=None,
        ee_orn_delta_rpy=None,
        gripper_target=None,
        ee_goal_orn_quat_world=None,
        ee_goal_orn_quat_world_enabled=None,
    ):
        env = self._runner.env
        row = slice(self.env_id, self.env_id + 1)
        base_command = _limit_base_commands(base_command)

        if self.command_buffers is not None:
            def _store(name, value):
                if value is not None:
                    target = self.command_buffers[name][row]
                    target[...] = _as_np(value).reshape(target.shape)

            _store("base", base_command)
            _store("ee_cart", ee_goal_cart)
            _store("ee_rpy", ee_orn_delta_rpy)
            _store("gripper", gripper_target)
            if ee_goal_orn_quat_world is not None:
                quat = _as_np(ee_goal_orn_quat_world).reshape(1, 4)
                quat /= max(float(np.linalg.norm(quat)), 1.0e-8)
                self.command_buffers["ee_quat"][row] = quat
                if ee_goal_orn_quat_world_enabled is None:
                    self.command_buffers["ee_quat_enabled"][row] = True
            if ee_goal_orn_quat_world_enabled is not None:
                self.command_buffers["ee_quat_enabled"][row] = bool(
                    np.asarray(ee_goal_orn_quat_world_enabled).reshape(-1)[0]
                )
            return

        def _copy(target, value):
            if value is not None:
                target[row] = torch.as_tensor(
                    value, device=env.device, dtype=target.dtype
                ).reshape_as(target[row])

        _copy(env.teleop_raw_commands, base_command)
        _copy(env.teleop_raw_ee_goal_cart, ee_goal_cart)
        _copy(env.teleop_raw_ee_goal_orn_delta_rpy, ee_orn_delta_rpy)
        _copy(env.gripper_pos_targets, gripper_target)
        if ee_goal_orn_quat_world is not None:
            quat = torch.as_tensor(
                ee_goal_orn_quat_world, device=env.device, dtype=torch.float32
            ).reshape(1, 4)
            quat = fixed_normalize(quat, eps=1.0e-8)
            env.teleop_world_ee_goal_orn_quat[row] = quat
            if ee_goal_orn_quat_world_enabled is None:
                env.teleop_world_ee_goal_orn_quat_enabled[row] = True
        if ee_goal_orn_quat_world_enabled is not None:
            env.teleop_world_ee_goal_orn_quat_enabled[row] = bool(
                torch.as_tensor(ee_goal_orn_quat_world_enabled).reshape(-1)[0]
            )
        env._teleop_inputs_dirty = True

def _make_batched_command_buffers(env):
    return {
        "base": _as_np(env.teleop_raw_commands),
        "ee_cart": _as_np(env.teleop_raw_ee_goal_cart),
        "ee_rpy": _as_np(env.teleop_raw_ee_goal_orn_delta_rpy),
        "gripper": _as_np(env.gripper_pos_targets),
        "ee_quat": _as_np(env.teleop_world_ee_goal_orn_quat),
        "ee_quat_enabled": env.teleop_world_ee_goal_orn_quat_enabled.detach().cpu().numpy().copy(),
    }

def _write_batched_episode_result(
    args, controller, runner, state, env_id, seed, step_count, max_finger_force,
    max_abs_hinge_speed, initial_root_state, initial_dof_position,
    initial_policy_observation, first_low_level_action,
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
        int(record["rejected_before_acceptance"])
        for record in arm_retract_records
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
            "position_radius_m": float(
                controller.arm_retract_position_noise_radius_m
            ),
            "rotation_bound_deg": float(
                controller.arm_retract_rotation_noise_deg
            ),
            "entries": arm_retract_records,
            "evaluated_candidates": arm_retract_evaluated,
            "unreachable_candidates": arm_retract_unreachable,
            "rejected_before_acceptance": arm_retract_resampled,
            "unreachable_frequency": (
                arm_retract_unreachable / arm_retract_evaluated
                if arm_retract_evaluated
                else 0.0
            ),
        },
        "camera": {"enabled": False, "names": [], "frame_count": {}},
        "scene": {
            "door_prim_path": door_path,
            "robot_side": controller.robot_side,
            "door_randomization": door_randomization,
        },
        "robot_state": {
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
            "steps": int(step_count),
            "max_finger_force_n": float(max_finger_force),
            "max_abs_hinge_speed_deg_s": math.degrees(max_abs_hinge_speed),
            "summary_only": True,
        },
    }
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


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
    reasons = () if success else (
        f"autoopen_failure:{controller.failure_stage or controller.stage.name}:"
        f"{controller.failure_reason or 'unknown'}",
    )
    episode = collector.finish(
        acceptance_reasons=reasons,
        provenance={
            "seed": episode_seed,
            "initial_root_pose_w": initial_root_state.tolist(),
            "initial_dof_position": initial_dof_position.tolist(),
            "initial_gripper_position": float(initial_gripper_position),
            "door_randomization": door_randomization,
            "door_asset": door_asset_snapshot,
            "camera_mount_randomization": dict(
                getattr(args, "camera_mount_randomizations", {}).get(
                    int(episode_seed), args.camera_mount_randomization
                )
                or {}
            ),
            "contact_audit": collector.contact_view_description,
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
        simulation_provenance=_standard_simulation_provenance(
            args, runner, episode_seed
        ),
    )
    writer.write_episode(episode)
    writer.finalize()




def _run_tensor_fsm_loop(
    args,
    *,
    stage_enum,
    runner,
    controllers,
    views,
    geometry_cache,
    device_state,
    steady_path_executor,
    approach_command_targets,
    approach_nominal_targets,
    initial_root_states,
    initial_dof_positions,
    initial_policy_observations,
    initial_gripper_positions,
    reference_buffers,
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
    )
    buffers = TensorCommandBuffers.from_env(env)
    planner = TensorPlanningCoordinator(
        controllers, views, steady_path_executor, Stage
    )
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
    step_counts = torch.zeros(num_envs, dtype=torch.int32, device=env.device)
    max_finger_forces = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    max_hinge_speeds = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    first_actions = torch.full(
        (num_envs, env.num_actions), torch.nan, dtype=torch.float32, device=env.device
    )

    for step in range(args.max_steps):
        active = (fsm.stage != int(Stage.SUCCESS)) & (
            fsm.stage != int(Stage.FAILURE)
        )
        if not bool(torch.any(active).item()):
            break
        step_counts[active] = step + 1

        planning_start = time.perf_counter()
        contact_rows = fsm.contact_plan_requests()
        retract_rows = fsm.push_retract_requests(device_state)
        if len(contact_rows) or len(retract_rows):
            _update_batched_reference_buffers(env, reference_buffers)
        if len(contact_rows):
            planner.plan_contact(fsm, device_state, contact_rows)
        if len(retract_rows):
            planner.plan_push_retract(fsm, device_state, retract_rows)
        timing["planning_s"] += time.perf_counter() - planning_start

        grasp_rows = torch.where(
            (fsm.stage == int(Stage.CENTER_HANDLE))
            | (fsm.stage == int(Stage.GRASP_HANDLE))
        )[0]
        geometry_start = time.perf_counter()
        grasp_geometry = _batched_geometry_metrics_gpu(
            device_state,
            geometry_cache,
            grasp_rows,
            as_numpy=False,
        ) if len(grasp_rows) else None
        timing["geometry_gpu_s"] += time.perf_counter() - geometry_start
        panel_rows = torch.where(
            (fsm.stage == int(Stage.CONTACT_PANEL))
            | (fsm.stage == int(Stage.PUSH_OPEN))
            | (fsm.stage == int(Stage.PUSH_ARM_RETRACT))
            | (fsm.stage == int(Stage.BASE_ADVANCE))
        )[0]
        clearance = torch.full(
            (num_envs,), torch.nan, dtype=torch.float32, device=env.device
        )
        if len(panel_rows):
            panel_start = time.perf_counter()
            clearance[panel_rows] = _batched_panel_finger_clearance_gpu(
                device_state, geometry_cache, panel_rows
            )
            timing["panel_clearance_backend_s"] += time.perf_counter() - panel_start

        forces = device_state.get("finger_contact_forces_w")
        if forces is not None:
            force = fixed_norm(forces).amax(dim=-1)
            max_finger_forces = torch.maximum(max_finger_forces, force)
            unsafe = active & (force > 600.0)
            fsm.fail(torch.where(unsafe)[0], 2)

        command_start = time.perf_counter()
        fsm.command(
            device_state,
            buffers,
            grasp_geometry=grasp_geometry,
            finger_clearance=clearance,
        )
        timing["command_s"] += time.perf_counter() - command_start
        prepare_start = time.perf_counter()
        buffers.flush(env)
        env._clip_teleop_targets()
        env._update_effective_teleop_inputs(force=True)
        timing["teleop_prepare_s"] += time.perf_counter() - prepare_start

        capture_env_ids = []
        capture_state_host = None
        if standard_collectors:
            standard_capture_start = time.perf_counter()
            capture_env_ids = torch.where(active)[0].detach().cpu().tolist()
            capture_state_host = _control_state_to_host(device_state, num_envs)
            capture_states = _slice_all_control_states(
                capture_state_host, num_envs
            )
            high_batch = torch.cat(
                (buffers.base, buffers.ee_cart, buffers.ee_rpy, buffers.gripper),
                dim=1,
            ).detach().cpu().numpy()
            stages_before_step = fsm.stage.detach().cpu().numpy()
            standard_collectors[0].contact_audit.begin_tick()
            for env_id in capture_env_ids:
                standard_collectors[env_id].capture_before_step(
                    capture_states[env_id],
                    high_batch[env_id],
                    stage_name=Stage(int(stages_before_step[env_id])).name,
                    begin_contact_tick=False,
                )
            timing["standard_capture_s"] += (
                time.perf_counter() - standard_capture_start
            )

        physics_start = time.perf_counter()
        result = runner.step_once()
        timing["policy_physics_s"] += time.perf_counter() - physics_start
        if step == 0:
            first_actions[:] = result["low_level_action"]
        if standard_collectors:
            standard_complete_start = time.perf_counter()
            contacts_by_env = standard_collectors[0].contact_audit.end_tick_by_env()
            post_state_host = _control_state_to_host(result["state"], num_envs)
            post_states = _slice_all_control_states(
                post_state_host, num_envs
            )
            target_poses = batched_target_height_invariant_poses(
                capture_state_host["root_states"],
                post_state_host["arm_position_target"],
            )
            joint_modes = (
                env.teleop_arm_joint_control_mask.detach().cpu().numpy()
            )
            for env_id in capture_env_ids:
                standard_collectors[env_id].complete_after_step(
                    {"state": post_states[env_id]},
                    contacts=contacts_by_env.get(env_id, ()),
                    target_height_invariant_pose=target_poses[env_id],
                    joint_mode=joint_modes[env_id],
                )
            timing["standard_complete_s"] += (
                time.perf_counter() - standard_complete_start
            )
        elif args.record_video:
            manager = runner.rgb_camera_manager
            if manager is None or not manager.enabled:
                raise RuntimeError("video regression requires enabled RGB cameras")
            timestamp = step * float(env.dt)
            for env_id in torch.where(active)[0].detach().cpu().tolist():
                images_by_name = manager.render_images(
                    env_id=env_id, update=False
                )
                for name, images in images_by_name.items():
                    if len(images) != 1:
                        raise RuntimeError(
                            f"camera {name} returned {len(images)} frames for "
                            f"environment {env_id}"
                        )
                    video_frames.setdefault(env_id, {}).setdefault(
                        name, []
                    ).append(images[0])
                    video_timestamps.setdefault(env_id, {}).setdefault(
                        name, []
                    ).append(timestamp)
        device_state = result["state"]
        hinge_speed = device_state["door_joint_velocity"][:, fsm.hinge_idx].abs()
        max_hinge_speeds = torch.maximum(max_hinge_speeds, hinge_speed)

        grasp_rows = torch.where(
            (fsm.stage == int(Stage.CENTER_HANDLE))
            | (fsm.stage == int(Stage.GRASP_HANDLE))
        )[0]
        geometry_start = time.perf_counter()
        grasp_geometry = _batched_geometry_metrics_gpu(
            device_state,
            geometry_cache,
            grasp_rows,
            as_numpy=False,
        ) if len(grasp_rows) else None
        timing["geometry_gpu_s"] += time.perf_counter() - geometry_start
        panel_rows = torch.where(
            (fsm.stage == int(Stage.CONTACT_PANEL))
            | (fsm.stage == int(Stage.PUSH_OPEN))
            | (fsm.stage == int(Stage.PUSH_ARM_RETRACT))
            | (fsm.stage == int(Stage.BASE_ADVANCE))
        )[0]
        clearance = torch.full(
            (num_envs,), torch.nan, dtype=torch.float32, device=env.device
        )
        if len(panel_rows):
            clearance[panel_rows] = _batched_panel_finger_clearance_gpu(
                device_state, geometry_cache, panel_rows
            )

        advance_start = time.perf_counter()
        events = fsm.advance(
            device_state,
            grasp_geometry=grasp_geometry,
            finger_clearance=clearance,
        )
        timing["advance_s"] += time.perf_counter() - advance_start
        planning_start = time.perf_counter()
        has_planning_event = any(
            len(events[name])
            for name in (
                "pregrasp_plan",
                "center_plan",
                "release_open_plan",
                "retreat_plan",
                "restore_plan",
            )
        )
        if has_planning_event:
            _update_batched_reference_buffers(env, reference_buffers)
        planner.plan_pregrasp(fsm, device_state, events["pregrasp_plan"])
        planner.plan_center(fsm, device_state, events["center_plan"])
        planner.plan_release_open(fsm, device_state, events["release_open_plan"])
        planner.plan_retreat(fsm, device_state, events["retreat_plan"])
        planner.plan_restore(fsm, device_state, events["restore_plan"])
        timing["planning_s"] += time.perf_counter() - planning_start

    final_state = _control_state_to_host(device_state, num_envs)
    state_rows = _slice_all_control_states(final_state, num_envs)
    stages = fsm.stage.detach().cpu().numpy()
    stage_steps = fsm.stage_steps.detach().cpu().numpy()
    failure_codes = fsm.failure_code.detach().cpu().numpy()
    failure_from_stages = fsm.failure_from_stage.detach().cpu().numpy()
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
    for env_id, controller in enumerate(controllers):
        controller.stage = Stage(int(stages[env_id]))
        controller.stage_steps = int(stage_steps[env_id])
        if controller.stage not in (Stage.SUCCESS, Stage.FAILURE):
            controller.stage = Stage.FAILURE
            controller.failure_reason = f"max_steps:{args.max_steps}"
        elif controller.stage == Stage.FAILURE and controller.failure_reason is None:
            origin = int(failure_from_stages[env_id])
            controller.failure_reason = f"tensor_fsm_failure_code:{int(failure_codes[env_id])}"
            controller.failure_stage = (
                Stage(origin).name if origin else Stage.FAILURE.name
            )
        root = final_state["root_states"][env_id]
        controller.last_acceptance_diagnostics.update(
            {
                "base_arrival_settle_steps": int(arrival_settle_np[env_id]),
                "base_arrival_max_settle_steps": int(arrival_max_settle_np[env_id]),
                "base_arrival_distance_m": float(arrival_distance_np[env_id]),
                "base_arrival_yaw_error_rad": float(arrival_yaw_np[env_id]),
                "base_arrival_linear_speed_m_s": float(np.linalg.norm(root[7:10])),
                "base_arrival_angular_speed_rad_s": float(np.linalg.norm(root[10:13])),
                "base_arrival_planar_speed_m_s": float(np.linalg.norm(root[7:9])),
                "base_arrival_yaw_rate_rad_s": float(abs(root[12])),
                "base_arrival_velocity_xyz_m_s": root[7:10].astype(float).tolist(),
                "base_arrival_angular_velocity_xyz_rad_s": root[10:13]
                .astype(float)
                .tolist(),
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
            video_dir = (
                args.output_dir / f"seed_{int(episode_seed)}" / "video"
            )
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
                dataset_dir=(
                    Path(args.standard_dataset_dir)
                    / f"episode-seed{episode_seed}"
                ),
                initial_root_state=initial_root_states[env_id],
                initial_dof_position=initial_dof_positions[env_id],
                initial_gripper_position=initial_gripper_positions[env_id],
                door_randomization=standard_door_randomizations[env_id],
                door_asset_snapshot=standard_door_asset_snapshots[env_id],
            )
        standard_collectors[0].contact_audit.detach()
        timing["standard_finalize_s"] += (
            time.perf_counter() - standard_finalize_start
        )
        timing.update(
            {
                f"collector_{name}": float(value)
                for name, value in {
                    key: sum(float(c.timing.get(key, 0.0)) for c in standard_collectors)
                    for key in {
                        key for c in standard_collectors for key in c.timing
                    }
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
        "panel_terminal_diagnostics": {
            "finger_clearance_m": fsm.panel_last_finger_clearance.detach().cpu().tolist(),
            "lateral_error_m": fsm.panel_last_lateral_error.detach().cpu().tolist(),
            "axis_error_rad": fsm.panel_last_axis_error.detach().cpu().tolist(),
            "ee_speed_m_s": fsm.panel_last_ee_speed.detach().cpu().tolist(),
            "settle_steps": fsm.panel_contact_settle_steps.detach().cpu().tolist(),
        },
        "push_terminal_diagnostics": {
            "hinge_rad": device_state["door_joint_position"][:, fsm.hinge_idx].detach().cpu().tolist(),
            "hinge_speed_rad_s": device_state["door_joint_velocity"][:, fsm.hinge_idx].detach().cpu().tolist(),
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
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        "[doorman][batch-timing] "
        + " ".join(f"{name}={value:.3f}" for name, value in timing.items()),
        flush=True,
    )
    print(f"[doorman][batch] complete: {successes}/{num_envs} active", flush=True)
    if steady_path_executor is not None:
        steady_path_executor.shutdown(
            wait=not args.headless,
            cancel_futures=bool(args.headless),
        )
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
            {
                env_id: configs[int(seed)]
                for env_id, seed in enumerate(episode_seeds)
            }
        )
    door_handle_paths = [
        str(prim.GetPath())
        for prim in env.sim.stage.Traverse()
        if prim.GetName() == "door_handle"
        and str(prim.GetPath()).startswith("/World/envs/")
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
    shared_steady_model = B2Z1SteadyEeForwardModel(
        LOWLEVEL_ROOT
        / "resources/robots/b2z1/urdf/generated"
        / "b2z1_mount_90_x0p22_y0_z0p17.urdf",
        stiffness=_arm_drive_stiffness(env),
        emit_tolerance_warnings=not args.summary_only,
    )
    steady_path_executor = None
    if args.steady_path_workers > 0:
        steady_path_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=int(args.steady_path_workers),
            mp_context=mp.get_context("spawn"),
            initializer=initialize_steady_path_worker,
            initargs=(
                str(shared_steady_model.urdf_path),
                shared_steady_model.stiffness,
            ),
        )
    if (
        (
            args.batch_contact_ik_backend == "exact_workers"
            or args.batch_path_backend == "exact_workers"
        )
        and steady_path_executor is None
    ):
        raise ValueError(
            "exact worker backends require "
            "--steady_path_workers greater than zero"
        )
    views = []
    controllers = []
    reference_buffers = {}
    _update_batched_reference_buffers(env, reference_buffers)
    command_buffers = _make_batched_command_buffers(env)
    arm_mode_buffer = (
        env.teleop_arm_joint_control_mask.detach().cpu().numpy().copy()
    )
    shared_robot_geometry = None
    controller_stdout = open(os.devnull, "w") if args.summary_only else None
    with (
        contextlib.redirect_stdout(controller_stdout)
        if controller_stdout is not None
        else contextlib.nullcontext()
    ):
        for env_id in range(num_envs):
            sliced_state = state_rows[env_id]
            view = _BatchedRunnerView(
                runner,
                env_id,
                sliced_state,
                reference_buffers=reference_buffers,
                arm_mode_buffer=arm_mode_buffer,
                command_buffers=command_buffers,
            )
            controller = DoorController(
                view,
                robot_side=args.robot_side,
                position_only_center=args.position_only_center,
                disable_push_base_assist=args.disable_push_base_assist,
                env_id=env_id,
                shared_steady_model=shared_steady_model,
                shared_robot_geometry=shared_robot_geometry,
                target_noise_seed=episode_seeds[env_id],
                target_noise_config=args.target_noise_config,
                arm_retract_pose_local=args.arm_retract_pose_local,
                arm_retract_position_noise_radius_m=(
                    args.arm_retract_position_noise_radius_m
                ),
                arm_retract_rotation_noise_deg=args.arm_retract_rotation_noise_deg,
            )
            controller.defer_batched_orientation_delta = True
            reset_controller(
                controller,
                state=sliced_state,
                reset_env=False,
                close_gripper=False,
            )
            views.append(view)
            controllers.append(controller)
            if shared_robot_geometry is None:
                shared_robot_geometry = controller.center_geometry
    geometry_cache = _make_batched_geometry_cache(controllers, env.device)

    # DoorController.reset() measures the open-jaw geometry and then closes
    # the jaw before a scalar episode starts.  Do that once for the complete
    # vectorized batch so every environment has the same initial semantics.
    gripper_joint_id = int(env.gripper_joint_ids[0].item())
    joint_pos = env.robot.data.joint_pos.clone()
    joint_vel = env.robot.data.joint_vel.clone()
    joint_pos[:, gripper_joint_id] = torch.as_tensor(
        [controller.q_closed for controller in controllers],
        device=env.device,
        dtype=joint_pos.dtype,
    )
    command_buffers["gripper"][:, 0] = np.asarray(
        [controller.q_closed for controller in controllers], dtype=np.float32
    )
    joint_vel[:, gripper_joint_id] = 0.0
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    env.robot.set_joint_position_target(joint_pos)
    runner.obs = env.get_observations()
    device_state = runner.get_control_state()
    full_state = _control_state_to_host(device_state, num_envs)
    handle_center_gpu_s = 0.0
    phase_start = time.perf_counter()
    handle_centers = _batched_handle_centers_gpu(full_state, geometry_cache)
    handle_center_gpu_s += time.perf_counter() - phase_start
    initial_root_states = _as_np(full_state["root_states"])[:, :7].copy()
    initial_dof_positions = _as_np(full_state["dof_pos"]).copy()
    initial_gripper_positions = _as_np(full_state["gripper_position"]).reshape(-1).copy()
    initial_policy_observations = _as_np(runner.obs).copy()
    first_low_level_actions = np.full(
        (num_envs, env.num_actions), np.nan, dtype=np.float32
    )
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
            shard_dir = (
                Path(args.standard_dataset_dir) / f"episode-seed{episode_seed}"
            )
            standard_collectors.append(
                AutoOpenStandardEpisodeCollector(
                    runner,
                    controllers[env_id],
                    episode_index=0,
                    seed=episode_seed,
                    video_staging_root=(
                        shard_dir / ".staging" / "episode_000000"
                    ),
                    capture_images=args.collection_camera_mode != "none",
                    camera_mode=args.collection_camera_mode,
                    env_id=env_id,
                    contact_audit=contact_audit,
                )
            )
            door_prim = env.sim.stage.GetPrimAtPath(
                _resolved_door_prim_path(env, env_id)
            )
            standard_door_randomizations.append(
                dict(door_prim.GetMetadata("customData") or {})
            )
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

    step_counts = np.zeros(num_envs, dtype=np.int32)
    max_finger_forces = np.zeros(num_envs, dtype=np.float32)
    max_abs_hinge_speeds = np.zeros(num_envs, dtype=np.float32)
    pending_metrics = np.zeros((num_envs, 2), dtype=np.float32)
    batch_timing = {
        "command_s": 0.0,
        "teleop_prepare_s": 0.0,
        "policy_physics_s": 0.0,
        "state_to_host_s": 0.0,
        "advance_s": 0.0,
        "steady_path_workers_s": 0.0,
        "geometry_gpu_s": 0.0,
        "geometry_numpy_backend_s": 0.0,
        "stability_gpu_s": 0.0,
        "orientation_gpu_s": 0.0,
        "contact_ik_backend_s": 0.0,
        "path_backend_s": 0.0,
        "panel_clearance_backend_s": 0.0,
    }
    contact_ik_backend_stats = {
        "batches": 0,
        "requests": 0,
        "worker_errors": 0,
    }
    path_backend_stats = {
        "batches": 0,
        "requests": 0,
        "worker_errors": 0,
        "labels": {},
    }
    geometry_numpy_backend_stats = {
        "batches": 0,
        "requests": 0,
        "accepted_requests": 0,
        "exact_scalar_requests": 0,
        "max_batch": 0,
    }
    orientation_gpu_calls = 0
    approach_nominal_targets = [
        _approach_base_target(controller) for controller in controllers
    ]
    approach_command_targets = [
        _perturb_base_target(controller, "approach", *target)
        for controller, target in zip(
            controllers, approach_nominal_targets, strict=True
        )
    ]
    if controller_stdout is not None:
        controller_stdout.close()
    return _run_tensor_fsm_loop(
        args,
        stage_enum=Stage,
        runner=runner,
        controllers=controllers,
        views=views,
        geometry_cache=geometry_cache,
        device_state=device_state,
        steady_path_executor=steady_path_executor,
        approach_command_targets=approach_command_targets,
        approach_nominal_targets=approach_nominal_targets,
        initial_root_states=initial_root_states,
        initial_dof_positions=initial_dof_positions,
        initial_policy_observations=initial_policy_observations,
        initial_gripper_positions=initial_gripper_positions,
        reference_buffers=reference_buffers,
        standard_collectors=standard_collectors,
        standard_door_randomizations=standard_door_randomizations,
        standard_door_asset_snapshots=standard_door_asset_snapshots,
        episode_seeds=episode_seeds,
        video_frames=video_frames,
        video_timestamps=video_timestamps,
    )
