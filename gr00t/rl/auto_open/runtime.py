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

from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy
from legged_gym.utils.ee_steady_state import B2Z1SteadyEeForwardModel
from legged_gym.utils.steady_path_worker import (
    initialize_steady_path_worker,
    solve_deferred_path_request,
    solve_steady_inverse_request,
)
from gr00t.rl.auto_open.cli import LOWLEVEL_ROOT
from gr00t.rl.auto_open.geometry import (
    CENTER_HANDLE_MIN_AXIAL_OVERLAP,
    _batched_geometry_metrics_gpu,
    _batched_geometry_metrics_numpy,
    _batched_handle_centers_gpu,
    _batched_panel_finger_clearance_gpu,
    _center_handle_axial_overlap,
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
    BASE_YAW_TOLERANCE_RAD,
    _approach_base_target,
    _arm_drive_stiffness,
    _base_commands_exact,
    _limit_base_commands,
    _perturb_base_target,
    _perturb_ee_target,
)
from gr00t.rl.auto_open.utils import as_numpy as _as_np
from gr00t.rl.auto_open.planning import (
    DeferredPathSolve,
    DeferredSteadyInverse,
    _apply_pregrasp_candidate,
    _prepare_panel_contact_command,
    _pregrasp_pitch_candidates,
    _steady_inverse_worker_request,
    _steady_path_request,
)


def _numpy_geometry_requires_exact_scalar(
    controller, metrics, center_handle_stage
):
    """Keep approximate batch values away from control-sensitive boundaries."""
    if controller.stage != center_handle_stage:
        return True
    numerical_guard_m = 1.0e-4
    boundaries = (
        ("pocket_normal_error", 0.005, True),
        ("channel_vertical_error", 0.010, True),
        ("grasp_depth", 0.015, False),
        ("panel_clearance", 0.005, False),
    )
    for name, threshold, absolute_value in boundaries:
        value = float(metrics[name])
        if not math.isfinite(value):
            return True
        if absolute_value:
            value = abs(value)
        if abs(value - threshold) <= numerical_guard_m:
            return True
    axial_overlap = _center_handle_axial_overlap(controller, metrics)
    if not math.isfinite(axial_overlap):
        return True
    if (
        abs(axial_overlap - CENTER_HANDLE_MIN_AXIAL_OVERLAP)
        <= numerical_guard_m
    ):
        return True
    return False

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

    def begin_cartesian_transition(self):
        self._env.begin_cartesian_transition([self._env_id])

    def cartesian_transition_is_active(self):
        return self._env.cartesian_transition_is_active([self._env_id])

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
            quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
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

def _flush_batched_command_buffers(env, buffers):
    for target, name in (
        (env.teleop_raw_commands, "base"),
        (env.teleop_raw_ee_goal_cart, "ee_cart"),
        (env.teleop_raw_ee_goal_orn_delta_rpy, "ee_rpy"),
        (env.gripper_pos_targets, "gripper"),
        (env.teleop_world_ee_goal_orn_quat, "ee_quat"),
        (env.teleop_world_ee_goal_orn_quat_enabled, "ee_quat_enabled"),
    ):
        target.copy_(torch.as_tensor(buffers[name], device=env.device, dtype=target.dtype))
    env._teleop_inputs_dirty = True

def _batched_grasp_orientation_delta_gpu(
    command_buffers, reference_buffers, arm_induced_pitch, env_ids, device
):
    """Compute teleop grasp-orientation RPY deltas for selected envs on GPU."""
    if not env_ids:
        return np.empty((0, 3), dtype=np.float32)
    rows = np.asarray(env_ids, dtype=np.int64)
    cart = torch.as_tensor(
        command_buffers["ee_cart"][rows], device=device, dtype=torch.float32
    )
    desired = torch.as_tensor(
        command_buffers["ee_quat"][rows], device=device, dtype=torch.float32
    )
    reference = torch.as_tensor(
        reference_buffers["get_goal_reference_quat"][rows],
        device=device,
        dtype=torch.float32,
    ).clone()
    reference[:, 1:] *= -1.0
    lw, lx, ly, lz = reference.unbind(dim=-1)
    rw, rx, ry, rz = desired.unbind(dim=-1)
    local = torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )
    w, x, y, z = local.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    radius = torch.linalg.vector_norm(cart, dim=-1).clamp_min(1.0e-8)
    sphere_pitch = torch.asin((cart[:, 2] / radius).clamp(-1.0, 1.0))
    sphere_yaw = torch.atan2(cart[:, 1], cart[:, 0])
    induced_pitch = torch.as_tensor(
        np.asarray(arm_induced_pitch, dtype=np.float32)[rows],
        device=device,
        dtype=torch.float32,
    )
    two_pi = 2.0 * math.pi
    result = torch.stack(
        (
            torch.remainder(roll + math.pi / 2.0 + math.pi, two_pi) - math.pi,
            torch.remainder(pitch + sphere_pitch - induced_pitch + math.pi, two_pi)
            - math.pi,
            torch.remainder(yaw - sphere_yaw + math.pi, two_pi) - math.pi,
        ),
        dim=-1,
    )
    return result.detach().cpu().numpy()


def _merge_tolerance_warning_stats(total, update):
    count = int(update.get("count", 0))
    if count == 0:
        return total
    total["count"] = int(total.get("count", 0)) + count
    for name in (
        "position_error_m",
        "orientation_error",
        "equilibrium_error_nm",
    ):
        low = float(update[f"{name}_min"])
        high = float(update[f"{name}_max"])
        total[f"{name}_min"] = min(total.get(f"{name}_min", math.inf), low)
        total[f"{name}_max"] = max(total.get(f"{name}_max", 0.0), high)
    return total

class _BatchedStabilityTracker:
    """GPU mirror of the per-controller 15-sample EE stability deque."""

    def __init__(self, num_envs, device, window_size=15):
        self.window_size = int(window_size)
        self.positions = torch.zeros(
            (num_envs, self.window_size, 3), dtype=torch.float32, device=device
        )
        self.orientations = torch.zeros(
            (num_envs, self.window_size, 4), dtype=torch.float32, device=device
        )
        self.counts = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update(self, state, controllers, env_ids):
        if not env_ids:
            return {}
        device = self.positions.device
        rows = torch.as_tensor(env_ids, dtype=torch.long, device=device)
        cleared = [
            env_id
            for env_id in env_ids
            if len(controllers[env_id].arm_stability_window) == 0
        ]
        if cleared:
            cleared_rows = torch.as_tensor(cleared, dtype=torch.long, device=device)
            self.counts[cleared_rows] = 0
        self.positions[rows] = torch.roll(self.positions[rows], -1, dims=1)
        self.orientations[rows] = torch.roll(
            self.orientations[rows], -1, dims=1
        )
        self.positions[rows, -1] = state["ee_pos"][rows]
        self.orientations[rows, -1] = state["ee_orn"][rows]
        self.counts[rows] = torch.clamp(self.counts[rows] + 1, max=self.window_size)

        positions = self.positions[rows]
        position_span = torch.linalg.vector_norm(
            positions - positions.mean(dim=1, keepdim=True), dim=-1
        ).amax(dim=1)
        orientations = self.orientations[rows]
        orientations = orientations / torch.linalg.vector_norm(
            orientations, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        current = orientations[:, -1]
        dots = torch.abs(torch.sum(orientations * current[:, None, :], dim=-1))
        orientation_span = (2.0 * torch.acos(dots.clamp(0.0, 1.0))).amax(dim=1)
        enough = self.counts[rows] >= self.window_size
        stable = enough & (position_span <= 0.001) & (orientation_span <= 0.003)
        position_span = position_span.cpu().numpy()
        orientation_span = orientation_span.cpu().numpy()
        enough = enough.cpu().numpy()
        stable = stable.cpu().numpy()
        return {
            env_id: {
                "enough": bool(enough[row]),
                "position_span": float(position_span[row]),
                "orientation_span": float(orientation_span[row]),
                "stable": bool(stable[row]),
            }
            for row, env_id in enumerate(env_ids)
        }

def _write_batched_episode_result(
    args, controller, runner, state, env_id, seed, step_count, max_finger_force,
    max_abs_hinge_speed, initial_root_state, initial_dof_position,
):
    episode_dir = args.output_dir / f"seed_{seed}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    door_path = _resolved_door_prim_path(runner.env, env_id)
    door_prim = runner.env.sim.stage.GetPrimAtPath(door_path)
    door_randomization = dict(door_prim.GetMetadata("customData") or {})
    arm_joint_ids = _as_np(runner.env.arm_joint_ids).astype(int)
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
        "camera": {"enabled": False, "names": [], "frame_count": {}},
        "scene": {
            "door_prim_path": door_path,
            "robot_side": controller.robot_side,
            "door_randomization": door_randomization,
        },
        "robot_state": {
            "initial_base_pose_w": initial_root_state.tolist(),
            "initial_arm_joint_position": initial_dof_position[arm_joint_ids].tolist(),
            "initial_gripper_position": float(
                initial_dof_position[int(runner.env.gripper_joint_ids[0].item())]
            ),
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
    if args.batch_active_env_ids is None:
        active_env_ids = tuple(range(num_envs))
    else:
        active_env_ids = tuple(
            sorted(
                {
                    int(value.strip())
                    for value in args.batch_active_env_ids.split(",")
                    if value.strip()
                }
            )
        )
        if not active_env_ids:
            raise ValueError("--batch_active_env_ids must select at least one env")
        if active_env_ids[0] < 0 or active_env_ids[-1] >= num_envs:
            raise ValueError(
                "--batch_active_env_ids contains an id outside --batched_envs"
            )
    active_env_mask = np.zeros(num_envs, dtype=bool)
    active_env_mask[list(active_env_ids)] = True
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    runner = ManipLoco_Policy(args)
    env = runner.env
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
    steady_tolerance_stats = {"count": 0}
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
                target_noise_seed=base_seed + env_id,
                target_noise_config=args.target_noise_config,
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
    stability_tracker = _BatchedStabilityTracker(num_envs, env.device)

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
    standard_collector = None
    standard_door_randomization = None
    standard_door_asset_snapshot = None
    video_frames = {}
    video_timestamps = {}
    if args.record_video and num_envs != 1:
        raise ValueError("video regression uses the unified runner with batch size one")
    if args.standard_dataset_dir is not None:
        if num_envs != 1 or active_env_ids != (0,):
            raise ValueError(
                "standard collection uses the unified runner with exactly one active environment"
            )
        from gr00t.rl.data_collection.auto_open_collector import (
            AutoOpenStandardEpisodeCollector,
        )

        standard_collector = AutoOpenStandardEpisodeCollector(
            runner,
            controllers[0],
            episode_index=0,
            seed=base_seed,
            video_staging_root=(
                Path(args.standard_dataset_dir) / ".staging" / "episode_000000"
            ),
        )
        door_prim = env.sim.stage.GetPrimAtPath(
            _resolved_door_prim_path(env, 0)
        )
        standard_door_randomization = dict(
            door_prim.GetMetadata("customData") or {}
        )
        standard_door_asset_snapshot = _door_asset_stage_snapshot(
            env.sim.stage, door_prim
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
    arm_induced_pitch = np.asarray(
        [controller.env.arm_induced_pitch for controller in controllers],
        dtype=np.float64,
    )
    approach_nominal_targets = [
        _approach_base_target(controller) for controller in controllers
    ]
    approach_command_targets = [
        _perturb_base_target(controller, "approach", *target)
        for controller, target in zip(
            controllers, approach_nominal_targets, strict=True
        )
    ]
    station_xy = np.stack([target[0] for target in approach_command_targets])
    station_yaw = np.asarray(
        [target[1] for target in approach_command_targets], dtype=np.float64
    )
    nominal_station_xy = np.stack(
        [target[0] for target in approach_nominal_targets]
    )
    nominal_station_yaw = np.asarray(
        [target[1] for target in approach_nominal_targets], dtype=np.float64
    )
    base_command_exact_s = 0.0
    pending_high_actions = [None] * num_envs
    for step in range(args.max_steps):
        pending_high_actions = [None] * num_envs
        phase_start = time.perf_counter()
        batched_base_command = _base_commands_exact(
            full_state,
            station_xy,
            station_yaw,
            nominal_station_xy,
            nominal_station_yaw,
        )
        base_command_exact_s += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        active = False
        deferred_contact_requests = []
        if args.batch_panel_clearance_backend == "gpu":
            for controller in controllers:
                controller.batched_panel_finger_clearance = None
            panel_clearance_env_ids = [
                env_id
                for env_id, controller in enumerate(controllers)
                if active_env_mask[env_id]
                and controller.stage
                in (
                    Stage.CONTACT_PANEL,
                    Stage.PUSH_OPEN,
                    Stage.PUSH_ARM_RETRACT,
                    Stage.BASE_ADVANCE,
                )
            ]
            if panel_clearance_env_ids:
                panel_start = time.perf_counter()
                panel_values = (
                    _batched_panel_finger_clearance_gpu(
                        device_state, geometry_cache, panel_clearance_env_ids
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                batch_timing["panel_clearance_backend_s"] += (
                    time.perf_counter() - panel_start
                )
                for env_id, value in zip(
                    panel_clearance_env_ids, panel_values, strict=True
                ):
                    controllers[env_id].batched_panel_finger_clearance = float(value)
        for env_id, (view, controller) in enumerate(zip(views, controllers)):
            sliced_state = state_rows[env_id]
            view.set_state(sliced_state)
            controller.batched_handle_center = handle_centers[env_id]
            controller.batched_default_base_command = tuple(
                value[env_id] for value in batched_base_command
            )
            if not active_env_mask[env_id]:
                command_buffers["base"][env_id] = 0.0
                continue
            if controller.stage in (Stage.SUCCESS, Stage.FAILURE):
                command_buffers["base"][env_id] = 0.0
                continue
            active = True
            step_counts[env_id] = step + 1
            finger_force = sliced_state.get("finger_contact_forces_w")
            if finger_force is not None:
                force = float(np.linalg.norm(_as_np(finger_force)[0], axis=-1).max())
                max_finger_forces[env_id] = max(max_finger_forces[env_id], force)
                if force > 600.0:
                    controller.failure_reason = f"safety_force_limit:{force:.1f}N"
                    controller.failure_stage = controller.stage.name
                    controller._transition_to(Stage.FAILURE, reset_steps=False)
                    command_buffers["base"][env_id] = 0.0
                    continue
            try:
                with (
                    contextlib.redirect_stdout(controller_stdout)
                    if controller_stdout is not None
                    else contextlib.nullcontext()
                ):
                    controller.defer_steady_inverse = (
                        args.batch_contact_ik_backend == "exact_workers"
                    )
                    if (
                        controller.stage == Stage.CONTACT_PANEL
                        and controller.defer_steady_inverse
                    ):
                        _prepare_panel_contact_command(controller, sliced_state)
                    high, dist, yaw_error = controller.command(sliced_state)
                    pending_high_actions[env_id] = high
            except DeferredSteadyInverse as request:
                deferred_contact_requests.append((env_id, request))
                continue
            except RuntimeError as exc:
                controller.failure_reason = f"controller_runtime_error:{exc}"
                controller.failure_stage = controller.stage.name
                controller._transition_to(Stage.FAILURE, reset_steps=False)
                command_buffers["base"][env_id] = 0.0
                continue
            finally:
                controller.defer_steady_inverse = False
            pending_metrics[env_id] = (dist, yaw_error)


        if deferred_contact_requests:
            requests = [item[1] for item in deferred_contact_requests]
            backend_start = time.perf_counter()
            worker_results = list(
                steady_path_executor.map(
                    solve_steady_inverse_request,
                    [_steady_inverse_worker_request(request) for request in requests],
                )
            )
            batch_timing["contact_ik_backend_s"] += (
                time.perf_counter() - backend_start
            )
            contact_ik_backend_stats["batches"] += 1
            contact_ik_backend_stats["requests"] += len(requests)

            for (env_id, request), worker_result in zip(
                deferred_contact_requests, worker_results, strict=True
            ):
                controller = request.controller
                _merge_tolerance_warning_stats(
                    steady_tolerance_stats,
                    worker_result["warning_stats"],
                )
                if worker_result["error"] is not None:
                    contact_ik_backend_stats["worker_errors"] += 1
                    controller.failure_reason = (
                        "controller_runtime_error:" + worker_result["error"]
                    )
                    controller.failure_stage = controller.stage.name
                    controller._transition_to(Stage.FAILURE, reset_steps=False)
                    command_buffers["base"][env_id] = 0.0
                    continue
                controller.panel_contact_inverse_command_world = np.asarray(
                    worker_result["command_position"], dtype=np.float32
                ).copy()
                controller.panel_contact_inverse_command_orientation_world = (
                    np.asarray(
                        worker_result["command_orientation"], dtype=np.float32
                    ).copy()
                )
                controller.panel_contact_inverse_target_world = (
                    request.desired_position_world.astype(np.float32)
                )
                controller.panel_contact_inverse_orientation_world = (
                    request.desired_orientation_world.astype(np.float32)
                )
                controller.prepared_panel_contact_command = (
                    controller.panel_contact_inverse_command_world.copy(),
                    controller.panel_contact_inverse_command_orientation_world.copy(),
                )
                sliced_state = state_rows[env_id]
                try:
                    with (
                        contextlib.redirect_stdout(controller_stdout)
                        if controller_stdout is not None
                        else contextlib.nullcontext()
                    ):
                        high, dist, yaw_error = controller.command(
                            sliced_state
                        )
                        pending_high_actions[env_id] = high
                except RuntimeError as exc:
                    controller.failure_reason = f"controller_runtime_error:{exc}"
                    controller.failure_stage = controller.stage.name
                    controller._transition_to(Stage.FAILURE, reset_steps=False)
                    command_buffers["base"][env_id] = 0.0
                    continue
                pending_metrics[env_id] = (dist, yaw_error)
        batch_timing["command_s"] += time.perf_counter() - phase_start
        if not active:
            break

        orientation_env_ids = [
            env_id
            for env_id, controller in enumerate(controllers)
            if active_env_mask[env_id]
            and controller.stage not in (Stage.SUCCESS, Stage.FAILURE)
        ]
        phase_start = time.perf_counter()
        orientation_delta = _batched_grasp_orientation_delta_gpu(
            command_buffers,
            reference_buffers,
            arm_induced_pitch,
            orientation_env_ids,
            env.device,
        )
        batch_timing["orientation_gpu_s"] += time.perf_counter() - phase_start
        if orientation_env_ids:
            orientation_gpu_calls += len(orientation_env_ids)
            command_buffers["ee_rpy"][orientation_env_ids] = orientation_delta

        phase_start = time.perf_counter()
        _flush_batched_command_buffers(env, command_buffers)
        env._clip_teleop_targets()
        env._update_effective_teleop_inputs(force=True)
        batch_timing["teleop_prepare_s"] += time.perf_counter() - phase_start
        if standard_collector is not None and pending_high_actions[0] is None:
            if controllers[0].stage in (Stage.SUCCESS, Stage.FAILURE):
                break
            raise RuntimeError(
                "unified standard collector has no command for an active frame"
            )
        if standard_collector is not None:
            high = pending_high_actions[0]
            standard_collector.capture_before_step(
                state_rows[0], high, stage_name=controllers[0].stage.name
            )
        phase_start = time.perf_counter()
        result = runner.step_once()
        if standard_collector is not None:
            standard_collector.complete_after_step(result)
        elif args.record_video:
            manager = runner.rgb_camera_manager
            if manager is None or not manager.enabled:
                raise RuntimeError("video regression requires enabled RGB cameras")
            images_by_name = manager.render_images(env_id=0, update=False)
            timestamp = step * float(env.dt)
            for name, images in images_by_name.items():
                if len(images) != 1:
                    raise RuntimeError(
                        f"camera {name} returned {len(images)} frames for one environment"
                    )
                video_frames.setdefault(name, []).append(images[0])
                video_timestamps.setdefault(name, []).append(timestamp)
        batch_timing["policy_physics_s"] += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        handle_centers = _batched_handle_centers_gpu(result["state"], geometry_cache)
        handle_center_gpu_s += time.perf_counter() - phase_start
        geometry_env_ids = [
            env_id
            for env_id, controller in enumerate(controllers)
            if controller.stage in (Stage.CENTER_HANDLE, Stage.GRASP_HANDLE)
        ]
        phase_start = time.perf_counter()
        geometry_metrics = _batched_geometry_metrics_gpu(
            result["state"], geometry_cache, geometry_env_ids
        )
        batch_timing["geometry_gpu_s"] += time.perf_counter() - phase_start
        geometry_rows = {
            env_id: row for row, env_id in enumerate(geometry_env_ids)
        }
        stability_env_ids = [
            env_id
            for env_id, controller in enumerate(controllers)
            if controller.stage
            in (
                Stage.PREGRASP,
                Stage.OPEN_GRIPPER,
                Stage.CENTER_HANDLE,
                Stage.RESTORE_GRIPPER,
            )
        ]
        phase_start = time.perf_counter()
        stability_results = stability_tracker.update(
            result["state"], controllers, stability_env_ids
        )
        batch_timing["stability_gpu_s"] += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        device_state = result["state"]
        full_state = _control_state_to_host(device_state, num_envs)
        state_rows = _slice_all_control_states(full_state, num_envs)
        _update_batched_reference_buffers(env, reference_buffers)
        batch_timing["state_to_host_s"] += time.perf_counter() - phase_start

        if args.batch_panel_clearance_backend == "gpu":
            for controller in controllers:
                controller.batched_panel_finger_clearance = None
            panel_clearance_env_ids = [
                env_id
                for env_id, controller in enumerate(controllers)
                if active_env_mask[env_id]
                and controller.stage
                in (
                    Stage.CONTACT_PANEL,
                    Stage.PUSH_OPEN,
                    Stage.PUSH_ARM_RETRACT,
                    Stage.BASE_ADVANCE,
                )
            ]
            if panel_clearance_env_ids:
                panel_start = time.perf_counter()
                panel_values = (
                    _batched_panel_finger_clearance_gpu(
                        device_state, geometry_cache, panel_clearance_env_ids
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                batch_timing["panel_clearance_backend_s"] += (
                    time.perf_counter() - panel_start
                )
                for env_id, value in zip(
                    panel_clearance_env_ids, panel_values, strict=True
                ):
                    controllers[env_id].batched_panel_finger_clearance = float(value)

        geometry_numpy_metrics = None
        geometry_numpy_rows = {}
        if args.batch_geometry_fallback_backend == "numpy_batch":
            fallback_env_ids = []
            for env_id in geometry_env_ids:
                row = geometry_rows[env_id]
                candidate_metrics = {
                    name: value[row] for name, value in geometry_metrics.items()
                }
                candidate_metrics["valid"] = True
                if controllers[env_id]._gpu_geometry_needs_scalar(
                    candidate_metrics
                ):
                    fallback_env_ids.append(env_id)
            if fallback_env_ids:
                numpy_start = time.perf_counter()
                geometry_numpy_metrics = _batched_geometry_metrics_numpy(
                    full_state,
                    geometry_cache,
                    fallback_env_ids,
                    handle_centers[fallback_env_ids],
                )
                batch_timing["geometry_numpy_backend_s"] += (
                    time.perf_counter() - numpy_start
                )
                candidate_rows = {
                    env_id: row for row, env_id in enumerate(fallback_env_ids)
                }
                for env_id, row in candidate_rows.items():
                    numpy_metrics = {
                        name: value[row]
                        for name, value in geometry_numpy_metrics.items()
                    }
                    if _numpy_geometry_requires_exact_scalar(
                        controllers[env_id], numpy_metrics, Stage.CENTER_HANDLE
                    ):
                        geometry_numpy_backend_stats["exact_scalar_requests"] += 1
                    else:
                        geometry_numpy_rows[env_id] = row
                geometry_numpy_backend_stats["batches"] += 1
                geometry_numpy_backend_stats["requests"] += len(fallback_env_ids)
                geometry_numpy_backend_stats["accepted_requests"] += len(
                    geometry_numpy_rows
                )
                geometry_numpy_backend_stats["max_batch"] = max(
                    geometry_numpy_backend_stats["max_batch"],
                    len(fallback_env_ids),
                )

        if steady_path_executor is not None:
            phase_start = time.perf_counter()
            planning = []
            for env_id, controller in enumerate(controllers):
                if not active_env_mask[env_id] or controller.stage != Stage.APPROACH:
                    continue
                sliced_state = state_rows[env_id]
                root_state = _as_np(sliced_state["root_states"])[0]
                dist, yaw_error = pending_metrics[env_id]
                base_stable = bool(
                    dist < controller.station_tolerance
                    and yaw_error < BASE_YAW_TOLERANCE_RAD
                    and float(np.linalg.norm(root_state[7:10])) < 0.02
                    and float(np.linalg.norm(root_state[10:13])) < 0.05
                )
                if not base_stable or controller.approach_settle_steps + 1 < 25:
                    continue
                request = _steady_path_request(
                    sliced_state,
                    _as_np(sliced_state["ee_pos"])[0],
                    controller.center_standoff_ee_world,
                    controller.grasp_orientation_quat,
                    start_orientation_world_wxyz=_as_np(sliced_state["ee_orn"])[0],
                )
                request_candidates = []
                for candidate in _pregrasp_pitch_candidates(controller):
                    command_position, command_orientation = (
                        _perturb_ee_target(
                            controller,
                            "pregrasp",
                            candidate["end_position"],
                            candidate["orientation"],
                        )
                    )
                    request_candidates.append(
                        {
                            "pitch_deg": candidate["pitch_deg"],
                            "orientation": command_orientation.astype(float).tolist(),
                            "end_position": command_position.astype(float).tolist(),
                            "nominal_orientation": candidate["orientation"].astype(float).tolist(),
                            "nominal_end_position": candidate["end_position"].astype(float).tolist(),
                        }
                    )
                request["pregrasp_candidates"] = request_candidates
                request["path_kind"] = "steady"
                planning.append((controller, request))
            if planning:
                solved = steady_path_executor.map(
                    solve_deferred_path_request,
                    [request for _, request in planning],
                )
                for (controller, _), solved_path in zip(
                    planning, solved, strict=True
                ):
                    _merge_tolerance_warning_stats(
                        steady_tolerance_stats, solved_path["warning_stats"]
                    )
                    if solved_path["error"] is not None:
                        path_backend_stats["worker_errors"] += 1
                        controller.failure_reason = (
                            "controller_runtime_error:" + solved_path["error"]
                        )
                        controller.failure_stage = controller.stage.name
                        controller._transition_to(Stage.FAILURE, reset_steps=False)
                        continue
                    selection = solved_path.get("pregrasp_selection")
                    if selection is None:
                        raise RuntimeError("PREGRASP worker omitted pitch selection")
                    _apply_pregrasp_candidate(controller,
                        {
                            "pitch_deg": selection["pitch_deg"],
                            "orientation": selection["nominal_orientation"],
                            "end_position": selection["nominal_end_position"],
                        }
                    )
                    controller.arm_align_desired_target_world = np.asarray(
                        selection["end_position"], dtype=np.float32
                    ).copy()
                    print(
                        f"[doorman][pregrasp-pitch] "
                        f"selected={controller.pregrasp_pitch_deg:.0f}deg "
                        f"exact={selection['exact']} "
                        f"candidates={selection['diagnostics']}",
                        flush=True,
                    )
                    controller.precomputed_steady_paths["arm_align_path"] = (
                        solved_path["positions"],
                        solved_path["orientations"],
                    )
            batch_timing["steady_path_workers_s"] += (
                time.perf_counter() - phase_start
            )

        phase_start = time.perf_counter()
        deferred_path_requests = []
        for env_id, (view, controller) in enumerate(zip(views, controllers)):
            if not active_env_mask[env_id]:
                continue
            if controller.stage in (Stage.SUCCESS, Stage.FAILURE):
                continue
            sliced_state = state_rows[env_id]
            view.set_state(sliced_state)
            controller.batched_handle_center = handle_centers[env_id]
            hinge_velocity = _as_np(sliced_state.get("door_joint_velocity"))
            if hinge_velocity is not None:
                max_abs_hinge_speeds[env_id] = max(
                    max_abs_hinge_speeds[env_id],
                    abs(float(hinge_velocity[0, controller.hinge_idx])),
                )
            controller.replan_counter = (
                controller.replan_counter + 1
            ) % controller.replan_interval
            dist, yaw_error = pending_metrics[env_id]
            try:
                precomputed_geometry_metrics = None
                precomputed_geometry_authoritative = False
                if env_id in geometry_rows:
                    if env_id in geometry_numpy_rows:
                        row = geometry_numpy_rows[env_id]
                        precomputed_geometry_metrics = {
                            name: value[row]
                            for name, value in geometry_numpy_metrics.items()
                        }
                        precomputed_geometry_authoritative = True
                    else:
                        row = geometry_rows[env_id]
                        precomputed_geometry_metrics = {
                            name: value[row]
                            for name, value in geometry_metrics.items()
                        }
                    precomputed_geometry_metrics["valid"] = True
                controller.batched_stability_result = stability_results.get(env_id)
                controller.defer_path_solve = (
                    args.batch_path_backend == "exact_workers"
                )
                with (
                    contextlib.redirect_stdout(controller_stdout)
                    if controller_stdout is not None
                    else contextlib.nullcontext()
                ):
                    controller.advance(
                        sliced_state,
                        dist,
                        yaw_error,
                        precomputed_geometry_metrics=precomputed_geometry_metrics,
                        precomputed_geometry_authoritative=(
                            precomputed_geometry_authoritative
                        ),
                    )
            except DeferredPathSolve as request:
                deferred_path_requests.append((env_id, request))
            except RuntimeError as exc:
                controller.failure_reason = f"controller_runtime_error:{exc}"
                controller.failure_stage = controller.stage.name
                controller._transition_to(Stage.FAILURE, reset_steps=False)
            finally:
                controller.defer_path_solve = False

        if deferred_path_requests:
            path_start = time.perf_counter()
            path_results = list(
                steady_path_executor.map(
                    solve_deferred_path_request,
                    [request.request for _, request in deferred_path_requests],
                )
            )
            batch_timing["path_backend_s"] += time.perf_counter() - path_start
            path_backend_stats["batches"] += 1
            path_backend_stats["requests"] += len(deferred_path_requests)
            for (env_id, request), result in zip(
                deferred_path_requests, path_results, strict=True
            ):
                labels = path_backend_stats["labels"]
                labels[request.label] = labels.get(request.label, 0) + 1
                _merge_tolerance_warning_stats(
                    steady_tolerance_stats, result["warning_stats"]
                )
                controller = request.controller
                if result["error"] is not None:
                    path_backend_stats["worker_errors"] += 1
                    controller.failure_reason = (
                        "controller_runtime_error:" + result["error"]
                    )
                    controller.failure_stage = controller.stage.name
                    controller._transition_to(Stage.FAILURE, reset_steps=False)
                    continue
                controller._complete_deferred_path(
                    request.label,
                    result["positions"],
                    result["orientations"],
                )
        batch_timing["advance_s"] += time.perf_counter() - phase_start

    print(
        "[doorman][batch-timing] "
        + " ".join(f"{name}={value:.3f}" for name, value in batch_timing.items()),
        flush=True,
    )
    print(
        "[doorman][geometry-gpu] "
        f"uses={sum(c.gpu_geometry_uses for c in controllers)} "
        f"fallbacks={sum(c.gpu_geometry_fallbacks for c in controllers)}",
        flush=True,
    )
    print(
        "[doorman][stability-gpu] "
        f"uses={sum(c.gpu_stability_uses for c in controllers)} "
        f"fallbacks={sum(c.gpu_stability_fallbacks for c in controllers)}",
        flush=True,
    )
    print(
        f"[doorman][handle-center-gpu] wall_s={handle_center_gpu_s:.3f}",
        flush=True,
    )
    print(
        f"[doorman][orientation-gpu] calls={orientation_gpu_calls}",
        flush=True,
    )
    print(
        f"[doorman][base-command-exact] wall_s={base_command_exact_s:.3f}",
        flush=True,
    )
    if steady_path_executor is not None:
        steady_path_executor.shutdown(wait=True)

    successes = 0
    for env_id in active_env_ids:
        controller = controllers[env_id]
        if controller.stage not in (Stage.SUCCESS, Stage.FAILURE):
            controller.failure_reason = f"max_steps:{args.max_steps}"
            controller._transition_to(Stage.FAILURE, reset_steps=False)
        successes += int(controller.stage == Stage.SUCCESS)
        sliced_state = state_rows[env_id]
        _write_batched_episode_result(
            args,
            controller,
            runner,
            sliced_state,
            env_id,
            base_seed + env_id,
            step_counts[env_id],
            max_finger_forces[env_id],
            max_abs_hinge_speeds[env_id],
            initial_root_states[env_id],
            initial_dof_positions[env_id],
        )
    if args.record_video:
        episode_dir = args.output_dir / f"seed_{base_seed}"
        video_dir = episode_dir / "video"
        video_dir.mkdir(exist_ok=True)
        for name, frames in video_frames.items():
            _save_rgb_video(
                frames,
                video_timestamps[name],
                video_dir / f"{name}.mp4",
            )
        _save_rgb_mosaic_video(
            video_frames,
            video_timestamps,
            video_dir / "all_cameras_mosaic.mp4",
        )
    if standard_collector is not None:
        controller = controllers[0]
        success = controller.stage == Stage.SUCCESS
        reasons = () if success else (
            f"autoopen_failure:{controller.failure_stage or controller.stage.name}:"
            f"{controller.failure_reason or 'unknown'}",
        )
        standard_episode = standard_collector.finish(
            acceptance_reasons=reasons,
            provenance={
                "seed": base_seed,
                "initial_root_pose_w": initial_root_states[0].tolist(),
                "initial_dof_position": initial_dof_positions[0].tolist(),
                "initial_gripper_position": float(initial_gripper_positions[0]),
                "door_randomization": standard_door_randomization,
                "door_asset": standard_door_asset_snapshot,
                "camera_mount_randomization": dict(
                    args.camera_mount_randomization or {}
                ),
                "contact_audit": standard_collector.contact_view_description,
            },
        )
        standard_collector.close()
        from gr00t.rl.data_collection.auto_open_writer import AutoOpenDatasetWriter

        camera_calibration = json.loads(
            Path(args.rgb_camera_config).read_text(encoding="utf-8")
        )
        standard_writer = AutoOpenDatasetWriter(
            args.standard_dataset_dir,
            camera_calibration=camera_calibration,
            simulation_provenance=_standard_simulation_provenance(
                args, runner, base_seed
            ),
        )
        standard_writer.write_episode(standard_episode)
        standard_writer.finalize()
    _merge_tolerance_warning_stats(
        steady_tolerance_stats,
        shared_steady_model.consume_tolerance_warning_stats(),
    )
    summary = {
        "batch_seed": base_seed,
        "num_envs": num_envs,
        "active_env_ids": list(active_env_ids),
        "successes": successes,
        "success_rate": successes / len(active_env_ids),
        "steady_path_workers": int(args.steady_path_workers),
        "steady_inverse_tolerance": steady_tolerance_stats,
        "batch_timing": batch_timing,
        "panel_clearance_backend": args.batch_panel_clearance_backend,
        "geometry_fallback_backend": args.batch_geometry_fallback_backend,
        "geometry_numpy_backend": (
            geometry_numpy_backend_stats
            if args.batch_geometry_fallback_backend == "numpy_batch"
            else None
        ),
        "contact_ik_backend": (
            contact_ik_backend_stats
            if args.batch_contact_ik_backend == "exact_workers"
            else None
        ),
        "path_backend": (
            path_backend_stats
            if args.batch_path_backend == "exact_workers"
            else None
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"[doorman][batch] complete: {successes}/{len(active_env_ids)} active "
        f"({num_envs} initialized)",
        flush=True,
    )
    print(
        "[doorman][steady-inverse-tolerance] "
        + " ".join(f"{key}={value}" for key, value in steady_tolerance_stats.items()),
        flush=True,
    )
    if args.headless:
        # Isaac Sim 5.1 spends tens of seconds unloading a large vectorized
        # stage after all results are durable.  The process owns this App, so
        # deterministic process teardown is the correct headless lifecycle.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    runner.close()
    return successes
