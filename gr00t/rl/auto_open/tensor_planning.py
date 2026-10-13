"""Sparse planning boundary for the GPU-resident AutoOpen FSM."""

from __future__ import annotations

import math

import numpy as np
import torch

from legged_gym.utils.steady_path_worker import (
    solve_deferred_path_request,
    solve_steady_inverse_request,
)
from gr00t.rl.auto_open.control import (
    _ee_world_orientation_target,
    _ee_world_target,
    _perturb_ee_target,
)
from gr00t.rl.auto_open.geometry import (
    _geometry_metrics,
    _physical_handle_grasp_position,
)
from gr00t.rl.auto_open.planning import (
    DeferredSteadyInverse,
    _apply_pregrasp_candidate,
    _prepare_panel_contact_command,
    _pregrasp_pitch_candidates,
    _release_path_request,
    _sample_reachable_arm_retract_pose,
    _steady_inverse_worker_request,
    _steady_path_request,
)


def selected_state_to_host(state, env_id: int, num_envs: int):
    """Transfer one sparse planning row while preserving shared metadata."""
    sliced = {}
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and value.shape[0] == num_envs:
                value = value[env_id : env_id + 1]
            sliced[name] = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            if value.ndim > 0 and value.shape[0] == num_envs:
                value = value[env_id : env_id + 1]
            sliced[name] = value
        else:
            sliced[name] = value
    return sliced


class TensorPlanningCoordinator:
    """Convert sparse FSM events into batched exact-planning requests."""

    def __init__(self, controllers, views, executor, stage_enum):
        self.controllers = controllers
        self.views = views
        self.executor = executor
        self.Stage = stage_enum
        self.num_envs = len(controllers)
        self.stats = {"batches": 0, "requests": 0, "errors": 0, "labels": {}}

    @staticmethod
    def _ids(rows) -> list[int]:
        if rows is None or not len(rows):
            return []
        return [int(value) for value in rows.detach().cpu().tolist()]

    def _state(self, device_state, env_id):
        state = selected_state_to_host(device_state, env_id, self.num_envs)
        self.views[env_id].set_state(state)
        return state

    def _run_paths(self, pending):
        if not pending:
            return []
        if self.executor is None:
            raise RuntimeError("tensor FSM sparse paths require the exact worker backend")
        results = list(
            self.executor.map(
                solve_deferred_path_request,
                [item[2] for item in pending],
            )
        )
        self.stats["batches"] += 1
        self.stats["requests"] += len(pending)
        for label, _, _ in pending:
            labels = self.stats["labels"]
            labels[label] = labels.get(label, 0) + 1
        return results

    def plan_pregrasp(self, fsm, device_state, rows):
        pending = []
        for env_id in self._ids(rows):
            controller = self.controllers[env_id]
            state = self._state(device_state, env_id)
            request = _steady_path_request(
                state,
                state["ee_pos"][0],
                controller.center_standoff_ee_world,
                controller.grasp_orientation_quat,
                start_orientation_world_wxyz=state["ee_orn"][0],
            )
            candidates = []
            for candidate in _pregrasp_pitch_candidates(controller):
                position, orientation = _perturb_ee_target(
                    controller,
                    "pregrasp",
                    candidate["end_position"],
                    candidate["orientation"],
                )
                candidates.append(
                    {
                        "pitch_deg": candidate["pitch_deg"],
                        "orientation": orientation.astype(float).tolist(),
                        "end_position": position.astype(float).tolist(),
                        "nominal_orientation": candidate["orientation"].astype(float).tolist(),
                        "nominal_end_position": candidate["end_position"].astype(float).tolist(),
                    }
                )
            request["pregrasp_candidates"] = candidates
            request["path_kind"] = "steady"
            pending.append(("arm_align_path", env_id, request))
        results = self._run_paths(pending)
        ids, positions, orientations, targets = [], [], [], []
        for (_, env_id, _), result in zip(pending, results, strict=True):
            if result["error"] is not None:
                self.stats["errors"] += 1
                fsm.fail(torch.as_tensor([env_id], device=fsm.device), 40)
                continue
            selection = result["pregrasp_selection"]
            controller = self.controllers[env_id]
            _apply_pregrasp_candidate(
                controller,
                {
                    "pitch_deg": selection["pitch_deg"],
                    "orientation": selection["nominal_orientation"],
                    "end_position": selection["nominal_end_position"],
                },
            )
            controller.arm_align_desired_target_world = np.asarray(
                selection["end_position"], dtype=np.float32
            )
            ids.append(env_id)
            positions.append(result["positions"])
            orientations.append(result["orientations"])
            targets.append(selection["end_position"])
        if ids:
            fsm.install_pregrasp_plan(ids, positions, orientations, targets)

    def plan_center(self, fsm, device_state, rows):
        pending = []
        metadata = {}
        for env_id in self._ids(rows):
            controller = self.controllers[env_id]
            state = self._state(device_state, env_id)
            handle_pose = state["door_body_state_w"][0, controller.handle_body_idx, :7]
            grasp_center = _physical_handle_grasp_position(controller, handle_pose)
            metrics = _geometry_metrics(controller, state, grasp_center)
            if not metrics.get("valid"):
                fsm.fail(torch.as_tensor([env_id], device=fsm.device), 41)
                continue
            fixed_upper = bool(np.asarray(metrics["fixed_center"])[2] >= grasp_center[2])
            fixed_gap = max(
                float(metrics["upper_gap" if fixed_upper else "lower_gap"]), 0.0
            )
            error = float(metrics["channel_vertical_error"])
            needed = max(abs(error) - 0.008, 0.0)
            available = max(0.019 - fixed_gap, 0.0)
            shift = -math.copysign(min(needed, available), error) if needed else 0.0
            if abs(error + shift) > 0.010:
                fsm.fail(torch.as_tensor([env_id], device=fsm.device), 42)
                continue
            controller.arm_forward_target_world = controller.grasp_waypoint_world.copy()
            controller.arm_forward_target_world[2] += shift
            target, orientation = _perturb_ee_target(
                controller,
                "center_handle",
                controller.arm_forward_target_world,
                controller.grasp_orientation_quat,
            )
            controller.arm_forward_desired_target_world = target.copy()
            request = _steady_path_request(
                state,
                state["ee_pos"][0],
                target,
                orientation,
                start_orientation_world_wxyz=controller.grasp_orientation_quat,
            )
            request["path_kind"] = "steady"
            pending.append(("arm_forward_path", env_id, request))
            metadata[env_id] = target
        results = self._run_paths(pending)
        ids, positions, orientations, targets = [], [], [], []
        for (_, env_id, _), result in zip(pending, results, strict=True):
            if result["error"] is not None:
                self.stats["errors"] += 1
                fsm.fail(torch.as_tensor([env_id], device=fsm.device), 43)
                continue
            ids.append(env_id)
            positions.append(result["positions"])
            orientations.append(result["orientations"])
            targets.append(metadata[env_id])
        if ids:
            fsm.install_center_plan(ids, positions, orientations, targets)

    def _release_request(self, controller, state, label, start, end, start_q, end_q, q0, q1, steps):
        return _release_path_request(
            state, start, end, start_q, end_q, q0, q1, steps,
            controller.pregrasp_position_tolerance,
            int(controller.env.gripper_joint_ids[0].item()),
        )

    def plan_release_open(self, fsm, device_state, rows):
        pending = []
        for env_id in self._ids(rows):
            controller = self.controllers[env_id]
            state = self._state(device_state, env_id)
            start = state["ee_pos"][0].copy()
            start_q = state["ee_orn"][0].copy()
            end = start + np.asarray(
                [0.0, 0.0, float(fsm.release_vertical_delta[env_id].item())],
                dtype=np.float32,
            )
            end, end_q = _perturb_ee_target(
                controller, "open_and_clear", end, start_q
            )
            request = self._release_request(
                controller, state, "release_open_path", start, end, start_q, end_q,
                controller.q_closed, controller.q_open, 35,
            )
            pending.append(("release_open_path", env_id, request))
        results = self._run_paths(pending)
        self._install_simple_path(fsm, pending, results, "open")

    def plan_retreat(self, fsm, device_state, rows):
        pending, targets = [], {}
        for env_id in self._ids(rows):
            controller = self.controllers[env_id]
            state = self._state(device_state, env_id)
            start = state["ee_pos"][0].copy()
            end = start + 0.10 * controller.door_normal + np.asarray([0, 0, 0.02], dtype=np.float32)
            end, end_q = _perturb_ee_target(
                controller, "retreat_arm", end, state["ee_orn"][0]
            )
            steps = max(50, int(math.ceil(float(np.linalg.norm(end - start)) / 0.002)))
            request = self._release_request(
                controller, state, "release_retreat_path", start, end,
                state["ee_orn"][0], end_q, controller.q_open, controller.q_open, steps,
            )
            pending.append(("release_retreat_path", env_id, request))
            targets[env_id] = end
        results = self._run_paths(pending)
        ids, positions, orientations, acceptance = self._successful_paths(
            fsm, pending, results
        )
        acceptance = [targets[env_id] for env_id in ids]
        if ids:
            fsm.install_retreat_plan(ids, positions, orientations, acceptance)

    def plan_restore(self, fsm, device_state, rows):
        pending, endpoints = [], {}
        for env_id in self._ids(rows):
            controller = self.controllers[env_id]
            state = self._state(device_state, env_id)
            try:
                _sample_reachable_arm_retract_pose(
                    controller, state, self.Stage.RESTORE_GRIPPER.name
                )
            except RuntimeError:
                self.stats["errors"] += 1
                fsm.fail(torch.as_tensor([env_id], device=fsm.device), 46)
                continue
            fsm.active_arm_retract_pose_local[env_id] = torch.as_tensor(
                controller.active_arm_retract_pose_local,
                device=fsm.device,
                dtype=torch.float32,
            )
            end = _ee_world_target(controller, controller.active_arm_retract_pose_local[:3])
            end_q = _ee_world_orientation_target(
                controller, controller.active_arm_retract_pose_local[3:]
            )
            end, end_q = _perturb_ee_target(
                controller, "restore_gripper", end, end_q
            )
            start = state["ee_pos"][0]
            steps = max(50, int(math.ceil(float(np.linalg.norm(end - start)) / 0.004)))
            request = self._release_request(
                controller, state, "release_restore_path", start, end,
                state["ee_orn"][0], end_q, controller.q_open, controller.q_closed, steps,
            )
            pending.append(("release_restore_path", env_id, request))
            endpoints[env_id] = (end, end_q)
        results = self._run_paths(pending)
        ids, positions, orientations, _ = self._successful_paths(fsm, pending, results)
        if ids:
            fsm.install_restore_plan(
                ids, positions, orientations,
                [endpoints[env_id][0] for env_id in ids],
                [endpoints[env_id][1] for env_id in ids],
            )

    def _successful_paths(self, fsm, pending, results):
        ids, positions, orientations = [], [], []
        for (_, env_id, _), result in zip(pending, results, strict=True):
            if result["error"] is not None:
                self.stats["errors"] += 1
                fsm.fail(torch.as_tensor([env_id], device=fsm.device), 44)
                continue
            ids.append(env_id)
            positions.append(result["positions"])
            orientations.append(result["orientations"])
        return ids, positions, orientations, None

    def _install_simple_path(self, fsm, pending, results, kind):
        ids, positions, orientations, _ = self._successful_paths(
            fsm, pending, results
        )
        if kind == "open" and ids:
            fsm.install_release_open_plan(ids, positions, orientations)

    def plan_contact(self, fsm, device_state, rows):
        pending = []
        ids, panel, desired, commands, orientations = [], [], [], [], []
        for env_id in self._ids(rows):
            controller = self.controllers[env_id]
            state = self._state(device_state, env_id)
            if bool(fsm.panel_contact_orientation_seed_valid[env_id].item()):
                controller.panel_contact_orientation = (
                    fsm.panel_contact_orientation_seed[env_id]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                )
                fsm.panel_contact_orientation_seed_valid[env_id] = False
            else:
                controller.panel_contact_orientation = None
            controller.prepared_panel_contact_command = None
            controller.defer_steady_inverse = True
            try:
                _prepare_panel_contact_command(controller, state)
                command_position, command_orientation = (
                    controller.prepared_panel_contact_command
                )
                ids.append(env_id)
                panel.append(controller.panel_contact_target_world)
                desired.append(controller.panel_contact_desired_ee_world)
                commands.append(command_position)
                orientations.append(command_orientation)
            except DeferredSteadyInverse as request:
                pending.append((env_id, request))
            except RuntimeError:
                self.stats["errors"] += 1
                fsm.fail(torch.as_tensor([env_id], device=fsm.device), 45)
            finally:
                controller.defer_steady_inverse = False
        if pending:
            if self.executor is None:
                raise RuntimeError("tensor FSM contact IK requires exact workers")
            results = list(
                self.executor.map(
                    solve_steady_inverse_request,
                    [_steady_inverse_worker_request(request) for _, request in pending],
                )
            )
            for (env_id, request), result in zip(pending, results, strict=True):
                if result["error"] is not None:
                    self.stats["errors"] += 1
                    fsm.fail(torch.as_tensor([env_id], device=fsm.device), 45)
                    continue
                controller = self.controllers[env_id]
                ids.append(env_id)
                panel.append(controller.panel_contact_target_world)
                desired.append(request.desired_position_world)
                commands.append(result["command_position"])
                orientations.append(result["command_orientation"])
        if ids:
            fsm.install_contact_plan(ids, panel, desired, commands, orientations)

    def plan_push_retract(self, fsm, device_state, rows):
        ids, poses = [], []
        for env_id in self._ids(rows):
            controller = self.controllers[env_id]
            state = self._state(device_state, env_id)
            try:
                _sample_reachable_arm_retract_pose(
                    controller, state, self.Stage.PUSH_ARM_RETRACT.name
                )
            except RuntimeError:
                self.stats["errors"] += 1
                fsm.fail(torch.as_tensor([env_id], device=fsm.device), 46)
                continue
            ids.append(env_id)
            poses.append(controller.active_arm_retract_pose_local.copy())
        if ids:
            fsm.install_push_retract_pose(device_state, ids, poses)
