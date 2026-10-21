"""Sparse planning boundary for the GPU-resident AutoOpen FSM."""

from __future__ import annotations

import math

import numpy as np
import torch

from legged_gym.utils.ee_steady_inverse_gpu import (
    batched_steady_ee_inverse_no_gravity,
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
    _attach_pregrasp_reachability,
    _apply_pregrasp_candidate,
    _prepare_panel_contact_command,
    _pregrasp_pitch_candidates,
    _quat_nlerp,
    _release_path_request,
    _sample_reachable_arm_retract_pose,
    _steady_path_request,
)
from gr00t.rl.auto_open.pregrasp_reachability import (
    select_pregrasp_candidate_from_errors,
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
        specs = [self._gpu_path_spec(env_id, request) for _, env_id, request in pending]
        flat = [
            (spec_index, sample_index, float(fraction))
            for spec_index, spec in enumerate(specs)
            for sample_index, fraction in enumerate(spec["fractions"])
        ]
        roots = np.stack([specs[index]["root"] for index, _, _ in flat])
        arms = torch.stack([specs[index]["arm"] for index, _, _ in flat])
        positions = np.asarray(
            [
                (1.0 - fraction) * specs[index]["start"]
                + fraction * specs[index]["end"]
                for index, _, fraction in flat
            ]
        )
        orientations = np.asarray(
            [
                _quat_nlerp(
                    specs[index]["start_q"], specs[index]["end_q"], fraction
                )
                for index, _, fraction in flat
            ]
        )
        result = self._inverse_gpu_batch(
            [specs[index]["env_id"] for index, _, _ in flat],
            roots,
            arms,
            positions,
            orientations,
            [specs[index]["tolerance"] for index, _, _ in flat],
        )
        for batch_index, (index, _, _) in enumerate(flat):
            specs[index]["positions"].append(
                result.command_position_world[batch_index].detach().cpu().numpy()
            )
            specs[index]["orientations"].append(
                result.command_orientation_world_wxyz[batch_index]
                .detach()
                .cpu()
                .numpy()
            )
        results = []
        for spec in specs:
            positions = np.asarray(spec["positions"], dtype=np.float32)
            desired = np.asarray(
                [(1.0 - value) * spec["start"] + value * spec["end"] for value in spec["fractions"]]
            )
            results.append(
                {
                    "error": None,
                    "positions": positions,
                    "orientations": np.asarray(spec["orientations"], dtype=np.float32),
                    "max_prediction_error_m": float(
                        np.linalg.norm(positions - desired, axis=1).max(initial=0.0)
                    ),
                    "warning_stats": {},
                }
            )
        self.stats["batches"] += 1
        self.stats["requests"] += len(pending)
        for label, _, _ in pending:
            labels = self.stats["labels"]
            labels[label] = labels.get(label, 0) + 1
        return results

    def _inverse_gpu_batch(self, env_ids, roots, arms, positions, orientations, tolerances):
        controller = self.controllers[env_ids[0]]
        limits = torch.stack(
            [
                self.controllers[env_id].env.dof_pos_limits[
                    self.controllers[env_id].env.arm_joint_ids
                ]
                for env_id in env_ids
            ]
        )
        return batched_steady_ee_inverse_no_gravity(
            torch.as_tensor(positions, device=limits.device, dtype=limits.dtype),
            torch.as_tensor(orientations, device=limits.device, dtype=limits.dtype),
            base_position_world=torch.as_tensor(roots[:, :3], device=limits.device, dtype=limits.dtype),
            base_orientation_world_xyzw=torch.as_tensor(roots[:, [4, 5, 6, 3]], device=limits.device, dtype=limits.dtype),
            initial_arm_joint_positions=arms,
            joint_limits=limits,
            position_tolerance_m=float(max(tolerances)),
            orientation_tolerance=1.0e-4,
        )

    def plan_turn_path(self, fsm, device_state, rows):
        """Resolve a complete press/push-pull/return path once at stage entry."""
        env_ids = self._ids(rows)
        if not env_ids:
            return
        row_tensor = torch.as_tensor(env_ids, dtype=torch.int64, device=fsm.device)
        lengths = fsm.turn_path.length[row_tensor].clone()
        max_samples = int(lengths.max().item())
        desired_positions = fsm.turn_path.positions[
            row_tensor, :max_samples
        ].clone()
        desired_orientations = fsm.turn_path.orientations[
            row_tensor, :max_samples
        ].clone()
        solved_positions = desired_positions.clone()
        solved_orientations = desired_orientations.clone()
        arm_ids = self.controllers[env_ids[0]].env.arm_joint_ids.to(
            device=fsm.device, dtype=torch.int64
        )
        arms = device_state["dof_pos"][row_tensor][:, arm_ids].clone()
        roots = device_state["root_states"][row_tensor]
        limits = torch.stack(
            [
                self.controllers[env_id].env.dof_pos_limits[
                    self.controllers[env_id].env.arm_joint_ids
                ]
                for env_id in env_ids
            ]
        )
        env_index, sample_index = torch.where(
            torch.arange(max_samples, device=fsm.device)[None, :]
            < lengths[:, None]
        )
        result = batched_steady_ee_inverse_no_gravity(
            desired_positions[env_index, sample_index],
            desired_orientations[env_index, sample_index],
            base_position_world=roots[env_index, :3],
            base_orientation_world_xyzw=roots[env_index][:, [4, 5, 6, 3]],
            initial_arm_joint_positions=arms[env_index],
            joint_limits=limits[env_index],
        )
        solved_positions[env_index, sample_index] = result.command_position_world
        solved_orientations[env_index, sample_index] = (
            result.command_orientation_world_wxyz
        )
        fsm.turn_path.positions[row_tensor, :max_samples] = solved_positions
        fsm.turn_path.orientations[row_tensor, :max_samples] = solved_orientations
        self.stats["batches"] += 1
        self.stats["requests"] += len(env_ids)
        labels = self.stats["labels"]
        labels["turn_stage_path"] = labels.get("turn_stage_path", 0) + len(env_ids)

    def _gpu_path_spec(self, env_id, request):
        start = np.asarray(request["start_position_world"], dtype=np.float64)
        end = np.asarray(request["end_position_world"], dtype=np.float64)
        start_q = np.asarray(request["start_orientation_world_wxyz"], dtype=np.float64)
        end_q = np.asarray(request["end_orientation_world_wxyz"], dtype=np.float64)
        if request["path_kind"] == "release":
            count = int(request["steps"])
            fractions = np.linspace(0.0, 1.0, count + 1)
            tolerance = float(request["position_tolerance_m"])
        else:
            orientation_distance = 2.0 * math.acos(
                float(np.clip(abs(np.dot(start_q, end_q)), 0.0, 1.0))
            )
            count = max(
                1,
                int(math.ceil(float(np.linalg.norm(end - start)) / 0.002)),
                int(math.ceil(orientation_distance / 0.03)),
            )
            fractions = np.linspace(1.0 / count, 1.0, count)
            tolerance = 5.0e-3
        names = list(request["dof_names"])
        values = np.asarray(request["dof_positions"], dtype=np.float32)
        arm_ids = [names.index(f"joint{index}") for index in range(1, 7)]
        controller = self.controllers[env_id]
        device = controller.env.device
        arm = torch.as_tensor(values[arm_ids], device=device, dtype=torch.float32)
        root = np.asarray(request["root_state"], dtype=np.float32)
        return {
            "env_id": env_id,
            "start": start,
            "end": end,
            "start_q": start_q,
            "end_q": end_q,
            "fractions": fractions,
            "tolerance": tolerance,
            "root": root,
            "arm": arm,
            "positions": [],
            "orientations": [],
        }

    def plan_pregrasp(self, fsm, device_state, rows):
        entries = []
        for env_id in self._ids(rows):
            controller = self.controllers[env_id]
            state = self._state(device_state, env_id)
            candidates = []
            for candidate in _pregrasp_pitch_candidates(controller):
                position, orientation = _perturb_ee_target(
                    controller,
                    "pregrasp",
                    candidate["end_position"],
                    candidate["orientation"],
                )
                candidates.append(
                    _attach_pregrasp_reachability(
                        controller,
                        {
                            "pitch_deg": candidate["pitch_deg"],
                            "orientation": orientation.astype(float).tolist(),
                            "end_position": position.astype(float).tolist(),
                            "nominal_orientation": candidate[
                                "orientation"
                            ].astype(float).tolist(),
                            "nominal_end_position": candidate[
                                "end_position"
                            ].astype(float).tolist(),
                        },
                        position,
                        state,
                    )
                )
            entries.append((env_id, controller, state, candidates))

        selections = self._select_pregrasp_gpu_batch(entries)
        pending = []
        metadata = {}
        for (env_id, controller, state, _), selection_result in zip(
            entries, selections, strict=True
        ):
            selection, exact, diagnostics = selection_result
            controller.pregrasp_pitch_diagnostics = diagnostics
            if not exact:
                print(
                    f"[doorman][pregrasp-pitch][error] env={env_id} "
                    f"no reachable pitch; candidates={diagnostics}",
                    flush=True,
                )
                fsm.fail(torch.as_tensor([env_id], device=fsm.device), 47)
                continue
            print(
                "[doorman][pregrasp-pitch] "
                f"env={env_id} selected={selection['pitch_deg']:.0f}deg "
                f"candidates={diagnostics}",
                flush=True,
            )
            request = _steady_path_request(
                state,
                state["ee_pos"][0],
                selection["end_position"],
                selection["orientation"],
                start_orientation_world_wxyz=state["ee_orn"][0],
            )
            request["path_kind"] = "steady"
            pending.append(("arm_align_path", env_id, request))
            metadata[env_id] = selection
        results = self._run_paths(pending)
        ids, positions, orientations, targets = [], [], [], []
        for (_, env_id, _), result in zip(pending, results, strict=True):
            if result["error"] is not None:
                self.stats["errors"] += 1
                fsm.fail(torch.as_tensor([env_id], device=fsm.device), 40)
                continue
            selection = metadata[env_id]
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

    def _select_pregrasp_gpu_batch(self, entries):
        if not entries:
            return []
        flat = []
        for entry_index, (env_id, _, state, candidates) in enumerate(entries):
            names = list(state["dof_names"])
            values = np.asarray(state["dof_pos"])[0]
            arm_ids = [names.index(f"joint{index}") for index in range(1, 7)]
            root = np.asarray(state["root_states"])[0]
            for candidate_index, candidate in enumerate(candidates):
                flat.append((entry_index, env_id, candidate_index, candidate, root, values[arm_ids]))
        device = self.controllers[flat[0][1]].env.device
        sample_counts = [len(item[3]["reachability_positions"]) for item in flat]
        sample_rows = np.repeat(np.arange(len(flat)), sample_counts)
        initial_arms = torch.as_tensor(
            np.asarray([item[5] for item in flat]), device=device, dtype=torch.float32
        )
        positions = np.concatenate([item[3]["reachability_positions"] for item in flat])
        orientations = np.concatenate([item[3]["reachability_orientations"] for item in flat])
        roots = np.asarray([item[4] for item in flat])[sample_rows]
        device_sample_rows = torch.as_tensor(sample_rows, device=device, dtype=torch.int64)
        limits = torch.stack(
            [
                self.controllers[item[1]].env.dof_pos_limits[
                    self.controllers[item[1]].env.arm_joint_ids
                ]
                for item in flat
            ]
        )
        result = batched_steady_ee_inverse_no_gravity(
            torch.as_tensor(positions, device=device, dtype=torch.float32),
            torch.as_tensor(orientations, device=device, dtype=torch.float32),
            base_position_world=torch.as_tensor(
                roots[:, :3], device=device, dtype=torch.float32
            ),
            base_orientation_world_xyzw=torch.as_tensor(
                roots[:, [4, 5, 6, 3]], device=device, dtype=torch.float32
            ),
            initial_arm_joint_positions=initial_arms[device_sample_rows],
            joint_limits=limits[device_sample_rows],
            position_tolerance_m=float("inf"),
            orientation_tolerance=float("inf"),
        )
        position_errors = (
            torch.linalg.vector_norm(result.position_error, dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        orientation_errors = (
            torch.linalg.vector_norm(result.orientation_error, dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        selections = []
        boundaries = np.cumsum(sample_counts)[:-1]
        position_errors = np.split(position_errors, boundaries)
        orientation_errors = np.split(orientation_errors, boundaries)
        offset = 0
        for _, _, _, candidates in entries:
            count = len(candidates)
            selections.append(
                select_pregrasp_candidate_from_errors(
                    candidates,
                    position_errors[offset : offset + count],
                    orientation_errors[offset : offset + count],
                )
            )
            offset += count
        return selections

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
            roots, arms, positions, target_orientations, tolerances = [], [], [], [], []
            for env_id, request in pending:
                request_state = request.state
                names = list(request_state["dof_names"])
                values = np.asarray(request_state["dof_pos"])[0]
                arm_ids = [names.index(f"joint{index}") for index in range(1, 7)]
                roots.append(np.asarray(request_state["root_states"])[0])
                arms.append(torch.as_tensor(values[arm_ids], device=fsm.device, dtype=torch.float32))
                positions.append(request.desired_position_world)
                target_orientations.append(request.desired_orientation_world)
                tolerances.append(
                        5.0e-3
                        if request.position_tolerance_m is None
                        else request.position_tolerance_m
                )
            result = self._inverse_gpu_batch(
                [env_id for env_id, _ in pending],
                np.asarray(roots),
                torch.stack(arms),
                np.asarray(positions),
                np.asarray(target_orientations),
                tolerances,
            )
            position_error = torch.linalg.vector_norm(result.position_error, dim=-1)
            orientation_error = torch.linalg.vector_norm(result.orientation_error, dim=-1)
            for batch_index, (env_id, request) in enumerate(pending):
                controller = self.controllers[env_id]
                exact = (
                    float(position_error[batch_index].item()) <= tolerances[batch_index]
                    and float(orientation_error[batch_index].item()) <= 1.0e-4
                )
                if not exact:
                    self.stats["errors"] += 1
                    fsm.fail(torch.as_tensor([env_id], device=fsm.device), 45)
                    continue
                ids.append(env_id)
                panel.append(controller.panel_contact_target_world)
                desired.append(request.desired_position_world)
                commands.append(
                    result.command_position_world[batch_index].detach().cpu().numpy()
                )
                orientations.append(
                    result.command_orientation_world_wxyz[batch_index]
                    .detach()
                    .cpu()
                    .numpy()
                )
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
