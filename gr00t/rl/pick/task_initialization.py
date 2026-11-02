"""Task admission on one shared policy/physics batch, preserving physical gates."""

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gr00t.rl.auto_open.acceptance import (
    TensorStabilityWindow,
    consecutive_steps,
    update_robot_stability,
)
from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
from gr00t.rl.auto_open.execution import TensorCommandBuffers
from gr00t.rl.auto_open.motion_math import _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.auto_open.tensor_metadata import to_host
from gr00t.rl.pick.admission_schedule import AdmissionSchedule
from gr00t.rl.pick.basket import BasketGeometry
from gr00t.rl.pick.basket_controller import BasketArmCollisionSamples
from gr00t.rl.pick.basket_geometry import BasketGeometryBatch
from gr00t.rl.pick.basket_initialization import (
    BasketAdmissionInputs,
    admission_report_row,
    admission_report_to_host,
    basket_grasp_access_batch,
    basket_object_checks_batch,
    resample_basket_payloads,
)
from gr00t.rl.pick.basket_intrinsic_proposals import cached_intrinsic_proposals
from gr00t.rl.pick.basket_layout_validation import BasketLayoutPathValidator
from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
from gr00t.rl.pick.collision import ArmCollisionSamples
from gr00t.rl.pick.external_initialization import ExternalAdmissionInputs, scene_object_checks_batch
from gr00t.rl.pick.initialization import RobotOcclusionGeometryBatch, initial_visibility
from gr00t.rl.pick.prepare import NoValidInitialScene
from gr00t.rl.pick.randomization import NoSafeRobotSpawn, sample_candidates_batch
from gr00t.rl.pick.scene_object_geometry import SceneObjectGeometryBatch
from gr00t.rl.pick.scene_selection import AdmissionSceneCache
from gr00t.rl.pick.tensor_planning import EpisodeEnvironmentView, episode_in_world
from gr00t.rl.simulator.isaacsim.vlnverse_scene import discover_vlnverse_scenes
from gr00t.rl.task_runtime import BatchedTaskLoop, reset_task_rows
from gr00t.rl.utils.b2z1_asset import ROBOT_URDF


def payload_pose_stability(window, state):
    """Use the shared pose window in the moving basket's reference frame."""
    root = state["root_states"]
    payload = torch.cat((state["object_state"][:, None], state["distractor_states"]), 1)
    inverse = root[:, None, 3:7].expand_as(payload[..., 3:7]).clone()
    inverse[..., 1:] *= -1
    inverse = inverse.reshape(-1, 4)
    positions = _quat_rotate(inverse, (payload[..., :3] - root[:, None, :3]).reshape(-1, 3))
    orientations = _quat_multiply(inverse, payload[..., 3:7].reshape(-1, 4))
    rows = torch.arange(positions.shape[0], device=root.device)
    result = window.update({"ee_pos": positions, "ee_orn": orientations}, rows)
    return tuple(value.reshape(payload.shape[:2]) for value in result)


def update_pick_pose_stability(state, base_steps, object_steps, other_steps, mask, payload_window):
    """Shared physical settle predicates for admission and its diagnostic probe."""
    stable, base_steps, _ = update_robot_stability(
        state["root_states"],
        base_steps,
        required_frames=25,
        max_linear_speed_m_s=0.10,
        max_angular_speed_rad_s=0.35,
        min_base_height_m=0.45,
        min_upright_cos=0.95,
    )
    position_spans = orientation_spans = None
    if payload_window is not None:
        payload_stable, position_spans, orientation_spans = payload_pose_stability(
            payload_window, state
        )
        counts = payload_window.count.reshape(len(base_steps), mask.shape[1] + 1)
        object_steps = torch.where(payload_stable[:, 0], counts[:, 0], 0)
        other_steps = torch.where((payload_stable[:, 1:] | ~mask).all(-1), counts[:, 0], 0)
    else:
        obj, others = state["object_state"], state["distractor_states"]
        object_steps = consecutive_steps(
            (obj[:, 7:10].norm(dim=-1) < 0.03) & (obj[:, 10:13].norm(dim=-1) < 0.15),
            object_steps,
        )
        still = (others[..., 7:10].norm(dim=-1) < 0.03) & (others[..., 10:13].norm(dim=-1) < 0.15)
        other_steps = consecutive_steps((still | ~mask).all(-1), other_steps)
    return stable, base_steps, object_steps, other_steps, position_spans, orientation_spans


def layout_grasp_hints(episodes, reference):
    metadata = [e.get("basket_sampling", {}) for e in episodes]
    valid = torch.tensor(
        [
            m.get("reachable_proposals", 0) > 0 and not m.get("reachability_fallback", True)
            for m in metadata
        ],
        device=reference.device,
        dtype=torch.bool,
    )
    local = reference.new_tensor(
        [m.get("grasp_seed_orientation_local_wxyz", [1, 0, 0, 0]) for m in metadata]
    )
    return valid, _quat_multiply(reference[:, 3:7], local)


def _external_resample_proposals(episodes, rows, cache, attempt):
    from gr00t.rl.pick.distractors import sample_nearby_objects_batch
    from gr00t.rl.pick.randomization import PickRandomizationConfig, select_and_serialize_batch

    grouped, proposals = {}, {}
    attempts_by_row = dict(
        zip(rows, [attempt] * len(rows) if isinstance(attempt, int) else attempt, strict=True)
    )
    config = PickRandomizationConfig()
    for row in rows:
        episode = episodes[row]
        grouped.setdefault(episode["scene_usd_path"], []).append(row)
    for path, group in grouped.items():
        episode = episodes[group[0]]
        record = discover_vlnverse_scenes(Path(path).parents[1], scene_ids=[episode["scene_id"]])[0]
        geometry = cache.get(record, None)
        seeds = [
            int(
                np.random.SeedSequence(
                    [episodes[row]["seed"], attempts_by_row[row], 0x41444D49]
                ).generate_state(1)[0]
            )
            for row in group
        ]
        try:
            candidates = sample_candidates_batch(
                record, [episodes[row]["asset"] for row in group], geometry, seeds, config
            )
            selected, counts = select_and_serialize_batch(
                candidates,
                record,
                [episodes[row]["asset"] for row in group],
                seeds,
                [episodes[row]["support_type"] for row in group],
                config,
            )
        except NoSafeRobotSpawn as error:
            for row in group:
                proposals[row] = NoValidInitialScene(str(error))
            continue
        accepted = []
        for offset, (row, seed, placement) in enumerate(zip(group, seeds, selected, strict=True)):
            if placement is None:
                proposals[row] = NoValidInitialScene(
                    "No valid initial scene in batched resampling",
                    {"counts": {key: value[offset] for key, value in counts.items()}},
                )
                continue
            replacement = {**episodes[row], **placement}
            replacement["seed"] = episodes[row]["seed"]
            replacement["randomization_attempt_seed"] = seed
            accepted.append((row, placement, replacement))
        if not accepted:
            continue
        assets = [
            [d["asset"] for d in episodes[row].get("distractors", [])] for row, _, _ in accepted
        ]
        distractors = sample_nearby_objects_batch(
            [placement for _, placement, _ in accepted],
            assets,
            geometry,
            counts=list(map(len, assets)),
            fixed_sequence=True,
        )
        for (row, _, replacement), asset_row, (objects, diagnostics) in zip(
            accepted, assets, distractors, strict=True
        ):
            if len(objects) != len(asset_row):
                proposals[row] = NoValidInitialScene(
                    "Installed distractor assets have no complete supported placement",
                    diagnostics,
                )
            else:
                replacement["distractors"] = objects
                replacement["distractor_sampling"] = diagnostics
                proposals[row] = replacement
    return proposals


def admit_pick_batch(runner, episodes, geometries, output, *, attempts=16, observer=None):
    if attempts < 1:
        raise ValueError("Scene admission requires at least one attempt")
    env, count = runner.env, runner.env.num_envs
    basket = (
        BasketGeometry.from_robot_usd(device=env.device)
        if episodes[0].get("task_type") == "pick_basket"
        else None
    )
    collision = None
    state = env.get_control_state()
    limits = torch.tensor(
        read_arm_joint_limits(ROBOT_URDF).copy(),
        device=env.device,
        dtype=state["root_states"].dtype,
    )
    env.pick_arm_joint_limits = limits
    schedule = AdmissionSchedule(
        count,
        env.device,
        attempts=attempts,
        warmup_steps=300 if basket is not None else 80,
        final_warmup_steps=300,
    )
    admitted = schedule.admitted
    cached_valid = torch.zeros_like(admitted)
    cached_orientation = torch.zeros(count, 4, device=env.device)
    episode_seeds = torch.tensor(
        [e["seed"] for e in episodes], device=env.device, dtype=torch.int64
    )
    phase_timings = []
    history, checks, planning = [[] for _ in episodes], [None] * count, list(geometries)
    views = [
        EpisodeEnvironmentView(
            env, row, state, distractor_count=len(episode.get("distractors", []))
        )
        for row, episode in enumerate(episodes)
    ]
    names = list(env.contact_sensor.body_names)
    nonfoot = torch.tensor(
        [i for i, name in enumerate(names) if not name.lower().endswith("foot")], device=env.device
    )
    basket_inputs = (
        BasketAdmissionInputs(episodes, state["distractor_states"].shape[1] + 1, env.device)
        if basket is not None
        else None
    )
    basket_geometry = (
        BasketGeometryBatch(
            [
                SimpleNamespace(static_scene=geometry, basket=basket, episode=episode)
                for geometry, episode in zip(geometries, episodes)
            ]
        )
        if basket is not None
        else None
    )
    external_rooms = BatchedSceneGeometry.from_views(geometries) if basket is None else None
    external_geometry = (
        SceneObjectGeometryBatch(episodes, external_rooms) if basket is None else None
    )
    robot_geometry = None
    resampling_scenes = AdmissionSceneCache(episodes, geometries) if basket is None else None
    origins_host = to_host(env.scene.env_origins)
    row_ids = torch.arange(count, device=env.device)
    reference = {
        key: state[key].clone()
        for key in ("root_states", "object_state", "distractor_states", "dof_pos")
    }
    base_steps = torch.zeros(count, device=env.device, dtype=torch.int32)
    object_steps, other_steps = torch.zeros_like(base_steps), torch.zeros_like(base_steps)
    maximum = state["distractor_states"].shape[1]
    mask = (
        torch.arange(maximum, device=env.device)[None]
        < torch.tensor([len(item.get("distractors", [])) for item in episodes], device=env.device)[
            :, None
        ]
    )
    payload_window = (
        TensorStabilityWindow(count * (maximum + 1), env.device) if basket is not None else None
    )
    layout_validator = None
    first_pose_stable = torch.full_like(schedule.age, -1)
    pose_stable_frames = torch.zeros_like(schedule.age)
    stable = torch.zeros_like(admitted)
    position_spans = orientation_spans = None

    def publish_timing(timing):
        phase_timings.append(timing)
        (output / "admission_timing.json").write_text(
            json.dumps(phase_timings, separators=(",", ":")) + "\n"
        )
        print(
            f"[task:init] cohort={timing['cohort']} admitted={int(admitted.sum())}/{count} "
            f"timing={timing}",
            flush=True,
        )

    maximum_cohorts = schedule.max_physics_steps // schedule.cadence + 2 * attempts * count + 2
    # Failed ready rows can resample while partners settle. Cached plans are
    # checked again against the current state before the shared batch is released.
    for cohort in range(maximum_cohorts):
        cohort_started = time.perf_counter()
        if schedule.tick >= schedule.max_physics_steps:
            schedule.expire_shared_budget()
        finishing = cohort > 0 and not bool(schedule.active.any())
        pending = row_ids if cohort == 0 else schedule.layout_rows()
        proposed = torch.zeros_like(admitted)
        proposed[pending] = True
        reset_rows = pending
        if cohort and len(pending):
            if basket is not None and layout_validator is None:
                layout_validator = BasketLayoutPathValidator(
                    [
                        [episode["asset"], *[d["asset"] for d in episode.get("distractors", [])]]
                        for episode in episodes
                    ],
                    [0] * count,
                    basket,
                    state["dof_pos"][:, env.arm_joint_ids],
                    limits,
                    collision,
                    geometry_factory=basket_geometry,
                    admission_inputs=basket_inputs,
                    dt=float(env.dt),
                    proposal_seeds=episode_seeds,
                )
            pending_host = to_host(
                {"rows": pending, "attempts": schedule.attempt_index[pending] + 1}
            )
            pending_rows = pending_host["rows"]
            try:
                proposals = (
                    _external_resample_proposals(
                        episodes, pending_rows, resampling_scenes, pending_host["attempts"]
                    )
                    if basket is None
                    else resample_basket_payloads(
                        episodes,
                        pending_rows,
                        basket,
                        schedule.attempt_index[pending] + 1,
                        collision=collision,
                        limits=limits,
                        initial_joints=state["dof_pos"][pending][:, env.arm_joint_ids],
                        layout_validator=layout_validator,
                        scene_rows=pending,
                    )
                )
            except NoValidInitialScene as error:
                proposals = dict.fromkeys(pending_rows, error)
            for request_index, row in enumerate(pending_rows):
                episode = episodes[row]
                try:
                    replacement = proposals[row]
                    if isinstance(replacement, NoValidInitialScene):
                        raise replacement
                    episode.clear()
                    episode.update(replacement)
                    target = output / f"seed_{episode['seed']}/resample_proposal.json"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(episode, separators=(",", ":")) + "\n")
                except NoValidInitialScene as error:
                    proposed[row] = False
                    checks[row] = {
                        "attempt": pending_host["attempts"][request_index],
                        "cohort": cohort,
                        "shared_physics_step": schedule.tick,
                        "accepted": False,
                        "reason": str(error),
                        "details": error.details,
                        "checks": {"nonpenetrating_resample_available": False},
                    }
                    history[row].append(checks[row])
                    target = output / f"seed_{episode['seed']}"
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "initial_attempts.json").write_text(
                        json.dumps(history[row], separators=(",", ":")) + "\n"
                    )
                    (target / "initial_acceptance.json").write_text(
                        json.dumps(checks[row], separators=(",", ":")) + "\n"
                    )
            reset_rows = schedule.proposals(pending, proposed[pending])
            if not len(reset_rows) and not bool(schedule.waiting.any()):
                continue
        # Reset only new proposals; other rows keep their physical state and history.
        proposal_s = time.perf_counter() - cohort_started
        warmup_started = time.perf_counter()
        preserve_robot = None
        full_reset_rows = reset_rows
        if len(reset_rows) and cohort and basket is not None and hasattr(env, "reset_payload_rows"):
            body_force = state["contact_forces_w"][:, nonfoot].norm(dim=-1).amax(-1)
            preserve_robot = (base_steps >= 25) & (body_force < 20)
            full_reset_rows = reset_rows[~preserve_robot[reset_rows]]
            reset_task_rows(runner, reset_rows, preserve_robot=preserve_robot)
        else:
            reset_task_rows(runner, reset_rows)
        base_steps[full_reset_rows] = 0
        object_steps[reset_rows], other_steps[reset_rows] = 0, 0
        first_pose_stable[reset_rows], pose_stable_frames[reset_rows] = -1, 0
        if payload_window is not None and len(reset_rows):
            payload_rows = (
                reset_rows[:, None] * (maximum + 1)
                + torch.arange(maximum + 1, device=env.device)[None]
            ).flatten()
            payload_window.clear(payload_rows)
        stage = torch.ones(count, device=env.device, dtype=torch.int32)
        warmup_frames = (
            0 if finishing or not bool(schedule.waiting.any()) else int(schedule.next_wait())
        )
        settle = (
            BatchedTaskLoop(
                runner,
                [item["seed"] for item in episodes],
                max_steps=warmup_frames,
                success_stage=2,
                failure_stage=3,
            )
            if warmup_frames
            else None
        )
        buffers = TensorCommandBuffers.from_env(env)
        buffers.base.zero_()
        buffers.arm_joint_mode.fill_(True)
        buffers.arm_joint.copy_(
            torch.tensor([item["initial_arm_joints"] for item in episodes], device=env.device)
        )
        buffers.gripper.fill_(-math.pi / 3)
        state = env.get_control_state()
        for step, _ in () if settle is None else settle.ticks(stage):
            settle.publish(buffers)
            state = settle.step()["state"]
            if observer is not None:
                observer(cohort, step)
            schedule.advance()
            (stable, base_steps, object_steps, other_steps, position_spans, orientation_spans) = (
                update_pick_pose_stability(
                    state, base_steps, object_steps, other_steps, mask, payload_window
                )
            )
            pose_stable_now = stable & (object_steps >= 15) & (other_steps >= 15)
            waiting_ready = pose_stable_now & schedule.waiting
            first_pose_stable = torch.where(
                (first_pose_stable < 0) & waiting_ready, schedule.age, first_pose_stable
            )
            pose_stable_frames += waiting_ready.to(pose_stable_frames.dtype)
        warmup_step_count = 0 if settle is None else settle.physics_steps
        proposed.zero_()
        ready = stable & (object_steps >= 15) & (other_steps >= 15)
        due = (
            torch.where(schedule.waiting)[0]
            if schedule.tick >= schedule.max_physics_steps
            else schedule.validation_rows(ready)
        )
        proposed[due] = True
        warmup_s = time.perf_counter() - warmup_started
        planning_started = time.perf_counter()
        if basket is not None:
            hint_mask, hint_orientation = layout_grasp_hints(episodes, state["root_states"])
            cached_valid[proposed] = hint_mask[proposed]
            cached_orientation[proposed] = hint_orientation[proposed]
        changed = (state["root_states"][:, :7] - reference["root_states"][:, :7]).abs().amax(
            -1
        ) > 0.001
        changed |= (state["object_state"][:, :7] - reference["object_state"][:, :7]).abs().amax(
            -1
        ) > 0.001
        changed |= (state["dof_pos"] - reference["dof_pos"]).abs().amax(-1) > 0.001
        if maximum:
            changed |= (
                state["distractor_states"][..., :7] - reference["distractor_states"][..., :7]
            ).abs().flatten(1).amax(-1) > 0.001
        force = state["contact_forces_w"][:, nonfoot].norm(dim=-1).amax(-1)
        changed |= (
            ~stable
            | (object_steps < 15)
            | (other_steps < 15)
            | (force >= (20 if basket is not None else 5))
        )
        payload_states = torch.cat((state["object_state"][:, None], state["distractor_states"]), 1)
        active_payloads = torch.cat(
            (torch.ones(count, 1, device=env.device, dtype=torch.bool), mask), 1
        )
        root = state["root_states"]
        relative_linear = (
            payload_states[..., 7:10]
            - root[:, None, 7:10]
            - torch.cross(
                root[:, None, 10:13].expand_as(payload_states[..., :3]),
                payload_states[..., :3] - root[:, None, :3],
                dim=-1,
            )
        )
        initial_values = {
            "warmup_pose_stable_first_step": first_pose_stable,
            "warmup_pose_stable_frames_total": pose_stable_frames,
            "warmup_elapsed_physics_steps": schedule.age,
            "nonfoot_contact_force_n": force,
            "base_stable_frames": base_steps,
            "object_stable_frames": object_steps,
            "distractors_stable_frames": other_steps,
            "root_state_world": root,
            "actual_joint_positions_rad": state["dof_pos"],
            "ee_position_world_m": state["ee_pos"],
            "ee_orientation_world_wxyz": state["ee_orn"],
            "initial_arm_joint_target_rad": buffers.arm_joint,
            "target_state_world": state["object_state"],
            "distractor_states_world": state["distractor_states"],
            "payload_present_mask": active_payloads,
            "payload_linear_speeds_world_m_s": payload_states[..., 7:10].norm(dim=-1),
            "payload_angular_speeds_world_rad_s": payload_states[..., 10:13].norm(dim=-1),
            "payload_linear_speeds_relative_to_base_m_s": relative_linear.norm(dim=-1),
            "payload_angular_speeds_relative_to_base_rad_s": (
                payload_states[..., 10:13] - root[:, None, 10:13]
            ).norm(dim=-1),
        }
        if "dof_vel" in state:
            initial_values["actual_joint_velocities_rad_s"] = state["dof_vel"]
        if payload_window is not None:
            initial_values.update(
                payload_position_spans_relative_to_base_m=position_spans.masked_fill(
                    ~active_payloads, torch.nan
                ),
                payload_orientation_spans_relative_to_base_rad=orientation_spans.masked_fill(
                    ~active_payloads, torch.nan
                ),
                payload_pose_window_frames=payload_window.count.reshape(count, maximum + 1),
            )
        for name in (
            "payload_linear_speeds_world_m_s",
            "payload_angular_speeds_world_rad_s",
            "payload_linear_speeds_relative_to_base_m_s",
            "payload_angular_speeds_relative_to_base_rad_s",
        ):
            initial_values[name] = initial_values[name].masked_fill(~active_payloads, torch.nan)
        if finishing:
            proposed |= admitted & changed
        if not bool(proposed.any()):
            if finishing:
                break
            publish_timing(
                {
                    "cohort": cohort,
                    "reset_environments": len(reset_rows),
                    "reset_robots": len(full_reset_rows),
                    "payload_only_resets": len(reset_rows) - len(full_reset_rows),
                    "retained_environments": count - len(reset_rows),
                    "proposal_s": proposal_s,
                    "physics_steps": warmup_step_count,
                    "warmup_s": warmup_s,
                    "planning_and_transfer_s": time.perf_counter() - planning_started,
                    "export_s": 0.0,
                }
            )
            continue
        # Collider samples are immutable asset-local data, shared by retries.
        if collision is None:
            collision = ArmCollisionSamples(env)
        if basket is not None:
            basket_predicates, basket_metrics = basket_object_checks_batch(
                basket, state, *basket_inputs.hulls
            )
            batch_force = state["contact_forces_w"][:, nonfoot].norm(dim=-1).amax(-1)
            basket_predicates.update(
                robot_stable_25_frames=stable,
                all_objects_stable_15_frames=(object_steps >= 15) & (other_steps >= 15),
                no_unexpected_nonfoot_contact=batch_force < 20,
            )
            candidates = torch.where(
                proposed & torch.stack(list(basket_predicates.values())).all(0)
            )[0]
            scene_geometry = basket_geometry.update(state)
            access_by_row = {}
            warm_probe_by_row = {}
            contact_probe_by_row = {}
            accessible = torch.zeros(count, device=env.device, dtype=torch.bool)
            if len(candidates):
                target_geometry, target_surface = basket_geometry.with_target(state)
            for probe in ("hint", "contacts", "full"):
                warm_start = probe == "hint"
                if probe == "contacts" and not state["root_states"].is_cuda:
                    continue
                selected = candidates[
                    cached_valid[candidates] if warm_start else ~accessible[candidates]
                ]
                if not len(selected):
                    continue
                options = (
                    {
                        "candidate_orientations": cached_orientation[selected, None],
                        "direct_only": True,
                    }
                    if warm_start
                    else {}
                )
                if probe == "contacts":
                    cached = cached_intrinsic_proposals(basket_inputs, episode_seeds).select(
                        selected
                    )
                    options = {
                        "candidate_bank": cached.candidate_bank(
                            {key: state[key][selected] for key in ("root_states", "object_state")},
                            BasketArmCollisionSamples(collision),
                            scene_geometry.select(selected),
                        ),
                        "direct_only": True,
                    }
                local_accessible, access_report, initial_path = basket_grasp_access_batch(
                    {
                        key: state[key][selected]
                        for key in (
                            "root_states",
                            "object_state",
                            "dof_pos",
                            "ee_pos",
                            "ee_orn",
                            "distractor_states",
                        )
                    },
                    basket_inputs.sizes[selected],
                    basket_inputs.payload_sizes[selected],
                    basket_inputs.hulls[2][selected],
                    basket,
                    BasketArmCollisionSamples(collision),
                    scene_geometry.select(selected),
                    target_geometry.select(selected),
                    target_surface.select(selected),
                    basket_inputs.support[selected],
                    basket_inputs.support_mask[selected],
                    basket_inputs.centers[selected],
                    limits,
                    float(env.dt),
                    lift_margin=basket_inputs.lift_margin[selected],
                    **options,
                )
                accessible[selected] = local_accessible
                direct = access_report["direct_path_available"]
                cached_valid[selected] = direct
                if "direct_full_path.entry_orientation_world_wxyz" in access_report:
                    cached_orientation[selected[direct]] = access_report[
                        "direct_full_path.entry_orientation_world_wxyz"
                    ][direct]
                access_host = admission_report_to_host(access_report)
                records = {
                    row: (offset, access_report, access_host, initial_path)
                    for offset, row in enumerate(to_host(selected))
                }
                access_by_row.update(records)
                if warm_start:
                    warm_probe_by_row.update(records)
                elif probe == "contacts":
                    contact_probe_by_row.update(records)
            acceptance_tensor = torch.stack(list(basket_predicates.values())).all(0)
            if len(candidates):
                acceptance_tensor[candidates] &= accessible[candidates]
            basket_host = to_host(
                {
                    "accepted": acceptance_tensor,
                    "checks": basket_predicates,
                    "metrics": basket_metrics,
                    "values": initial_values,
                    "accessible": accessible,
                }
            )
        else:
            external_inputs = ExternalAdmissionInputs(
                episodes, env.scene.env_origins, state["distractor_states"].shape[1] + 1
            )
            external_predicates, external_metrics = scene_object_checks_batch(
                state, external_rooms, external_inputs
            )
            scene_geometry = external_geometry.update(state)
            if robot_geometry is None:
                robot_geometry = RobotOcclusionGeometryBatch(env)
            robot_occlusion = robot_geometry.update(env.robot.data.body_state_w)
            eye, heading, fov, self_clear, scene_clear = initial_visibility(
                state, scene_geometry, robot_occlusion
            )
            force = state["contact_forces_w"][:, nonfoot].norm(dim=-1).amax(-1)
            error = (state["object_state"][:, 2] - external_inputs.initial_positions[:, 0, 2]).abs()
            external_predicates.update(
                camera_distance=(eye >= 1) & (eye <= 3),
                heading_within_60deg=heading >= 0.5,
                in_camera_frame=fov,
                robot_not_occluding=self_clear,
                scene_not_occluding=scene_clear,
                object_stable_15_frames=object_steps >= 15,
                robot_stable_25_frames=stable,
                support_error=error < 0.02,
                no_nonfoot_contact=force < 5,
                distractors_stable_15_frames=other_steps >= 15,
            )
            external_acceptance_tensor = torch.stack(list(external_predicates.values())).all(0)
            acceptance_tensor = external_acceptance_tensor
            external_visibility = {
                name: value
                for name, value in {
                    "camera_distance_m": eye,
                    "heading_cosine": heading,
                    "in_camera_frame": fov,
                    "robot_not_occluding": self_clear,
                    "scene_not_occluding": scene_clear,
                }.items()
            }
            external_values = {
                name: value
                for name, value in {
                    "nonfoot_contact_force_n": force,
                    "target_support_error_m": error,
                    "base_stable_frames": base_steps,
                    "object_stable_frames": object_steps,
                    "distractors_stable_frames": other_steps,
                    "root_state_world": state["root_states"],
                    "target_state_world": state["object_state"],
                }.items()
            }
            external_values.update(initial_values)
            external_host = to_host(
                {
                    "accepted": acceptance_tensor,
                    "checks": external_predicates,
                    "visibility": external_visibility,
                    "values": external_values,
                    "metrics": external_metrics,
                }
            )
            external_accepted = external_host["accepted"]
            external_checks = external_host["checks"]
            external_visibility = external_host["visibility"]
            external_values = external_host["values"]
            external_metrics = external_host["metrics"]
        stage_host = to_host(
            {
                "proposed": proposed,
                "base_height": state["root_states"][:, 2],
                "attempt_indices": schedule.attempt_index,
                "row_steps": schedule.age,
            }
        )
        planning_s = time.perf_counter() - planning_started
        export_started = time.perf_counter()
        for row, requested in enumerate(stage_host["proposed"]):
            if not requested:
                continue
            episode = episode_in_world(episodes[row], origins_host[row])
            if basket is not None:
                payloads = 1 + len(episode.get("distractors", []))
                metrics = {
                    name: value[row][:payloads] for name, value in basket_host["metrics"].items()
                }
                metrics["initial_acceptance_values"] = {
                    name: value[row] for name, value in basket_host["values"].items()
                }
                plan = scene_geometry.select(row_ids[row : row + 1])
                plan.basket_static_scene = geometries[row]
                plan.basket_model = basket
                plan.basket_geometry_factory = basket_geometry
                plan.object_support_points = basket_inputs.supports[row]
                plan.initial_grasp_plan = None
                if row in access_by_row:
                    offset, access_report, access_host, initial_path = access_by_row[row]
                    access = admission_report_row(access_report, access_host, offset)
                    metrics["grasp_access"] = access
                    if row in warm_probe_by_row:
                        probe_offset, probe_report, probe_host, _ = warm_probe_by_row[row]
                        metrics["cached_grasp_revalidation"] = admission_report_row(
                            probe_report, probe_host, probe_offset
                        )
                    if row in contact_probe_by_row:
                        probe_offset, probe_report, probe_host, _ = contact_probe_by_row[row]
                        metrics["cached_contact_revalidation"] = admission_report_row(
                            probe_report, probe_host, probe_offset
                        )
                    episode["basket_initial_reposition_required"] = access[
                        "initial_reposition_required"
                    ]
                    episodes[row]["basket_initial_reposition_required"] = episode[
                        "basket_initial_reposition_required"
                    ]
                    if access["direct_path_available"]:
                        plan.initial_grasp_plan = {
                            "paths": {
                                "approach": tuple(
                                    value[offset : offset + 1] for value in initial_path
                                )
                            },
                            "diagnostics": {
                                key.split(".", 1)[1]: value[offset : offset + 1]
                                if torch.is_tensor(value)
                                else value
                                for key, value in access_report.items()
                                if key.startswith("direct_full_path.")
                            },
                            "state": {key: state[key][row : row + 1].clone() for key in reference},
                            "maximum_tool_tilt_deg": 30.0,
                        }
            else:
                plan = scene_geometry.select(row_ids[row : row + 1])
                plan.external_geometry_factory = external_geometry
                metrics = {
                    name: [
                        value[row][: external_inputs.counts[row] - (name == "distractor_states")]
                    ]
                    for name, value in external_metrics.items()
                }
                metrics["initial_visibility"] = {
                    name: values[row] for name, values in external_visibility.items()
                }
                metrics["initial_acceptance_values"] = {
                    name: values[row] for name, values in external_values.items()
                }
            accepted = (
                basket_host["accepted"][row] if basket is not None else external_accepted[row]
            )
            checks[row] = {
                "attempt": stage_host["attempt_indices"][row],
                "cohort": cohort,
                "shared_physics_step": schedule.tick,
                "accepted": accepted,
                "settle_steps": stage_host["row_steps"][row],
                "checks": (
                    {
                        **{key: value[row] for key, value in basket_host["checks"].items()},
                        **(
                            {
                                "target_has_grasp_or_safe_reposition_path": basket_host[
                                    "accessible"
                                ][row]
                            }
                            if row in access_by_row
                            else {}
                        ),
                    }
                    if basket is not None
                    else {key: values[row] for key, values in external_checks.items()}
                ),
                "base_height_m": stage_host["base_height"][row],
                **metrics,
            }
            history[row].append(checks[row])
            planning[row] = plan
            target = output / f"seed_{episodes[row]['seed']}"
            target.mkdir(exist_ok=True)
            (target / "initial_acceptance.json").write_text(
                json.dumps(checks[row], separators=(",", ":")) + "\n"
            )
            (target / "initial_attempts.json").write_text(
                json.dumps(history[row], separators=(",", ":")) + "\n"
            )
            (target / "episode.json").write_text(json.dumps(episode, separators=(",", ":")) + "\n")
        validated = torch.where(proposed)[0]
        schedule.finish(validated, acceptance_tensor[validated])
        revalidated = admitted & proposed
        for key in reference:
            reference[key][revalidated] = state[key][revalidated]
        timing = {
            "cohort": cohort,
            "final_revalidation": finishing,
            "reset_environments": len(reset_rows),
            "reset_robots": len(full_reset_rows),
            "payload_only_resets": len(reset_rows) - len(full_reset_rows),
            "retained_environments": count - len(reset_rows),
            "proposal_s": proposal_s,
            "physics_steps": warmup_step_count,
            "warmup_s": warmup_s,
            "planning_and_transfer_s": planning_s,
            "export_s": time.perf_counter() - export_started,
        }
        publish_timing(timing)
    else:
        raise RuntimeError("Admission scheduling exceeded its finite cohort bound")
    for view in views:
        view.state = state
    env.pick_initial_admitted = admitted.clone()
    return state, views, collision, checks, planning
