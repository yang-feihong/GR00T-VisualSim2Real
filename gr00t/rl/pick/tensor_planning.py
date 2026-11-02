"""Pick stage-entry planners; the shared runtime owns policy/physics stepping."""

import copy

import torch
from torch.nn.utils.rnn import pad_sequence

from gr00t.rl.auto_open.motion_math import _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.auto_open.stages import StagePlanningEvents
from gr00t.rl.pick.basket_carry import plan_basket_carry_batch
from gr00t.rl.pick.basket_clear import plan_clear_for_nudge_batch
from gr00t.rl.pick.basket_controller import basket_required_height
from gr00t.rl.pick.basket_entry import transport_entry_orientation
from gr00t.rl.pick.basket_geometry import BasketGeometryBatch
from gr00t.rl.pick.basket_grasp import plan_basket_grasp_batch
from gr00t.rl.pick.basket_push import plan_closed_nudge_batch
from gr00t.rl.pick.stages import BasketStage
from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
from gr00t.rl.pick.gaze import plan_approach_gaze_batch
from gr00t.rl.pick.grasp_inverse import solve_default_grasp_inverse
from gr00t.rl.pick.grasp_paths import score_default_grasp_paths
from gr00t.rl.pick.measured_commands import load_measured_commands
from gr00t.rl.pick.measured_station_batch import select_measured_station_batch
from gr00t.rl.pick.planning import grasp_candidate_bank, plan_carry_batch
from gr00t.rl.pick.station_batch import STATION_POLICIES, select_initial_pick_station_batch

_BASKET_PLANNING_FAILURES = {
    "lift": {
        1: "no_collision_free_held_object_lift",
        2: "no_robust_actual_body_lift_path",
        3: "held_lift_path_exceeds_capacity",
        4: "held_lift_dense_geometry_failed",
    },
    "clear_for_reposition": {
        1: "no_open_jaw_clear_geometry",
        2: "no_open_jaw_clear_inverse_path",
        3: "open_jaw_clear_path_exceeds_capacity",
        4: "open_jaw_clear_dense_geometry",
    },
    "reposition_object": {
        1: "no_improving_collision_free_basket_shift",
        2: "no_shift_improves_grasp_access_or_space",
        3: "no_closed_jaw_flank_entry",
        4: "no_closed_nudge_inverse_path",
        5: "closed_nudge_path_exceeds_capacity",
        6: "closed_nudge_dense_clearance_failed",
    },
}


class EpisodeEnvironmentView:
    """A stage planner's row view, never a separate simulator or policy runner."""

    def __init__(self, env, row, state, *, distractor_count):
        self.batch, self.row, self.state = env, row, state
        self.num_envs = 1
        self.distractor_count = distractor_count
        self.gripper_control_options = dict(env.gripper_control_options)

    def __getattr__(self, name):
        value = getattr(self.batch, name)
        if name in ("robot", "pick_object"):
            return _AssetRow(value, self.row)
        if name == "pick_distractors":
            return [_AssetRow(item, self.row) for item in value[: self.distractor_count]]
        if torch.is_tensor(value) and name not in (
            "arm_joint_ids",
            "gripper_joint_ids",
            "dof_pos_limits",
            "pick_arm_joint_limits",
        ):
            if value.ndim and value.shape[0] == self.batch.num_envs:
                return value[self.row : self.row + 1]
        return value

    def get_control_state(self):
        state = {
            name: value[self.row : self.row + 1]
            if torch.is_tensor(value) and value.ndim and value.shape[0] == self.batch.num_envs
            else value
            for name, value in self.state.items()
        }
        state["distractor_states"] = state["distractor_states"][:, : self.distractor_count]
        return state

    def get_goal_reference_origin(self):
        return self.batch.get_goal_reference_origin()[self.row : self.row + 1]

    def get_goal_reference_quat(self):
        return self.batch.get_goal_reference_quat()[self.row : self.row + 1]

    def get_goal_center_offset_local(self):
        return self.batch.get_goal_center_offset_local()[self.row : self.row + 1]


class _DataRow:
    def __init__(self, data, row):
        self.source, self.row = data, row

    def __getattr__(self, name):
        value = getattr(self.source, name)
        return value[self.row : self.row + 1] if torch.is_tensor(value) and value.ndim else value


class _AssetRow:
    def __init__(self, asset, row):
        self.source, self.row = asset, row
        self.data = _DataRow(asset.data, row)

    def write_root_state_to_sim(self, state, env_ids=None):
        ids = torch.tensor([self.row], device=state.device)
        self.source.write_root_state_to_sim(state, env_ids=ids)


def episode_in_world(episode, origin):
    result = copy.deepcopy(episode)
    from gr00t.rl.auto_open.tensor_metadata import to_host

    offset = to_host(origin) if torch.is_tensor(origin) else origin
    for descriptor in (result, *result.get("distractors", [])):
        descriptor["object_position"] = [
            a + b for a, b in zip(descriptor["object_position"], offset)
        ]
        descriptor["support_z"] += offset[2]
    for name in ("base_xy", "station_xy"):
        result[name] = [a + b for a, b in zip(result[name], offset[:2])]
    result["batch_scene_origin_m"] = offset
    return result


class TensorPickPlanningCoordinator(StagePlanningEvents):
    planning_events = {
        "move_to_grasp_plan": "plan_move_to_grasp",
        "grasp_object_plan": None,
        "lift_object_plan": "plan_lift",
        "withdraw_with_object_plan": "plan_withdraw",
        "verify_retention_plan": None,
    }

    def __init__(self, controllers):
        self.controllers = controllers
        self.planning_events = dict(type(self).planning_events)
        self.external_geometry = getattr(controllers[0].geometry, "external_geometry_factory", None)
        self.carry_geometry = (
            self.external_geometry.geometry if self.external_geometry is not None else None
        )
        self.basket_geometry = getattr(controllers[0].geometry, "basket_geometry_factory", None)
        self.carry_diagnostics = []
        self.grasp_metadata = None
        if getattr(controllers[0], "basket", None) is not None:
            # Basket entry must refresh and execute insertion before closure.
            self.planning_events["grasp_object_plan"] = "plan_grasp_entry"
            self.planning_events["lift_object_plan"] = "plan_basket_lift"
            self.planning_events["move_to_grasp_plan"] = "plan_basket_grasp"
            self.planning_events["clear_for_reposition_plan"] = "plan_basket_clear"
            self.planning_events["open_for_reposition_plan"] = "plan_basket_open"
            self.planning_events["close_for_reposition_plan"] = "plan_basket_hold"
            self.planning_events["reposition_object_plan"] = "plan_basket_push"
            self.planning_events["reopen_after_reposition_plan"] = "plan_basket_reopen"

    def _external_scene_geometry(self, state=None):
        factory = getattr(self, "external_geometry", None)
        if factory is not None:
            self.carry_geometry = factory.geometry if state is None else factory.update(state)
        elif self.carry_geometry is None:
            self.carry_geometry = BatchedSceneGeometry.from_views(
                [c.geometry for c in self.controllers]
            )
        return self.carry_geometry

    def _cache_initial_joint_references(self, fsm):
        if hasattr(self, "initial_joint_references"):
            return
        one = torch.ones(1, device=fsm.device, dtype=torch.bool)
        self.initial_joint_references = torch.cat(
            [
                c.initial_grasp_joint_reference
                if c.initial_grasp_joint_reference is not None
                else c.initial_joints
                for c in self.controllers
            ]
        )
        self.initial_joint_reference_mask = torch.tensor(
            [c.initial_grasp_joint_reference is not None for c in self.controllers],
            device=fsm.device,
        )
        self.initial_joint_reference_mask &= torch.cat(
            [
                getattr(
                    c,
                    "initial_grasp_joint_reference_valid",
                    getattr(c, "initial_station_valid", one),
                )
                for c in self.controllers
            ]
        )

    def install_initial_external_gaze(self, fsm):
        """Pool external gaze initialization before the first common control tick."""
        configured = [
            (row, controller)
            for row, controller in enumerate(self.controllers)
            if hasattr(controller, "approach_root_path")
        ]
        if not configured:
            return
        self._external_scene_geometry()
        self._cache_initial_joint_references(fsm)
        rows = torch.tensor([row for row, _ in configured], device=fsm.device)
        paths = [c.approach_root_path[0] for _, c in configured]
        lengths = torch.cat(
            [
                c.approach_path_length
                if hasattr(c, "approach_path_length")
                else torch.tensor([len(path)], device=fsm.device)
                for (_, c), path in zip(configured, paths)
            ]
        )
        roots = pad_sequence(paths, batch_first=True)
        active = torch.arange(roots.shape[1], device=fsm.device)[None] < lengths[:, None]
        roots = torch.where(
            active[..., None],
            roots,
            roots[torch.arange(len(rows), device=fsm.device), lengths - 1, None],
        )
        position, orientation, diagnostics = plan_approach_gaze_batch(
            roots,
            torch.cat([c.initial_joints for _, c in configured]),
            fsm.target[rows],
            self.controllers[0].limits,
            float(fsm.env.dt),
            self.controllers[0].collision,
            self.carry_geometry.select(rows),
            fsm.gaze_offsets[rows],
            lengths=lengths,
            next_grasp_joints=self.initial_joint_references[rows],
            next_grasp_mask=self.initial_joint_reference_mask[rows],
        )
        fsm.arm_path.install_tensor(rows, position, orientation, lengths)
        fsm.approach_gaze_valid[rows] = diagnostics["valid"]
        fsm.fail(rows[~diagnostics["valid"]], 7)
        for key in ("candidate_q2_offset_deg", "candidate_optical_roll_offset_deg"):
            diagnostics[key] = diagnostics[key][None].expand(len(rows), -1)
        diagnostics["wrist_route_enabled"] = self.initial_joint_reference_mask[rows]
        offsets = torch.cat([c.approach_random_offsets for _, c in configured])
        for index, name in enumerate(
            ("base_heading", "optical_yaw", "optical_pitch", "optical_roll")
        ):
            diagnostics["random_offsets_rad." + name] = offsets[:, index]
        self.carry_diagnostics.append((rows, "approach_gaze", diagnostics["valid"], diagnostics))
        for _, controller in configured:
            del controller.approach_root_path
            del controller.approach_random_offsets
            if hasattr(controller, "approach_path_length"):
                del controller.approach_path_length

    def install_initial_external_stations(self, fsm, state):
        """Prepare initial stations and common base paths in one device batch."""
        import time

        started = time.monotonic()
        configured = [
            (row, c)
            for row, c in enumerate(self.controllers)
            if getattr(c, "deferred_initial_station", False)
        ]
        if not configured:
            return
        self._cache_initial_joint_references(fsm)
        rows = torch.tensor([row for row, _ in configured], device=fsm.device)
        roots = state["root_states"][rows]
        destination = roots.new_tensor(
            [[*c.episode["station_xy"], c.episode["station_yaw"]] for _, c in configured]
        )
        offsets = torch.cat([c.approach_random_offsets for _, c in configured])
        destination[:, 2] += offsets[:, 0]
        if self.grasp_metadata is None:
            self._cache_grasp_metadata(fsm)
        _, sizes, hulls, hull_mask, preferred = self.grasp_metadata
        supported = torch.tensor(
            [c.object_support_points is not None for _, c in configured],
            device=fsm.device,
            dtype=torch.bool,
        )
        measured_success = self._select_initial_measured_stations(
            fsm, state, configured, destination
        )
        measured_done = time.monotonic()
        print(f"[pick:planning] measured_stations_s={measured_done - started:.3f}", flush=True)
        ordinary = supported & ~measured_success[rows]
        requested = rows[ordinary]
        if len(requested):
            self._external_scene_geometry(state)
            pose, orientation, joint_reference, valid, diagnostics = (
                select_initial_pick_station_batch(
                    {
                        key: state[key][requested]
                        for key in ("root_states", "object_state", "dof_pos", "ee_orn")
                    }
                    | {
                        "episode_seed": torch.tensor(
                            [c.episode["seed"] for c in self.controllers], device=fsm.device
                        )[requested],
                        "support_z": torch.tensor(
                            [c.episode["support_z"] for c in self.controllers], device=fsm.device
                        )[requested],
                        "on_floor": torch.tensor(
                            [c.episode["support_type"] == "floor" for c in self.controllers],
                            device=fsm.device,
                        )[requested],
                    },
                    destination[ordinary],
                    sizes[requested],
                    hulls[requested],
                    self.controllers[0].limits,
                    self.controllers[0].collision,
                    self.carry_geometry.select(requested),
                    support_mask=hull_mask[requested],
                )
            )
            destination[ordinary] = pose
            preferred[requested] = orientation
            self.grasp_preferred_mask[requested] = valid
            self.initial_joint_references[requested] = joint_reference
            self.initial_joint_reference_mask[requested] = valid
            self.carry_diagnostics.append((requested, "initial_station", valid, diagnostics))
            diagnostics["measured_reach_fallback_code"] = torch.tensor(
                [
                    0
                    if getattr(c, "measured_reach", None) is None
                    else 2
                    if c.measured_reach.compatible
                    else 1
                    for c in self.controllers
                ],
                device=fsm.device,
            )[requested]
        print(
            f"[pick:planning] ordinary_stations_s={time.monotonic() - measured_done:.3f}",
            flush=True,
        )
        initial_yaw = roots.new_tensor([c.episode["base_yaw"] for _, c in configured])
        fsm.base_path.install(
            rows,
            torch.cat((roots[:, :2], initial_yaw[:, None]), -1),
            destination[:, None],
            float(fsm.env.dt),
        )
        # One batch-wide allocation bound, not a per-environment planning decision.
        count = int(fsm.base_path.length[rows].max())
        planned = fsm.base_path.poses[rows, :count]
        root_path = roots[:, None].expand(-1, count, -1).clone()
        root_path[..., :2] = planned[..., :2]
        delta = planned[..., 2] - initial_yaw[:, None]
        rotation = torch.stack(((delta / 2).cos(), delta * 0, delta * 0, (delta / 2).sin()), -1)
        root_path[..., 3:7] = _quat_multiply(rotation, root_path[..., 3:7])
        fsm.hold_xy[rows], fsm.hold_yaw[rows] = destination[:, :2], destination[:, 2]
        for index, (row, controller) in enumerate(configured):
            controller.approach_root_path = root_path[index : index + 1]
            controller.approach_path_length = fsm.base_path.length[row : row + 1].clone()
            controller.hold_xy = destination[index : index + 1, :2]
            controller.hold_yaw = destination[index : index + 1, 2]
            if controller.object_support_points is not None:
                controller.initial_grasp_orientation = preferred[row : row + 1]
                controller.initial_station_valid = self.grasp_preferred_mask[row : row + 1]
                controller.initial_grasp_joint_reference = self.initial_joint_references[
                    row : row + 1
                ]
                controller.initial_grasp_joint_reference_valid = self.initial_joint_reference_mask[
                    row : row + 1
                ]
            del controller.deferred_initial_station

    def _select_initial_measured_stations(self, fsm, state, configured, destination):
        """Group immutable catalogs, while all per-scene admission stays on device."""
        success = torch.zeros(fsm.num_envs, device=fsm.device, dtype=torch.bool)
        catalogs = {}
        for index, (row, controller) in enumerate(configured):
            measured = getattr(controller, "measured_reach", None)
            if (
                measured is not None
                and measured.compatible
                and controller.object_support_points is not None
            ):
                count = len(controller.object_support_points)
                hull_bucket = 1 << (count - 1).bit_length()
                catalogs.setdefault((str(measured.catalog), hull_bucket), []).append(
                    (index, row, controller)
                )
        if not catalogs:
            return success
        self._external_scene_geometry(state)
        _, _, hulls, hull_mask, preferred = self.grasp_metadata
        for (catalog, _), group in catalogs.items():
            rows = torch.tensor([row for _, row, _ in group], device=fsm.device)
            local_rows = torch.tensor([index for index, _, _ in group], device=fsm.device)
            floor = destination.new_tensor([c.episode["support_z"] for _, _, c in group])
            commands = load_measured_commands([catalog], state["root_states"])
            # Immutable sample counts are metadata, not GPU decisions. Avoid
            # evaluating every small fruit against the batch's largest hull pad.
            hull_width = max(len(c.object_support_points) for _, _, c in group)
            pose, orientation, depth, valid, diagnostics = select_measured_station_batch(
                {key: state[key][rows] for key in ("root_states", "object_state", "ee_orn")},
                destination[local_rows],
                hulls[rows, :hull_width],
                hull_mask[rows, :hull_width],
                floor,
                commands,
                self.controllers[0].collision,
                self.carry_geometry.select(rows),
            )
            destination[local_rows] = pose
            success[rows] = valid
            preferred[rows] = orientation
            self.grasp_preferred_mask[rows] = valid
            self.measured_grasp_mask[rows] = valid
            self.measured_grasp_orientations[rows] = orientation
            self.measured_grasp_depth[rows] = depth
            self.initial_joint_reference_mask[rows] = False
            diagnostics["command_catalog_sha256"] = group[0][2].measured_reach.catalog_sha256
            self.carry_diagnostics.append((rows, "measured_station", valid, diagnostics))
            for _, row, controller in group:
                controller.measured_reach.orientation = self.measured_grasp_orientations[
                    row : row + 1
                ]
                controller.measured_reach.depth = self.measured_grasp_depth[row]
                controller.measured_reach.valid = self.measured_grasp_mask[row : row + 1]
        return success

    def install_initial_basket_plans(self, fsm, state):
        """Install unchanged prefixes; refresh stale plans once after the final warm-up."""
        configured = [
            (row, c)
            for row, c in enumerate(self.controllers)
            if c.current_stage == BasketStage.MOVE_TO_GRASP
        ]
        if any(c.initial_grasp_plan is None for _, c in configured):
            raise ValueError("Admitted basket approach is missing its verified initial path")
        if configured:
            rows = torch.tensor([row for row, _ in configured], device=fsm.device)
            caches = [c.initial_grasp_plan for _, c in configured]
            unchanged = torch.ones(len(rows), device=fsm.device, dtype=torch.bool)
            for key in ("root_states", "dof_pos", "object_state", "distractor_states"):
                expected = torch.cat([cache["state"][key] for cache in caches])
                same = state[key][rows] == expected
                if key == "distractor_states":
                    same |= ~fsm.payload_mask[rows, 1:, None]
                unchanged &= same.flatten(1).all(-1)
            fresh = rows[unchanged]
            if len(fresh):
                p, q, lengths = (
                    torch.cat([cache["paths"]["approach"][i] for cache in caches])[unchanged]
                    for i in range(3)
                )
                fsm.arm_path.install_tensor(fresh, p, q, lengths)
                fsm.awaiting_entry_refresh[fresh] = True
                fsm.basket_entry_orientation[fresh] = q[:, -1]
                fsm.basket_entry_object_orientation[fresh] = state["object_state"][fresh, 3:7]
                fsm.tried_grasp_orientations[fresh, 0] = q[:, -1]
                fsm.tried_grasp_mask[fresh, 0] = True
                diagnostic = {
                    key: (
                        torch.cat([cache["diagnostics"][key] for cache in caches])
                        if value.ndim < 2
                        else pad_sequence(
                            [cache["diagnostics"][key][0] for cache in caches],
                            batch_first=True,
                            padding_value=(
                                torch.nan
                                if value.is_floating_point()
                                else 0
                                if value.dtype == torch.bool
                                else -1
                            ),
                        )
                    )[unchanged]
                    if torch.is_tensor(value)
                    else value
                    for key, value in caches[0]["diagnostics"].items()
                }
                diagnostic["reused_initial_admission_plan"] = True
                fsm.basket_entry_depth_fraction[fresh] = diagnostic.get(
                    "selected_grasp_depth_fraction", fsm.basket_entry_depth_fraction[fresh]
                )
                self.carry_diagnostics.append(
                    (
                        fresh.clone(),
                        "grasp",
                        torch.ones(len(fresh), device=fsm.device, dtype=torch.bool),
                        diagnostic,
                    )
                )
            stale = rows[~unchanged]
            if len(stale):
                fsm.stage_entry_from[stale] = int(BasketStage.APPROACH_OBJECT)
                fsm.tried_grasp_mask[stale] = False
                options = {}
                if all(
                    "entry_orientation_world_wxyz" in cache.get("diagnostics", {})
                    for cache in caches
                ):
                    options["initial_orientations"] = torch.cat(
                        [cache["diagnostics"]["entry_orientation_world_wxyz"] for cache in caches]
                    )[~unchanged]
                self.plan_basket_grasp(fsm, state, stale, initial_refresh=True, **options)
            for _, controller in configured:
                controller.initial_grasp_plan = None
                controller.geometry.initial_grasp_plan = None
        nudge = [
            row
            for row, c in enumerate(self.controllers)
            if c.current_stage == BasketStage.OPEN_FOR_REPOSITION
        ]
        if nudge:
            self.plan_basket_open(fsm, state, torch.tensor(nudge, device=fsm.device))

    def plan_grasp_entry(self, fsm, state, rows):
        refresh = rows[fsm.awaiting_entry_refresh[rows]]
        if len(refresh):
            fsm.basket_entry_orientation[refresh] = transport_entry_orientation(
                fsm.basket_entry_orientation[refresh],
                fsm.basket_entry_object_orientation[refresh],
                state["object_state"][refresh, 3:7],
            )
            fsm.awaiting_entry_refresh[refresh] = False
            fsm.basket_final_entry[refresh] = True
            fsm.basket_entry_refresh_count[refresh] += 1
            fsm.tried_grasp_mask[refresh] = False
            fsm.transition(refresh, BasketStage.MOVE_TO_GRASP)
            self.plan_basket_grasp(fsm, state, refresh)

    def plan_basket_grasp(
        self, fsm, state, rows, *, initial_refresh=False, initial_orientations=None
    ):
        if not len(rows):
            return
        if initial_orientations is not None and (
            not initial_refresh or initial_orientations.shape != (len(rows), 4)
        ):
            raise ValueError("Admission orientation hints require one quaternion per initial row")
        if self.grasp_metadata is None:
            self._cache_grasp_metadata(fsm)
        if self.basket_geometry is None:
            self.basket_geometry = BasketGeometryBatch(self.controllers)
        geometry = self.basket_geometry.update(state)
        retry = fsm.stage_entry_from[rows] == int(BasketStage.MOVE_TO_GRASP)
        fsm.grasp_retries[rows] += retry.long()
        from_reposition = rows[
            fsm.stage_entry_from[rows] == int(BasketStage.REOPEN_AFTER_REPOSITION)
        ]
        fsm.tried_grasp_mask[from_reposition] = False
        fsm.grasp_retries[from_reposition] = 0
        fsm.basket_final_entry[from_reposition] = False
        for final_entry in (False, True):
            selected = rows[fsm.basket_final_entry[rows] == final_entry]
            if len(selected):
                if initial_orientations is not None and not final_entry:
                    valid = self._plan_basket_grasp_group(
                        fsm,
                        state,
                        selected,
                        geometry,
                        final_entry,
                        initial_refresh=True,
                        candidate_orientations=initial_orientations[
                            ~fsm.basket_final_entry[rows], None
                        ],
                        probe_only=True,
                    )
                    selected = selected[~valid]
                if not len(selected):
                    continue
                self._plan_basket_grasp_group(
                    fsm, state, selected, geometry, final_entry, initial_refresh=initial_refresh
                )

    def _plan_basket_grasp_group(
        self,
        fsm,
        state,
        rows,
        geometry,
        final_entry,
        *,
        initial_refresh=False,
        candidate_orientations=None,
        probe_only=False,
    ):
        local = {
            key: state[key][rows]
            for key in ("root_states", "ee_pos", "ee_orn", "object_state", "dof_pos")
        }
        first = self.controllers[0]
        minimum = basket_required_height(
            first.basket,
            local,
            fsm.object_support[rows],
            fsm.basket_lift_margin[rows],
            support_mask=fsm.object_support_mask[rows],
        )
        previous_end = (fsm.arm_path.length[rows] - 1).clamp_min(0).long()
        previous_error = (local["ee_pos"] - fsm.arm_path.positions[rows, previous_end]).norm(dim=-1)
        path, valid, diagnostics = plan_basket_grasp_batch(
            local,
            self.grasp_metadata[1][rows],
            first.limits,
            first.collision,
            geometry.select(rows),
            float(fsm.env.dt),
            support_points=fsm.object_support[rows],
            support_mask=fsm.object_support_mask[rows],
            initial_height=fsm.initial_object_z[rows],
            minimum_height=minimum,
            prefer_realized_body=fsm.grasp_retries[rows] > 0,
            realized_body=fsm.grasp_retries[rows] > 0,
            excluded=fsm.tried_grasp_orientations[rows],
            excluded_mask=fsm.tried_grasp_mask[rows],
            entry_orientation=fsm.basket_entry_orientation[rows] if final_entry else None,
            **(
                {"entry_depth_fraction": fsm.basket_entry_depth_fraction[rows]}
                if final_entry
                else {}
            ),
            maximum_tool_tilt_deg=float(getattr(first, "maximum_tool_tilt_deg", 30.0)),
            capacity=fsm.arm_path.capacity,
            **(
                {"candidate_orientations": candidate_orientations}
                if candidate_orientations is not None
                else {}
            ),
        )
        diagnostics.update(
            initial_admission_cache_refreshed=initial_refresh,
            initial_admission_orientation_probe=probe_only,
            execution_entry_is_retry=fsm.stage_entry_from[rows] == int(BasketStage.MOVE_TO_GRASP),
            entry_previous_goal_error_m=previous_error,
            entry_base_position_world=local["root_states"][:, :3],
            grasp_retry_index=fsm.grasp_retries[rows].clone(),
            entry_refresh_count=fsm.basket_entry_refresh_count[rows].clone(),
        )
        fsm.hold_xy[rows] = local["root_states"][:, :2]
        w, x, y, z = local["root_states"][:, 3:7].unbind(-1)
        fsm.hold_yaw[rows] = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        fsm.arm_path.install_tensor(rows, *path)
        fsm.stage_steps[rows] = 0
        fsm.settled[rows] = 0
        fsm.grasp_retry_steps[rows] = 0
        fsm.arm_stability.clear(rows)
        fsm.awaiting_entry_refresh[rows] = valid & ~torch.full_like(valid, final_entry)
        fsm.basket_entry_orientation[rows] = diagnostics["entry_orientation_world_wxyz"]
        fsm.basket_entry_object_orientation[rows] = local["object_state"][:, 3:7]
        fsm.basket_entry_depth_fraction[rows] = diagnostics.get(
            "selected_grasp_depth_fraction", fsm.basket_entry_depth_fraction[rows]
        )
        accepted = rows[valid]
        slots = fsm.tried_grasp_mask[accepted].sum(-1)
        torch._assert_async(
            (slots < fsm.tried_grasp_mask.shape[1]).all(), "Grasp history capacity exceeded"
        )
        fsm.tried_grasp_orientations[accepted, slots] = fsm.basket_entry_orientation[accepted]
        fsm.tried_grasp_mask[accepted, slots] = True
        self.carry_diagnostics.append((rows.clone(), "grasp", valid, diagnostics))
        if probe_only:
            return valid
        rejected = rows[~valid]
        reposition = rejected[fsm.nudge_attempts[rejected] < 2]
        fsm.fail(rejected[fsm.nudge_attempts[rejected] >= 2], 12)
        if len(reposition):
            fsm.transition(reposition, BasketStage.OPEN_FOR_REPOSITION)
            self.plan_basket_open(fsm, state, reposition)
        return valid

    def plan_basket_hold(self, fsm, state, rows):
        fsm.arm_path.install_tensor(
            rows,
            state["ee_pos"][rows, None],
            state["ee_orn"][rows, None],
            torch.ones(len(rows), device=fsm.device, dtype=torch.long),
        )
        fsm.gripper_stability.clear(rows)
        fsm.arm_stability.clear(rows)

    def plan_basket_open(self, fsm, state, rows):
        if not len(rows):
            return
        fsm.nudge_attempts[rows] += 1
        fsm.nudge_current_path_unavailable[rows] = True
        fsm.nudge_performed[rows] = False
        self.plan_basket_hold(fsm, state, rows)
        self.carry_diagnostics.append(
            (
                rows.clone(),
                "reposition_start",
                torch.ones(len(rows), device=fsm.device, dtype=torch.bool),
                {
                    "attempt": fsm.nudge_attempts[rows].clone(),
                    "trigger_stage_id": fsm.stage_entry_from[rows].clone(),
                },
            )
        )

    def plan_basket_push(self, fsm, state, rows):
        if not len(rows):
            return
        if self.grasp_metadata is None:
            self._cache_grasp_metadata(fsm)
        if self.basket_geometry is None:
            self.basket_geometry = BasketGeometryBatch(self.controllers)
        geometry = self.basket_geometry.update(state)
        target_geometry, target_surface = self.basket_geometry.with_target(state)
        local = {
            key: state[key][rows]
            for key in (
                "root_states",
                "ee_pos",
                "ee_orn",
                "object_state",
                "dof_pos",
                "distractor_states",
            )
        }
        first = self.controllers[0]
        path, valid, diagnostics = plan_closed_nudge_batch(
            local,
            fsm.basket_payload_sizes[rows],
            self.grasp_metadata[1][rows],
            fsm.payload_mask[rows],
            first.basket,
            first.limits,
            first.collision,
            geometry.select(rows),
            target_geometry.select(rows),
            target_surface.select(rows),
            float(fsm.env.dt),
            support_points=fsm.object_support[rows],
            support_mask=fsm.object_support_mask[rows],
            current_path_unavailable=fsm.nudge_current_path_unavailable[rows],
            capacity=fsm.arm_path.capacity,
        )
        diagnostics["attempt"] = fsm.nudge_attempts[rows].clone()
        diagnostics["planned_path_valid"] = valid
        fsm.arm_path.install_tensor(rows, *path)
        fsm.nudge_performed[rows] = valid
        accepted = rows[valid]
        fsm.nudge_start[accepted] = local["object_state"][valid, :3]
        if "direction_world" in diagnostics:
            fsm.nudge_direction[accepted] = diagnostics["direction_world"][valid]
        fsm.nudge_space_before[accepted] = diagnostics["space_before_m"][valid]
        not_required = ~valid & diagnostics["not_required"]
        fsm.fail(rows[~valid & ~not_required], 9)
        self.carry_diagnostics.append(
            (rows.clone(), "reposition_object", valid | not_required, diagnostics)
        )
        reopen = rows[not_required]
        if len(reopen):
            fsm.transition(reopen, BasketStage.REOPEN_AFTER_REPOSITION)
            self.plan_basket_reopen(fsm, state, reopen)

    def plan_basket_reopen(self, fsm, state, rows):
        if not len(rows):
            return
        self.plan_basket_hold(fsm, state, rows)
        if self.grasp_metadata is None:
            self._cache_grasp_metadata(fsm)
        if self.basket_geometry is None:
            self.basket_geometry = BasketGeometryBatch(self.controllers)
        geometry = self.basket_geometry.update(state)
        local = {
            key: state[key][rows]
            for key in ("root_states", "ee_pos", "ee_orn", "object_state", "dof_pos")
        }
        first = self.controllers[0]
        minimum = basket_required_height(
            first.basket,
            local,
            fsm.object_support[rows],
            fsm.basket_lift_margin[rows],
            support_mask=fsm.object_support_mask[rows],
        )
        _, valid, diagnostics = plan_basket_grasp_batch(
            local,
            self.grasp_metadata[1][rows],
            first.limits,
            first.collision,
            geometry.select(rows),
            float(fsm.env.dt),
            support_points=fsm.object_support[rows],
            support_mask=fsm.object_support_mask[rows],
            initial_height=fsm.initial_object_z[rows],
            minimum_height=minimum,
            prefer_realized_body=torch.zeros(len(rows), device=fsm.device, dtype=torch.bool),
            realized_body=torch.ones(len(rows), device=fsm.device, dtype=torch.bool),
            maximum_tool_tilt_deg=float(getattr(first, "maximum_tool_tilt_deg", 30.0)),
            capacity=fsm.arm_path.capacity,
        )
        shift = (
            (local["object_state"][:, :3] - fsm.nudge_start[rows]) * fsm.nudge_direction[rows]
        ).sum(-1)
        diagnostics.update(
            attempt=fsm.nudge_attempts[rows].clone(),
            measured_shift_m=torch.where(fsm.nudge_performed[rows], shift, torch.nan),
            nudge_performed=fsm.nudge_performed[rows].clone(),
        )
        self.carry_diagnostics.append((rows.clone(), "reopen_after_reposition", valid, diagnostics))
        rejected = rows[~valid]
        retry = rejected[fsm.nudge_attempts[rejected] < 2]
        fsm.fail(rejected[fsm.nudge_attempts[rejected] >= 2], 9)
        if len(retry):
            fsm.transition(retry, BasketStage.OPEN_FOR_REPOSITION)
            self.plan_basket_open(fsm, state, retry)

    def plan_move_to_grasp(self, fsm, state, rows):
        if self.grasp_metadata is None:
            self._cache_grasp_metadata(fsm)
        eligible = self.grasp_metadata[0]
        retry = fsm.stage_entry_from[rows] == int(BasketStage.MOVE_TO_GRASP)
        fsm.grasp_retries[rows] += retry.long()
        measured = rows[eligible[rows] & self.measured_grasp_mask[rows]]
        rejected = rows[:0]
        if len(measured):
            rejected = self._plan_grasp_group(fsm, state, measured, measured=True)
        ordinary = rows[eligible[rows] & ~self.measured_grasp_mask[rows]]
        ordinary = torch.cat((ordinary, rejected))
        for prefer_orientation in (True, False):
            mask = self.grasp_preferred_mask[ordinary]
            selected = ordinary[mask if prefer_orientation else ~mask]
            if len(selected):
                self._plan_grasp_group(fsm, state, selected, prefer_orientation=prefer_orientation)
        remaining = rows[~eligible[rows]]
        fsm.fail(remaining, 13)

    def _plan_grasp_group(
        self, fsm, state, rows, *, measured=False, prefer_orientation=True, preferred_only=False
    ):
        if prefer_orientation and not measured and not preferred_only and len(rows):
            fresh = rows[fsm.grasp_retries[rows] == 0]
            repeated = rows[fsm.grasp_retries[rows] > 0]
            if len(fresh):
                rows = torch.cat(
                    (
                        repeated,
                        self._plan_grasp_group(
                            fsm, state, fresh, prefer_orientation=True, preferred_only=True
                        ),
                    )
                )
        eligible, sizes, hulls, hull_mask, preferred = self.grasp_metadata
        grouped = rows[eligible[rows]]
        if len(grouped):
            self._external_scene_geometry(state)
            local = {
                key: state[key][grouped]
                for key in ("root_states", "ee_pos", "ee_orn", "object_state", "dof_pos")
            }
            bank = grasp_candidate_bank(
                local,
                local["object_state"][:, :3],
                sizes[grouped],
                self.controllers[0].collision,
                self.carry_geometry.select(grouped),
                hulls[grouped],
                preferred_orientation=preferred[grouped] if prefer_orientation else None,
                support_mask=hull_mask[grouped],
                **({"candidate_orientations": preferred[grouped, None]} if preferred_only else {}),
                **(
                    {
                        "candidate_orientations": self.measured_grasp_orientations[grouped, None],
                        "pocket_depth_preference": self.measured_grasp_depth[grouped],
                    }
                    if measured
                    else {}
                ),
            )
            inverse = solve_default_grasp_inverse(
                local,
                bank,
                self.controllers[0].limits,
                self.controllers[0].collision.arm_joint_ids,
                fsm.initial_object_z[grouped],
                float(fsm.env.dt),
                excluded=fsm.tried_grasp_orientations[grouped],
                excluded_mask=fsm.tried_grasp_mask[grouped],
            )
            retry = fsm.stage_entry_from[grouped] == int(BasketStage.MOVE_TO_GRASP)
            previous_end = (fsm.arm_path.length[grouped] - 1).clamp_min(0).long()
            previous_error = (local["ee_pos"] - fsm.arm_path.positions[grouped, previous_end]).norm(
                dim=-1
            )
            path, valid, diagnostics = score_default_grasp_paths(
                local,
                bank,
                inverse,
                self.controllers[0].limits,
                self.controllers[0].collision,
                self.carry_geometry.select(grouped),
                float(fsm.env.dt),
                prefer_realized_body=fsm.grasp_retries[grouped] > 0,
                capacity=fsm.arm_path.capacity,
                elevated_transit=measured,
                prefer_orientation=prefer_orientation,
            )
            diagnostics.update(
                preferred_pose_fast_path=preferred_only,
                execution_entry_is_retry=retry,
                entry_previous_goal_error_m=previous_error,
                entry_base_position_world=local["root_states"][:, :3],
                grasp_retry_index=fsm.grasp_retries[grouped].clone(),
                measured_candidate_used=measured,
                measured_grasp_fallback=self.measured_grasp_mask[grouped]
                if not measured
                else torch.zeros_like(retry),
            )
            rejected = grouped[:0]
            if preferred_only:
                valid &= diagnostics["preferred_orientation_preservable"][:, 0]
                diagnostics["selected"] = torch.where(valid, diagnostics["selected"], -1)
            if measured or preferred_only:
                if measured:
                    diagnostics["pocket_depth_preference"] = self.measured_grasp_depth[grouped]
                rejected = grouped[~valid]
                if measured and len(rejected):
                    rejected_report = {
                        key: value[~valid] if torch.is_tensor(value) else value
                        for key, value in diagnostics.items()
                    }
                    self.carry_diagnostics.append(
                        (
                            rejected.clone(),
                            "measured_grasp_rejection",
                            valid[~valid],
                            rejected_report,
                        )
                    )
                grouped = grouped[valid]
                path = tuple(value[valid] for value in path)
                diagnostics = {
                    key: value[valid] if torch.is_tensor(value) else value
                    for key, value in diagnostics.items()
                }
                local = {key: value[valid] for key, value in local.items()}
                valid = torch.ones(len(grouped), device=fsm.device, dtype=torch.bool)
                if not len(grouped):
                    return rejected
            fsm.hold_xy[grouped] = local["root_states"][:, :2]
            w, x, y, z = local["root_states"][:, 3:7].unbind(-1)
            fsm.hold_yaw[grouped] = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            fsm.arm_path.install_tensor(grouped, *path)
            fsm.stage_steps[grouped] = 0
            fsm.settled[grouped] = 0
            fsm.grasp_retry_steps[grouped] = 0
            fsm.arm_stability.clear(grouped)
            fsm.withdraw_displacement[grouped] = diagnostics["withdraw_displacement_m"]
            accepted = grouped[valid]
            slots = fsm.tried_grasp_mask[accepted].sum(-1)
            torch._assert_async(
                (slots < fsm.tried_grasp_mask.shape[1]).all(), "Grasp history capacity exceeded"
            )
            ends = fsm.arm_path.length[accepted].long() - 1
            fsm.tried_grasp_orientations[accepted, slots] = fsm.arm_path.orientations[
                accepted, ends
            ]
            fsm.tried_grasp_mask[accepted, slots] = True
            fsm.fail(grouped[~valid], 12)
            self.carry_diagnostics.append((grouped.clone(), "grasp", valid, diagnostics))
            return rejected
        return rows[:0]

    def _cache_grasp_metadata(self, fsm):
        """Upload immutable asset metadata once; stage geometry remains device-side."""

        eligible = torch.tensor(
            [
                getattr(c, "basket", None) is None and c.object_support_points is not None
                for c in self.controllers
            ],
            device=fsm.device,
        )
        sizes = torch.tensor(
            [
                c.episode["asset"].get("intrinsic_size_m", c.episode["asset"]["size_m"])
                for c in self.controllers
            ],
            device=fsm.device,
        )
        from gr00t.rl.pick.tensor_rows import pack_scalars, pad_device_rows

        empty = sizes.new_empty(0, 3)
        hulls, mask = pad_device_rows(
            [
                c.object_support_points if c.object_support_points is not None else empty
                for c in self.controllers
            ],
            fill=torch.nan,
        )
        zero_q = sizes.new_zeros(1, 4)
        preferred = torch.cat(
            [
                c.initial_grasp_orientation if c.initial_grasp_orientation is not None else zero_q
                for c in self.controllers
            ]
        )
        one = torch.ones(1, device=fsm.device, dtype=torch.bool)
        self.grasp_preferred_mask = torch.tensor(
            [c.initial_grasp_orientation is not None for c in self.controllers], device=fsm.device
        )
        self.grasp_preferred_mask &= torch.cat(
            [getattr(c, "initial_station_valid", one) for c in self.controllers]
        )
        measured = [getattr(c, "measured_reach", None) for c in self.controllers]
        self.measured_grasp_mask = torch.tensor(
            [value is not None and value.orientation is not None for value in measured],
            device=fsm.device,
        )
        self.measured_grasp_mask &= torch.cat([getattr(value, "valid", one) for value in measured])
        self.measured_grasp_orientations = torch.cat(
            [
                value.orientation if value is not None and value.orientation is not None else zero_q
                for value in measured
            ]
        )
        self.measured_grasp_depth = pack_scalars(
            [
                value.depth if value is not None and value.orientation is not None else 0.0
                for value in measured
            ],
            sizes,
        )
        self.grasp_metadata = eligible, sizes, hulls, mask, preferred

    def plan_lift(self, fsm, state, rows):
        self._plan_carry(fsm, state, rows, withdraw=False)

    def plan_basket_lift(self, fsm, state, rows):
        if not len(rows):
            return
        if self.basket_geometry is None:
            self.basket_geometry = BasketGeometryBatch(self.controllers)
        geometry = self.basket_geometry.update(state).select(rows)
        local = {
            key: state[key][rows]
            for key in ("ee_pos", "ee_orn", "root_states", "dof_pos", "object_state")
        }
        first = self.controllers[0]
        minimum = basket_required_height(
            first.basket,
            local,
            fsm.object_support[rows],
            fsm.basket_lift_margin[rows],
            support_mask=fsm.object_support_mask[rows],
        )
        options = (
            {"tilt_angles_deg": first.carry_tilt_angles_deg}
            if hasattr(first, "carry_tilt_angles_deg")
            else {}
        )
        path, valid, diagnostics = plan_basket_carry_batch(
            local,
            fsm.initial_object_z[rows],
            -local["dof_pos"][:, fsm.env.gripper_joint_ids[0]],
            first.limits,
            first.collision,
            geometry,
            float(fsm.env.dt),
            basket=first.basket,
            support_points=fsm.object_support[rows],
            support_mask=fsm.object_support_mask[rows],
            minimum_object_height=minimum,
            capacity=fsm.arm_path.capacity,
            **options,
        )
        fsm.arm_path.install_tensor(rows, *path)
        inverse = local["ee_orn"].clone()
        inverse[:, 1:] *= -1
        fsm.grasp_object_relative[rows] = _quat_rotate(
            inverse, local["object_state"][:, :3] - local["ee_pos"]
        )
        fsm.fail(rows[~valid], 11)
        self.carry_diagnostics.append((rows.clone(), "lift", valid, diagnostics))

    def plan_basket_clear(self, fsm, state, rows):
        if not len(rows):
            return
        if self.basket_geometry is None:
            self.basket_geometry = BasketGeometryBatch(self.controllers)
        geometry = self.basket_geometry.update(state)
        target_geometry, target_surface = self.basket_geometry.with_target(state)
        local = {
            key: state[key][rows]
            for key in ("root_states", "ee_pos", "ee_orn", "object_state", "dof_pos")
        }
        first = self.controllers[0]
        path, valid, diagnostics = plan_clear_for_nudge_batch(
            local,
            first.basket,
            first.limits,
            first.collision,
            geometry.select(rows),
            target_geometry.select(rows),
            target_surface.select(rows),
            float(fsm.env.dt),
            capacity=fsm.arm_path.capacity,
        )
        diagnostics["attempt"] = fsm.nudge_attempts[rows].clone()
        fsm.arm_path.install_tensor(rows, *path)
        fsm.gripper_stability.clear(rows)
        fsm.fail(rows[~valid], 9)
        self.carry_diagnostics.append((rows.clone(), "clear_for_reposition", valid, diagnostics))

    def plan_withdraw(self, fsm, state, rows):
        self._plan_carry(fsm, state, rows, withdraw=True)

    def _plan_carry(self, fsm, state, rows, *, withdraw):
        self._external_scene_geometry(state)
        local = {
            key: state[key][rows]
            for key in ("ee_pos", "ee_orn", "root_states", "dof_pos", "object_state")
        }
        first = self.controllers[0]
        displacement = (
            fsm.withdraw_displacement[rows] if withdraw else torch.zeros_like(local["ee_pos"])
        )
        path, valid, diagnostics = plan_carry_batch(
            local,
            displacement,
            fsm.initial_object_z[rows],
            -local["dof_pos"][:, fsm.env.gripper_joint_ids[0]],
            first.limits,
            first.collision,
            self.carry_geometry.select(rows),
            float(fsm.env.dt),
            capacity=fsm.arm_path.capacity,
        )
        fsm.arm_path.install_tensor(rows, *path)
        fsm.fail(rows[~valid], 11)
        if not withdraw:
            inverse = local["ee_orn"].clone()
            inverse[:, 1:] *= -1
            fsm.grasp_object_relative[rows] = _quat_rotate(
                inverse, local["object_state"][:, :3] - local["ee_pos"]
            )
        self.carry_diagnostics.append(
            (rows.clone(), "withdraw" if withdraw else "lift", valid, diagnostics)
        )

    def finish_diagnostics(self):
        """Only serialization crosses to the host, after all control ticks finish."""
        from gr00t.rl.auto_open.tensor_metadata import to_host

        host_records = to_host(self.carry_diagnostics)
        for (_, name, _, diagnostics), (rows, _, accepted, host) in zip(
            self.carry_diagnostics, host_records, strict=True
        ):
            for offset, row in enumerate(rows):
                report = {
                    key: value[offset] if torch.is_tensor(diagnostics[key]) else value
                    for key, value in host.items()
                }
                if not accepted[offset]:
                    report["failure"] = (
                        _BASKET_PLANNING_FAILURES[name][report["planning_failure_code"]]
                        if name in _BASKET_PLANNING_FAILURES and "planning_failure_code" in report
                        else "post_nudge_no_safe_grasp_candidate"
                        if name == "reopen_after_reposition"
                        else "no_safe_measured_grasp_candidate"
                        if name == "measured_grasp_rejection"
                        else "no_safe_grasp_candidate"
                        if name == "grasp"
                        else "approach_gaze_no_collision_free_aimed_trajectory"
                        if name == "approach_gaze"
                        else "no_scene_safe_measured_match"
                        if name == "measured_station"
                        else {
                            1: "no_environment_safe_pocket",
                            2: "no_safe_exact_grasp_lift_withdraw_posture",
                            3: "no_safe_nominal_station_candidate",
                        }[report["planning_failure_code"]]
                        if name == "initial_station"
                        else "seated_carry_path_geometry_or_inverse_failed"
                    )
                if name == "approach_gaze":
                    report["random_offsets_rad"] = {
                        key.split(".", 1)[1]: report.pop(key)
                        for key in list(report)
                        if key.startswith("random_offsets_rad.")
                    }
                    if not report.get("wrist_route_enabled", False):
                        report["wrist_route_length_rad"] = None
                if name == "measured_station" and accepted[offset]:
                    self.controllers[row].planning_diagnostics["initial_station"] = report
                if name == "initial_station":
                    report["selection_policy"] = STATION_POLICIES[report["selection_policy_code"]]
                    if report.get("measured_reach_fallback_code") == 1:
                        report["measured_reach_fallback"] = "catalog_missing_or_policy_mismatch"
                    elif report.get("measured_reach_fallback_code") == 2:
                        report["measured_reach_fallback"] = "no_scene_safe_measured_match"
                if name in ("grasp", "measured_grasp_rejection"):
                    report["selected"] = None if report["selected"] < 0 else report["selected"]
                    for key in (
                        "candidate_orientations_world_wxyz",
                        "candidate_grasp_origins_world_m",
                    ):
                        if key in report:
                            report[key] = [report[key]]
                    model = {
                        key.split(".", 1)[1]: [report.pop(key)]
                        for key in list(report)
                        if key.startswith("convex_contact_model.")
                    }
                    if model:
                        report["convex_contact_model"] = model
                    measured = getattr(self.controllers[row], "measured_reach", None)
                    if measured is not None and measured.orientation is not None:
                        report["command_catalog_sha256"] = measured.catalog_sha256
                    if name == "measured_grasp_rejection":
                        self.controllers[row].planning_diagnostics.setdefault(
                            "measured_grasp_rejections", []
                        ).append(report)
                        continue
                    if report.get("measured_grasp_fallback"):
                        report["measured_grasp_fallback_reason"] = (
                            "no_safe_measured_grasp_candidate"
                        )
                    attempts = self.controllers[row].planning_diagnostics.setdefault(
                        "grasp_attempts", []
                    )
                    if report.get("execution_entry_is_retry") and attempts:
                        attempts[-1]["execution_rejection"] = {
                            "reason": "steady_ee_pose_outside_tolerance",
                            "position_error_m": [report["entry_previous_goal_error_m"]],
                            "base_position_world": [report["entry_base_position_world"]],
                        }
                    attempts.append(report)
                if name in (
                    "reposition_start",
                    "clear_for_reposition",
                    "reposition_object",
                    "reopen_after_reposition",
                ):
                    attempts = self.controllers[row].planning_diagnostics.setdefault(
                        "reposition_attempts", []
                    )
                    while len(attempts) < report["attempt"]:
                        attempts.append({"attempt": len(attempts) + 1})
                    attempt = attempts[report["attempt"] - 1]
                    if name == "reposition_start":
                        report["trigger"] = {
                            int(BasketStage.MOVE_TO_GRASP): "no_safe_grasp_candidate",
                            int(BasketStage.GRASP_OBJECT): "stage_acceptance_timeout",
                            int(
                                BasketStage.OPEN_FOR_REPOSITION
                            ): "initial_layout_requires_reachable_grasp",
                            int(
                                BasketStage.REOPEN_AFTER_REPOSITION
                            ): "post_nudge_no_safe_grasp_candidate",
                        }.get(report["trigger_stage_id"], "stage_reposition_request")
                        attempt.update(report)
                    else:
                        attempt[name.upper()] = report
                        if name == "reopen_after_reposition":
                            attempt["post_nudge_grasp"] = report
                            if report["nudge_performed"]:
                                attempt["measured_shift_m"] = report["measured_shift_m"]
                self.controllers[row].planning_diagnostics[name] = report
