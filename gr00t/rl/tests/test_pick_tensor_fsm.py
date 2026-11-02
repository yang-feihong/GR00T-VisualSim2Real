"""Pick stages use the door tensor lifecycle and sparse planning events."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.execution import TensorCommandBuffers
from gr00t.rl.pick.controller import PickStage
from gr00t.rl.pick.tensor_fsm import TensorPickFSM
from gr00t.rl.pick.tensor_planning import TensorPickPlanningCoordinator
from gr00t.rl.tests.test_pick_workflow import fake_controller


def batch(monkeypatch, count=2, **options):
    pairs = [fake_controller(monkeypatch, **options) for _ in range(count)]
    controllers = [pair[0] for pair in pairs]
    env = SimpleNamespace(**vars(controllers[0].env))
    env.num_envs = count
    for name, value in vars(env).copy().items():
        if (
            name not in ("arm_joint_ids", "gripper_joint_ids")
            and torch.is_tensor(value)
            and value.ndim
            and value.shape[0] == 1
        ):
            setattr(env, name, value.expand(count, *value.shape[1:]).clone())
    env.get_goal_reference_origin = lambda: torch.zeros(count, 3)
    env.get_goal_reference_quat = lambda: torch.tensor([[1.0, 0, 0, 0]]).expand(count, -1)
    env.get_goal_center_offset_local = lambda: torch.tensor([[0.22, 0, 0.7]]).expand(count, -1)
    state = {key: torch.cat([pair[1][key] for pair in pairs]) for key in pairs[0][1]}
    state["contact_forces_w"] = torch.zeros(count, 0, 3)
    state["object_finger_contact_points"] = torch.zeros(count, 2, 3)
    return env, controllers, state


def test_batch_approach_publishes_same_references_as_single_task_configuration(monkeypatch):
    env, controllers, state = batch(monkeypatch)
    fsm = TensorPickFSM(env, controllers)
    buffers = TensorCommandBuffers.from_env(env)
    fsm.command(state, buffers)
    for row, controller in enumerate(controllers):
        local = {key: value[row : row + 1] for key, value in state.items()}
        single_fsm = TensorPickFSM(controller.env, [controller])
        single = TensorCommandBuffers.from_env(controller.env)
        single_fsm.command(local, single)
        for name in ("base", "ee_cart", "ee_quat", "gripper", "arm_joint_mode"):
            torch.testing.assert_close(getattr(buffers, name)[row : row + 1], getattr(single, name))


def test_one_rows_collision_does_not_fail_other_rows(monkeypatch):
    env, controllers, state = batch(monkeypatch)
    for controller in controllers:
        controller.nonfoot_contact_ids = torch.tensor([0])
    fsm = TensorPickFSM(env, controllers)
    state["contact_forces_w"] = torch.tensor([[[21.0, 0, 0]], [[0.0, 0, 0]]])
    for _ in range(3):
        fsm.advance(state)
    assert fsm.stage.tolist() == [int(PickStage.FAILURE), int(PickStage.APPROACH_OBJECT)]
    assert fsm.failure_code.tolist() == [1, 0]


def test_low_command_correction_does_not_redefine_path_or_arrival_acceptance(monkeypatch):
    env, controllers, state = batch(monkeypatch)
    fsm = TensorPickFSM(env, controllers)
    buffers = TensorCommandBuffers.from_env(env)
    physical_goal = torch.tensor([[0.65, 0, 0.105], [0.65, 0, 0.105]])
    orientation = torch.tensor([[1.0, 0, 0, 0]]).expand(2, -1)
    fsm.arm_path.install_tensor(
        fsm.rows, physical_goal[:, None], orientation[:, None], torch.ones(2, dtype=torch.long)
    )
    fsm.stage.fill_(int(PickStage.MOVE_TO_GRASP))
    fsm.stage_steps.fill_(10)
    state["ee_pos"] = physical_goal + torch.tensor([0, 0, 0.023])
    state["ee_orn"] = orientation.clone()
    for _ in range(30):
        fsm.command(state, buffers)
    torch.testing.assert_close(fsm.arm_path.positions[:, 0], physical_goal)
    published = buffers.ee_cart + env.get_goal_center_offset_local()
    assert (published[:, 2] < physical_goal[:, 2] - 0.019).all()
    assert (published[:, 2] >= physical_goal[:, 2] - 0.020001).all()
    for _ in range(30):
        fsm.advance(state)
    assert (fsm.stage == int(PickStage.MOVE_TO_GRASP)).all()


def test_grasp_acceptance_waits_for_position_stability_without_dof_velocity_gate(monkeypatch):
    env, controllers, state = batch(monkeypatch)
    for controller in controllers:
        controller.current_stage = PickStage.GRASP_OBJECT
    fsm = TensorPickFSM(env, controllers)
    state["object_finger_forces"].fill_(2)
    state["object_finger_contact_points"] = (
        state["object_state"][:, None, :3].expand(-1, 2, -1).clone()
    )
    state["object_finger_contact_points"][:, 0, 2] -= 0.02
    state["object_finger_contact_points"][:, 1, 2] += 0.02
    for _ in range(14):
        events = fsm.advance(state)
        assert not len(events.get("lift_object_plan", []))
        assert fsm.stage.tolist() == [int(PickStage.GRASP_OBJECT)] * 2
    for _ in range(10):
        fsm.advance(state)
    assert fsm.stage.tolist() == [int(PickStage.LIFT_OBJECT)] * 2


def test_nonplanning_stage_entries_reset_gpu_windows_without_row_dispatch(monkeypatch):
    env, controllers, state = batch(monkeypatch)
    fsm = TensorPickFSM(env, controllers)
    planner = TensorPickPlanningCoordinator(controllers)
    monkeypatch.setattr(
        planner,
        "plan_stage",
        lambda *a: (_ for _ in ()).throw(AssertionError("serial dispatch")),
        raising=False,
    )
    fsm.stage.fill_(int(PickStage.MOVE_TO_GRASP))
    fsm.settled.fill_(11)
    fsm.gripper_stability.count.fill_(12)
    fsm.transition(fsm.rows, PickStage.GRASP_OBJECT)
    planner.plan_events(fsm, state, {"grasp_object_plan": fsm.rows})
    assert (fsm.settled == 0).all()
    assert (fsm.gripper_stability.count == 0).all()
    assert (fsm.stage_entry_from == int(PickStage.MOVE_TO_GRASP)).all()
    fsm.stage.fill_(int(PickStage.WITHDRAW_WITH_OBJECT))
    fsm.object_stability.count.fill_(12)
    fsm.transition(fsm.rows, PickStage.VERIFY_RETENTION)
    planner.plan_events(fsm, state, {"verify_retention_plan": fsm.rows})
    assert (fsm.object_stability.count == 0).all()
    assert (fsm.stage_entry_from == int(PickStage.WITHDRAW_WITH_OBJECT)).all()


def test_missing_grasp_support_fails_only_requested_rows_without_serial_fallback(monkeypatch):
    env, controllers, state = batch(monkeypatch, count=3)
    assert all(controller.object_support_points is None for controller in controllers)
    fsm = TensorPickFSM(env, controllers)
    planner = TensorPickPlanningCoordinator(controllers)
    rows = torch.tensor([2, 0])
    fsm.transition(rows, PickStage.MOVE_TO_GRASP)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Missing grasp metadata must not invoke another planner or read host tensors"
        )

    planner._cache_grasp_metadata(fsm)
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        planner.plan_events(fsm, state, {"move_to_grasp_plan": rows})
    assert fsm.failure_code.tolist() == [13, 0, 13]
    assert fsm.stage[1] == int(PickStage.APPROACH_OBJECT)
    assert not hasattr(planner, "plan_stage")


@pytest.mark.parametrize("destination", [PickStage.GRASP_OBJECT, PickStage.VERIFY_RETENTION])
def test_hold_plan_install_keeps_executed_endpoint_and_other_rows_untouched(
    monkeypatch, destination
):
    env, controllers, state = batch(monkeypatch)
    fsm = TensorPickFSM(env, controllers)
    positions = torch.tensor([[[0.1, 0, 0.4], [0.2, 0, 0.5], [0.3, 0, 0.6]]])
    orientations = torch.tensor([[[1.0, 0, 0, 0]]]).expand(1, 3, 4)
    fsm.arm_path.install_tensor(fsm.rows[:1], positions, orientations, torch.tensor([3]))
    fsm.arm_path.index[0] = 2
    other_position = fsm.arm_path.positions[1].clone()
    other_cursor = fsm.arm_path.index[1].clone()
    controllers[0].install_arm_path(positions, orientations)
    controllers[0].current_stage = destination
    fsm.stage[0] = int(destination)
    fsm.install_plans(fsm.rows[:1], controllers[:1])
    p, q = fsm.arm_path.current(fsm.rows[:1], advance=False)
    torch.testing.assert_close(p, positions[:, -1])
    torch.testing.assert_close(q, orientations[:, -1])
    assert fsm.arm_path.index[0] == 2
    torch.testing.assert_close(fsm.arm_path.positions[1], other_position)
    torch.testing.assert_close(fsm.arm_path.index[1], other_cursor)


def test_external_carry_events_submit_one_batch_and_defer_diagnostics(monkeypatch):
    from gr00t.rl.pick import tensor_planning

    env, controllers, state = batch(monkeypatch)
    fsm = TensorPickFSM(env, controllers)
    planner = TensorPickPlanningCoordinator(controllers)
    planner.carry_geometry = SimpleNamespace(select=lambda rows: None)
    calls = []

    def carry(local, displacement, *args, **kwargs):
        calls.append((local["ee_pos"].shape[0], displacement.clone()))
        p = local["ee_pos"][:, None].expand(-1, 2, -1)
        q = local["ee_orn"][:, None].expand(-1, 2, -1)
        return (
            (p, q, torch.tensor([2, 2])),
            torch.tensor([True, False]),
            {"value": torch.tensor([1.0, 2.0])},
        )

    monkeypatch.setattr(tensor_planning, "plan_carry_batch", carry)
    monkeypatch.setattr(
        planner,
        "plan_stage",
        lambda *a: (_ for _ in ()).throw(AssertionError("serial dispatch")),
        raising=False,
    )
    fsm.stage.fill_(int(PickStage.LIFT_OBJECT))
    planner.plan_events(fsm, state, {"lift_object_plan": fsm.rows})
    assert len(calls) == 1 and calls[0][0] == 2
    assert (calls[0][1] == 0).all()
    assert fsm.failure_code.tolist() == [0, 11]
    assert "lift" not in controllers[0].planning_diagnostics
    planner.finish_diagnostics()
    assert controllers[0].planning_diagnostics["lift"] == {"value": 1.0}
    assert (
        controllers[1].planning_diagnostics["lift"]["failure"]
        == "seated_carry_path_geometry_or_inverse_failed"
    )


@pytest.mark.parametrize("failure_code", [1, 2, 3, 4])
def test_basket_lift_entry_batches_rows_and_defers_precise_failures(monkeypatch, failure_code):
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.pick import tensor_planning

    env, controllers, state = batch(monkeypatch, count=3)
    basket = SimpleNamespace(boxes=torch.tensor([[[0.0, 0, 0.2]] * 8]))
    for row, controller in enumerate(controllers):
        controller.basket = basket
        controller.lift_margin = 0.1 + row * 0.01
        controller.object_support_points = torch.tensor([[0.0, 0, -0.02], [0, 0, 0.02]])[
            : row % 2 + 1
        ]
    state["root_states"][:, 0] = torch.tensor([0.0, 10, 20])
    state["object_state"][:, 0] = torch.tensor([0.65, 10.65, 20.65])
    state["ee_orn"][1] = torch.tensor([0.0, 0, 0, 1])
    fsm = TensorPickFSM(env, controllers)
    planner = TensorPickPlanningCoordinator(controllers)
    updates, calls = [], []

    class Geometry:
        def __init__(self, configured):
            assert configured is controllers

        def update(self, full_state):
            assert full_state is state
            updates.append(full_state["root_states"].clone())
            return self

        def select(self, rows):
            return rows

    def carry(local, initial, aperture, limits, collision, geometry, dt, **kw):
        calls.append((local, geometry.clone(), kw))
        assert local["ee_pos"].shape[0] == 2 and dt == 0.02
        assert kw["basket"] is basket
        assert kw["capacity"] == fsm.arm_path.capacity
        p = local["ee_pos"][:, None].expand(-1, 2, -1).clone()
        p[:, 1, 2] += 0.2
        q = local["ee_orn"][:, None].expand(-1, 2, -1)
        return (
            (p, q, torch.tensor([2, 2])),
            torch.tensor([True, False]),
            {"planning_failure_code": torch.tensor([0, failure_code])},
        )

    monkeypatch.setattr(tensor_planning, "BasketGeometryBatch", Geometry)
    monkeypatch.setattr(tensor_planning, "plan_basket_carry_batch", carry)

    def forbidden(*args, **kwargs):
        raise AssertionError("Basket lift must not perform scalar planning or host reads")

    monkeypatch.setattr(planner, "plan_stage", forbidden, raising=False)
    rows = torch.tensor([1, 0])
    other_path = fsm.arm_path.positions[2].clone()
    fsm.stage[rows] = int(PickStage.LIFT_OBJECT)
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        planner.plan_events(fsm, state, {"lift_object_plan": rows})
    assert len(calls) == 1 and len(updates) == 1
    torch.testing.assert_close(calls[0][1], rows)
    torch.testing.assert_close(calls[0][2]["support_mask"], fsm.object_support_mask[rows])
    expected_height = state["root_states"][rows, 2] + 0.2 + fsm.basket_lift_margin[rows] + 0.02
    torch.testing.assert_close(calls[0][2]["minimum_object_height"], expected_height)
    assert fsm.failure_code.tolist() == [11, 0, 0]
    torch.testing.assert_close(fsm.arm_path.positions[2], other_path)
    inverse = state["ee_orn"][rows].clone()
    inverse[:, 1:] *= -1
    torch.testing.assert_close(
        fsm.grasp_object_relative[rows],
        _quat_rotate(inverse, state["object_state"][rows, :3] - state["ee_pos"][rows]),
    )
    assert all("lift" not in controller.planning_diagnostics for controller in controllers)
    planner.finish_diagnostics()
    failures = {
        1: "no_collision_free_held_object_lift",
        2: "no_robust_actual_body_lift_path",
        3: "held_lift_path_exceeds_capacity",
        4: "held_lift_dense_geometry_failed",
    }
    assert controllers[0].planning_diagnostics["lift"]["failure"] == failures[failure_code]
    assert "failure" not in controllers[1].planning_diagnostics["lift"]
    assert "lift" not in controllers[2].planning_diagnostics
    planner.plan_basket_lift(fsm, state, rows[:0])
    assert len(calls) == 1 and len(updates) == 1


def test_basket_refreshes_and_executes_insertion_before_publishing_close(monkeypatch):
    from gr00t.rl.pick import tensor_planning
    from gr00t.rl.pick.basket_controller import BasketPickController

    env, originals, state = batch(monkeypatch, count=1)
    control = BasketPickController.__new__(BasketPickController)
    control.__dict__.update(vars(originals[0]))
    control.basket = SimpleNamespace(boxes=torch.tensor([[[0.0, 0, 0.2]] * 8]))
    control.lift_margin = 0.1
    control.object_support_points = torch.zeros(8, 3)
    control.current_stage = PickStage.MOVE_TO_GRASP
    control.final_entry, control.awaiting_entry_refresh = False, True
    control.entry_object_orientation = state["object_state"][:, 3:7].clone()
    control.entry_orientation = state["ee_orn"].clone()
    control.entry_refresh_count = 0
    control.refresh_basket_geometry = lambda state: None
    calls = []

    def insertion(local, *args, **kwargs):
        calls.append(local["object_state"].clone())
        torch.testing.assert_close(kwargs["entry_orientation"], local["ee_orn"])
        p = torch.stack((local["ee_pos"], local["object_state"][:, :3]), 1)
        path = (p, local["ee_orn"][:, None].expand(-1, 2, -1), torch.tensor([2]))
        return path, torch.tensor([True]), {"entry_orientation_world_wxyz": local["ee_orn"]}

    monkeypatch.setattr(tensor_planning, "plan_basket_grasp_batch", insertion)
    fsm = TensorPickFSM(env, [control])
    planner = TensorPickPlanningCoordinator([control])
    geometry = SimpleNamespace(select=lambda rows: None)
    planner.basket_geometry = SimpleNamespace(update=lambda state: geometry)
    fsm.transition(fsm.rows, PickStage.GRASP_OBJECT)
    planner.plan_events(fsm, state, {"grasp_object_plan": fsm.rows})
    assert calls and fsm.basket_final_entry.all() and not fsm.awaiting_entry_refresh.any()
    assert fsm.basket_entry_refresh_count.tolist() == [1]
    assert fsm.stage.tolist() == [int(PickStage.MOVE_TO_GRASP)]
    buffers = TensorCommandBuffers.from_env(env)
    fsm.command(state, buffers)
    torch.testing.assert_close(buffers.gripper, torch.full_like(buffers.gripper, -torch.pi / 3))
    fsm.arm_path.index[:] = fsm.arm_path.length - 1
    insertion_end = fsm.arm_path.positions[:, 1].clone()
    monkeypatch.setattr(
        planner,
        "plan_stage",
        lambda *a: (_ for _ in ()).throw(AssertionError("Closure must not invoke scalar planning")),
        raising=False,
    )
    fsm.transition(fsm.rows, PickStage.GRASP_OBJECT)
    planner.plan_events(fsm, state, {"grasp_object_plan": fsm.rows})
    assert len(calls) == 1
    fsm.command(state, buffers)
    assert (buffers.gripper == 0).all()
    position, _ = fsm.arm_path.current(fsm.rows, advance=False)
    torch.testing.assert_close(position, insertion_end)
    assert (fsm.arm_path.index == fsm.arm_path.length - 1).all()


def test_mixed_basket_grasp_entries_only_dispatch_rows_needing_insertion_refresh(monkeypatch):
    env, controllers, state = batch(monkeypatch)
    for controller in controllers:
        controller.basket = object()
        controller.lift_margin = 0.1
        controller.object_support_points = torch.zeros(8, 3)
    fsm = TensorPickFSM(env, controllers)
    planner = TensorPickPlanningCoordinator(controllers)
    fsm.awaiting_entry_refresh[:] = torch.tensor([False, True])
    dispatched = []
    monkeypatch.setattr(
        planner, "plan_basket_grasp", lambda f, s, rows: dispatched.append(rows.clone())
    )
    planner.plan_events(fsm, state, {"grasp_object_plan": fsm.rows})
    assert len(dispatched) == 1
    assert dispatched[0].tolist() == [1]


def test_basket_above_object_entry_batches_paths_and_failure_without_host_reads(monkeypatch):
    from gr00t.rl.pick import tensor_planning

    env, controllers, state = batch(monkeypatch, count=3)
    basket = SimpleNamespace(boxes=torch.tensor([[[0.0, 0, 0.2]] * 8]))
    for controller in controllers:
        controller.basket = basket
        controller.lift_margin = 0.1
        controller.object_support_points = torch.zeros(8, 3)
    fsm = TensorPickFSM(env, controllers)
    planner = TensorPickPlanningCoordinator(controllers)
    geometry = SimpleNamespace(select=lambda rows: rows)
    planner.basket_geometry = SimpleNamespace(update=lambda state: geometry)
    fsm.stage.fill_(int(PickStage.MOVE_TO_GRASP))
    fsm.stage_entry_from.fill_(int(PickStage.APPROACH_OBJECT))
    fsm.nudge_attempts.fill_(2)
    rows = torch.tensor([1, 0])
    other_path = fsm.arm_path.positions[2].clone()
    calls = []

    def plan(local, sizes, limits, collision, selected, dt, **kw):
        calls.append(selected.clone())
        assert kw["entry_orientation"] is None
        assert kw["support_mask"].shape == (2, 8)
        p = torch.stack((local["ee_pos"], local["ee_pos"] + 0.01), 1)
        q = local["ee_orn"][:, None].expand(-1, 2, -1)
        return (
            (p, q, torch.tensor([2, 2])),
            torch.tensor([True, False]),
            {"entry_orientation_world_wxyz": local["ee_orn"]},
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("Basket approach must use one device-side stage plan")

    monkeypatch.setattr(tensor_planning, "plan_basket_grasp_batch", plan)
    monkeypatch.setattr(planner, "plan_stage", forbidden, raising=False)
    # Immutable asset metadata is uploaded once, before stage-time tensor checks.
    planner._cache_grasp_metadata(fsm)
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        planner.plan_events(fsm, state, {"move_to_grasp_plan": rows})
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], rows)
    assert fsm.failure_code.tolist() == [12, 0, 0]
    assert fsm.awaiting_entry_refresh.tolist() == [False, True, False]
    assert fsm.tried_grasp_mask.sum(-1).tolist() == [0, 1, 0]
    torch.testing.assert_close(fsm.arm_path.positions[2], other_path)


def test_basket_clear_event_uses_one_batch_and_records_attempt_failures(monkeypatch):
    from gr00t.rl.pick import tensor_planning
    from gr00t.rl.pick.stages import BasketStage

    env, controllers, state = batch(monkeypatch, count=3)
    for controller in controllers:
        controller.basket = object()
        controller.lift_margin = 0.1
        controller.object_support_points = torch.zeros(8, 3)
        controller.planning_diagnostics["reposition_attempts"] = [{"attempt": 1}]
    fsm = TensorPickFSM(env, controllers)
    planner = TensorPickPlanningCoordinator(controllers)
    geometry = SimpleNamespace(select=lambda rows: rows)
    planner.basket_geometry = SimpleNamespace(
        update=lambda state: geometry, with_target=lambda state: (geometry, geometry)
    )
    fsm.stage.fill_(int(BasketStage.CLEAR_FOR_REPOSITION))
    fsm.nudge_attempts.fill_(1)
    rows = torch.tensor([1, 0])
    other_path = fsm.arm_path.positions[2].clone()
    calls = []

    def clear(local, basket, limits, collision, scene, obstacles, target, dt, **kw):
        calls.append(scene.clone())
        torch.testing.assert_close(scene, rows)
        torch.testing.assert_close(obstacles, rows)
        torch.testing.assert_close(target, rows)
        assert kw["capacity"] == fsm.arm_path.capacity
        p = torch.stack((local["ee_pos"], local["ee_pos"] + 0.02), 1)
        q = local["ee_orn"][:, None].expand(-1, 2, -1)
        return (
            (p, q, torch.tensor([2, 2])),
            torch.tensor([True, False]),
            {"planning_failure_code": torch.tensor([0, 2])},
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("Basket release must not use scalar dispatch or host reads")

    monkeypatch.setattr(tensor_planning, "plan_clear_for_nudge_batch", clear)
    monkeypatch.setattr(planner, "plan_stage", forbidden, raising=False)
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        planner.plan_events(fsm, state, {"clear_for_reposition_plan": rows})
    assert len(calls) == 1
    assert fsm.failure_code.tolist() == [9, 0, 0]
    torch.testing.assert_close(fsm.arm_path.positions[2], other_path)
    planner.finish_diagnostics()
    report = controllers[0].planning_diagnostics["reposition_attempts"][0]["CLEAR_FOR_REPOSITION"]
    assert report["failure"] == "no_open_jaw_clear_inverse_path"
    assert report["attempt"] == 1
    assert (
        "CLEAR_FOR_REPOSITION" not in controllers[2].planning_diagnostics["reposition_attempts"][0]
    )


@pytest.mark.parametrize("preferred", [False, True])
def test_external_stage_prepares_one_candidate_batch_and_consumes_each_rows_bank(
    monkeypatch, preferred
):
    from gr00t.rl.pick import controller as planning_config
    from gr00t.rl.pick import tensor_planning

    env, controllers, state = batch(monkeypatch)
    from gr00t.rl.pick.measured_station import MeasuredLowReachPlanner

    measured = MeasuredLowReachPlanner.__new__(MeasuredLowReachPlanner)
    measured.orientation = None
    controllers[1].measured_reach = measured
    for row, controller in enumerate(controllers):
        controller.object_support_points = torch.zeros(8 + row, 3)
        controller.initial_grasp_orientation = (
            state["ee_orn"][row : row + 1].clone() if preferred else None
        )
    fsm = TensorPickFSM(env, controllers)
    planner = TensorPickPlanningCoordinator(controllers)
    planner.carry_geometry = SimpleNamespace(select=lambda rows: SimpleNamespace(rows=rows))
    submissions, consumed = [], []
    accepted = torch.ones(2, dtype=torch.bool)

    def candidates(local, target, size, collision, geometry, hulls, **options):
        assert (options["preferred_orientation"] is not None) == preferred
        submissions.append(
            (target.shape[0], geometry.rows.clone(), options["support_mask"].clone())
        )
        p = target[:, None].clone()
        p[:, :, 1] = torch.arange(len(target))[:, None] * 0.1
        q = local["ee_orn"][:, None].clone()
        return q, p, {"pocket_fit": torch.ones(len(target), 1, dtype=torch.bool)}, 1, 0

    def score(local, bank, *args, **kwargs):
        assert kwargs["prefer_orientation"] == preferred
        consumed.append(bank[1].clone())
        path = bank[1], bank[0], torch.ones(2, dtype=torch.long)
        return (
            path,
            accepted.clone(),
            {
                "withdraw_displacement_m": torch.zeros(2, 3),
                "selected": torch.where(accepted, 0, -1),
                "preferred_orientation_preservable": torch.ones(2, 1, dtype=torch.bool),
            },
        )

    monkeypatch.setattr(tensor_planning, "grasp_candidate_bank", candidates)
    monkeypatch.setattr(tensor_planning, "solve_default_grasp_inverse", lambda *a, **kw: {})
    monkeypatch.setattr(tensor_planning, "score_default_grasp_paths", score)
    assert not hasattr(planning_config, "plan_grasp")
    monkeypatch.setattr(
        planner, "plan_stage", lambda *a, **kw: pytest.fail("Scalar stage dispatch"), raising=False
    )
    fsm.transition(fsm.rows, PickStage.MOVE_TO_GRASP)

    def forbidden(*args, **kwargs):
        raise AssertionError("The regular grasp entry must not read tensor values on the host")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        planner.plan_events(fsm, state, {"move_to_grasp_plan": fsm.rows})
    assert len(submissions) == 1 and submissions[0][0] == 2
    assert submissions[0][1].tolist() == [0, 1]
    assert submissions[0][2].sum(-1).tolist() == [8, 9]
    assert len(consumed) == 1
    torch.testing.assert_close(consumed[0][..., 1], torch.tensor([[0.0], [0.1]]))
    torch.testing.assert_close(fsm.arm_path.positions[:, 0, 1], torch.tensor([0.0, 0.1]))
    assert fsm.tried_grasp_mask.sum(-1).tolist() == [1, 1]
    assert "grasp" not in controllers[0].planning_diagnostics
    accepted[1] = False
    fsm.transition(fsm.rows, PickStage.MOVE_TO_GRASP)
    planner.plan_events(fsm, state, {"move_to_grasp_plan": fsm.rows})
    assert fsm.failure_code.tolist() == [0, 12]
    assert fsm.grasp_retries.tolist() == [1, 1]
    assert fsm.tried_grasp_mask.sum(-1).tolist() == [2, 1]
    planner.finish_diagnostics()
    assert controllers[0].planning_diagnostics["grasp"]["selected"] == 0
    assert len(controllers[0].planning_diagnostics["grasp_attempts"]) == 2
    assert "execution_rejection" in controllers[0].planning_diagnostics["grasp_attempts"][0]
    assert controllers[1].planning_diagnostics["grasp"]["selected"] is None
    assert controllers[1].planning_diagnostics["grasp"]["failure"] == "no_safe_grasp_candidate"


def test_measured_grasp_batch_preserves_preference_and_fallback_without_double_retry(monkeypatch):
    from gr00t.rl.pick import tensor_planning
    from gr00t.rl.pick.measured_station import MeasuredLowReachPlanner

    env, controllers, state = batch(monkeypatch, count=3)
    for row, controller in enumerate(controllers):
        controller.object_support_points = torch.zeros(8, 3)
        controller.initial_grasp_orientation = state["ee_orn"][row : row + 1].clone()
        if row < 2:
            measured = MeasuredLowReachPlanner.__new__(MeasuredLowReachPlanner)
            measured.orientation = controller.initial_grasp_orientation.clone()
            measured.depth = 0.25 + row * 0.5
            measured.catalog_sha256 = "a" * 64
            controller.measured_reach = measured
    fsm = TensorPickFSM(env, controllers)
    planner = TensorPickPlanningCoordinator(controllers)
    planner.carry_geometry = SimpleNamespace(select=lambda rows: SimpleNamespace(rows=rows))
    candidate_calls, score_calls = [], []

    def candidates(local, target, sizes, collision, geometry, hulls, **kwargs):
        candidate_calls.append((geometry.rows.clone(), kwargs))
        p = target[:, None].clone()
        p[..., 1] = geometry.rows[:, None] * 0.1
        return local["ee_orn"][:, None], p, {}, 1, 0

    def score(local, bank, inverse, limits, collision, geometry, dt, **kwargs):
        elevated = kwargs["elevated_transit"]
        score_calls.append((geometry.rows.clone(), elevated))
        valid = geometry.rows != 1 if elevated else torch.ones(len(geometry.rows), dtype=torch.bool)
        assert kwargs["prefer_realized_body"].all()
        return (
            (bank[1], bank[0], torch.ones(len(valid), dtype=torch.long)),
            valid,
            {
                "withdraw_displacement_m": torch.zeros(len(valid), 3),
                "selected": torch.where(valid, 0, -1),
            },
        )

    monkeypatch.setattr(tensor_planning, "grasp_candidate_bank", candidates)
    monkeypatch.setattr(tensor_planning, "solve_default_grasp_inverse", lambda *a, **kw: {})
    monkeypatch.setattr(tensor_planning, "score_default_grasp_paths", score)
    monkeypatch.setattr(
        planner, "plan_stage", lambda *a: pytest.fail("Serial grasp dispatch"), raising=False
    )
    fsm.transition(fsm.rows, PickStage.MOVE_TO_GRASP)
    fsm.transition(fsm.rows, PickStage.MOVE_TO_GRASP)
    planner.plan_events(fsm, state, {"move_to_grasp_plan": fsm.rows})
    assert score_calls[0][0].tolist() == [0, 1] and score_calls[0][1]
    assert score_calls[1][0].tolist() == [2, 1] and not score_calls[1][1]
    torch.testing.assert_close(
        candidate_calls[0][1]["pocket_depth_preference"], torch.tensor([0.25, 0.75])
    )
    assert "candidate_orientations" in candidate_calls[0][1]
    assert "candidate_orientations" not in candidate_calls[1][1]
    assert fsm.grasp_retries.tolist() == [1, 1, 1]
    assert fsm.failure_code.tolist() == [0, 0, 0]
    assert fsm.tried_grasp_mask.sum(-1).tolist() == [1, 1, 1]
    planner.finish_diagnostics()
    assert controllers[0].planning_diagnostics["grasp"]["measured_candidate_used"]
    assert controllers[1].planning_diagnostics["grasp"]["measured_grasp_fallback"]
    assert len(controllers[1].planning_diagnostics["grasp_attempts"]) == 1
    rejected = controllers[1].planning_diagnostics["measured_grasp_rejections"]
    assert rejected[0]["failure"] == "no_safe_measured_grasp_candidate"
    assert rejected[0]["command_catalog_sha256"] == "a" * 64
