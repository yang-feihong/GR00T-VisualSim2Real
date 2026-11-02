"""Basket reposition stages use shared tensor events, with no scalar planner fallback."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick import tensor_planning
from gr00t.rl.pick.stages import BasketStage
from gr00t.rl.pick.tensor_fsm import TensorPickFSM
from gr00t.rl.pick.tensor_planning import TensorPickPlanningCoordinator
from gr00t.rl.tests.test_pick_tensor_fsm import batch


def setup(monkeypatch, count=3):
    env, controllers, state = batch(monkeypatch, count=count)
    basket = SimpleNamespace(boxes=torch.tensor([[[0.0, 0, 0.2]] * 8]))
    for controller in controllers:
        controller.basket = basket
        controller.lift_margin = 0.1
        controller.object_support_points = torch.zeros(8, 3)
    state["distractor_states"] = torch.zeros(count, 0, 13)
    fsm = TensorPickFSM(env, controllers)
    planner = TensorPickPlanningCoordinator(controllers)
    geometry = SimpleNamespace(select=lambda rows: rows)
    planner.basket_geometry = SimpleNamespace(
        update=lambda s: geometry, with_target=lambda s: (geometry, geometry)
    )
    planner._cache_grasp_metadata(fsm)
    monkeypatch.setattr(planner, "plan_stage", forbidden, raising=False)
    return fsm, planner, controllers, state


def forbidden(*args, **kwargs):
    raise AssertionError("Reposition stages must not dispatch scalar planners or read host tensors")


def test_coordinator_has_no_serial_stage_dispatch_or_registered_fallback():
    assert not hasattr(TensorPickPlanningCoordinator, "plan_stage")
    assert "plan_stage" not in TensorPickPlanningCoordinator.planning_events.values()


def test_open_and_close_entries_hold_actual_pose_and_reset_only_selected_rows(monkeypatch):
    fsm, planner, _, state = setup(monkeypatch)
    rows = torch.tensor([1, 0])
    fsm.stage[rows] = int(BasketStage.GRASP_OBJECT)
    fsm.transition(rows, BasketStage.OPEN_FOR_REPOSITION)
    fsm.gripper_stability.count.fill_(12)
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        planner.plan_events(fsm, state, {"open_for_reposition_plan": rows})
    assert fsm.nudge_attempts.tolist() == [1, 1, 0]
    assert fsm.gripper_stability.count.tolist() == [0, 0, 12]
    assert fsm.nudge_current_path_unavailable.tolist() == [True, True, False]
    torch.testing.assert_close(fsm.arm_path.positions[rows, 0], state["ee_pos"][rows])
    state["ee_pos"][rows] += 0.1
    fsm.stage[rows] = int(BasketStage.CLOSE_FOR_REPOSITION)
    planner.plan_events(fsm, state, {"close_for_reposition_plan": rows})
    torch.testing.assert_close(fsm.arm_path.positions[rows, 0], state["ee_pos"][rows])
    assert fsm.nudge_attempts.tolist() == [1, 1, 0]
    planner.finish_diagnostics()
    assert (
        planner.controllers[0].planning_diagnostics["reposition_attempts"][0]["trigger"]
        == "stage_acceptance_timeout"
    )


@pytest.mark.parametrize("not_required", [False, True])
def test_closed_nudge_event_isolates_failure_or_skips_only_unnecessary_row(
    monkeypatch, not_required
):
    fsm, planner, controllers, state = setup(monkeypatch)
    rows = torch.tensor([1, 0])
    fsm.stage[rows] = int(BasketStage.REPOSITION_OBJECT)
    fsm.nudge_attempts[rows] = 1
    calls, reopened = [], []

    def plan(
        local,
        sizes,
        intrinsic,
        mask,
        basket,
        limits,
        collision,
        scene,
        obstacles,
        surface,
        dt,
        **kw,
    ):
        calls.append(scene.clone())
        assert len(local["root_states"]) == 2 and kw["capacity"] == fsm.arm_path.capacity
        p = torch.stack((local["ee_pos"], local["ee_pos"] + 0.02), 1)
        q = local["ee_orn"][:, None].expand(-1, 2, -1)
        return (
            (p, q, torch.tensor([2, 2])),
            torch.tensor([True, False]),
            {
                "planning_failure_code": torch.tensor([0, 4]),
                "direction_world": torch.tensor([[1.0, 0, 0], [0, 0, 0]]),
                "space_before_m": torch.tensor([0.01, 0.02]),
                "not_required": torch.tensor([False, not_required]),
            },
        )

    monkeypatch.setattr(tensor_planning, "plan_closed_nudge_batch", plan)
    monkeypatch.setattr(
        planner, "plan_basket_reopen", lambda f, s, selected: reopened.append(selected.clone())
    )
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        planner.plan_events(fsm, state, {"reposition_object_plan": rows})
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], rows)
    assert fsm.nudge_performed.tolist() == [False, True, False]
    torch.testing.assert_close(fsm.nudge_start[1], state["object_state"][1, :3])
    if not_required:
        assert fsm.failure_code.tolist() == [0, 0, 0]
        assert reopened[0].tolist() == [0]
    else:
        assert fsm.failure_code.tolist() == [9, 0, 0] and not reopened
    planner.finish_diagnostics()
    report = controllers[0].planning_diagnostics["reposition_attempts"][0]["REPOSITION_OBJECT"]
    assert report["planned_path_valid"] is False
    assert ("failure" not in report) == not_required


def test_reopen_requires_actual_body_grasp_and_bounds_retries_on_gpu(monkeypatch):
    fsm, planner, controllers, state = setup(monkeypatch, count=2)
    fsm.stage.fill_(int(BasketStage.REOPEN_AFTER_REPOSITION))
    fsm.nudge_attempts.fill_(1)
    fsm.nudge_start[:] = state["object_state"][:, :3]
    fsm.nudge_direction[:, 0] = 1
    fsm.nudge_performed.fill_(True)
    state["object_state"][:, 0] += 0.012
    state["ee_pos"][1, 0] = 2.4
    calls = []

    def verify(local, *args, **kw):
        assert kw["realized_body"].all() and not kw["prefer_realized_body"].any()
        assert kw.get("entry_orientation") is None and kw.get("excluded") is None
        calls.append(len(local["ee_pos"]))
        valid = local["ee_pos"][:, 0] < 1
        return None, valid, {"selected": torch.where(valid, 0, -1)}

    monkeypatch.setattr(tensor_planning, "plan_basket_grasp_batch", verify)
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        planner.plan_events(fsm, state, {"reopen_after_reposition_plan": fsm.rows})
    assert calls == [2]
    assert fsm.stage.tolist() == [
        int(BasketStage.REOPEN_AFTER_REPOSITION),
        int(BasketStage.OPEN_FOR_REPOSITION),
    ]
    assert fsm.nudge_attempts.tolist() == [1, 2] and not fsm.failure_code.any()
    fsm.stage[1] = int(BasketStage.REOPEN_AFTER_REPOSITION)
    planner.plan_events(fsm, state, {"reopen_after_reposition_plan": fsm.rows[1:]})
    assert calls == [2, 1] and fsm.failure_code.tolist() == [0, 9]
    assert fsm.nudge_attempts.tolist() == [1, 2]
    planner.finish_diagnostics()
    success_attempt = controllers[0].planning_diagnostics["reposition_attempts"][0]
    assert success_attempt["measured_shift_m"] == float(
        state["object_state"][0, 0] - fsm.nudge_start[0, 0]
    )
    rejected = controllers[1].planning_diagnostics["reposition_attempts"]
    assert len(rejected) == 2
    assert rejected[0]["post_nudge_grasp"]["failure"] == "post_nudge_no_safe_grasp_candidate"
    assert rejected[1]["trigger"] == "post_nudge_no_safe_grasp_candidate"
