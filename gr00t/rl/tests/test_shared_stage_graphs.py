"""Tasks change stage definitions, not the execution or planning event protocol."""

import pytest
import torch

from gr00t.rl.auto_open.stages import (
    StagePlanningEvents,
    StageSpec,
    TensorStageLifecycle,
    validate_stage_graph,
)
from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM
from gr00t.rl.auto_open.tensor_planning import TensorPlanningCoordinator
from gr00t.rl.pick.stages import stage_definitions
from gr00t.rl.pick.tensor_fsm import TensorPickFSM
from gr00t.rl.pick.tensor_planning import TensorPickPlanningCoordinator


@pytest.mark.parametrize("basket", [False, True])
def test_pick_graphs_have_the_existing_stage_specs_and_closed_control_contracts(basket):
    stage, specs, interfaces = stage_definitions(basket=basket)
    assert set(specs) == set(stage) == set(interfaces)
    assert all(isinstance(spec, StageSpec) for spec in specs.values())
    assert specs[stage.LIFT_OBJECT].successors == (
        (stage.VERIFY_RETENTION,) if basket else (stage.WITHDRAW_WITH_OBJECT,)
    )
    missing = dict(interfaces)
    del missing[stage.GRASP_OBJECT]
    with pytest.raises(ValueError, match="Every stage"):
        validate_stage_graph(stage, specs, missing)


def test_tasks_share_lifecycle_and_event_dispatch_implementations():
    assert TensorDoorFSM.fail is TensorPickFSM.fail is TensorStageLifecycle.fail
    assert TensorDoorFSM.transition is TensorStageLifecycle.transition
    assert TensorPlanningCoordinator.plan_events is StagePlanningEvents.plan_events
    assert TensorPickPlanningCoordinator.plan_events is StagePlanningEvents.plan_events
    assert not hasattr(TensorPickPlanningCoordinator, "resolve")


def test_planning_events_fail_on_unknown_labels_and_ignore_empty_stage_entries():
    planner = StagePlanningEvents()
    planner.planning_events = {"grasp_plan": "plan_grasp"}
    calls = []
    planner.plan_grasp = lambda *args: calls.append(args)
    planner.plan_events(None, {}, {"grasp_plan": torch.empty(0, dtype=torch.long)})
    assert not calls
    planner.plan_events(None, {}, {"grasp_plan": torch.tensor([1, 3])})
    assert calls[0][-1].tolist() == [1, 3]
    with pytest.raises(ValueError, match="Unregistered"):
        planner.plan_events(None, {}, {"new_private_request_protocol": torch.tensor([0])})
