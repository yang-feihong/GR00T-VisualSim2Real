import ast
from pathlib import Path

import pytest

from gr00t.rl.scripts.doorman_auto_open import DoorController, Stage, STAGE_SPECS


REPO_ROOT = Path(__file__).resolve().parents[3]
AUTO_OPEN_SOURCE = REPO_ROOT / "gr00t" / "rl" / "scripts" / "doorman_auto_open.py"
LIFECYCLE_SOURCE = REPO_ROOT / "gr00t" / "rl" / "auto_open" / "lifecycle.py"


EXPECTED_STAGES = (
    Stage.APPROACH,
    Stage.PREGRASP,
    Stage.OPEN_GRIPPER,
    Stage.CENTER_HANDLE,
    Stage.GRASP_HANDLE,
    Stage.TURN_HANDLE,
    Stage.RETURN_HANDLE,
    Stage.OPEN_AND_CLEAR,
    Stage.RETREAT_ARM,
    Stage.RESTORE_GRIPPER,
    Stage.ALIGN_PANEL,
    Stage.CONTACT_PANEL,
    Stage.PUSH_OPEN,
    Stage.PUSH_ARM_RETRACT,
    Stage.BASE_ADVANCE,
    Stage.THROUGH,
    Stage.SUCCESS,
    Stage.FAILURE,
)


EXPECTED_BUDGETS = {
    Stage.APPROACH: 1000,
    Stage.PREGRASP: 500,
    Stage.OPEN_GRIPPER: 100,
    Stage.CENTER_HANDLE: 800,
    Stage.GRASP_HANDLE: 400,
    Stage.TURN_HANDLE: 900,
    Stage.RETURN_HANDLE: 150,
    Stage.OPEN_AND_CLEAR: 100,
    Stage.RETREAT_ARM: 200,
    Stage.RESTORE_GRIPPER: 300,
    Stage.ALIGN_PANEL: 400,
    Stage.CONTACT_PANEL: 400,
    Stage.PUSH_OPEN: 900,
    Stage.PUSH_ARM_RETRACT: 300,
    Stage.BASE_ADVANCE: 900,
    Stage.THROUGH: 1100,
    Stage.SUCCESS: None,
    Stage.FAILURE: None,
}


EXPECTED_SUCCESSORS = {
    Stage.APPROACH: (Stage.PREGRASP,),
    Stage.PREGRASP: (Stage.OPEN_GRIPPER,),
    Stage.OPEN_GRIPPER: (Stage.CENTER_HANDLE,),
    Stage.CENTER_HANDLE: (Stage.GRASP_HANDLE,),
    Stage.GRASP_HANDLE: (Stage.TURN_HANDLE,),
    Stage.TURN_HANDLE: (Stage.RETURN_HANDLE,),
    Stage.RETURN_HANDLE: (Stage.OPEN_AND_CLEAR,),
    Stage.OPEN_AND_CLEAR: (Stage.RETREAT_ARM,),
    Stage.RETREAT_ARM: (Stage.RESTORE_GRIPPER,),
    Stage.RESTORE_GRIPPER: (Stage.ALIGN_PANEL,),
    Stage.ALIGN_PANEL: (Stage.CONTACT_PANEL,),
    Stage.CONTACT_PANEL: (Stage.PUSH_OPEN, Stage.THROUGH),
    Stage.PUSH_OPEN: (Stage.PUSH_ARM_RETRACT, Stage.THROUGH),
    Stage.PUSH_ARM_RETRACT: (Stage.BASE_ADVANCE, Stage.THROUGH),
    Stage.BASE_ADVANCE: (Stage.CONTACT_PANEL, Stage.THROUGH),
    Stage.THROUGH: (Stage.SUCCESS,),
    Stage.SUCCESS: (),
    Stage.FAILURE: (),
}


def test_stage_spec_is_the_complete_ordered_runtime_contract():
    assert tuple(STAGE_SPECS) == EXPECTED_STAGES
    assert {stage: spec.budget_steps for stage, spec in STAGE_SPECS.items()} == (
        EXPECTED_BUDGETS
    )
    assert {stage: spec.successors for stage, spec in STAGE_SPECS.items()} == (
        EXPECTED_SUCCESSORS
    )
    assert all(spec.control and spec.acceptance for spec in STAGE_SPECS.values())


def test_transition_validation_consumes_the_declared_graph():
    controller = DoorController.__new__(DoorController)
    controller.stage = Stage.PUSH_OPEN
    controller.stage_steps = 73

    controller._transition_to(Stage.PUSH_ARM_RETRACT)
    assert controller.stage == Stage.PUSH_ARM_RETRACT
    assert controller.stage_steps == 0

    with pytest.raises(RuntimeError, match="illegal AutoOpen transition"):
        controller._transition_to(Stage.GRASP_HANDLE)

    controller.stage_steps = 19
    controller._transition_to(Stage.FAILURE, reset_steps=False)
    assert controller.stage == Stage.FAILURE
    assert controller.stage_steps == 19


def test_runtime_code_cannot_bypass_the_stage_transition_contract():
    direct_writes = []
    for source_path in (AUTO_OPEN_SOURCE, LIFECYCLE_SOURCE):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        for node in ast.walk(tree):
            targets = []
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in {"self", "controller"}
                    and target.attr == "stage"
                ):
                    owner = node
                    while owner is not None and not isinstance(owner, ast.FunctionDef):
                        owner = parents.get(owner)
                    direct_writes.append(
                        (source_path.name, target.value.id, owner.name)
                    )

    assert direct_writes == [
        ("doorman_auto_open.py", "self", "__init__"),
        ("doorman_auto_open.py", "self", "_transition_to"),
        ("lifecycle.py", "self", "reset_controller"),
    ]
