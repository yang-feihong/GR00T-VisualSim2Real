from pathlib import Path

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
    Stage.PRESS_HANDLE,
    Stage.PUSH_PULL_DOOR,
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
    Stage.PRESS_HANDLE: 900,
    Stage.PUSH_PULL_DOOR: 900,
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
    Stage.GRASP_HANDLE: (Stage.PRESS_HANDLE,),
    Stage.PRESS_HANDLE: (Stage.PUSH_PULL_DOOR,),
    Stage.PUSH_PULL_DOOR: (Stage.RETURN_HANDLE,),
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


def test_door_controller_has_no_second_fsm_execution_path():
    assert not hasattr(DoorController, "command")
    assert not hasattr(DoorController, "advance")
