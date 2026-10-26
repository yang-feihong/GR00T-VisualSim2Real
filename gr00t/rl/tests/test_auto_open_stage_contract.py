from pathlib import Path

from gr00t.rl.scripts.doorman_auto_open import DoorController, Stage, STAGE_SPECS


REPO_ROOT = Path(__file__).resolve().parents[3]
AUTO_OPEN_SOURCE = REPO_ROOT / "gr00t" / "rl" / "scripts" / "doorman_auto_open.py"
LIFECYCLE_SOURCE = REPO_ROOT / "gr00t" / "rl" / "auto_open" / "lifecycle.py"


EXPECTED_STAGES = (
    Stage.APPROACH_HANDLE,
    Stage.MOVE_TO_PREGRASP,
    Stage.OPEN_GRIPPER,
    Stage.INSERT_AROUND_HANDLE,
    Stage.GRASP_HANDLE,
    Stage.PRESS_HANDLE,
    Stage.OPEN_DOOR_WITH_HANDLE,
    Stage.RESET_HANDLE,
    Stage.RELEASE_HANDLE,
    Stage.WITHDRAW_FROM_HANDLE,
    Stage.STOW_ARM_AND_CLOSE_GRIPPER,
    Stage.MOVE_TO_GAP_STATION,
    Stage.INSERT_INTO_DOOR_GAP,
    Stage.PRY_DOOR_OPEN,
    Stage.STOW_ARM_AFTER_PULL,
    Stage.FOLLOW_WALL_TO_PANEL,
    Stage.MOVE_TO_PANEL_STATION,
    Stage.MOVE_TO_PANEL_STANDOFF,
    Stage.PUSH_PANEL_OPEN,
    Stage.STOW_ARM_AFTER_PUSH,
    Stage.ADVANCE_TO_NEXT_PUSH_STATION,
    Stage.PASS_THROUGH_DOORWAY,
    Stage.SUCCESS,
    Stage.FAILURE,
)


EXPECTED_BUDGETS = {
    Stage.APPROACH_HANDLE: 1000,
    Stage.MOVE_TO_PREGRASP: 500,
    Stage.OPEN_GRIPPER: 100,
    Stage.INSERT_AROUND_HANDLE: 800,
    Stage.GRASP_HANDLE: 400,
    Stage.PRESS_HANDLE: 900,
    Stage.OPEN_DOOR_WITH_HANDLE: 900,
    Stage.RESET_HANDLE: 150,
    Stage.RELEASE_HANDLE: 100,
    Stage.WITHDRAW_FROM_HANDLE: 200,
    Stage.STOW_ARM_AND_CLOSE_GRIPPER: 300,
    Stage.MOVE_TO_GAP_STATION: 1000,
    Stage.INSERT_INTO_DOOR_GAP: 900,
    Stage.PRY_DOOR_OPEN: 900,
    Stage.STOW_ARM_AFTER_PULL: 300,
    Stage.FOLLOW_WALL_TO_PANEL: 1000,
    Stage.MOVE_TO_PANEL_STATION: 400,
    Stage.MOVE_TO_PANEL_STANDOFF: 400,
    Stage.PUSH_PANEL_OPEN: 900,
    Stage.STOW_ARM_AFTER_PUSH: 300,
    Stage.ADVANCE_TO_NEXT_PUSH_STATION: 900,
    Stage.PASS_THROUGH_DOORWAY: 1100,
    Stage.SUCCESS: None,
    Stage.FAILURE: None,
}


EXPECTED_SUCCESSORS = {
    Stage.APPROACH_HANDLE: (Stage.MOVE_TO_PREGRASP,),
    Stage.MOVE_TO_PREGRASP: (Stage.OPEN_GRIPPER,),
    Stage.OPEN_GRIPPER: (Stage.INSERT_AROUND_HANDLE,),
    Stage.INSERT_AROUND_HANDLE: (Stage.GRASP_HANDLE,),
    Stage.GRASP_HANDLE: (Stage.PRESS_HANDLE,),
    Stage.PRESS_HANDLE: (Stage.OPEN_DOOR_WITH_HANDLE,),
    Stage.OPEN_DOOR_WITH_HANDLE: (Stage.RESET_HANDLE,),
    Stage.RESET_HANDLE: (Stage.RELEASE_HANDLE,),
    Stage.RELEASE_HANDLE: (Stage.WITHDRAW_FROM_HANDLE,),
    Stage.WITHDRAW_FROM_HANDLE: (Stage.STOW_ARM_AND_CLOSE_GRIPPER,),
    Stage.STOW_ARM_AND_CLOSE_GRIPPER: (Stage.MOVE_TO_PANEL_STATION, Stage.MOVE_TO_GAP_STATION),
    Stage.MOVE_TO_GAP_STATION: (Stage.INSERT_INTO_DOOR_GAP,),
    Stage.INSERT_INTO_DOOR_GAP: (
        Stage.PRY_DOOR_OPEN,
        Stage.STOW_ARM_AFTER_PULL,
        Stage.MOVE_TO_GAP_STATION,
    ),
    Stage.PRY_DOOR_OPEN: (Stage.STOW_ARM_AFTER_PULL,),
    Stage.STOW_ARM_AFTER_PULL: (Stage.FOLLOW_WALL_TO_PANEL, Stage.PASS_THROUGH_DOORWAY),
    Stage.FOLLOW_WALL_TO_PANEL: (Stage.MOVE_TO_PANEL_STANDOFF, Stage.PASS_THROUGH_DOORWAY),
    Stage.MOVE_TO_PANEL_STATION: (Stage.MOVE_TO_PANEL_STANDOFF,),
    Stage.MOVE_TO_PANEL_STANDOFF: (Stage.PUSH_PANEL_OPEN, Stage.PASS_THROUGH_DOORWAY),
    Stage.PUSH_PANEL_OPEN: (
        Stage.STOW_ARM_AFTER_PUSH,
        Stage.PASS_THROUGH_DOORWAY,
        Stage.STOW_ARM_AFTER_PULL,
    ),
    Stage.STOW_ARM_AFTER_PUSH: (Stage.ADVANCE_TO_NEXT_PUSH_STATION, Stage.PASS_THROUGH_DOORWAY),
    Stage.ADVANCE_TO_NEXT_PUSH_STATION: (Stage.MOVE_TO_PANEL_STANDOFF, Stage.PASS_THROUGH_DOORWAY),
    Stage.PASS_THROUGH_DOORWAY: (Stage.SUCCESS,),
    Stage.SUCCESS: (),
    Stage.FAILURE: (),
}


def test_stage_spec_is_the_complete_ordered_runtime_contract():
    assert tuple(STAGE_SPECS) == EXPECTED_STAGES
    assert {stage: spec.budget_steps for stage, spec in STAGE_SPECS.items()} == (EXPECTED_BUDGETS)
    assert {stage: spec.successors for stage, spec in STAGE_SPECS.items()} == (EXPECTED_SUCCESSORS)
    assert all(spec.control and spec.acceptance for spec in STAGE_SPECS.values())


def test_stage_names_have_stable_numeric_ids_and_no_aliases():
    expected = {
        "APPROACH_HANDLE": 1,
        "MOVE_TO_PREGRASP": 2,
        "OPEN_GRIPPER": 3,
        "INSERT_AROUND_HANDLE": 4,
        "GRASP_HANDLE": 5,
        "PRESS_HANDLE": 6,
        "OPEN_DOOR_WITH_HANDLE": 7,
        "RESET_HANDLE": 8,
        "RELEASE_HANDLE": 9,
        "WITHDRAW_FROM_HANDLE": 10,
        "STOW_ARM_AND_CLOSE_GRIPPER": 11,
        "MOVE_TO_PANEL_STATION": 12,
        "MOVE_TO_PANEL_STANDOFF": 13,
        "PUSH_PANEL_OPEN": 14,
        "STOW_ARM_AFTER_PUSH": 15,
        "ADVANCE_TO_NEXT_PUSH_STATION": 16,
        "PASS_THROUGH_DOORWAY": 17,
        "SUCCESS": 18,
        "FAILURE": 19,
        "MOVE_TO_GAP_STATION": 20,
        "INSERT_INTO_DOOR_GAP": 21,
        "PRY_DOOR_OPEN": 22,
        "STOW_ARM_AFTER_PULL": 23,
        "FOLLOW_WALL_TO_PANEL": 24,
    }
    assert {name: int(stage) for name, stage in Stage.__members__.items()} == expected


def test_door_controller_has_no_second_fsm_execution_path():
    assert not hasattr(DoorController, "command")
    assert not hasattr(DoorController, "advance")
