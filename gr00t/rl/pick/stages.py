"""Pick task graphs using the same StageSpec and execution contracts as doors."""

from enum import IntEnum

from gr00t.rl.auto_open.stage_interfaces import ArmMode, BaseMode, GoalSource, StageInterface
from gr00t.rl.auto_open.stages import StageSpec, validate_stage_graph


class PickStage(IntEnum):
    APPROACH_OBJECT = 1
    MOVE_TO_GRASP = 2
    GRASP_OBJECT = 3
    LIFT_OBJECT = 4
    WITHDRAW_WITH_OBJECT = 5
    VERIFY_RETENTION = 6
    SUCCESS = 7
    FAILURE = 8


BasketStage = IntEnum(
    "BasketStage",
    {
        **{stage.name: int(stage) for stage in PickStage},
        "OPEN_FOR_REPOSITION": 9,
        "CLEAR_FOR_REPOSITION": 10,
        "CLOSE_FOR_REPOSITION": 11,
        "REPOSITION_OBJECT": 12,
        "REOPEN_AFTER_REPOSITION": 13,
    },
)


def stage_definitions(*, basket):
    S = BasketStage if basket else PickStage
    specs = {
        S.APPROACH_OBJECT: StageSpec(
            "follow base approach and aimed-camera EE trajectories",
            "base reaches the approach station and is stable",
            1500,
            (S.MOVE_TO_GRASP,),
        ),
        S.MOVE_TO_GRASP: StageSpec(
            "hold base and follow the selected grasp trajectory",
            "EE trajectory completed with stable accurate pose",
            700,
            (S.GRASP_OBJECT, S.MOVE_TO_GRASP),
        ),
        S.GRASP_OBJECT: StageSpec(
            "hold EE pose and close the gripper",
            "stable force-qualified opposed jaw contacts",
            700,
            (S.LIFT_OBJECT,),
        ),
        S.LIFT_OBJECT: StageSpec(
            "hold base and lift the grasped object along an EE trajectory",
            "stable EE at lift goal, object retained above required height",
            700,
            (S.VERIFY_RETENTION,) if basket else (S.WITHDRAW_WITH_OBJECT,),
        ),
        S.WITHDRAW_WITH_OBJECT: StageSpec(
            "hold base and follow the object withdrawal trajectory",
            "stable EE at withdrawal goal with object retained",
            700,
            (S.VERIFY_RETENTION,),
        ),
        S.VERIFY_RETENTION: StageSpec(
            "hold base and grasped object",
            "bilateral grasp and lifted object remain stable for 50 frames",
            700,
            (S.SUCCESS,),
        ),
        S.SUCCESS: StageSpec("stop base and retain object", "terminal success", None, ()),
        S.FAILURE: StageSpec("stop base", "terminal failure", None, ()),
    }
    if basket:
        for stage in (S.MOVE_TO_GRASP, S.GRASP_OBJECT):
            spec = specs[stage]
            extra = (S.OPEN_FOR_REPOSITION,)
            if stage == S.GRASP_OBJECT:
                extra += (S.MOVE_TO_GRASP,)
            specs[stage] = StageSpec(
                spec.control, spec.acceptance, spec.budget_steps, spec.successors + extra
            )
        for stage, control, acceptance, successor in (
            (
                S.OPEN_FOR_REPOSITION,
                "hold EE and open jaws",
                "jaws open and stable",
                S.CLEAR_FOR_REPOSITION,
            ),
            (
                S.CLEAR_FOR_REPOSITION,
                "follow clearance EE trajectory",
                "EE stable at clearance goal",
                S.CLOSE_FOR_REPOSITION,
            ),
            (
                S.CLOSE_FOR_REPOSITION,
                "hold EE and close jaws",
                "jaws closed and stable",
                S.REPOSITION_OBJECT,
            ),
            (
                S.REPOSITION_OBJECT,
                "follow closed-jaw nudge trajectory",
                "object shifted with insertion space",
                S.REOPEN_AFTER_REPOSITION,
            ),
            (
                S.REOPEN_AFTER_REPOSITION,
                "hold EE and reopen jaws",
                "jaws open and safe grasp available",
                S.MOVE_TO_GRASP,
            ),
        ):
            specs[stage] = StageSpec(control, acceptance, 700, (successor,))
        spec = specs[S.REOPEN_AFTER_REPOSITION]
        specs[S.REOPEN_AFTER_REPOSITION] = StageSpec(
            spec.control,
            spec.acceptance,
            spec.budget_steps,
            spec.successors + (S.OPEN_FOR_REPOSITION,),
        )
    paths = {S.APPROACH_OBJECT, S.MOVE_TO_GRASP, S.LIFT_OBJECT, S.WITHDRAW_WITH_OBJECT}
    candidates = {S.APPROACH_OBJECT, S.MOVE_TO_GRASP, S.LIFT_OBJECT, S.WITHDRAW_WITH_OBJECT}
    if basket:
        paths |= {S.CLEAR_FOR_REPOSITION, S.REPOSITION_OBJECT}
        candidates |= {S.CLEAR_FOR_REPOSITION, S.REPOSITION_OBJECT}
    interfaces = {
        stage: StageInterface(
            BaseMode.TRAJECTORY if stage == S.APPROACH_OBJECT and not basket else BaseMode.TARGET,
            None
            if stage == S.FAILURE
            else ArmMode.TRAJECTORY
            if stage in paths
            else ArmMode.TARGET,
            GoalSource.CANDIDATES if stage in candidates else GoalSource.DIRECT,
        )
        for stage in S
    }
    validate_stage_graph(S, specs, interfaces)
    return S, specs, interfaces
