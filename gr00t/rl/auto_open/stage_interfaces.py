"""Closed execution vocabulary for task stages, independent of door direction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BaseMode(str, Enum):
    TARGET = "target"
    TRAJECTORY = "trajectory"


class ArmMode(str, Enum):
    TARGET = "ee_target"
    TRAJECTORY = "ee_trajectory"
    JOINT = "joint_target"


class GoalSource(str, Enum):
    DIRECT = "direct"
    CANDIDATES = "candidates"


@dataclass(frozen=True)
class StageInterface:
    base: BaseMode
    arm: ArmMode | None
    goal_source: GoalSource = GoalSource.DIRECT


def stage_interfaces(stage_enum, *, pull):
    """Fail at construction when a new task stage lacks a standard contract."""
    base_paths = {"MOVE_TO_GAP_STATION", "FOLLOW_WALL_TO_PANEL", "PASS_THROUGH_DOORWAY", "SUCCESS"}
    joint = {
        "APPROACH_HANDLE",
        "MOVE_TO_GAP_STATION",
        "FOLLOW_WALL_TO_PANEL",
        "MOVE_TO_PANEL_STATION",
        "ADVANCE_TO_NEXT_PUSH_STATION",
        "PASS_THROUGH_DOORWAY",
        "SUCCESS",
        "FAILURE",
    }
    arm_paths = {
        "MOVE_TO_PREGRASP",
        "OPEN_GRIPPER",
        "INSERT_AROUND_HANDLE",
        "PRESS_HANDLE",
        "OPEN_DOOR_WITH_HANDLE",
        "RESET_HANDLE",
        "RELEASE_HANDLE",
        "WITHDRAW_FROM_HANDLE",
        "STOW_ARM_AND_CLOSE_GRIPPER",
        "STOW_ARM_AFTER_PULL",
        "INSERT_INTO_DOOR_GAP",
        "PRY_DOOR_OPEN",
    }
    arm_targets = {
        "GRASP_HANDLE",
        "MOVE_TO_PANEL_STANDOFF",
        "PUSH_PANEL_OPEN",
        "STOW_ARM_AFTER_PUSH",
    }
    if not pull:
        arm_paths.add("MOVE_TO_PANEL_STANDOFF")
        arm_targets.remove("MOVE_TO_PANEL_STANDOFF")
    candidates = {
        "MOVE_TO_PREGRASP",
        "STOW_ARM_AND_CLOSE_GRIPPER",
        "MOVE_TO_GAP_STATION",
        "INSERT_INTO_DOOR_GAP",
        "STOW_ARM_AFTER_PULL",
        "MOVE_TO_PANEL_STANDOFF",
        "STOW_ARM_AFTER_PUSH",
    }
    result = {}
    for stage in stage_enum:
        name = stage.name
        if name not in joint | arm_paths | arm_targets:
            raise ValueError(f"No standard execution contract for {name}")
        result[stage] = StageInterface(
            BaseMode.TRAJECTORY if name in base_paths else BaseMode.TARGET,
            None
            if name == "FAILURE"
            else (
                ArmMode.JOINT
                if name in joint
                else (ArmMode.TRAJECTORY if name in arm_paths else ArmMode.TARGET)
            ),
            GoalSource.CANDIDATES if name in candidates else GoalSource.DIRECT,
        )
    return result
