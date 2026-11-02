"""Stage specifications and tensor lifecycle shared by all task graphs."""

from dataclasses import dataclass
from enum import IntEnum


@dataclass(frozen=True)
class StageSpec:
    control: str
    acceptance: str
    budget_steps: int | None
    successors: tuple[IntEnum, ...]


def validate_stage_graph(stage_enum, specs, interfaces):
    stages = set(stage_enum)
    if set(specs) != stages or set(interfaces) != stages:
        raise ValueError("Every stage requires a specification and standard execution interface")
    for stage, spec in specs.items():
        if not spec.control or not spec.acceptance:
            raise ValueError(f"Missing stage definition: {stage.name}")
        if spec.budget_steps is not None and spec.budget_steps <= 0:
            raise ValueError(f"Invalid stage budget: {stage.name}")
        if not set(spec.successors) <= stages:
            raise ValueError(f"Unknown successor: {stage.name}")


class TensorStageLifecycle:
    """Existing door FSM row transitions, including per-visit diagnostic reset."""

    def mask(self, stage):
        return self.stage == int(stage)

    def transition(self, rows, stage):
        self.stage[rows] = int(stage)
        self.stage_steps[rows] = 0
        if hasattr(self, "stage_visit"):
            self.stage_visit[rows, int(stage)] += 1
        for (stage_id, _), entry in getattr(self, "_acceptance", {}).items():
            if stage_id != int(stage):
                continue
            for name in (
                "current",
                "true_frames",
                "streak",
                "max_streak",
                "sole_blocker_frames",
                "failed_streak",
                "max_failed_streak",
            ):
                entry[name][rows] = 0
            entry["value"][rows] = float("nan")
            for name in ("min_value", "min_failed_value"):
                entry[name][rows] = float("inf")
            for name in ("max_value", "max_failed_value"):
                entry[name][rows] = -float("inf")
            for name in (
                "min_value_step",
                "max_value_step",
                "min_failed_value_step",
                "max_failed_value_step",
                "max_failed_streak_end_step",
            ):
                entry[name][rows] = -1

    def fail(self, rows, code):
        if not len(rows):
            return
        self.failure_from_stage[rows] = self.stage[rows].to(self.failure_from_stage.dtype)
        self.failure_code[rows] = int(code)
        self.stage[rows] = int(self.Stage.FAILURE)


class StagePlanningEvents:
    """Dispatch the existing FSM event dictionary to stage-entry GPU planners."""

    def plan_events(self, fsm, state, events):
        unknown = set(events) - set(self.planning_events) - {"success", "timeout"}
        if unknown:
            raise ValueError(f"Unregistered stage planning events: {sorted(unknown)}")
        for label, method in self.planning_events.items():
            rows = events.get(label)
            if rows is not None and len(rows) and method is not None:
                getattr(self, method)(fsm, state, rows)
