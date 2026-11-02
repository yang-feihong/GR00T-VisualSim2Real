"""Architecture extraction must preserve the committed door decision algorithms."""

import ast
import subprocess
from pathlib import Path

import pytest
import torch

from gr00t.rl.auto_open.stages import TensorStageLifecycle


ROOT = Path(__file__).parents[3]
BASELINE = "b56eae4"


def original(relative):
    return ast.parse(
        subprocess.check_output(["git", "show", f"{BASELINE}:{relative}"], cwd=ROOT, text=True)
    )


def definitions(tree, *, cls=None):
    body = tree.body
    if cls is not None:
        body = next(
            node for node in body if isinstance(node, ast.ClassDef) and node.name == cls
        ).body
    return {node.name: node for node in body if isinstance(node, ast.FunctionDef)}


@pytest.mark.parametrize(
    "relative,cls,excluded",
    [
        (
            "gr00t/rl/auto_open/tensor_fsm.py",
            "TensorDoorFSM",
            {"__init__", "mask", "transition", "fail"},
        ),
        ("gr00t/rl/auto_open/tensor_planning.py", "TensorPlanningCoordinator", set()),
        ("gr00t/rl/auto_open/tensor_stage_planning.py", "GPUStagePlanningMixin", set()),
        ("gr00t/rl/auto_open/geometry.py", None, set()),
        ("gr00t/rl/auto_open/pull_entry_gpu.py", None, set()),
    ],
)
def test_existing_door_decision_functions_are_ast_identical(relative, cls, excluded):
    before = definitions(original(relative), cls=cls)
    after = definitions(ast.parse((ROOT / relative).read_text()), cls=cls)
    removed_dead_helpers = {
        "_axes",
        "_axial_end_margins",
        "_axial_target_correction",
        "_quat_align_vectors_np",
        "_center_handle_axial_overlap",
    }
    assert set(before) - set(after) <= excluded | removed_dead_helpers
    for name in before.keys() & after.keys() - excluded:
        assert ast.dump(before[name]) == ast.dump(after[name]), f"{relative}:{name}"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_extracted_lifecycle_matches_original_for_every_row_and_diagnostic(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    before = definitions(original("gr00t/rl/auto_open/tensor_fsm.py"), cls="TensorDoorFSM")
    scope = {"torch": torch}
    for name in ("mask", "transition", "fail"):
        exec(compile(ast.Module(body=[before[name]], type_ignores=[]), "baseline", "exec"), scope)
    from types import SimpleNamespace

    def fixture():
        return SimpleNamespace(
            stage=torch.tensor([1, 2, 3, 2], device=device),
            stage_steps=torch.arange(4, device=device),
            stage_visit=torch.ones(4, 6, dtype=torch.int32, device=device),
            failure_from_stage=torch.zeros(4, dtype=torch.int16, device=device),
            failure_code=torch.zeros(4, dtype=torch.int16, device=device),
            Stage=SimpleNamespace(FAILURE=5),
        )

    left, right = fixture(), fixture()
    fields = {
        "current",
        "true_frames",
        "streak",
        "max_streak",
        "sole_blocker_frames",
        "value",
        "min_value",
        "max_value",
        "min_failed_value",
        "max_failed_value",
        "min_value_step",
        "max_value_step",
        "min_failed_value_step",
        "max_failed_value_step",
        "failed_streak",
        "max_failed_streak",
        "max_failed_streak_end_step",
    }
    for obj in (left, right):
        obj._acceptance = {
            (stage, "metric"): {key: torch.ones(4, device=device) for key in fields}
            for stage in (2, 3)
        }
    rows = torch.tensor([1, 3], device=device)
    scope["transition"](left, rows, 3)
    TensorStageLifecycle.transition(right, rows, 3)
    scope["fail"](left, rows[:1], 7)
    TensorStageLifecycle.fail(right, rows[:1], 7)
    for key in ("stage", "stage_steps", "stage_visit", "failure_from_stage", "failure_code"):
        torch.testing.assert_close(getattr(left, key), getattr(right, key), rtol=0, atol=0)
    for key, values in left._acceptance.items():
        for name, value in values.items():
            torch.testing.assert_close(
                value, right._acceptance[key][name], rtol=0, atol=0, equal_nan=True
            )
