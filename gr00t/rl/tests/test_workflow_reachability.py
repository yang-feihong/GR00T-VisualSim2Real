"""Task modules must be reachable from collection or an explicitly supported tool."""

import ast
import hashlib
from collections import Counter
from pathlib import Path

import pytest
import torch

from gr00t.rl.pick.entry import TOOLS


ROOT = Path(__file__).parents[3]


def workflow_sources():
    paths = [
        *ROOT.glob("gr00t/rl/auto_open/*.py"),
        *ROOT.glob("gr00t/rl/pick/*.py"),
        *ROOT.glob("gr00t/rl/data_collection/*.py"),
        ROOT / "gr00t/rl/task_runtime.py",
        ROOT / "gr00t/rl/batch_workflow.py",
        ROOT / "gr00t/rl/scripts/doorman_auto_open.py",
    ]
    return {
        ".".join(path.relative_to(ROOT).with_suffix("").parts): ast.parse(path.read_text())
        for path in paths
        if path.stem != "__init__"
    }


def test_no_orphan_task_module():
    sources = workflow_sources()
    roots = {
        "gr00t.rl.scripts.doorman_auto_open",
        "gr00t.rl.pick.entry",
        "gr00t.rl.pick.probe_policy",
        "gr00t.rl.batch_workflow",
        *(f"gr00t.rl.pick.{name}" for name in TOOLS.values()),
    }
    # The shared shell invokes these data tools through `python -m`.
    launcher = (ROOT / "gr00t/rl/scripts/run_auto_open.sh").read_text()
    roots.update(module for module in sources if module in launcher)
    reachable, pending = set(), list(roots)
    while pending:
        module = pending.pop()
        if module in reachable or module not in sources:
            continue
        reachable.add(module)
        for node in ast.walk(sources[module]):
            dependencies = []
            if isinstance(node, ast.Import):
                dependencies = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join(module.split(".")[: -node.level] + ([base] if base else []))
                dependencies = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            pending.extend(dependencies)
    assert (
        sorted(module for module in sources if ".pick." in module and module not in reachable) == []
    )


def test_no_unreferenced_private_module_function():
    sources = workflow_sources()
    references = Counter()
    for tree in sources.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                references[node.id] += 1
            elif isinstance(node, ast.Attribute):
                references[node.attr] += 1
            elif isinstance(node, ast.ImportFrom):
                references.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Hooks are sometimes looked up by their exact method name.
                references[node.value] += 1
    unused = [
        f"{module}:{node.name}"
        for module, tree in sources.items()
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("_")
        and not references[node.name]
    ]
    assert not unused, "\n".join(unused)


@pytest.mark.parametrize("initialize", [False, True])
def test_cleanup_preserves_all_consumed_fsm_tensors_and_commands(monkeypatch, initialize):
    """Golden values captured from the verified pre-cleanup constructor and command."""
    from gr00t.rl.tests.test_pick_workflow import command, fake_controller, tensor_task

    current, state = fake_controller(monkeypatch, initialize=initialize)

    def signature(task):
        published = command(task, state)
        digest = hashlib.sha256()
        groups = {
            "fsm": vars(task),
            "base_path": vars(task.base_path),
            "arm_path": vars(task.arm_path),
            "command": vars(published),
        }
        for group, values in sorted(groups.items()):
            for name, value in sorted(values.items()):
                if torch.is_tensor(value):
                    digest.update(f"{group}.{name}:{value.dtype}:{tuple(value.shape)}".encode())
                    digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    actual = signature(tensor_task(current, state))
    reference = ROOT / "datasets/pick/.cache/init_speed_gaze_tmp/gr00t/rl/pick/controller.py"
    if reference.exists():
        from gr00t.rl.auto_open.acceptance import (
            StageAcceptanceMixin,
            TensorContactWindow,
            TensorJointStabilityWindow,
            TensorStabilityWindow,
        )
        from gr00t.rl.pick import controller

        cls = next(
            node
            for node in ast.parse(reference.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "PickController"
        )
        scope = dict(vars(controller))
        scope.update(
            StageAcceptanceMixin=StageAcceptanceMixin,
            TensorContactWindow=TensorContactWindow,
            TensorJointStabilityWindow=TensorJointStabilityWindow,
            TensorStabilityWindow=TensorStabilityWindow,
        )
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(reference), "exec"), scope)
        previous = scope[cls.name](
            current.env,
            current.episode,
            current.geometry,
            collision=current.collision,
            initialize=initialize,
            defer_initial_station=True,
            defer_approach_gaze=True,
        )
        assert signature(tensor_task(previous, state)) == actual
    expected = {
        False: "f461fbda1615ce4ba983ac410276c67c7f7462d519712b17626e201503d712e7",
        True: "f584c69fcb2c9a3d54318fa6fd9fbe4b9214ccbbcd9eb15c3befdd200d53a914",
    }
    assert actual == expected[initialize]


def test_scalar_pick_stage_lifecycle_and_reposition_protocol_are_retired():
    from gr00t.rl.pick.basket_controller import BasketPickController
    from gr00t.rl.pick.controller import PickController

    assert not (ROOT / "gr00t/rl/pick/basket_reposition.py").exists()
    for cls in (PickController, BasketPickController):
        for name in (
            "prepare_approach",
            "transition",
            "fail",
            "begin_reposition",
            "enter_reposition",
        ):
            assert not hasattr(cls, name), (cls.__name__, name)
