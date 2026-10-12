import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "gr00t" / "rl" / "scripts"


def test_auto_open_has_one_experiment_launcher():
    launchers = sorted(path.name for path in SCRIPTS.glob("run_auto_open*.sh"))
    assert launchers == ["run_auto_open.sh"]


def test_direct_python_entry_requires_the_fixed_launcher_identity():
    source = (SCRIPTS / "doorman_auto_open.py").read_text(encoding="utf-8")
    assert 'AUTO_OPEN_FIXED_ENTRY") != "run_auto_open.sh"' in source
    assert "run_auto_open_*.sh" not in source


def test_no_alternate_python_script_wraps_the_auto_open_entry():
    wrappers = []
    for path in SCRIPTS.glob("*.py"):
        if path.name == "doorman_auto_open.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "from doorman_auto_open import" in source:
            wrappers.append(path.name)
    assert wrappers == []


def test_one_door_controller_class_owns_scalar_and_batch_state_machine():
    source = (SCRIPTS / "doorman_auto_open.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    controller_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DoorController"
    ]
    assert len(controller_classes) == 1
    methods = {
        node.name
        for node in controller_classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"command", "advance"} <= methods
    assert "ScalarDoorController" not in source
    assert "BatchedDoorController" not in source


def test_one_vectorized_episode_runner_handles_batch_size_one_and_many():
    source = (SCRIPTS / "doorman_auto_open.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_run_batched" in top_level_functions
    assert "_serialize_episode" not in top_level_functions

    main_calls = {
        node.func.id
        for node in ast.walk(top_level_functions["main"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_run_batched" in main_calls
    assert "_serialize_episode" not in main_calls


def test_every_fixed_workflow_supplies_vectorized_environment_count():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    python_invocations = [
        block for block in launcher.split('python "${ENTRY}"')[1:]
    ]
    assert python_invocations
    assert all("--batched_envs" in block.split(">", 1)[0] for block in python_invocations)
