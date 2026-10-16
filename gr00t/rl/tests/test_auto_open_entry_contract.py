import ast
import builtins
import importlib
import symtable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "gr00t" / "rl" / "scripts"
AUTO_OPEN = REPO_ROOT / "gr00t" / "rl" / "auto_open"


def test_all_auto_open_modules_resolve_their_runtime_imports():
    module_names = (
        "cli",
        "control",
        "geometry",
        "io",
        "lifecycle",
        "planning",
        "runtime",
        "target_noise",
        "utils",
    )
    importlib.import_module("gr00t.rl.scripts.doorman_auto_open")
    for name in module_names:
        importlib.import_module(f"gr00t.rl.auto_open.{name}")


def test_auto_open_modules_have_no_unbound_global_references():
    paths = [SCRIPTS / "doorman_auto_open.py", *AUTO_OPEN.glob("*.py")]
    unresolved = []
    for path in paths:
        table = symtable.symtable(
            path.read_text(encoding="utf-8"), str(path), "exec"
        )
        module_bindings = {
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_imported() or symbol.is_assigned() or symbol.is_namespace()
        }
        global_references = set()

        def collect_references(scope):
            for symbol in scope.get_symbols():
                if symbol.is_referenced() and (
                    scope.get_type() == "module" or symbol.is_global()
                ):
                    global_references.add(symbol.get_name())
            for child in scope.get_children():
                collect_references(child)

        collect_references(table)
        missing = global_references - module_bindings - set(dir(builtins))
        missing -= {"__file__", "__name__"}
        unresolved.extend((path.name, name) for name in sorted(missing))

    assert unresolved == []


def test_extracted_static_geometry_helpers_keep_their_original_call_arity():
    expected_arity = {
        "_quat_distance": 2,
        "_rotate_about_z": 3,
        "_yaw_quat": 1,
        "_quat_nlerp": 3,
    }
    invalid_calls = []
    paths = [SCRIPTS / "doorman_auto_open.py", *AUTO_OPEN.glob("*.py")]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in expected_arity
            ):
                continue
            expected = expected_arity[node.func.id]
            if len(node.args) != expected:
                invalid_calls.append(
                    (path.name, node.lineno, node.func.id, len(node.args), expected)
                )

    assert invalid_calls == []


def test_domain_modules_do_not_call_removed_controller_methods():
    orchestrator_tree = ast.parse(
        (SCRIPTS / "doorman_auto_open.py").read_text(encoding="utf-8")
    )
    controller = next(
        node
        for node in orchestrator_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DoorController"
    )
    controller_methods = {
        node.name for node in controller.body if isinstance(node, ast.FunctionDef)
    }

    stale_calls = []
    sources = [
        (AUTO_OPEN / filename, {"self", "controller"})
        for filename in (
            "control.py",
            "geometry.py",
            "lifecycle.py",
            "planning.py",
            "runtime.py",
        )
    ]
    sources.append(
        (
            REPO_ROOT / "gr00t" / "rl" / "data_collection" / "auto_open_collector.py",
            {"controller"},
        )
    )
    for path, controller_names in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in controller_names
                and node.func.attr.startswith("_")
            ):
                continue
            if node.func.attr not in controller_methods:
                stale_calls.append((path.name, node.lineno, node.func.attr))

    assert stale_calls == []


def test_auto_open_has_one_experiment_launcher():
    launchers = sorted(path.name for path in SCRIPTS.glob("run_auto_open*.sh"))
    assert launchers == ["run_auto_open.sh"]


def test_auto_open_uses_batch_invariant_physx_partitioning():
    source = (
        REPO_ROOT
        / "gr00t"
        / "rl"
        / "third_party"
        / "visual_wholebody_lowlevel"
        / "legged_gym"
        / "utils"
        / "helpers.py"
    ).read_text(encoding="utf-8")
    assert "cfg.sim.physx.enable_enhanced_determinism = True" in source
    assert "cfg.sim.physx.gpu_max_num_partitions = 1" in source


def test_scripts_contains_no_auto_open_domain_modules():
    assert sorted(path.name for path in SCRIPTS.glob("auto_open_*.py")) == []
    assert (AUTO_OPEN / "__init__.py").is_file()


def test_domain_package_never_imports_from_executable_scripts():
    violations = []
    for path in AUTO_OPEN.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.module or ""
            ).startswith("gr00t.rl.scripts"):
                violations.append((path.name, node.lineno, node.module))
    assert violations == []


def test_domain_package_dependency_graph_is_acyclic():
    module_paths = {
        path.stem: path for path in AUTO_OPEN.glob("*.py") if path.stem != "__init__"
    }
    dependencies = {name: set() for name in module_paths}
    for name, path in module_paths.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            prefix = "gr00t.rl.auto_open."
            if node.module.startswith(prefix):
                dependency = node.module.removeprefix(prefix)
                if dependency in module_paths:
                    dependencies[name].add(dependency)

    visiting = set()
    visited = set()

    def visit(name):
        assert name not in visiting, f"cyclic AutoOpen dependency through {name}"
        if name in visited:
            return
        visiting.add(name)
        for dependency in dependencies[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in dependencies:
        visit(name)


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
    orchestrator_source = (SCRIPTS / "doorman_auto_open.py").read_text(
        encoding="utf-8"
    )
    runtime_source = (AUTO_OPEN / "runtime.py").read_text(encoding="utf-8")
    orchestrator_tree = ast.parse(orchestrator_source)
    runtime_tree = ast.parse(runtime_source)
    runtime_functions = {
        node.name: node
        for node in runtime_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    orchestrator_functions = {
        node.name: node
        for node in orchestrator_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_run_batched" in runtime_functions
    assert "_run_batched" not in orchestrator_functions
    assert "_serialize_episode" not in runtime_functions

    main_calls = {
        node.func.id
        for node in ast.walk(orchestrator_functions["main"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_run_batched" in main_calls
    assert "_serialize_episode" not in main_calls


def test_every_fixed_workflow_supplies_vectorized_environment_count():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    assert launcher.count('python "${ENTRY}"') == 1
    assert '--batched_envs "${#rank_seeds[@]}"' in launcher
    assert '--episode_seeds "${rank_seeds[@]}"' in launcher


def test_fixed_launcher_has_one_python_invocation_and_one_common_contract():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    assert launcher.count('python "${ENTRY}"') == 1
    assert launcher.count("COMMON_RUNTIME_ARGS=(") == 1
    for option in (
        "--headless",
        "--device cuda:0",
        "--batch_contact_ik_backend exact_workers",
        "--batch_path_backend exact_workers",
        "--batch_panel_clearance_backend gpu",
        "--batch_geometry_fallback_backend numpy_batch",
    ):
        assert launcher.count(option) == 1


def test_standard_collection_uses_one_vectorized_process_per_gpu():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    assert 'rank_count="${gpu_count}"' in launcher
    assert '--episode_seeds "${rank_seeds[@]}"' in launcher
    assert '--batched_envs "${#rank_seeds[@]}"' in launcher
    assert "STANDARD_WORKERS_PER_GPU" not in launcher
    assert "for ((worker_index=" not in launcher


def test_collection_camera_mode_is_configuration_not_a_workflow():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    assert 'CAMERA_MODE="${CAMERA_MODE:-none}"' in launcher
    assert 'CAMERA_MODE}" != none' in launcher
    assert 'CAMERA_MODE}" != dual' in launcher
    assert 'CAMERA_MODE}" != six' in launcher
    assert "AUTO_OPEN_WORKFLOW" not in launcher
