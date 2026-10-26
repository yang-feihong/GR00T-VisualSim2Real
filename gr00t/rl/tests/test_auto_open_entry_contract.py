import ast
import builtins
import importlib
import symtable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "gr00t" / "rl" / "scripts"
AUTO_OPEN = REPO_ROOT / "gr00t" / "rl" / "auto_open"
MANIP_LOCO = (
    REPO_ROOT
    / "gr00t/rl/third_party/visual_wholebody_lowlevel/legged_gym/envs/manip_loco/manip_loco.py"
)


def test_retired_cpu_planning_paths_are_not_present():
    utils = MANIP_LOCO.parents[2] / "utils"
    for path in (
        AUTO_OPEN / "planning.py",
        utils / "ee_steady_state.py",
        utils / "steady_path_worker.py",
    ):
        assert not path.exists(), path
    retired = {"ee_steady_state", "steady_path_worker", "gr00t.rl.auto_open.planning"}
    for path in (*AUTO_OPEN.glob("*.py"), SCRIPTS / "doorman_auto_open.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                modules.extend(alias.name for alias in node.names)
            assert not any(
                module == name or module.endswith("." + name)
                for module in modules
                for name in retired
            ), path


def test_default_arm_presets_match_requested_joint_angles():
    config = ast.parse((MANIP_LOCO.parent / "b2z1_config.py").read_text())
    xr = next(n for n in ast.walk(config) if isinstance(n, ast.ClassDef) and n.name == "xr_teleop")
    presets = {
        n.targets[0].id: ast.literal_eval(n.value) for n in xr.body if isinstance(n, ast.Assign)
    }
    assert presets["front_preset_q"] == "-1.57,2.6,-1.5,0.4,0,0"
    assert presets["back_preset_q"] == "1.57,1.57,-1.3,1.1,0,0"
    cli = ast.parse((AUTO_OPEN / "cli.py").read_text())
    middle = next(
        k.value
        for n in ast.walk(cli)
        if isinstance(n, ast.Call)
        for k in n.keywords
        if k.arg == "doorman_arm_joint_init_center"
    )
    assert ast.literal_eval(middle) == (-1.57, 1.75, -0.785, -0.93, 0.0, 0.0)


def test_all_auto_open_modules_resolve_their_runtime_imports():
    module_names = sorted(path.stem for path in AUTO_OPEN.glob("*.py") if path.stem != "__init__")
    importlib.import_module("gr00t.rl.scripts.doorman_auto_open")
    for name in module_names:
        importlib.import_module(f"gr00t.rl.auto_open.{name}")


def test_auto_open_modules_have_no_unbound_global_references():
    paths = [SCRIPTS / "doorman_auto_open.py", *AUTO_OPEN.glob("*.py")]
    unresolved = []
    for path in paths:
        table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
        module_bindings = {
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_imported() or symbol.is_assigned() or symbol.is_namespace()
        }
        global_references = set()

        def collect_references(scope):
            for symbol in scope.get_symbols():
                if symbol.is_referenced() and (scope.get_type() == "module" or symbol.is_global()):
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
    orchestrator_tree = ast.parse((SCRIPTS / "doorman_auto_open.py").read_text(encoding="utf-8"))
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
    assert launchers == ["run_auto_open.sh", "run_auto_open_pitch_probe.sh"]


def test_debug_workflow_always_publishes_six_camera_mosaics():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    assert "write_six_camera_mosaics" in launcher
    assert 'if [[ "${AUTO_OPEN_MODE}" == debug ]]' in launcher
    assert "common_args+=(--debug_trace)" in launcher


def test_canonical_mode_reuses_official_no_camera_phase_without_replay():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    assert 'AUTO_OPEN_MODE}" == canonical' in launcher
    assert "canonical validation completed" in launcher


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
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "gr00t.rl.scripts"
            ):
                violations.append((path.name, node.lineno, node.module))
    assert violations == []


def test_domain_package_dependency_graph_is_acyclic():
    module_paths = {path.stem: path for path in AUTO_OPEN.glob("*.py") if path.stem != "__init__"}
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
    literals = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"run_auto_open.sh", "run_auto_open_pitch_probe.sh"} <= literals
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


def test_tensor_fsm_is_the_only_auto_open_execution_path():
    source = (SCRIPTS / "doorman_auto_open.py").read_text(encoding="utf-8")
    runtime_source = (AUTO_OPEN / "runtime.py").read_text(encoding="utf-8")
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
    assert "command" not in methods
    assert "advance" not in methods
    assert "fsm.command(" in runtime_source
    assert "fsm.advance(" in runtime_source
    assert "controller.command(" not in runtime_source
    assert "controller.advance(" not in runtime_source
    assert "ScalarDoorController" not in source
    assert "BatchedDoorController" not in source


def test_one_vectorized_episode_runner_handles_batch_size_one_and_many():
    orchestrator_source = (SCRIPTS / "doorman_auto_open.py").read_text(encoding="utf-8")
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
    assert launcher.count("common_args=(") == 1
    for option in (
        "--headless",
        "--device cuda:0",
    ):
        assert launcher.count(option) == 1


def test_gpu_runtime_has_no_obsolete_backend_selection():
    removed = (
        "--steady_path_workers",
        "--batch_contact_ik_backend",
        "--batch_path_backend",
        "--batch_panel_clearance_backend",
        "--batch_geometry_fallback_backend",
    )
    for path in (
        AUTO_OPEN / "cli.py",
        SCRIPTS / "run_auto_open.sh",
        SCRIPTS / "run_auto_open_pitch_probe.sh",
    ):
        source = path.read_text(encoding="utf-8")
        assert all(option not in source for option in removed)


def test_runtime_keeps_reference_snapshots_out_of_the_online_loop():
    tree = ast.parse((AUTO_OPEN / "runtime.py").read_text(encoding="utf-8"))
    loop = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_tensor_fsm_loop"
    )
    calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(node.func, ast.Name)
        and node.func.id
        in (
            "_update_batched_reference_buffers",
            "_snapshot_initial_reference_buffers",
            "batched_target_height_invariant_poses",
        )
        for node in calls
    )
    finalizers = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "finalize_diagnostics"
    ]
    assert len(finalizers) == 1
    online_loop = next(node for node in loop.body if isinstance(node, ast.For))
    assert finalizers[0].lineno > online_loop.end_lineno


def test_runtime_has_no_scalar_runner_adapters_or_executor_lifecycle():
    source = (AUTO_OPEN / "runtime.py").read_text(encoding="utf-8")
    for removed in (
        "steady_path_executor",
        "_BatchedRunnerView",
        "_BatchedEnvView",
        "_make_batched_command_buffers",
        "_batched_state_rows",
        "_update_batched_reference_buffers",
        "_snapshot_initial_reference_buffers",
    ):
        assert removed not in source
    tree = ast.parse(source)
    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TensorPlanningCoordinator"
    ]
    assert len(constructors) == 1
    assert [arg.id for arg in constructors[0].args] == ["controllers", "Stage"]
    assert constructors[0].keywords == []


def test_runtime_audit_fk_uses_pre_step_device_root_snapshot():
    tree = ast.parse((AUTO_OPEN / "runtime.py").read_text(encoding="utf-8"))
    loop = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_tensor_fsm_loop"
    )
    calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)]
    step = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "step_once"
    )
    snapshot = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "capture_root_states"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
    )
    assert snapshot.lineno < step.lineno
    assert ast.unparse(snapshot.value) == "device_state['root_states'].detach().clone()"
    for name in ("_target_height_invariant_poses_gpu", "b2z1_gripper_pose_and_jacobian"):
        call = next(
            node for node in calls if isinstance(node.func, ast.Name) and node.func.id == name
        )
        assert call.lineno > step.lineno
        assert "capture_root_states" in ast.unparse(call.args[0])


def test_standard_collection_uses_one_vectorized_process_per_gpu():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    assert 'rank_count="${#GPU_IDS_ARRAY[@]}"' in launcher
    assert '--episode_seeds "${rank_seeds[@]}"' in launcher
    assert '--batched_envs "${#rank_seeds[@]}"' in launcher
    assert "STANDARD_WORKERS_PER_GPU" not in launcher
    assert "for ((worker_index=" not in launcher


def test_fixed_launcher_owns_the_complete_official_two_phase_workflow():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    assert 'run_phase "${canonical_root}" none "" ALL_SEEDS' in launcher
    assert 'run_phase "${pending_root}" dual "${canonical_root}" ACCEPTED_SEEDS' in launcher
    assert 'if bool(row.get("accepted"))' in launcher
    assert 'report.get("quarantined_episodes") != 0' in launcher
    assert "ln -sfn" not in launcher


def test_fixed_launcher_exposes_only_range_and_two_workflow_modes():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    assert "SEED_START and SEED_END" in launcher
    assert "for ((seed=SEED_START; seed<=SEED_END; seed++))" in launcher
    assert 'AUTO_OPEN_MODE="${AUTO_OPEN_MODE:-official}"' in launcher
    assert 'AUTO_OPEN_MODE}" != official' in launcher
    assert 'AUTO_OPEN_MODE}" != debug' in launcher
    for removed_assignment in (
        "DATASET_ROOT=",
        "CAMERA_MODE=",
        "REPLAY_SOURCE_DATASET_ROOT=",
        "NUM_EPISODES=",
    ):
        assert removed_assignment not in launcher
    assert "SEMANTIC_TARGET_NOISE" not in launcher


def test_debug_workflow_runs_six_camera_collection_then_strict_selection():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    assert 'if [[ "${AUTO_OPEN_MODE}" == debug ]]' in launcher
    assert 'run_phase "${debug_pending_root}" six "" ALL_SEEDS' in launcher
    assert 'if bool(row.get("accepted"))' in launcher
    assert '"${FORMAL_DATA_ROOT}/debug/${debug_name}"' in launcher
    assert "latest" not in launcher


def test_official_release_name_contains_range_count_camera_and_date():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    assert 'range_tag="seeds${SEED_START}-${SEED_END}"' in launcher
    assert "episodes${accepted_count}_dual_camera_${RELEASE_STAMP}" in launcher
    assert "canonical_no_camera_${RELEASE_STAMP}" in launcher
    assert 'FORMAL_DATA_ROOT="${REPO_ROOT}/datasets/auto_open"' in launcher


def test_every_official_run_is_an_independent_timestamped_dataset():
    launcher = (SCRIPTS / "run_auto_open.sh").read_text(encoding="utf-8")
    assert "canonical_no_camera_${RELEASE_STAMP}" in launcher
    assert "reusing complete canonical" not in launcher
    assert "official release already complete" not in launcher
    assert '"source_fingerprint_sha256": sys.argv[7]' in launcher


def test_forced_camera_refresh_advances_the_sensor_clock():
    source = MANIP_LOCO.read_text(encoding="utf-8")
    refresh = "update(float(self.dt), force_recompute=True)"
    assert source.count(refresh) == 3
    assert "def render_stereo_rgb_camera_rigs_batch" in source
    assert (
        "update(0.0, force_recompute=True)"
        not in source[
            source.index("def _camera_rgb_to_numpy") : source.index(
                "def render_stereo_rgb_camera_rigs"
            )
            + 2500
        ]
    )
