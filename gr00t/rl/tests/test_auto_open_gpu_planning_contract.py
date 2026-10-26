"""Guard the online GPU planning boundary against host/row dispatch regressions."""

import ast
import inspect
import json
import textwrap
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.tensor_planning import TensorPlanningCoordinator
from gr00t.rl.auto_open.tensor_stage_planning import GPUStagePlanningMixin
from gr00t.rl.auto_open.pull_entry_gpu import PullEntryGPU
from gr00t.rl.auto_open.tensor_fsm import PullTensorHistory


ONLINE_METHODS = [
    TensorPlanningCoordinator._inverse_pose_bank,
    TensorPlanningCoordinator.plan_pose_path,
    TensorPlanningCoordinator.plan_pull_insert,
    TensorPlanningCoordinator.plan_pull_open,
    TensorPlanningCoordinator.plan_initial_pull_station,
    TensorPlanningCoordinator.retry_first_pull_station,
    PullEntryGPU.plan,
] + [
    getattr(GPUStagePlanningMixin, name)
    for name in dir(GPUStagePlanningMixin)
    if name.startswith("plan_") or name in ("_stage_path", "_sample_gpu_retract")
]


@pytest.mark.parametrize("method", ONLINE_METHODS, ids=lambda method: method.__name__)
def test_online_planning_never_reads_device_values_on_host(method):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"cpu", "numpy", "tolist", "item"}, (
                method.__name__,
                node.func.attr,
            )
        if isinstance(node, ast.Name):
            assert node.id not in {"np", "scipy"}
        if isinstance(node, ast.Attribute):
            assert node.attr != "controllers", method.__name__


def test_unobserved_candidate_diagnostics_export_as_json_null():
    history = PullTensorHistory(1)
    history.append_batch(
        torch.tensor([0]),
        {
            "errors": torch.tensor([[float("inf"), float("nan"), 0.002]]),
            "selected_attempt": torch.tensor([2]),
        },
    )
    record = history[0][0]
    assert record == {"errors": [None, None, pytest.approx(0.002)], "selected_attempt": 2}
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("rejection", [None, "inverse", "geometry", "radius"])
@pytest.mark.parametrize("phase", ["later", "first", "exhausted"])
def test_pull_candidates_share_one_inverse_and_rejections_never_install(
    monkeypatch, rejection, phase
):
    from gr00t.rl.auto_open import tensor_planning, pose_planning
    from gr00t.rl.auto_open.panel_motion import PullPanelConfig
    from gr00t.rl.auto_open.pull_station import PullStationConfig
    from gr00t.rl.scripts.doorman_auto_open import Stage

    calls, installs, records, failures = [], [], [], []
    end = torch.zeros(2, 3, 3)
    end[..., 2] = 1.0
    if rejection == "radius":
        end[..., 0] = 0.61
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 3, 4).clone()
    geometry = dict(
        positions=end[:, :, None].expand(-1, -1, 2, -1).clone(),
        orientations=quat[:, :, None].expand(-1, -1, 2, -1).clone(),
        lengths=torch.full((2, 3), 2),
        feasible=torch.full((2, 3), rejection != "geometry"),
        endpoint=end,
        orientation=quat,
        face_offset_local=torch.zeros_like(end),
        contact_point=end + torch.tensor([0.002, 0.0, 0.0]),
        normal=torch.tensor([1.0, 0.0, 0.0]).expand_as(end),
        tip_radius_m=torch.ones(2, 3),
        ee_tangent_radius_m=torch.ones(2, 3),
        gripper_height_bounds_m=torch.zeros(2, 3, 2),
    )

    def inverse(position, orientation, **kwargs):
        calls.append(len(position))
        error = torch.zeros_like(position)
        bad = (
            torch.ones(len(position), dtype=torch.bool)
            if rejection == "inverse"
            else torch.arange(len(position)) % 6 < 2
        )
        error[bad, 0] = 0.02
        return SimpleNamespace(
            command_position_world=position + error,
            command_orientation_world_wxyz=orientation,
            position_error=error,
            orientation_error=torch.zeros_like(error),
            arm_joint_positions=torch.zeros(len(position), 6),
        )

    monkeypatch.setattr(pose_planning, "solve_pose_targets", inverse)
    planner = TensorPlanningCoordinator.__new__(TensorPlanningCoordinator)
    planner.Stage, planner.arm_joint_ids = Stage, torch.arange(6)
    planner.joint_limits = torch.tensor([[-3.0, 3.0]] * 6)
    planner.pull_geometry = SimpleNamespace(plan=lambda *args, **kwargs: geometry)
    planner.pull_geometry.hinge_world = torch.zeros(2, 3)
    planner.pull_geometry.door_normal = torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1)
    planner.pull_geometry.tangent = torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1)
    planner._first_pull_wall = SimpleNamespace(
        clearance=lambda roots, *args: torch.ones(len(roots))
    )
    monkeypatch.setattr(tensor_planning, "plan_first_pull_entry", lambda *args, **kwargs: geometry)
    retries = []
    planner.retry_first_pull_station = lambda fsm, state, rows: retries.append(rows.tolist())
    planner.stats = dict(batches=0, requests=0, errors=0, labels={})
    fsm = SimpleNamespace(
        device="cpu",
        pull_cycle=torch.full((2,), int(phase == "later"), dtype=torch.int64),
        pull_first_station_attempts=torch.full((2,), 4 if phase == "exhausted" else 1),
        pull_station_config=PullStationConfig(),
        pull_panel_config=PullPanelConfig(height_offsets_m=(0.0,)),
        pull_frame_edge=torch.ones(2),
        pull_wall_front=torch.zeros(2),
        pull_hardware_z=torch.zeros(2, 2),
        pull_cycle_samples=torch.zeros(2, 4),
        q_closed=torch.zeros(2),
        sample_pull_cycle_candidates=lambda rows: torch.zeros(len(rows), 3, 4),
        record_acceptance=lambda *args: None,
        record_pull_cycle=lambda rows, record: records.append(record),
        fail=lambda rows, code: failures.append((rows.tolist(), code)),
        install_pull_insert_tensor=lambda rows, *args, **kwargs: installs.append(rows.tolist()),
    )
    root = torch.zeros(2, 13)
    root[:, 3] = 1.0
    fsm.env = SimpleNamespace(
        get_goal_reference_origin=lambda: root[:, :3],
        get_goal_reference_quat=lambda: root[:, 3:7],
        get_goal_center_offset_local=lambda: torch.zeros(2, 3),
    )
    planner.plan_pull_insert(
        fsm,
        dict(
            root_states=root,
            dof_pos=torch.zeros(2, 6),
            ee_pos=torch.zeros(2, 3),
            ee_orn=root[:, 3:7],
            robot_body_state_w=torch.zeros(2, 3, 13),
        ),
        torch.arange(2),
    )
    assert calls == ([] if rejection in ("geometry", "radius") else [12])
    assert installs == ([[]] if rejection else [[0, 1]])
    should_retry = phase == "first" and rejection is not None
    assert retries == ([[0, 1]] if should_retry else [])
    assert [item for item in failures if item[0]] == (
        []
        if rejection is None or should_retry
        else [([0, 1], 50 if rejection in ("geometry", "radius") else 49)]
    )
    assert records[0]["selected_attempt"].tolist() == ([-1, -1] if rejection else [1, 1])


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_routing_minkowski_boundary_agrees_with_convex_hull_reference(dtype):
    import numpy as np
    from scipy.spatial import ConvexHull
    from gr00t.rl.auto_open.pull_entry_gpu import _minkowski_polygon, _polygon_halfspaces, _free

    angles = torch.arange(257, dtype=torch.float64) * (2.0 * torch.pi / 257)
    polygon = torch.stack((angles.cos() * 0.07, angles.sin() * 0.035), -1)[None].to(dtype)
    rectangle = torch.tensor([[[0.1, 0.8], [0.14, 0.8], [0.14, 1.8], [0.1, 1.8]]], dtype=dtype)
    boundary = _minkowski_polygon(polygon, rectangle)
    obstacles = _polygon_halfspaces(boundary)
    combinations = (
        (rectangle.double()[:, :, None] - polygon.double()[:, None]).flatten(1, 2)[0].numpy()
    )
    oracle = ConvexHull(combinations)
    vertices = torch.from_numpy(combinations[oracle.vertices])[None]
    assert _free(vertices, [obstacles]).all()
    rng = np.random.default_rng(42)
    queries = rng.uniform([-0.1, 0.6], [0.4, 2.0], (2000, 2))
    expected = ~((queries @ oracle.equations[:, :2].T + oracle.equations[:, 2]) < -1.0e-8).all(1)
    assert np.array_equal(_free(torch.from_numpy(queries)[None], [obstacles])[0].numpy(), expected)


@pytest.mark.parametrize("capacity, expected", [(16, False), (128, True)])
def test_visibility_graph_does_not_confuse_waypoint_capacity_with_connectivity(
    monkeypatch, capacity, expected
):
    from gr00t.rl.auto_open import pull_entry_gpu as gpu

    x = torch.cat((torch.tensor([0.0, 64.0]), torch.arange(1.0, 64.0)))
    points = torch.stack((x, torch.zeros_like(x)), -1)[None]
    available = torch.ones(1, len(x), dtype=torch.bool)
    obstacles = gpu._Obstacles(
        torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1), torch.ones(1, 1, 1, dtype=torch.bool)
    )
    monkeypatch.setattr(
        gpu, "_visible", lambda a, b, obstacles: (a[..., 0] - b[..., 0]).abs() <= 1.0
    )
    route, length, safe = gpu._solve_visibility(points, available, obstacles, capacity, 1024)
    assert safe.item() is expected
    if expected:
        assert length.item() == 65
        torch.testing.assert_close(route[0, :65, 0], torch.arange(65.0))


def test_entry_route_capacity_matches_physical_path_bank_capacity():
    from gr00t.rl.tests.test_auto_open_pull_entry_gpu import fixture

    cache, _, _ = fixture(max_samples=1024)
    assert cache.max_route_nodes == cache.max_samples == 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_pose_path_rejection_reduction_runs_on_cuda(monkeypatch):
    from gr00t.rl.auto_open import tensor_planning, pose_planning
    from gr00t.rl.auto_open.motion_targets import PaddedPathBank
    from gr00t.rl.scripts.doorman_auto_open import Stage

    device = "cuda"
    planner = TensorPlanningCoordinator.__new__(TensorPlanningCoordinator)
    planner.Stage = Stage
    planner.arm_joint_ids = torch.arange(6, device=device)
    planner.joint_limits = torch.tensor([[-3.0, 3.0]] * 6, device=device)
    planner.path_diagnostics = []
    planner.stats = dict(batches=0, requests=0, errors=0, labels={})
    bank = PaddedPathBank(2, 4, device)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(2, 3, -1)
    bank.install_tensor(
        torch.arange(2, device=device),
        torch.zeros(2, 3, 3, device=device),
        quat,
        torch.full((2,), 3, device=device),
    )
    failed = []
    fsm = SimpleNamespace(
        device=device,
        pull_open_path=bank,
        stage=torch.full((2,), int(Stage.PRY_DOOR_OPEN), device=device),
        fail=lambda rows, code: failed.append(rows.clone()),
        record_acceptance=lambda *args: None,
    )

    def inverse(position, orientation, **kwargs):
        error = torch.zeros_like(position)
        error[-1, 0] = 0.02
        return SimpleNamespace(
            command_position_world=position,
            command_orientation_world_wxyz=orientation,
            position_error=error,
            orientation_error=torch.zeros_like(error),
            arm_joint_positions=torch.zeros(len(position), 6, device=device),
        )

    monkeypatch.setattr(pose_planning, "solve_pose_targets", inverse)
    root = torch.zeros(2, 13, device=device)
    root[:, 3] = 1.0
    planner.plan_pose_path(
        fsm,
        dict(root_states=root, dof_pos=torch.zeros(2, 6, device=device)),
        torch.arange(2, device=device),
        bank_name="pull_open_path",
        prediction_tolerance=0.004,
    )
    assert torch.equal(failed[0], torch.tensor([1], device=device))
