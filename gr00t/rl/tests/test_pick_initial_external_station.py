"""Ordinary station initialization feeds the same standard GPU base/gaze execution."""

from types import SimpleNamespace

import torch

from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory
from gr00t.rl.pick import tensor_planning
from gr00t.rl.pick.controller import PickController
from gr00t.rl.pick.measured_commands import read_measured_report
from gr00t.rl.pick.measured_station import COMMAND_CATALOG
from gr00t.rl.pick.tensor_fsm import TensorPickFSM
from gr00t.rl.pick.tensor_planning import TensorPickPlanningCoordinator
from gr00t.rl.tests.test_pick_tensor_fsm import batch
from gr00t.rl.tests.test_pick_workflow import fake_controller


def test_deferred_stations_install_common_base_paths_and_preserve_optional_reference_masks(
    monkeypatch,
):
    env, controls, state = batch(monkeypatch, count=5, defer_initial_station=True)
    assert all(c.arm_path.length.tolist() == [1] for c in controls)
    assert all(not c.base_path.length.any() for c in controls)
    for row in (0, 1, 3):
        controls[row].object_support_points = torch.zeros(8, 3)
    # Rejected or already-configured rows must not request station planning.
    del controls[3].deferred_initial_station
    del controls[4].deferred_initial_station
    fsm = TensorPickFSM(env, controls)
    fsm.fail(torch.tensor([3]), 10)
    old_base = fsm.base_path.poses[3:].clone()
    old_lengths = fsm.base_path.length[3:].clone()
    planner = TensorPickPlanningCoordinator(controls)
    planner.carry_geometry = SimpleNamespace(select=lambda rows: rows)
    requests, gaze_requests = [], []

    def plan(local, destination, sizes, hulls, limits, collision, geometry, **kwargs):
        requests.append(geometry.clone())
        assert hulls.shape[:2] == kwargs["support_mask"].shape
        valid = geometry != 1
        pose = destination + 0.1
        pose = torch.where(valid[:, None], pose, destination)
        q = local["ee_orn"].clone()
        return (
            pose,
            q,
            local["dof_pos"].clone(),
            valid,
            {
                "planning_failure_code": torch.where(valid, 0, 3),
                "selection_policy_code": torch.where(valid, 3, 5),
            },
        )

    def gaze(roots, joints, target, limits, dt, collision, geometry, offsets, **kwargs):
        gaze_requests.append((geometry.clone(), kwargs["next_grasp_mask"]))
        return (
            roots[..., :3],
            roots[..., 3:7],
            {
                "valid": torch.ones(len(roots), dtype=torch.bool),
                "samples": kwargs["lengths"],
                "candidate_q2_offset_deg": torch.zeros(1),
                "candidate_optical_roll_offset_deg": torch.zeros(1),
            },
        )

    monkeypatch.setattr(tensor_planning, "select_initial_pick_station_batch", plan)
    monkeypatch.setattr(tensor_planning, "plan_approach_gaze_batch", gaze)
    planner.install_initial_external_stations(fsm, state)
    assert len(requests) == 1 and requests[0].tolist() == [0, 1]
    assert planner.grasp_preferred_mask.tolist() == [True, False, False, False, False]
    assert fsm.failure_code.tolist() == [0, 0, 0, 10, 0]
    for row in range(3):
        c = controls[row]
        assert c.approach_root_path.shape[1] == int(fsm.base_path.length[:3].max())
        assert c.approach_path_length == fsm.base_path.length[row]
        reference = BasePoseTrajectory(1, "cpu")
        reference.install(
            torch.tensor([0]),
            torch.cat(
                (state["root_states"][row : row + 1, :2], torch.tensor([[c.episode["base_yaw"]]])),
                -1,
            ),
            torch.cat((c.hold_xy, c.hold_yaw[:, None]), -1)[:, None],
            env.dt,
        )
        for name in ("poses", "velocities", "length"):
            torch.testing.assert_close(
                getattr(fsm.base_path, name)[row : row + 1], getattr(reference, name)
            )
    torch.testing.assert_close(fsm.base_path.poses[3:], old_base)
    torch.testing.assert_close(fsm.base_path.length[3:], old_lengths)
    planner.install_initial_external_gaze(fsm)
    assert [rows.tolist() for rows, _ in gaze_requests] == [[0, 1, 2]]
    assert gaze_requests[0][1].tolist() == [True, False, False]
    assert all(not hasattr(c, "deferred_initial_station") for c in controls)
    assert all(not hasattr(c, "approach_root_path") for c in controls)
    assert fsm.failure_code.tolist() == [0, 0, 0, 10, 0]
    planner.finish_diagnostics()
    assert (
        controls[0].planning_diagnostics["initial_station"]["selection_policy"]
        == "standing_arrival"
    )
    assert (
        controls[1].planning_diagnostics["initial_station"]["failure"]
        == "no_safe_nominal_station_candidate"
    )
    assert "initial_station" not in controls[2].planning_diagnostics


def test_no_deferred_stations_never_changes_existing_base_or_loads_geometry(monkeypatch):
    planner = TensorPickPlanningCoordinator([SimpleNamespace(geometry=None)])

    def forbidden(*args, **kwargs):
        raise AssertionError("No station planning was requested")

    monkeypatch.setattr(tensor_planning.BatchedSceneGeometry, "from_views", forbidden)
    monkeypatch.setattr(tensor_planning, "select_initial_pick_station_batch", forbidden)
    planner.install_initial_external_stations(SimpleNamespace(), {})


def test_floor_with_missing_policy_catalog_uses_the_same_deferred_ordinary_station(monkeypatch):
    original, state = fake_controller(monkeypatch, initialize=False)
    episode = dict(original.episode, support_type="floor")

    def forbidden(*args, **kwargs):
        raise AssertionError("Incompatible catalog must not run a serial fallback planner")

    assert not hasattr(PickController, "prepare_approach")
    control = PickController(
        original.env,
        episode,
        original.geometry,
        collision=original.collision,
        initialize=True,
    )
    assert not control.measured_reach.compatible
    assert control.deferred_initial_station
    assert control.arm_path.length.tolist() == [1]
    torch.testing.assert_close(control.initial_joints, state["dof_pos"])


def test_compatible_catalog_never_invokes_scene_scoped_station_planning_at_construction(
    monkeypatch,
):
    original, state = fake_controller(monkeypatch, initialize=False)
    metadata, _ = read_measured_report(COMMAND_CATALOG)
    original.env.pick_policy_checkpoint_sha256 = metadata["checkpoint_sha256"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Compatible measured stations must be planned in the common batch")

    assert not hasattr(PickController, "prepare_approach")
    control = PickController(
        original.env,
        dict(original.episode, support_type="floor"),
        original.geometry,
        collision=original.collision,
        object_support_points=torch.zeros(8, 3),
        initialize=True,
    )
    assert control.measured_reach.compatible and control.deferred_initial_station
    assert control.arm_path.length.tolist() == [1]
    assert control.hold_yaw.dtype == state["root_states"].dtype


def test_measured_success_and_gpu_selected_fallback_share_station_installation(monkeypatch):
    env, controls, state = batch(monkeypatch, count=4, defer_initial_station=True)
    for row in (0, 1, 2):
        controls[row].object_support_points = torch.zeros(8, 3)
    for row in (0, 1):
        controls[row].measured_reach = SimpleNamespace(
            compatible=True,
            catalog="catalog.json",
            catalog_sha256="a" * 64,
            orientation=None,
            depth=0.0,
        )
        controls[row].episode["support_z"] = 0.0
    fsm = TensorPickFSM(env, controls)
    planner = TensorPickPlanningCoordinator(controls)
    planner.carry_geometry = SimpleNamespace(select=lambda rows: rows)
    catalogs, ordinary = [], []

    def load(paths, reference):
        catalogs.append(paths)
        return object()

    def measured(local, destination, hulls, hull_mask, floor, commands, collision, geometry):
        assert geometry.tolist() == [0, 1]
        valid = geometry == 0
        pose = torch.where(valid[:, None], destination + 0.3, destination)
        orientation = local["ee_orn"].clone()
        return (
            pose,
            orientation,
            torch.full((2,), 0.375),
            valid,
            {
                "selection_policy": "shortest_scene_clear_measured_low_command",
                "selected_station": pose,
                "selected_orientation": orientation,
            },
        )

    def default(local, destination, sizes, hulls, limits, collision, geometry, **kwargs):
        ordinary.append(geometry.clone())
        valid = torch.ones(len(geometry), dtype=torch.bool)
        return (
            destination + 0.1,
            local["ee_orn"],
            local["dof_pos"],
            valid,
            {
                "planning_failure_code": torch.zeros(len(geometry), dtype=torch.long),
                "selection_policy_code": torch.full((len(geometry),), 3),
            },
        )

    monkeypatch.setattr(tensor_planning, "load_measured_commands", load)
    monkeypatch.setattr(tensor_planning, "select_measured_station_batch", measured)
    monkeypatch.setattr(tensor_planning, "select_initial_pick_station_batch", default)
    planner.install_initial_external_stations(fsm, state)
    assert catalogs == [["catalog.json"]]
    assert len(ordinary) == 1 and ordinary[0].tolist() == [1, 2]
    assert planner.measured_grasp_mask.tolist() == [True, False, False, False]
    assert planner.grasp_preferred_mask.tolist() == [True, True, True, False]
    assert planner.initial_joint_reference_mask.tolist() == [False, True, True, False]
    torch.testing.assert_close(fsm.hold_xy[:, 0], torch.tensor([0.4, 0.2, 0.2, 0.1]))
    assert not fsm.failure_code.any()
    # Rebuilding metadata cannot turn an unsuccessful measured proposal into a measured grasp.
    planner._cache_grasp_metadata(fsm)
    assert planner.measured_grasp_mask.tolist() == [True, False, False, False]
    assert planner.grasp_preferred_mask.tolist() == [True, True, True, False]
    planner.finish_diagnostics()
    assert controls[0].planning_diagnostics["initial_station"]["selection_policy"] == (
        "shortest_scene_clear_measured_low_command"
    )
    assert (
        controls[1].planning_diagnostics["measured_station"]["failure"]
        == "no_scene_safe_measured_match"
    )
    assert (
        controls[1].planning_diagnostics["initial_station"]["measured_reach_fallback"]
        == "no_scene_safe_measured_match"
    )
