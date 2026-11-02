"""Formal initialization pools gaze requests and installs standard task paths."""

from types import SimpleNamespace

import torch

from gr00t.rl.pick import tensor_planning
from gr00t.rl.pick.controller import PickStage
from gr00t.rl.pick.gaze import plan_approach_gaze
from gr00t.rl.pick.tensor_fsm import TensorPickFSM
from gr00t.rl.pick.tensor_planning import TensorPickPlanningCoordinator
from gr00t.rl.tests.test_pick_tensor_fsm import batch


def test_deferred_gaze_pools_optional_wrist_routes_and_preserves_rejected_rows(monkeypatch):
    env, controllers, state = batch(monkeypatch, count=4, defer_approach_gaze=True)
    assert all(c.arm_path.length.tolist() == [1] for c in controllers)
    assert all("approach_gaze" not in c.planning_diagnostics for c in controllers)
    for row, length in enumerate((3, 5, 2, 1)):
        roots = state["root_states"][row : row + 1, None].expand(-1, length, -1).clone()
        roots[..., 0] = row
        controllers[row].approach_root_path = roots
        controllers[row].approach_path_length = torch.tensor([length])
        controllers[row].initial_grasp_joint_reference = (
            torch.full((1, 6), float(row)) if row in (0, 2) else None
        )
    del controllers[3].approach_root_path
    del controllers[3].approach_random_offsets
    fsm = TensorPickFSM(env, controllers)
    fsm.fail(torch.tensor([3]), 10)
    base = fsm.base_path.poses.clone()
    planner = TensorPickPlanningCoordinator(controllers)
    planner.carry_geometry = SimpleNamespace(select=lambda rows: rows)
    requests = []

    def plan(roots, initial, target, limits, dt, collision, geometry, offsets, **kwargs):
        requests.append((geometry.clone(), kwargs["lengths"].clone()))
        lengths = kwargs["lengths"]
        for row in range(len(roots)):
            # Last-pose padding retains the actual scene's frame, including quaternion.
            torch.testing.assert_close(roots[row, -1], roots[row, lengths[row] - 1])
        reference = kwargs["next_grasp_joints"]
        if reference is not None:
            mask = kwargs["next_grasp_mask"]
            torch.testing.assert_close(reference[mask, 0], geometry[mask].float())
        valid = geometry != 2
        return (
            roots[..., :3] + 0.1,
            roots[..., 3:7],
            {
                "valid": valid,
                "samples": lengths,
                "candidate_q2_offset_deg": roots.new_tensor([0, 10]),
                "candidate_optical_roll_offset_deg": roots.new_tensor([0, 90]),
                "candidate_valid": valid[:, None].expand(-1, 2),
            },
        )

    monkeypatch.setattr(tensor_planning, "plan_approach_gaze_batch", plan)

    def forbidden(*args, **kwargs):
        raise AssertionError("Initializing pooled paths must not serialize GPU state")

    with monkeypatch.context() as scope:
        for name in ("item", "tolist", "cpu"):
            scope.setattr(torch.Tensor, name, forbidden)
        planner.install_initial_external_gaze(fsm)
    assert [rows.tolist() for rows, _ in requests] == [[0, 1, 2]]
    assert [length.tolist() for _, length in requests] == [[3, 5, 2]]
    assert fsm.failure_code.tolist() == [0, 0, 7, 10]
    assert fsm.arm_path.length.tolist() == [3, 5, 2, 1]
    torch.testing.assert_close(fsm.arm_path.positions[1, 4], torch.tensor([1.1, 0.1, 0.636]))
    torch.testing.assert_close(fsm.base_path.poses, base)
    assert not fsm.grasp_retries.any() and not fsm.stage_steps.any()
    assert all(not hasattr(c, "approach_root_path") for c in controllers)
    planner.finish_diagnostics()
    assert "failure" not in controllers[0].planning_diagnostics["approach_gaze"]
    assert controllers[2].planning_diagnostics["approach_gaze"]["failure"] == (
        "approach_gaze_no_collision_free_aimed_trajectory"
    )
    assert set(controllers[1].planning_diagnostics["approach_gaze"]["random_offsets_rad"]) == {
        "base_heading",
        "optical_yaw",
        "optical_pitch",
        "optical_roll",
    }
    assert "approach_gaze" not in controllers[3].planning_diagnostics


def test_shared_initial_gaze_matches_single_scene_planner_and_no_retiming(monkeypatch):
    env, controllers, state = batch(monkeypatch, count=2, defer_approach_gaze=True)
    # Different physical paths, but padding must not change either scene's timing.
    controllers[1].approach_root_path = controllers[1].approach_root_path[:, :2]
    controllers[1].approach_path_length = torch.tensor([2])
    expected = [
        plan_approach_gaze(
            c.approach_root_path,
            c.initial_joints,
            c.target,
            c.limits,
            env.dt,
            c.collision,
            None,
            c.gaze_offsets,
        )
        for c in controllers
    ]
    fsm = TensorPickFSM(env, controllers)
    planner = TensorPickPlanningCoordinator(controllers)
    planner.carry_geometry = SimpleNamespace(select=lambda rows: None)
    planner.install_initial_external_gaze(fsm)
    for row, (p, q, diagnostics) in enumerate(expected):
        count = len(p[0])
        assert fsm.arm_path.length[row] == count
        assert fsm.approach_gaze_valid[row] == diagnostics["valid"]
        torch.testing.assert_close(fsm.arm_path.positions[row, :count], p[0])
        torch.testing.assert_close(fsm.arm_path.orientations[row, :count], q[0])
    assert fsm.stage[0] in (PickStage.APPROACH_OBJECT, PickStage.FAILURE)


def test_rejected_or_basket_initialization_does_not_build_external_gaze_geometry(monkeypatch):
    planner = TensorPickPlanningCoordinator([SimpleNamespace(geometry=None)])

    def forbidden(*args, **kwargs):
        raise AssertionError("No external initialization was requested")

    monkeypatch.setattr(tensor_planning.BatchedSceneGeometry, "from_views", forbidden)
    monkeypatch.setattr(tensor_planning, "plan_approach_gaze_batch", forbidden)
    planner.install_initial_external_gaze(SimpleNamespace())
