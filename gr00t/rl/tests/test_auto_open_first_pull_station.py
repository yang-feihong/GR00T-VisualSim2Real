import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.first_pull_station import station_candidates, station_clearances
from gr00t.rl.auto_open.pull_station import PullStationConfig
from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM, PullTensorHistory
from gr00t.rl.auto_open.tensor_planning import TensorPlanningCoordinator
from gr00t.rl.scripts.doorman_auto_open import Stage
from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory


def test_initial_station_has_backward_waypoint_before_lateral_target():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage, fsm.device, fsm.pull_door = Stage, "cpu", True
    fsm.stage = torch.full((2,), int(Stage.STOW_ARM_AND_CLOSE_GRIPPER))
    fsm.stage_steps = torch.zeros(2, dtype=torch.int64)
    fsm.pull_base_path = torch.zeros(2, 2, 3)
    fsm.pull_base_path_length = torch.ones(2, dtype=torch.int64)
    fsm.pull_base_path_index = torch.zeros(2, dtype=torch.int64)
    fsm.pull_base_entry_pose = torch.zeros(2, 3)
    fsm.pull_base_trajectory = BasePoseTrajectory(2, "cpu")
    fsm.dt = 0.02
    fsm.pull_station_xy = torch.zeros(2, 2)
    fsm.pull_station_yaw = torch.zeros(2)
    fsm.pull_initial_station_xy = torch.zeros(2, 2)
    fsm.pull_initial_station_yaw = torch.zeros(2)
    fsm.pull_gap_xy = torch.zeros(2, 2)
    fsm.pull_planned_frame_clearance = torch.zeros(2)
    fsm.pull_planned_wall_clearance = torch.zeros(2)
    fsm.pull_frame_clear_steps = torch.zeros(2, dtype=torch.int64)
    fsm.align_panel_arm_joint_target = torch.zeros(2, 6)
    fsm.env = SimpleNamespace(arm_joint_ids=torch.arange(6))
    fsm.reset_base_arrival = lambda rows, profile: None
    fsm.transition = lambda rows, stage: fsm.stage.__setitem__(rows, int(stage))
    root = torch.zeros(2, 13)
    root[:, :2] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    root[0, 3] = 1.0
    root[1, 3:7] = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    rows = torch.arange(2)
    fsm.begin_move_to_gap_station({"root_states": root, "dof_pos": torch.zeros(2, 6)}, rows)
    target = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    fsm.install_initial_pull_station(
        rows, target, torch.zeros(2), target, torch.ones(2), torch.ones(2)
    )
    torch.testing.assert_close(fsm.pull_base_path[:, 0, :2], torch.tensor([[0.8, 2.0], [3.0, 3.8]]))
    torch.testing.assert_close(fsm.pull_base_path[:, 0, 2], torch.tensor([0.0, math.pi / 2]))
    torch.testing.assert_close(fsm.pull_base_path[:, 1, :2], target)
    assert fsm.pull_base_path_length.tolist() == [2, 2]
    torch.testing.assert_close(fsm.pull_station_xy, target)
    assert fsm.pull_base_trajectory.length.min() > 2
    torch.testing.assert_close(
        fsm.pull_base_trajectory.poses[rows, fsm.pull_base_trajectory.length - 1, :2], target
    )


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_local_station_coordinates_include_wall_and_frame_origin(side):
    config = PullStationConfig()
    hinge = torch.tensor([[3.0, -2.0, 0.0]])
    normal, tangent = torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, side]])
    xy, yaw, local = station_candidates(
        hinge,
        normal,
        tangent,
        torch.tensor([1.2]),
        torch.tensor([0.01]),
        torch.full((1, 2, 3), 0.5),
        config,
    )
    torch.testing.assert_close(local[0, 0], torch.tensor([0.545, 0.640, -129.0]))
    torch.testing.assert_close(xy[0, 0], torch.tensor([3.555, -2.0 + side * 1.840]))
    forward = torch.stack((yaw.cos(), yaw.sin()), -1)
    torch.testing.assert_close(
        forward[0, 0],
        torch.tensor([math.cos(math.radians(-129)), side * math.sin(math.radians(-129))]),
    )


def test_initial_station_random_stream_is_batch_independent_and_does_not_consume_pull_cycle():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage = Stage
    fsm.pull_station_config = PullStationConfig()
    fsm.pull_cycle_seeds = torch.tensor([1700, 1701, 1702, 1703])
    fsm.stage_visit = torch.ones(4, max(int(s) for s in Stage) + 1, dtype=torch.int64)
    fsm.pull_cycle = torch.zeros(4, dtype=torch.int64)
    full = fsm.sample_initial_pull_stations(torch.arange(4))
    torch.testing.assert_close(
        full[[3, 1]], fsm.sample_initial_pull_stations(torch.tensor([3, 1])), atol=0, rtol=0
    )
    assert not torch.equal(full[0], full[1])
    assert not fsm.pull_cycle.any()


def test_clearance_uses_limbs_not_only_trunk():
    root = torch.tensor([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]])
    bodies = root[:, None].expand(-1, 2, -1).clone()
    bodies[:, 1, 0] = -0.3
    wall, frame = station_clearances(
        root,
        bodies,
        torch.zeros(2, 1, 3),
        torch.arange(2),
        torch.tensor([[[0.35, 0.8]]]),
        torch.zeros(1, 1),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.zeros(1, 3),
        torch.zeros(1),
        torch.zeros(1),
    )
    torch.testing.assert_close(wall, torch.tensor([[0.05]]))
    torch.testing.assert_close(frame, torch.tensor([[0.8]]))


@pytest.mark.parametrize("accepted", [True, False])
@pytest.mark.parametrize("use_forecast", [True, False])
def test_coordinator_selects_only_path_valid_station_and_records_all_candidates(
    monkeypatch, accepted, use_forecast
):
    from gr00t.rl.auto_open import tensor_planning

    config = PullStationConfig(first_candidate_count=3)
    installs, failures, conditions = [], [], []
    fsm = SimpleNamespace(
        device="cpu",
        pull_station_config=config,
        env=SimpleNamespace(
            get_goal_reference_origin=lambda: torch.zeros(1, 3),
            get_goal_reference_quat=lambda: torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            get_goal_center_offset_local=lambda: torch.zeros(1, 3),
        ),
        pull_first_station_attempts=torch.zeros(1, dtype=torch.int64),
        hinge_world=torch.zeros(1, 3),
        door_normal=torch.tensor([[1.0, 0.0, 0.0]]),
        pull_free_direction=torch.tensor([[0.0, 1.0]]),
        pull_frame_edge=torch.ones(1),
        pull_wall_front=torch.zeros(1),
        sample_initial_pull_stations=lambda rows: torch.full((1, 3, 3), 0.5),
        pull_motion_points=None,
        pull_motion_body_indices=None,
        pull_cycle_pose_grid=torch.zeros(27, 2),
        pull_hardware_z=torch.zeros(1, 2),
        pull_panel_config=SimpleNamespace(position_tolerance_m=0.004),
        hinge_axis_sign=torch.ones(1),
        hinge_idx=0,
        pull_free_edge=torch.ones(1),
        install_initial_pull_station=lambda rows, *args: installs.append(rows.clone()),
        pull_initial_station_history=PullTensorHistory(1),
        record_acceptance=lambda *args: conditions.append(args),
        fail=lambda rows, code: failures.append((rows.clone(), code)),
    )
    monkeypatch.setattr(
        tensor_planning, "station_clearances", lambda *args: (torch.ones(1, 3), torch.ones(1, 3))
    )
    inverse_safe = torch.zeros(1, 6 if use_forecast else 3, 27, dtype=torch.bool)
    inverse_safe[0, 1, 5] = accepted
    monkeypatch.setattr(
        tensor_planning,
        "first_insertion_residuals",
        lambda *args: (
            inverse_safe,
            torch.full_like(inverse_safe, torch.inf, dtype=torch.float32),
            torch.full_like(inverse_safe, torch.inf, dtype=torch.float32),
        ),
    )
    planner = TensorPlanningCoordinator.__new__(TensorPlanningCoordinator)
    planner.Stage, planner.pull_geometry, planner.arm_joint_ids, planner.joint_limits = (
        Stage,
        None,
        torch.arange(6),
        None,
    )
    planner.stats = dict(batches=0, requests=0, errors=0, labels={})
    planner.plan_initial_pull_station(
        fsm,
        dict(
            root_states=torch.zeros(1, 13),
            robot_body_state_w=torch.zeros(1, 1, 13),
            door_joint_position=torch.zeros(1, 1),
            dof_pos=torch.zeros(1, 6),
        ),
        torch.tensor([0]),
        **(
            dict(arrival_xy_error=torch.zeros(1, 2), arrival_yaw_error=torch.zeros(1))
            if use_forecast
            else {}
        ),
    )
    record = fsm.pull_initial_station_history[0][0]
    assert record["selected_attempt"] == (1 if accepted else -1)
    assert len(record["path_max_position_error_m"]) == 3
    assert len(record["forecast_path_max_position_error_m"]) == (3 if use_forecast else 0)
    assert installs[0].tolist() == ([0] if accepted else [])
    assert failures[0][0].tolist() == ([] if accepted else [0])
    assert failures[0][1] == 55


def test_failure_reason_codes_are_unique():
    tree = ast.parse((Path(__file__).parents[1] / "auto_open/runtime.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            codes = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and type(key.value) is int
            ]
            assert len(codes) == len(set(codes))


def test_retry_reuses_station_entry_preserves_other_cycles_and_passes_measured_error():
    root = torch.zeros(2, 13)
    root[:, 3] = 1.0
    root[0, :2] = torch.tensor([0.42, 0.73])
    entries, plans = [], []
    fsm = SimpleNamespace(
        pull_cycle=torch.tensor([1, 3]),
        pull_station_xy=torch.tensor([[0.40, 0.79], [0.0, 0.0]]),
        pull_station_yaw=torch.zeros(2),
        begin_move_to_gap_station=lambda state, rows: entries.append(rows.tolist()),
    )
    planner = TensorPlanningCoordinator.__new__(TensorPlanningCoordinator)
    planner.plan_initial_pull_station = lambda fsm, state, rows, **kwargs: plans.append(kwargs)
    planner.retry_first_pull_station(fsm, dict(root_states=root), torch.tensor([0]))
    assert entries == [[0]]
    assert fsm.pull_cycle.tolist() == [0, 3]
    torch.testing.assert_close(plans[0]["arrival_xy_error"], torch.tensor([[0.02, -0.06]]))
    torch.testing.assert_close(plans[0]["arrival_yaw_error"], torch.zeros(1))
