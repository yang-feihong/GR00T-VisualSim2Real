import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.auto_open.panel_motion import (
    PullPanelConfig,
    hinge_speed_target,
    outside_comfort_workspace,
    rotate_base_station,
    base_move_clearance,
    base_station_waypoints,
    follow_station_rotation,
    base_station_candidate_paths,
    fingertip_arc_position,
    panel_fingertip_reference,
    PUSH_TIP_MAX_ANGULAR_LEAD_RAD,
)
from gr00t.rl.auto_open.tensor_fsm import (
    TensorDoorFSM,
    PullTensorHistory,
    _pull_station_history_record,
)
from gr00t.rl.auto_open.motion_targets import PaddedPathBank
from gr00t.rl.scripts.doorman_auto_open import Stage


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("yaw_deg", [0.0, 10.0, 30.0])
def test_push_four_degree_lead_is_on_tip_not_ee(sign, yaw_deg):
    hinge = torch.tensor([[2.0, -3.0, 0.0]], dtype=torch.float64)
    surface = hinge + torch.tensor([[0.2, 0.0, 0.9]], dtype=torch.float64)
    yaw = math.radians(yaw_deg)
    offset = torch.tensor([[0.1 * math.cos(yaw), 0.1 * math.sin(yaw), 0.01]], dtype=torch.float64)
    angle = torch.tensor([sign * (math.radians(25.0) + PUSH_TIP_MAX_ANGULAR_LEAD_RAD)])
    ee = fingertip_arc_position(surface, offset, hinge, angle)
    tip = ee + offset - hinge
    actual = torch.atan2(tip[:, 1], tip[:, 0]) / sign - math.radians(25.0)
    torch.testing.assert_close(
        actual, torch.full_like(actual, math.radians(4.0)), atol=1.0e-7, rtol=0.0
    )
    assert torch.allclose(ee[:, 2], surface[:, 2] - offset[:, 2])


def test_tip_surface_reference_removes_standoff_without_hidden_preload():
    target = torch.tensor([[0.2, 0.13, 0.9]])
    offsets = torch.tensor([[[0.0, -0.1, 0.0], [0.01, -0.1, 0.0], [0.0, 0.0, 0.0]]])
    depth = torch.tensor([[0.0, -1.0, 0.0]])
    normal = -depth
    tip_offset, surface = panel_fingertip_reference(
        target, offsets, depth, normal, torch.tensor([0.03])
    )
    torch.testing.assert_close(surface[:, 1], torch.zeros(1), atol=1.0e-7, rtol=0.0)
    zero_lead = fingertip_arc_position(
        surface, tip_offset, torch.zeros_like(surface), torch.zeros(1)
    )
    torch.testing.assert_close(zero_lead + tip_offset, surface)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("hinge_angle", [0.25, math.radians(88.7794), math.radians(90.0)])
def test_command_push_uses_four_degree_tip_limit_and_fixed_attitude(sign, hinge_angle):
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage = Stage
    fsm.stage = torch.tensor([int(Stage.PUSH_PANEL_OPEN)])
    fsm.hinge_idx = 0
    fsm.dt = 0.02
    fsm.hinge_axis_sign = torch.tensor([sign])
    fsm.hinge_world = torch.zeros(1, 3)
    fsm.push_tip_offset_world = torch.tensor([[0.08, 0.03, 0.0]])
    surface = torch.tensor([[0.2, 0.0, 0.9]])
    fsm.push_contact_world = surface - fsm.push_tip_offset_world
    fsm.push_speed_error_integral = torch.zeros(1)
    fsm.push_target_angular_speed = torch.tensor([math.radians(20.0)])
    fsm.push_speed_kp, fsm.push_speed_ki = torch.tensor([0.45]), torch.tensor([0.5])
    fsm.push_speed_integral_limit = torch.tensor([math.radians(8.0)])
    fsm.push_max_angular_lead = torch.tensor([math.radians(2.0)])
    fsm.push_tip_max_angular_lead = torch.tensor([PUSH_TIP_MAX_ANGULAR_LEAD_RAD])
    fsm.push_goal_hinge_angle = torch.tensor([math.radians(89.0)])
    fsm.push_planned_stop_hinge = fsm.push_goal_hinge_angle.clone()
    fsm.push_start_hinge_angle = torch.zeros(1)
    fsm.push_command_hinge_angle = torch.zeros(1)
    fsm.push_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    fsm.ee_orientation_command = torch.zeros(1, 4)
    fsm.ee_command_world = fingertip_arc_position(
        surface,
        fsm.push_tip_offset_world,
        fsm.hinge_world,
        torch.tensor([sign * (hinge_angle + PUSH_TIP_MAX_ANGULAR_LEAD_RAD)]),
    )
    fsm.q_closed = torch.zeros(1)
    fsm.push_previous_finger_clearance = torch.zeros(1)
    fsm._base_hold_command = lambda state, rows, buffers=None: (torch.zeros(1, 3), None, None)
    fsm.arm_pose_target = lambda *args: None
    buffers = SimpleNamespace(base=torch.zeros(1, 3), gripper=torch.zeros(1, 1))
    state = {
        "door_joint_position": torch.tensor([[hinge_angle]]),
        "door_joint_velocity": torch.zeros(1, 1),
    }
    fsm.command_push_panel_open(state, torch.tensor([0.03]), buffers)
    tip = fsm.ee_command_world + fsm.push_tip_offset_world
    lead = torch.atan2(tip[:, 1], tip[:, 0]) / sign - hinge_angle
    torch.testing.assert_close(lead, torch.full_like(lead, PUSH_TIP_MAX_ANGULAR_LEAD_RAD))
    torch.testing.assert_close(fsm.ee_orientation_command, fsm.push_orientation)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_push_pi_keeps_bounded_lead_beyond_acceptance_angle(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    hinge = torch.deg2rad(torch.tensor([88.7794, 89.0, 90.0], device=device))
    target, _ = hinge_speed_target(
        hinge, torch.zeros_like(hinge), torch.zeros_like(hinge),
        dt=0.02, target_speed=math.radians(20.0), kp=0.45, ki=0.5,
        integral_limit=math.radians(8.0), max_lead=PUSH_TIP_MAX_ANGULAR_LEAD_RAD,
        start=torch.zeros_like(hinge),
    )
    torch.testing.assert_close(
        target - hinge, torch.full_like(hinge, PUSH_TIP_MAX_ANGULAR_LEAD_RAD)
    )
    assert torch.all(target > math.radians(89.0))


def test_shared_pi_is_bitwise_equal_to_push_baseline():
    hinge = torch.tensor([0.1, 0.5, 1.55])
    velocity = torch.tensor([0.0, 0.4, -0.1])
    integral = torch.tensor([0.01, -0.02, 0.03])
    target_speed, kp, ki, limit = math.radians(20.0), 0.45, 0.5, math.radians(8.0)
    max_lead = math.radians(2.0)
    goal, start = torch.full((3,), math.pi / 2), torch.full((3,), 0.1)
    expected_integral = (integral + (target_speed - velocity) * 0.02).clamp(-limit, limit)
    lead = (kp * (target_speed - velocity) + ki * expected_integral).clamp(-max_lead, max_lead)
    expected = torch.maximum(torch.minimum(hinge + lead, goal), start - math.radians(2.0))
    target, updated = hinge_speed_target(
        hinge,
        velocity,
        integral,
        dt=0.02,
        target_speed=target_speed,
        kp=kp,
        ki=ki,
        integral_limit=limit,
        max_lead=max_lead,
        goal=goal,
        start=start,
    )
    assert torch.equal(target, expected)
    assert torch.equal(updated, expected_integral)
    assert torch.max((target - hinge).abs()) <= max_lead + 1e-7


def test_comfort_workspace_has_same_xy_radius_as_push():
    assert outside_comfort_workspace(
        torch.tensor([[0.3, 0.4, 3.0], [0.71, 0.0, 0.0]]), torch.tensor([0.7, 0.7])
    ).tolist() == [False, True]


def test_pull_position_arc_holds_world_orientation_and_stops_at_workspace_boundary():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device, fsm.Stage = "cpu", Stage
    fsm.stage = torch.tensor([int(Stage.INSERT_INTO_DOOR_GAP)])
    fsm.stage_steps = torch.zeros(1, dtype=torch.int32)
    for name in (
        "push_start_hinge_angle",
        "push_last_progress_hinge",
        "push_speed_error_integral",
        "push_command_hinge_angle",
    ):
        setattr(fsm, name, torch.zeros(1))
    for name in ("push_no_progress_steps", "push_overspeed_steps"):
        setattr(fsm, name, torch.zeros(1, dtype=torch.int32))
    fsm.push_goal_hinge_angle = torch.tensor([math.pi / 2])
    fsm.push_comfort_reach_radius = torch.tensor([0.7])
    fsm.pull_move_start_hinge = torch.zeros(1)
    fsm.hinge_axis_sign, fsm.hinge_idx = torch.ones(1), 0
    fsm.hinge_world = torch.zeros(1, 3)
    fsm.pull_open_path = PaddedPathBank(1, 256, "cpu")
    fsm.env = SimpleNamespace(
        get_goal_reference_origin=lambda: torch.tensor([[1.0, 0.0, 0.0]]),
        get_goal_reference_quat=lambda: torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        get_goal_center_offset_local=lambda: torch.zeros(1, 3),
    )
    state = {
        "door_joint_position": torch.zeros(1, 1),
        "ee_pos": torch.tensor([[1.0, 0.0, 0.8]]),
        "ee_orn": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }
    fsm.begin_pry_door_open(state, torch.tensor([0]))
    count = int(fsm.pull_open_path.length[0])
    assert 2 < count < 181
    points = fsm.pull_open_path.positions[0, :count]
    torch.testing.assert_close(torch.linalg.vector_norm(points[:, :2], dim=-1), torch.ones(count))
    assert torch.equal(points[:, 2], torch.full((count,), 0.8))
    assert torch.equal(
        fsm.pull_open_path.orientations[0, :count], state["ee_orn"].expand(count, -1)
    )
    local = points[:, :2] - torch.tensor([1.0, 0.0])
    assert torch.linalg.vector_norm(local[-1]) > 0.7
    assert torch.linalg.vector_norm(local[-2]) <= 0.7
    assert fsm.stage.item() == int(Stage.PRY_DOOR_OPEN)


@pytest.mark.parametrize("truncate", [False, True])
def test_complete_pose_sequences_use_one_inverse_batch_and_report_bad_env(monkeypatch, truncate):
    from gr00t.rl.auto_open import tensor_planning, pose_planning

    calls = []

    def inverse(positions, orientations, **kwargs):
        calls.append(len(positions))
        error = torch.zeros_like(positions)
        error[-1, 0] = 0.02
        return SimpleNamespace(
            command_position_world=positions + 0.001,
            command_orientation_world_wxyz=orientations,
            position_error=error,
            orientation_error=torch.zeros_like(error),
            arm_joint_positions=torch.zeros((len(positions), 6)),
        )

    monkeypatch.setattr(pose_planning, "solve_pose_targets", inverse)
    coordinator = tensor_planning.TensorPlanningCoordinator.__new__(
        tensor_planning.TensorPlanningCoordinator
    )
    env = SimpleNamespace(
        arm_joint_ids=torch.arange(6), dof_pos_limits=torch.tensor([[-3.0, 3.0]] * 6)
    )
    coordinator.controllers = [
        SimpleNamespace(env=env, last_acceptance_diagnostics={}) for _ in range(2)
    ]
    coordinator.arm_joint_ids, coordinator.joint_limits = env.arm_joint_ids, env.dof_pos_limits
    coordinator.Stage, coordinator.path_diagnostics = Stage, []
    coordinator.gpu_stage_metadata = []
    coordinator.stats = {"batches": 0, "requests": 0, "errors": 0, "labels": {}}
    bank = PaddedPathBank(2, 8, "cpu")
    positions = torch.zeros(2, 5, 3)
    positions[1] = 1.0
    orientations = torch.zeros(2, 5, 4)
    orientations[..., 0] = 1.0
    bank.install_tensor(torch.arange(2), positions, orientations, torch.tensor([3, 5]))
    failures, diagnostics = [], []
    fsm = SimpleNamespace(
        device="cpu",
        pull_insert_path=bank,
        stage=torch.tensor([int(Stage.PRY_DOOR_OPEN)] * 2),
        fail=lambda rows, code: failures.append((rows.tolist(), code)),
        record_acceptance=lambda *args, **kwargs: diagnostics.append(args),
    )
    root = torch.zeros(2, 13)
    root[:, 3] = 1.0
    coordinator.plan_pose_path(
        fsm,
        {"dof_pos": torch.zeros(2, 6), "root_states": root},
        torch.arange(2),
        bank_name="pull_insert_path",
        label="test",
        prediction_tolerance=0.004,
        truncate_at_first_unreachable=truncate,
    )
    assert calls == [8]
    assert [entry for entry in failures if entry[0]] == ([] if truncate else [([1], 49)])
    assert bank.length.tolist() == ([3, 4] if truncate else [3, 5])
    nonempty = next(entry for entry in diagnostics if len(entry[1]))
    assert nonempty[2]["path_inverse_position_residual_m"][1].tolist() == pytest.approx(
        [0.0, 0.0 if truncate else 0.02]
    )
    coordinator.finalize_diagnostics()
    report = coordinator.controllers[1].last_acceptance_diagnostics["test_inverse"]
    assert report["predicted_position_world_m"] == pytest.approx(
        [1.0, 1.0, 1.0] if truncate else [1.02, 1.0, 1.0]
    )
    assert report["inverse_command_position_world_m"] == pytest.approx([1.001, 1.001, 1.001])


def test_workspace_boundary_requires_thirty_degree_opening_before_retraction():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    from gr00t.rl.tests.test_auto_open_doorway_clearance import install_doorway
    install_doorway(fsm, 2)
    fsm.Stage, fsm.pull_door, fsm.device = Stage, True, "cpu"
    fsm.pull_panel_config = PullPanelConfig()
    fsm.terminal_stage = Stage.PASS_THROUGH_DOORWAY
    fsm.stage = torch.tensor([int(Stage.PRY_DOOR_OPEN)] * 2)
    fsm.stage_steps = torch.zeros(2, dtype=torch.int32)
    fsm.hinge_idx = 0
    fsm.push_max_angular_speed = torch.full((2,), 0.5)
    fsm.push_overspeed_steps = torch.zeros(2, dtype=torch.int32)
    fsm.push_no_progress_steps = torch.zeros(2, dtype=torch.int32)
    fsm.push_last_progress_hinge = torch.full((2,), 0.1)
    fsm.push_start_hinge_angle = torch.full((2,), 0.1)
    fsm.push_command_hinge_angle = torch.full((2,), 0.2)
    fsm.push_comfort_reach_radius = torch.full((2,), 0.7)
    fsm.push_goal_hinge_angle = torch.full((2,), math.pi / 2)
    fsm.pull_open_path = PaddedPathBank(2, 256, "cpu")
    fsm.pull_open_path.length[:] = 181
    fsm.ee_command_world = torch.zeros(2, 3)
    fsm.pull_finished_for_reposition = torch.zeros(2, dtype=torch.bool)
    fsm.pull_door_opened = torch.zeros(2, dtype=torch.bool)
    state = {
        "door_joint_position": torch.tensor([[0.2], [0.1]]),
        "door_joint_velocity": torch.full((2, 1), 0.1),
        "ee_goal_cart_local": torch.tensor([[0.71, 0.0, 0.0]] * 2),
    }
    assert fsm.advance_pry_door_open(state).tolist() == []
    state["door_joint_position"][:] = math.radians(31.0)
    assert fsm.advance_pry_door_open(state).tolist() == [0, 1]
    assert fsm.stage.tolist() == [int(Stage.STOW_ARM_AFTER_PULL)] * 2
    fsm.stage[:] = int(Stage.PRY_DOOR_OPEN)
    state["ee_goal_cart_local"][:, 0] = 0.6
    fsm.pull_open_path.length[:] = 2
    fsm.pull_open_path.positions[:, 1, 0] = 0.01
    assert fsm.advance_pry_door_open(state).tolist() == []
    fsm.ee_command_world[:, 0] = 0.01
    assert fsm.advance_pry_door_open(state).tolist() == [0, 1]


def test_small_progress_station_noise_stays_in_range_and_cannot_cancel_reposition():
    limit = math.radians(5.0)
    progress = torch.full((3,), math.radians(0.218))
    noise = torch.tensor([-limit, 0.0, limit])
    radius = torch.full((3,), 1.6)
    angle, effective, feasible = follow_station_rotation(
        progress, noise, radius, max_noise_rad=limit, inner_distance_m=0.05, inner_yaw_rad=0.05
    )
    assert feasible.all() and (effective.abs() <= limit + 1e-7).all()
    assert (2.0 * radius * torch.sin(angle / 2.0) > 0.1).all()
    assert (angle > 0.05).all()
    ordinary = torch.full((3,), math.radians(20.0))
    angle, effective, feasible = follow_station_rotation(
        ordinary, noise, radius, max_noise_rad=limit, inner_distance_m=0.05, inner_yaw_rad=0.05
    )
    assert feasible.all() and torch.equal(effective, noise)
    assert torch.equal(angle, ordinary + noise)


def test_pull_insertion_accepts_stable_standoff_without_any_contact_input():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage, fsm.pull_door, fsm.device = Stage, True, "cpu"
    fsm.stage = torch.tensor([int(Stage.INSERT_INTO_DOOR_GAP)])
    fsm.stage_steps = torch.zeros(1, dtype=torch.int32)
    fsm.terminal_stage = Stage.INSERT_INTO_DOOR_GAP
    fsm.pull_insert_goal = torch.tensor([[0.1, 0.5, 0.8]])
    fsm.pull_insert_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    fsm.pull_face_offset_local = torch.zeros(1, 3)
    fsm.pull_panel_contact = fsm.pull_insert_goal + torch.tensor([[0.002, 0.0, 0.0]])
    fsm.pull_insert_stable_steps = torch.zeros(1, dtype=torch.int32)
    fsm.pull_cycle = torch.ones(1, dtype=torch.int64)
    fsm.pull_insert_path = PaddedPathBank(1, 8, "cpu")
    fsm.pull_insert_path.length[:] = 1
    fsm.arm_stability = SimpleNamespace(
        update=lambda state, rows: (torch.ones(1, dtype=torch.bool),)
    )
    fsm.hinge_idx, fsm.hinge_axis_sign = 0, torch.ones(1)
    fsm.door_normal = torch.tensor([[1.0, 0.0, 0.0]])
    fsm.q_closed = torch.zeros(1)
    state = {
        "ee_pos": fsm.pull_insert_goal.clone(),
        "ee_orn": fsm.pull_insert_orientation.clone(),
        "door_joint_position": torch.zeros(1, 1),
        "root_states": torch.zeros(1, 13),
        "gripper_position": torch.zeros(1, 1),
    }
    for _ in range(7):
        fsm.advance_insert_into_door_gap(state)
        assert fsm.stage.item() == int(Stage.INSERT_INTO_DOOR_GAP)
    fsm.advance_insert_into_door_gap(state)
    assert fsm.stage.item() == int(Stage.SUCCESS)


def test_cycle_draws_are_seed_and_visit_scoped():
    from gr00t.rl.auto_open.tensor_fsm import _pull_cycle_uniform

    seeds, cycles = torch.tensor([1700]), torch.tensor([0])
    first = _pull_cycle_uniform(seeds, cycles, 8)
    assert torch.equal(first, _pull_cycle_uniform(seeds, cycles, 8))
    assert not torch.equal(first, _pull_cycle_uniform(seeds, cycles + 1, 8))
    assert ((first >= 0.0) & (first < 1.0)).all()


def test_follow_station_rotates_position_and_yaw_about_same_hinge():
    root = torch.tensor([[2.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0]])
    xy, yaw = rotate_base_station(
        root, torch.tensor([[1.0, 1.0, 0.0]]), torch.tensor([math.pi / 2])
    )
    torch.testing.assert_close(xy, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(yaw, torch.tensor([math.pi / 2]))


def test_station_sweep_rejects_crossing_wall_outside_frame():
    body = torch.tensor([[x, y, 0.0] for x in (-0.1, 0.1) for y in (-0.1, 0.1)])
    root = torch.tensor([[1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    assert base_move_clearance(
        root,
        torch.tensor([[0.8, 2.0], [-0.2, 2.0]]),
        torch.zeros(2),
        body,
        hinge=torch.zeros(2, 3),
        normal=torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
        tangent=torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1),
        frame_edge=torch.ones(2),
        wall_front=torch.full((2,), 0.01),
        panel_center=torch.tensor([[0.0, 0.5, 1.0]]).repeat(2, 1),
        panel_normal=torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
        panel_tangent=torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1),
        panel_width=torch.ones(2),
        panel_half_thickness=torch.full((2,), 0.02),
    ).tolist() == [True, False]


def test_station_sweep_preserves_empty_space_between_separate_colliders():
    part = torch.tensor(
        [[x, y, z] for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (0.0, 0.1)]
    )
    separate = torch.stack(
        (part + torch.tensor([-0.3, 0.5, 0.0]), part + torch.tensor([0.3, 0.5, 0.0]))
    )[None]
    root = torch.tensor([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]])
    geometry = dict(
        hinge=torch.zeros(1, 3),
        normal=torch.tensor([[1.0, 0.0, 0.0]]),
        tangent=torch.tensor([[0.0, 1.0, 0.0]]),
        frame_edge=torch.ones(1),
        wall_front=torch.zeros(1),
        panel_center=torch.tensor([[0.0, 0.5, 1.0]]),
        panel_normal=torch.tensor([[1.0, 0.0, 0.0]]),
        panel_tangent=torch.tensor([[0.0, 1.0, 0.0]]),
        panel_width=torch.ones(1),
        panel_half_thickness=torch.full((1,), 0.02),
    )
    assert base_move_clearance(root, root[:, :2], torch.zeros(1), separate, **geometry).item()
    combined = torch.tensor(
        [[x, y, z] for x in (-0.35, 0.35) for y in (0.45, 0.55) for z in (0.0, 0.1)]
    )
    assert not base_move_clearance(root, root[:, :2], torch.zeros(1), combined, **geometry).item()


def test_station_pose_resampling_selects_safe_target_instead_of_reusing_bad_one():
    root = torch.tensor([[1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0]])
    body = torch.tensor([[x, y, 0.0] for x in (-0.1, 0.1) for y in (-0.1, 0.1)])
    result = base_station_candidate_paths(
        root,
        torch.tensor([[[-0.2, 2.0], [0.8, 2.0]]]),
        torch.zeros(1, 2),
        body,
        hinge=torch.zeros(1, 3),
        normal=torch.tensor([[1.0, 0.0, 0.0]]),
        tangent=torch.tensor([[0.0, 1.0, 0.0]]),
        frame_edge=torch.ones(1),
        wall_front=torch.zeros(1),
        panel_center=torch.tensor([[0.0, 0.5, 1.0]]),
        panel_normal=torch.tensor([[1.0, 0.0, 0.0]]),
        panel_tangent=torch.tensor([[0.0, 1.0, 0.0]]),
        panel_width=torch.ones(1),
        panel_half_thickness=torch.full((1,), 0.02),
    )
    assert result[2].item() and result[4].item() == 1
    torch.testing.assert_close(result[0][0, -1, :2], torch.tensor([0.8, 2.0]))


def test_door_coasting_open_during_insertion_returns_a_retract_planning_request():
    fsm = _station_fsm([Stage.INSERT_INTO_DOOR_GAP])
    fsm.terminal_stage = Stage.PASS_THROUGH_DOORWAY
    fsm.pull_insert_goal = torch.zeros(1, 3)
    fsm.pull_insert_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    fsm.pull_face_offset_local = torch.zeros(1, 3)
    fsm.pull_panel_contact = torch.tensor([[0.002, 0.0, 0.0]])
    fsm.door_normal = torch.tensor([[1.0, 0.0, 0.0]])
    fsm.hinge_axis_sign = torch.ones(1)
    fsm.q_closed = torch.zeros(1)
    fsm.arm_stability = SimpleNamespace(update=lambda *args: (torch.ones(1, dtype=torch.bool),))
    fsm.pull_insert_stable_steps = torch.tensor([7])
    fsm.pull_cycle = torch.ones(1, dtype=torch.int64)
    fsm.pull_insert_path = SimpleNamespace(
        complete=lambda rows: torch.ones(len(rows), dtype=torch.bool),
        index=torch.zeros(1, dtype=torch.int64),
        length=torch.ones(1, dtype=torch.int64),
    )
    state = {
        "ee_pos": torch.zeros(1, 3),
        "ee_orn": fsm.pull_insert_orientation,
        "door_joint_position": torch.tensor([[math.radians(94.0)]]),
        "root_states": torch.zeros(1, 13),
        "gripper_position": torch.zeros(1, 1),
    }
    completed = fsm.advance_insert_into_door_gap(state)
    assert completed.tolist() == [0]
    assert fsm.stage.item() == int(Stage.STOW_ARM_AFTER_PULL)


def test_retract_completion_selects_follow_station_or_direct_through():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    from gr00t.rl.tests.test_auto_open_doorway_clearance import install_doorway
    install_doorway(fsm, 2)
    from gr00t.rl.auto_open.pull_station import PullStationConfig

    fsm.pull_station_config = PullStationConfig()
    fsm.Stage, fsm.device = Stage, "cpu"
    fsm.stage = torch.full((2,), int(Stage.STOW_ARM_AFTER_PULL))
    fsm.stage_steps = torch.ones(2, dtype=torch.int32)
    fsm.pull_panel_config = PullPanelConfig()
    fsm.pull_door_opened = torch.zeros(2, dtype=torch.bool)
    fsm.front_shoulder_body_indices = torch.tensor([0])
    fsm.pull_shoulder_anchor = torch.zeros(2, 2)
    fsm.pull_shoulder_local = torch.zeros(2, 2)
    fsm.pull_panel_side = torch.ones(2)
    fsm.pending_station_distance = torch.zeros(2)
    fsm.pending_yaw_error = torch.zeros(2)
    from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory

    fsm.pull_base_trajectory = BasePoseTrajectory(2, "cpu")
    fsm.pull_follow_holding = torch.zeros(2, dtype=torch.bool)
    fsm.pull_follow_velocity = torch.zeros(2, 3)
    fsm.dt = 0.02
    handoff = []

    def through(rows, *, rolling=False):
        handoff.append((rows.tolist(), rolling))
        fsm.stage[rows] = int(Stage.PASS_THROUGH_DOORWAY)

    fsm.begin_pass_through_doorway = through
    fsm.hinge_idx, fsm.hinge_axis_sign = 0, -torch.ones(2)
    fsm.hinge_world = torch.zeros(2, 3)
    fsm.pull_move_start_hinge = torch.tensor([0.1, 0.1])
    fsm.pull_cycle_samples = torch.zeros(2, 4)
    fsm.door_forward = torch.tensor([[-1.0, 0.0, 0.0]]).repeat(2, 1)
    fsm.door_normal = -fsm.door_forward
    fsm.initial_door_root = torch.zeros(2, 3)
    fsm.pull_station_xy, fsm.pull_station_yaw = torch.zeros(2, 2), torch.zeros(2)
    fsm.pull_station_history = PullTensorHistory(2, _pull_station_history_record)
    fsm.pull_base_path = torch.zeros(2, 2, 3)
    fsm.pull_base_path_length = torch.zeros(2, dtype=torch.int64)
    fsm.pull_base_path_index = torch.zeros(2, dtype=torch.int64)
    fsm.align_panel_arm_joint_target = torch.zeros(2, 6)
    fsm.env = SimpleNamespace(arm_joint_ids=torch.arange(6))
    fsm.reset_base_arrival = lambda rows, profile: None
    fsm.pull_free_direction = torch.tensor([[0.0, 1.0]]).repeat(2, 1)
    fsm.panel_center_world = torch.tensor([[0.0, 0.5, 1.0]]).repeat(2, 1)
    fsm.pull_base_points = torch.zeros(4, 3)
    fsm.pull_motion_points = torch.zeros(4, 3)
    fsm.pull_motion_body_indices = torch.zeros(4, dtype=torch.int64)
    fsm.pull_frame_edge, fsm.pull_free_edge = torch.ones(2), torch.ones(2)
    fsm.pull_wall_front, fsm.panel_half_thickness = torch.zeros(2), torch.full((2,), 0.02)
    state = {
        "root_states": torch.tensor([[1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        "door_joint_position": torch.tensor([[0.3], [math.radians(89.0)]]),
        "robot_body_state_w": torch.zeros(2, 1, 13),
        "dof_pos": torch.ones(2, 6),
    }
    fsm.begin_follow_wall_to_panel(state, torch.arange(2))
    assert fsm.stage.tolist() == [int(Stage.FOLLOW_WALL_TO_PANEL), int(Stage.PASS_THROUGH_DOORWAY)]
    assert handoff == [([1], True)]
    assert fsm.pull_base_trajectory.length[0] > 1
    assert fsm.pull_base_trajectory.length[1] == 0


def _station_fsm(stages):
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    from gr00t.rl.tests.test_auto_open_doorway_clearance import install_doorway
    install_doorway(fsm, len(stages))
    fsm.Stage, fsm.device, fsm.pull_door = Stage, "cpu", True
    fsm.stage = torch.tensor([int(s) for s in stages])
    n = len(stages)
    fsm.stage_steps = torch.ones(n, dtype=torch.int32)
    fsm.hinge_idx, fsm.pull_panel_config = 0, PullPanelConfig()
    fsm.pull_door_opened = torch.zeros(n, dtype=torch.bool)
    fsm.base_arrival_confirmed = torch.zeros(n, dtype=torch.bool)
    fsm.base_stability_confirmed = torch.zeros(n, dtype=torch.bool)
    fsm.base_arrival_settle_steps = torch.zeros(n, dtype=torch.int32)
    fsm.base_stability_steps = torch.zeros(n, dtype=torch.int32)
    fsm.base_arrival_required_frames = torch.full((n,), 25)
    fsm.pull_base_path_index = torch.zeros(n, dtype=torch.int64)
    fsm.pull_base_path_length = torch.ones(n, dtype=torch.int64)
    fsm.pending_station_distance = torch.full((n,), 0.04)
    fsm.pending_yaw_error = torch.full((n,), 0.03)
    fsm.record_acceptance = lambda *args, **kwargs: None
    return fsm


def test_open_door_during_base_motion_redirects_without_another_pry():
    fsm = _station_fsm([Stage.FOLLOW_WALL_TO_PANEL])
    redirected = []
    fsm.begin_pass_through_doorway = lambda rows, *, rolling=False: redirected.append(
        (rows.tolist(), rolling)
    )
    fsm.advance_follow_wall_to_panel({"door_joint_position": torch.tensor([[math.radians(82.0)]])})
    assert redirected == [([0], True)]
    assert fsm.stage_steps.tolist() == [1]


def test_open_door_hands_off_without_stationary_window_or_alignment():
    fsm = _station_fsm([Stage.FOLLOW_WALL_TO_PANEL, Stage.FOLLOW_WALL_TO_PANEL])
    fsm.pending_yaw_error[1] = 0.2
    handoff = []
    fsm.begin_pass_through_doorway = lambda rows, *, rolling=False: handoff.append(
        (rows.tolist(), rolling)
    )
    fsm.advance_follow_wall_to_panel(
        {"door_joint_position": torch.full((2, 1), math.radians(94.8))}
    )
    assert handoff == [([0, 1], True)]
    assert not fsm.base_stability_confirmed.any()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_free_edge_bridge_avoids_leg_collision_and_is_batch_independent(device):
    root = torch.tensor(
        [[-6.3821716, 8.6028814, 0.5119839, -0.4024237, -0.0159009, 0.0104338, 0.9152560]]
    )
    xy, yaw = torch.tensor([[-7.23, 8.0]]), torch.tensor([-math.pi])
    body = torch.tensor([[x, y, 0.0] for x in (-0.665482, 0.53) for y in (-0.315258, 0.288757)])
    geometry = {
        key: torch.tensor(value)
        for key, value in {
            "hinge": [[-7.98, 8.507818, 0.0]],
            "normal": [[1.0, 0.0, 0.0]],
            "tangent": [[0.0, -1.0, 0.0]],
            "frame_edge": [1.015636],
            "wall_front": [0.01],
            "panel_center": [[-7.472034, 8.5236, 1.026306]],
            "panel_normal": [[-0.07036424, 0.9975214, 0.0]],
            "panel_tangent": [[0.9975214, 0.07036424, 0.0]],
            "panel_width": [1.013636],
            "panel_half_thickness": [0.02],
        }.items()
    }
    root, xy, yaw, body = (value.to(device) for value in (root, xy, yaw, body))
    geometry = {key: value.to(device) for key, value in geometry.items()}
    assert not base_move_clearance(root, xy, yaw, body, **geometry).item()
    path, lengths, safe, margins = base_station_waypoints(root, xy, yaw, body, **geometry)
    assert safe.item() and lengths.item() == 2
    assert all(value.item() >= 0.0 for value in margins.values())
    torch.testing.assert_close(path[0, -1], torch.cat((xy[0], yaw)))
    repeated = {key: value.expand(2, *value.shape[1:]) for key, value in geometry.items()}
    batched = base_station_waypoints(
        root.expand(2, -1),
        xy.expand(2, -1),
        yaw.expand(2),
        body[None].expand(2, -1, -1),
        **repeated,
    )
    assert torch.equal(batched[0][0], path[0])
    assert torch.equal(batched[0][1], path[0])


@pytest.mark.parametrize("rolling", [False, True])
def test_through_entry_motion_ready_matches_handoff(rolling):
    fsm = _station_fsm([Stage.FOLLOW_WALL_TO_PANEL])
    from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory

    fsm.through_trajectory = BasePoseTrajectory(1, "cpu")
    fsm.through_path_previous_index = torch.zeros(1, dtype=torch.int64)
    for name in (
        "through_best_station_error",
        "through_no_progress_steps",
        "through_settle_steps",
        "through_middle_ready_steps",
        "through_entry_stable_steps",
    ):
        setattr(fsm, name, torch.zeros(1))
    fsm.through_stop_latched = torch.ones(1, dtype=torch.bool)
    fsm.through_arm_restore_ready = torch.ones(1, dtype=torch.bool)
    fsm.through_arm_carry_initialized = torch.ones(1, dtype=torch.bool)
    fsm.through_arm_carry_ready = torch.ones(1, dtype=torch.bool)
    fsm.through_motion_ready = torch.zeros(1, dtype=torch.bool)
    fsm.begin_pass_through_doorway(torch.tensor([0]), rolling=rolling)
    assert fsm.through_motion_ready.item() == rolling
    assert fsm.stage.item() == int(Stage.PASS_THROUGH_DOORWAY)
