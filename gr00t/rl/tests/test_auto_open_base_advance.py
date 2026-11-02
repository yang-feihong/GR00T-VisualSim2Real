"""Forward-ray fallback when the old contact's 40 cm circle cannot be reached."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.control import APPROACH_ARRIVAL
from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM


@pytest.fixture(autouse=True, params=['cpu', 'cuda'])
def device_context(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    with torch.device(request.param):
        yield


def _planner(contacts):
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    count = len(contacts)
    fsm.Stage = SimpleNamespace(ADVANCE_TO_NEXT_PUSH_STATION=1)
    fsm.stage = torch.ones(count, dtype=torch.int64)
    fsm.stage_steps = torch.zeros(count, dtype=torch.int32)
    fsm.mask = lambda stage: fsm.stage == stage
    root = torch.zeros(count, 13)
    fsm.device = root.device
    from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory
    fsm.panel_station_trajectory = BasePoseTrajectory(count, fsm.device)
    fsm.panel_station_prefix_steps = torch.zeros(count, dtype=torch.int32)
    fsm.base_arrival_inner_distance = torch.full((count,), .02)
    fsm.dt = .02
    root[:, 2:4] = torch.tensor([0.5, 1.0])
    fsm.door_forward = torch.tensor([[1., 0., 0.]]).expand(count, -1)
    fsm.panel_station_xy = torch.zeros(count, 2)
    fsm.hinge_idx = 0
    fsm.hinge_axis_sign = torch.ones(count)
    fsm.hinge_world = torch.tensor([[10., 10., 0.]]).expand(count, -1).clone()
    fsm.panel_center_world = torch.tensor([[10., 10.5, 0.]]).expand(count, -1).clone()
    fsm.front_shoulder_body_indices = torch.tensor([0, 1])
    fsm.push_shoulder_collision_radius = torch.full((count,), .155)
    fsm.panel_half_thickness = torch.full((count,), .02)
    fsm.push_shoulder_safety_margin = torch.full((count,), .02)
    fsm.push_reengage_reach_radius = torch.full((count,), .4)
    fsm.push_reengage_release_radius = torch.full((count,), .45)
    fsm.push_reengage_target_lead = torch.full((count,), .05)
    fsm.push_reengage_reachable = torch.zeros(count, dtype=torch.bool)
    fsm.base_advance_closest_fallback = torch.zeros(count, dtype=torch.bool)
    fsm.base_advance_target_longitudinal = torch.full((count,), -torch.inf)
    fsm.env = SimpleNamespace(
        get_goal_reference_origin=lambda: root[:, :3],
        get_goal_reference_quat=lambda: root[:, 3:7],
        get_goal_center_offset_local=lambda: torch.zeros(count, 3),
    )
    contact = torch.tensor(contacts)
    fsm._tracked_push_contact = lambda state, rows: contact[rows]
    state = {
        'root_states': root,
        'door_joint_position': torch.zeros(count, 1),
        'door_joint_velocity': torch.zeros(count, 1),
        'robot_body_state_w': torch.full((count, 2, 13), 100.),
    }
    return fsm, state


def test_forward_target_uses_closest_point_when_circle_has_no_intersection():
    fsm, state = _planner([[.8, .5, .7], [.8, .2, .7], [-.8, .5, .7]])
    target, _ = fsm._base_advance_targets(state, torch.arange(3))
    expected = torch.tensor([[.8, 0.], [.8 - (.4**2 - .2**2)**.5 + .05, 0.], [0., 0.]])
    torch.testing.assert_close(target, expected)
    assert fsm.base_advance_closest_fallback.tolist() == [True, False, True]
    # The fallback is a true minimum on the forward ray, not an extra overshoot.
    point = torch.tensor([.8, .5])
    for dx in (-.1, .1):
        assert torch.linalg.vector_norm(target[0] - point) < torch.linalg.vector_norm(
            target[0] + torch.tensor([dx, 0.]) - point
        )


def test_first_search_center_is_converted_from_surface_to_ee_reference():
    fsm, state = _planner([[.8, .2, .7]])
    fsm.door_normal = torch.tensor([[-1., 0., 0.]])
    fsm.depth_axis_local = torch.tensor([[1., 0., 0.]])
    fsm.panel_station_search_center = torch.tensor([[.8, .2, .7]])
    fsm.panel_station_half_thickness = torch.zeros(1)
    fsm.panel_station_fixed_body, fsm.panel_station_moving_body = 0, 1
    fsm.panel_station_fixed_points = torch.tensor([[.1, 0., 0.], [.2, 0., 0.]])
    fsm.panel_station_moving_points = fsm.panel_station_fixed_points.clone()
    state['ee_pos'] = torch.tensor([[.2, 0., .7]])
    state['ee_orn'] = torch.tensor([[1., 0., 0., 0.]])
    state['robot_body_state_w'].zero_()
    state['robot_body_state_w'][:, :, :3] = state['ee_pos'][:, None]
    state['robot_body_state_w'][:, :, 3] = 1
    fsm.push_contact_world = torch.zeros(1, 3)
    fsm.push_tip_offset_world = torch.zeros(1, 3)
    fsm.push_start_hinge_angle = torch.zeros(1)
    fsm.push_orientation = torch.zeros(1, 4)
    fsm._initialize_first_push_reference(state, torch.tensor([0]))
    torch.testing.assert_close(fsm.push_contact_world, torch.tensor([[.6, .2, .7]]))
    torch.testing.assert_close(fsm.push_contact_world + fsm.push_tip_offset_world,
                               fsm.panel_station_search_center)


@pytest.mark.parametrize('direction', [1., -1.])
@pytest.mark.parametrize('scene_x', [0., 8.])
def test_trunk_target_cannot_cross_frame_even_with_old_forward_history(direction, scene_x):
    contact_x = scene_x + direction * .8
    fsm, state = _planner([[contact_x, .5, .7]])
    fsm.door_forward = torch.tensor([[direction, 0., 0.]])
    fsm.panel_station_xy[:, 0] = scene_x
    state['root_states'][:, 0] = scene_x
    fsm.hinge_world[:, 0] = scene_x + direction * .5
    fsm.panel_center_world[:, 0] = fsm.hinge_world[:, 0]
    rows = torch.tensor([0])
    target, _ = fsm._base_advance_targets(state, rows)
    torch.testing.assert_close(target, torch.tensor([[scene_x + direction * .5, 0.]]))
    assert fsm.base_advance_closest_fallback.item()
    # A previous target or an actual overshoot must not defeat the hard bound.
    fsm.base_advance_target_longitudinal[:] = 1.
    state['root_states'][:, 0] = scene_x + direction * .6
    target, _ = fsm._base_advance_targets(state, rows)
    torch.testing.assert_close(target, torch.tensor([[scene_x + direction * .5, 0.]]))
    assert fsm.base_advance_target_longitudinal.item() == pytest.approx(.5)


def test_frame_limit_also_caps_the_five_centimeter_forward_lead():
    fsm, state = _planner([[.8, .2, .7]])
    fsm.hinge_world[:, 0] = .5
    fsm.panel_center_world[:, 0] = .5
    target, _ = fsm._base_advance_targets(state, torch.tensor([0]))
    torch.testing.assert_close(target, torch.tensor([[.5, 0.]]))
    assert fsm.base_advance_closest_fallback.item()


def test_first_panel_station_is_one_path_with_only_final_acceptance():
    from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory

    fsm, state = _planner([[.8, .2, .7]])
    fsm.Stage = SimpleNamespace(MOVE_TO_PANEL_STATION=1, MOVE_TO_PANEL_STANDOFF=2)
    state['root_states'][:, 1] = .4
    state['ee_pos'] = torch.tensor([[.2, .4, .8]])
    state['ee_orn'] = torch.tensor([[1., 0., 0., 0.]])
    fsm._initialize_first_push_reference = lambda *args: None
    fsm.panel_station_command_xy = torch.zeros(1, 2)
    fsm.panel_station_trajectory = BasePoseTrajectory(1, fsm.device)
    fsm.dt = .02
    fsm.turn_release_max_hinge_speed = torch.tensor([.05])
    for name in ('pending_station_distance', 'pending_yaw_error', 'base_hold_target_yaw', 'base_hold_nominal_yaw'):
        setattr(fsm, name, torch.zeros(1))
    for name in ('base_hold_target_xy', 'base_hold_nominal_xy', 'ee_command_world'):
        setattr(fsm, name, torch.zeros(1, 3 if name == 'ee_command_world' else 2))
    fsm.ee_orientation_command = torch.zeros(1, 4)
    fsm.ee_command_valid = torch.zeros(1, dtype=torch.bool)
    fsm.align_panel_arm_joint_target = torch.zeros(1, 6)
    fsm.q_closed = torch.zeros(1)
    fsm.arm_joint_target = fsm.gripper_target = lambda *args: None
    fsm.base_target = lambda *args, **kwargs: (None, torch.zeros(1), torch.zeros(1))
    fsm.record_acceptance = lambda *args, **kwargs: None
    fsm.base_arrival_confirmed = fsm.base_stability_confirmed = torch.ones(1, dtype=torch.bool)
    fsm.base_arrival_settle_steps = fsm.base_stability_steps = fsm.base_arrival_required_frames = torch.ones(1)
    fsm.panel_contact_plan_valid = torch.ones(1, dtype=torch.bool)
    fsm.panel_contact_settle_steps = torch.ones(1, dtype=torch.int32)
    fsm.transition = lambda rows, stage: fsm.stage.__setitem__(rows, stage)
    fsm.through_min_base_height = torch.tensor([.3])
    fsm.push_recovery_settle_steps = torch.zeros(1, dtype=torch.int32)
    fsm.push_start_hinge_angle = torch.zeros(1)
    fsm.push_orientation = torch.tensor([[1., 0., 0., 0.]])
    fsm.panel_contact_orientation_seed = torch.zeros(1, 4)
    fsm.panel_contact_orientation_seed_valid = torch.zeros(1, dtype=torch.bool)
    rows = torch.tensor([0])
    fsm.command_move_to_panel_station(state, SimpleNamespace())
    torch.testing.assert_close(fsm.panel_station_trajectory.poses[0, 0, :2], state['root_states'][0, :2])
    final = fsm.panel_station_trajectory.poses[0, fsm.panel_station_trajectory.length[0] - 1].clone()
    assert final[0] > .8
    fsm.panel_station_trajectory.index[:] = fsm.panel_station_prefix_steps - 1
    fsm.command_move_to_panel_station(state, SimpleNamespace())
    fsm.push_reengage_reachable[:] = True
    assert not len(fsm.advance_move_to_panel_station(state))
    assert fsm.stage.item() == fsm.Stage.MOVE_TO_PANEL_STATION
    state['root_states'][0, :2] = torch.tensor([.5, 0.])
    for _ in range(11):
        fsm.command_move_to_panel_station(state, SimpleNamespace())
    assert fsm.panel_station_trajectory.index.item() < fsm.panel_station_trajectory.length.item() - 1
    assert fsm.advance_move_to_panel_station(state).tolist() == [0]


def test_closest_fallback_preserves_shoulder_clearance_and_monotonic_target():
    fsm, state = _planner([[.8, .5, .7]])
    fsm.hinge_world[:] = torch.tensor([.5, -1., 0.])
    fsm.panel_center_world[:] = torch.tensor([.5, 0., 0.])
    state['robot_body_state_w'][:, :, :2] = 0.
    target, _ = fsm._base_advance_targets(state, torch.tensor([0]))
    assert 0. < target[0, 0] <= .5 - (.155 + .02 + .02)
    previous = target.clone()
    fsm._tracked_push_contact = lambda state, rows: torch.tensor([[.1, .5, .7]])
    target, _ = fsm._base_advance_targets(state, torch.tensor([0]))
    torch.testing.assert_close(target, previous)


@pytest.mark.parametrize('yaw_offset', [0., .1])
@pytest.mark.parametrize('stop_x', [.8, .5])
def test_closest_station_stops_and_can_replan_without_old_point_being_reachable(yaw_offset, stop_x):
    fsm, state = _planner([[.8, .5, .7]])
    if stop_x == .5:
        fsm.hinge_world[:, 0] = stop_x
        fsm.panel_center_world[:, 0] = stop_x
    fsm.through_min_base_height = torch.tensor([.3])
    for name in ('base_arrival_confirmed', 'base_stability_confirmed'):
        setattr(fsm, name, torch.zeros(1, dtype=torch.bool))
    for name in (
        'base_arrival_settle_steps', 'base_arrival_max_settle_steps',
        'base_arrival_required_frames', 'base_stability_steps', 'push_recovery_settle_steps',
    ):
        setattr(fsm, name, torch.zeros(1, dtype=torch.int32))
    for name in (
        'base_arrival_inner_distance', 'base_arrival_outer_distance',
        'base_arrival_inner_yaw', 'base_arrival_outer_yaw',
        'base_arrival_max_linear_speed', 'base_arrival_max_angular_speed',
    ):
        setattr(fsm, name, torch.zeros(1))
    fsm.reset_base_arrival(torch.tensor([0]), APPROACH_ARRIVAL)
    fsm.base_command_previous = torch.zeros(1, 3)
    for name in ('base_reversal_pending_sign', 'base_reversal_pending_steps', 'base_reversal_lock_steps'):
        dtype = torch.int8 if name == 'base_reversal_pending_sign' else torch.int32
        setattr(fsm, name, torch.zeros(1, 3, dtype=dtype))
    fsm.turn_release_max_hinge_speed = torch.tensor([.035])
    fsm.pending_station_distance = torch.zeros(1)
    fsm.pending_yaw_error = torch.zeros(1)
    fsm.base_hold_target_xy = torch.zeros(1, 2)
    fsm.base_hold_target_yaw = torch.zeros(1)
    fsm.base_hold_nominal_xy = torch.zeros(1, 2)
    fsm.base_hold_nominal_yaw = torch.zeros(1)
    fsm.base_advance_arm_joint_target = torch.zeros(1, 6)
    fsm.q_closed = torch.zeros(1)
    fsm.arm_joint_target = lambda *args: None
    fsm.gripper_target = lambda *args: None
    fsm.record_acceptance = lambda *args, **kwargs: None
    buffers = SimpleNamespace(base=torch.zeros(1, 3))
    import math
    state['root_states'][0, 3] = math.cos(yaw_offset / 2)
    state['root_states'][0, 6] = math.sin(yaw_offset / 2)
    fsm.command_advance_to_next_push_station(state, buffers)
    for _ in range(5):
        fsm.command_advance_to_next_push_station(state, buffers)
    assert buffers.base[0, 0] > 0.
    state['root_states'][0, 0] = stop_x
    # Actual arrival can precede the time-indexed reference; it must still stop.
    state['root_states'][0, 3] = math.cos(yaw_offset / 2)
    state['root_states'][0, 6] = math.sin(yaw_offset / 2)
    fsm.base_command_previous.zero_()
    for _ in range(5):
        fsm.command_advance_to_next_push_station(state, buffers)
        assert not buffers.base[0, :2].any()
    state['root_states'][0, 3] = 1.
    state['root_states'][0, 6] = 0.
    for _ in range(40):
        fsm.command_advance_to_next_push_station(state, buffers)
        assert not buffers.base.any()
    assert not fsm.push_reengage_reachable.item()
    assert fsm.base_arrival_confirmed.item()
    assert fsm.push_recovery_settle_steps.item() >= 10
