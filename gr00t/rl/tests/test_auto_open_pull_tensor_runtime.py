"""Device-only pull runtime contracts; no simulator imports or launches."""

import inspect
import math
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.panel_motion import PullPanelConfig
from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory
from gr00t.rl.auto_open.target_noise import EpisodeTargetNoise
from gr00t.rl.auto_open.tensor_fsm import (
    PaddedPathBank, PullTensorHistory, TensorDoorFSM, _pull_cycle_uniform,
    _pull_station_history_record,
)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def device(request):
    return request.param


def test_tensor_install_sparse_ragged_and_empty(device):
    bank = PaddedPathBank(4, 7, device)
    rows = torch.tensor([3, 1], device=device)
    positions = torch.arange(30, device=device, dtype=torch.float32).reshape(2, 5, 3)
    orientations = torch.zeros(2, 5, 4, device=device)
    orientations[..., 0] = 1.
    lengths = torch.tensor([2, 5], device=device)
    bank.install_tensor(rows, positions, orientations, lengths)
    torch.testing.assert_close(bank.length, torch.tensor([0, 5, 0, 2], device=device, dtype=torch.int32))
    bank.index[rows] = 6
    actual, _ = bank.current(rows, advance=True)
    torch.testing.assert_close(actual, positions[torch.arange(2, device=device), lengths - 1])
    assert bank.complete(rows).all()
    bank.install_tensor(rows[:0], positions[:0], orientations[:0], lengths[:0])
    with pytest.raises(ValueError, match="orientations/lengths"):
        bank.install_tensor(rows, positions, orientations[:, :2], lengths)


def test_follow_station_entry_reads_environment_arm_indices(device):
    from gr00t.rl.scripts.doorman_auto_open import Stage
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device, fsm.Stage = device, Stage
    fsm.env = SimpleNamespace(arm_joint_ids=torch.arange(6, device=device))
    fsm.front_shoulder_body_indices = torch.tensor([0, 1], device=device)
    fsm.hinge_idx = 0
    fsm.hinge_axis_sign = torch.ones(2, device=device)
    fsm.door_normal = torch.tensor([[1., 0., 0.]], device=device).repeat(2, 1)
    fsm.hinge_world = torch.zeros(2, 3, device=device)
    fsm.pull_shoulder_anchor = torch.zeros(2, 2, device=device)
    fsm.pull_shoulder_local = torch.zeros(2, 2, device=device)
    fsm.pull_panel_side = torch.zeros(2, device=device)
    fsm.pull_door_opened = torch.zeros(2, dtype=torch.bool, device=device)
    fsm.align_panel_arm_joint_target = torch.zeros(2, 6, device=device)
    fsm.pull_base_path_length = torch.zeros(2, dtype=torch.int32, device=device)
    fsm.pull_base_path_index = torch.zeros_like(fsm.pull_base_path_length)
    fsm.pull_base_path = torch.zeros(2, 2, 3, device=device)
    fsm.pull_base_trajectory = BasePoseTrajectory(2, device)
    fsm.pull_follow_holding = torch.zeros(2, dtype=torch.bool, device=device)
    fsm.pull_follow_beta = torch.zeros(2, device=device)
    fsm.pull_follow_velocity = torch.zeros(2, 3, device=device)
    fsm.pull_base_points = torch.tensor([[0., -.15, 0.], [0., .15, 0.]], device=device)
    fsm.pull_wall_front = torch.zeros(2, device=device)
    fsm.dt = .02
    fsm.pull_station_xy = torch.zeros(2, 2, device=device)
    fsm.pull_station_yaw = torch.zeros(2, device=device)
    fsm.pending_station_distance = torch.zeros(2, device=device)
    fsm.pending_yaw_error = torch.zeros(2, device=device)
    fsm.pull_free_direction = torch.tensor([[0., 1.]], device=device).repeat(2, 1)
    fsm.door_forward = torch.tensor([[-1., 0., 0.]], device=device).repeat(2, 1)
    fsm.initial_door_root = torch.zeros(2, 3, device=device)
    fsm.push_shoulder_collision_radius = torch.full((2,), .155, device=device)
    fsm.push_shoulder_safety_margin = torch.full((2,), .02, device=device)
    fsm.panel_half_thickness = torch.full((2,), .02, device=device)
    fsm.pull_station_config = SimpleNamespace(panel_side_clearance_m=.10, shoulder_hinge_radius_m=(.40, .60))
    fsm.pull_panel_config = PullPanelConfig()
    fsm.stage_steps = torch.zeros_like(fsm.pull_base_path_length)
    fsm.reset_base_arrival = lambda rows, profile: None
    fsm.transition = lambda rows, stage: None
    fsm.record_acceptance = lambda *args, **kwargs: None
    def reject_failures(rows, code):
        assert not len(rows)
    fsm.fail = reject_failures
    root = torch.tensor([[.8, 1.5, .5, 1., 0., 0., 0.]], device=device).repeat(2, 1)
    bodies = root[:, None].repeat(1, 2, 1)
    bodies[:, :, 0] += .3
    bodies[:, 0, 1] += .15
    bodies[:, 1, 1] -= .15
    state = dict(root_states=root, robot_body_state_w=bodies,
        door_joint_position=torch.full((2, 1), .6, device=device),
        dof_pos=torch.arange(12, device=device, dtype=torch.float32).reshape(2, 6))
    fsm.begin_pull_follow_station(state, torch.tensor([1, 0], device=device))
    assert torch.isinf(fsm.pending_station_distance).all()
    assert torch.isinf(fsm.pending_yaw_error).all()
    torch.testing.assert_close(fsm.align_panel_arm_joint_target, state['dof_pos'])
    assert (fsm.pull_base_trajectory.length > 1).all()
    assert torch.isfinite(fsm.pull_shoulder_local).all()
    planned = fsm.pull_base_trajectory.poses.clone()
    state['door_joint_position'] += .1
    fsm.update_pull_panel_station(state, torch.tensor([0, 1], device=device))
    assert torch.equal(fsm.pull_base_trajectory.poses, planned)


def test_follow_reference_gate_is_monotonic_and_uses_only_the_door_angle(device):
    from gr00t.rl.scripts.doorman_auto_open import Stage
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device, fsm.Stage = device, Stage
    fsm.hinge_idx = 0
    fsm.pull_panel_config = PullPanelConfig()
    fsm.pull_door_opened = torch.zeros(1, dtype=torch.bool, device=device)
    fsm.pull_free_direction = torch.tensor([[0., 1.]], device=device)
    fsm.door_normal = torch.tensor([[1., 0., 0.]], device=device)
    fsm.pull_base_trajectory = BasePoseTrajectory(1, device, capacity=8)
    bank = fsm.pull_base_trajectory
    beta = torch.deg2rad(torch.tensor([20., 40., 60.], device=device))
    bank.length[:] = 3
    bank.poses[0, :3, 2] = torch.atan2(-beta.cos(), -beta.sin())
    bank.velocities[0, :2, 0] = .25
    fsm.pull_station_xy, fsm.pull_station_yaw = torch.zeros(1, 2, device=device), torch.zeros(1, device=device)
    fsm.pull_follow_holding = torch.zeros(1, dtype=torch.bool, device=device)
    fsm.pull_follow_beta = torch.zeros(1, device=device)
    fsm.pull_follow_velocity = torch.zeros(1, 3, device=device)
    for door_angle, expected_index, expected_beta, held in (
        (35., 1, 20., False), (35., 1, 40., True),
        (25., 1, 40., True), (50., 2, 40., False), (50., 2, 60., True)):
        state = {'door_joint_position': torch.tensor([[math.radians(door_angle)]], device=device)}
        fsm.update_pull_panel_station(state, torch.tensor([0], device=device))
        assert bank.index.item() == expected_index
        assert math.degrees(fsm.pull_follow_beta.item()) == pytest.approx(expected_beta, abs=1.e-4)
        assert fsm.pull_follow_holding.item() == held
        if held:
            assert not fsm.pull_follow_velocity.any()


def test_push_retract_preserves_push_station_controller(device):
    from gr00t.rl.scripts.doorman_auto_open import Stage
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device, fsm.Stage, fsm.pull_door = device, Stage, False
    rows = torch.tensor([0, 1], device=device)
    quat = torch.tensor([[1., 0., 0., 0.]], device=device).repeat(2, 1)
    zero = torch.zeros(2, 3, device=device)
    fsm.env = SimpleNamespace(arm_joint_ids=torch.arange(6, device=device),
        get_goal_reference_origin=lambda: zero, get_goal_reference_quat=lambda: quat,
        get_goal_center_offset_local=lambda: zero)
    fsm.stage = torch.full((2,), int(Stage.PUSH_ARM_RETRACT), device=device)
    fsm.stage_steps = torch.ones(2, dtype=torch.int32, device=device)
    fsm.active_arm_retract_pose_local = torch.zeros(2, 6, device=device)
    fsm.push_recovery_settle_steps = torch.full((2,), 9, device=device)
    fsm.base_advance_arm_joint_target = torch.zeros(2, 6, device=device)
    fsm.push_reengage_reachable = torch.ones(2, dtype=torch.bool, device=device)
    fsm.base_advance_target_longitudinal = torch.zeros(2, device=device)
    fsm._world_orientation_from_local_rpy = lambda rows, rpy: quat[rows]
    fsm.record_acceptance = lambda *args, **kwargs: None
    fsm.transition = lambda rows, stage: fsm.stage.index_fill_(0, rows, int(stage))
    followed = []
    def follow(state, rows):
        followed.append(rows.clone())
        fsm.transition(rows, Stage.PULL_BASE_ADVANCE)
    fsm.begin_pull_follow_station = follow
    state = dict(ee_pos=zero, ee_orn=quat, ee_vel=zero, dof_pos=torch.ones(2, 6, device=device))
    torch.testing.assert_close(fsm.advance_push_retract(state), rows)
    assert (fsm.stage == int(Stage.BASE_ADVANCE)).all()
    assert not followed


def test_shared_panel_clearance_uses_selected_contact_side(device):
    from gr00t.rl.auto_open.geometry import _batched_panel_finger_clearance_gpu
    bodies = torch.zeros(2, 2, 13, device=device)
    bodies[:, :, 3] = 1.
    bodies[0, :, 0] = torch.tensor([.05, .06], device=device)
    bodies[1, :, 0] = torch.tensor([-.05, -.06], device=device)
    state = dict(robot_body_state_w=bodies, door_body_state_w=torch.zeros_like(bodies),
        door_joint_position=torch.zeros(2, 1, device=device))
    cache = dict(fixed_body_idx=0, moving_body_idx=1, panel_body_idx=0, hinge_idx=0,
        fixed_points=torch.zeros(1, 3, device=device), moving_points=torch.zeros(1, 3, device=device),
        hinge_axis_sign=torch.ones(2, device=device), panel_half_thickness=torch.full((2,), .02, device=device),
        door_normal=torch.tensor([[1., 0., 0.]], device=device).repeat(2, 1))
    rows = torch.tensor([0, 1], device=device)
    clearance = _batched_panel_finger_clearance_gpu(state, cache, rows,
        normal_sign=torch.tensor([1., -1.], device=device))
    torch.testing.assert_close(clearance, torch.full((2,), .03, device=device))
    baseline = _batched_panel_finger_clearance_gpu(state, cache, rows[:1])
    torch.testing.assert_close(baseline, clearance[:1], rtol=0., atol=0.)


def test_push_angle_guard_remains_active(device):
    from gr00t.rl.scripts.doorman_auto_open import Stage
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device, fsm.Stage, fsm.pull_door = device, Stage, False
    fsm.stage = torch.full((2,), int(Stage.PUSH_ARM_RETRACT), device=device)
    fsm.hinge_idx = 0
    fsm.pull_door_opened = torch.tensor([True, False], device=device)
    fsm.push_max_observed_angular_speed = torch.zeros(2, device=device)
    fsm.push_max_angular_speed = torch.full((2,), math.radians(30.), device=device)
    fsm.push_overspeed_steps = torch.zeros(2, dtype=torch.int32, device=device)
    failures = []
    def fail(rows, code):
        failures.extend((int(row), code) for row in rows)
        fsm.stage[rows] = int(Stage.FAILURE)
    fsm.fail = fail
    state = dict(door_joint_position=torch.tensor([[math.radians(95.)], [math.radians(80.)]], device=device),
                 door_joint_velocity=torch.zeros(2, 1, device=device))
    opened = fsm.advance_push_common(state)
    assert failures == [(0, 22)]
    assert opened.tolist() == []


def test_insert_retains_desired_endpoint_not_inverse_command(device):
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device = device
    fsm.pull_insert_path = PaddedPathBank(4, 7, device)
    for name, width in (("pull_insert_goal", 3), ("pull_insert_orientation", 4),
                        ("pull_insert_alignment_goal", 3), ("pull_insert_alignment_orientation", 4),
                        ("pull_face_offset_local", 3), ("pull_panel_contact", 3)):
        setattr(fsm, name, torch.zeros(4, width, device=device))
    fsm.pull_insert_aligned = torch.zeros(4, device=device, dtype=torch.bool)
    fsm.pull_cycle = torch.zeros(4, device=device, dtype=torch.int64)
    rows = torch.tensor([2, 0], device=device)
    position = torch.ones(2, 5, 3, device=device)
    orientation = torch.tensor([[1., 0., 0., 0.]], device=device).expand(2, -1)
    endpoint = torch.full((2, 3), 2., device=device)
    contact = torch.full((2, 3), 3., device=device)
    face = torch.full((2, 3), .02, device=device)
    fsm.install_pull_insert_tensor(rows, position, orientation[:, None].expand(-1, 5, -1),
        torch.tensor([2, 5], device=device), contact, orientation, face, endpoint=endpoint)
    torch.testing.assert_close(fsm.pull_insert_goal[rows], endpoint)
    torch.testing.assert_close(fsm.pull_panel_contact[rows], contact)
    torch.testing.assert_close(fsm.pull_face_offset_local[rows], face)


def test_stateless_candidates_are_batch_independent_and_in_range(device):
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device, fsm.num_envs = device, 3
    fsm.pull_panel_config = PullPanelConfig()
    controllers = [SimpleNamespace(target_noise=EpisodeTargetNoise(seed)) for seed in (19, 5, 19)]
    fsm.pull_cycle = torch.tensor([1, 0, 1], device=device, dtype=torch.int64)
    fsm.prepare_pull_cycle_samples(controllers)
    rows = torch.tensor([2, 0], device=device)
    samples = fsm.sample_pull_cycle_candidates(rows)
    torch.testing.assert_close(samples[0], samples[1], rtol=0., atol=0.)
    torch.testing.assert_close(fsm.pull_cycle, torch.tensor([2, 0, 2], device=device, dtype=torch.int64))
    assert (samples >= fsm.pull_cycle_sample_lower).all()
    assert (samples <= fsm.pull_cycle_sample_lower + fsm.pull_cycle_sample_span).all()
    assert not torch.equal(samples[:, 0], samples[:, 1])
    assert samples.shape == (2, 27, 4)
    assert torch.count_nonzero(samples[..., 2]) == 0
    expected_grid = torch.cartesian_prod(
        torch.tensor([.8, .85, .9, .95, 1., 1.05, 1.1, 1.15, 1.2], device=device),
        torch.tensor([.02, .04, .06], device=device))
    distances = (samples[0, :, None, :2] - expected_grid[None]).abs().amax(-1)
    assert (distances.amin(0) < 1.e-6).all()
    assert (distances.amin(1) < 1.e-6).all()
    torch.testing.assert_close(samples[0, 0, :2], torch.tensor([1., .04], device=device))
    fsm.pull_cycle[0] = 1
    single = fsm.sample_pull_cycle_candidates(rows[1:])
    torch.testing.assert_close(single[0], samples[1], rtol=0., atol=0.)
    next_cycle = fsm.sample_pull_cycle_candidates(rows[1:])
    assert torch.count_nonzero(next_cycle[..., 2]) == 0
    fsm.pull_cycle[0] = 2 ** 40
    assert torch.isfinite(fsm.sample_pull_cycle_candidates(rows[1:])).all()
    assert fsm.sample_pull_cycle_candidates(rows[:0]).shape == (0, 27, 4)


def test_first_insert_advances_precomputed_approach_without_point_hold(device):
    from gr00t.rl.scripts.doorman_auto_open import Stage
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device, fsm.Stage, fsm.pull_door = device, Stage, True
    fsm.stage = torch.full((3,), int(Stage.PULL_INSERT), device=device)
    fsm.pull_cycle = torch.tensor([1, 2, 1], device=device)
    fsm.pull_insert_path = PaddedPathBank(3, 3, device)
    for name, width in (("pull_insert_goal", 3), ("pull_insert_orientation", 4),
                        ("pull_insert_alignment_goal", 3), ("pull_insert_alignment_orientation", 4),
                        ("pull_face_offset_local", 3), ("pull_panel_contact", 3),
                        ("ee_command_world", 3), ("ee_orientation_command", 4)):
        setattr(fsm, name, torch.zeros(3, width, device=device))
    rows = torch.arange(3, device=device)
    positions = torch.tensor([[[1., 0., 1.], [.98, 0., 1.], [.96, 0., 1.]]], device=device).repeat(3, 1, 1)
    orientation = torch.tensor([[1., 0., 0., 0.]], device=device).repeat(3, 1)
    fsm.install_pull_insert_tensor(rows, positions, orientation[:, None].expand(-1, 3, -1),
        torch.full((3,), 3, device=device), positions[:, -1], orientation,
        torch.zeros(3, 3, device=device))
    fsm._base_hold_command = lambda state, rows: (torch.zeros(len(rows), 3, device=device),)
    fsm._set_cartesian_targets = lambda *args: None
    fsm.q_closed = torch.zeros(3, device=device)
    buffers = SimpleNamespace(base=torch.zeros(3, 3, device=device), gripper=torch.zeros(3, 1, device=device))
    state = dict(ee_pos=torch.zeros(3, 3, device=device), ee_orn=orientation.clone())
    state['ee_pos'][2] = positions[2, 0]
    state['ee_orn'][2] = torch.tensor([0., 1., 0., 0.], device=device)
    fsm.command_pull_insert(state, buffers)
    assert fsm.pull_insert_path.index.tolist() == [1, 1, 1]
    torch.testing.assert_close(fsm.ee_command_world, positions[:, 0])
    state['ee_pos'][0] = positions[0, 0]
    state['ee_orn'][2] = orientation[2]
    fsm.command_pull_insert(state, buffers)
    assert fsm.pull_insert_path.index.tolist() == [2, 2, 2]
    state['ee_pos'].zero_()
    fsm.command_pull_insert(state, buffers)
    assert fsm.pull_insert_path.index.tolist() == [2, 2, 2]
    torch.testing.assert_close(fsm.ee_command_world, positions[:, 2])


def test_hash_stream_matches_cpu_at_integer_limits(device):
    seeds = torch.tensor([0, 2 ** 32 - 1, 2 ** 63 - 1], dtype=torch.int64)
    cycles = torch.tensor([2 ** 32 + 1, 2 ** 40, 2 ** 63 - 1], dtype=torch.int64)
    expected = _pull_cycle_uniform(seeds, cycles, 8)
    actual = _pull_cycle_uniform(seeds.to(device), cycles.to(device), 8)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0., atol=0.)
    assert (actual >= 0).all() and (actual < 1).all()


def test_legacy_list_installation_apis_are_removed():
    assert not hasattr(PaddedPathBank, "install")
    for name in ("install_pregrasp_plan", "install_center_plan", "install_release_open_plan",
                 "install_retreat_plan", "install_restore_plan", "install_contact_plan",
                 "install_pull_insert_plan", "install_push_retract_pose"):
        assert not hasattr(TensorDoorFSM, name)


def test_history_snapshots_online_and_materializes_only_on_access(device, monkeypatch):
    history = PullTensorHistory(3)
    rows = torch.tensor([2, 0], device=device)
    value = torch.tensor([[1., 2.], [3., 4.]], device=device)
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError("online history copied data to the host")
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        history.append_batch(rows, {"samples": value, "nested": {"valid": value > 0}})
    value.zero_()
    rows.zero_()
    assert history[2] == [{"samples": [1., 2.], "nested": {"valid": [True, True]}}]
    assert history[1] == []


def test_station_history_output_schema():
    values = list(range(26))
    values[8] = 1
    record = _pull_station_history_record(values)
    assert record["waypoints_xy_yaw"] == [[16, 17, 18]]
    assert record["robot_envelope_min_local_m"] == [10, 11, 12]
    assert record["sampled_station_angle_noise_rad"] == 22
    assert record["tracking_margin_m"] == 25


def test_pull_runtime_has_no_host_materialization():
    for method in (TensorDoorFSM.begin_pull_clear_sweep, TensorDoorFSM.begin_pull_follow_station,
                   TensorDoorFSM.sample_pull_cycle_candidates, TensorDoorFSM.install_pull_insert_tensor,
                   TensorDoorFSM.begin_pull_open):
        source = inspect.getsource(method)
        assert ".cpu(" not in source
        assert ".tolist(" not in source
        assert ".item(" not in source
