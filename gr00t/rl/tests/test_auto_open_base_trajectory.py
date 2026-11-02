import math
import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.base_trajectory import (
    BasePoseTrajectory,
    frame_traversal_waypoints,
    wrapped_angle,
    panel_station_waypoints,
)
from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM
from gr00t.rl.auto_open.motion_math import track_base_pose


DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_forward_route_allows_early_stop_and_endpoint_updates(device):
    bank = BasePoseTrajectory(1, device)
    rows = torch.tensor([0], device=device)
    start = torch.tensor([[0., .4, 0.]], device=device)
    forward = torch.tensor([[1., 0.]], device=device)
    route = panel_station_waypoints(
        start, torch.zeros(1, 2, device=device), forward,
        torch.tensor([[2., 0.]], device=device), torch.zeros(1, device=device),
    )
    bank.install(rows, start, route, .02)
    prefix = torch.tensor([80], device=device)
    target = torch.tensor([[.3, 0.]], device=device)
    bank.index[:] = 140
    pose, velocity, prefix_done, at_target = bank.sample_forward_target(rows, target, forward, prefix)
    assert prefix_done.all() and at_target.all()
    torch.testing.assert_close(pose[:, :2], target)
    assert not velocity.any()
    assert bank.index.item() == 140 < bank.length.item() - 1
    target[:, 0] = .6
    previous = pose.clone()
    pose, velocity, _, at_target = bank.sample_forward_target(rows, target, forward, prefix)
    assert not at_target.any()
    assert bank.index.item() == 141
    assert (pose[:, :2] - previous[:, :2]).abs().max() <= .005001
    assert velocity[0, 0] > 0


@pytest.mark.parametrize("device", DEVICES)
def test_forward_target_does_not_clip_or_stop_lateral_prefix(device):
    bank = BasePoseTrajectory(1, device)
    rows = torch.tensor([0], device=device)
    start = torch.tensor([[0., .4, 0.]], device=device)
    forward = torch.tensor([[1., 0.]], device=device)
    route = panel_station_waypoints(
        start, torch.zeros(1, 2, device=device), forward,
        torch.tensor([[2., 0.]], device=device), torch.zeros(1, device=device),
    )
    bank.install(rows, start, route, .02)
    _, velocity, prefix_done, at_target = bank.sample_forward_target(
        rows, torch.zeros(1, 2, device=device), forward, torch.tensor([80], device=device)
    )
    assert not prefix_done.any() and not at_target.any()
    assert velocity[0, 1] < 0 and bank.index.item() == 1


@pytest.mark.parametrize("device", DEVICES)
def test_runtime_progress_counter_matches_trajectory_index_dtype(device):
    owner = SimpleNamespace(
        stage_steps=torch.zeros(2, dtype=torch.int32, device=device),
        through_trajectory=BasePoseTrajectory(2, device),
    )
    source = ast.parse(textwrap.dedent(inspect.getsource(TensorDoorFSM.__init__)))
    assignment = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "through_path_previous_index"
            for target in node.targets
        )
    )
    counter = eval(
        compile(ast.Expression(assignment.value), "<runtime-init>", "eval"),
        {"torch": torch, "self": owner},
    )
    rows = torch.tensor([1, 0], device=device)
    counter[rows] = owner.through_trajectory.index[rows]
    assert counter.dtype == owner.through_trajectory.index.dtype


@pytest.mark.parametrize("device", DEVICES)
def test_speed_limits_shortest_yaw_and_batch_independence(device):
    start = torch.tensor([[1.0, 2.0, math.radians(179.0)], [0.0, 0.0, 0.0]], device=device)
    waypoints = torch.tensor(
        [
            [[0.8, 2.0, math.radians(179.0)], [2.0, 3.0, math.radians(-179.0)]],
            [[-0.2, 0.0, 0.0], [0.5, 0.5, 1.0]],
        ],
        device=device,
    )
    bank = BasePoseTrajectory(2, device)
    bank.install(torch.arange(2, device=device), start, waypoints, 0.02)
    for row in range(2):
        n = bank.length[row]
        delta = bank.poses[row, 1:n] - bank.poses[row, : n - 1]
        assert torch.linalg.vector_norm(delta[:, :2], dim=-1).max() <= 0.25 * 0.02 + 1e-6
        assert wrapped_angle(delta[:, 2]).abs().max() <= 0.40 * 0.02 + 1e-6
        torch.testing.assert_close(bank.poses[row, n - 1], waypoints[row, -1], atol=0, rtol=0)
        assert not bank.velocities[row, n - 1 :].any()
        single = BasePoseTrajectory(1, device)
        single.install(
            torch.tensor([0], device=device), start[row : row + 1], waypoints[row : row + 1], 0.02
        )
        assert torch.equal(bank.poses[row], single.poses[0])
        assert torch.equal(bank.velocities[row], single.velocities[0])


def test_sampling_advances_every_frame_and_holds_the_endpoint():
    bank = BasePoseTrajectory(1, "cpu", capacity=512)
    bank.install(
        torch.tensor([0]),
        torch.zeros(1, 3),
        torch.tensor([[[0.1, 0.0, 1.0], [1.0, 0.0, 1.0]]]),
        0.02,
    )
    for frame in range(20):
        pose, _, _ = bank.sample(torch.tensor([0]))
        torch.testing.assert_close(pose, bank.poses[:, frame])
        assert bank.index.item() == frame + 1
    bank.index[:] = bank.length - 1
    pose, velocity, complete = bank.sample(torch.tensor([0]))
    assert complete.item() and not velocity.any()
    torch.testing.assert_close(pose, torch.tensor([[1.0, 0.0, 1.0]]))


def test_through_route_uses_actual_depth_without_retreating():
    start = torch.tensor([[0.4, -0.7, 2.0]])
    route = frame_traversal_waypoints(
        start,
        torch.zeros(1, 2),
        torch.tensor([[-1.0, 0.0]]),
        torch.tensor([[-1.4, 0.0, math.pi]]),
    )
    torch.testing.assert_close(route[0, 0], torch.tensor([0.4, 0.0, math.pi]))
    assert route[0, 1, 0] < 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_panel_station_is_one_lateral_then_forward_trajectory(device):
    start = torch.tensor([[-1., .4, 0.], [1., -.4, math.pi]], device=device)
    forward = torch.tensor([[1., 0.], [-1., 0.]], device=device)
    finish = torch.tensor([[-.4, 0.], [.4, 0.]], device=device)
    yaw = torch.tensor([0., math.pi], device=device)
    route = panel_station_waypoints(start, torch.zeros(2, 2, device=device), forward, finish, yaw)
    torch.testing.assert_close(route[:, 0, :2], torch.tensor([[-1., 0.], [1., 0.]], device=device))
    bank = BasePoseTrajectory(2, device)
    rows = torch.arange(2, device=device)
    bank.install(rows, start, route, .02)
    torch.testing.assert_close(bank.poses[:, 0, :2], start[:, :2])
    torch.testing.assert_close(wrapped_angle(bank.poses[:, 0, 2] - start[:, 2]), torch.zeros(2, device=device))
    lateral_end = torch.ceil(torch.tensor(.4 / .25 / .02, device=device)).long()
    bank.index[:] = lateral_end
    pose, velocity, complete = bank.sample(rows)
    assert not complete.any()
    assert (velocity[:, :2] * forward).sum(-1).min() > 0
    bank.index[:] = bank.length - 1
    pose, velocity, complete = bank.sample(rows)
    assert complete.all()
    torch.testing.assert_close(pose[:, :2], finish)


@pytest.mark.parametrize("device", DEVICES)
def test_through_reference_never_moves_backward_from_any_entry_depth(device):
    start = torch.tensor([[.2, .3, 2.], [-.2, -.3, -2.]], device=device)
    forward = torch.tensor([[-1., 0.], [1., 0.]], device=device)
    finish = torch.tensor([[-1.4, 0., math.pi], [1.4, 0., 0.]], device=device)
    route = frame_traversal_waypoints(start, torch.zeros(2, 2, device=device), forward, finish)
    bank = BasePoseTrajectory(2, device)
    bank.install(torch.arange(2, device=device), start, route, .02)
    delta = bank.poses[:, 1:, :2] - bank.poses[:, :-1, :2]
    assert (delta * forward[:, None]).sum(-1).min() >= -1e-7


def test_tracking_feedforward_preserves_nominal_velocity():
    root = torch.zeros(1, 13)
    root[:, 3] = 1.0
    root[:, 7] = 0.25
    root[:, 12] = 0.4
    command, _, _ = track_base_pose(
        root,
        torch.zeros(1, 2),
        torch.zeros(1),
        torch.zeros(1, 2),
        torch.zeros(1),
        minimum_speed_toward_target_only=False,
        reference_velocity_world=torch.tensor([[0.25, 0.0, 0.4]]),
    )
    torch.testing.assert_close(command, torch.tensor([[0.25, 0.0, 0.4]]))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pull_door", [False, True])
def test_through_builds_start_only_when_motion_is_released(device, pull_door):
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.pull_door = pull_door
    fsm.device = torch.device(device)
    fsm.Stage = SimpleNamespace(PASS_THROUGH_DOORWAY=1, SUCCESS=2)
    fsm.stage = torch.ones(1, dtype=torch.int64, device=device)
    fsm.through_trajectory = BasePoseTrajectory(1, device)
    fsm.through_motion_ready = torch.zeros(1, dtype=torch.bool, device=device)
    fsm.through_stop_latched = torch.zeros_like(fsm.through_motion_ready)
    fsm.through_command_xy = torch.tensor([[-1.4, 0.]], device=device)
    fsm.through_command_yaw = torch.tensor([math.pi], device=device)
    fsm.through_station_xy = fsm.through_command_xy.clone()
    fsm.through_station_yaw = fsm.through_command_yaw.clone()
    fsm.initial_door_root = torch.zeros(1, 3, device=device)
    fsm.door_forward = torch.tensor([[-1., 0., 0.]], device=device)
    fsm.dt = .02
    fsm._tail_clearance = lambda state, rows: torch.zeros(len(rows), device=device)
    fsm.base_target = lambda *args, **kwargs: (None, torch.ones(1, device=device), torch.zeros(1, device=device))
    arm_targets = []
    fsm.arm_joint_target = lambda rows, target, buffers: arm_targets.append(target.clone())
    fsm.gripper_target = lambda *args: None
    fsm.middle_arm_joint_target = torch.zeros(1, 6, device=device)
    fsm.through_arm_hold_joint_target = torch.zeros_like(fsm.middle_arm_joint_target)
    fsm.through_arm_restore_ready = torch.zeros(1, dtype=torch.bool, device=device)
    fsm.through_arm_carry_initialized = torch.zeros(1, dtype=torch.bool, device=device)
    fsm.through_arm_carry_ready = torch.zeros(1, dtype=torch.bool, device=device)
    fsm.env = SimpleNamespace(arm_joint_ids=torch.arange(6, device=device))
    fsm.q_closed = torch.zeros(1, device=device)
    fsm.pending_station_distance = torch.zeros(1, device=device)
    fsm.pending_yaw_error = torch.zeros(1, device=device)
    root = torch.zeros(1, 13, device=device)
    root[:, :4] = torch.tensor([[.2, .3, .5, 1.]], device=device)
    state = {"root_states": root, "dof_pos": torch.arange(6, device=device)[None].float()*.1+.3}
    fsm.command_pass_through_doorway(state, SimpleNamespace())
    assert fsm.through_trajectory.length.item() == 0
    assert fsm.through_trajectory.index.item() == 0
    root[:, :2] = torch.tensor([[.25, .2]], device=device)
    fsm.through_motion_ready[:] = True
    if pull_door:
        fsm.command_pass_through_doorway(state, SimpleNamespace())
        assert fsm.through_trajectory.length.item() == 0
        state["dof_pos"][:] = fsm.through_arm_hold_joint_target
    fsm.command_pass_through_doorway(state, SimpleNamespace())
    torch.testing.assert_close(fsm.through_trajectory.poses[:, 0, :2], root[:, :2])
    assert fsm.through_trajectory.index.item() == 1
    expected = state["dof_pos"].clone() if pull_door else fsm.middle_arm_joint_target
    torch.testing.assert_close(arm_targets[-1], expected)
    state["dof_pos"] += .05
    fsm.command_pass_through_doorway(state, SimpleNamespace())
    torch.testing.assert_close(arm_targets[-1], expected)
    fsm._tail_clearance = lambda state, rows: torch.full((len(rows),), 1.01, device=device)
    fsm.command_pass_through_doorway(state, SimpleNamespace())
    torch.testing.assert_close(arm_targets[-1], fsm.middle_arm_joint_target)
    fsm._tail_clearance = lambda state, rows: torch.full((len(rows),), .94, device=device)
    fsm.command_pass_through_doorway(state, SimpleNamespace())
    torch.testing.assert_close(arm_targets[-1], fsm.middle_arm_joint_target)
