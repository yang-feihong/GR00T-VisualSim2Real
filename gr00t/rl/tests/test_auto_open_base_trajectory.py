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


def test_through_route_aligns_outside_frame_before_crossing():
    start = torch.tensor([[0.4, -0.7, 2.0]])
    route = frame_traversal_waypoints(
        start,
        torch.zeros(1, 2),
        torch.tensor([[-1.0, 0.0]]),
        torch.tensor([[-1.4, 0.0, math.pi]]),
        torch.tensor([0.7]),
    )
    torch.testing.assert_close(route[0, 0], torch.tensor([0.75, -0.7, 2.0]))
    torch.testing.assert_close(route[0, 1], torch.tensor([0.75, 0.0, math.pi]))
    assert route[0, 2, 0] < 0.0


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
