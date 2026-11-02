"""Basket admission uses the shared pose window, not instantaneous twists."""

import os

import pytest
import torch

from gr00t.rl.auto_open.acceptance import TensorStabilityWindow
from gr00t.rl.pick.task_initialization import payload_pose_stability


def replay(states, device):
    states = {name: value.to(device) for name, value in states.items()}
    count, frames = states["root_states"].shape[:2]
    slots = states["distractor_states"].shape[2] + 1
    window = TensorStabilityWindow(count * slots, device)
    for frame in range(frames):
        result = payload_pose_stability(
            window, {name: value[:, frame] for name, value in states.items()}
        )
    return result


def test_pose_stability_rejects_rotation_and_ignores_common_base_translation():
    root = torch.zeros(3, 30, 13)
    root[..., 3] = 1
    root[0, :, 0] = torch.arange(30) * 0.01
    target = root.clone()
    target[..., 2] += 0.1
    target[1, :, 0] += torch.arange(30) * 0.002
    angle = torch.arange(30) * 0.01
    target[2, :, 3] = torch.cos(angle / 2)
    target[2, :, 6] = torch.sin(angle / 2)
    stable, _, _ = replay(
        {
            "root_states": root,
            "object_state": target,
            "distractor_states": torch.empty(3, 30, 0, 13),
        },
        "cpu",
    )
    assert stable[:, 0].tolist() == [True, False, False]


@pytest.mark.skipif(
    not os.environ.get("PICK_SETTLE_TRACE"), reason="requires recorded physical probe"
)
def test_recorded_pose_stability_is_batch_independent():
    trace = torch.load(os.environ["PICK_SETTLE_TRACE"], weights_only=True)
    states = trace["states"]
    actual = replay(states, "cuda:0")
    permutation = torch.arange(len(trace["seeds"]) - 1, -1, -1)
    reverse = replay({name: value[permutation] for name, value in states.items()}, "cuda:0")
    for left, right in zip(actual, reverse):
        torch.testing.assert_close(left[permutation], right, rtol=0, atol=0)
    for row in (0, 1, 42, 99):
        single = replay({name: value[row : row + 1] for name, value in states.items()}, "cuda:0")
        for left, right in zip(actual, single):
            torch.testing.assert_close(left[row : row + 1], right, rtol=0, atol=0)
    slots = actual[0].shape[1]
    present = (
        torch.arange(slots, device="cuda:0")[None]
        < torch.tensor(trace["payload_counts"], device="cuda:0")[:, None]
    )
    passed = (actual[0] | ~present).all(-1)
    print(
        {
            "recorded_rows": len(passed),
            "pose_stable": int(passed.sum()),
            "pose_unstable": int((~passed).sum()),
        }
    )
    assert passed.sum() >= 90
    assert (~passed).any()


@pytest.mark.skipif(not os.environ.get("PICK_SETTLE_TRACE"), reason="requires recorded physical probe")
def test_recorded_first_stable_opportunity():
    from gr00t.rl.auto_open.acceptance import update_robot_stability

    trace = torch.load(os.environ["PICK_SETTLE_TRACE"], weights_only=True)
    states = {key: value.cuda() for key, value in trace["states"].items()}
    n, frames = states["root_states"].shape[:2]
    slots = states["distractor_states"].shape[2] + 1
    present = torch.arange(slots, device="cuda")[None] < torch.tensor(trace["payload_counts"], device="cuda")[:, None]
    window = TensorStabilityWindow(n * slots, "cuda")
    base_steps = torch.zeros(n, device="cuda", dtype=torch.int32)
    first = torch.full((n,), -1, device="cuda", dtype=torch.long)
    for frame in range(frames):
        state = {key: value[:, frame] for key, value in states.items()}
        stable, _, _ = payload_pose_stability(window, state)
        base, base_steps, _ = update_robot_stability(
            state["root_states"], base_steps, required_frames=25,
            max_linear_speed_m_s=0.10, max_angular_speed_rad_s=0.35,
            min_base_height_m=0.45, min_upright_cos=0.95,
        )
        ready = base & (stable | ~present).all(-1)
        first[(first < 0) & ready] = frame
    print({"ever_stable": int((first >= 0).sum()), "never_stable_seeds": [seed for seed, value in zip(trace["seeds"], first.tolist()) if value < 0], "latest_first_stable_frame": int(first.max())})
