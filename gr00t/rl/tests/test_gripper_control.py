import math

import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.envs.manip_loco.gripper_control import (
    advance_gripper_actuator_targets,
)


def _advance(semantic, actual, actuator, hold, forces):
    return advance_gripper_actuator_targets(
        torch.tensor(semantic, dtype=torch.float32).reshape(-1, 1),
        torch.tensor(actual, dtype=torch.float32).reshape(-1, 1),
        torch.tensor(actuator, dtype=torch.float32).reshape(-1, 1),
        torch.tensor(hold, dtype=torch.bool),
        torch.tensor(forces, dtype=torch.float32),
    )


def test_open_command_is_applied_directly_and_clears_hold():
    target, hold = _advance([-0.7], [-0.2], [-0.2], [True], [[8.0, 9.0]])
    torch.testing.assert_close(target[:, 0], torch.tensor([-0.7]))
    assert hold.tolist() == [False]


def test_free_close_slews_without_rewriting_semantic_command():
    semantic = torch.zeros(1, 1)
    target, hold = advance_gripper_actuator_targets(
        semantic,
        torch.tensor([[-0.90]]),
        torch.tensor([[-1.00]]),
        torch.tensor([False]),
        torch.zeros(1, 2),
    )
    torch.testing.assert_close(target[:, 0], torch.tensor([-0.984]))
    torch.testing.assert_close(semantic, torch.zeros_like(semantic))
    assert hold.tolist() == [False]


def test_unilateral_contact_slows_closure():
    target, hold = _advance([0.0], [-0.50], [-0.52], [False], [[2.0, 0.0]])
    torch.testing.assert_close(target[:, 0], torch.tensor([-0.512]))
    assert hold.tolist() == [False]


def test_bilateral_contact_latches_actual_position():
    target, hold = _advance([0.0], [-0.2397], [-0.30], [False], [[2.0, 3.0]])
    torch.testing.assert_close(target[:, 0], torch.tensor([-0.2397]))
    assert hold.tolist() == [True]


def test_contact_loss_resumes_gradual_closure():
    target, hold = _advance([0.0], [-0.24], [-0.24], [True], [[0.0, 0.0]])
    torch.testing.assert_close(target[:, 0], torch.tensor([-0.224]))
    assert hold.tolist() == [False]


def test_overload_releases_actuator_without_changing_semantic_command():
    semantic = torch.zeros(1, 1)
    target, hold = advance_gripper_actuator_targets(
        semantic,
        torch.tensor([[-0.24]]),
        torch.tensor([[-0.24]]),
        torch.tensor([True]),
        torch.tensor([[51.0, 4.0]]),
    )
    torch.testing.assert_close(target[:, 0], torch.tensor([-0.241]))
    torch.testing.assert_close(semantic, torch.zeros_like(semantic))
    assert hold.tolist() == [True]
    assert target.item() >= -math.pi / 3.0


def test_rows_transition_independently_in_one_batch():
    target, hold = _advance(
        [-0.8, 0.0, 0.0],
        [-0.3, -0.5, -0.2],
        [-0.3, -0.6, -0.3],
        [True, False, False],
        [[4.0, 4.0], [0.0, 0.0], [3.0, 3.0]],
    )
    torch.testing.assert_close(target[:, 0], torch.tensor([-0.8, -0.584, -0.2]))
    assert hold.tolist() == [False, False, True]
