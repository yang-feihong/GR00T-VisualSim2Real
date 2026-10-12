import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.envs.manip_loco.arm_control import (
    advance_cartesian_joint_transition,
)


def test_active_cartesian_entry_is_limited_per_joint():
    raw = torch.tensor([[0.5, -0.707, 0.01]], dtype=torch.float32)
    previous = torch.zeros_like(raw)
    output, limited, active = advance_cartesian_joint_transition(
        raw, previous, torch.tensor([True]), torch.full((3,), 0.02)
    )
    torch.testing.assert_close(output, torch.tensor([[0.02, -0.02, 0.01]]))
    torch.testing.assert_close(limited, output)
    assert active.tolist() == [True]


def test_transition_retires_only_after_raw_target_is_reached():
    raw = torch.tensor([[0.03, -0.01]], dtype=torch.float32)
    output, limited, active = advance_cartesian_joint_transition(
        raw,
        torch.tensor([[0.02, 0.0]]),
        torch.tensor([True]),
        torch.full((2,), 0.02),
    )
    torch.testing.assert_close(output, raw)
    torch.testing.assert_close(limited, raw)
    assert active.tolist() == [False]


def test_scalar_and_batch_rows_use_identical_transition_math():
    raw = torch.tensor([[0.6, -0.7], [-0.4, 0.3]], dtype=torch.float32)
    previous = torch.tensor([[0.0, 0.0], [-0.1, 0.1]], dtype=torch.float32)
    max_step = torch.tensor([0.02, 0.02])
    batched = advance_cartesian_joint_transition(
        raw, previous, torch.tensor([True, True]), max_step
    )
    scalar_rows = [
        advance_cartesian_joint_transition(
            raw[index : index + 1],
            previous[index : index + 1],
            torch.tensor([True]),
            max_step,
        )
        for index in range(2)
    ]
    for output_index in range(3):
        expected = torch.cat([row[output_index] for row in scalar_rows], dim=0)
        torch.testing.assert_close(batched[output_index], expected)


def test_inactive_rows_receive_unmodified_ik_targets():
    raw = torch.tensor([[0.5, -0.7]], dtype=torch.float32)
    output, limited, active = advance_cartesian_joint_transition(
        raw,
        torch.zeros_like(raw),
        torch.tensor([False]),
        torch.full((2,), 0.02),
    )
    torch.testing.assert_close(output, raw)
    torch.testing.assert_close(limited, torch.zeros_like(raw))
    assert active.tolist() == [False]
