import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.envs.manip_loco import (
    gripper_control,
)

advance_gripper_actuator_targets = gripper_control.advance_gripper_actuator_targets


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


def test_high_contact_force_keeps_latched_actuator_target():
    semantic = torch.zeros(1, 1)
    target, hold = advance_gripper_actuator_targets(
        semantic,
        torch.tensor([[-0.23]]),
        torch.tensor([[-0.24]]),
        torch.tensor([True]),
        torch.tensor([[500.0, 400.0]]),
    )
    torch.testing.assert_close(target[:, 0], torch.tensor([-0.24]))
    torch.testing.assert_close(semantic, torch.zeros_like(semantic))
    assert hold.tolist() == [True]


def test_optional_held_preload_tracks_relaxation_without_reopening_or_rewriting_command():
    semantic = torch.zeros(1, 1)
    actual = torch.tensor([[-0.5735]])
    old = torch.tensor([[-0.5712]])
    target, hold = advance_gripper_actuator_targets(
        semantic,
        actual,
        old,
        torch.tensor([True]),
        torch.tensor([[10.0, 12.0]]),
        contact_command_lead=0.006,
        maintain_contact_preload=True,
    )
    torch.testing.assert_close(target, actual + 0.006)
    assert (target >= old).all()
    assert hold.all()
    torch.testing.assert_close(semantic, torch.zeros_like(semantic))
    unchanged, _ = advance_gripper_actuator_targets(
        semantic,
        actual,
        old,
        torch.tensor([True]),
        torch.tensor([[10.0, 12.0]]),
        contact_command_lead=0.006,
    )
    torch.testing.assert_close(unchanged, old)


def test_optional_contact_preload_is_retained_and_capped_at_contact_lead():
    target, hold = advance_gripper_actuator_targets(
        torch.zeros(1, 1),
        torch.tensor([[-0.72]]),
        torch.tensor([[-0.70]]),
        torch.tensor([False]),
        torch.tensor([[14.0, 17.0]]),
        contact_force_threshold=11.0,
        contact_command_lead=0.006,
        retain_contact_preload=True,
    )
    torch.testing.assert_close(target, torch.tensor([[-0.714]]))
    assert hold.tolist() == [True]


def test_faster_single_contact_closing_does_not_increase_latched_preload():
    actual = torch.full((3, 1), -0.7)
    previous = actual + 0.012
    target, held = advance_gripper_actuator_targets(
        torch.zeros_like(actual), actual, previous, torch.tensor([False, False, True]),
        torch.tensor([[2., 0.], [2., 2.], [2., 2.]]),
        contact_close_step=0.006, contact_command_lead=0.018,
        hold_command_lead=0.006, retain_contact_preload=True,
    )
    torch.testing.assert_close(target - actual, torch.tensor([[0.018], [0.006], [0.012]]))
    assert held.tolist() == [False, True, True]
    maintained, _ = advance_gripper_actuator_targets(
        torch.zeros(1, 1), actual[:1], actual[:1], torch.tensor([True]),
        torch.tensor([[2., 2.]]), contact_command_lead=0.018, hold_command_lead=0.006,
        maintain_contact_preload=True,
    )
    torch.testing.assert_close(maintained, actual[:1] + 0.006)


def test_optional_hold_hysteresis_does_not_reclose_during_load_transfer():
    target, hold = advance_gripper_actuator_targets(
        torch.zeros(1, 1),
        torch.tensor([[-0.59]]),
        torch.tensor([[-0.60]]),
        torch.tensor([True]),
        torch.tensor([[2.0, 3.0]]),
        contact_force_threshold=8.0,
        hold_contact_force_threshold=0.3,
    )
    torch.testing.assert_close(target, torch.tensor([[-0.60]]))
    assert hold.tolist() == [True]


def test_opt_in_preload_bound_removes_impact_ratchet_without_changing_default():
    actual = torch.tensor([[-0.7798]])
    previous = torch.tensor([[-0.7701]])
    args = (
        torch.zeros_like(actual), actual, previous, torch.tensor([True]),
        torch.tensor([[50.0, 53.7]]),
    )
    default, _ = advance_gripper_actuator_targets(
        *args, maintain_contact_preload=True, hold_command_lead=0.006,
    )
    bounded, held = advance_gripper_actuator_targets(
        *args, maintain_contact_preload=True, hold_command_lead=0.006,
        bound_contact_preload=True,
    )
    torch.testing.assert_close(default, previous)
    torch.testing.assert_close(bounded, actual + 0.006)
    assert held.item()
    frozen, _ = advance_gripper_actuator_targets(
        torch.zeros_like(actual), actual - 0.003, bounded, held,
        torch.tensor([[20.0, 22.0]]),
        maintain_contact_preload=False, prevent_reopening_while_closing=True,
    )
    torch.testing.assert_close(frozen, bounded)


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


def test_pick_recontact_does_not_relax_preload_or_open_jaw():
    actual = torch.tensor([[-0.58]])
    previous = torch.tensor([[-0.536]])
    target, hold = advance_gripper_actuator_targets(
        torch.zeros(1, 1),
        actual,
        previous,
        torch.tensor([False]),
        torch.tensor([[25.0, 22.0]]),
        contact_force_threshold=11.0,
        contact_command_lead=0.006,
        retain_contact_preload=True,
        prevent_reopening_while_closing=True,
    )
    torch.testing.assert_close(target, previous)
    assert hold.item()


def test_pick_insufficient_load_force_resumes_closure_without_reopening():
    previous = torch.tensor([[-0.60]])
    target, hold = advance_gripper_actuator_targets(
        torch.zeros(1, 1),
        torch.tensor([[-0.61]]),
        previous,
        torch.tensor([True]),
        torch.tensor([[2.0, 3.0]]),
        contact_force_threshold=11.0,
        hold_contact_force_threshold=11.0,
        prevent_reopening_while_closing=True,
    )
    assert (target >= previous).all()
    assert not hold.item()
