"""Command calibration never moves the physical reference or another batch row."""

import pytest
import torch
from gr00t.rl.pick.command_compensation import LowTargetCommandCompensation


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bounded_low_compensation_and_lift_release_preserve_physical_goals(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU command compensation")
    correction = LowTargetCommandCompensation(4, device)
    rows = torch.arange(4, device=device)
    intended = torch.zeros(4, 3, device=device)
    intended[:, 2] = intended.new_tensor([0.105, 0.105, 0.5, 0.105])
    original = intended.clone()
    actual = intended.clone()
    actual[:, 2] += 0.023
    actual[3, 0] += 0.10
    enabled = torch.ones(4, device=device, dtype=torch.bool)
    disabled = ~enabled

    def forbidden(*args, **kwargs):
        raise AssertionError("Calibration must not read individual GPU values on the host")

    with monkeypatch.context() as scope:
        for name in ("cpu", "item", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        for _ in range(100):
            command = correction.apply(
                rows, intended, actual, adapt=enabled, retain=enabled, reset=disabled, dt=0.02
            )
            actual[:2, 2] = command[:2, 2] + 0.023
    torch.testing.assert_close(intended, original)
    assert ((actual[:2, 2] - intended[:2, 2]).abs() < 0.012).all()
    assert (correction.applied >= -0.020001).all() and (correction.applied <= 0).all()
    torch.testing.assert_close(correction.bias[2:], intended.new_zeros(2))
    torch.testing.assert_close(command[:, :2], intended[:, :2])
    frozen = correction.bias.clone()
    closing = correction.apply(
        rows, intended, actual, adapt=disabled, retain=enabled, reset=disabled, dt=0.02
    )
    torch.testing.assert_close(correction.bias, frozen)
    torch.testing.assert_close(closing[:, 2] - intended[:, 2], frozen)
    lifted = intended.clone()
    lifted[:, 2] = 0.305
    released = correction.apply(
        rows, lifted, actual, adapt=disabled, retain=enabled, reset=disabled, dt=0.02
    )
    torch.testing.assert_close(released, lifted)
    reset = correction.apply(
        rows, intended, actual, adapt=disabled, retain=enabled, reset=enabled, dt=0.02
    )
    torch.testing.assert_close(reset, intended)


def test_compensation_is_batch_independent_and_inactive_until_enabled():
    correction = LowTargetCommandCompensation(3, "cpu")
    reference = torch.tensor([[0.1, 0, 0.1], [0.2, 0, 0.12], [0.3, 0, 0.11]])
    actual = reference + torch.tensor([0, 0, 0.025])
    order = torch.tensor([2, 0, 1])
    enabled = torch.ones(3, dtype=torch.bool)
    command = correction.apply(
        order,
        reference[order],
        actual[order],
        adapt=~enabled,
        retain=enabled,
        reset=~enabled,
        dt=0.02,
    )
    torch.testing.assert_close(command, reference[order])
    for _ in range(5):
        command = correction.apply(
            order,
            reference[order],
            actual[order],
            adapt=enabled,
            retain=enabled,
            reset=~enabled,
            dt=0.02,
        )
    for row in range(3):
        single = LowTargetCommandCompensation(1, "cpu")
        for _ in range(5):
            expected = single.apply(
                torch.tensor([0]),
                reference[row : row + 1],
                actual[row : row + 1],
                adapt=enabled[:1],
                retain=enabled[:1],
                reset=~enabled[:1],
                dt=0.02,
            )
        torch.testing.assert_close(correction.applied[row : row + 1], single.applied)
        torch.testing.assert_close(command[order == row], expected)
