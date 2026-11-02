"""Diagnostic command offsets do not redefine the physical intended endpoint."""

import pytest
import torch

from gr00t.rl.pick.probe_policy import offset_command_goals


def test_offsets_preserve_intent_and_have_one_value_per_command():
    intended = torch.tensor([[1.0, 2.0, 0.105]]).expand(4, -1).clone()
    command = offset_command_goals(intended, torch.tensor([0.0, -0.02, -0.04, -0.06]))
    torch.testing.assert_close(command[:, 2], torch.tensor([0.105, 0.085, 0.065, 0.045]))
    torch.testing.assert_close(intended[:, 2], torch.full((4,), 0.105))
    torch.testing.assert_close(command[:, :2], intended[:, :2])
    with pytest.raises(ValueError, match="own height"):
        offset_command_goals(intended, torch.zeros(1))
