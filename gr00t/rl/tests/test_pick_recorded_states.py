"""Offline planners consume the shared record, without a second trajectory dump."""

import pytest
import torch

from gr00t.rl.pick.recorded_states import load_recorded_states


def test_shared_control_record_exposes_lazy_post_step_singleton_states(tmp_path):
    post = {"root_states": torch.arange(39).reshape(3, 13), "dof_pos": torch.ones(3, 19)}
    torch.save({"post": post}, tmp_path / "control_record.pt")
    trace = load_recorded_states(tmp_path)
    assert len(trace) == 3
    torch.testing.assert_close(trace[0]["root_states"], post["root_states"][:1])
    torch.testing.assert_close(trace[-1]["root_states"], post["root_states"][-1:])
    assert len(trace[1:]) == 2
    assert trace[0]["dof_pos"].shape == (1, 19)
    with pytest.raises(IndexError):
        trace[3]
    with pytest.raises(IndexError):
        trace[-4]


def test_historical_trace_remains_readable_for_regression_comparison(tmp_path):
    torch.save([{"root_states": torch.zeros(1, 13)}], tmp_path / "trajectory.pt")
    trace = load_recorded_states(tmp_path)
    assert len(trace) == 1
