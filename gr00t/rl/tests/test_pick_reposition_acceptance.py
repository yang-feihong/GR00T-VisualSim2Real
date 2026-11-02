"""Reposition acceptance exercises the active batched FSM, never a scalar lifecycle."""

import math
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick.stages import BasketStage, stage_definitions
from gr00t.rl.pick.tensor_fsm import TensorPickFSM
from gr00t.rl.tests.test_pick_workflow import fake_controller, tensor_task


REPOSITION_STAGES = (
    BasketStage.OPEN_FOR_REPOSITION,
    BasketStage.CLEAR_FOR_REPOSITION,
    BasketStage.CLOSE_FOR_REPOSITION,
    BasketStage.REPOSITION_OBJECT,
    BasketStage.REOPEN_AFTER_REPOSITION,
)


@pytest.mark.parametrize("stage", REPOSITION_STAGES)
def test_reposition_command_is_standard_ee_path_with_zero_base(stage):
    control = TensorPickFSM.__new__(TensorPickFSM)
    control.stage = torch.tensor([int(stage)])
    control.rows, control.num_envs = torch.tensor([0]), 1
    control.env, control.arm_path = object(), object()
    buffers = SimpleNamespace(base=torch.ones(1, 3))
    control.stop_base = lambda ids, command: command.base.zero_()
    calls = []
    control.arm_trajectory = lambda ids, path, command, **kw: calls.append(kw["advance"])
    control.gripper_target = lambda ids, value, command: calls.append(value.clone())
    control.command({"ee_pos": torch.zeros(1, 3)}, buffers)
    assert not buffers.base.any()
    assert calls[0] == (stage in (BasketStage.CLEAR_FOR_REPOSITION, BasketStage.REPOSITION_OBJECT))
    closed = stage in (BasketStage.CLOSE_FOR_REPOSITION, BasketStage.REPOSITION_OBJECT)
    assert float(calls[1][0]) == pytest.approx(0 if closed else -math.pi / 3)


def make_tensor_control(monkeypatch, stage):
    config, state = fake_controller(monkeypatch)
    control = tensor_task(config, state)
    control.arm_path.install_tensor(
        control.rows,
        state["ee_pos"][:, None],
        state["ee_orn"][:, None],
        torch.ones(1, dtype=torch.long),
    )
    control.Stage, control.stage_specs, control.interfaces = stage_definitions(basket=True)
    for key, spec in control.stage_specs.items():
        control.stage_limits[int(key)] = spec.budget_steps or 0
    control.stage.fill_(int(stage))
    control.basket = object()
    state["object_state"][:, :3] = torch.tensor([[0.15, 0, 0.15]])
    state["dof_pos"][:, control.env.gripper_joint_ids] = -math.pi / 3
    control._basket_measures = lambda state: (
        torch.tensor([0.1]),
        0.18 - state["object_state"][:, 0],
        torch.tensor([0.2]),
    )
    observed, entries = [], []
    original = control.record_acceptance

    def record(current, rows, conditions, **kwargs):
        if int(current) == int(stage):
            observed.append(conditions)
        return original(current, rows, conditions, **kwargs)

    control.record_acceptance = record
    return control, state, entries, observed


def advance(control, state, entries):
    previous = int(control.stage[0])
    control.advance(state)
    if int(control.stage[0]) != previous:
        entries.append(control.Stage(int(control.stage[0])))


def test_push_does_not_start_until_empty_gripper_is_closed(monkeypatch):
    control, state, entries, observed = make_tensor_control(
        monkeypatch, BasketStage.CLOSE_FOR_REPOSITION
    )
    for _ in range(12):
        advance(control, state, entries)
    assert not entries
    assert not bool(observed[-1]["gripper_target_error_rad"][0])
    state["dof_pos"].zero_()
    for _ in range(35):
        advance(control, state, entries)
    assert entries == [BasketStage.REPOSITION_OBJECT]


def test_push_requires_actual_object_motion_before_actual_grasp_revalidation(monkeypatch):
    control, state, entries, observed = make_tensor_control(
        monkeypatch, BasketStage.REPOSITION_OBJECT
    )
    control.nudge_start = state["object_state"][:, :3].clone()
    control.nudge_direction = torch.tensor([-1.0, 0, 0])
    control.nudge_space_before, control.active_nudge_report = 0.03, {}
    for _ in range(12):
        advance(control, state, entries)
    assert not entries
    assert not bool(observed[-1]["measured_object_shift_m"][0])
    state["object_state"][:, 0] = 0.12
    for _ in range(35):
        advance(control, state, entries)
        if entries:
            break
    assert entries == [BasketStage.REOPEN_AFTER_REPOSITION]
    assert float(observed[-1]["measured_object_shift_m"][1]) == pytest.approx(0.03)


@pytest.mark.parametrize("fractions", [(), (1,), (-1,), (float("nan"),)])
def test_nudge_contact_height_experiments_reject_invalid_fractions(fractions):
    from gr00t.rl.pick.basket_nudge import plan_closed_nudge

    with pytest.raises(ValueError, match="Contact heights"):
        plan_closed_nudge(
            None, None, None, None, None, None, 0.02, contact_height_fractions=fractions
        )
