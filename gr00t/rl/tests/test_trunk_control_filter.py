"""Lab30 trunk-transform semantics without launching Isaac Sim."""

import ast
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from legged_gym.utils import trunk_control_filter as module
from legged_gym.utils.isaaclab_math import (
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def test_two_poles_match_scalar_recurrence_and_batch_independence(device):
    measured = torch.tensor([[0.52, 0.0, 0.0], [0.41, 0.1, -0.2]], device=device)
    selected = module.TrunkControlFilter()
    independent = [module.TrunkControlFilter(), module.TrunkControlFilter()]
    first, second = measured.clone(), measured.clone()
    alpha = -math.expm1(-0.02 / 0.27)
    for step in range(100):
        value = measured + math.sin(step * 0.3) * 0.02
        if step:
            first += alpha * (value - first)
            second += alpha * (first - second)
        actual = selected.update(step * 0.02, value)
        torch.testing.assert_close(actual, second)
        for i, single in enumerate(independent):
            torch.testing.assert_close(single.update(step * 0.02, value[i : i + 1])[0], actual[i])


def test_repeated_clock_partial_reset_rewind_gap_and_dc(device):
    selected = module.TrunkControlFilter()
    value = torch.tensor([[0.52, 0.0, 0.0], [0.4, 0.2, -0.1]], device=device)
    torch.testing.assert_close(selected.update(0, value), value)
    previous = selected.update(0.02, value + 0.1).clone()
    torch.testing.assert_close(selected.update(0.02, value + 1), previous)
    selected.reset(torch.tensor([0], device=device))
    current = selected.update(0.02, value + 0.5)
    torch.testing.assert_close(current[0], value[0] + 0.5)
    torch.testing.assert_close(current[1], previous[1])
    torch.testing.assert_close(selected.update(-1, value), value)
    torch.testing.assert_close(selected.update(2, value + 0.2), value + 0.2)
    for step in range(1, 301):
        final = selected.update(2 + step * 0.02, value)
    torch.testing.assert_close(final, value, atol=1e-6, rtol=0)
    selected.reset()
    torch.testing.assert_close(selected.update(0, value), value)


def test_transform_equivalence_fast_commands_and_unmodified_state(device, monkeypatch):
    monkeypatch.setattr(module, "TRUNK_CONTROL_FILTER_ENABLED", True)
    selected = module.TrunkControlFilter()
    position = torch.tensor([[1.0, 2.0, 0.52]], device=device, dtype=torch.float64)
    rpy = torch.tensor([[0.0, 0.0, 0.4]], device=device, dtype=torch.float64)
    root_quat = quat_from_euler_xyz(*rpy.unbind(-1))
    goal = torch.tensor([[1.4, 2.1, 0.2]], device=device, dtype=torch.float64)
    orientation = root_quat.clone()
    ground = torch.tensor([0.1], device=device, dtype=torch.float64)
    selected.control_target(0, position, root_quat, ground, goal, orientation)
    position[:, 2] -= 0.07
    rpy[:, :2] = torch.tensor([[0.1, -0.2]], device=device)
    root_quat = quat_from_euler_xyz(*rpy.unbind(-1))
    originals = [x.clone() for x in (position, root_quat, goal, orientation)]
    output, output_quat = selected.control_target(
        0.02, position, root_quat, ground, goal, orientation
    )
    filtered = selected.stages[-1]
    filtered_position = position.clone()
    filtered_position[:, 2] = ground + filtered[:, 0]
    filtered_quat = quat_from_euler_xyz(filtered[:, 1], filtered[:, 2], rpy[:, 2])
    torch.testing.assert_close(
        quat_apply(quat_conjugate(root_quat), output - position),
        quat_apply(quat_conjugate(filtered_quat), goal - filtered_position),
    )
    torch.testing.assert_close(
        quat_mul(quat_conjugate(root_quat), output_quat),
        quat_mul(quat_conjugate(filtered_quat), orientation),
    )
    jump = torch.tensor([[0.3, -0.2, 0.4]], device=device, dtype=torch.float64)
    jump_quat = quat_from_euler_xyz(*(rpy + 0.5).unbind(-1))
    new_output, new_quat = selected.control_target(
        0.02, position, root_quat, ground, goal + jump, jump_quat
    )
    correction = quat_mul(root_quat, quat_conjugate(filtered_quat))
    torch.testing.assert_close(new_output - output, quat_apply(correction, jump))
    torch.testing.assert_close(new_quat, quat_mul(correction, jump_quat))
    for actual, original in zip((position, root_quat, goal, orientation), originals):
        torch.testing.assert_close(actual, original)
    assert module.TRUNK_CONTROL_FILTER_ENABLED
    monkeypatch.setattr(module, "TRUNK_CONTROL_FILTER_ENABLED", False)
    direct = selected.control_target(0.04, position, root_quat, ground, goal, orientation)
    assert direct[0] is goal and direct[1] is orientation


def test_environment_invariant_only_no_walking_gate_and_reset_wiring(monkeypatch):
    monkeypatch.setattr(module, "TRUNK_CONTROL_FILTER_ENABLED", True)
    source = Path(module.__file__).parents[1] / "envs/manip_loco/manip_loco.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    scope = {"torch": torch}
    exec(
        compile(
            ast.Module(body=[methods["_get_ik_control_goal"]], type_ignores=[]), str(source), "exec"
        ),
        scope,
    )
    raw = torch.tensor([[1.4, 2.0, 0.2]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    root = torch.tensor([[1.0, 2.0, 0.52]])
    env = SimpleNamespace(
        teleop_mode=True,
        curr_ee_goal_cart_world=raw,
        ee_goal_orn_quat=identity,
        robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=root)),
        base_quat=identity,
        global_steps=0,
        dt=0.02,
        _trunk_control_filter=module.TrunkControlFilter(),
        uses_goal_height_reference_mask=True,
        goal_height_follow_mask=torch.tensor([False]),
        get_invariant_goal_reference_origin=lambda: torch.tensor([[1.0, 2.0, 0.0]]),
    )
    get_goal = MethodType(scope["_get_ik_control_goal"], env)
    get_goal()
    root[:, 2] -= 0.05
    env.global_steps = 1
    assert get_goal()[0][0, 2] < raw[0, 2] - 0.049
    env.goal_height_follow_mask.fill_(True)
    torch.testing.assert_close(get_goal()[0], raw)
    env.teleop_mode = False
    assert get_goal()[0] is raw
    reset_source = ast.unparse(methods["_reset_task_specific_buffers_on_reset"])
    assert "self._trunk_control_filter.reset(env_ids)" in reset_source
    target_source = ast.unparse(methods["_get_arm_pos_targets"])
    assert "self._get_ik_control_goal()" in target_source
    assert "orientation_error(control_goal[1], ee_orn_norm)" in target_source
