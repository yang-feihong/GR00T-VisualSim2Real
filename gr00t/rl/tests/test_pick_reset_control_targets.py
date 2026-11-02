"""Custom episode reset must align rate-limiter history with the physical preset."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_episode_reset_does_not_command_a_stale_arm_or_gripper_pose(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = Path(__file__).parents[1] / "pick/environment.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PickEnvironment"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_reset_batch"
    )
    scope = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    presets = [
        [-1.57, 2.6, -1.5, 0.4, 0, 0],
        [-1.57, 1.75, -0.785, -0.93, 0, 0],
        [1.57, 1.57, -1.3, 1.1, 0, 0],
    ]
    episodes = [
        {
            "base_xy": [i, 0],
            "base_yaw": 0,
            "object_position": [i + 1, 0, 0.03],
            "object_yaw": 0,
            "initial_arm_joints": preset,
            "distractors": [],
        }
        for i, preset in enumerate(presets)
    ]
    calls = {}

    def write_joints(positions, velocity, *, env_ids):
        calls["physical_joints"] = positions.clone()

    env = SimpleNamespace(
        pick_episode=episodes,
        pick_distractors=[],
        scene=SimpleNamespace(env_origins=torch.zeros(3, 3, device=device)),
        default_dof_pos=torch.zeros(3, 7, device=device),
        arm_joint_ids=torch.arange(6, device=device),
        gripper_joint_ids=torch.tensor([6], device=device),
        robot=SimpleNamespace(
            write_root_state_to_sim=lambda *a, **kw: None,
            write_joint_state_to_sim=write_joints,
            set_joint_position_target=lambda *a, **kw: None,
        ),
        _joint_position_target_buf=torch.zeros(3, 7, device=device),
        teleop_arm_joint_pos_targets=torch.zeros(3, 6, device=device),
        teleop_arm_joint_limited_targets=torch.zeros(3, 6, device=device),
        teleop_cartesian_transition_targets=torch.zeros(3, 6, device=device),
        teleop_cartesian_transition_active=torch.tensor([False, True, False], device=device),
        teleop_arm_joint_control_mask=torch.zeros(3, dtype=torch.bool, device=device),
        gripper_pos_targets=torch.zeros(3, 1, device=device),
        gripper_actuator_targets=torch.zeros(3, 1, device=device),
    )
    rows = torch.tensor([0, 2], device=device)
    scope["_reset_batch"](env, rows)
    expected = torch.tensor(presets, device=device)[rows]
    torch.testing.assert_close(calls["physical_joints"][:, :6], expected)
    for name in (
        "teleop_arm_joint_pos_targets",
        "teleop_arm_joint_limited_targets",
        "teleop_cartesian_transition_targets",
    ):
        torch.testing.assert_close(getattr(env, name)[rows], expected)
        assert not getattr(env, name)[1].any()
    assert env.teleop_arm_joint_control_mask.tolist() == [True, False, True]
    assert env.teleop_cartesian_transition_active.tolist() == [False, True, False]
    torch.testing.assert_close(
        env.gripper_actuator_targets[rows], torch.full((2, 1), -math.pi / 3, device=device)
    )
    assert env.gripper_actuator_targets[1].eq(0).all()
    first_target = env.teleop_arm_joint_limited_targets[rows] + (
        env.teleop_arm_joint_pos_targets[rows] - env.teleop_arm_joint_limited_targets[rows]
    ).clamp(-0.01, 0.01)
    torch.testing.assert_close(first_target, calls["physical_joints"][:, :6])
