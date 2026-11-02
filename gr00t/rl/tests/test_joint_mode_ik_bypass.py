"""Joint mode bypasses discarded IK without losing trunk-filter or EE transitions."""

import ast
import copy
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from legged_gym.utils.trunk_control_filter import TrunkControlFilter


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("teleop", [False, True])
def test_joint_only_bypass_matches_published_targets_and_filter_history(device, teleop):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU arm targets")
    root = Path(__file__).resolve().parents[1]
    source = root / "third_party/visual_wholebody_lowlevel/legged_gym/envs/manip_loco/manip_loco.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    original = copy.deepcopy(methods["_get_arm_pos_targets"])
    original.name = "original_arm_targets"
    guards = [
        node
        for node in original.body
        if isinstance(node, ast.If) and "teleop_arm_joint_control_mask" in ast.unparse(node.test)
    ]
    assert len(guards) == 1
    original.body.remove(guards[0])
    calls = []

    def dls(jacobian, delta, joints, limits, damping):
        calls.append(True)
        return delta * damping

    scope = {
        "torch": torch,
        "fixed_normalize": lambda q, eps: q / q.norm(dim=-1, keepdim=True).clamp_min(eps),
        "orientation_error": lambda q, actual: q[:, 1:] - actual[:, 1:],
        "batched_constrained_dls_delta": dls,
    }
    exec(
        compile(
            ast.Module(
                body=[
                    methods["_get_ik_control_goal"],
                    methods["_get_arm_pos_targets"],
                    original,
                ],
                type_ignores=[],
            ),
            str(source),
            "exec",
        ),
        scope,
    )

    def environment():
        count = 4
        identity = torch.tensor([1, 0, 0, 0], device=device, dtype=torch.float32).expand(count, -1)
        position = torch.zeros(count, 3, device=device)
        position[:, 2] = 0.52
        joints = torch.zeros(count, 6, device=device)
        env = SimpleNamespace(
            teleop_mode=teleop,
            num_envs=count,
            teleop_arm_joint_control_mask=torch.ones(count, device=device, dtype=torch.bool),
            teleop_arm_joint_pos_targets=torch.full_like(joints, 0.4),
            target_pos=joints.clone(),
            dof_pos=joints,
            dof_pos_limits=torch.tensor([[-3, 3]] * 6, device=device, dtype=torch.float32),
            arm_joint_ids=torch.arange(6, device=device),
            ee_pos=position + position.new_tensor([0.3, 0, 0.2]),
            ee_orn=identity,
            ee_j_eef=torch.eye(6, device=device).expand(count, -1, -1),
            curr_ee_goal_cart_world=position + position.new_tensor([0.4, 0, 0.2]),
            ee_goal_orn_quat=identity,
            robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=position)),
            base_quat=identity,
            global_steps=0,
            dt=0.02,
            _trunk_control_filter=TrunkControlFilter(),
            uses_goal_height_reference_mask=False,
            get_invariant_goal_reference_origin=lambda: position * 0,
            cfg=SimpleNamespace(
                goal_ee=SimpleNamespace(ik_damping=0.1, teleop_ik_orientation_gain=1)
            ),
            _control_ik=lambda delta: delta.squeeze(-1) * 0.1,
        )
        env._get_ik_control_goal = MethodType(scope["_get_ik_control_goal"], env)
        return env

    old, new = environment(), environment()
    patterns = [[True] * 4] * 3 + [[True, False, True, False]] * 3 + [[False] * 4] * 3
    for step, pattern in enumerate(patterns):
        for env in (old, new):
            env.teleop_arm_joint_control_mask[:] = torch.tensor(pattern, device=device)
            env.global_steps = step
            env.robot.data.root_pos_w[:, 2] = 0.52 + 0.003 * step
        expected = scope["original_arm_targets"](old)
        previous = len(calls)
        actual = scope["_get_arm_pos_targets"](new)
        if teleop:
            expected = torch.where(
                old.teleop_arm_joint_control_mask[:, None],
                old.teleop_arm_joint_pos_targets,
                expected,
            )
            actual = torch.where(
                new.teleop_arm_joint_control_mask[:, None], new.teleop_arm_joint_pos_targets, actual
            )
            assert len(calls) - previous == (0 if all(pattern) else 1)
            torch.testing.assert_close(
                old._trunk_control_filter.stages, new._trunk_control_filter.stages, rtol=0, atol=0
            )
            assert old._trunk_control_filter.timestamp == new._trunk_control_filter.timestamp
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
