"""Execution contracts apply to every task stage, not a seed whitelist."""

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open import execution, pose_planning, tensor_fsm
from gr00t.rl.auto_open.acceptance import (
    RobotStabilityProfile,
    TensorStabilityWindow,
    consecutive_steps,
    update_robot_stability,
)
from gr00t.rl.auto_open.execution import StageExecutionMixin
from gr00t.rl.auto_open.motion_math import track_base_pose
from gr00t.rl.auto_open.motion_targets import PaddedPathBank
from gr00t.rl.auto_open.stage_interfaces import ArmMode, BaseMode, stage_interfaces
from gr00t.rl.scripts.doorman_auto_open import Stage


@pytest.mark.parametrize("pull", [False, True])
def test_every_stage_has_a_closed_execution_contract(pull):
    registry = stage_interfaces(Stage, pull=pull)
    assert set(registry) == set(Stage)
    assert registry[Stage.PASS_THROUGH_DOORWAY].base == BaseMode.TRAJECTORY
    assert registry[Stage.MOVE_TO_PANEL_STANDOFF].arm == (
        ArmMode.TARGET if pull else ArmMode.TRAJECTORY
    )
    assert registry[Stage.INSERT_AROUND_HANDLE].arm == ArmMode.TRAJECTORY
    assert registry[Stage.FAILURE].arm is None  # Hold the last issued target.
    assert {r.base for r in registry.values()} <= set(BaseMode)
    assert {r.arm for r in registry.values()} <= set(ArmMode) | {None}


def test_stage_orchestration_cannot_write_low_level_commands():
    tree = ast.parse(inspect.getsource(tensor_fsm))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                assert not any(
                    isinstance(n, ast.Name) and n.id == "buffers" for n in ast.walk(target)
                ), ast.dump(target)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "track_base_pose"


@pytest.mark.parametrize("module", [execution, pose_planning])
def test_execution_and_planning_do_not_read_device_values_on_host(module):
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"cpu", "numpy", "tolist", "item"}


def controller(count, device):
    control = StageExecutionMixin()
    control.base_command_previous = torch.zeros(count, 3, device=device)
    control.base_reversal_pending_sign = torch.zeros(count, 3, dtype=torch.int8, device=device)
    control.base_reversal_pending_steps = torch.zeros(count, 3, dtype=torch.int32, device=device)
    control.base_reversal_lock_steps = torch.zeros(count, 3, dtype=torch.int32, device=device)
    return control


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_point_and_trajectory_references_share_tracking_and_task_stop(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    control = controller(3, device)
    roots = torch.zeros(3, 13, device=device)
    roots[:, 3] = 1.0
    xy = torch.tensor([[1.0, 0.2], [0.5, -0.3], [0.0, 0.0]], device=device)
    yaw = torch.tensor([0.3, -0.2, 0.1], device=device)
    velocity = torch.tensor([[0.1, 0.0, 0.2], [0.0, 0.1, -0.1], [0.0, 0.0, 0.0]], device=device)
    stopped = torch.tensor([False, True, False], device=device)
    buffers = SimpleNamespace(base=torch.empty(3, 3, device=device))
    expected = track_base_pose(
        roots,
        xy,
        yaw,
        xy,
        yaw,
        minimum_speed_toward_target_only=False,
        reference_velocity_world=velocity,
    )[0]
    expected[stopped] = 0.0
    control.base_target(
        {"root_states": roots},
        torch.arange(3, device=device),
        xy,
        yaw,
        buffers,
        task_stop=stopped,
        reference_velocity_world=velocity,
    )
    torch.testing.assert_close(buffers.base, expected, rtol=0.0, atol=0.0)
    one = controller(1, device)
    single = SimpleNamespace(base=torch.empty(1, 3, device=device))
    one.base_target(
        {"root_states": roots[:1]},
        torch.arange(1, device=device),
        xy[:1],
        yaw[:1],
        single,
        reference_velocity_world=velocity[:1],
    )
    torch.testing.assert_close(single.base[0], buffers.base[0], rtol=0.0, atol=0.0)


def test_shared_arm_trajectory_holds_endpoint_and_joint_mode_is_explicit():
    control = StageExecutionMixin()
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    control.env = SimpleNamespace(
        get_goal_reference_origin=lambda: torch.zeros(1, 3),
        get_goal_reference_quat=lambda: q,
        get_goal_center_offset_local=lambda: torch.zeros(1, 3),
    )
    control.arm_induced_pitch = torch.zeros(1)
    buffers = SimpleNamespace(
        ee_cart=torch.zeros(1, 3),
        ee_quat=q.clone(),
        ee_quat_enabled=torch.zeros(1, dtype=torch.bool),
        ee_rpy=torch.zeros(1, 3),
        arm_joint_mode=torch.ones(1, dtype=torch.bool),
        arm_joint=torch.zeros(1, 6),
    )
    path = PaddedPathBank(1, 2, "cpu")
    rows = torch.tensor([0])
    path.install_tensor(
        rows,
        torch.tensor([[[0.2, 0.0, 0.3], [0.3, 0.0, 0.3]]]),
        q[:, None].expand(-1, 2, -1),
        torch.tensor([2]),
    )
    for _ in range(4):
        control.arm_trajectory(rows, path, buffers)
    torch.testing.assert_close(buffers.ee_cart, torch.tensor([[0.3, 0.0, 0.3]]))
    assert not buffers.arm_joint_mode.any()
    control.arm_joint_target(rows, torch.ones(1, 6), buffers)
    assert buffers.arm_joint_mode.all()
    assert buffers.arm_joint.eq(1).all()


def test_stability_profile_and_task_windows_are_independent():
    roots = torch.zeros(2, 13)
    roots[:, 3] = 1.0
    roots[1, 7] = 0.1
    steps = torch.zeros(2, dtype=torch.int32)
    profile = RobotStabilityProfile(required_frames=2)
    for _ in range(2):
        done, steps, stable = update_robot_stability(roots, steps, profile=profile)
    assert done.tolist() == [True, False]
    assert consecutive_steps(torch.tensor([False, True]), steps).tolist() == [0, 1]
    assert TensorStabilityWindow(2, "cpu").max_position_span_m == 0.001


def test_one_solver_submission_for_ragged_candidate_and_trajectory_bank(monkeypatch):
    calls = []

    def solve(p, q, **kwargs):
        calls.append(p.clone())
        return p

    monkeypatch.setattr(pose_planning, "solve_pose_targets", solve)
    planner = pose_planning.PosePlanningMixin()
    planner.arm_joint_ids = torch.arange(6)
    planner.joint_limits = torch.tensor([[-3.0, 3.0]] * 6)
    roots = torch.zeros(2, 13)
    roots[:, 3] = 1.0
    positions = torch.arange(18.0).reshape(2, 3, 3)
    _, owner, sample = planner._inverse_pose_bank(
        SimpleNamespace(device="cpu"),
        {"root_states": roots, "dof_pos": torch.zeros(2, 6)},
        torch.arange(2),
        positions,
        torch.zeros(2, 3, 4),
        torch.tensor([1, 3]),
    )
    assert len(calls) == 1
    assert owner.tolist() == [0, 1, 1, 1]
    assert sample.tolist() == [0, 0, 1, 2]
