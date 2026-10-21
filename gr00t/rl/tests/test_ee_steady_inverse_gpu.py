import math
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.rl.envs.door.b2z1_reset_geometry import (
    b2z1_gripper_pose_and_jacobian,
)
from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.ee_steady_inverse_gpu import (
    batched_steady_ee_inverse_no_gravity,
)
from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.ee_steady_state import (
    B2Z1SteadyEeForwardModel,
)


_B2_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_inverse_recovers_batched_reachable_poses_without_gravity():
    device = torch.device("cuda")
    dtype = torch.float64
    root_position = torch.tensor(
        [[0.0, 0.0, 0.55], [0.1, -0.2, 0.52]], device=device, dtype=dtype
    )
    root_quaternion = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, math.sin(0.15), math.cos(0.15)]],
        device=device,
        dtype=dtype,
    )
    initial = torch.tensor(
        [[0.0, 1.2, -1.8, -1.0, 0.0, 0.0]] * 2,
        device=device,
        dtype=dtype,
    )
    target_joints = initial + torch.tensor(
        [[0.08, -0.05, 0.04, -0.03, 0.06, -0.04], [-0.06, 0.04, -0.03, 0.05, -0.04, 0.03]],
        device=device,
        dtype=dtype,
    )
    target_position, target_rotation, _ = b2z1_gripper_pose_and_jacobian(
        root_position, root_quaternion, target_joints
    )
    from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

    target_orientation = _matrix_to_quaternion_wxyz(target_rotation)
    limits = torch.tensor([[-3.0, 3.0]] * 6, device=device, dtype=dtype)
    result = batched_steady_ee_inverse_no_gravity(
        target_position,
        target_orientation,
        base_position_world=root_position,
        base_orientation_world_xyzw=root_quaternion,
        initial_arm_joint_positions=initial,
        joint_limits=limits,
        position_tolerance_m=1.0e-4,
        orientation_tolerance=1.0e-4,
    )
    assert torch.all(result.exact), result.position_error
    torch.testing.assert_close(
        result.command_position_world, target_position, rtol=0.0, atol=1.0e-4
    )
    assert torch.equal(result.arm_joint_positions, result.arm_position_targets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_inverse_matches_cpu_without_gravity_at_joint_limit():
    device = torch.device("cuda")
    dtype = torch.float64
    root_position = torch.tensor([[0.0, 0.0, 0.54]], device=device, dtype=dtype)
    root_quaternion = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype
    )
    initial = torch.tensor(
        [[-1.514, 1.80, -0.30, -1.40, -0.24, -2.10]],
        device=device,
        dtype=dtype,
    )
    raw_limits = torch.tensor(
        [
            [-2.6179938779914944, 2.6179938779914944],
            [0.0, 2.9670597283903604],
            [-2.8797932657906435, 0.0],
            [-1.5184364492350666, 1.5184364492350666],
            [-1.3439035240356338, 1.3439035240356338],
            [-2.792526803190927, 2.792526803190927],
        ],
        device=device,
        dtype=dtype,
    )
    outside_limit = initial.clone()
    outside_limit[:, 1] += 0.06
    outside_limit[:, 2] += 0.08
    outside_limit[:, 3] = -1.55
    target_position, target_rotation, _ = b2z1_gripper_pose_and_jacobian(
        root_position, root_quaternion, outside_limit
    )
    from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

    target_orientation = _matrix_to_quaternion_wxyz(target_rotation)
    gpu = batched_steady_ee_inverse_no_gravity(
        target_position,
        target_orientation,
        base_position_world=root_position,
        base_orientation_world_xyzw=root_quaternion,
        initial_arm_joint_positions=initial,
        joint_limits=raw_limits,
    )

    urdf = (
        Path(__file__).resolve().parents[1]
        / "third_party/visual_wholebody_lowlevel/resources/robots/b2z1/urdf/generated"
        / "b2z1_mount_90_x0p22_y0_z0p17.urdf"
    )
    cpu_model = B2Z1SteadyEeForwardModel(
        urdf,
        stiffness=10_000.0,
        gravity_magnitude=0.0,
        emit_tolerance_warnings=False,
    )
    joint_positions = dict.fromkeys(_B2_JOINT_NAMES, 0.0)
    joint_positions["jointGripper"] = 0.0
    joint_positions.update(
        zip(cpu_model.arm_joint_names, initial[0].cpu().tolist(), strict=True)
    )
    cpu = cpu_model.inverse(
        target_position[0].cpu().numpy(),
        target_orientation[0].cpu().numpy(),
        base_position_world=root_position[0].cpu().numpy(),
        base_orientation_world_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        joint_positions=joint_positions,
        max_function_evaluations=100,
    )

    torch.testing.assert_close(
        gpu.command_position_world[0].cpu(),
        torch.from_numpy(cpu.command_position_world),
        rtol=0.0,
        atol=2.0e-5,
    )
    torch.testing.assert_close(
        gpu.arm_joint_positions[0].cpu(),
        torch.from_numpy(cpu.forward_result.arm_joint_positions),
        rtol=0.0,
        atol=2.0e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [9, 250])
def test_gpu_inverse_result_is_batch_shape_independent(dtype, batch_size):
    device = torch.device("cuda")
    root = torch.tensor([[0.0, 0.0, 0.55]], device=device, dtype=dtype)
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype)
    initial = torch.tensor(
        [[0.0, 1.2, -1.8, -1.0, 0.0, 0.0]], device=device, dtype=dtype
    )
    target = initial + torch.tensor(
        [[0.04, -0.03, 0.02, -0.01, 0.03, -0.02]], device=device, dtype=dtype
    )
    position, rotation, _ = b2z1_gripper_pose_and_jacobian(root, quaternion, target)
    from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

    orientation = _matrix_to_quaternion_wxyz(rotation)
    limits = torch.tensor([[-3.0, 3.0]] * 6, device=device, dtype=dtype)

    def solve(repeats):
        return batched_steady_ee_inverse_no_gravity(
            position.repeat(repeats, 1),
            orientation.repeat(repeats, 1),
            base_position_world=root.repeat(repeats, 1),
            base_orientation_world_xyzw=quaternion.repeat(repeats, 1),
            initial_arm_joint_positions=initial.repeat(repeats, 1),
            joint_limits=limits,
        )

    single = solve(1)
    repeated = solve(batch_size)
    torch.testing.assert_close(
        single.arm_joint_positions[0],
        repeated.arm_joint_positions[0],
        rtol=0.0,
        atol=0.0,
    )

    offsets = torch.linspace(-0.03, 0.03, batch_size, device=device, dtype=dtype)
    mixed_initial = initial.repeat(batch_size, 1)
    mixed_initial[:, 0] += offsets
    mixed_targets = target.repeat(batch_size, 1)
    mixed_targets[:, 1] += offsets
    mixed_position, mixed_rotation, _ = b2z1_gripper_pose_and_jacobian(
        root.repeat(batch_size, 1), quaternion.repeat(batch_size, 1), mixed_targets
    )
    mixed_position[4] = position[0]
    mixed_rotation[4] = rotation[0]
    mixed_initial[4] = initial[0]
    mixed = batched_steady_ee_inverse_no_gravity(
        mixed_position,
        _matrix_to_quaternion_wxyz(mixed_rotation),
        base_position_world=root.repeat(batch_size, 1),
        base_orientation_world_xyzw=quaternion.repeat(batch_size, 1),
        initial_arm_joint_positions=mixed_initial,
        joint_limits=limits,
    )
    torch.testing.assert_close(
        single.arm_joint_positions[0],
        mixed.arm_joint_positions[4],
        rtol=0.0,
        atol=0.0,
    )
