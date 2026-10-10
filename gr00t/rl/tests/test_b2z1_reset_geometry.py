import math
import unittest
from pathlib import Path

import numpy as np
import pinocchio as pin
import torch

from gr00t.rl.envs.door.b2z1_reset_geometry import (
    b2z1_gripper_collision_obb,
    b2z1_gripper_pose_and_jacobian,
    initial_gripper_door_collision,
    oriented_boxes_overlap,
    solve_b2z1_arm_ik,
)


class B2Z1ResetGeometryTest(unittest.TestCase):
    def test_forward_kinematics_matches_production_urdf(self):
        urdf = (
            Path(__file__).parents[1]
            / "third_party/visual_wholebody_lowlevel/resources/robots/b2z1/urdf/generated"
            / "b2z1_mount_90_x0p22_y0_z0p17.urdf"
        )
        model = pin.buildModelFromUrdf(str(urdf))
        data = model.createData()
        frame_id = model.getFrameId("gripper_link")
        arm_names = tuple(f"joint{index}" for index in range(1, 7))
        samples = torch.tensor(
            [
                [-1.57, 1.60, -0.63, -1.10, 0.0, 0.0],
                [-1.49, 1.51, -0.56, -1.02, 0.08, -0.09],
            ],
            dtype=torch.float64,
        )
        root_position = torch.tensor(
            [[0.7, -0.3, 0.45], [1.2, 0.8, 0.45]], dtype=torch.float64
        )
        yaw = torch.tensor([0.4, -1.0], dtype=torch.float64)
        root_quaternion = torch.stack(
            (
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                torch.sin(0.5 * yaw),
                torch.cos(0.5 * yaw),
            ),
            dim=-1,
        )
        center, axes, half_extent = b2z1_gripper_collision_obb(
            root_position, root_quaternion, samples
        )
        local_center = torch.tensor(
            [0.02203577, 0.0, -0.001962435], dtype=torch.float64
        )
        ee_positions = center - torch.einsum("bij,j->bi", axes, local_center)

        for sample_index in range(samples.shape[0]):
            configuration = pin.neutral(model)
            for joint_name, value in zip(arm_names, samples[sample_index].tolist()):
                joint = model.joints[model.getJointId(joint_name)]
                configuration[joint.idx_q] = value
            pin.forwardKinematics(model, data, configuration)
            pin.updateFramePlacements(model, data)
            pose_base = data.oMf[frame_id]
            yaw_value = float(yaw[sample_index])
            rotation_root = np.asarray(
                [
                    [math.cos(yaw_value), -math.sin(yaw_value), 0.0],
                    [math.sin(yaw_value), math.cos(yaw_value), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            expected_position = (
                root_position[sample_index].numpy()
                + rotation_root @ np.asarray(pose_base.translation)
            )
            expected_rotation = rotation_root @ np.asarray(pose_base.rotation)
            np.testing.assert_allclose(
                ee_positions[sample_index].numpy(), expected_position, atol=1.0e-9
            )
            np.testing.assert_allclose(
                axes[sample_index].numpy(), expected_rotation, atol=1.0e-9
            )
        self.assertTrue(torch.all(half_extent > 0.0))

    def test_obb_overlap_respects_rotation_and_margin(self):
        center_a = torch.tensor([[0.0, 0.0, 0.0]])
        center_b = torch.tensor([[2.01, 0.0, 0.0]])
        identity = torch.eye(3).unsqueeze(0)
        half = torch.ones(1, 3)
        self.assertFalse(
            bool(oriented_boxes_overlap(center_a, identity, half, center_b, identity, half))
        )
        self.assertTrue(
            bool(
                oriented_boxes_overlap(
                    center_a,
                    identity,
                    half,
                    center_b,
                    identity,
                    half,
                    separation_margin=0.02,
                )
            )
        )

    def test_batched_inverse_kinematics_recovers_reachable_gripper_pose(self):
        target_joint_position = torch.tensor(
            [
                [-1.42, 1.45, -0.72, -0.95, 0.20, -0.18],
                [-1.66, 1.72, -0.52, -1.22, -0.16, 0.24],
            ],
            dtype=torch.float64,
        )
        root_position = torch.tensor(
            [[0.75, -0.2, 0.45], [0.9, 0.35, 0.45]], dtype=torch.float64
        )
        yaw = torch.tensor([0.15, -0.3], dtype=torch.float64)
        root_quaternion = torch.stack(
            (
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                torch.sin(0.5 * yaw),
                torch.cos(0.5 * yaw),
            ),
            dim=-1,
        )
        target_position, target_rotation, _ = b2z1_gripper_pose_and_jacobian(
            root_position, root_quaternion, target_joint_position
        )
        initial = torch.tensor(
            [-1.57, 1.60, -0.63, -1.10, 0.0, 0.0], dtype=torch.float64
        ).expand(2, -1)
        solution, position_error, orientation_error = solve_b2z1_arm_ik(
            root_position,
            root_quaternion,
            target_position,
            target_rotation,
            initial,
            torch.full((6,), -3.0, dtype=torch.float64),
            torch.full((6,), 3.0, dtype=torch.float64),
        )
        self.assertEqual(solution.shape, (2, 6))
        self.assertTrue(torch.all(position_error < 1.0e-4), position_error)
        self.assertTrue(torch.all(orientation_error < 1.0e-4), orientation_error)
    def test_closed_door_collision_changes_with_base_distance(self):
        arm = torch.tensor([[-1.57, 1.60, -0.63, -1.10, 0.0, 0.0]]).repeat(2, 1)
        root_position = torch.tensor([[0.55, 0.0, 0.45], [2.0, 0.0, 0.45]])
        root_quaternion = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(2, 1)
        door_position = torch.zeros(2, 3)
        door_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        collision = initial_gripper_door_collision(
            root_position,
            root_quaternion,
            arm,
            door_position,
            door_quaternion,
            torch.full((2,), 0.9),
            torch.full((2,), 2.0),
            torch.full((2,), 0.9),
            torch.full((2,), 0.1),
            torch.ones(2),
            torch.full((2,), 0.2),
            torch.full((2,), 0.12),
            torch.full((2,), 0.013),
            torch.full((2,), 2.7),
            separation_margin=0.01,
        )
        self.assertEqual(collision.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
