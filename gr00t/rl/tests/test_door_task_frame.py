import math
import unittest

import torch

from gr00t.rl.envs.door.task_frame import (
    build_door_task_basis,
    canonicalize_jaw_axial,
    canonicalize_jaw_order,
    frame_axial_to_task,
    frame_polar_to_task,
    task_axial_to_world,
    task_axial_to_frame,
    task_polar_to_world,
    task_polar_to_frame,
    world_axial_to_task,
    world_polar_to_task,
)


class DoorTaskFrameTest(unittest.TestCase):
    def setUp(self):
        yaw = torch.tensor([-1.1, -0.3, 0.0, 0.7, 2.2], dtype=torch.float64)
        self.normal = torch.stack(
            (torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), dim=-1
        )
        self.tangent = torch.stack(
            (-torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)), dim=-1
        )
        self.side = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
        self.basis, self.handedness = build_door_task_basis(
            self.normal, self.tangent, self.side
        )

    def test_basis_semantics_and_handedness(self):
        self.assertTrue(
            torch.allclose(self.basis[..., 0], -self.normal, atol=1.0e-12)
        )
        self.assertTrue(
            torch.allclose(
                self.basis[..., 1], self.side[:, None] * self.tangent, atol=1.0e-12
            )
        )
        self.assertTrue(
            torch.allclose(self.handedness, -self.side, atol=1.0e-12)
        )
        identity = torch.eye(3, dtype=torch.float64).expand(len(self.side), -1, -1)
        self.assertTrue(
            torch.allclose(
                self.basis.transpose(-1, -2) @ self.basis,
                identity,
                atol=1.0e-12,
            )
        )

    def test_polar_and_axial_round_trip(self):
        vectors = torch.tensor(
            [
                [0.2, -0.1, 0.3],
                [-0.4, 0.8, -0.2],
                [1.0, 2.0, 3.0],
                [-1.2, -0.7, 0.5],
                [0.0, 0.6, -0.9],
            ],
            dtype=torch.float64,
        )
        polar_world = task_polar_to_world(vectors, self.basis)
        axial_world = task_axial_to_world(vectors, self.basis, self.handedness)
        self.assertTrue(
            torch.allclose(
                world_polar_to_task(polar_world, self.basis), vectors, atol=1.0e-12
            )
        )
        self.assertTrue(
            torch.allclose(
                world_axial_to_task(axial_world, self.basis, self.handedness),
                vectors,
                atol=1.0e-12,
            )
        )

    def test_positive_task_roll_matches_positive_handle_joint_axis(self):
        canonical_roll = torch.zeros(len(self.side), 3, dtype=torch.float64)
        canonical_roll[:, 0] = math.radians(1.0)
        world_rotation = task_axial_to_world(
            canonical_roll, self.basis, self.handedness
        )
        expected = self.side[:, None] * self.normal * math.radians(1.0)
        self.assertTrue(torch.allclose(world_rotation, expected, atol=1.0e-12))

    def test_linear_motion_is_mirrored_only_along_free_end_axis(self):
        canonical = torch.tensor(
            [[0.2, 0.1, -0.05]], dtype=torch.float64
        ).expand(len(self.side), -1)
        world = task_polar_to_world(canonical, self.basis)
        expected = (
            -0.2 * self.normal
            + 0.1 * self.side[:, None] * self.tangent
            + torch.tensor([0.0, 0.0, -0.05], dtype=torch.float64)
        )
        self.assertTrue(torch.allclose(world, expected, atol=1.0e-12))

    def test_random_base_yaw_velocity_round_trip(self):
        base_yaw = torch.tensor([0.9, -1.7, 0.2, 2.6, -0.4], dtype=torch.float64)
        base_forward = torch.stack(
            (
                torch.cos(base_yaw),
                torch.sin(base_yaw),
                torch.zeros_like(base_yaw),
            ),
            dim=-1,
        )
        base_left = torch.stack(
            (
                -torch.sin(base_yaw),
                torch.cos(base_yaw),
                torch.zeros_like(base_yaw),
            ),
            dim=-1,
        )
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(
            len(self.side), -1
        )
        base_basis = torch.stack((base_forward, base_left, up), dim=-1)
        task_linear = torch.tensor(
            [
                [0.20, -0.10, 0.00],
                [-0.30, 0.40, 0.00],
                [0.00, 0.15, 0.00],
                [0.50, 0.00, 0.00],
                [-0.25, -0.35, 0.00],
            ],
            dtype=torch.float64,
        )
        task_angular = torch.tensor(
            [
                [0.0, 0.0, 0.3],
                [0.0, 0.0, -0.2],
                [0.0, 0.0, 0.1],
                [0.0, 0.0, -0.4],
                [0.0, 0.0, 0.5],
            ],
            dtype=torch.float64,
        )
        body_linear = task_polar_to_frame(task_linear, self.basis, base_basis)
        body_angular = task_axial_to_frame(
            task_angular, self.basis, self.handedness, base_basis
        )
        self.assertTrue(
            torch.allclose(
                frame_polar_to_task(body_linear, base_basis, self.basis),
                task_linear,
                atol=1.0e-12,
            )
        )
        self.assertTrue(
            torch.allclose(
                frame_axial_to_task(
                    body_angular, base_basis, self.basis, self.handedness
                ),
                task_angular,
                atol=1.0e-12,
            )
        )

    def test_canonical_jaw_order_swaps_only_right_hinged_variant(self):
        values = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        side = torch.tensor([1.0, -1.0])
        expected = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[7.0, 8.0], [5.0, 6.0]],
            ]
        )
        self.assertTrue(torch.equal(canonicalize_jaw_order(values, side), expected))

    def test_canonical_jaw_axial_unifies_free_end_direction_and_jaw_slots(self):
        values = torch.tensor(
            [
                [0.4, -0.5],
                [0.5, -0.4],
            ]
        )
        side = torch.tensor([1.0, -1.0])
        expected = torch.tensor(
            [
                [0.4, -0.5],
                [0.4, -0.5],
            ]
        )
        self.assertTrue(torch.equal(canonicalize_jaw_axial(values, side), expected))


if __name__ == "__main__":
    unittest.main()
