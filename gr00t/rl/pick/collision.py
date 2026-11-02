"""Conservative arm collider samples transformed by the shared B2Z1 kinematics."""

import math

import numpy as np
import torch
from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

from gr00t.rl.auto_open.first_pull_wall import _asset_envelopes
from gr00t.rl.auto_open.geometry import _read_collider_points
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.envs.door.b2z1_reset_geometry import (
    _ARM_JOINT_AXES,
    _arm_offset_tensors,
    _axis_rotation,
    _fixed_matmul_3x3,
    _fixed_matvec_3,
    quaternion_xyzw_to_matrix,
)
from gr00t.rl.utils.b2z1_asset import read_arm_mount


def moving_jaw_in_wrist(points, aperture_rad=torch.pi / 3):
    """Use deployed joint frames, independent of any environment's measured jaw pose."""
    from gr00t.rl.utils.b2z1_asset import ROBOT_USD

    _, first, second, axis = _asset_envelopes(str(ROBOT_USD))
    first, second = points.new_tensor(first), points.new_tensor(second)
    rotation = _axis_rotation(points.new_tensor([-aperture_rad]), axis, points.dtype)[0]
    position = first[:3, 3] + first[:3, :3] @ rotation @ second[:3, 3]
    rotation = first[:3, :3] @ rotation @ second[:3, :3]
    return points @ rotation.T + position


class ArmCollisionSamples:
    def __init__(self, env, spacing_m=0.025):
        self._populate(
            env.sim.stage,
            env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot"),
            env.device,
            env.arm_joint_ids,
            spacing_m,
        )

    @classmethod
    def from_robot_usd(cls, *, device, spacing_m=0.025):
        """Read the deployed asset without starting a simulator or borrowing an episode."""
        from pxr import Usd

        from gr00t.rl.utils.b2z1_asset import ROBOT_USD

        stage = Usd.Stage.Open(str(ROBOT_USD))
        if stage is None or not stage.GetDefaultPrim():
            raise ValueError(f"Deployed robot has no valid root: {ROBOT_USD}")
        model = cls.__new__(cls)
        model._populate(
            stage, stage.GetDefaultPrim(), device, torch.arange(6, device=device), spacing_m
        )
        return model

    def _populate(self, stage, root, device, arm_joint_ids, spacing_m):
        from pxr import Usd

        self.parts = []
        self.gripper_surface = []
        self.arm_joint_ids = arm_joint_ids
        for index in range(7):
            name = f"link0{index}"
            body = next(p for p in Usd.PrimRange(root) if p.GetName() == name)
            geom = _read_collider_points(stage, body, ("/collisions/",))
            if geom is None:
                continue
            lower, upper = geom.bounds()
            if index == 6:
                self.gripper_surface.append(
                    torch.tensor(geom.points_body, device=device, dtype=torch.float32)
                )
            axes = [
                np.linspace(a, b, max(2, int(np.ceil((b - a) / spacing_m)) + 1))
                for a, b in zip(lower, upper)
            ]
            cloud = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            self.parts.append((index, torch.tensor(cloud, device=device, dtype=torch.float32)))
        # Open moving jaw is rigid relative to link06 throughout approach/insertion.
        mover = next(p for p in Usd.PrimRange(root) if p.GetName() == "gripperMover")
        geom = _read_collider_points(stage, mover, ("/collisions/",))
        points = torch.tensor(geom.points_body, device=device, dtype=torch.float32)
        local = moving_jaw_in_wrist(points)
        self.parts.append((6, local))
        self.gripper_surface.append(local)

    def link_transforms(self, roots, joints):
        rotation = quaternion_xyzw_to_matrix(roots[:, [4, 5, 6, 3]])
        offsets = _arm_offset_tensors(roots.dtype, roots.device)
        position = roots[:, :3] + _fixed_matvec_3(rotation, offsets[0])
        yaw = torch.full_like(joints[:, 0], read_arm_mount()[1])
        rotation = _fixed_matmul_3x3(rotation, _axis_rotation(yaw, 2, roots.dtype))
        transforms = [(position, rotation)]
        for i, axis in enumerate(_ARM_JOINT_AXES):
            position = position + _fixed_matvec_3(rotation, offsets[i + 1])
            rotation = _fixed_matmul_3x3(rotation, _axis_rotation(joints[:, i], axis, roots.dtype))
            transforms.append((position, rotation))
        return transforms

    def world_points(self, roots, joints):
        transforms = self.link_transforms(roots, joints)
        clouds = []
        for index, local in self.parts:
            origin, orientation = transforms[index]
            clouds.append(torch.einsum("bij,kj->bki", orientation, local) + origin[:, None, :])
        return torch.cat(clouds, 1)

    def clearance(
        self,
        geometry,
        roots,
        joints,
        *,
        maximum_distance_m=2.0,
        include_gripper=True,
        query_mask=None,
    ):
        shape = roots.shape[:-1]
        if query_mask is not None:
            if query_mask.shape != shape or query_mask.device != roots.device:
                raise ValueError("Arm query masks must match the device-side pose grid")
            active = torch.where(query_mask.reshape(-1))[0]
            result = roots.new_full(shape, maximum_distance_m)
            if not len(active):
                return result
            selected = (
                geometry.select(active // math.prod(shape[1:]))
                if hasattr(geometry, "select")
                else geometry
            )
            values = self.clearance(
                selected,
                roots.reshape(-1, 13)[active, None],
                joints.reshape(-1, 6)[active, None],
                maximum_distance_m=maximum_distance_m,
                include_gripper=include_gripper,
            )
            result.reshape(-1)[active] = values.reshape(-1)
            return result
        transforms = self.link_transforms(roots.reshape(-1, 13), joints.reshape(-1, 6))
        minima = roots.new_full(shape, maximum_distance_m)
        for index, local in self.parts:
            if not include_gripper and index == 6:
                continue
            position, rotation = transforms[index]
            distance = geometry.rigid_surface_clearance(
                position.reshape(*shape, 3),
                _matrix_to_quaternion_wxyz(rotation).reshape(*shape, 4),
                local,
                check_rays=False,
                maximum_distance_m=maximum_distance_m,
                **({} if query_mask is None else {"query_mask": query_mask}),
            )
            minima = torch.minimum(minima, distance)
        return minima

    def _gripper_local_surfaces(self, positions):
        if hasattr(self, "_local_gripper_surfaces"):
            return self._local_gripper_surfaces
        if not hasattr(self, "gripper_surface"):
            from pxr import Usd

            from gr00t.rl.utils.b2z1_asset import ROBOT_USD

            stage = Usd.Stage.Open(str(ROBOT_USD))
            body = next(p for p in stage.Traverse() if p.GetName() == "link06")
            geom = _read_collider_points(stage, body, ("/collisions/",))
            self.gripper_surface = [
                torch.tensor(geom.points_body, device=positions.device, dtype=positions.dtype),
                self.parts[-1][1],
            ]
        points = self.gripper_surface[0]
        offsets = _arm_offset_tensors(points.dtype, points.device)
        wrist_to_tcp = offsets[7] + offsets[8]
        rotation = points.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        self._local_gripper_surfaces = tuple(
            torch.unique((points - wrist_to_tcp) @ rotation, dim=0)
            for points in self.gripper_surface
        )
        self._combined_gripper_surface = torch.unique(torch.cat(self._local_gripper_surfaces), dim=0)
        return self._local_gripper_surfaces

    def gripper_clearance(
        self, geometry, positions, orientations, *, query_mask=None, maximum_distance_m=None
    ):
        self._gripper_local_surfaces(positions)
        local = self._combined_gripper_surface
        options = {} if query_mask is None else {"query_mask": query_mask}
        if maximum_distance_m is not None:
            options["maximum_distance_m"] = maximum_distance_m
        distance = geometry.rigid_surface_clearance(positions, orientations, local, **options)
        blocked = self._moving_envelope_intersects(geometry, positions, orientations, query_mask)
        return distance.masked_fill(blocked, -1.0)

    def _moving_envelope_intersects(self, geometry, positions, orientations, query_mask=None):
        """Conservative jaw-volume edges catch furniture between sparse surface samples."""
        if not hasattr(self, "_moving_envelope_corners"):
            from gr00t.rl.utils.b2z1_asset import ROBOT_USD

            bounds = positions.new_tensor(_asset_envelopes(str(ROBOT_USD))[0][-1])
            corners = torch.cartesian_prod(*[bounds[:, axis] for axis in range(3)])
            wrist = moving_jaw_in_wrist(corners)
            offsets = _arm_offset_tensors(positions.dtype, positions.device)
            self._moving_envelope_corners = (
                wrist - offsets[7] - offsets[8]
            ) @ positions.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        if not hasattr(self, "_moving_envelope_edges"):
            self._moving_envelope_edges = torch.tensor(
                [
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [1, 3],
                    [1, 5],
                    [2, 3],
                    [2, 6],
                    [3, 7],
                    [4, 5],
                    [4, 6],
                    [5, 7],
                    [6, 7],
                ],
                device=positions.device,
            )
        world = positions[..., None, :] + _quat_rotate(
            orientations[..., None, :], self._moving_envelope_corners.to(positions)
        )
        starts = world[..., self._moving_envelope_edges[:, 0], :]
        ends = world[..., self._moving_envelope_edges[:, 1], :]
        if query_mask is not None:
            starts = torch.where(query_mask[..., None, None], starts, 0)
            ends = torch.where(query_mask[..., None, None], ends, 0)
        blocked = ~geometry.visible(starts, ends, tolerance_m=0.0005).all(-1)
        return blocked if query_mask is None else blocked & query_mask

    def gripper_pose_clearance(
        self,
        geometry,
        positions,
        orientations,
        aperture,
        *,
        query_mask=None,
        maximum_distance_m=None,
    ):
        """Actual held aperture, not the open-to-closed approach sweep."""
        from gr00t.rl.auto_open.motion_math import _quat_multiply

        fixed, moving = self._gripper_local_surfaces(positions)
        delta = torch.pi / 3 - aperture
        rotation = torch.stack(((delta / 2).cos(), delta * 0, (delta / 2).sin(), delta * 0), -1)
        options = {} if query_mask is None else {"query_mask": query_mask}
        if maximum_distance_m is not None:
            options["maximum_distance_m"] = maximum_distance_m
        fixed_clear = geometry.rigid_surface_clearance(positions, orientations, fixed, **options)
        moving_clear = geometry.rigid_surface_clearance(
            positions, _quat_multiply(orientations, rotation), moving, **options
        )
        distance = torch.minimum(fixed_clear, moving_clear)
        blocked = self._moving_envelope_intersects(
            geometry, positions, _quat_multiply(orientations, rotation), query_mask
        )
        return distance.masked_fill(blocked, -1.0)

    def gripper_closing_clearance(
        self,
        geometry,
        positions,
        orientations,
        aperture,
        *,
        query_mask=None,
        maximum_distance_m=None,
    ):
        from gr00t.rl.auto_open.motion_math import _quat_multiply

        if not hasattr(self, "_closing_gripper_surface"):
            self._gripper_local_surfaces(positions)
            points = self.gripper_surface[1]
            offsets = _arm_offset_tensors(points.dtype, points.device)
            self._closing_gripper_surface = torch.unique(
                (points - offsets[7] - offsets[8]) @ points.new_tensor(
                    [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
                ), dim=0,
            )
        local = self._closing_gripper_surface
        weight = torch.linspace(0, 1, 7, device=local.device)
        delta = (torch.pi / 3 - aperture)[..., None] * weight
        rotation = torch.stack(
            (
                (delta / 2).cos(),
                delta * 0,
                (delta / 2).sin(),
                delta * 0,
            ),
            -1,
        )
        quats = _quat_multiply(orientations[..., None, :], rotation)
        positions = positions[..., None, :].expand_as(quats[..., :3])
        options = (
            {}
            if query_mask is None
            else {"query_mask": query_mask[..., None].expand_as(delta).flatten(1, 2)}
        )
        if maximum_distance_m is not None:
            options["maximum_distance_m"] = maximum_distance_m
        clearance = geometry.rigid_surface_clearance(
            positions.flatten(1, 2), quats.flatten(1, 2), local, **options
        )
        blocked = self._moving_envelope_intersects(
            geometry,
            positions.flatten(1, 2),
            quats.flatten(1, 2),
            None if query_mask is None else query_mask[..., None].expand_as(delta).flatten(1, 2),
        )
        clearance = clearance.masked_fill(blocked, -1.0)
        return clearance.reshape(*aperture.shape, len(weight)).amin(-1)
