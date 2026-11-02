"""Physical admission, including self-occlusion, before a randomized episode may run."""

import math

import torch

from gr00t.rl.auto_open.motion_math import (
    _euler_xyz_quat,
    _quat_multiply,
    _quat_rotate,
    _quat_rotate_inverse,
)
from gr00t.rl.pick.batched_geometry import BatchedCompositeGeometry, BatchedSceneGeometry
from gr00t.rl.pick.scene_geometry import SceneGeometry
from gr00t.rl.pick.scene_object_geometry import bind_rigid_frames


def robot_collision_meshes(env):
    """Load body-local collider triangles once; never read an environment's joint state."""
    from pxr import Usd

    from gr00t.rl.auto_open.geometry import _read_collider_points

    stage = env.sim.stage
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    names = {name: index for index, name in enumerate(env.body_names)}
    meshes = []
    for body in Usd.PrimRange(root):
        if body.GetName() not in names:
            continue
        collider = _read_collider_points(stage, body, ("/collisions/",))
        if collider is None or not len(collider.triangles_body):
            continue
        points = torch.as_tensor(
            collider.triangles_body.reshape(-1, 3), device=env.device, dtype=torch.float32
        )
        faces = torch.arange(len(points), device=env.device).reshape(-1, 3)
        meshes.append((names[body.GetName()], SceneGeometry(points, faces, device=env.device)))
    if not meshes:
        raise RuntimeError("Robot collider triangles unavailable for visibility admission")
    return meshes


class RobotOcclusionGeometryBatch:
    """Reuse body-local BVHs while querying each environment's actual rigid poses."""

    def __init__(self, env, *, meshes=None):
        self.count = env.num_envs
        self.body_meshes = tuple(robot_collision_meshes(env) if meshes is None else meshes)
        if not self.body_meshes:
            raise ValueError("Robot visibility requires collider meshes")
        origins = torch.zeros(self.count, 3, device=env.device)
        rows = torch.zeros(self.count, device=env.device, dtype=torch.long)
        self.parts = tuple(
            (index, BatchedSceneGeometry([mesh], rows, origins)) for index, mesh in self.body_meshes
        )
        self.geometry = BatchedCompositeGeometry([part for _, part in self.parts])

    def update(self, body_states):
        if body_states.ndim != 3 or body_states.shape[0] != self.count or body_states.shape[-1] < 7:
            raise ValueError("Robot visibility requires scene/body rigid-state tensors")
        if max(index for index, _ in self.parts) >= body_states.shape[1]:
            raise ValueError("Robot visibility body states omit a configured collider")
        for index, view in self.parts:
            bind_rigid_frames(view, body_states[:, index, :3], body_states[:, index, 3:7])
        return self.geometry


def initial_visibility(state, geometry, robot_geometry):
    root, target = state["root_states"], state["object_state"][:, :3]
    head = root[:, :3] + _quat_rotate(root[:, 3:7], root.new_tensor([[0.46, 0, 0.01]]))
    delta = target - head
    distance = delta.norm(dim=-1)
    forward = _quat_rotate(root[:, 3:7], root.new_tensor([[1, 0, 0]]))
    cosine = (delta[:, :2] * forward[:, :2]).sum(-1) / (
        delta[:, :2].norm(dim=-1) * forward[:, :2].norm(dim=-1)
    ).clamp_min(1e-6)
    mount = _euler_xyz_quat(root.new_tensor([[-math.pi / 2, 0, 0]]))
    camera_quat = _quat_multiply(root[:, 3:7], mount)
    local = _quat_rotate_inverse(camera_quat, delta)
    h_angle = torch.atan2(local[:, 1], local[:, 0]).abs()
    v_angle = torch.atan2(local[:, 2], local[:, 0]).abs()
    v_limit = math.atan(math.tan(math.radians(40)) * 480 / 640)
    in_frame = (
        (local[:, 0] > 0) & (h_angle < math.radians(38)) & (v_angle < v_limit - math.radians(2))
    )
    self_clear = robot_geometry.visible(head, target)
    static_clear = geometry.visible(head, target)
    return distance, cosine, in_frame, self_clear, static_clear
