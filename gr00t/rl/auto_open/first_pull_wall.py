"""Stage-entry GPU arm envelopes and timing for the first pull insertion."""

from functools import lru_cache

import torch
import torch.nn.functional as F

from gr00t.rl.auto_open.pull_entry_gpu import _nlerp
from gr00t.rl.envs.door.b2z1_reset_geometry import (
    _ARM_JOINT_AXES, _ARM_JOINT_ORIGINS, _axis_rotation,
    quaternion_xyzw_to_matrix,
)
from gr00t.rl.utils.b2z1_asset import read_arm_mount


@lru_cache(maxsize=4)
def _asset_envelopes(path):
    """Read static collision envelopes once, never per environment or frame."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from gr00t.rl.auto_open.pull_station import collider_local_corners

    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    names = [f'link{i:02d}' for i in range(7)] + ['gripperMover']
    bounds = []
    for name in names:
        body = next(p for p in stage.Traverse() if p.GetName() == name)
        inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        points = []
        for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
                continue
            xf = cache.GetLocalToWorldTransform(p)*inverse
            # Bounding boxes enclose meshes and capsules, not only their axes.
            points.extend(xf.Transform(Gf.Vec3d(*v)) for v in collider_local_corners(p))
        if not points:
            bounds.append(np.zeros((2, 3)))
        else:
            points = np.asarray(points)
            bounds.append(np.stack((points.min(0), points.max(0))))
    joint = next(p for p in stage.Traverse() if p.GetName() == 'jointGripper')
    frames = []
    for side in (0, 1):
        q = joint.GetAttribute(f'physics:localRot{side}').Get()
        frame = Gf.Matrix4d(1.)
        frame.SetRotate(Gf.Quatd(q))
        frame.SetTranslateOnly(Gf.Vec3d(joint.GetAttribute(f'physics:localPos{side}').Get()))
        frames.append(np.asarray(frame).T)
    return np.asarray(bounds), frames[0], np.linalg.inv(frames[1]), 'XYZ'.index(joint.GetAttribute('physics:axis').Get())


class FirstPullArmWallGPU:
    def __init__(self, *, device, asset_path=None):
        from gr00t.rl.utils.b2z1_asset import ROBOT_USD
        bounds, first, second, self.gripper_axis = _asset_envelopes(str(asset_path or ROBOT_USD))
        self.bounds = torch.as_tensor(bounds, device=device, dtype=torch.float64)
        self.first = torch.as_tensor(first, device=device, dtype=torch.float64)
        self.second = torch.as_tensor(second, device=device, dtype=torch.float64)
        mount, yaw = read_arm_mount(asset_path)
        self.mount = torch.tensor(mount, device=device, dtype=torch.float64)
        self.mount_yaw = yaw

    def clearance(self, roots, arms, gripper, hinge, normal, tangent, frame_edge, wall_front,
                  *, panel_plane=False):
        """Lower bound against both side walls, using conservative link OBBs.

        Fixed-size joint loops are tensor batches; there are no per-env loops,
        host state copies, mesh point clouds or per-frame calls.
        """
        roots, arms = roots.double(), arms.double()
        rotation = quaternion_xyzw_to_matrix(roots[:, [4, 5, 6, 3]])
        position = roots[:, :3] + (rotation*self.mount[None, None, :]).sum(-1)
        rotation = rotation @ _axis_rotation(torch.full_like(arms[:, 0], self.mount_yaw), 2, arms.dtype)
        poses = [(position, rotation)]
        for i, (origin, axis) in enumerate(zip(_ARM_JOINT_ORIGINS, _ARM_JOINT_AXES)):
            offset = arms.new_tensor(origin)
            position = position + (rotation*offset[None, None, :]).sum(-1)
            rotation = rotation @ _axis_rotation(arms[:, i], axis, arms.dtype)
            poses.append((position, rotation))
        position = position + (rotation*self.first[None, :3, 3][:, None, :]).sum(-1)
        rotation = rotation @ self.first[:3, :3]
        rotation = rotation @ _axis_rotation(gripper.reshape(-1).double(), self.gripper_axis, arms.dtype)
        position = position + (rotation*self.second[None, :3, 3][:, None, :]).sum(-1)
        rotation = rotation @ self.second[:3, :3]
        poses.append((position, rotation))
        gaps = []
        for i, (position, rotation) in enumerate(poses):
            center = self.bounds[i].mean(0)
            half = (self.bounds[i, 1]-self.bounds[i, 0])/2.
            world_center = position + (rotation*center[None, None, :]).sum(-1)-hinge.double()
            projected_n = (normal.double()[:, :, None]*rotation).sum(1)
            projected_t = (tangent.double()[:, :, None]*rotation).sum(1)
            half_n = (projected_n.abs()*half).sum(-1)
            half_t = (projected_t.abs()*half).sum(-1)
            n = (world_center*normal).sum(-1)
            t = (world_center*tangent).sum(-1)
            # The opening is between hinge and free frame edge. Requiring an
            # entire link box outside the wall or inside the opening is safe
            # but conservative at the wall corner; it is not exact mesh SAT.
            outside = n-half_n-wall_front
            inside = torch.minimum(t-half_t, frame_edge-t-half_t)
            gaps.append(outside if panel_plane else torch.maximum(outside, inside))
        return torch.stack(gaps, -1).amin(-1)


def retime_first_pull_paths(positions, orientations, joints, lengths, *, max_joint_step=.01,
                           max_position_step=None,max_rotation_step=None):
    """Slow only this stage's cached reference when joint motion needs time."""
    count, capacity, _ = positions.shape
    valid = torch.arange(capacity-1, device=positions.device)[None] < lengths[:, None]-1
    delta = (joints[:, 1:]-joints[:, :-1]).abs().amax(-1)
    required = torch.ceil(delta/max_joint_step).clamp_min(1)
    if max_position_step is not None:
        required = torch.maximum(required,torch.ceil(torch.linalg.vector_norm(
            positions[:,1:]-positions[:,:-1],dim=-1)/max_position_step))
    if max_rotation_step is not None:
        angle = 2*torch.acos((orientations[:,1:]*orientations[:,:-1]).sum(-1).abs().clamp(0.,1.))
        required = torch.maximum(required,torch.ceil(angle/max_rotation_step))
    steps = torch.where(valid,required.long(),0)
    ends = steps.cumsum(1)
    new_lengths = steps.sum(1)+1
    ticks = torch.arange(capacity, device=positions.device)[None].expand(count, -1)
    ticks = torch.minimum(ticks, (new_lengths-1)[:, None])
    segment = torch.searchsorted(ends.contiguous(), ticks.contiguous(), right=False).clamp_max(capacity-2)
    previous = F.pad(ends[:, :-1], (1, 0)).gather(1, segment)
    fraction = (ticks-previous).to(positions.dtype)/steps.gather(1, segment).clamp_min(1)
    def gather(values, index):
        return values.gather(1, index[..., None].expand(-1, -1, values.shape[-1]))
    p0, p1 = gather(positions, segment), gather(positions, segment+1)
    q0, q1 = gather(orientations, segment), gather(orientations, segment+1)
    p = p0+(p1-p0)*fraction[..., None]
    q = _nlerp(q0, q1, fraction)
    p[:, 0], q[:, 0] = positions[:, 0], orientations[:, 0]
    return p, q, new_lengths.clamp_max(capacity), new_lengths > capacity
