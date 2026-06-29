"""Small IsaacGym torch_utils-compatible helper module.
Quaternion convention in these helpers is **wxyz**, matching IsaacLab/IsaacSim.
"""
from __future__ import annotations
import math
import torch


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    # q = [w, x, y, z]
    return torch.cat((q[..., 0:1], -q[..., 1:4]), dim=-1)


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack((
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ), dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = normalize(q)
    q_xyz = q[..., 1:]
    q_w = q[..., 0:1]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def quat_from_euler_xyz(roll, pitch, yaw):
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5); sy = torch.sin(yaw * 0.5)
    return torch.stack((
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ), dim=-1)


def euler_from_quat(q: torch.Tensor):
    # q = [w, x, y, z]
    q = normalize(q)
    w, x, y, z = q.unbind(-1)
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2 * (x*x + y*y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w*y - z*x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def orientation_error(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    q_r = quat_mul(desired, quat_conjugate(current))
    return q_r[..., 1:4] * torch.sign(q_r[..., 0:1])


def get_axis_params(value: float, axis_idx: int):
    out = [0.0, 0.0, 0.0]
    out[axis_idx] = value
    return out


def to_torch(x, device='cpu', dtype=torch.float, requires_grad=False):
    return torch.tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def wrap_to_pi(angles):
    return (angles + math.pi) % (2 * math.pi) - math.pi


def torch_wrap_to_pi_minuspi(angles):
    return wrap_to_pi(angles)


def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def torch_rand_sqrt_float(lower, upper, shape, device):
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.0) / 2.0
    return (upper - lower) * r + lower
