"""GPU-batched, speed-bounded planar pose trajectories for locomotion."""

from dataclasses import dataclass

import torch

from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.utils.batch_invariant_math import (
    fixed_norm,
)


@dataclass(frozen=True)
class BaseTrajectoryConfig:
    linear_speed_m_s: float = 0.25
    angular_speed_rad_s: float = 0.40


def wrapped_angle(angle):
    return torch.atan2(angle.sin(), angle.cos())


def frame_traversal_waypoints(start, frame_xy, forward, finish, robot_radius, margin_m=0.05):
    """Align outside the frame, then cross it; no separate FSM stage."""
    depth = ((frame_xy - start[:, :2]) * forward).sum(-1)
    depth = torch.maximum(depth, robot_radius + margin_m)
    retreat_xy = (
        start[:, :2] + (((frame_xy - start[:, :2]) * forward).sum(-1) - depth)[:, None] * forward
    )
    alignment_xy = frame_xy - depth[:, None] * forward
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    return torch.stack(
        (
            torch.cat((retreat_xy, start[:, 2:3]), -1),
            torch.cat((alignment_xy, heading[:, None]), -1),
            finish,
        ),
        dim=1,
    )


class BasePoseTrajectory:
    """Precompute once; sample on device and hold the exact final pose."""

    def __init__(self, num_envs, device, capacity=4096, config=BaseTrajectoryConfig()):
        self.config = config
        self.capacity = capacity
        self.poses = torch.zeros(num_envs, capacity, 3, device=device)
        self.velocities = torch.zeros_like(self.poses)
        self.length = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.index = torch.zeros_like(self.length)

    def clear(self, rows):
        self.length[rows] = 0
        self.index[rows] = 0

    def install(self, rows, start, waypoints, dt):
        if not len(rows):
            return
        vertices = torch.cat((start[:, None], waypoints), dim=1)
        delta = vertices[:, 1:] - vertices[:, :-1]
        delta[..., 2] = wrapped_angle(delta[..., 2])
        duration = torch.maximum(
            fixed_norm(delta[..., :2]) / self.config.linear_speed_m_s,
            delta[..., 2].abs() / self.config.angular_speed_rad_s,
        )
        steps = torch.ceil(duration / dt).to(torch.int64)
        total = steps.sum(-1)
        if torch.any(total >= self.capacity):
            raise ValueError("base trajectory exceeds sample capacity")
        times = torch.arange(self.capacity, device=start.device)[None]
        ends = steps.cumsum(-1)
        sample_times = times.expand(len(rows), -1).contiguous()
        segment = torch.searchsorted(ends, sample_times, right=True).clamp_max(
            waypoints.shape[1] - 1
        )
        count = steps.gather(1, segment)
        offset = (ends - steps).gather(1, segment)
        gather = segment[..., None].expand(-1, -1, 3)
        fraction = ((sample_times - offset) / count.clamp_min(1)).clamp(0.0, 1.0)
        poses = vertices[:, :-1].gather(1, gather) + fraction[..., None] * delta.gather(1, gather)
        poses[..., 2] = wrapped_angle(poses[..., 2])
        poses = torch.where((times >= total[:, None])[..., None], waypoints[:, -1, None], poses)
        velocity = (poses[:, 1:] - poses[:, :-1]) / dt
        velocity[..., 2] = wrapped_angle(poses[:, 1:, 2] - poses[:, :-1, 2]) / dt
        self.poses[rows] = poses
        self.velocities[rows] = 0.0
        self.velocities[rows, :-1] = velocity
        self.length[rows] = total + 1
        self.index[rows] = 0

    def sample(self, rows, *, advance=None):
        index = self.index[rows]
        pose = self.poses[rows, index]
        velocity = self.velocities[rows, index]
        complete = index >= self.length[rows] - 1
        if advance is None:
            advance = torch.ones_like(index, dtype=torch.bool)
        velocity = torch.where(advance[:, None], velocity, 0.0)
        self.index[rows] = torch.minimum(index + advance.to(torch.int64), self.length[rows] - 1)
        return pose, velocity, complete
