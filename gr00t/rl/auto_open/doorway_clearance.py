"""Door/centered-quadruped projection clearance, shared by push and pull."""

import math

import torch

THROUGH_PROJECTION_CLEARANCE_M = 0.10


class DoorwayProjection:
    def __init__(self, cosine, sine, robot_lower, robot_upper):
        self.cosine = cosine
        self.sine = sine
        self.robot_lower = robot_lower
        self.robot_upper = robot_upper

    @classmethod
    def from_world(cls, corners, hinge, tangent, hinge_sign, center, forward, robot_y_bounds):
        relative = corners[..., :2] - hinge[:, None, :2]
        cosine = (relative * tangent[:, None, :2]).sum(-1)
        perpendicular = torch.stack((-relative[..., 1], relative[..., 0]), -1)
        sine = (perpendicular * tangent[:, None, :2]).sum(-1) * hinge_sign[:, None]
        left = torch.stack((-forward[:, 1], forward[:, 0]), -1)
        direction = (left * tangent[:, :2]).sum(-1)
        offset = ((center[:, :2] - hinge[:, :2]) * tangent[:, :2]).sum(-1)
        bounds = offset[:, None] + direction[:, None] * robot_y_bounds
        return cls(cosine, sine, bounds.amin(-1), bounds.amax(-1))

    def gap(self, angle, rows):
        """Signed horizontal interval gap; negative means the projections overlap."""
        points = self.cosine[rows] * angle.cos()[:, None] + self.sine[rows] * angle.sin()[:, None]
        return torch.maximum(self.robot_lower[rows] - points.amax(-1),
                             points.amin(-1) - self.robot_upper[rows])

    def required_angle(self):
        """Forecast a planning endpoint; actual handoff must still check the gap."""
        rows = torch.arange(len(self.robot_lower), device=self.cosine.device)
        low = torch.zeros_like(self.robot_lower)
        high = torch.full_like(low, math.pi)
        for _ in range(24):
            middle = (low + high) * .5
            clear = self.gap(middle, rows) >= THROUGH_PROJECTION_CLEARANCE_M
            high = torch.where(clear, middle, high)
            low = torch.where(clear, low, middle)
        return high
