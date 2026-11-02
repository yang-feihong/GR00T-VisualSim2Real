"""Arm-only gravity feedforward, aligned with visual wholebody lab30 c32e252."""

import torch


class ArmGravityCompensation:
    def __init__(self, robot, joint_ids, effort_limits):
        self.view = robot.root_physx_view
        self.joint_ids = joint_ids
        self.limits = effort_limits[joint_ids]
        offset = 0 if robot.is_fixed_base else 6
        self.force_ids = joint_ids + offset
        forces = self.view.get_gravity_compensation_forces()
        expected = (robot.num_instances, robot.num_joints + offset)
        if tuple(forces.shape) != expected or not torch.isfinite(forces).all():
            raise RuntimeError(
                f"Invalid gravity force layout {tuple(forces.shape)}; expected {expected}"
            )

    def apply(self, torques, enabled):
        if enabled:
            # PhysX returns compensating torque, not gravitational load.
            gravity = self.view.get_gravity_compensation_forces()[:, self.force_ids]
            torques[:, self.joint_ids] = torch.clamp(gravity, -self.limits, self.limits)
        else:
            # The leg-only fast path reuses its torque buffer between substeps.
            torques[:, self.joint_ids] = 0.0
