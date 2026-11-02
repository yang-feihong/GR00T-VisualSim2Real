"""Migrate URDF arm/gripper friction to effort units with the requested scale."""

import torch

ARM_GRIPPER_COULOMB_FRICTION_SCALE = 0.5


def restore_urdf_friction_efforts(robot, joint_ids):
    # Isaac Sim's legacy importer interprets URDF Nm values as load-scaled
    # coefficients; move these to the explicit effort-valued properties once.
    view = robot.root_physx_view
    legacy = view.get_dof_friction_coefficients().clone()
    ids_cpu = joint_ids.cpu()
    friction = legacy[:, ids_cpu].to(robot.device)
    if not torch.isfinite(friction).all() or (friction < 0).any():
        raise RuntimeError("Invalid imported URDF joint friction")
    properties = view.get_dof_friction_properties().to(robot.device)
    old = friction > 0
    # Scale on migration only, so retrying initialization cannot halve twice.
    scaled = friction * ARM_GRIPPER_COULOMB_FRICTION_SCALE
    static = torch.where(old, scaled, properties[:, joint_ids, 0])
    dynamic = torch.where(old, scaled, properties[:, joint_ids, 1])
    viscous = properties[:, joint_ids, 2]
    legacy[:, ids_cpu] = 0.0
    view.set_dof_friction_coefficients(legacy, torch.arange(robot.num_instances, dtype=torch.int32))
    robot.write_joint_friction_coefficient_to_sim(static, dynamic, viscous, joint_ids=joint_ids)
