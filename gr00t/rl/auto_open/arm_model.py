"""Static arm metadata, without constructing a CPU kinematics/solver model."""

import xml.etree.ElementTree as ET

import numpy as np

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))


def read_arm_joint_limits(urdf_path, *, margin_rad=1.0e-3):
    """Read the same URDF bounds and inward margin used by GPU retract IK."""
    root = ET.parse(urdf_path).getroot()
    limits = []
    for name in ARM_JOINT_NAMES:
        limit = root.find(f"joint[@name='{name}']/limit")
        if limit is None:
            raise ValueError(f"URDF is missing arm joint limits: {name}")
        limits.append((float(limit.attrib["lower"]), float(limit.attrib["upper"])))
    limits = np.asarray(limits, dtype=np.float64)
    limits[:, 0] += margin_rad
    limits[:, 1] -= margin_rad
    if not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Invalid arm joint limit range or margin")
    limits.setflags(write=False)
    return limits
