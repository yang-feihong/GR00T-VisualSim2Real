"""Shared Z1 position-drive defaults, matching visual wholebody lab30 c32e252."""

Z1_STIFFNESS = {
    "joint1": 264.0 / 30.0 * 25.6,
    "joint2": 328.0 / 30.0 * 25.6,
    "joint3": 264.0 / 30.0 * 25.6,
    "joint4": 264.0 / 30.0 * 25.6,
    "joint5": 264.0 / 30.0 * 25.6,
    "joint6": 264.0 / 30.0 * 25.6,
    "jointGripper": 100.0 / 30.0 * 25.6,
}

Z1_DAMPING = {
    "joint1": 1.5,
    "joint2": 3.0,
    "joint3": 1.5,
    "joint4": 1.5,
    "joint5": 1.5,
    "joint6": 1.5,
    "jointGripper": 1.0,
}
