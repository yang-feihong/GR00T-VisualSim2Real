from __future__ import annotations

import torch


ROOT_POS = (-5.0, 0.0, 0.5)
VELOCITY_COMMAND = (0.0, 0.0, 0.0)
EE_GOAL_CART = (0.4619397663, 0.0, 0.1913417162)
JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", "joint1",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint", "joint2",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "joint3", "joint4", "joint5", "joint6", "jointGripper",
)
JOINT_POS = (
    0.2, -0.2, 0.2, -0.2, 0.0,
    0.8, 0.8, 0.8, 0.8, 1.48,
    -1.5, -1.5, -1.5, -1.5, -0.63, -0.84, 0.0, 1.57, -0.785,
)
POLICY_ACTION = (
    0.0480948463, -0.2489279509, 0.0248126425,
    -0.0960438848, -0.3345853686, -0.3029152751,
    -0.0759676322, 0.2507962584, -0.4769482613,
    -0.0119658224, 0.3151202202, -0.4884460866,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)


def tensor(values, device):
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)


def joint_state_by_name(dof_names, values, device):
    """Map a named fixture into an articulation's native joint order."""
    value_by_name = dict(zip(JOINT_NAMES, values, strict=True))
    missing = set(dof_names) - value_by_name.keys()
    extra = value_by_name.keys() - set(dof_names)
    if missing or extra:
        raise RuntimeError(
            f"B2Z1 fixture joint mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return tensor([value_by_name[name] for name in dof_names], device)


def snapshot(value: torch.Tensor) -> torch.Tensor:
    """Copy mutable Isaac Lab buffers into a stable CPU trace record."""
    return value.detach().cpu().clone()


def canonical_root_state(root_state_wxyz: torch.Tensor) -> torch.Tensor:
    root = root_state_wxyz.clone()
    root[:, 3:7] = root_state_wxyz[:, [4, 5, 6, 3]]
    return root
