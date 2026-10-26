"""Standard direct-pose and candidate/path-bank planning on the input device.

Tasks generate geometry and feasibility predicates; they must not implement
another inverse or dispatch one inverse call per candidate or path point.
"""

import torch


def solve_pose_targets(*args, **kwargs):
    from legged_gym.utils.ee_steady_inverse_gpu import batched_steady_ee_inverse_no_gravity

    return batched_steady_ee_inverse_no_gravity(*args, **kwargs)


def first_feasible(feasible):
    """First valid entry of an already ordered candidate bank; no host reads."""
    return feasible.to(torch.int64).argmax(-1), feasible.any(-1)


def best_feasible(feasible, cost):
    """Minimum-cost valid entry, preserving the first index on ties."""
    return cost.masked_fill(~feasible, torch.inf).argmin(-1), feasible.any(-1)


class PosePlanningMixin:
    def _inverse_pose_bank(
        self,
        fsm,
        device_state,
        env_ids,
        desired_positions,
        desired_orientations,
        lengths,
        *,
        initial_arm_joint_positions=None,
    ):
        rows = torch.as_tensor(env_ids, dtype=torch.int64, device=fsm.device)
        arms = device_state["dof_pos"][rows][:, self.arm_joint_ids]
        roots = device_state["root_states"][rows]
        owner, sample = torch.where(
            torch.arange(desired_positions.shape[1], device=fsm.device)[None, :] < lengths[:, None]
        )
        result = solve_pose_targets(
            desired_positions[owner, sample],
            desired_orientations[owner, sample],
            base_position_world=roots[owner, :3],
            base_orientation_world_xyzw=roots[owner][:, [4, 5, 6, 3]],
            initial_arm_joint_positions=(
                arms[owner]
                if initial_arm_joint_positions is None
                else initial_arm_joint_positions[owner, sample]
            ),
            joint_limits=self.joint_limits,
        )
        return result, owner, sample
