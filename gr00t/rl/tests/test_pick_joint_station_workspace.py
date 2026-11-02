"""Bounded GPU station workspace diagnosis at conservative body postures."""

import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not os.environ.get("PICK_WORKSPACE_RECORD"), reason="Records required")
def test_joint_station_workspace(tmp_path):
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
    from gr00t.rl.auto_open.pose_planning import solve_pose_grid
    from gr00t.rl.auto_open.tensor_random import episode_uniform
    from gr00t.rl.pick.station import station_corridors_clear
    from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch

    path = Path(os.environ["PICK_WORKSPACE_RECORD"])
    state, _, sizes, support, mask, geometry, collision, limits = _recorded_batch(path, count=4)
    row = 2
    report = json.loads(next(path.glob("logs/rank2-control/seed_*/summary.json")).read_text())
    selected = report["planning"]["initial_station"]
    goal = limits.new_tensor(selected["selected_grasp_goal_world_m"])
    quat = limits.new_tensor(selected["selected_orientation"])
    samples = episode_uniform(state["episode_seed"][row : row + 1], 0x57534B, 1000, lanes=4)[0]
    goals = goal.expand(1000, 3)
    quats = quat.expand(1000, 4)
    if os.environ.get("PICK_WORKSPACE_ALL_GRASPS"):
        from gr00t.rl.pick.planning import grasp_candidate_bank

        indices = torch.tensor([row], device="cuda")
        bank = grasp_candidate_bank(
            {key: value[row : row + 1] for key, value in state.items()},
            state["object_state"][row : row + 1, :3],
            sizes[row : row + 1],
            collision,
            geometry.select(indices),
            support[row : row + 1],
            support_mask=mask[row : row + 1],
            preferred_orientation=quat[None],
            retain_depth_candidates=True,
        )
        physical = (
            bank[2]["pocket_fit"][0]
            & (bank[2]["antipodal_margin"][0] >= 0)
            & (bank[2]["environment_clearance_m"][0] >= 0.002)
        )
        available = torch.where(physical)[0]
        choice = available[(samples[:, 3] * len(available)).long()]
        goals, quats = bank[1][0, choice], bank[0][0, choice]
    angle = selected["selected_station"][2] + (samples[:, 0] * 2 - 1) * torch.pi / 2
    radius = 0.35 + samples[:, 1] * 0.50
    xy = goals[:, :2] - radius[:, None] * torch.stack((angle.cos(), angle.sin()), -1)
    yaw = angle + (samples[:, 2] * 2 - 1) * torch.pi / 12
    stations = torch.cat((xy, yaw[:, None]), -1)
    corridor = station_corridors_clear(
        state["root_states"][row : row + 1],
        stations[None],
        geometry.select(torch.tensor([row], device="cuda")),
    )[0]
    bodies = state["root_states"][row].expand(3, 1000, -1).clone()
    bodies[..., :2] = xy
    bodies[..., 2] = limits.new_tensor([0.34, 0.37, 0.38])[:, None]
    bodies[..., 3:7] = _euler_xyz_quat(
        torch.stack((yaw * 0, yaw * 0 + 8 * torch.pi / 180, yaw), -1)
    )[None]
    actual = state["dof_pos"][row, collision.arm_joint_ids]
    seeds = torch.stack((actual, limits.mean(-1), limits.sum(-1) - actual))
    seeds = seeds[:, None, None].expand(-1, 3, 1000, -1).reshape(9, 1000, 6)
    result = solve_pose_grid(
        goals[:, None],
        quats[:, None],
        bodies[None].expand(3, -1, -1, -1).reshape(9, 1000, 13),
        seeds,
        limits,
    )
    exact = result.exact[:, :, 0].reshape(3, 3, 1000)
    margin = (
        torch.minimum(
            result.arm_joint_positions - limits[:, 0], limits[:, 1] - result.arm_joint_positions
        )
        .amin(-1)[:, :, 0]
        .reshape(3, 3, 1000)
    )
    valid = exact & corridor[None, None] & (margin >= 0.02)
    score = (xy - state["root_states"][row, :2]).norm(dim=-1)[None, None] - 0.1 * margin
    output = {
        "heights": [0.34, 0.37, 0.38],
        "corridor_count": corridor.sum().item(),
        "exact_counts": exact.any(0).sum(-1).tolist(),
        "safe_counts": valid.any(0).sum(-1).tolist(),
    }
    for h in range(3):
        values = torch.where(valid[:, h])
        indices = values[1]
        order = score[:, h][values].argsort()[:5]
        output[str(h)] = {
            "stations": stations[indices[order]].tolist(),
            "goals": goals[indices[order]].tolist(),
            "quats": quats[indices[order]].tolist(),
            "joints": result.arm_joint_positions.reshape(3, 3, 1000, 1, 6)[
                values[0][order], h, indices[order], 0
            ].tolist(),
        }
    (tmp_path / "workspace.json").write_text(json.dumps(output, indent=2))
    print(output, flush=True)
