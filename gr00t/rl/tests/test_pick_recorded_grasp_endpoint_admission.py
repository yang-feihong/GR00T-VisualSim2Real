"""Replay grasp inverse admission from recorded candidate banks without physics."""

import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    not os.environ.get("PICK_GRASP_ADMISSION_ROOT"), reason="Recorded probe required"
)
def test_recorded_grasp_candidates_have_a_coherent_endpoint_solution(tmp_path):
    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.auto_open.pose_planning import solve_pose_grid, solve_pose_targets
    from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
    from gr00t.rl.pick.grasp_inverse import solve_default_grasp_inverse
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    source = Path(os.environ["PICK_GRASP_ADMISSION_ROOT"])
    report = {"physics_simulated": False, "episodes": []}
    for path in sorted(source.glob("logs/rank*-control/seed_*/control_record.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        summary = json.loads(path.with_name("summary.json").read_text())
        episode = json.loads(path.with_name("episode.json").read_text())
        detail = summary["planning"]["grasp"]
        if not detail.get("candidate_grasp_origins_world_m"):
            continue
        entries = [i for i, stage in enumerate(record["stages"]) if stage == "MOVE_TO_GRASP"]
        # Retries change the bank. Use this probe only for the first planning attempt.
        attempts = summary["planning"].get("grasp_attempts", [])
        if len(attempts) > 1:
            continue
        index = entries[0] if entries else len(record["stages"]) - 1
        snapshot = record["pre"] if entries else record["post"]
        state = {
            key: snapshot[key][index : index + 1].cuda()
            for key in ("root_states", "dof_pos", "object_state", "ee_pos", "ee_orn")
        }

        def tensor(value, dtype=torch.float32):
            return torch.tensor(value, device="cuda", dtype=dtype)

        orientations = tensor(detail["candidate_orientations_world_wxyz"])
        goals = tensor(detail["candidate_grasp_origins_world_m"])
        quality = {
            key: tensor(detail["convex_contact_model"][key], dtype)
            for key, dtype in (
                ("pocket_fit", torch.bool),
                ("antipodal_margin", torch.float32),
                ("environment_clearance_m", torch.float32),
            )
        }
        full_axes = _quat_rotate(orientations[0], tensor([1, 0, 0]))
        axis_z = full_axes[:, 2]
        fit = quality["pocket_fit"][0]
        antipodal = fit & (quality["antipodal_margin"][0] >= 0)
        clearance = quality["environment_clearance_m"][0]
        queried = tensor(
            detail["convex_contact_model"]["environment_query_evaluated"], torch.bool
        ).reshape_as(quality["pocket_fit"])[0]
        geometry_groups = {}
        for name, mask in (
            ("horizontal", axis_z.abs() <= 0.5),
            ("oblique_down", (axis_z < -0.5) & (axis_z > -0.8)),
            ("steep_down", axis_z <= -0.8),
        ):
            geometry_groups[name] = {
                "candidates": int(mask.sum()),
                "pocket_fit": int((mask & fit).sum()),
                "pocket_and_antipodal": int((mask & antipodal).sum()),
                "environment_queried": int((mask & queried).sum()),
                "environment_clear": int((mask & antipodal & (clearance >= 0.002)).sum()),
                "best_queried_clearance_m": (
                    None
                    if not bool((mask & queried).any())
                    else float(clearance.masked_fill(~(mask & queried), -torch.inf).max())
                ),
            }
        if os.environ.get("PICK_GRASP_GEOMETRY_ONLY"):
            report["episodes"].append({"seed": summary["seed"], "geometry_groups": geometry_groups})
            continue
        limits = tensor(read_arm_joint_limits(ROBOT_URDF).copy())
        fk, _, _ = b2z1_gripper_pose_and_jacobian(
            state["root_states"][:, :3],
            state["root_states"][:, [4, 5, 6, 3]],
            state["dof_pos"][:, record["arm_joint_ids"]],
        )
        self_inverse = solve_pose_targets(
            state["ee_pos"],
            state["ee_orn"],
            base_position_world=state["root_states"][:, :3],
            base_orientation_world_xyzw=state["root_states"][:, [4, 5, 6, 3]],
            initial_arm_joint_positions=state["dof_pos"][:, record["arm_joint_ids"]],
            joint_limits=limits,
        )
        with torch.inference_mode():
            inverse = solve_default_grasp_inverse(
                state,
                (orientations, goals, quality, None, None),
                limits,
                record["arm_joint_ids"].cuda(),
                tensor([episode["object_position"][2]]),
                0.02,
            )
        owner, candidate = torch.where(inverse["lookup"] >= 0)
        if not len(owner):
            continue
        endpoints = inverse["counts"][owner].cumsum(-1) - 1
        slots = torch.searchsorted(
            inverse["references"][owner].contiguous(), endpoints.contiguous()
        )
        raw = inverse["result"]
        endpoint_exact = raw["exact"].gather(
            -1, slots[None, None].expand(*raw["exact"].shape[:3], -1)
        )
        coherent = endpoint_exact[..., 1:].all(-1).any(0).any(0)
        selected = detail.get("selected")
        selected_slot = None if selected is None else int(inverse["lookup"][0, selected])
        pe = raw["position_error"].norm(dim=-1)
        endpoint_error = pe.gather(-1, slots[None, None].expand(*pe.shape[:3], -1))
        goal = goals[owner, candidate]
        lifted = goal.clone()
        lifted[:, 2] = (
            goal[:, 2]
            + (inverse["minimum_object_height"][owner] - state["object_state"][owner, 2]).clamp_min(
                0
            )
            + 0.005
        ).clamp_min(0.305)
        forward = state["object_state"][:, :2] - state["root_states"][:, :2]
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        withdrawn = lifted.clone()
        withdrawn[:, :2] -= inverse["withdraw_distance_m"][owner, None] * forward[owner]
        positions = torch.stack((goal, lifted, withdrawn), 1)
        rotations = orientations[owner, candidate, None].expand(-1, 3, -1)
        shifts = tensor([0, 0.05, 0.10, 0.15, 0.20, 0.25])
        heights = torch.cat(
            (
                tensor([0.18, 0.22, 0.30, 0.38, 0.46, 0.536]),
                state["root_states"][0, 2] + tensor([-0.01, 0, 0.01]),
            )
        )
        actual = state["dof_pos"][:, record["arm_joint_ids"]]
        seeds = torch.cat((actual, limits.mean(-1)[None], limits.sum(-1)[None] - actual))
        body = (
            state["root_states"][owner][None, None, None]
            .expand(len(shifts), len(heights), len(seeds), -1, -1)
            .clone()
        )
        body[..., :2] += shifts[:, None, None, None, None] * forward[None, None, None]
        body[..., 2] = heights[None, :, None, None]
        shifted = solve_pose_grid(
            positions,
            rotations,
            body,
            seeds[None, None, :, None].expand(len(shifts), len(heights), -1, len(owner), -1),
            limits,
        )
        shifted_support = shifted.exact.any(2).any(1).all(-1)
        axes = _quat_rotate(orientations[owner, candidate], tensor([1, 0, 0]))
        report["episodes"].append(
            {
                "seed": summary["seed"],
                "geometry_groups": geometry_groups,
                "entry_fk_vs_physics_position_error_m": float((fk - state["ee_pos"]).norm()),
                "entry_self_inverse_exact": bool(self_inverse.exact[0]),
                "recorded_selected": selected,
                "inverse_candidate_count": len(candidate),
                "geometrically_valid_tool_axis_world_z_range": [
                    float(axes[:, 2].min()),
                    float(axes[:, 2].max()),
                ],
                "coherent_endpoint_candidate_count": int(coherent.sum()),
                "selected_has_coherent_endpoint_solution": (
                    None
                    if selected_slot is None or selected_slot < 0
                    else bool(coherent[selected_slot])
                ),
                "selected_minimum_position_error_by_endpoint_m": (
                    None
                    if selected_slot is None or selected_slot < 0
                    else endpoint_error[:, :, selected_slot].amin((0, 1)).tolist()
                ),
                "forward_station_shift_probe": {
                    "collision_checked": False,
                    "forward_offsets_m": shifts.tolist(),
                    "endpoint_available_candidate_counts": shifted_support.sum(-1).tolist(),
                },
            }
        )
    (tmp_path / "grasp_endpoint_admission.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["episodes"], "No first-attempt grasp bank was available for replay"
