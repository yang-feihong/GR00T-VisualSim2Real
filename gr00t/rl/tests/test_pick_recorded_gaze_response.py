"""Recorded-scene counterfactuals for walking gaze posture robustness."""

import json
import math
import os
from pathlib import Path

import pytest
import torch

from gr00t.rl.tests.test_pick_recorded_batch_invariance import _recorded_batch


def walking_band_inverse(
    p,
    q,
    template,
    roots,
    times,
    active,
    limits,
    max_requests,
    collision=None,
    geometry=None,
    settled=None,
):
    from gr00t.rl.auto_open.pose_planning import solve_pose_grid

    shape = p.shape[:-1]
    body = roots[None].expand(3, *roots.shape).clone()
    body[..., 2] += (
        p.new_tensor([-0.03, 0.04, 0.10])[:, None, None, None]
        * (times / 1.5).clamp_max(1)[None, :, None]
    )
    result = solve_pose_grid(
        p.reshape(-1, 1, 3),
        q.reshape(-1, 1, 4),
        body.reshape(3, -1, 13),
        template[None].expand(3, *template.shape).reshape(3, -1, 6),
        limits,
        query_mask=active.reshape(-1, 1),
        max_requests=max_requests,
    )
    joints = result.arm_joint_positions[..., 0, :]
    margin = torch.minimum(joints - limits[:, 0], limits[:, 1] - joints).amin(-1)
    error = result.position_error[..., 0, :].norm(dim=-1).nan_to_num(nan=torch.inf)
    angle = result.orientation_error[..., 0, :].norm(dim=-1).nan_to_num(nan=torch.inf)
    error = error.masked_fill(margin < math.radians(2), torch.inf)
    values = (
        torch.where(active[..., None], joints[0].reshape(*shape, 6), template),
        body[0],
        error.amax(0).reshape(shape),
        angle.amax(0).reshape(shape),
    )
    if collision is None:
        return values
    batch, candidates, count = shape
    checked_roots = body.permute(1, 2, 0, 3, 4).flatten(1, 2)
    checked_joints = joints.reshape(3, *shape, 6).permute(1, 2, 0, 3, 4).flatten(1, 2)
    checked_joints = torch.where(
        active[:, :, None].expand(-1, -1, 3, -1).flatten(1, 2)[..., None],
        checked_joints,
        template[:, :, None].expand(-1, -1, 3, -1, -1).flatten(1, 2),
    )
    clearance = (
        collision.clearance(geometry, checked_roots, checked_joints, maximum_distance_m=0.0201)
        .reshape(batch, candidates, 3, count)
        .amin(2)
    )
    return (*values, clearance)


@pytest.mark.skipif(not os.environ.get("PICK_RECORDED_GAZE_ROOT"), reason="Recorded probe required")
def test_recorded_walking_gaze_height_band_counterfactual(monkeypatch, tmp_path):
    from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory
    from gr00t.rl.auto_open.planning_geometry import _quat_multiply
    from gr00t.rl.pick import gaze

    source = Path(os.environ["PICK_RECORDED_GAZE_ROOT"])
    state, _, _, _, _, geometry, collision, limits = _recorded_batch(source, count=4)
    paths = sorted(
        source.glob("logs/rank*-control/seed_*/control_record.pt"),
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
    )
    episodes = [json.loads(path.with_name("episode.json").read_text()) for path in paths]
    summaries = [json.loads(path.with_name("summary.json").read_text()) for path in paths]
    records = [torch.load(path, weights_only=False) for path in paths]
    batch = len(paths)
    targets = state["object_state"].new_tensor([e["object_position"] for e in episodes])
    offsets = gaze.sample_approach_offsets_batch([e["seed"] for e in episodes], "cuda")
    starts = torch.cat(
        (
            state["root_states"][:, :2],
            offsets.new_tensor([e["base_yaw"] for e in episodes])[:, None],
        ),
        -1,
    )
    ends = offsets.new_tensor(
        [
            [*s["planning"]["approach_gaze"]["base_path_end_m"][:2], e["station_yaw"]]
            for s, e in zip(summaries, episodes, strict=True)
        ]
    )
    # Failed station search retains the episode target, including its yaw noise.
    for row, summary in enumerate(summaries):
        detail = summary["planning"]["initial_station"]
        if detail["selected"] >= 0:
            ends[row] = offsets.new_tensor(detail["station_poses"][detail["selected"]])
        else:
            ends[row, 2] += offsets[row, 0]
    base = BasePoseTrajectory(batch, "cuda")
    rows = torch.arange(batch, device="cuda")
    base.install(rows, starts, ends[:, None], 0.02)
    length = int(base.length.max())
    roots = state["root_states"][:, None].expand(-1, length, -1).clone()
    roots[..., :2] = base.poses[:, :length, :2]
    yaw = base.poses[:, :length, 2] - starts[:, None, 2]
    delta = torch.stack(((yaw / 2).cos(), yaw * 0, yaw * 0, (yaw / 2).sin()), -1)
    roots[..., 3:7] = _quat_multiply(delta, roots[..., 3:7])
    expected_lengths = base.length.new_tensor(
        [s["planning"]["approach_gaze"]["samples"] for s in summaries]
    )
    torch.testing.assert_close(base.length, expected_lengths)
    references = roots.new_tensor(
        [s["planning"]["initial_station"]["selected_endpoint_joints_rad"][0] for s in summaries]
    ).nan_to_num()
    reference_mask = torch.tensor(
        [s["planning"]["approach_gaze"]["wrist_route_enabled"] for s in summaries], device="cuda"
    )
    original = gaze._gaze_inverse_samples
    report = {"physics_simulated": False, "seeds": state["episode_seed"].tolist(), "cases": {}}
    actual_roots = torch.stack(
        [
            r["pre"]["root_states"][torch.arange(length).clamp_max(len(r["stages"]) - 1)]
            for r in records
        ]
    ).cuda()
    actual_joints = torch.stack(
        [r["pre"]["dof_pos"][torch.arange(length).clamp_max(len(r["stages"]) - 1)] for r in records]
    ).cuda()[..., collision.arm_joint_ids]
    for name, solver in (("current", original), ("walking_band", walking_band_inverse)):
        monkeypatch.setattr(gaze, "_gaze_inverse_samples", solver)
        with torch.inference_mode():
            p, q, detail = gaze.plan_approach_gaze_batch(
                roots,
                state["dof_pos"][:, collision.arm_joint_ids],
                targets,
                limits,
                0.02,
                collision,
                geometry,
                offsets[:, 1:],
                lengths=base.length,
                next_grasp_joints=references,
                next_grasp_mask=reference_mask,
            )
            from gr00t.rl.auto_open.pose_planning import solve_pose_grid

            _, _, templates, _, shoulder, rolls = gaze.natural_gaze_candidates(
                roots,
                state["dof_pos"][:, collision.arm_joint_ids],
                targets,
                limits,
                0.02,
                offsets[:, 1:],
            )
            matches = (
                shoulder[None] * (180 / math.pi) - detail["selected_q2_offset_deg"][:, None]
            ).abs() < 0.001
            matches &= (
                rolls[None] * (180 / math.pi) - detail["selected_optical_roll_offset_deg"][:, None]
            ).abs() < 0.001
            chosen_template = templates.reshape(batch, len(shoulder), length, 6)[
                rows, matches.long().argmax(-1)
            ]
            seeds = torch.stack(
                (
                    actual_joints,
                    limits.mean(-1).expand_as(actual_joints),
                    limits.sum(-1) - actual_joints,
                    chosen_template,
                )
            ).reshape(4, -1, 6)
            replay = solve_pose_grid(
                p.reshape(-1, 1, 3),
                q.reshape(-1, 1, 4),
                actual_roots.reshape(1, -1, 13).expand(4, -1, -1),
                seeds,
                limits,
            )
            pe = replay.position_error[..., 0, :].norm(dim=-1)
            ae = replay.orientation_error[..., 0, :].norm(dim=-1)
            good = (pe <= 0.02) & (ae <= 0.15)
            cost = pe + 0.1 * ae
            choice = torch.where(
                good.any(0), cost.masked_fill(~good, torch.inf).argmin(0), cost.argmin(0)
            )
            flat_rows = torch.arange(batch * length, device="cuda")
            error = pe[choice, flat_rows].reshape(batch, length)
            orientation_error = ae[choice, flat_rows].reshape(batch, length)
            active = torch.arange(length, device="cuda")[None] < base.length[:, None]
            settled = active & (torch.arange(length, device="cuda")[None] >= 75)
            worst = error.masked_fill(~settled, -1).argmax(-1)
            root_delta = actual_roots[rows, worst, :3] - roots[rows, worst, :3]
        report["cases"][name] = {
            "valid": detail["valid"].tolist(),
            "selected_q2_offset_deg": detail["selected_q2_offset_deg"].tolist(),
            "selected_roll_deg": detail["selected_optical_roll_offset_deg"].tolist(),
            "selected_end_m": detail["selected_path_end_m"].tolist(),
            "predicted_end_joints": detail["predicted_gaze_end_joints_rad"].tolist(),
            "max_recorded_body_position_error_m": error.masked_fill(~settled, 0).amax(-1).tolist(),
            "recorded_body_frames_above_2cm": ((error > 0.02) & settled).sum(-1).tolist(),
            "max_recorded_body_orientation_error_rad": (2 * orientation_error.clamp_max(1).asin())
            .masked_fill(~settled, 0)
            .amax(-1)
            .tolist(),
            "recorded_body_frames_outside_pose_tolerance": (
                ((error > 0.02) | (orientation_error > 0.15)) & settled
            )
            .sum(-1)
            .tolist(),
            "worst_frame": worst.tolist(),
            "worst_actual_minus_planned_root_m": root_delta.tolist(),
            "worst_actual_root_quat": actual_roots[rows, worst, 3:7].tolist(),
            "worst_planned_root_quat": roots[rows, worst, 3:7].tolist(),
            "worst_target_m": p[rows, worst].tolist(),
        }
        del p, q
    report["limitations"] = "Hypothetical walking-height band, no physical policy response proof."
    (tmp_path / "recorded_gaze_response.json").write_text(json.dumps(report, indent=2) + "\n")
