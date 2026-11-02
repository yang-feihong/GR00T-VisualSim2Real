"""Recorded observations test GPU planning independence without simulating physics."""

import json
import os
from pathlib import Path

import pytest
import torch


def _recorded_batch(root, *, count=8, frame=0):
    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
    from gr00t.rl.pick.collision import ArmCollisionSamples
    from gr00t.rl.pick.gaze import sample_approach_offsets_batch
    from gr00t.rl.pick.geometry_views import TranslatedGeometry
    from gr00t.rl.pick.grasp_quality import load_grasp_support
    from gr00t.rl.pick.scene_geometry import read_scene_triangles, static_scene_geometry
    from gr00t.rl.pick.scene_object_geometry import SceneObjectGeometryBatch
    from gr00t.rl.pick.tensor_rows import pad_device_rows
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    available = sorted(
        root.glob("logs/rank*-control/seed_*/control_record.pt"),
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
    )
    if count < 2 or len(available) < count:
        raise ValueError("Recorded batch regression has too few completed records")
    paths = [available[i * (len(available) - 1) // (count - 1)] for i in range(count)]
    episodes = [json.loads(path.with_name("episode.json").read_text()) for path in paths]
    records = [torch.load(path, weights_only=False)["pre"] for path in paths]
    state = {
        key: torch.stack([record[key][frame] for record in records]).cuda()
        for key in ("root_states", "object_state", "dof_pos", "ee_orn", "ee_pos")
    }
    state["distractor_states"], _ = pad_device_rows(
        [record["distractor_states"][frame].cuda() for record in records]
    )
    state["episode_seed"] = torch.tensor([e["seed"] for e in episodes], device="cuda")
    state["support_z"] = state["object_state"].new_tensor([e["support_z"] for e in episodes])
    state["on_floor"] = torch.tensor(
        [e["support_type"] == "floor" for e in episodes], device="cuda"
    )
    destination = state["object_state"].new_tensor(
        [[*e["station_xy"], e["station_yaw"]] for e in episodes]
    )
    destination[:, 2] += sample_approach_offsets_batch([e["seed"] for e in episodes], "cuda")[:, 0]
    sizes = destination.new_tensor(
        [e["asset"].get("intrinsic_size_m", e["asset"]["size_m"]) for e in episodes]
    )
    support, mask = pad_device_rows(
        [load_grasp_support(e["asset"]["usd_path"], "cuda") for e in episodes]
    )
    rooms, views = {}, []
    for episode in episodes:
        path = episode["scene_usd_path"]
        if path not in rooms:
            points, faces, labels = read_scene_triangles(
                Path(path), Path(episode["geometry_cache_root"]), return_labels=True
            )
            rooms[path] = static_scene_geometry(
                points, faces, "cuda", labels=labels, backend="native", ground_plane_z_m=0.0
            )
        views.append(
            TranslatedGeometry(rooms[path], destination.new_tensor(episode["batch_scene_origin_m"]))
        )
    geometry = SceneObjectGeometryBatch(episodes, BatchedSceneGeometry.from_views(views)).update(
        state
    )
    model = torch.load(paths[0].with_name("collision_model.pt"), weights_only=False)
    collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
    collision.parts = [(i, points.cuda()) for i, points in model["parts"]]
    collision.gripper_surface = [points.cuda() for points in model["gripper_surface"]]
    collision.arm_joint_ids = model["arm_joint_ids"].cuda()
    limits = torch.tensor(
        read_arm_joint_limits(ROBOT_URDF).copy(), device="cuda", dtype=torch.float32
    )
    return state, destination, sizes, support, mask, geometry, collision, limits


@pytest.mark.skipif(
    not os.environ.get("PICK_RECORDED_BATCH_ROOT"), reason="Recorded batch required"
)
def test_recorded_station_planning_matches_singletons_permutation_and_chunking(tmp_path):
    from gr00t.rl.pick.station_batch import select_initial_pick_station_batch

    state, destination, sizes, support, mask, geometry, collision, limits = _recorded_batch(
        Path(os.environ["PICK_RECORDED_BATCH_ROOT"])
    )

    def solve(rows, chunks):
        with torch.inference_mode():
            return select_initial_pick_station_batch(
                {key: value[rows] for key, value in state.items()},
                destination[rows],
                sizes[rows],
                support[rows],
                limits,
                collision,
                geometry.select(rows),
                support_mask=mask[rows],
                max_scenes=chunks,
            )

    def compare(actual, expected, rows):
        for value, reference in zip(actual[:4], expected[:4], strict=True):
            torch.testing.assert_close(value, reference[rows], atol=1e-6, rtol=0, equal_nan=True)
        for key in (
            "selected",
            "selected_candidate",
            "planning_failure_code",
            "selected_arrival_coverage",
            "selected_endpoint_joints_rad",
        ):
            torch.testing.assert_close(
                actual[4][key], expected[4][key][rows], atol=1e-6, rtol=0, equal_nan=True
            )

    rows = torch.arange(len(destination), device="cuda")
    baseline = solve(rows, 4)
    compare(solve(rows, 8), baseline, rows)
    reverse = rows.flip(0)
    compare(solve(reverse, 8), baseline, reverse)
    for row in rows.split(1):
        compare(solve(row, 1), baseline, row)
    report = {
        "seeds": state["episode_seed"].tolist(),
        "rows": len(rows),
        "valid": baseline[3].tolist(),
        "position_joint_atol": 1e-6,
        "tested_chunk_sizes": [1, 4, 8],
        "permutation": "reverse",
        "physics_simulated": False,
    }
    (tmp_path / "recorded_batch_invariance.json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.skipif(
    not os.environ.get("PICK_STATION_COUNTERFACTUAL_ROOT"), reason="Recorded probe required"
)
def test_recorded_station_body_hypotheses_counterfactual(tmp_path):
    from gr00t.rl.pick.station_batch import select_initial_pick_station_batch

    state, destination, sizes, support, mask, geometry, collision, limits = _recorded_batch(
        Path(os.environ["PICK_STATION_COUNTERFACTUAL_ROOT"]), count=4, frame=-1
    )
    report = {
        "seeds": state["episode_seed"].tolist(),
        "physics_simulated": False,
        "not_a_body_policy_or_path_tracking_guarantee": True,
        "cases": {},
    }
    for name, options in (
        ("original_hypotheses", {}),
        (
            "moderate_crouch_band",
            {
                "body_heights": (0.34, 0.38),
                "body_pitches_deg": (0.0, 4.0),
                "require_all_postures": True,
            },
        ),
        (
            "moderate_crouch_wider_station",
            {
                "body_heights": (0.34, 0.38),
                "body_pitches_deg": (0.0, 4.0),
                "require_all_postures": True,
                "forward_offsets_m": (-0.32, 0.0, 0.32),
            },
        ),
    ):
        with torch.inference_mode():
            pose, orientation, joints, valid, detail = select_initial_pick_station_batch(
                state,
                destination,
                sizes,
                support,
                limits,
                collision,
                geometry,
                support_mask=mask,
                **options,
            )
        report["cases"][name] = {
            "valid": valid.tolist(),
            "station_poses": pose.tolist(),
            "orientations": orientation.tolist(),
            "arm_joint_reference": joints.tolist(),
            "failure_code": detail["planning_failure_code"].tolist(),
            "arrival_coverage": detail["selected_arrival_coverage"].tolist(),
            "joint_margin_rad": detail["predicted_joint_margin_rad"].tolist(),
        }
    (tmp_path / "station_body_counterfactual.json").write_text(json.dumps(report, indent=2) + "\n")
