"""Replay initial basket refresh at real control-entry states, without simulator startup."""

import json
import os
import time
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not os.environ.get("PICK_INITIAL_REFRESH_ROOT"), reason="Real records required")
def test_initial_orientation_refresh_preserves_full_search_successes(tmp_path, monkeypatch):
    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.pick.basket import BasketGeometry
    from gr00t.rl.pick.basket_controller import BasketArmCollisionSamples, basket_required_height
    from gr00t.rl.pick.basket_geometry import BatchedBasketPlanningGeometry
    from gr00t.rl.pick.basket_grasp import plan_basket_grasp_batch
    from gr00t.rl.pick.basket_initialization import BasketAdmissionInputs
    from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
    from gr00t.rl.pick.collision import ArmCollisionSamples
    from gr00t.rl.pick.geometry_views import TranslatedGeometry
    from gr00t.rl.pick.scene_geometry import read_scene_triangles, static_scene_geometry
    from gr00t.rl.pick.scene_object_geometry import SceneObjectGeometryBatch
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    root = Path(os.environ["PICK_INITIAL_REFRESH_ROOT"])
    episodes, snapshots, hints, views, rooms = [], [], [], [], {}
    collision_path = None
    fields = ("root_states", "object_state", "dof_pos", "ee_pos", "ee_orn", "distractor_states")
    for path in sorted(root.glob("seed_*/control_record.pt")):
        record = torch.load(path, weights_only=False)
        if not record["stages"] or record["stages"][0] != "MOVE_TO_GRASP":
            continue
        admission = json.loads(path.with_name("initial_acceptance.json").read_text())
        if not admission.get("grasp_access", {}).get("direct_path_available", False):
            continue
        episode = json.loads(path.with_name("episode.json").read_text())
        episodes.append(episode)
        collision_path = collision_path or path.with_name("collision_model.pt")
        snapshots.append({key: record["pre"][key][0].cuda() for key in fields})
        hints.append(admission["grasp_access"]["direct_full_path"]["entry_orientation_world_wxyz"])
        scene = episode["scene_usd_path"]
        if scene not in rooms:
            p, f, labels = read_scene_triangles(
                Path(scene), Path(episode["geometry_cache_root"]), return_labels=True
            )
            rooms[scene] = static_scene_geometry(
                p, f, "cuda", labels=labels, backend="native", ground_plane_z_m=0.0
            )
        views.append(
            TranslatedGeometry(
                rooms[scene], torch.tensor(episode["batch_scene_origin_m"], device="cuda")
            )
        )
    assert snapshots
    state = {key: torch.stack([snapshot[key] for snapshot in snapshots]) for key in fields}
    count = len(episodes)
    inputs = BasketAdmissionInputs(episodes, state["distractor_states"].shape[1] + 1, "cuda")
    factory = SceneObjectGeometryBatch(episodes, BatchedSceneGeometry.from_views(views))
    model = BasketGeometry.from_robot_usd(device="cuda")
    basket = BatchedSceneGeometry(
        [model.geometry],
        torch.zeros(count, device="cuda", dtype=torch.long),
        state["root_states"][:, :3],
        state["root_states"][:, 3:7],
    )
    geometry = BatchedBasketPlanningGeometry(factory.update(state), basket, model)
    recorded = torch.load(collision_path, weights_only=False)
    arm = ArmCollisionSamples.__new__(ArmCollisionSamples)
    arm.parts = [(index, points.cuda()) for index, points in recorded["parts"]]
    arm.gripper_surface = [points.cuda() for points in recorded["gripper_surface"]]
    arm.arm_joint_ids = recorded["arm_joint_ids"].cuda()
    collision = BasketArmCollisionSamples(arm)
    limits = state["ee_pos"].new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    minimum = basket_required_height(
        model, state, inputs.support, inputs.lift_margin, support_mask=inputs.support_mask
    )
    hints = state["ee_pos"].new_tensor(hints)
    rows = torch.arange(count, device="cuda")

    def plan(rows, orientation=None):
        return plan_basket_grasp_batch(
            {key: value[rows] for key, value in state.items()},
            inputs.sizes[rows],
            limits,
            collision,
            geometry.select(rows),
            0.02,
            support_points=inputs.support[rows],
            support_mask=inputs.support_mask[rows],
            initial_height=state["object_state"][rows, 2],
            minimum_height=minimum[rows],
            prefer_realized_body=torch.zeros(len(rows), device="cuda", dtype=torch.bool),
            realized_body=torch.zeros(len(rows), device="cuda", dtype=torch.bool),
            **({"candidate_orientations": orientation[:, None]} if orientation is not None else {}),
        )

    compare_contacts = bool(os.environ.get("PICK_CONTACT_SHORT_CIRCUIT_COMPARE"))
    if compare_contacts:
        from gr00t.rl.pick import contact_profile_kernel
        from gr00t.rl.tests.test_pick_contact_short_circuit import exhaustive_profile_contact

        optimized_contact = contact_profile_kernel.first_profile_contact
        monkeypatch.setattr(
            contact_profile_kernel, "first_profile_contact", exhaustive_profile_contact
        )
    contact_report = {}
    with torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
        baseline_result = plan(rows)
        baseline = baseline_result[1]
        torch.cuda.synchronize()
        baseline_s = time.perf_counter() - started
        if compare_contacts:
            monkeypatch.setattr(contact_profile_kernel, "first_profile_contact", optimized_contact)
            started = time.perf_counter()
            optimized_result = plan(rows)
            torch.cuda.synchronize()
            contact_report["short_circuit_full_search_s"] = time.perf_counter() - started
            torch.testing.assert_close(optimized_result[1], baseline, rtol=0, atol=0)
            for actual, expected in zip(optimized_result[0], baseline_result[0], strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            contact_report["full_search_paths_bitwise_equal"] = True
        started = time.perf_counter()
        valid = plan(rows, hints)[1]
        hinted = int(valid.sum())
        pending = rows[~valid]
        if len(pending):
            valid[pending] = plan(pending)[1]
        torch.cuda.synchronize()
        candidate_s = time.perf_counter() - started
    report = {
        "physics_simulated": False,
        "rows": count,
        "room_obstacles_included": True,
        "robot_collision_source": "recorded_runtime_samples",
        "baseline": int(baseline.sum()),
        "hinted": hinted,
        "candidate": int(valid.sum()),
        "baseline_s": baseline_s,
        "candidate_s": candidate_s,
        **contact_report,
        "regressed_seeds": [
            episodes[i]["seed"] for i in torch.where(baseline & ~valid)[0].tolist()
        ],
    }
    (tmp_path / "initial_refresh_comparison.json").write_text(json.dumps(report, indent=2))
    assert not (baseline & ~valid).any(), report
