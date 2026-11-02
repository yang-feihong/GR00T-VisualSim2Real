"""Compare hint frames on saved actual admission poses, without simulator startup."""

import json
import math
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    not os.environ.get("PICK_HINT_RECORD_ROOT"), reason="Recorded admission required"
)
def test_recorded_hint_frames(monkeypatch, tmp_path):
    from gr00t.rl.auto_open.motion_math import _quat_multiply
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.pick.basket import BASKET_GRASP_CLEARANCE_M, BasketGeometry
    from gr00t.rl.pick.basket_geometry import BatchedBasketPlanningGeometry
    from gr00t.rl.pick.basket_initialization import BasketAdmissionInputs
    from gr00t.rl.pick.batched_geometry import BatchedCompositeGeometry, BatchedSceneGeometry
    from gr00t.rl.pick.collision import ArmCollisionSamples
    from gr00t.rl.pick.geometry_views import TranslatedGeometry
    from gr00t.rl.pick.planning import pocket_pose_targets
    from gr00t.rl.pick.scene_geometry import read_scene_triangles, static_scene_geometry
    from gr00t.rl.pick.scene_object_geometry import SceneObjectGeometryBatch

    directory = Path(os.environ["PICK_HINT_RECORD_ROOT"])
    prepared = json.loads((directory / "prepared_scenes.json").read_text())
    episodes, poses, rooms = [], [], {}
    cached_evaluated, cached_geometry = [], []
    views = []
    for item in prepared:
        episode = item["episode"]
        seed_root = directory / f"seed_{episode['seed']}"
        record = json.loads((seed_root / "initial_attempts.json").read_text())[0]
        if "initial_acceptance_values" not in record:
            continue
        current = json.loads((seed_root / "episode.json").read_text())
        episode = {**episode, "batch_scene_origin_m": current["batch_scene_origin_m"]}
        episodes.append(episode)
        poses.append(record["initial_acceptance_values"])
        cached = record.get("cached_grasp_revalidation")
        cached_evaluated.append(cached is not None)
        cached_geometry.append(
            bool(cached)
            and sum(cached.get("direct_access", {}).get("geometric_grasp_counts", [0])) > 0
        )
        scene = episode["scene_usd_path"]
        if scene not in rooms:
            points, faces, labels = read_scene_triangles(
                Path(scene), Path(episode["geometry_cache_root"]), return_labels=True
            )
            rooms[scene] = static_scene_geometry(points, faces, "cuda", labels=labels)
        views.append(
            TranslatedGeometry(
                rooms[scene], torch.tensor(episode["batch_scene_origin_m"], device="cuda")
            )
        )
    state = {
        "root_states": torch.tensor([p["root_state_world"] for p in poses], device="cuda"),
        "object_state": torch.tensor([p["target_state_world"] for p in poses], device="cuda"),
        "distractor_states": torch.tensor(
            [p["distractor_states_world"] for p in poses], device="cuda"
        ),
    }
    metadata = BasketAdmissionInputs(episodes, state["distractor_states"].shape[1] + 1, "cuda")
    factory = SceneObjectGeometryBatch(episodes, BatchedSceneGeometry.from_views(views))
    nonrobot = factory.update(state)
    model = BasketGeometry.from_robot_usd(device="cuda")
    root = state["root_states"]
    basket = BatchedSceneGeometry(
        [model.geometry],
        torch.zeros(len(root), device="cuda", dtype=torch.long),
        root[:, :3],
        root[:, 3:7],
    )
    geometry = BatchedBasketPlanningGeometry(nonrobot, basket, model)
    collision = ArmCollisionSamples.from_robot_usd(device="cuda")
    local = root.new_tensor(
        [
            e["basket_sampling"].get("grasp_seed_orientation_local_wxyz", [1, 0, 0, 0])
            for e in episodes
        ]
    )
    hint_valid = torch.tensor(
        [e["basket_sampling"].get("reachable_proposals", 0) > 0 for e in episodes], device="cuda"
    )
    yaw = root.new_tensor([e["base_yaw"] for e in episodes]) / 2
    old_frame = torch.stack((yaw.cos(), yaw * 0, yaw * 0, yaw.sin()), -1)
    normal = _quat_rotate(root[:, 3:7], root.new_tensor([0, 0, 1]))

    geometry_diagnostics = []

    def check(q, z_offset=0.0, valid_mask=None, *, label="full", query_geometry=None):
        q = q[:, None]
        target = state["object_state"][:, :3].clone()
        target[:, 2] += z_offset
        _, quality = pocket_pose_targets(
            q,
            target,
            state["object_state"][:, 3:7],
            metadata.support,
            collision,
            geometry if query_geometry is None else query_geometry,
            support_mask=metadata.support_mask,
            minimum_environment_clearance_m=BASKET_GRASP_CLEARANCE_M,
            minimum_center_depth_m=0.005,
            support_normal_world=normal,
            retain_depth_candidates=True,
        )
        eligible = quality["pocket_fit"] & (quality["antipodal_margin"] >= 0)
        eligible &= quality["support_pinch_compatible"] & (
            quality["environment_clearance_m"] >= BASKET_GRASP_CLEARANCE_M
        )
        tilt = _quat_rotate(q, root.new_tensor([1, 0, 0]))[..., 2] <= -math.cos(math.radians(30))
        eligible &= tilt
        valid = hint_valid if valid_mask is None else valid_mask
        components = {
            "pocket_fit": quality["pocket_fit"],
            "antipodal": quality["antipodal_margin"] >= 0,
            "support_pinch": quality["support_pinch_compatible"],
            "environment": quality["environment_clearance_m"] >= BASKET_GRASP_CLEARANCE_M,
            "tilt": tilt,
        }
        intrinsic = (
            components["pocket_fit"] & components["antipodal"] & components["support_pinch"] & tilt
        )
        geometry_diagnostics.append(
            {
                "z_offset_m": z_offset,
                "obstacles": label,
                "eligible_rows": int(valid.sum()),
                "component_pass_rows": {
                    name: int((value.any(-1) & valid).sum()) for name, value in components.items()
                },
                "intrinsic_pass_rows": int((intrinsic.any(-1) & valid).sum()),
                "environment_only_rejection_rows": int(
                    (intrinsic.any(-1) & ~eligible.any(-1) & valid).sum()
                ),
            }
        )
        return eligible.any(-1) & (hint_valid if valid_mask is None else valid_mask)

    with torch.inference_mode():
        old_q = _quat_multiply(old_frame, local)
        before, after = check(old_q), check(_quat_multiply(root[:, 3:7], local))
        local_yaw = (
            root.new_tensor(
                [e.get("basket_local_yaw", e["object_yaw"] - e["base_yaw"]) for e in episodes]
            )
            / 2
        )
        inverse_object = torch.stack(
            (local_yaw.cos(), local_yaw * 0, local_yaw * 0, -local_yaw.sin()), -1
        )
        transported = _quat_multiply(
            state["object_state"][:, 3:7], _quat_multiply(inverse_object, local)
        )
        object_frame = check(transported)
        raised3 = check(old_q, 0.003)
        raised6 = check(old_q, 0.006)
        transported_raised3 = check(transported, 0.003)
    evaluated = torch.tensor(cached_evaluated, device="cuda")
    expected = torch.tensor(cached_geometry, device="cuda")
    baseline = after if os.environ.get("PICK_HINT_ACTUAL_FRAME") else before
    torch.testing.assert_close(baseline[evaluated], expected[evaluated], rtol=0, atol=0)
    if os.environ.get("PICK_HINT_SPLIT_OBSTACLES"):
        with torch.inference_mode():
            actual_q = _quat_multiply(root[:, 3:7], local)
            check(actual_q, label="basket_only", query_geometry=basket)
            check(actual_q, label="room_only", query_geometry=factory.rooms)
            check(
                actual_q,
                label="distractors_only",
                query_geometry=BatchedCompositeGeometry(tuple(factory.props)),
            )
    report = {
        "physics_simulated": False,
        "actual_recorded_rows": len(root),
        "hints": int(hint_valid.sum()),
        "old_geometry_pass": int(before.sum()),
        "actual_root_geometry_pass": int(after.sum()),
        "object_transport_geometry_pass": int(object_frame.sum()),
        "old_orientation_raise3mm_geometry_pass": int(raised3.sum()),
        "old_orientation_raise6mm_geometry_pass": int(raised6.sum()),
        "object_transport_raise3mm_geometry_pass": int(transported_raised3.sum()),
        "matches_actual_cached_geometry_checks": True,
        "geometry_diagnostics": geometry_diagnostics,
        "improved_seeds": [episodes[i]["seed"] for i in torch.where(after & ~before)[0].tolist()],
        "regressed_seeds": [episodes[i]["seed"] for i in torch.where(before & ~after)[0].tolist()],
    }
    if os.environ.get("PICK_HINT_SETTLED_PREFLIGHT"):
        from gr00t.rl.pick.basket_layout_validation import BasketLayoutPathValidator

        from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
        from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

        rows = [[e, *e.get("distractors", [])] for e in episodes]
        slots = max(map(len, rows))
        assets = [[d["asset"] for d in row] for row in rows]
        positions = root.new_tensor(
            [
                [d.get("basket_local_position", [0, 0, -10000]) for d in row]
                + [[0, 0, -10000]] * (slots - len(row))
                for row in rows
            ]
        )
        sizes = root.new_tensor(
            [[d["asset"]["size_m"] for d in row] + [[0, 0, 0]] * (slots - len(row)) for row in rows]
        )
        local_yaws = root.new_tensor(
            [[d.get("basket_local_yaw", 0) for d in row] + [0] * (slots - len(row)) for row in rows]
        )
        mask = (
            torch.arange(slots, device="cuda")[None]
            < torch.tensor(list(map(len, rows)), device="cuda")[:, None]
        )
        valid = torch.tensor(["basket_local_position" in e for e in episodes], device="cuda")[
            :, None
        ]
        data = {
            "valid": valid,
            "chosen": torch.arange(slots, device="cuda")[None].expand(len(rows), -1),
            "mask": mask,
            "xy": positions[:, None, :, :2],
            "height": (positions[..., 2] - sizes[..., 2] / 2 - 0.003)[:, None],
            "sizes": sizes,
            "yaw": local_yaws[:, None],
        }
        joints = root.new_tensor([e["initial_arm_joints"] for e in episodes])
        limits = root.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
        validator = BasketLayoutPathValidator(
            assets, [0] * len(rows), model, joints, limits, collision, admission_inputs=metadata
        )
        with torch.inference_mode():
            certified, detail = validator(
                data,
                valid,
                {"orientation_local_wxyz": local[:, None].clone()},
                torch.arange(len(rows), device="cuda"),
                joints,
            )
            effective = torch.where(certified, detail["orientation_local_wxyz"][:, 0], local)
            new_pass = check(
                _quat_multiply(root[:, 3:7], effective), valid_mask=hint_valid | certified[:, 0]
            )
        report.update(
            settled_preflight_paths=int(certified.sum()),
            settled_actual_geometry_pass=int(new_pass.sum()),
            settled_actual_geometry_gains=int((new_pass & ~before).sum()),
            settled_actual_geometry_regressions=int((before & ~new_pass).sum()),
        )
    if os.environ.get("PICK_HINT_INTRINSIC_BANK"):
        from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

        from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
        from gr00t.rl.pick.grasp_candidates import cone_grasp_candidate_bank
        from gr00t.rl.pick.planning import (
            grasp_approach_origin,
            grasp_pose_bank,
            object_frame_grasp_bank,
        )
        from gr00t.rl.pick.pocket import BASKET_MINIMUM_CENTER_DEPTH_M

        with torch.inference_mode():
            joints = root.new_tensor([e["initial_arm_joints"] for e in episodes])
            _, rotation, _ = b2z1_gripper_pose_and_jacobian(
                root[:, :3], root[:, [4, 5, 6, 3]], joints
            )
            state["ee_orn"] = _matrix_to_quaternion_wxyz(rotation)
            target = state["object_state"][:, :3]
            heading = torch.atan2(target[:, 1] - root[:, 1], target[:, 0] - root[:, 0])
            _, orientations, _ = grasp_pose_bank(
                target,
                heading,
                metadata.sizes,
                grasp_approach_origin(root),
                reference_orientation=state["ee_orn"],
            )
            orientations = torch.cat(
                (
                    state["ee_orn"][:, None],
                    orientations,
                    object_frame_grasp_bank(state["object_state"][:, 3:7], metadata.sizes),
                ),
                1,
            )

            def unobstructed(_geometry, positions, *args, **kwargs):
                return positions.new_full(positions.shape[:-1], torch.inf)

            with monkeypatch.context() as scope:
                scope.setattr(collision, "gripper_clearance", unobstructed)
                scope.setattr(collision, "gripper_closing_clearance", unobstructed)
                bank = cone_grasp_candidate_bank(
                    state,
                    metadata.sizes,
                    collision,
                    geometry,
                    metadata.support,
                    metadata.support_mask,
                    orientations,
                    30.0,
                    normal,
                    collider_clearance_m=BASKET_GRASP_CLEARANCE_M,
                    minimum_center_depth_m=BASKET_MINIMUM_CENTER_DEPTH_M,
                )
            quality = bank[2]
            intrinsic = (
                quality["pocket_fit"]
                & (quality["antipodal_margin"] >= 0)
                & quality["support_pinch_compatible"]
            ).any(-1)
        report["intrinsic_bank"] = {
            "environment_obstacles_disabled": True,
            "joint_inverse_not_evaluated": True,
            "reference_arm_joints": "initial_command_not_recorded_actual",
            "rows_with_intrinsic_grasp": int(intrinsic.sum()),
            "rejected": [
                {
                    "seed": episodes[i]["seed"],
                    "asset": episodes[i]["asset"]["name"],
                    "size_m": episodes[i]["asset"]["size_m"],
                }
                for i in torch.where(~intrinsic)[0].tolist()
            ],
        }
    (tmp_path / "hint_frame_comparison.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
