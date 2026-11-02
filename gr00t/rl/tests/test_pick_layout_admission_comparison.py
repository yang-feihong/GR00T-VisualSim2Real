"""Offline paired layout comparison with real basket, objects and arm colliders."""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from gr00t.rl.pick.basket_layout_reachability import screen_layouts
from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
from gr00t.rl.pick.basket import BasketConfig, BasketGeometry
from gr00t.rl.pick.basket_controller import BasketArmCollisionSamples
from gr00t.rl.pick.basket_geometry import BatchedBasketPlanningGeometry
from gr00t.rl.pick.basket_initialization import BasketAdmissionInputs, basket_grasp_access_batch
from gr00t.rl.pick.basket_layout import sample_layouts
from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
from gr00t.rl.pick.collision import ArmCollisionSamples
from gr00t.rl.pick.scene_geometry import SceneGeometry
from gr00t.rl.pick.scene_object_geometry import SceneObjectGeometryBatch
from gr00t.rl.pick.task_initialization import layout_grasp_hints


def evaluate(layouts, episodes, model, arm, limits, *, include_paths=False, contact_proposals=None):
    count = len(layouts)
    root = model.boxes.new_zeros(count, 13)
    root[:, 2], root[:, 3] = 0.55, 1
    descriptors, objects, others = [], [], []
    for layout, episode in zip(layouts, episodes, strict=True):
        items, target, _ = layout
        records, states = [], []
        for item in items:
            pose = [*item["basket_local_position"], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            pose[2] += 0.55
            if os.environ.get("PICK_LAYOUT_SETTLED_EVAL"):
                from gr00t.rl.pick.pocket import PICK_OBJECT_CONTACT_OFFSET_M

                pose[2] -= 0.003 + PICK_OBJECT_CONTACT_OFFSET_M
            yaw = model.boxes.new_tensor(item["basket_local_yaw"])
            pose[3], pose[6] = float((yaw / 2).cos()), float((yaw / 2).sin())
            records.append({**item, "object_position": pose[:3]})
            states.append(pose)
        descriptors.append(
            {
                **records[target],
                "seed": episode["seed"],
                "base_yaw": 0.0,
                "basket_sampling": layout[2],
                "distractors": [r for i, r in enumerate(records) if i != target],
            }
        )
        objects.append(states[target])
        distractors = [s for i, s in enumerate(states) if i != target]
        others.append(
            distractors + [[0, 0, -10000, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * (3 - len(distractors))
        )
    joints = root.new_tensor([e["initial_arm_joints"] for e in episodes])
    p, rotation, _ = b2z1_gripper_pose_and_jacobian(root[:, :3], root[:, [4, 5, 6, 3]], joints)
    dof = root.new_zeros(count, int(arm.arm_joint_ids.max()) + 1)
    dof[:, arm.arm_joint_ids] = joints
    state = {
        "root_states": root,
        "dof_pos": dof,
        "object_state": root.new_tensor(objects),
        "distractor_states": root.new_tensor(others),
        "ee_pos": p,
        "ee_orn": _matrix_to_quaternion_wxyz(rotation),
    }
    ground = SceneGeometry(
        root.new_tensor([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]]),
        torch.tensor([[0, 1, 2], [0, 2, 3]], device=root.device),
        root.device,
    )
    ids = torch.zeros(count, device=root.device, dtype=torch.long)
    rooms = BatchedSceneGeometry([ground], ids, root.new_zeros(count, 3))
    factory = SceneObjectGeometryBatch(descriptors, rooms)
    nonrobot = factory.update(state)
    with_target, target = factory.with_target(state)
    basket = BatchedSceneGeometry([model.geometry], ids, root[:, :3], root[:, 3:7])
    geometry = BatchedBasketPlanningGeometry(nonrobot, basket, model)
    target_geometry = BatchedBasketPlanningGeometry(with_target, basket, model)
    metadata = BasketAdmissionInputs(descriptors, 4, root.device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    accepted = torch.zeros(count, device=root.device, dtype=torch.bool)
    report = {}
    if include_paths:
        report["validated_path.position_world_m"] = (
            state["ee_pos"][:, None].expand(-1, 4096, -1).clone()
        )
        report["validated_path.orientation_world_wxyz"] = (
            state["ee_orn"][:, None].expand(-1, 4096, -1).clone()
        )
        report["validated_path.length"] = torch.ones(count, device=root.device, dtype=torch.long)
    hinted, hint_q = layout_grasp_hints(descriptors, root)
    from gr00t.rl.pick.basket_initialization import _record_batch_fields

    for warm in ("cache", True, False) if contact_proposals is not None else (True, False):
        rows = torch.where((hinted if warm is True else ~accepted) & ~accepted)[0]
        if not len(rows):
            continue
        options = {}
        if warm == "cache":
            options = {
                "candidate_bank": contact_proposals.select(rows).candidate_bank(
                    {key: value[rows] for key, value in state.items()},
                    BasketArmCollisionSamples(arm),
                    geometry.select(rows),
                ),
                "direct_only": True,
            }
        elif warm is True:
            options = {"candidate_orientations": hint_q[rows, None], "direct_only": True}
        allowed, local_report, path = basket_grasp_access_batch(
            {key: value[rows] for key, value in state.items()},
            metadata.sizes[rows],
            metadata.payload_sizes[rows],
            metadata.hulls[2][rows],
            model,
            BasketArmCollisionSamples(arm),
            geometry.select(rows),
            target_geometry.select(rows),
            target.select(rows),
            metadata.support[rows],
            metadata.support_mask[rows],
            metadata.centers[rows],
            limits,
            0.02,
            lift_margin=root.new_full((len(rows),), 0.1),
            **options,
        )
        accepted[rows] = allowed
        if include_paths:
            for name, value in zip(
                ("position_world_m", "orientation_world_wxyz", "length"), path, strict=True
            ):
                report[f"validated_path.{name}"][rows] = value
        padded = {}
        _record_batch_fields(padded, "batch", local_report, rows, count)
        for name, value in padded.items():
            name = name.removeprefix("batch.")
            if name in report and torch.is_tensor(value):
                if report[name].shape[1:] != value.shape[1:]:
                    # Candidate banks differ in width between warm and full search.
                    # Only fixed-width acceptance/selected pose diagnostics are merged.
                    continue
                report[name][rows] = value[rows]
            else:
                report[name] = value
    torch.cuda.synchronize()
    return accepted, report, time.perf_counter() - started


@pytest.mark.skipif(not os.environ.get("PICK_LAYOUT_MANIFEST"), reason="Real episodes required")
@pytest.mark.parametrize("optimization", ["capture", "contact_reuse", "nudge_pruning"])
def test_capture_precheck_preserves_mixed_layout_admission(monkeypatch, tmp_path, optimization):
    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.pick import basket_grasp, basket_initialization, basket_push
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    raw = json.loads(Path(os.environ["PICK_LAYOUT_MANIFEST"]).read_text())
    episodes = [record["episode"] for record in raw if "episode" in record]
    model = BasketGeometry.from_robot_usd(device="cuda")
    arm = ArmCollisionSamples.from_robot_usd(device="cuda")
    limits = model.boxes.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    assets = [[e["asset"], *[d["asset"] for d in e.get("distractors", [])]] for e in episodes]
    layouts = sample_layouts(
        assets, model, [e["seed"] for e in episodes], BasketConfig(), counts=list(map(len, assets))
    )
    valid = [i for i, layout in enumerate(layouts) if isinstance(layout, tuple)]
    indices = [valid[i * (len(valid) - 1) // 19] for i in range(20)]
    split = os.environ.get("PICK_CONTACT_REUSE_SPLIT", "development")
    if split == "holdout":
        remaining = [i for i in valid if i not in set(indices)]
        assert len(remaining) >= 20
        indices = [remaining[i * (len(remaining) - 1) // 19] for i in range(20)]
    elif split != "development":
        raise ValueError("Unknown contact reuse comparison split")
    chosen = [layouts[i] for i in indices]
    local = [episodes[i] for i in indices]
    with torch.inference_mode():
        with monkeypatch.context() as scope:
            if optimization == "capture":
                scope.setattr(
                    basket_grasp,
                    "realized_grasp_candidate_mask",
                    lambda state, bank, *args: bank[2]["pocket_fit"],
                )
            elif optimization == "contact_reuse":
                native = basket_initialization.plan_basket_grasp_batch

                def rebuild(*args, **kwargs):
                    kwargs.pop("candidate_bank", None)
                    return native(*args, **kwargs)

                scope.setattr(basket_initialization, "plan_basket_grasp_batch", rebuild)
            else:
                precheck = basket_initialization.closed_nudge_geometry_precheck
                access = basket_push.shifted_grasp_counts_batch

                def all_rows(*args, **kwargs):
                    result = precheck(*args, **kwargs)
                    return {**result, "good": torch.ones_like(result["good"])}

                def original_target(*args, **kwargs):
                    mask = kwargs["target_mask"].clone()
                    mask[:, 0] = True
                    return access(*args, **{**kwargs, "target_mask": mask})

                scope.setattr(basket_initialization, "closed_nudge_geometry_precheck", all_rows)
                scope.setattr(basket_push, "shifted_grasp_counts_batch", original_target)
            old, old_report, old_s = evaluate(chosen, local, model, arm, limits, include_paths=True)
        new, new_report, new_s = evaluate(chosen, local, model, arm, limits, include_paths=True)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    direct = old_report["direct_path_available"]
    torch.testing.assert_close(direct, new_report["direct_path_available"], rtol=0, atol=0)
    assert direct.any() and not direct.all(), "Comparison needs both feasible and failing layouts"
    for key in ("selected", "entry_orientation_world_wxyz", "selected_pregrasp_position_world_m"):
        name = f"direct_full_path.{key}"
        torch.testing.assert_close(
            old_report[name][direct], new_report[name][direct], rtol=0, atol=0
        )
    for key in ("position_world_m", "orientation_world_wxyz", "length"):
        torch.testing.assert_close(
            old_report[f"validated_path.{key}"], new_report[f"validated_path.{key}"], rtol=0, atol=0
        )
    report = {
        "physics_simulated": False,
        "optimization": optimization,
        "split": split,
        "seeds": [e["seed"] for e in local],
        "room_obstacles_included": False,
        "rows": len(indices),
        "accepted": int(new.sum()),
        "direct_paths": int(direct.sum()),
        "baseline_s": old_s,
        "batch_capture_precheck_s": new_s,
        "optimized_s": new_s,
        "full_paths_bitwise_equal": True,
        "acceptance_and_selected_poses_bitwise_equal": True,
    }
    (tmp_path / "capture_mixed_comparison.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


@pytest.mark.skipif(
    not os.environ.get("PICK_LAYOUT_MANIFEST"),
    reason="Real episodes required",
)
def test_paired_layout_admission(tmp_path, monkeypatch):

    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    records = json.loads(Path(os.environ["PICK_LAYOUT_MANIFEST"]).read_text())
    episodes = [r["episode"] for r in records if "episode" in r]
    exclusion = json.loads(os.environ.get("PICK_LAYOUT_EXCLUDE_REPORTS", "[]"))
    if not isinstance(exclusion, list) or not all(isinstance(path, str) for path in exclusion):
        raise ValueError("Excluded paired reports must be a JSON list of paths")
    if os.environ.get("PICK_LAYOUT_EXCLUDE_REPORT"):
        exclusion.append(os.environ["PICK_LAYOUT_EXCLUDE_REPORT"])
    if exclusion:
        excluded = {
            seed for path in exclusion for seed in json.loads(Path(path).read_text())["seeds"]
        }
        episodes = [e for e in episodes if e["seed"] not in excluded]
    limit = int(os.environ.get("PICK_LAYOUT_ROW_LIMIT", len(episodes)))
    if limit < len(episodes):
        episodes = [episodes[i * (len(episodes) - 1) // (limit - 1)] for i in range(limit)]
    assets = [[e["asset"], *[d["asset"] for d in e.get("distractors", [])]] for e in episodes]
    seeds = [e["seed"] for e in episodes]
    basket = BasketGeometry.from_robot_usd(device="cuda")
    limits = basket.boxes.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    joints = basket.boxes.new_tensor([e["initial_arm_joints"] for e in episodes])
    seed_tensor = torch.tensor(seeds, device="cuda", dtype=torch.int64)
    arm = ArmCollisionSamples.from_robot_usd(device="cuda")
    validator = None
    if os.environ.get("PICK_LAYOUT_CERTIFY"):
        from gr00t.rl.pick.basket_layout_validation import BasketLayoutPathValidator

        from gr00t.rl.pick import basket_layout_validation

        validator = BasketLayoutPathValidator(
            assets, [0] * len(assets), basket, joints, limits, arm
        )
        native_plan = basket_layout_validation.plan_basket_grasp_batch
        phases = []

        def timed_plan(state, *args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = native_plan(state, *args, **kwargs)
            torch.cuda.synchronize()
            phases.append(
                {
                    "rows": len(state["root_states"]),
                    "planning_s": time.perf_counter() - started,
                    "valid": int(result[1].sum()),
                }
            )
            (tmp_path / "shortlist_phase_timing.json").write_text(json.dumps(phases, indent=2))
            print({"nominal_layout_cohort": phases[-1]}, flush=True)
            return result

        monkeypatch.setattr(basket_layout_validation, "plan_basket_grasp_batch", timed_plan)

    coarse_screens = []
    intrinsic_bank = None
    compare_boxes = bool(os.environ.get("PICK_LAYOUT_BOX_SCREEN_COMPARE"))
    preserve_box_choices = bool(os.environ.get("PICK_LAYOUT_BOX_PRESERVE_CHOICES"))
    relaxed_boxes = False

    def prefer(data, *, use_bank=None):
        nonlocal intrinsic_bank
        use_bank = (
            bool(os.environ.get("PICK_LAYOUT_INTRINSIC_BANK")) if use_bank is None else use_bank
        )
        if use_bank and intrinsic_bank is None:
            from gr00t.rl.pick.basket_intrinsic_proposals import IntrinsicGraspProposals

            if validator is None:
                raise ValueError("Intrinsic proposals require target geometry metadata")
            started = time.perf_counter()
            intrinsic_bank = IntrinsicGraspProposals(
                validator.inputs.support,
                validator.inputs.support_mask,
                validator.inputs.sizes,
                seed_tensor,
            )
            if os.environ.get("PICK_LAYOUT_CONTACT_CACHE_COMPARE") or compare_boxes:
                validator.proposal_seeds = seed_tensor
                validator.intrinsic_proposals = intrinsic_bank
            torch.cuda.synchronize()
            (tmp_path / "intrinsic_bank_work.json").write_text(
                json.dumps(
                    {
                        **intrinsic_bank.work,
                        "build_s": time.perf_counter() - started,
                        "rows_with_contacts": int((intrinsic_bank.counts > 0).sum()),
                        "valid_contacts": int(intrinsic_bank.counts.sum()),
                    },
                    indent=2,
                )
            )
        with monkeypatch.context() as scope:
            if relaxed_boxes and not preserve_box_choices:
                from gr00t.rl.pick import basket_layout_reachability

                def uncertain_boxes(
                    low, high, obstacle_low, obstacle_high, mask=None, *, reduce=True
                ):
                    shape = torch.broadcast_shapes(low[..., None, :].shape, obstacle_low.shape)[:-1]
                    possible = torch.ones(shape, device=low.device, dtype=torch.bool)
                    return possible.all(-1) if reduce else possible

                scope.setattr(basket_layout_reachability, "_boxes_clear", uncertain_boxes)
            preferred, report = screen_layouts(
                data,
                joints,
                limits,
                seed_tensor,
                collision=arm,
                basket=basket,
                box_overlap_fallback=relaxed_boxes and preserve_box_choices,
                **({"intrinsic_proposals": intrinsic_bank} if use_bank else {}),
            )
        coarse_screens.append(
            {
                "rows": len(assets),
                "intrinsic_bank": use_bank,
                "box_overlap_not_a_collision_proof": relaxed_boxes,
                "rows_with_preferred_layout": int(preferred.any(-1).sum()),
                "counts": {
                    name: int(value.sum())
                    for name, value in report.items()
                    if (name.endswith("_count") or name == "joint_queries_evaluated")
                    and torch.is_tensor(value)
                },
            }
        )
        (tmp_path / "coarse_screen_diagnostics.json").write_text(
            json.dumps(coarse_screens, indent=2)
        )
        if os.environ.get("PICK_LAYOUT_SCREEN_ONLY"):
            return preferred, report
        if validator is not None:
            return validator(
                data, preferred, report, torch.arange(len(assets), device="cuda"), joints
            )
        return preferred, report

    with torch.inference_mode():
        compare_choices = int(os.environ.get("PICK_LAYOUT_COMPARE_CHOICES", 0))
        compare_bank = bool(os.environ.get("PICK_LAYOUT_BANK_COMPARE"))
        compare_hint = bool(os.environ.get("PICK_LAYOUT_HINT_COMPARE"))
        compare_contacts = bool(os.environ.get("PICK_LAYOUT_CONTACT_CACHE_COMPARE"))
        proposal_bank = None
        if compare_contacts:
            if validator is None:
                raise ValueError("Contact-cache comparison requires full certification")
            proposal_bank = validator.proposal_bank
            validator.proposal_bank = lambda rows=None: None
        certify = None
        if compare_hint:
            if validator is None:
                raise ValueError("Hint comparison requires full certification")
            certify = validator.certify

            def without_hint(state, rows, geometry, minimum, mask, orientation):
                return certify(state, rows, geometry, minimum, torch.zeros_like(mask), orientation)

            validator.certify = without_hint
        if compare_choices and (validator is None or not 1 <= compare_choices <= 4):
            raise ValueError("Cohort comparison requires certification and one to four layouts")
        if compare_choices:
            validator.maximum_layouts = 1
        original = sample_layouts(
            assets,
            basket,
            seeds,
            BasketConfig(),
            counts=list(map(len, assets)),
            target_asset_indices=[0] * len(assets),
            **(
                {"proposal_filter": prefer}
                if compare_hint or compare_contacts or compare_boxes
                else {"proposal_filter": lambda data: prefer(data, use_bank=False)}
                if compare_bank
                else {"proposal_filter": prefer}
                if compare_choices
                else {}
            ),
        )
        if compare_hint:
            validator.certify = certify
        if compare_contacts:
            validator.proposal_bank = proposal_bank
        if compare_choices:
            validator.maximum_layouts = compare_choices
        relaxed_boxes = compare_boxes
        torch.cuda.synchronize()
        proposals_started = time.perf_counter()
        improved = sample_layouts(
            assets,
            basket,
            seeds,
            BasketConfig(),
            counts=list(map(len, assets)),
            target_asset_indices=[0] * len(assets),
            proposal_filter=prefer,
        )
        torch.cuda.synchronize()
        proposal_s = time.perf_counter() - proposals_started
        (tmp_path / "proposal_timing.json").write_text(json.dumps({"proposal_s": proposal_s}))
        (tmp_path / "layout_candidates.json").write_text(
            json.dumps(
                {
                    "episodes": episodes,
                    "original": [
                        layout if isinstance(layout, tuple) else {"failure": str(layout)}
                        for layout in original
                    ],
                    "proposed": [
                        layout if isinstance(layout, tuple) else {"failure": str(layout)}
                        for layout in improved
                    ],
                    "proposal_batch_s": proposal_s,
                }
            )
        )
        if os.environ.get("PICK_LAYOUT_SCREEN_ONLY"):
            assert coarse_screens
            assert all(
                r["counts"]["joint_queries_evaluated"] <= len(episodes) * 1000
                for r in coarse_screens
            )
            print(
                {
                    "screen_only": True,
                    "formal_admission_evaluated": False,
                    "coarse_screens": coarse_screens,
                },
                flush=True,
            )
            return
        valid = [
            i
            for i in range(len(episodes))
            if isinstance(original[i], tuple) and isinstance(improved[i], tuple)
        ]
        assert len(valid) >= 20, "Independent comparison requires twenty distinct placeable rows"
        indices = [valid[i * (len(valid) - 1) // 19] for i in range(20)]
        split = os.environ.get("PICK_LAYOUT_COMPARE_SPLIT", "development")
        if split == "holdout":
            remaining = [i for i in valid if i not in set(indices)]
            indices = [remaining[i * (len(remaining) - 1) // 19] for i in range(20)]
        elif split != "development":
            raise ValueError("Unknown layout comparison split")
        local = [episodes[i] for i in indices]
        baseline, baseline_report, baseline_s = evaluate(
            [original[i] for i in indices], local, basket, arm, limits
        )
        proposed, proposed_report, proposed_s = evaluate(
            [improved[i] for i in indices], local, basket, arm, limits
        )
    result = {
        "physics_simulated": False,
        "room_obstacles_included": False,
        "basket_and_objects_original_collision": True,
        "rows": 20,
        "proposal_batch_rows": len(episodes),
        "proposal_batch_s": proposal_s,
        "target_identity": "recorded_asset_preserved",
        "baseline_layout_policy": "original_box_screen_with_full_path_certification"
        if compare_boxes
        else "hint_then_full_without_contact_cache"
        if compare_contacts
        else "full_bank_without_coarse_hint"
        if compare_hint
        else "legacy_preflight"
        if compare_bank
        else "one_cohort"
        if compare_choices
        else "spatial_only",
        "nominal_object_height": "settled_envelope"
        if os.environ.get("PICK_LAYOUT_SETTLED_EVAL")
        else "spawn",
        "split": "independent_from_report" if exclusion else split,
        "baseline": int(baseline.sum()),
        "prefiltered": int(proposed.sum()),
        "baseline_s": baseline_s,
        "prefiltered_s": proposed_s,
        "improved_seeds": [local[i]["seed"] for i in torch.where(proposed & ~baseline)[0].tolist()],
        "regressed_seeds": [
            local[i]["seed"] for i in torch.where(baseline & ~proposed)[0].tolist()
        ],
    }
    (tmp_path / "layout_admission_comparison.json").write_text(json.dumps(result, indent=2))
    from gr00t.rl.auto_open.tensor_metadata import to_host

    reports = to_host({"baseline": baseline_report, "prefiltered": proposed_report})
    reports["seeds"] = [e["seed"] for e in local]
    result["box_overlap_policy"] = (
        ("preserve_original_choices_then_uncertain" if preserve_box_choices else "uncertain")
        if compare_boxes
        else None
    )
    result["seeds"] = reports["seeds"]
    (tmp_path / "layout_admission_comparison.json").write_text(json.dumps(result, indent=2))
    (tmp_path / "paired_admission_reports.json").write_text(
        json.dumps(reports, separators=(",", ":"))
    )
    print(result, flush=True)
    assert baseline_s < 60 and proposed_s < 60
    if compare_hint or compare_contacts or preserve_box_choices:
        assert not (baseline & ~proposed).any(), "Preferred admission must preserve successes"
    else:
        assert proposed.sum() > baseline.sum(), "Proposal prefilter must improve paired admission"
