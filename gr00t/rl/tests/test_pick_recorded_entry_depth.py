"""Check whether entry resampling drops an admitted continuous contact depth."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not os.environ.get("PICK_ENTRY_DEPTH_ROOT"), reason="Recorded debug required")
def test_recorded_entry_depth(tmp_path, monkeypatch):
    from gr00t.rl.pick.basket_intrinsic_proposals import IntrinsicGraspProposals

    from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
    from gr00t.rl.pick import planning
    from gr00t.rl.pick.basket import BasketGeometry
    from gr00t.rl.pick.basket_controller import BasketArmCollisionSamples, basket_required_height
    from gr00t.rl.pick.basket_entry import transport_entry_orientation
    from gr00t.rl.pick.basket_geometry import BasketGeometryBatch
    from gr00t.rl.pick.basket_grasp import plan_basket_grasp_batch
    from gr00t.rl.pick.basket_initialization import BasketAdmissionInputs
    from gr00t.rl.pick.collision import ArmCollisionSamples
    from gr00t.rl.pick.scene_geometry import SceneGeometry
    from gr00t.rl.utils.b2z1_asset import ROBOT_URDF

    root = Path(os.environ["PICK_ENTRY_DEPTH_ROOT"])
    record = torch.load(root / "control_record.pt", weights_only=False)
    episode = json.loads((root / "episode.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    # GRASP is replanned before action recording; use the resulting recovery entry.
    frame = record["stages"].index("OPEN_FOR_REPOSITION")
    state = {k: v[frame : frame + 1].cuda() for k, v in record["pre"].items() if torch.is_tensor(v)}
    basket = BasketGeometry.from_robot_usd(device="cuda")
    saved = torch.load(root / "collision_model.pt", weights_only=False)
    arm = ArmCollisionSamples.__new__(ArmCollisionSamples)
    arm.parts = [(i, p.cuda()) for i, p in saved["parts"]]
    arm.gripper_surface = [p.cuda() for p in saved["gripper_surface"]]
    arm.arm_joint_ids = saved["arm_joint_ids"].cuda()
    collision = BasketArmCollisionSamples(arm)
    p = state["ee_pos"]
    ground = SceneGeometry(
        p.new_tensor([[-4000, -4000, 0], [4000, -4000, 0], [4000, 4000, 0], [-4000, 4000, 0]]),
        torch.tensor([[0, 1, 2], [0, 2, 3]], device=p.device),
    )
    geometry = BasketGeometryBatch(
        [SimpleNamespace(static_scene=ground, basket=basket, episode=episode)]
    ).update(state)
    inputs = BasketAdmissionInputs([episode], 4, p.device)
    limits = p.new_tensor(read_arm_joint_limits(ROBOT_URDF).copy())
    first = summary["planning"]["grasp_attempts"][0]
    slot = first["candidate_source_indices"][first["selected"]]
    proposals = IntrinsicGraspProposals(
        inputs.support,
        inputs.support_mask,
        inputs.sizes,
        torch.tensor([episode["seed"]], device=p.device),
    )
    fraction = proposals.fractions[:, slot]
    q = transport_entry_orientation(
        p.new_tensor(first["entry_orientation_world_wxyz"])[None],
        record["pre"]["object_state"][0:1, 3:7].cuda(),
        state["object_state"][:, 3:7],
    )
    native = planning.convex_grasp_quality
    outcomes = []
    for reuse in (False, True, "production"):
        with monkeypatch.context() as scope:
            if reuse is True:

                def with_original_depth(*args, **kwargs):
                    kwargs["depth_fractions"] = torch.cat(
                        (fraction[:, None], torch.linspace(0, 1, 17, device=p.device)[None]), -1
                    )[:, None]
                    return native(*args, **kwargs)

                scope.setattr(planning, "convex_grasp_quality", with_original_depth)
            _, valid, detail = plan_basket_grasp_batch(
                state,
                inputs.sizes,
                limits,
                collision,
                geometry,
                0.02,
                support_points=inputs.support,
                support_mask=inputs.support_mask,
                initial_height=state["object_state"][:, 2],
                minimum_height=basket_required_height(
                    basket,
                    state,
                    inputs.support,
                    inputs.lift_margin,
                    support_mask=inputs.support_mask,
                ),
                prefer_realized_body=torch.zeros(1, device=p.device, dtype=torch.bool),
                realized_body=torch.zeros(1, device=p.device, dtype=torch.bool),
                entry_orientation=q,
                **({"entry_depth_fraction": fraction} if reuse == "production" else {}),
            )
            outcomes.append(
                {
                    "reuse_depth": reuse,
                    "valid": bool(valid[0]),
                    "geometry_candidates": int(
                        detail["uncompressed_geometry_pass_counts.pocket_fit"][0]
                    ),
                }
            )
    result = {
        "seed": episode["seed"],
        "frame": frame,
        "source_slot": slot,
        "depth_fraction": float(fraction[0]),
        "physics_simulated": False,
        "room_geometry_included": False,
        "outcomes": outcomes,
    }
    (tmp_path / "entry_depth_comparison.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)
    assert outcomes[1]["valid"], (
        "Original depth must still pass complete original basket path checks"
    )
    assert outcomes[2]["valid"], "Production entry interface must preserve the admitted depth"
