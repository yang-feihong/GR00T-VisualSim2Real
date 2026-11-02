"""Top-down basket targets using the shared GPU grasp candidate/inverse/path tools."""

import math

import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.basket import BASKET_GRASP_CLEARANCE_M
from gr00t.rl.pick.basket_entry import above_object_path_batch
from gr00t.rl.pick.grasp_candidates import cone_grasp_candidate_bank
from gr00t.rl.pick.grasp_inverse import realized_grasp_candidate_mask, solve_default_grasp_inverse
from gr00t.rl.pick.grasp_paths import score_default_grasp_paths
from gr00t.rl.pick.grasp_search import (
    balanced_pocket_priority,
    compact_grasp_bank,
    search_grasp_batches,
)
from gr00t.rl.pick.planning import (
    geometric_grasp_mask,
    grasp_approach_origin,
    grasp_pose_bank,
    object_frame_grasp_bank,
)
from gr00t.rl.pick.pocket import BASKET_MINIMUM_CENTER_DEPTH_M


def plan_basket_grasp_batch(
    state,
    sizes,
    limits,
    collision,
    geometry,
    dt,
    *,
    support_points,
    support_mask,
    initial_height,
    minimum_height,
    prefer_realized_body,
    realized_body,
    excluded=None,
    excluded_mask=None,
    entry_orientation=None,
    maximum_tool_tilt_deg=30.0,
    capacity=4096,
    minimum_realized_grasp_margin_rad=0.0,
    candidate_orientations=None,
    body_heights=None,
    candidate_bank=None,
    prefer_balanced_contact=True,
    entry_depth_fraction=None,
):
    """A fixed candidate axis and tensor masks preserve independent scene decisions."""
    root = state["root_states"]
    target = state["object_state"][:, :3]
    final_entry = entry_orientation is not None
    if final_entry:
        # The current pregrasp is intentional reuse, not a failed orientation retry.
        excluded, excluded_mask = None, None
    normal = _quat_rotate(root[:, 3:7], target.new_tensor([0, 0, 1]))
    if candidate_bank is None:
        if candidate_orientations is not None:
            orientations = candidate_orientations
        elif final_entry:
            orientations = entry_orientation[:, None]
        else:
            heading = torch.atan2(target[:, 1] - root[:, 1], target[:, 0] - root[:, 0])
            _, orientations, _ = grasp_pose_bank(
                target,
                heading,
                sizes,
                grasp_approach_origin(root),
                reference_orientation=state["ee_orn"],
            )
            orientations = torch.cat((state["ee_orn"][:, None], orientations), 1)
        initial_axis = _quat_rotate(orientations, target.new_tensor([1, 0, 0]))
        initial_available = (
            initial_axis[..., 2] <= -math.cos(math.radians(maximum_tool_tilt_deg))
        ).any(-1)
        if candidate_orientations is None and not final_entry:
            orientations = torch.cat(
                (orientations, object_frame_grasp_bank(state["object_state"][:, 3:7], sizes)), 1
            )
        bank = cone_grasp_candidate_bank(
            state,
            sizes,
            collision,
            geometry,
            support_points,
            support_mask,
            orientations,
            maximum_tool_tilt_deg,
            normal,
            collider_clearance_m=BASKET_GRASP_CLEARANCE_M,
            minimum_center_depth_m=BASKET_MINIMUM_CENTER_DEPTH_M,
            **(
                {
                    "depth_fractions": torch.cat(
                        (
                            entry_depth_fraction.nan_to_num(0)[:, None],
                            torch.linspace(0, 1, 17, device=root.device, dtype=root.dtype)[
                                None
                            ].expand(len(root), -1),
                        ),
                        -1,
                    )[:, None]
                }
                if final_entry and entry_depth_fraction is not None
                else {}
            ),
        )
    else:
        if final_entry:
            raise ValueError("Admission candidate reuse cannot override a locked entry pose")
        bank = candidate_bank
        if bank[0].shape[0] != len(root) or bank[1].shape[:2] != bank[0].shape[:2]:
            raise ValueError("Admission candidate rows must match the current planning batch")
        initial_available = (
            _quat_rotate(bank[0], target.new_tensor([1, 0, 0]))[..., 2]
            <= -math.cos(math.radians(maximum_tool_tilt_deg))
        ).any(-1)
    # Filtering retains fixed GPU dimensions without admitting a forbidden initial cone.
    bank_axis = _quat_rotate(bank[0], target.new_tensor([1, 0, 0]))
    bank[2]["pocket_fit"] &= initial_available[:, None] & (
        bank_axis[..., 2] <= -math.cos(math.radians(maximum_tool_tilt_deg))
    )
    bank[2]["pocket_fit"] &= geometric_grasp_mask(
        bank[0],
        bank[2],
        BASKET_GRASP_CLEARANCE_M,
        normal,
        excluded,
        excluded_mask=excluded_mask,
    )
    bank, compact_report = compact_grasp_bank(bank)
    capture_mask = (
        realized_grasp_candidate_mask(
            state,
            bank,
            limits,
            collision.arm_joint_ids,
            realized_body,
            minimum_realized_grasp_margin_rad,
        )
        if realized_body is not None
        else bank[2]["pocket_fit"]
    )

    def evaluate(rows, local_bank):
        local = {key: value[rows] for key, value in state.items()}
        inverse = solve_default_grasp_inverse(
            local,
            local_bank,
            limits,
            collision.arm_joint_ids,
            initial_height[rows],
            dt,
            excluded=None if excluded is None else excluded[rows],
            excluded_mask=None if excluded_mask is None else excluded_mask[rows],
            collider_clearance_m=BASKET_GRASP_CLEARANCE_M,
            support_normal_world=normal[rows],
            minimum_object_height_m=minimum_height[rows],
            withdraw_distance_m=0.0,
            body_heights=None if body_heights is None else body_heights[rows],
        )
        return score_default_grasp_paths(
            local,
            local_bank,
            inverse,
            limits,
            collision,
            geometry.select(rows),
            dt,
            prefer_realized_body=prefer_realized_body[rows],
            realized_body=realized_body[rows],
            prefer_orientation=False,
            collider_clearance_m=BASKET_GRASP_CLEARANCE_M,
            require_safe_arm_path=True,
            validate_carry_preview=False,
            wrist_motion_weight=1e-5,
            minimum_realized_grasp_margin_rad=minimum_realized_grasp_margin_rad,
            capacity=capacity,
        )

    priority, contact_lever, contact_depth = balanced_pocket_priority(bank, state["ee_orn"])
    path, valid, report = search_grasp_batches(
        state,
        bank,
        evaluate,
        capacity=capacity,
        candidate_mask=capture_mask,
        candidate_priority=priority if prefer_balanced_contact else None,
    )
    selected_slot = report["selected"].clamp_min(0).long()
    selected_rows = torch.arange(len(root), device=root.device)
    report["contact_balance_priority_enabled"] = prefer_balanced_contact
    report["selected_contact_lever_arm_m"] = contact_lever[
        selected_rows, selected_slot
    ].masked_fill(~valid, torch.nan)
    report["selected_object_center_depth_m"] = contact_depth[
        selected_rows, selected_slot
    ].masked_fill(~valid, torch.nan)
    if "depth_fraction" in bank[2]:
        report["selected_grasp_depth_fraction"] = bank[2]["depth_fraction"][
            selected_rows, selected_slot
        ].masked_fill(~valid, torch.nan)
    report["capture_precheck_candidate_mask"] = capture_mask
    report.update(compact_report)
    rows = torch.arange(len(root), device=root.device)
    selected_orientation = path[1][rows, path[2].long() - 1]
    if not final_entry:
        destination = report.get("selected_pregrasp_position_world_m", state["ee_pos"])
        path, selected_orientation = above_object_path_batch(path, destination)
    report.update(
        task="stationary_basket_top_down",
        entry_leg="insert_at_measured_object_pose" if final_entry else "above_object",
        maximum_tool_tilt_deg=maximum_tool_tilt_deg,
        support_surface_planning_clearance_m=BASKET_GRASP_CLEARANCE_M,
        other_obstacle_planning_clearance_m=BASKET_GRASP_CLEARANCE_M,
        clearance_reference="raw_surface_distance",
        minimum_realized_grasp_margin_rad=minimum_realized_grasp_margin_rad,
        initial_top_down_orientation_available=initial_available,
        entry_orientation_world_wxyz=selected_orientation,
    )
    return path, valid, report
