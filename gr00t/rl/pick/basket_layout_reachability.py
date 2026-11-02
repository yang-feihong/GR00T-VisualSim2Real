"""Bounded GPU proposal ranking; physical admission remains authoritative."""

import math

import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.auto_open.pose_planning import solve_pose_grid
from gr00t.rl.auto_open.tensor_random import episode_uniform
from gr00t.rl.pick.planning import BALANCED_GRASP_PITCH_OFFSETS_DEG, balanced_grasp_orientations
from gr00t.rl.pick.pocket import gripper_pocket_geometry


def prefer_materialized_layouts(layouts, basket, seeds, initial_joints, limits, collision, config):
    """Improve chosen assets' placement, preserving original tuples on fallback."""
    from gr00t.rl.pick.basket_layout import sample_layouts
    from gr00t.rl.pick.basket_layout_validation import BasketLayoutPathValidator

    rows = [i for i, item in enumerate(layouts) if isinstance(item, tuple)]
    if not rows:
        return layouts
    source = [layouts[i] for i in rows]
    asset_rows = [[item["asset"] for item in layout[0]] for layout in source]
    row_ids = torch.tensor(rows, device=initial_joints.device, dtype=torch.long)
    local_seeds = [seeds[i] for i in rows]
    seed_tensor = torch.tensor(local_seeds, device=initial_joints.device, dtype=torch.int64)
    joints = initial_joints[row_ids]
    validator = BasketLayoutPathValidator(
        asset_rows,
        [layout[1] for layout in source],
        basket,
        joints,
        limits,
        collision,
        proposal_seeds=seed_tensor,
    )
    selected_rows = torch.arange(len(source), device=initial_joints.device)

    def preflight(data):
        bank = validator.proposal_bank()
        preferred, report = screen_layouts(
            data,
            joints,
            limits,
            seed_tensor,
            collision=collision,
            basket=basket,
            intrinsic_proposals=bank,
            box_overlap_fallback=bank is not None,
        )
        return validator(data, preferred, report, selected_rows, joints)

    proposals = sample_layouts(
        asset_rows,
        basket,
        local_seeds,
        config,
        counts=list(map(len, asset_rows)),
        target_asset_indices=[layout[1] for layout in source],
        gaps=[[layout[2]["object_gap_m"], layout[2]["wall_gap_m"]] for layout in source],
        proposal_filter=preflight,
    )
    result = list(layouts)
    for row, old, proposal in zip(rows, source, proposals, strict=True):
        if isinstance(proposal, tuple) and not proposal[2]["reachability_fallback"]:
            items, target, diagnostics = proposal
            result[row] = items, target, {**old[2], **diagnostics}
        elif isinstance(proposal, tuple):
            attempted = {
                key: proposal[2][key]
                for key in (
                    "reachability_fallback",
                    "full_path_layouts_evaluated",
                    "full_path_layouts_passed",
                    "full_path_hint_layouts_evaluated",
                    "full_path_hint_layouts_passed",
                    "intrinsic_contact_candidates",
                    "intrinsic_contact_fallback",
                    "intrinsic_contact_work",
                )
                if key in proposal[2]
            }
            result[row] = old[0], old[1], {**old[2], "layout_proposal_diagnostics": attempted}
    return result


def _boxes_clear(low, high, obstacle_low, obstacle_high, mask=None, *, reduce=True):
    gap = torch.maximum(obstacle_low - high[..., None, :], low[..., None, :] - obstacle_high)
    separated = gap.clamp_min(0).square().sum(-1) >= 0.006**2
    if mask is not None:
        separated |= ~mask
    return separated.all(-1) if reduce else separated


def _gripper_boxes_clear(
    low,
    high,
    positions,
    orientations,
    local_low,
    local_high,
    obstacle_low,
    obstacle_high,
    mask=None,
):
    """Conservative box separation in world and gripper axes, without mesh queries."""
    world = _boxes_clear(low, high, obstacle_low, obstacle_high, mask, reduce=False)
    center, half = (obstacle_low + obstacle_high) / 2, (obstacle_high - obstacle_low) / 2
    inverse = orientations.clone()
    inverse[..., 1:] *= -1
    local_center = _quat_rotate(inverse[..., None, :], center - positions[..., None, :])
    axes = _quat_rotate(
        orientations[..., None, :], torch.eye(3, device=positions.device, dtype=positions.dtype)
    )
    local_half = (axes.abs()[..., :, None, :] * half[..., None, :, :]).sum(-1).transpose(-1, -2)
    local = _boxes_clear(
        local_low,
        local_high,
        local_center - local_half,
        local_center + local_half,
        mask,
        reduce=False,
    )
    return (world | local).all(-1)


def screen_layouts(
    data,
    initial_joints,
    limits,
    seeds,
    *,
    maximum_candidates=1000,
    collision=None,
    basket=None,
    intrinsic_proposals=None,
    box_overlap_fallback=False,
):
    """Jointly draw orientation/depth/standoff/IK seed once per valid layout."""
    if not 1 <= maximum_candidates <= 1000:
        raise ValueError("Layout reachability requires 1..1000 joint proposals")
    if box_overlap_fallback and (collision is None or intrinsic_proposals is None):
        raise ValueError("Box-overlap fallback requires original-surface collision checks")
    valid, target = data["valid"], data["target"]
    n, width = valid.shape
    if intrinsic_proposals is not None and intrinsic_proposals.counts.shape != (n,):
        raise ValueError("Intrinsic source rows must match layout proposal rows")
    counts = valid.sum(1)
    order = valid.long().argsort(dim=1, descending=True, stable=True)
    layout_draw = episode_uniform(seeds, 0x4C41594C, maximum_candidates, 1)[..., 0]
    rank = (layout_draw * counts[:, None]).long().clamp_max(width - 1)
    columns = order.gather(1, rank)
    owner, joint_sample = torch.where((counts > 0)[:, None].expand(-1, maximum_candidates))
    column = columns[owner, joint_sample]
    eligible = torch.zeros_like(valid)
    eligible[owner, column] = True
    passed = torch.zeros_like(valid)
    orientations = data["xy"].new_zeros(n, width, 4)
    report = {
        "orientation_local_wxyz": orientations,
        "evaluated": eligible,
        "joint_queries_evaluated": (counts > 0).long() * maximum_candidates,
    }
    if intrinsic_proposals is not None:
        report["intrinsic_contact_counts"] = intrinsic_proposals.counts
        report["intrinsic_contact_work"] = intrinsic_proposals.work

    def record(name, actors, accepted):
        total = torch.zeros(n, device=valid.device, dtype=torch.long)
        total.index_add_(0, actors, accepted.long())
        report[name] = total

    if not len(owner):
        return passed, report
    roots = data["xy"].new_zeros(n, 13)
    roots[:, 2], roots[:, 3] = 0.55, 1
    draw = episode_uniform(seeds, 0x4C41594B, maximum_candidates, 6)[owner, joint_sample]
    sizes = data["sizes"][owner, target[owner]]
    radius = sizes[:, :2].amin(-1) / 2
    pocket = gripper_pocket_geometry(str(roots.device), roots.dtype)
    signs = roots.new_tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    )
    minimum_depth = pocket["back_x"] + radius + 0.0015
    depth = minimum_depth + draw[:, 5] * (pocket["front_x"] - 0.0015 - minimum_depth).clamp_min(0)
    offset = torch.stack((depth, radius * 0, pocket["fixed_plane_z"] + radius), -1)
    objects = torch.cat(
        (
            data["xy"][owner, column, target[owner]],
            (data["height"][owner, column, target[owner]] + sizes[:, 2] / 2 + 0.003)[:, None],
        ),
        -1,
    )
    objects[:, 2] += roots[owner, 2]
    heading = torch.atan2(objects[:, 1] - roots[owner, 1], objects[:, 0] - roots[owner, 0])
    choices = len(BALANCED_GRASP_PITCH_OFFSETS_DEG) * 16
    index = (draw[:, 2] * choices).long()
    q = balanced_grasp_orientations(heading, sizes, candidate_indices=index[:, None])[:, 0]
    if intrinsic_proposals is not None:
        contact_q, contact_center, available = intrinsic_proposals.sample(
            owner, draw[:, 2], data["yaw"][owner, column, target[owner]]
        )
        q = torch.where(available[:, None], contact_q, q)
        offset = torch.where(available[:, None], contact_center, offset)
        objects[:, 2] -= available.to(objects.dtype) * 0.005
        record("intrinsic_candidate_available_count", owner, available)
    capture = objects - _quat_rotate(q, offset)
    approach_axis = _quat_rotate(
        q, q.new_tensor([math.cos(-math.pi / 6), 0, -math.sin(-math.pi / 6)])
    )
    pre = capture - (0.08 + 0.24 * draw[:, 3, None]) * approach_axis
    minimum_height = torch.maximum(
        objects[:, 2] + 0.10,
        roots[owner, 2] + data["rim_z"] + data.get("lift_margin", 0.10) + sizes.norm(dim=-1) / 2,
    )
    lifted = capture.clone()
    lifted[:, 2] += (minimum_height - objects[:, 2]).clamp_min(0) + 0.005
    positions = torch.stack((pre, capture, lifted), 1)
    if collision is not None:
        if basket is None:
            raise ValueError("Self-collision proposal checks require the basket model")
        surface = torch.cat(collision._gripper_local_surfaces(positions))
        corners = surface.amin(0) + signs * (surface.amax(0) - surface.amin(0))
        gripper = (
            positions[:, :, None]
            - roots[owner, None, None, :3]
            + _quat_rotate(q[:, None, None], corners[None, None])
        )
        other_size = data["sizes"][owner]
        other_yaw = data["yaw"][owner, column]
        c, s = other_yaw.cos().abs(), other_yaw.sin().abs()
        hx = (c * other_size[..., 0] + s * other_size[..., 1]) / 2
        hy = (s * other_size[..., 0] + c * other_size[..., 1]) / 2
        other_position = torch.cat(
            (
                data["xy"][owner, column],
                (
                    data["height"][owner, column]
                    + other_size[..., 2] / 2
                    + (
                        0.003
                        if intrinsic_proposals is None
                        else torch.where(intrinsic_proposals.counts[owner, None] > 0, -0.002, 0.003)
                    )
                )[..., None],
            ),
            -1,
        )
        extent = torch.stack((hx, hy, other_size[..., 2] / 2), -1)
        other_mask = data["mask"][owner] & (
            torch.arange(other_size.shape[1], device=roots.device)[None] != target[owner, None]
        )
        neighbor_clear = _gripper_boxes_clear(
            gripper.amin(2),
            gripper.amax(2),
            positions - roots[owner, None, :3],
            q[:, None],
            surface.amin(0),
            surface.amax(0),
            (other_position - extent)[:, None],
            (other_position + extent)[:, None],
            other_mask[:, None],
        ).all(-1)
        record("preinverse_neighbor_pass_count", owner, neighbor_clear)
        separated = neighbor_clear.clone()
        if box_overlap_fallback:
            neighbor_clear = torch.ones_like(neighbor_clear)
        if intrinsic_proposals is not None:
            probe_indices = torch.cat((surface.argmin(0), surface.argmax(0)))
            probe_rows = torch.where(neighbor_clear)[0]
            sparse_clear = torch.zeros_like(neighbor_clear)
            if len(probe_rows):
                # Six existing surface samples can prove rejection, never acceptance.
                distance = basket.geometry.rigid_surface_clearance(
                    positions[probe_rows] - roots[owner[probe_rows], None, :3],
                    q[probe_rows, None].expand(-1, 3, -1),
                    surface[probe_indices],
                    check_rays=False,
                    maximum_distance_m=0.0061,
                )
                sparse_clear[probe_rows] = (distance >= 0.00599).all(-1)
            record("preinverse_sparse_basket_pass_count", owner, sparse_clear)
            neighbor_clear &= sparse_clear
        # This is the same final neighbor predicate, evaluated before expensive IK.
        keep = torch.where(neighbor_clear)[0]
        owner, column, joint_sample = owner[keep], column[keep], joint_sample[keep]
        positions, q, draw = positions[keep], q[keep], draw[keep]
        separated = separated[keep]
        if not len(owner):
            return passed, report
    record("inverse_candidates_evaluated_count", owner, torch.ones_like(owner, dtype=torch.bool))
    hypothesis = (draw[:, 4] * 3).long()
    actual = initial_joints[owner]
    seed = torch.where(
        (hypothesis == 0)[:, None],
        actual,
        torch.where((hypothesis == 1)[:, None], limits.mean(-1), limits.sum(-1) - actual),
    )
    result = solve_pose_grid(positions, q[:, None].expand(-1, 3, -1), roots[owner], seed, limits)
    margin = torch.minimum(
        result.arm_joint_positions - limits[:, 0], limits[:, 1] - result.arm_joint_positions
    )
    feasible = result.exact.all(-1) & (result.position_error.norm(dim=-1) <= 0.002).all(-1)
    feasible &= margin[:, 1].amin(-1) >= math.radians(2)
    feasible &= _quat_rotate(q, q.new_tensor([1, 0, 0]))[:, 2] <= -math.cos(math.radians(30))
    record("inverse_pregrasp_pass_count", owner, result.exact[:, 0])
    record("inverse_capture_pass_count", owner, result.exact[:, 1])
    record("inverse_lift_pass_count", owner, result.exact[:, 2])
    record("inverse_all_endpoints_pass_count", owner, feasible)
    if collision is not None:
        if basket is None:
            raise ValueError("Self-collision proposal checks require the basket model")
        ids = torch.where(feasible)[0]
        if not len(ids):
            return passed, report
        local_roots = roots[owner[ids], None].expand(-1, 3, -1).clone()
        local_roots[..., :3] = 0
        transforms = collision.link_transforms(
            local_roots.reshape(-1, 13), result.arm_joint_positions[ids].reshape(-1, 6)
        )
        obstacle_low, obstacle_high = basket.boxes.amin(1), basket.boxes.amax(1)
        clear = torch.ones(len(ids), 3, device=roots.device, dtype=torch.bool)
        box_clear = separated[ids, None].expand(-1, 3).clone()
        for index, points in collision.parts:
            if index == 6:
                continue
            corners = points.amin(0) + signs * (points.amax(0) - points.amin(0))
            origin, rotation = transforms[index]
            world = (torch.einsum("bij,kj->bki", rotation, corners) + origin[:, None]).reshape(
                len(ids), 3, 8, 3
            )
            box_clear &= _boxes_clear(world.amin(2), world.amax(2), obstacle_low, obstacle_high)
            record(f"arm_part_{index}_cumulative_pass_count", owner[ids], box_clear.all(-1))
        if not box_overlap_fallback:
            clear &= box_clear
        surface = torch.cat(collision._gripper_local_surfaces(positions))
        corners = surface.amin(0) + signs * (surface.amax(0) - surface.amin(0))
        gripper = (
            positions[ids, :, None]
            - roots[owner[ids], None, None, :3]
            + _quat_rotate(q[ids, None, None], corners[None, None])
        )
        low, high = gripper.amin(2), gripper.amax(2)
        if intrinsic_proposals is None:
            clear &= _gripper_boxes_clear(
                low,
                high,
                positions[ids] - roots[owner[ids], None, :3],
                q[ids, None],
                surface.amin(0),
                surface.amax(0),
                obstacle_low,
                obstacle_high,
            )
        else:
            from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry

            selected = torch.where(clear.all(-1))[0]
            precise = torch.zeros_like(clear)
            if len(selected):
                view = BatchedSceneGeometry(
                    [basket.geometry],
                    torch.zeros(len(selected), device=roots.device, dtype=torch.long),
                    roots.new_zeros(len(selected), 3),
                )
                precise[selected] = (
                    collision.gripper_clearance(
                        view,
                        positions[ids[selected]] - roots[owner[ids[selected]], None, :3],
                        q[ids[selected], None].expand(-1, 3, -1),
                        maximum_distance_m=0.0061,
                    )
                    >= 0.006
                )
            clear &= precise
        record("gripper_basket_pass_count", owner[ids], clear.all(-1))
        actor = owner[ids]
        other_size = data["sizes"][actor]
        other_yaw = data["yaw"][actor, column[ids]]
        c, s = other_yaw.cos().abs(), other_yaw.sin().abs()
        hx = (c * other_size[..., 0] + s * other_size[..., 1]) / 2
        hy = (s * other_size[..., 0] + c * other_size[..., 1]) / 2
        other_position = torch.cat(
            (
                data["xy"][actor, column[ids]],
                (
                    data["height"][actor, column[ids]]
                    + other_size[..., 2] / 2
                    + (
                        0.003
                        if intrinsic_proposals is None
                        else torch.where(intrinsic_proposals.counts[actor, None] > 0, -0.002, 0.003)
                    )
                )[..., None],
            ),
            -1,
        )
        extent = torch.stack((hx, hy, other_size[..., 2] / 2), -1)
        other_mask = data["mask"][actor] & (
            torch.arange(other_size.shape[1], device=roots.device)[None] != target[actor, None]
        )
        neighbor_clear = _gripper_boxes_clear(
            low,
            high,
            positions[ids] - roots[owner[ids], None, :3],
            q[ids, None],
            surface.amin(0),
            surface.amax(0),
            (other_position - extent)[:, None],
            (other_position + extent)[:, None],
            other_mask[:, None],
        )
        box_clear &= neighbor_clear
        if not box_overlap_fallback:
            clear &= neighbor_clear
        record("gripper_neighbors_pass_count", owner[ids], clear.all(-1))
        feasible = torch.zeros_like(feasible)
        feasible[ids] = clear.all(-1)
        if box_overlap_fallback:
            conservative = torch.zeros_like(feasible)
            conservative[ids] = clear.all(-1) & box_clear.all(-1)
            preferred_rows = torch.zeros(n, device=valid.device, dtype=torch.long)
            preferred_rows.index_add_(0, owner, conservative.long())
            # Preserve every original row's ordering; uncertain proposals are last resort.
            feasible &= conservative | (preferred_rows[owner] == 0)
            record("box_overlap_fallback_candidate_count", owner, feasible & ~conservative)
    sentinel = len(owner)
    candidate = torch.arange(sentinel, device=valid.device)
    chosen = torch.full((n * width,), sentinel, device=valid.device, dtype=torch.long)
    chosen.scatter_reduce_(
        0, owner * width + column, candidate.masked_fill(~feasible, sentinel), reduce="amin"
    )
    passed = (chosen < sentinel).reshape(n, width)
    orientations.copy_(
        torch.where(
            passed[..., None], q[chosen.clamp_max(sentinel - 1)].reshape(n, width, 4), orientations
        )
    )
    return passed, report
