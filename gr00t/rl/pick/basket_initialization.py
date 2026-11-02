"""Physical basket admission, without the external-task camera-distance requirement."""

import math
from pathlib import Path

import numpy as np
import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.basket import (
    BasketConfig,
)
from gr00t.rl.pick.basket_access import shifted_grasp_counts_batch
from gr00t.rl.pick.basket_clear import plan_clear_for_nudge_batch
from gr00t.rl.pick.basket_controller import (
    basket_required_height,
    plan_basket_grasp_path,
)
from gr00t.rl.pick.basket_grasp import plan_basket_grasp_batch
from gr00t.rl.pick.basket_nudge import plan_clear_for_nudge, plan_closed_nudge, shifted_grasp_counts
from gr00t.rl.pick.basket_nudge_geometry import closed_nudge_geometry_precheck
from gr00t.rl.pick.basket_push import plan_closed_nudge_batch
from gr00t.rl.pick.grasp_quality import load_grasp_center, load_grasp_support


class BasketAdmissionInputs:
    """Immutable payload metadata shared by physical and grasp admission checks."""

    def __init__(self, episodes, maximum_objects, device):
        from gr00t.rl.pick.tensor_rows import pad_device_rows

        self.hulls = load_basket_payload_hulls(episodes, maximum_objects, device)
        descriptors = [[episode, *episode.get("distractors", [])] for episode in episodes]
        self.payload_sizes = torch.tensor(
            [
                [d["asset"]["size_m"] for d in row] + [[0, 0, 0]] * (maximum_objects - len(row))
                for row in descriptors
            ],
            device=device,
            dtype=torch.float32,
        )
        self.sizes = torch.tensor(
            [e["asset"].get("intrinsic_size_m", e["asset"]["size_m"]) for e in episodes],
            device=device,
        )
        self.lift_margin = torch.tensor(
            [e.get("lift_above_rim_m", 0.1) for e in episodes], device=device
        )
        assets = list(dict.fromkeys(e["asset"]["usd_path"] for e in episodes))
        supports = {path: load_grasp_support(path, device) for path in assets}
        self.supports = [supports[e["asset"]["usd_path"]] for e in episodes]
        centers = {path: load_grasp_center(path, str(device)) for path in assets}
        self.support, self.support_mask = pad_device_rows(self.supports, fill=torch.nan)
        self.centers = torch.stack([centers[e["asset"]["usd_path"]] for e in episodes])


def admission_report_to_host(report):
    """Transfer batch diagnostics once; callers only index metadata while publishing."""
    from gr00t.rl.auto_open.tensor_metadata import to_host

    return to_host(report)


def resample_basket_payloads(
    episodes,
    rows,
    basket,
    attempt,
    *,
    collision=None,
    limits=None,
    initial_joints=None,
    layout_validator=None,
    scene_rows=None,
):
    """One spatial proposal batch; the shared env.reset applies accepted descriptors."""
    from gr00t.rl.auto_open.tensor_metadata import to_host
    from gr00t.rl.pick.basket_layout import sample_layouts

    source = [episodes[row] for row in rows]
    descriptors = [[episode, *episode.get("distractors", [])] for episode in source]
    assets = [[item["asset"] for item in row] for row in descriptors]
    seeds = [int(episode["seed"]) for episode in source]
    attempt_tensor = torch.as_tensor(attempt, device=basket.boxes.device, dtype=torch.int64)
    if attempt_tensor.ndim == 0:
        attempt_tensor = attempt_tensor.expand(len(source))
    if attempt_tensor.shape != (len(source),):
        raise ValueError("Basket resample attempts must match the selected rows")
    seed_tensor = (
        torch.tensor(seeds, device=basket.boxes.device, dtype=torch.int64)
        + attempt_tensor * 0x4241534B
    ) & 0x7FFFFFFFFFFFFFFF
    gaps = [
        [episode["basket_sampling"]["object_gap_m"], episode["basket_sampling"]["wall_gap_m"]]
        for episode in source
    ]
    options = {}
    if collision is not None:
        from gr00t.rl.pick.basket_layout_reachability import screen_layouts

        if limits is None:
            raise ValueError("Reachable resampling requires shared arm limits")
        joints = (
            initial_joints
            if initial_joints is not None
            else basket.boxes.new_tensor([episode["initial_arm_joints"] for episode in source])
        )

        def preflight(data):
            if layout_validator is not None and scene_rows is None:
                raise ValueError("Layout path validation requires GPU source row indices")
            bank = None if layout_validator is None else layout_validator.proposal_bank(scene_rows)
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
            if layout_validator is not None:
                return layout_validator(data, preferred, report, scene_rows, joints)
            return preferred, report

        options = {
            "target_asset_indices": [0] * len(source),
            "proposal_filter": preflight,
        }
    layouts, tensors = sample_layouts(
        assets,
        basket,
        seeds,
        BasketConfig(),
        counts=list(map(len, assets)),
        gaps=gaps,
        return_tensors=True,
        randomization_seeds=seed_tensor,
        **options,
    )
    # Transform every payload on device, then transfer just the final descriptors.
    chosen = tensors["chosen"]
    order = torch.arange(chosen.shape[1], device=chosen.device)[None].expand_as(chosen)
    counts = chosen.new_tensor(list(map(len, assets)))
    inverse = torch.full_like(chosen, chosen.shape[1])
    inverse.scatter_reduce_(
        1, chosen, order.masked_fill(order >= counts[:, None], chosen.shape[1]), reduce="amin"
    )
    inverse.clamp_max_(chosen.shape[1] - 1)
    positions = tensors["positions"].gather(1, inverse[..., None].expand(-1, -1, 3))
    yaw = tensors["yaw"].gather(1, inverse)
    height = tensors["height"].gather(1, inverse)
    root = positions.new_tensor(
        [[*e["base_xy"], e["support_z"] - e["support_z_local"]] for e in source]
    )
    heading = positions.new_tensor([e["base_yaw"] for e in source])
    quat = torch.stack(((heading / 2).cos(), heading * 0, heading * 0, (heading / 2).sin()), -1)
    host = to_host(
        {
            "positions": _quat_rotate(quat[:, None], positions) + root[:, None],
            "yaw": yaw + heading[:, None],
            "support_z": height + root[:, 2, None],
            "resample_attempts": attempt_tensor,
            "layout_seeds": seed_tensor,
        }
    )
    results = {}
    for index, (row, layout, original) in enumerate(zip(rows, layouts, descriptors, strict=True)):
        if isinstance(layout, Exception):
            results[row] = layout
            continue
        items, _, diagnostics = layout
        by_name = {item["asset"]["name"]: item for item in items}
        updated = []
        for slot, item in enumerate(original):
            updated.append(
                {
                    **item,
                    **by_name[item["asset"]["name"]],
                    "object_position": host["positions"][index][slot],
                    "object_yaw": host["yaw"][index][slot],
                    "support_z": host["support_z"][index][slot],
                }
            )
        episode = updated[0]
        episode["distractors"] = updated[1:]
        episode["basket_sampling"] = {
            **source[index]["basket_sampling"],
            **diagnostics,
            "target_index": next(
                i
                for i, item in enumerate(items)
                if item["asset"]["name"] == episode["asset"]["name"]
            ),
            "resample_attempt": host["resample_attempts"][index],
            "layout_seed": host["layout_seeds"][index],
        }
        results[row] = episode
    return results


def admission_report_row(report, host, row):
    result = {}
    for name, value in host.items():
        value = value[row] if torch.is_tensor(report[name]) else value
        target = result
        parts = name.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def _record_batch_fields(report, prefix, values, rows, batch):
    for name, value in values.items():
        if isinstance(value, dict):
            _record_batch_fields(report, f"{prefix}.{name}", value, rows, batch)
        elif torch.is_tensor(value):
            fill = (
                torch.nan
                if value.is_floating_point()
                else False
                if value.dtype == torch.bool
                else -1
            )
            field = torch.full(
                (batch, *value.shape[1:]), fill, device=value.device, dtype=value.dtype
            )
            field[rows] = value
            report[f"{prefix}.{name}"] = field
        else:
            report[f"{prefix}.{name}"] = value


def _cached_direct_grasp_access(
    state,
    sizes,
    basket,
    collision,
    geometry,
    support,
    support_mask,
    limits,
    dt,
    lift_margin,
    candidate_bank,
    capacity,
):
    from gr00t.rl.pick.basket_inverse import safe_inverse_paths_batch

    batch = len(state["root_states"])
    if candidate_bank[0].shape[0] != batch:
        raise ValueError("Cached admission contacts must match current scene rows")
    minimum = basket_required_height(basket, state, support, lift_margin, support_mask=support_mask)
    path, accepted, diagnostic = plan_basket_grasp_batch(
        state,
        sizes,
        limits,
        collision,
        geometry,
        dt,
        support_points=support,
        support_mask=support_mask,
        initial_height=state["object_state"][:, 2],
        minimum_height=minimum,
        prefer_realized_body=torch.zeros(batch, device=limits.device, dtype=torch.bool),
        realized_body=torch.ones(batch, device=limits.device, dtype=torch.bool),
        minimum_realized_grasp_margin_rad=math.radians(2),
        capacity=capacity,
        candidate_bank=candidate_bank,
    )
    checked = torch.zeros_like(accepted)
    rows = torch.where(accepted)[0]
    if len(rows):
        selected = diagnostic["selected"][rows].long()
        source = diagnostic["candidate_source_indices"][rows, selected].long()
        goals = candidate_bank[1][rows, source]
        quats = candidate_bank[0][rows, source]
        endpoints = torch.stack((diagnostic["selected_pregrasp_position_world_m"][rows], goals), 1)
        safe, _, detail = safe_inverse_paths_batch(
            {key: state[key][rows] for key in ("root_states", "dof_pos")},
            endpoints[:, None],
            quats[:, None, None].expand(-1, 1, 2, -1),
            limits,
            collision,
            geometry.select(rows),
            body_height_offsets=(-0.01, 0.0, 0.01),
            final_joint_margin_rad=math.radians(2),
            minimum_arm_clearance_m=0.006,
            maximum_arm_query_distance_m=0.0061,
        )
        accepted[rows] &= safe[:, 0]
        checked[rows] = True
    else:
        detail = {}
    report = {}
    _record_batch_fields(
        report, "direct_full_path", diagnostic, torch.arange(batch, device=limits.device), batch
    )
    _record_batch_fields(report, "direct_access.selected_endpoint_precheck", detail, rows, batch)
    quality = candidate_bank[2]
    geometric = quality["pocket_fit"] & (quality["antipodal_margin"] >= 0)
    geometric &= quality["support_pinch_compatible"] & (quality["environment_clearance_m"] >= 0.006)
    report.update(
        {
            "direct_access.geometric_grasp_counts": geometric.sum(-1)[:, None],
            "direct_access.reachable_grasp_count_is_lower_bound": torch.ones_like(accepted),
            "direct_access.selected_endpoints_evaluated": checked,
            "direct_access.selected_endpoints_passed": accepted.clone(),
            "direct_reachable_grasp_candidates": accepted.long(),
            "direct_path_available": accepted,
            "initial_reposition_required": torch.zeros_like(accepted),
            "admission_failure_code": (~accepted).long(),
        }
    )
    return accepted, report, path


def basket_grasp_access_batch(
    state,
    sizes,
    payload_sizes,
    payload_mask,
    basket,
    collision,
    geometry,
    target_geometry,
    target_surface,
    support,
    support_mask,
    object_centers_local,
    limits,
    dt,
    *,
    lift_margin,
    capacity=4096,
    candidate_orientations=None,
    direct_only=False,
    candidate_bank=None,
):
    """Batch the original direct-path then safe-reposition admission, without physics."""
    if candidate_bank is not None:
        if not direct_only or candidate_orientations is not None:
            raise ValueError(
                "Cached contacts require a direct-only probe without an orientation override"
            )
        return _cached_direct_grasp_access(
            state,
            sizes,
            basket,
            collision,
            geometry,
            support,
            support_mask,
            limits,
            dt,
            lift_margin,
            candidate_bank,
            capacity,
        )
    root = state["root_states"]
    batch = len(root)
    counts, report = shifted_grasp_counts_batch(
        state,
        sizes,
        state["object_state"][:, None, :3],
        collision,
        geometry,
        support,
        limits,
        support_mask=support_mask,
        return_candidate_bank=True,
        **(
            {}
            if candidate_orientations is None
            else {"candidate_orientations": candidate_orientations}
        ),
    )
    candidate_bank = report.pop("candidate_bank", None)
    report = {f"direct_access.{key}": value for key, value in report.items()}
    direct = torch.zeros(batch, device=root.device, dtype=torch.bool)
    path_p = state["ee_pos"][:, None].expand(-1, capacity, -1).clone()
    path_q = state["ee_orn"][:, None].expand(-1, capacity, -1).clone()
    lengths = torch.ones(batch, device=root.device, dtype=torch.long)
    rows = torch.where(counts[:, 0] > 0)[0]
    if len(rows):
        if candidate_bank is not None and len(rows) != batch:
            candidate_bank = (
                candidate_bank[0][rows],
                candidate_bank[1][rows],
                {key: value[rows] for key, value in candidate_bank[2].items()},
                *candidate_bank[3:],
            )
        local = {
            key: state[key][rows]
            for key in (
                "root_states",
                "ee_pos",
                "ee_orn",
                "object_state",
                "dof_pos",
                "distractor_states",
            )
        }
        minimum = basket_required_height(
            basket, local, support[rows], lift_margin[rows], support_mask=support_mask[rows]
        )
        path, valid, diagnostic = plan_basket_grasp_batch(
            local,
            sizes[rows],
            limits,
            collision,
            geometry.select(rows),
            dt,
            support_points=support[rows],
            support_mask=support_mask[rows],
            initial_height=local["object_state"][:, 2],
            minimum_height=minimum,
            prefer_realized_body=torch.zeros(len(rows), device=root.device, dtype=torch.bool),
            realized_body=torch.ones(len(rows), device=root.device, dtype=torch.bool),
            minimum_realized_grasp_margin_rad=math.radians(2),
            capacity=capacity,
            **({"candidate_bank": candidate_bank} if candidate_bank is not None else {}),
            **(
                {}
                if candidate_orientations is None
                else {"candidate_orientations": candidate_orientations[rows]}
            ),
        )
        direct[rows] = valid
        accepted = rows[valid]
        path_p[accepted], path_q[accepted], lengths[accepted] = (
            path[0][valid],
            path[1][valid],
            path[2][valid],
        )
        _record_batch_fields(report, "direct_full_path", diagnostic, rows, batch)
    admitted = direct.clone()
    failure = torch.ones(batch, device=root.device, dtype=torch.long)
    rows = (
        torch.where(~direct)[0]
        if not direct_only
        else torch.empty(0, device=root.device, dtype=torch.long)
    )
    if len(rows):
        precheck = closed_nudge_geometry_precheck(
            {key: value[rows] for key, value in state.items()},
            payload_sizes[rows],
            payload_mask[rows],
            basket,
            target_surface.select(rows),
            support_points=support[rows],
            support_mask=support_mask[rows],
            object_centers_local=object_centers_local[rows],
            require_sliding_stability=True,
        )
        eligible = precheck["good"].any(-1)
        _record_batch_fields(
            report,
            "nudge_precheck",
            {
                "safe_direction_mask": precheck["good"],
                "bbox_good_direction_mask": precheck["bbox_good"],
                "arm_path_search_eligible": eligible,
            },
            rows,
            batch,
        )
        failure[rows[~eligible]] = 2
        rows = rows[eligible]
    if len(rows):
        local = {
            key: state[key][rows]
            for key in (
                "root_states",
                "ee_pos",
                "ee_orn",
                "object_state",
                "dof_pos",
                "distractor_states",
            )
        }
        clear, valid, diagnostic = plan_clear_for_nudge_batch(
            local,
            basket,
            limits,
            collision,
            geometry.select(rows),
            target_geometry.select(rows),
            target_surface.select(rows),
            dt,
            capacity=capacity,
        )
        _record_batch_fields(report, "clear_for_nudge", diagnostic, rows, batch)
        ready = rows[valid]
        if len(ready):
            virtual = {
                key: state[key][ready]
                for key in (
                    "root_states",
                    "ee_pos",
                    "ee_orn",
                    "object_state",
                    "dof_pos",
                    "distractor_states",
                )
            }
            local_rows = torch.arange(len(rows), device=root.device)[valid]
            ends = clear[2][valid].long() - 1
            virtual["ee_pos"], virtual["ee_orn"] = (
                clear[0][local_rows, ends],
                clear[1][local_rows, ends],
            )
            _, safe, diagnostic = plan_closed_nudge_batch(
                virtual,
                payload_sizes[ready],
                sizes[ready],
                payload_mask[ready],
                basket,
                limits,
                collision,
                geometry.select(ready),
                target_geometry.select(ready),
                target_surface.select(ready),
                dt,
                support_points=support[ready],
                support_mask=support_mask[ready],
                object_centers_local=object_centers_local[ready],
                current_path_unavailable=torch.ones(
                    len(ready), device=root.device, dtype=torch.bool
                ),
                require_sliding_stability=True,
                capacity=capacity,
            )
            admitted[ready] = safe
            failure[ready] = 2
            _record_batch_fields(report, "closed_nudge", diagnostic, ready, batch)
    report.update(
        direct_reachable_grasp_candidates=counts[:, 0],
        direct_path_available=direct,
        initial_reposition_required=admitted & ~direct,
        admission_failure_code=torch.where(admitted, 0, failure),
    )
    return admitted, report, (path_p, path_q, lengths)


def basket_grasp_access(
    state, episode, basket, collision, geometry, support, limits, dt, *, plan_cache=None
):
    """Admit actual geometry, including layouts which first need a safe closed-jaw nudge."""
    target = state["object_state"][:, :3]
    report = {}
    count = shifted_grasp_counts(
        state, episode, target, collision, geometry, support, limits=limits, diagnostics=report
    )
    report["direct_reachable_grasp_candidates"] = int(count[0])
    report["direct_path_available"] = False
    if bool((count > 0).all()):
        paths, detail = plan_basket_grasp_path(
            state,
            state["object_state"][:, :3],
            episode["asset"].get("intrinsic_size_m", episode["asset"]["size_m"]),
            limits,
            collision,
            geometry,
            dt,
            minimum_height=basket_required_height(
                basket, state, support, float(episode.get("lift_above_rim_m", 0.10))
            ),
            object_support_points=support,
            initial_object_height=state["object_state"][:, 2],
            realized_body=True,
            minimum_realized_grasp_margin_rad=math.radians(2),
        )
        report["direct_full_path"] = detail
        report["direct_path_available"] = paths is not None
        if paths is not None:
            if plan_cache is not None:
                plan_cache.update(
                    paths=paths,
                    diagnostics=detail,
                    maximum_tool_tilt_deg=30.0,
                    state={
                        key: state[key].clone()
                        for key in ("root_states", "dof_pos", "object_state", "distractor_states")
                    },
                )
            return True, report
    clear, clear_report = plan_clear_for_nudge(
        state, basket, limits, collision, geometry, dt, episode=episode
    )
    report["clear_for_nudge"] = clear_report
    if clear is None:
        return False, report
    virtual = {**state, "ee_pos": clear[0][:, -1], "ee_orn": clear[1][:, -1]}
    path, detail = plan_closed_nudge(
        virtual,
        episode,
        basket,
        limits,
        collision,
        geometry,
        dt,
        object_support_points=support,
        current_path_unavailable=True,
        require_sliding_stability=True,
    )
    report["closed_nudge"] = detail
    return path is not None, report


def load_basket_payload_hulls(episodes, maximum_objects, device):
    """Load immutable collider vertices once; all physical predicates stay on-device."""
    descriptors = [[episode, *episode.get("distractors", [])] for episode in episodes]
    if any(len(row) > maximum_objects for row in descriptors):
        raise ValueError("Basket object state capacity omits configured payloads")
    cache = {}
    for row in descriptors:
        for item in row:
            path = Path(item["asset"]["usd_path"]).with_name("collision_support.npy")
            if path not in cache:
                points = np.load(path)
                if points.ndim != 2 or points.shape[1] != 3 or not len(points):
                    raise ValueError(f"Invalid basket collision hull: {path}")
                cache[path] = torch.as_tensor(points, device=device, dtype=torch.float32)
    from gr00t.rl.pick.tensor_rows import pad_device_rows

    paths = list(cache)
    indices = {path: index for index, path in enumerate(paths)}
    empty = next(iter(cache.values())).new_empty(0, 3)
    table, table_mask = pad_device_rows([*cache.values(), empty], fill=torch.nan)
    owner = torch.tensor(
        [
            [
                indices[Path(item["asset"]["usd_path"]).with_name("collision_support.npy")]
                for item in row
            ]
            + [len(paths)] * (maximum_objects - len(row))
            for row in descriptors
        ],
        device=device,
        dtype=torch.long,
    )
    return table[owner], table_mask[owner], owner != len(paths)


def basket_object_checks_batch(basket, state, points, point_mask, payload_mask):
    poses = torch.cat((state["object_state"][:, None], state["distractor_states"]), 1)
    if (
        points.ndim != 4
        or points.shape[:2] != poses.shape[:2]
        or point_mask.shape != points.shape[:-1]
    ):
        raise ValueError("Basket hulls and point masks must match the padded object states")
    if payload_mask.shape != poses.shape[:2]:
        raise ValueError("Basket object mask must match the padded object states")
    valid_points = point_mask & payload_mask[..., None]
    world = _quat_rotate(poses[..., None, 3:7], torch.where(valid_points[..., None], points, 0))
    world += poses[..., None, :3]
    root = state["root_states"]
    inverse = root[:, 3:7].clone()
    inverse[:, 1:] *= -1
    local = _quat_rotate(inverse[:, None, None], world - root[:, None, None, :3])
    low = local.masked_fill(~valid_points[..., None], torch.inf).amin(-2)
    high = local.masked_fill(~valid_points[..., None], -torch.inf).amax(-2)
    margins = torch.minimum(
        low[..., :2] - basket.floor_lower[:2], basket.floor_upper[:2] - high[..., :2]
    ).amin(-1)
    centers = _quat_rotate(inverse[:, None], poses[..., :3] - root[:, None, :3])
    xy = torch.where(payload_mask[..., None], centers[..., :2], 0)
    height, _ = basket.support(xy)
    errors = low[..., 2] - height
    checks = {
        "all_objects_inside_basket": ((margins >= 0) & valid_points.any(-1) | ~payload_mask).all(
            -1
        ),
        "all_objects_supported": ((errors.abs() < 0.02) & valid_points.any(-1) | ~payload_mask).all(
            -1
        ),
    }
    return checks, {
        "object_wall_margin_m": margins.masked_fill(~payload_mask, torch.nan),
        "object_support_error_m": errors.masked_fill(~payload_mask, torch.nan),
    }
