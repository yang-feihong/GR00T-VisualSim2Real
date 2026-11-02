"""Batched basket placement using the deployed floor and wall collision geometry."""

import math

import torch

from gr00t.rl.auto_open.tensor_metadata import to_host


def sample_layouts(
    asset_rows,
    basket,
    seeds,
    config,
    *,
    counts=None,
    gaps=None,
    return_tensors=False,
    proposal_filter=None,
    target_asset_indices=None,
    randomization_seeds=None,
):
    from gr00t.rl.pick.basket import NoValidBasketLayout

    if len(asset_rows) != len(seeds) or not seeds:
        raise ValueError("Basket layouts require matching nonempty seed and asset rows")
    counts = [None] * len(seeds) if counts is None else counts
    if len(counts) != len(seeds):
        raise ValueError("Basket layout counts must match seeds")
    # A partner's object count must not change this seed's random stream stride.
    width = config.max_objects
    from gr00t.rl.auto_open.tensor_random import episode_uniform

    targets = [None] * len(seeds) if target_asset_indices is None else target_asset_indices
    if len(targets) != len(seeds):
        raise ValueError("Layout target identities must match seeds")
    for assets, count, target in zip(asset_rows, counts, targets, strict=True):
        maximum = min(config.max_objects, len(assets))
        if not assets or len({a["name"] for a in assets}) != len(assets):
            raise ValueError("Basket asset categories must be nonempty and unique")
        if not 1 <= config.min_objects <= maximum:
            raise ValueError("Invalid basket object count range")
        if count is not None and not config.min_objects <= count <= maximum:
            raise ValueError("Requested basket object count outside configured range")
        if target is not None and (count != len(assets) or not 0 <= target < len(assets)):
            raise ValueError("Fixed targets require all supplied assets and a valid asset index")
    device = basket.boxes.device
    if randomization_seeds is None:
        seed_tensor = torch.tensor(seeds, device=device, dtype=torch.int64)
    else:
        seed_tensor = randomization_seeds
        if (
            not torch.is_tensor(seed_tensor)
            or seed_tensor.shape != (len(seeds),)
            or seed_tensor.dtype != torch.int64
            or seed_tensor.device != device
        ):
            raise ValueError(
                "Layout randomization seeds must be an aligned device-side int64 vector"
            )
    asset_counts = torch.tensor(list(map(len, asset_rows)), device=device)
    table_width = max(width, max(map(len, asset_rows)))
    size_table = basket.boxes.new_tensor(
        [
            [asset["size_m"] for asset in assets] + [[0, 0, 0]] * (table_width - len(assets))
            for assets in asset_rows
        ]
    )
    priority = episode_uniform(seed_tensor, 0x5045524D, table_width, 1)[..., 0]
    slots = torch.arange(table_width, device=device)
    chosen = priority.masked_fill(slots[None] >= asset_counts[:, None], torch.inf).argsort(
        dim=-1, stable=True
    )[:, :width]
    chosen = torch.where(chosen < asset_counts[:, None], chosen, 0)
    sizes = size_table.gather(1, chosen[..., None].expand(-1, -1, 3))
    random = episode_uniform(seed_tensor, 0x4C41594F, config.placement_candidates * width, 3)
    random = random.reshape(len(seeds), config.placement_candidates, width, 3)
    selection = episode_uniform(seed_tensor, 0x53454C45, 1, 3)[:, 0]
    requested = asset_counts.new_tensor([-1 if count is None else count for count in counts])
    maximum = asset_counts.clamp_max(config.max_objects)
    sampled_count = (
        config.min_objects + (selection[:, 2] * (maximum - config.min_objects + 1)).long()
    )
    counts_gpu = torch.where(requested < 0, sampled_count, requested)
    mask = torch.arange(width, device=sizes.device)[None] < counts_gpu[:, None]
    yaw = random[..., 2] * (2 * math.pi)
    half = sizes[:, None, :, :2] / 2
    c, s = yaw.cos().abs(), yaw.sin().abs()
    extent = torch.stack(
        (c * half[..., 0] + s * half[..., 1], s * half[..., 0] + c * half[..., 1]), -1
    )
    gap_tensor = sizes.new_tensor(
        gaps if gaps is not None else [[config.object_gap_m, config.wall_gap_m]] * len(seeds)
    )
    low = basket.floor_lower[:2] + extent + gap_tensor[:, None, None, 1, None]
    high = basket.floor_upper[:2] - extent - gap_tensor[:, None, None, 1, None]
    xy = low + random[..., :2] * (high - low)
    height, normal = basket.support(xy)
    corners = (
        xy[..., None, :]
        + xy.new_tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * extent[..., None, :]
    )
    corner_height, corner_normal = basket.support(corners)
    supported = (normal > 0.97) & (corner_normal > 0.97).all(-1)
    supported &= (corner_height - height[..., None]).abs().amax(-1) < 0.003
    supported &= height < basket.rim_z - 0.01
    supported &= (high > low).all(-1)
    delta = (xy[..., :, None, :] - xy[..., None, :, :]).abs()
    separation = (
        delta
        >= (
            extent[..., :, None, :]
            + extent[..., None, :, :]
            + gap_tensor[:, None, None, None, 0, None]
        )
    ).any(-1)
    diagonal = torch.eye(width, device=sizes.device, dtype=torch.bool)
    irrelevant = ~(mask[:, :, None] & mask[:, None, :])
    valid = (supported | ~mask[:, None]).all(-1)
    valid &= (separation | diagonal | irrelevant[:, None]).all(dim=(-1, -2))
    target = (selection[:, 1] * counts_gpu).long()
    if target_asset_indices is not None:
        requested_targets = chosen.new_tensor(target_asset_indices)
        if requested_targets.shape != (len(seeds),):
            raise ValueError("Layout target identities must match the seed batch")
        matches = (chosen == requested_targets[:, None]) & mask
        torch._assert_async(matches.any(-1).all(), "Required target asset omitted from layout")
        target = matches.long().argmax(-1)
    filter_report = None
    if proposal_filter is not None:
        preferred, filter_report = proposal_filter(
            {
                "valid": valid,
                "target": target,
                "xy": xy,
                "height": height,
                "yaw": yaw,
                "sizes": sizes,
                "chosen": chosen,
                "mask": mask,
                "rim_z": basket.rim_z,
                "lift_margin": config.lift_above_rim_m,
            }
        )
        preferred &= valid
        preferred_count = preferred.sum(-1)
        valid = torch.where(preferred.any(-1)[:, None], preferred, valid)
    valid_count = valid.sum(-1)
    rank = (selection[:, 0] * valid_count).long()
    selected = (valid.cumsum(-1) > rank[:, None]).long().argmax(-1)
    owner = torch.arange(len(seeds), device=sizes.device)
    positions = torch.cat(
        (xy[owner, selected], (height[owner, selected] + sizes[..., 2] / 2 + 0.003)[..., None]), -1
    )
    host = to_host(
        {
            "count": counts_gpu,
            "chosen": chosen,
            "position": positions,
            "yaw": yaw[owner, selected],
            "height": height[owner, selected],
            "valid_count": valid_count,
            "target": target,
            "floor": [basket.floor_lower, basket.floor_upper],
            "rim": basket.rim_z,
            "boxes": basket.boxes,
            "randomization_seeds": seed_tensor,
            **(
                {}
                if filter_report is None
                else {
                    "reachability_count": preferred_count,
                    "reachability_evaluated": filter_report["evaluated"].sum(-1),
                    "joint_queries": filter_report.get(
                        "joint_queries_evaluated", filter_report["evaluated"].sum(-1)
                    ),
                    "selected_orientation": filter_report["orientation_local_wxyz"][
                        owner, selected
                    ],
                    **(
                        {
                            "intrinsic_contact_counts": filter_report["intrinsic_contact_counts"],
                            "intrinsic_contact_work": filter_report["intrinsic_contact_work"],
                        }
                        if "intrinsic_contact_counts" in filter_report
                        else {}
                    ),
                    **(
                        {
                            "full_path_layouts_evaluated": filter_report[
                                "full_path_layouts_evaluated"
                            ],
                            "full_path_layouts_passed": filter_report["full_path_layouts_passed"],
                        }
                        if "full_path_layouts_evaluated" in filter_report
                        else {}
                    ),
                    **{
                        key: filter_report[key]
                        for key in (
                            "full_path_hint_layouts_evaluated",
                            "full_path_hint_layouts_passed",
                        )
                        if key in filter_report
                    },
                }
            ),
        }
    )
    results = []
    for row, assets in enumerate(asset_rows):
        count = host["count"][row]
        if not host["valid_count"][row]:
            results.append(
                NoValidBasketLayout(
                    f"No nonpenetrating basket layout in {config.placement_candidates} proposals",
                    {
                        "object_count": count,
                        "asset_names": [assets[i]["name"] for i in host["chosen"][row][:count]],
                    },
                )
            )
            continue
        descriptors = [
            {
                "asset": assets[host["chosen"][row][slot]],
                "basket_local_position": host["position"][row][slot],
                "basket_local_yaw": host["yaw"][row][slot],
                "support_type": "basket",
                "support_kind": "basket",
                "support_z_local": host["height"][row][slot],
            }
            for slot in range(count)
        ]
        diagnostics = {
            "proposal_count": config.placement_candidates,
            "valid_count": host["valid_count"][row],
            "target_index": host["target"][row],
            "object_count": count,
            "floor_bounds_local_m": host["floor"],
            "rim_height_local_m": host["rim"],
            "robot_usd": basket.source,
            "collider_corners_local_m": host["boxes"],
            "object_gap_m": config.object_gap_m if gaps is None else gaps[row][0],
            "wall_gap_m": config.wall_gap_m if gaps is None else gaps[row][1],
            "layout_seed": host["randomization_seeds"][row],
        }
        if filter_report is not None:
            diagnostics.update(
                {
                    key: host[key][row]
                    for key in ("full_path_hint_layouts_evaluated", "full_path_hint_layouts_passed")
                    if key in host
                }
            )
            diagnostics.update(
                reachable_proposals=host["reachability_count"][row],
                reachability_evaluated=host["reachability_evaluated"][row],
                reachability_joint_queries=host["joint_queries"][row],
                reachability_fallback=host["reachability_count"][row] == 0,
                grasp_seed_orientation_local_wxyz=host["selected_orientation"][row],
            )
            if "intrinsic_contact_counts" in host:
                diagnostics.update(
                    intrinsic_contact_candidates=host["intrinsic_contact_counts"][row],
                    intrinsic_contact_fallback=host["intrinsic_contact_counts"][row] == 0,
                    intrinsic_contact_work=host["intrinsic_contact_work"],
                )
            if "full_path_layouts_evaluated" in host:
                diagnostics.update(
                    full_path_layouts_evaluated=host["full_path_layouts_evaluated"][row],
                    full_path_layouts_passed=host["full_path_layouts_passed"][row],
                    preflight_definition="nominal_flat_basket_original_meshes_current_height_plus_minus_1cm_physical_revalidation_required",
                )
        results.append((descriptors, host["target"][row], diagnostics))
    if return_tensors:
        return results, {
            "positions": positions,
            "yaw": yaw[owner, selected],
            "height": height[owner, selected],
            "chosen": chosen,
        }
    return results
