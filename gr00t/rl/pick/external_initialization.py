"""External scene admission predicates on padded, scene-owned GPU object states."""

import torch

from gr00t.rl.pick.distractors import object_category
from gr00t.rl.pick.scene_geometry import SURFACE_NAMES


class ExternalAdmissionInputs:
    """Upload descriptor metadata once per proposal, not once per spatial query."""

    def __init__(self, episodes, origins, slots):
        descriptors = [[episode, *episode.get("distractors", [])] for episode in episodes]
        if len(episodes) != len(origins) or any(len(row) > slots for row in descriptors):
            raise ValueError("External admission metadata must match the scene and object slots")
        self.counts = [len(row) for row in descriptors]
        categories = sorted({object_category(item["asset"]) for row in descriptors for item in row})
        category_ids = {name: index for index, name in enumerate(categories)}
        support, category, positions, active = [], [], [], []
        for row in descriptors:
            support.append(
                [SURFACE_NAMES.index(item["support_type"]) for item in row]
                + [-1] * (slots - len(row))
            )
            category.append(
                [category_ids[object_category(item["asset"])] for item in row]
                + [-1] * (slots - len(row))
            )
            positions.append(
                [item["object_position"] for item in row] + [[0, 0, 0]] * (slots - len(row))
            )
            active.append([True] * len(row) + [False] * (slots - len(row)))
        self.expected_support = torch.tensor(support, device=origins.device, dtype=torch.long)
        self.category_ids = torch.tensor(category, device=origins.device, dtype=torch.long)
        self.mask = torch.tensor(active, device=origins.device, dtype=torch.bool)
        # Match descriptor/world conversion before narrowing to the simulator dtype.
        self.initial_positions = (
            torch.tensor(positions, device=origins.device, dtype=torch.float64)
            + origins[:, None].double()
        ).to(origins.dtype)


def scene_object_checks_batch(state, geometry, metadata):
    """Keep all four original support/category/error/stability gates unchanged."""
    poses = torch.cat((state["object_state"][:, None], state["distractor_states"]), 1)
    if poses.shape[:2] != metadata.mask.shape:
        raise ValueError("External object states must match the padded admission metadata")
    active = metadata.mask
    positions = torch.where(active[..., None], poses[..., :3], state["root_states"][:, None, :3])
    ends = positions.clone()
    ends[..., 2] = -0.20
    hits, _, _ = geometry.rays(positions, ends)
    labels = geometry.last_hit_types.long()
    labels = torch.where((labels == 5) & (hits[..., 2] < 0.12), 0, labels)
    pairs = metadata.category_ids * len(SURFACE_NAMES) + labels
    duplicated = (
        (pairs[..., :, None] == pairs[..., None, :]) & active[..., :, None] & active[..., None, :]
    )
    duplicated &= ~torch.eye(poses.shape[1], device=poses.device, dtype=torch.bool)[None]
    error = (positions[..., 2] - metadata.initial_positions[..., 2]).abs()
    speed = poses[..., 7:10].norm(dim=-1)
    angular_speed = poses[..., 10:13].norm(dim=-1)
    stable = (speed < 0.03) & (angular_speed < 0.15)
    return {
        "object_support_labels_match": ((labels == metadata.expected_support) | ~active).all(-1),
        "object_category_support_pairs_unique": ~duplicated.any(dim=(-1, -2)),
        "all_object_support_errors_below_2cm": ((error < 0.02) | ~active).all(-1),
        "all_objects_stable": (stable | ~active).all(-1),
    }, {
        "actual_support_type_ids": labels.masked_fill(~active, -1),
        "object_support_error_m": error.masked_fill(~active, torch.nan),
        "all_object_speeds_m_s": speed.masked_fill(~active, torch.nan),
        "all_object_angular_speeds_rad_s": angular_speed.masked_fill(~active, torch.nan),
        "distractor_states": state["distractor_states"],
    }
