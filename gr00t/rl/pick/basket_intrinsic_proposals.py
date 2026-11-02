"""Reusable object-frame contacts before drawing coupled basket layout requests."""

import math

import torch

from gr00t.rl.auto_open.motion_math import _quat_multiply
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.auto_open.tensor_random import episode_uniform
from gr00t.rl.pick.grasp_quality import convex_grasp_quality, supported_pinch_alignment
from gr00t.rl.pick.planning import grasp_environment_clearance, object_frame_grasp_bank
from gr00t.rl.pick.pocket import BASKET_MINIMUM_CENTER_DEPTH_M
from gr00t.rl.pick.support_features import uniform_surface_support


def cached_intrinsic_proposals(inputs, seeds):
    cached = getattr(inputs, "_intrinsic_proposal_cache", None)
    if cached is None:
        print(
            f"[task:layout] intrinsic_cache objects={len(seeds)} contacts_per_object=1000",
            flush=True,
        )
        cached = IntrinsicGraspProposals(inputs.support, inputs.support_mask, inputs.sizes, seeds)
        inputs._intrinsic_proposal_cache = cached
        inputs._intrinsic_proposal_seeds = seeds.clone()
    else:
        torch._assert_async(
            (inputs._intrinsic_proposal_seeds == seeds).all(), "Intrinsic source seed order changed"
        )
    return cached


class IntrinsicGraspProposals:
    """One contact depth per draw; no layout, joint-seed or body-height grid."""

    def __init__(self, support, support_mask, sizes, seeds, *, count=1000, sample_budget=4_000_000):
        if not 1 <= count <= 1000:
            raise ValueError("Intrinsic proposal count must be in 1..1000")
        n = len(sizes)
        if support.ndim != 3 or support.shape[0] != n or support.shape[-1] != 3:
            raise ValueError("Intrinsic support must have shape [objects, points, xyz]")
        if support_mask.shape != support.shape[:2] or sizes.shape != (n, 3) or seeds.shape != (n,):
            raise ValueError("Intrinsic proposal metadata must match the source object rows")
        if not n or not support.shape[1] or support_mask.dtype != torch.bool:
            raise ValueError("Intrinsic proposal sources must be nonempty with a boolean mask")
        if (
            not support.is_cuda
            or sizes.device != support.device
            or seeds.device != support.device
            or support_mask.device != support.device
            or sizes.dtype != support.dtype
            or support.dtype != torch.float32
            or seeds.dtype != torch.int64
        ):
            raise ValueError(
                "Intrinsic contacts require aligned CUDA float32 geometry and int64 seeds"
            )
        torch._assert_async(
            support_mask.any(-1).all(), "Each intrinsic source needs support points"
        )
        source_points = support.shape[1]
        if source_points > 9984:
            support, support_mask = uniform_surface_support(support, support_mask, count=9984)
        self.support, self.support_mask = support, support_mask
        if sample_budget < support.shape[1]:
            raise ValueError("Contact workspace must fit one original support query")
        identity = sizes.new_zeros(n, 4)
        identity[:, 0] = 1
        bank = object_frame_grasp_bank(identity, sizes)
        allowed = _quat_rotate(bank, sizes.new_tensor([1, 0, 0]))[..., 2]
        allowed = allowed <= -math.cos(math.radians(30))
        order = allowed.long().argsort(dim=-1, descending=True, stable=True)
        draw = episode_uniform(seeds, 0x494E5452, count, 2)
        rank = (draw[..., 0] * allowed.sum(-1)[:, None]).long().clamp_max(bank.shape[1] - 1)
        chosen = order.gather(1, rank)
        selected_allowed = allowed.gather(1, chosen).flatten()
        self.orientations = bank.gather(1, chosen[..., None].expand(-1, -1, 4))
        self.fractions = draw[..., 1]
        self.centers = sizes.new_zeros(n, count, 3)
        self.valid = torch.zeros(n, count, device=sizes.device, dtype=torch.bool)
        self.qualities = {}
        capacity = sample_budget // support.shape[1]
        flat_q = self.orientations.flatten(0, 1)
        for start in range(0, n * count, capacity):
            end = min(start + capacity, n * count)
            source = torch.arange(start, end, device=sizes.device) // count
            quality = convex_grasp_quality(
                flat_q[start:end, None],
                identity[source],
                support[source],
                support_mask=support_mask[source],
                depth_fractions=self.fractions.flatten()[start:end, None, None],
                support_normal_world=sizes.new_tensor([0, 0, 1]).expand(len(source), -1),
                minimum_center_depth_m=BASKET_MINIMUM_CENTER_DEPTH_M,
            )
            for name, value in quality.items():
                if name not in self.qualities:
                    self.qualities[name] = value.new_empty(n, count, *value.shape[2:])
                self.qualities[name].flatten(0, 1)[start:end] = value[:, 0]
            self.centers.view(-1, 3)[start:end] = quality["center_local"][:, 0]
            self.valid.flatten()[start:end] = (
                quality["pocket_fit"]
                & (quality["antipodal_margin"] >= 0)
                & quality["support_pinch_compatible"]
            )[:, 0] & selected_allowed[start:end]
        self.centers[..., 2] -= 0.0005
        self.order = self.valid.long().argsort(dim=-1, descending=True, stable=True)
        self.counts = self.valid.sum(-1)
        self.work = {
            "objects": n,
            "contacts_per_object": count,
            "depths_per_contact": 1,
            "source_support_points": source_points,
            "used_support_points": support.shape[1],
            "contact_requests": n * count,
            "peak_transformed_points": min(capacity, n * count) * support.shape[1],
        }

    def sample(self, owners, draw, yaw):
        if owners.ndim != 1 or draw.shape != owners.shape or yaw.shape != owners.shape:
            raise ValueError("Intrinsic sample owners, draws and yaws must be aligned vectors")
        torch._assert_async(((draw >= 0) & (draw <= 1)).all(), "Intrinsic draws must be in [0, 1]")
        rank = torch.minimum(
            (draw * self.counts[owners]).long(), (self.counts[owners] - 1).clamp_min(0)
        )
        chosen = self.order[owners, rank]
        half = yaw / 2
        rotation = torch.stack((half.cos(), half * 0, half * 0, half.sin()), -1)
        return (
            _quat_multiply(rotation, self.orientations[owners, chosen]),
            self.centers[owners, chosen],
            self.counts[owners] > 0,
        )

    def select(self, rows):
        """Gather a GPU view of cached contacts; never rerun source contact queries."""
        result = type(self).__new__(type(self))
        for name in ("orientations", "fractions", "centers", "valid", "order", "counts"):
            setattr(result, name, getattr(self, name)[rows])
        result.qualities = {name: value[rows] for name, value in self.qualities.items()}
        result.support, result.support_mask = self.support[rows], self.support_mask[rows]
        result.work = {**self.work, "selected_objects": len(rows), "cached_view": True}
        return result

    def candidate_bank(
        self, state, collision, geometry, *, minimum_clearance_m=0.006, maximum_tool_tilt_deg=30.0
    ):
        """Transport intrinsic contacts; recompute support, cone and full obstacle checks."""
        objects = state["object_state"]
        if len(objects) != len(self.counts):
            raise ValueError("Transported contacts must match their source rows")
        orientations = _quat_multiply(objects[:, None, 3:7], self.orientations)
        quality = {name: value.clone() for name, value in self.qualities.items()}
        normal = _quat_rotate(state["root_states"][:, 3:7], objects.new_tensor([0, 0, 1]))
        fit, sine = supported_pinch_alignment(
            orientations, quality["fixed_contact_local"], quality["moving_contact_local"], normal
        )
        axis = _quat_rotate(orientations, objects.new_tensor([1, 0, 0]))
        eligible = (
            self.valid & fit & (axis[..., 2] <= -math.cos(math.radians(maximum_tool_tilt_deg)))
        )
        quality["support_pinch_compatible"] = fit
        quality["support_contact_line_sine"] = sine
        owners, slots = torch.where(eligible)
        capacity = max(1, 4_000_000 // self.support.shape[1])
        # Profile interpolation can change at floating-point boundaries. Recheck
        # only the sparse intrinsic survivors with the original contact operator.
        for start in range(0, len(owners), capacity):
            rows, columns = owners[start : start + capacity], slots[start : start + capacity]
            fresh = convex_grasp_quality(
                orientations[rows, columns, None],
                objects[rows, 3:7],
                self.support[rows],
                support_mask=self.support_mask[rows],
                depth_fractions=self.fractions[rows, columns, None, None],
                support_normal_world=normal[rows],
                minimum_center_depth_m=BASKET_MINIMUM_CENTER_DEPTH_M,
            )
            for name, value in fresh.items():
                quality[name][rows, columns] = value[:, 0]
        eligible &= quality["pocket_fit"] & (quality["antipodal_margin"] >= 0)
        eligible &= quality["support_pinch_compatible"]
        quality["pocket_fit"] &= eligible
        center = quality["center_local"].clone()
        center[..., 2] -= 0.0005
        goals = objects[:, None, :3] - _quat_rotate(orientations, center)
        quality["command_center_local"] = center
        quality["environment_query_evaluated"] = eligible
        quality["environment_clearance_m"] = grasp_environment_clearance(
            collision,
            geometry,
            goals,
            orientations,
            quality["aperture_angle_rad"],
            query_mask=eligible,
            minimum_clearance_m=minimum_clearance_m,
        )
        quality["environment_clearance_query_cap_m"] = torch.full_like(
            quality["environment_clearance_m"], minimum_clearance_m + 0.0001
        )
        return orientations, goals, quality, orientations.shape[1], 0
