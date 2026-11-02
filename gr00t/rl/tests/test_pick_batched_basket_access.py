"""Padded basket payloads and hypothetical shifts preserve single-scene screening."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open import pose_planning
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick import basket_access, basket_nudge
from gr00t.rl.pick.basket_access import (
    bbox_space_batch,
    object_boxes_in_trunk_batch,
    shifted_grasp_counts_batch,
    slide_stability_batch,
)
from gr00t.rl.tests.test_pick_batched_basket_carry import cases as basket_cases
from gr00t.rl.tests.test_pick_batched_candidate_bank import cases
from gr00t.rl.tests.test_pick_batched_grasp_inverse import fake_inverse

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU basket access")


def test_sliding_stability_matches_scalar_in_rotated_roots_with_padded_hulls(monkeypatch):
    state, points, mask, _ = basket_cases()
    root, obj = state["root_states"], state["object_state"]
    offsets = points.new_tensor([[-0.02, 0, -0.015], [-0.02, 0, 0.03], [-0.02, 0, 0.01]])
    hits = obj[:, None, :3] + _quat_rotate(root[:, None, 3:7], offsets[None])
    direction = _quat_rotate(root[:, None, 3:7], points.new_tensor([[1, 0, 0]]).expand(2, 3, -1))
    normal = _quat_rotate(
        root[:, None, 3:7], points.new_tensor([[-1, 0, 0], [-1, 0, 0], [-1, 0, 0.4]])[None]
    )
    center = points.new_tensor([[0, 0, 0], [0.005, 0, 0]])
    owners = torch.arange(2, device="cuda").repeat_interleave(3)

    def forbidden(*args, **kwargs):
        raise AssertionError("Sliding stability must not read tensors on the host")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        result = slide_stability_batch(
            root[owners],
            obj[owners],
            points[owners],
            center[owners],
            hits.flatten(0, 1),
            normal.flatten(0, 1),
            direction.flatten(0, 1),
            0.5,
            support_mask=mask[owners],
        )
    assert result[0][0] and not result[0][1] and not result[0][2]
    for row in range(2):
        scalar = basket_nudge.slide_stability(
            root[row],
            obj[row],
            points[row, mask[row]],
            center[row],
            hits[row],
            normal[row],
            direction[row],
            0.5,
        )
        for combined, single in zip(result, scalar):
            torch.testing.assert_close(combined.reshape(2, 3)[row], single)


def test_batched_payload_boxes_and_clearance_match_scalar_with_absent_objects():
    state, target, sizes, _, _ = cases()
    distractors = state["object_state"][:, None].expand(-1, 2, -1).clone()
    distractors[:, :, 0] += target.new_tensor([0.1, 0.3])
    state["distractor_states"] = distractors
    all_sizes = sizes[:, None].expand(-1, 3, -1)
    present = torch.tensor([[True, True, True], [True, True, False]], device="cuda")
    basket = SimpleNamespace(
        floor_lower=target.new_tensor([-1, -1, 0]), floor_upper=target.new_tensor([1, 1, 0])
    )
    centers, extents = object_boxes_in_trunk_batch(state, all_sizes)
    proposals = centers[:, :1, :2] + target.new_tensor([[[0, 0], [0.03, 0], [0, 0.05]]])
    together = bbox_space_batch(
        proposals, extents[:, 0, :2], basket, centers[:, 1:, :2], extents[:, 1:, :2], present[:, 1:]
    )
    for row in range(2):
        local = {key: value[row : row + 1] for key, value in state.items()}
        c, e = basket_nudge.object_boxes_in_trunk({}, local, sizes=all_sizes[row])
        torch.testing.assert_close(centers[row], c)
        torch.testing.assert_close(extents[row], e)
        expected = basket_nudge.bbox_space(
            proposals[row],
            e[0, :2],
            basket,
            c[1:, :2][present[row, 1:]],
            e[1:, :2][present[row, 1:]],
        )
        torch.testing.assert_close(together[row], expected)


def test_future_grasp_counts_match_scalar_and_sparse_target_mask(monkeypatch):
    state, target, sizes, hulls, mask = cases()
    state["ee_pos"] = target + target.new_tensor([-0.2, 0, 0.2])
    state["dof_pos"] = target.new_zeros(2, 6)
    targets = target[:, None] + target.new_tensor([[[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0]]])
    allowed = torch.tensor([[True, True, True], [True, True, False]], device="cuda")
    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6, device="cuda"),
        clearance=lambda g, r, j, **kw: r.new_full(r.shape[:-1], 0.02),
        gripper_clearance=lambda g, p, q, **kw: p.new_full(p.shape[:-1], 0.02),
        gripper_closing_clearance=lambda g, p, q, *a, **kw: p.new_full(p.shape[:-1], 0.02),
    )
    geometry = SimpleNamespace(supports_query_mask=True)
    geometry.select = lambda rows: geometry
    limits = target.new_tensor([[-3, 3]] * 6)
    monkeypatch.setattr(pose_planning, "solve_pose_targets", fake_inverse)

    def forbidden(*args, **kwargs):
        raise AssertionError("Future grasp checks must remain on the GPU")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        counts, report = shifted_grasp_counts_batch(
            state,
            sizes,
            targets,
            collision,
            geometry,
            hulls,
            limits,
            support_mask=mask,
            target_mask=allowed,
        )
    assert counts.any() and counts[1, 2] == 0
    with pytest.raises(ValueError, match="one unmasked target"):
        shifted_grasp_counts_batch(
            state,
            sizes,
            targets,
            collision,
            geometry,
            hulls,
            limits,
            return_candidate_bank=True,
        )
    with pytest.raises(ValueError, match="one unmasked target"):
        shifted_grasp_counts_batch(
            state,
            sizes,
            targets[:, :1],
            collision,
            geometry,
            hulls,
            limits,
            target_mask=allowed[:, :1],
            return_candidate_bank=True,
        )
    reused, direct_report = shifted_grasp_counts_batch(
        state,
        sizes,
        targets[:, :1],
        collision,
        geometry,
        hulls,
        limits,
        support_mask=mask,
        return_candidate_bank=True,
    )
    torch.testing.assert_close(reused, counts[:, :1])
    bank = direct_report["candidate_bank"]
    assert bank[0].shape[:2] == bank[1].shape[:2]
    assert bank[0].shape[0] == len(target)
    assert bank[3] == bank[0].shape[1]
    monkeypatch.setattr(
        basket_nudge,
        "safe_inverse_paths",
        lambda s, p, q, *a, **kw: (
            torch.ones(len(p), device=p.device, dtype=torch.bool),
            p.new_zeros(len(p)),
        ),
    )
    for row in range(2):
        local = {key: value[row : row + 1] for key, value in state.items()}
        legacy = basket_nudge.shifted_grasp_counts(
            local,
            {"asset": {"size_m": sizes[row].tolist()}},
            targets[row, allowed[row]],
            collision,
            geometry,
            hulls[row, mask[row]],
            limits=limits,
        )
        torch.testing.assert_close(counts[row, allowed[row]], legacy)
    monkeypatch.setattr(basket_access, "MAX_ACCESS_POSES_PER_BLOCK", 13)
    chunked, _ = shifted_grasp_counts_batch(
        state,
        sizes,
        targets,
        collision,
        geometry,
        hulls,
        limits,
        support_mask=mask,
        target_mask=allowed,
    )
    torch.testing.assert_close(chunked, counts)
    empty, _ = shifted_grasp_counts_batch(
        state,
        sizes,
        targets,
        collision,
        geometry,
        hulls,
        limits,
        support_mask=mask,
        target_mask=torch.zeros_like(allowed),
    )
    assert not empty.any()
