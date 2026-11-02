"""Object-frame contact proposals stay independent of partner rows and world yaw."""

import pytest
import torch
from gr00t.rl.pick.basket_intrinsic_proposals import IntrinsicGraspProposals

from gr00t.rl.pick.grasp_quality import convex_grasp_quality
from gr00t.rl.pick.pocket import BASKET_MINIMUM_CENTER_DEPTH_M


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native GPU contact profiles")
def test_intrinsic_bank_matches_split_batches_and_transported_formal_contacts(monkeypatch):
    from gr00t.rl.pick import basket_intrinsic_proposals as module

    signs = torch.tensor(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        device="cuda",
        dtype=torch.float32,
    )
    sizes = signs.new_tensor([[0.04, 0.04, 0.04], [0.05, 0.05, 0.05]])
    support = signs[None] * sizes[:, None] / 2
    mask = torch.ones(2, 8, device="cuda", dtype=torch.bool)
    seeds = torch.tensor([1700, 1702], device="cuda", dtype=torch.int64)
    evaluated = []

    def tracked(*args, **kwargs):
        assert kwargs["depth_fractions"].shape[-1] == 1
        assert args[2].shape[0] * args[2].shape[1] <= 256
        evaluated.append(len(args[0]))
        return convex_grasp_quality(*args, **kwargs)

    monkeypatch.setattr(module, "convex_grasp_quality", tracked)
    batch = IntrinsicGraspProposals(support, mask, sizes, seeds, count=64, sample_budget=256)
    assert sum(evaluated) == 128 and batch.work["peak_transformed_points"] <= 256
    assert batch.counts.min() > 0
    for row in range(2):
        single = IntrinsicGraspProposals(
            support[row : row + 1],
            mask[row : row + 1],
            sizes[row : row + 1],
            seeds[row : row + 1],
            count=64,
            sample_budget=256,
        )
        for name in ("orientations", "centers", "valid", "order", "counts"):
            torch.testing.assert_close(
                getattr(batch, name)[row], getattr(single, name)[0], rtol=0, atol=0
            )
    owners = torch.tensor([1, 0, 1], device="cuda")
    draw = signs.new_tensor([0.0, 0.5, 0.99])
    yaw = signs.new_tensor([0.3, -0.2, 0.8])
    q, center, available = batch.sample(owners, draw, yaw)
    calls_before_view = len(evaluated)
    view = batch.select(owners)
    view_q, view_center, view_available = view.sample(torch.arange(3, device="cuda"), draw, yaw)
    assert len(evaluated) == calls_before_view
    torch.testing.assert_close(view_q, q, rtol=0, atol=0)
    torch.testing.assert_close(view_center, center, rtol=0, atol=0)
    torch.testing.assert_close(view_available, available, rtol=0, atol=0)
    empty = batch.select(owners[:1])
    empty.counts.zero_()
    assert not empty.sample(
        torch.zeros(2, device="cuda", dtype=torch.long),
        signs.new_tensor([0, 1]),
        signs.new_zeros(2),
    )[2].any()
    assert batch.counts.min() > 0
    final_q, final_center, _ = batch.sample(owners, torch.ones_like(draw), yaw)
    close_q, close_center, _ = batch.sample(owners, draw.new_full(draw.shape, 0.999999), yaw)
    torch.testing.assert_close(final_q, close_q, rtol=0, atol=0)
    torch.testing.assert_close(final_center, close_center, rtol=0, atol=0)
    index = batch.order[owners, (draw * batch.counts[owners]).long()]
    object_q = torch.stack(((yaw / 2).cos(), yaw * 0, yaw * 0, (yaw / 2).sin()), -1)
    oracle = convex_grasp_quality(
        q[:, None],
        object_q,
        support[owners],
        support_mask=mask[owners],
        depth_fractions=batch.fractions[owners, index, None, None],
        support_normal_world=signs.new_tensor([0, 0, 1]).expand(3, -1),
        minimum_center_depth_m=BASKET_MINIMUM_CENTER_DEPTH_M,
    )
    expected = oracle["center_local"][:, 0].clone()
    expected[:, 2] -= 0.0005
    torch.testing.assert_close(center, expected, rtol=0, atol=1e-6)
    assert available.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native GPU contact profiles")
def test_intrinsic_bank_is_independent_of_padded_partner_geometry():
    signs = torch.tensor(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        device="cuda",
        dtype=torch.float32,
    )
    support = signs[None] * 0.02
    mask = torch.ones(1, 8, device="cuda", dtype=torch.bool)
    sizes = signs.new_full((1, 3), 0.04)
    seeds = torch.tensor([1700], device="cuda", dtype=torch.int64)
    compact = IntrinsicGraspProposals(support, mask, sizes, seeds, count=32)
    padded = signs.new_full((1, 10005, 3), torch.nan)
    padded[:, :8] = support
    padded_mask = torch.arange(10005, device="cuda")[None] < 8
    expanded = IntrinsicGraspProposals(padded, padded_mask, sizes, seeds, count=32)
    assert expanded.work["used_support_points"] == 9984
    for name in ("orientations", "centers", "valid", "order", "counts"):
        torch.testing.assert_close(getattr(compact, name), getattr(expanded, name), rtol=0, atol=0)


def test_layout_metadata_serializes_intrinsic_candidates_and_fallback():
    import json
    from types import SimpleNamespace

    from gr00t.rl.pick.basket import BasketConfig
    from gr00t.rl.pick.basket_layout import sample_layouts

    basket = SimpleNamespace(
        boxes=torch.zeros(1, 8, 3),
        floor_lower=torch.tensor([-0.3, -0.2, 0.0]),
        floor_upper=torch.tensor([0.3, 0.2, 0.1]),
        rim_z=torch.tensor(0.3),
        source="test",
        support=lambda xy: (torch.full(xy.shape[:-1], 0.1), torch.ones(xy.shape[:-1])),
    )

    def report(data):
        valid = data["valid"]
        return valid, {
            "evaluated": valid,
            "orientation_local_wxyz": torch.tensor([1, 0, 0, 0]).expand(*valid.shape, 4),
            "intrinsic_contact_counts": torch.tensor([7, 0]),
            "intrinsic_contact_work": {"contacts_per_object": 1000, "depths_per_contact": 1},
        }

    layouts = sample_layouts(
        [[{"name": "target", "size_m": [0.04, 0.04, 0.04]}]] * 2,
        basket,
        [7, 9],
        BasketConfig(min_objects=1, max_objects=1, placement_candidates=8),
        counts=[1, 1],
        proposal_filter=report,
    )
    assert [layout[2]["intrinsic_contact_candidates"] for layout in layouts] == [7, 0]
    assert [layout[2]["intrinsic_contact_fallback"] for layout in layouts] == [False, True]
    json.dumps(layouts)
