"""Bounded candidate cohorts preserve ownership, exhaustion and acceptance."""

import pytest
import torch

from gr00t.rl.pick.grasp_search import (
    balanced_pocket_priority,
    compact_grasp_bank,
    search_grasp_batches,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_balanced_pocket_priority_prefers_central_deep_seating_and_preserves_rows(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU search")
    q = torch.tensor([1.0, 0, 0, 0], device=device).expand(2, 3, -1).clone()
    q[:, 2] = q.new_tensor([0, 1, 0, 0])
    center = q.new_zeros(2, 3, 3)
    fixed, moving = center.clone(), center.clone()
    fixed[..., 2], moving[..., 2] = -0.02, 0.02
    fixed[:, 0, 1], moving[:, 0, 1] = 0.03, 0.03
    depth = q.new_tensor([[0.005, 0.015, 0.025], [0.025, 0.015, 0.005]])
    quality = {
        "fixed_contact_local": fixed,
        "moving_contact_local": moving,
        "center_local": center,
        "center_depth_margin_m": depth,
        "pocket_fit": torch.ones(2, 3, device=device, dtype=torch.bool),
    }
    bank = (q, center, quality, 3, 0)
    actual, lever, _ = balanced_pocket_priority(bank, q[:, 0])
    assert actual[0].argmin() == 2
    torch.testing.assert_close(lever[0], q.new_tensor([0.03, 0, 0]))
    for row in range(2):
        single = (
            q[row : row + 1],
            center[row : row + 1],
            {k: v[row : row + 1] for k, v in quality.items()},
            3,
            0,
        )
        expected = balanced_pocket_priority(single, q[row : row + 1, 0])[0]
        torch.testing.assert_close(actual[row : row + 1], expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_candidate_priority_preserves_late_fallback(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU search")
    p = torch.zeros(1, 6, 3, device=device)
    q = p.new_tensor([1, 0, 0, 0]).expand(1, 6, -1)
    bank = (q, p, {"pocket_fit": torch.ones(1, 6, device=device, dtype=torch.bool)}, 6, 0)
    state = {"ee_pos": p[:, 0], "ee_orn": q[:, 0]}
    calls = []

    def evaluate(rows, local):
        mask = local[2]["pocket_fit"]
        calls.append(mask.clone())
        accepted = mask[:, 0]
        return (
            (p[:, :1], q[:, :1], rows.new_ones(1)),
            accepted,
            {"selected": torch.where(accepted, 0, -1)},
        )

    _, valid, _ = search_grasp_batches(
        state,
        bank,
        evaluate,
        capacity=1,
        cohort_size=1,
        candidate_priority=p.new_tensor([[5, 4, 3, 2, 1, 0]]),
    )
    assert bool(valid[0]) and len(calls) == 6
    assert bool(calls[0][0, 5]) and bool(calls[-1][0, 0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compaction_preserves_all_eligible_candidates_and_selected_paths(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU search")
    width = 30481
    goals = torch.zeros(3, width, 3, device=device)
    goals[..., 0] = torch.arange(width, device=device)
    quats = goals.new_tensor([1, 0, 0, 0]).expand(3, width, -1)
    eligible = torch.zeros(3, width, device=device, dtype=torch.bool)
    eligible[0, [111, 30000]] = True
    eligible[1, [15, 20000]] = True
    bank = (quats, goals, {"pocket_fit": eligible}, width, 0)
    state = {"ee_pos": goals[:, 0], "ee_orn": quats[:, 0]}

    def evaluate(rows, bank):
        mask = bank[2]["pocket_fit"]
        selected = mask.long().argmax(1)
        index = torch.arange(len(rows), device=device)
        p = bank[1][index, selected][:, None]
        q = bank[0][index, selected][:, None]
        return (
            (p, q, torch.ones_like(rows)),
            mask.any(1),
            {
                "selected": selected,
                "nested": {"chosen_goal": bank[1][index, selected]},
            },
        )

    expected = search_grasp_batches(state, bank, evaluate, capacity=2)

    def forbidden(*args, **kwargs):
        raise AssertionError("Compaction must not transfer geometry to the host")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        packed, report = compact_grasp_bank(bank)
        actual = search_grasp_batches(state, packed, evaluate, capacity=2)
    assert packed[0].shape[1] == 4
    assert report["uncompressed_geometry_pass_counts.pocket_fit"].tolist() == [2, 2, 0]
    for left, right in zip(expected[0], actual[0]):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(expected[1], actual[1])
    torch.testing.assert_close(
        expected[2]["nested"]["chosen_goal"][:2], actual[2]["nested"]["chosen_goal"][:2]
    )
    for row in (0, 1):
        source = report["candidate_source_indices"][row, actual[2]["selected"][row]]
        assert source == expected[2]["selected"][row]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_candidate_cohorts_stop_solved_rows_without_dropping_late_feasible_poses(
    device, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU search")
    goals = torch.zeros(3, 12, 3, device=device)
    goals[:, :, 0] = torch.arange(12, device=device)
    q = goals.new_tensor([1, 0, 0, 0]).expand(3, 12, -1)
    state = {"ee_pos": goals[:, 0], "ee_orn": q[:, 0]}
    eligible = torch.ones(3, 12, device=device, dtype=torch.bool)
    eligible[2] = False
    calls = []

    def evaluate(rows, bank):
        mask = bank[2]["pocket_fit"]
        calls.append((rows.clone(), mask.clone()))
        wanted = rows.new_tensor([1, 9, 0])[rows]
        accepted = mask.gather(1, wanted[:, None])[:, 0]
        p = goals[rows, wanted][:, None].expand(-1, 2, -1)
        length = torch.full_like(rows, 2)
        selected = torch.where(accepted, wanted, -1)
        return (p, q[rows, :2], length), accepted, {"selected": selected}

    def forbidden(*args, **kwargs):
        raise AssertionError("Search must not read candidate tensors on the host")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        path, accepted, diagnostic = search_grasp_batches(
            state, (q, goals, {"pocket_fit": eligible}, 12, 0), evaluate, capacity=4
        )
    torch.testing.assert_close(accepted, eligible.new_tensor([True, True, False]))
    torch.testing.assert_close(diagnostic["selected"], torch.tensor([1, 9, -1], device=device))
    torch.testing.assert_close(calls[0][0], torch.arange(2, device=device))
    assert len(calls) == 3
    assert calls[1][0].tolist() == calls[2][0].tolist() == [1]
    assert all(mask.sum(-1).max() <= 4 for _, mask in calls)
    torch.testing.assert_close(path[0][0, :, 0], goals.new_full((4,), 1))
    torch.testing.assert_close(path[0][1, :, 0], goals.new_full((4,), 9))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_endpoint_mask_preserves_original_cohort_boundaries(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU search")
    goals = torch.zeros(2, 6, 3, device=device)
    goals[..., 0] = torch.arange(6, device=device)
    q = goals.new_tensor([1, 0, 0, 0]).expand(2, 6, -1)
    state = {"ee_pos": goals[:, 0], "ee_orn": q[:, 0]}
    available = torch.ones(2, 6, device=device, dtype=torch.bool)
    mask = torch.zeros_like(available)
    mask[0, [1, 5]] = True
    calls = []

    def evaluate(rows, bank):
        candidates = bank[2]["pocket_fit"]
        calls.append(rows)
        valid = candidates & mask[rows]
        # The later candidate is cheaper, but must not enter the earlier cohort.
        cost = goals.new_tensor([100, 10, 100, 100, 100, 0]).expand(len(rows), -1)
        selected = cost.masked_fill(~valid, torch.inf).argmin(-1)
        accepted = valid.any(-1)
        p = goals[rows, selected][:, None].expand(-1, 2, -1)
        return (
            (p, q[rows, :2], torch.full_like(rows, 2)),
            accepted,
            {"selected": torch.where(accepted, selected, -1)},
        )

    bank = q, goals, {"pocket_fit": available}, 6, 0
    baseline = search_grasp_batches(state, bank, evaluate, capacity=4)
    calls.clear()
    screened = search_grasp_batches(state, bank, evaluate, capacity=4, candidate_mask=mask)
    torch.testing.assert_close(baseline[1], screened[1])
    torch.testing.assert_close(screened[2]["selected"], goals.new_tensor([1, -1]).long())
    torch.testing.assert_close(baseline[0][0][0], screened[0][0][0], rtol=0, atol=0)
    assert len(calls) == 1 and calls[0].tolist() == [0]
