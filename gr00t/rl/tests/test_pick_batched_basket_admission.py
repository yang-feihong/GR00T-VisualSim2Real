"""Admission preserves direct-path precedence, stronger margins and safe-nudge fallback."""

import math
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick import basket_initialization as initial


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("safe_nudge", [False, True])
@pytest.mark.parametrize("reuse_contacts", [False, True])
@pytest.mark.parametrize("prune_impossible", [False, True])
def test_basket_admission_batches_each_branch_without_weakening_initial_gates(
    monkeypatch, device, safe_nudge, reuse_contacts, prune_impossible
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU basket admission")
    root = torch.zeros(3, 13, device=device)
    root[:, 0], root[:, 2], root[:, 3] = root.new_tensor([0, 10, 20]), 0.5, 1
    obj = root.clone()
    obj[:, 2] = 0.7
    state = {
        "root_states": root,
        "object_state": obj,
        "ee_pos": root[:, :3] + root.new_tensor([0.3, 0, 0.4]),
        "ee_orn": root[:, 3:7],
        "dof_pos": root.new_zeros(3, 6),
        "distractor_states": root.new_zeros(3, 0, 13),
    }
    basket = SimpleNamespace(boxes=root.new_tensor([[[0, 0, 0.2]] * 8]))
    support = root.new_tensor([[[0, 0, -0.02], [0, 0, 0.02]]]).expand(3, -1, -1)
    mask = torch.ones(3, 2, device=device, dtype=torch.bool)
    sizes = root.new_full((3, 3), 0.04)
    geometry = SimpleNamespace(select=lambda rows: rows)
    calls = []

    def counts(s, *args, **kwargs):
        assert kwargs["return_candidate_bank"]
        report = {}
        if reuse_contacts:
            report["candidate_bank"] = (
                s["ee_orn"][:, None].expand(-1, 2, -1),
                s["ee_pos"][:, None].expand(-1, 2, -1),
                {"row_token": s["root_states"][:, :1].expand(-1, 2)},
                2,
                0,
            )
        return (s["root_states"][:, :1] < 20).long(), report

    monkeypatch.setattr(initial, "shifted_grasp_counts_batch", counts)
    monkeypatch.setattr(
        initial,
        "closed_nudge_geometry_precheck",
        lambda s, *args, **kwargs: {
            "good": (s["root_states"][:, :1] < 15).expand(-1, 24)
            if prune_impossible
            else torch.ones(len(s["ee_pos"]), 24, device=device, dtype=torch.bool),
            "bbox_good": torch.ones(len(s["ee_pos"]), 24, device=device, dtype=torch.bool),
        },
    )

    def direct(local, *args, **kw):
        assert kw["realized_body"].all() and not kw["prefer_realized_body"].any()
        assert kw["minimum_realized_grasp_margin_rad"] == math.radians(2)
        if reuse_contacts:
            bank = kw["candidate_bank"]
            torch.testing.assert_close(bank[0][:, 0], local["ee_orn"])
            torch.testing.assert_close(bank[1][:, 0], local["ee_pos"])
            torch.testing.assert_close(bank[2]["row_token"][:, 0], local["root_states"][:, 0])
        else:
            assert "candidate_bank" not in kw
        calls.append(("direct", local["root_states"][:, 0].clone()))
        count = len(local["ee_pos"])
        p = (local["ee_pos"] + 0.1)[:, None].expand(-1, 16, -1)
        q = local["ee_orn"][:, None].expand(-1, 16, -1)
        valid = local["root_states"][:, 0] < 5
        return (
            (p, q, torch.full((count,), 4, device=device)),
            valid,
            {
                "selected": torch.where(valid, 0, -1),
                "convex_contact_model": {"center": local["object_state"][:, :3]},
            },
        )

    def clear(local, *args, **kw):
        calls.append(("clear", local["root_states"][:, 0].clone()))
        count = len(local["ee_pos"])
        p = (local["ee_pos"] + 0.2)[:, None].expand(-1, 16, -1)
        q = local["ee_orn"][:, None].expand(-1, 16, -1)
        valid = local["root_states"][:, 0] < 15
        return (p, q, torch.full((count,), 3, device=device)), valid, {"geometric": valid}

    def nudge(local, *args, **kw):
        calls.append(("nudge", local["root_states"][:, 0].clone()))
        assert kw["require_sliding_stability"] and kw["current_path_unavailable"].all()
        assert kw["object_centers_local"].shape == (1, 3)
        torch.testing.assert_close(local["ee_pos"], state["ee_pos"][1:2] + 0.2)
        valid = torch.full((len(local["ee_pos"]),), safe_nudge, device=device, dtype=torch.bool)
        return None, valid, {"stable": valid}

    monkeypatch.setattr(initial, "plan_basket_grasp_batch", direct)
    monkeypatch.setattr(initial, "plan_clear_for_nudge_batch", clear)
    monkeypatch.setattr(initial, "plan_closed_nudge_batch", nudge)

    def run(rows):
        return initial.basket_grasp_access_batch(
            {key: value[rows] for key, value in state.items()},
            sizes[rows],
            sizes[rows, None],
            torch.ones(3, 1, device=device, dtype=torch.bool),
            basket,
            None,
            geometry,
            geometry,
            geometry,
            support[rows],
            mask[rows],
            root.new_zeros(3, 3),
            root.new_tensor([[-3, 3]] * 6),
            0.02,
            lift_margin=root.new_full((3,), 0.1),
            capacity=16,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("Admission planning must not read tensors on the host")

    rows = torch.arange(3, device=device)
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        accepted, report, cache = run(rows)
    assert accepted.tolist() == [True, safe_nudge, False]
    assert report["direct_path_available"].tolist() == [True, False, False]
    assert report["initial_reposition_required"].tolist() == [False, safe_nudge, False]
    assert not any("candidate_bank" in name for name in report)
    assert report["admission_failure_code"].tolist() == [
        0,
        0 if safe_nudge else 2,
        2 if prune_impossible else 1,
    ]
    assert [name for name, _ in calls] == ["direct", "clear", "nudge"]
    torch.testing.assert_close(
        report["direct_full_path.convex_contact_model.center"][:2], obj[:2, :3]
    )
    host = initial.admission_report_to_host(report)
    row_report = initial.admission_report_row(report, host, 1)
    assert row_report["direct_full_path"]["convex_contact_model"]["center"] == [
        10,
        0,
        pytest.approx(0.7),
    ]
    assert calls[0][1].tolist() == [0, 10]
    assert calls[1][1].tolist() == ([10] if prune_impossible else [10, 20])
    assert calls[2][1].tolist() == [10]
    assert cache[2].tolist() == [4, 1, 1]
    torch.testing.assert_close(cache[0][1:, 0], state["ee_pos"][1:])
    reverse = rows.flip(0)
    reordered = run(reverse)
    torch.testing.assert_close(reordered[0][reverse], accepted)
    for key in ("direct_path_available", "initial_reposition_required", "admission_failure_code"):
        torch.testing.assert_close(reordered[1][key][reverse], report[key])
    for value, expected in zip(reordered[2], cache):
        torch.testing.assert_close(value[reverse], expected)
