"""Basket approach prefixes and shared grasp composition stay row-local on the GPU."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick import basket_grasp, grasp_inverse
from gr00t.rl.pick.basket_entry import above_object_path, above_object_path_batch
from gr00t.rl.pick.grasp_candidates import cone_grasp_candidate_bank
from gr00t.rl.pick.planning import grasp_candidate_bank
from gr00t.rl.tests.test_pick_batched_candidate_bank import cases
from gr00t.rl.tests.test_pick_batched_grasp_inverse import fake_inverse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU cone compaction")
def test_cone_compaction_matches_allowed_original_contact_queries():
    state, target, sizes, hulls, mask = cases()
    q = target.new_tensor([[1, 0, 0, 0], [2**-0.5, 0, 2**-0.5, 0], [0, 0, 1, 0]]).expand(2, -1, -1)
    normal = target.new_tensor([0, 0, 1]).expand(2, -1)
    geometry = SimpleNamespace(supports_query_mask=True)
    geometry.select = lambda rows: geometry
    collision = SimpleNamespace(
        gripper_clearance=lambda g, p, q, **kw: p.new_full(p.shape[:-1], 0.02),
        gripper_closing_clearance=lambda g, p, q, *args, **kw: p.new_full(p.shape[:-1], 0.02),
    )
    packed = cone_grasp_candidate_bank(
        state,
        sizes,
        collision,
        geometry,
        hulls,
        mask,
        q,
        30.0,
        normal,
        collider_clearance_m=0.006,
        minimum_center_depth_m=0.005,
    )
    reference = grasp_candidate_bank(
        state,
        target,
        sizes,
        collision,
        geometry,
        hulls,
        candidate_orientations=q,
        object_frame_fallback=False,
        collider_clearance_m=0.006,
        retain_depth_candidates=True,
        minimum_center_depth_m=0.005,
        support_normal_world=normal,
        support_mask=mask,
    )
    depth = packed[0].shape[1] // 3
    allowed = torch.zeros(packed[0].shape[:2], device="cuda", dtype=torch.bool)
    allowed[:, depth : 2 * depth] = True
    torch.testing.assert_close(packed[0], reference[0])
    torch.testing.assert_close(packed[1][allowed], reference[1][allowed])
    for key in reference[2]:
        torch.testing.assert_close(
            packed[2][key][allowed], reference[2][key][allowed], equal_nan=True, msg=key
        )
    assert not packed[2]["pocket_fit"][~allowed].any()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_above_object_prefix_matches_single_and_ignores_nan_padding(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU basket prefixes")
    p = torch.full((2, 7, 3), torch.nan, device=device)
    p[0, :4] = p.new_tensor([[0, 0, 0.9], [0, 0, 0.8], [0, 0, 0.68], [0, 0, 0.6]])
    p[1, :3] = p.new_tensor([[1, 0, 0.9], [1, 0, 0.8], [1, 0, 0.7]])
    q = p.new_tensor([2**-0.5, 0, 2**-0.5, 0]).expand(2, 7, -1).clone()
    lengths = torch.tensor([4, 3], device=device)
    destination = p.new_tensor([[0, 0, 0.68], [1, 0, 0.78]])

    def forbidden(*args, **kwargs):
        raise AssertionError("Basket path trimming must remain device-side")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        path, selected = above_object_path_batch((p, q, lengths), destination)
    for row in range(2):
        n = int(lengths[row])
        single, orientation = above_object_path(
            (p[row : row + 1, :n], q[row : row + 1, :n], lengths[row : row + 1]),
            {"selected": 0, "inverse_candidate_indices": [0], "pregrasp_standoff_m": [0.08]},
        )
        stop = int(path[2][row])
        assert stop == int(single[2][0])
        torch.testing.assert_close(path[0][row, :stop], single[0][0])
        torch.testing.assert_close(path[1][row, :stop], single[1][0])
        torch.testing.assert_close(selected[row], orientation[0])
        torch.testing.assert_close(path[0][row, stop:], destination[row].expand(7 - stop, -1))
    reverse = torch.tensor([1, 0], device=device)
    reordered, _ = above_object_path_batch(
        (p[reverse], q[reverse], lengths[reverse]), destination[reverse]
    )
    for value, expected in zip(reordered, path):
        torch.testing.assert_close(value[reverse], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU basket grasp composition")
@pytest.mark.parametrize("final_entry", [False, True])
def test_native_basket_pocket_composition_preserves_mixed_hulls_and_split_results(
    monkeypatch, final_entry
):
    state, target, sizes, hulls, mask = cases()
    state["ee_pos"] = target + target.new_tensor([-0.2, 0, 0.2])
    state["dof_pos"] = target.new_zeros(2, 6)
    monkeypatch.setattr(grasp_inverse, "solve_pose_targets", fake_inverse)
    geometry = SimpleNamespace(supports_query_mask=True)
    geometry.select = lambda rows: geometry
    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6, device="cuda"),
        clearance=lambda g, r, j, **kw: r.new_full(r.shape[:-1], 0.02),
        gripper_clearance=lambda g, p, q, **kw: p.new_full(p.shape[:-1], 0.02),
        gripper_closing_clearance=lambda g, p, q, *args, **kw: p.new_full(p.shape[:-1], 0.02),
    )
    limits = target.new_tensor([[-3, 3]] * 6)
    entry = target.new_tensor([2**-0.5, 0, 2**-0.5, 0]).expand(2, -1)
    minimum = target.new_tensor([0.9, 1.1])

    def run(rows, with_history=False):
        return basket_grasp.plan_basket_grasp_batch(
            {key: value[rows] for key, value in state.items()},
            sizes[rows],
            limits,
            collision,
            geometry,
            0.02,
            support_points=hulls[rows],
            support_mask=mask[rows],
            initial_height=target[rows, 2],
            minimum_height=minimum[rows],
            prefer_realized_body=torch.zeros(len(rows), device="cuda", dtype=torch.bool),
            realized_body=torch.zeros(len(rows), device="cuda", dtype=torch.bool),
            entry_orientation=entry[rows] if final_entry else None,
            excluded=entry[rows, None] if with_history else None,
            excluded_mask=torch.ones(len(rows), 1, device="cuda", dtype=torch.bool)
            if with_history
            else None,
        )

    rows = torch.arange(2, device="cuda")
    # The real pocket geometry and path scorers are used, only the inverse is a fixture.
    combined = run(rows)
    if final_entry:
        assert (combined[2]["uncompressed_candidate_count"] == 17).all()
        reused = run(rows, with_history=True)
        counts_key = "uncompressed_geometry_pass_counts.pocket_fit"
        assert combined[2][counts_key].sum() > 0
        torch.testing.assert_close(reused[2][counts_key], combined[2][counts_key], rtol=0, atol=0)
        torch.testing.assert_close(reused[1], combined[1], rtol=0, atol=0)
        for actual, expected in zip(reused[0], combined[0], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for row in range(2):
        single = run(rows[row : row + 1])
        torch.testing.assert_close(combined[1][row : row + 1], single[1])
        source = combined[2]["candidate_source_indices"][row]
        single_source = single[2]["candidate_source_indices"][0]
        columns = torch.searchsorted(source, single_source)
        for key in ("candidate_cost", "feasible"):
            if key not in single[2]:
                assert not bool(single[1][0])
                continue
            torch.testing.assert_close(combined[2][key][row, columns], single[2][key][0])
        selected = combined[2]["selected"][row]
        single_selected = single[2]["selected"][0]
        if selected >= 0:
            torch.testing.assert_close(source[selected], single_source[single_selected])
        else:
            assert single_selected < 0
        torch.testing.assert_close(
            combined[2]["entry_orientation_world_wxyz"][row : row + 1],
            single[2]["entry_orientation_world_wxyz"],
        )
        n = int(combined[0][2][row])
        assert n == int(single[0][2][0])
        for i in (0, 1):
            torch.testing.assert_close(combined[0][i][row, :n], single[0][i][0, :n])

    def forbidden(*args, **kwargs):
        raise AssertionError("Basket grasp composition must not read tensors on the host")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        run(rows.flip(0))
