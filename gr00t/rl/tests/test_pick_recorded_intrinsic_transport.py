"""Audit transported geometry against fresh profiles at recorded physical object poses."""

import json
import os
import time
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    not os.environ.get("PICK_HINT_RECORD_ROOT"), reason="Recorded admission required"
)
def test_real_asset_transport_preserves_contact_acceptance(tmp_path, monkeypatch):
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.auto_open.tensor_metadata import to_host
    from gr00t.rl.pick import basket_intrinsic_proposals as module
    from gr00t.rl.pick.basket_initialization import BasketAdmissionInputs
    from gr00t.rl.pick.grasp_quality import convex_grasp_quality
    from gr00t.rl.pick.pocket import BASKET_MINIMUM_CENTER_DEPTH_M
    from gr00t.rl.pick.support_features import uniform_surface_support

    directory = Path(os.environ["PICK_HINT_RECORD_ROOT"])
    records = json.loads((directory / "prepared_scenes.json").read_text())
    episodes, snapshots = [], []
    for record in records:
        if "episode" not in record:
            continue
        episode = record["episode"]
        path = directory / f"seed_{episode['seed']}/initial_acceptance.json"
        snapshot = json.loads(path.read_text()).get("initial_acceptance_values")
        if snapshot is not None:
            episodes.append(episode)
            snapshots.append(snapshot)
    metadata = BasketAdmissionInputs(episodes, 4, "cuda")
    points, mask = metadata.support, metadata.support_mask
    if points.shape[1] > 9984:
        points, mask = uniform_surface_support(points, mask, count=9984)
    seeds = torch.tensor([e["seed"] for e in episodes], device="cuda", dtype=torch.int64)
    bank = module.IntrinsicGraspProposals(points, mask, metadata.sizes, seeds)
    state = {
        "root_states": points.new_tensor([s["root_state_world"] for s in snapshots]),
        "object_state": points.new_tensor([s["target_state_world"] for s in snapshots]),
    }
    monkeypatch.setattr(
        module,
        "grasp_environment_clearance",
        lambda c, g, p, q, a, **kw: p.new_full(p.shape[:-1], kw["minimum_clearance_m"] + 0.0001),
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    orientations, _, quality, width, _ = bank.candidate_bank(state, None, None)
    torch.cuda.synchronize()
    transport_s = time.perf_counter() - started
    normal = _quat_rotate(state["root_states"][:, 3:7], points.new_tensor([0, 0, 1]))
    active = quality["pocket_fit"].flatten()
    rejected = torch.zeros(len(seeds), device="cuda", dtype=torch.long)
    totals = torch.zeros_like(rejected)
    error = points.new_zeros(len(seeds))
    margin_error = error.clone()
    capacity = 4_000_000 // points.shape[1]
    started = time.perf_counter()
    for start in range(0, len(seeds) * width, capacity):
        stop = min(start + capacity, len(seeds) * width)
        source = torch.arange(start, stop, device="cuda") // width
        oracle = convex_grasp_quality(
            orientations.flatten(0, 1)[start:stop, None],
            state["object_state"][source, 3:7],
            points[source],
            support_mask=mask[source],
            depth_fractions=bank.fractions.flatten()[start:stop, None, None],
            minimum_center_depth_m=BASKET_MINIMUM_CENTER_DEPTH_M,
            support_normal_world=normal[source],
        )
        valid = active[start:stop]
        fresh = oracle["pocket_fit"][:, 0] & (oracle["antipodal_margin"][:, 0] >= 0)
        fresh &= oracle["support_pinch_compatible"][:, 0]
        rejected.index_add_(0, source, (valid & ~fresh).long())
        totals.index_add_(0, source, valid.long())
        difference = (
            oracle["center_local"][:, 0] - bank.qualities["center_local"].flatten(0, 1)[start:stop]
        ).norm(dim=-1)
        error.scatter_reduce_(0, source, difference.masked_fill(~valid, 0), reduce="amax")
        difference = (
            oracle["antipodal_margin"][:, 0]
            - bank.qualities["antipodal_margin"].flatten()[start:stop]
        ).abs()
        margin_error.scatter_reduce_(0, source, difference.masked_fill(~valid, 0), reduce="amax")
    torch.cuda.synchronize()
    report = {
        "physics_simulated": False,
        "environment_obstacles_included": False,
        "rows": len(seeds),
        "contacts_per_object": width,
        "transport_s": transport_s,
        "recomputation_s": time.perf_counter() - started,
        "values": to_host(
            {
                "seeds": seeds,
                "eligible_counts": totals,
                "false_accept_counts": rejected,
                "maximum_center_error_m": error,
                "maximum_antipodal_error": margin_error,
            }
        ),
    }
    (tmp_path / "intrinsic_transport_audit.json").write_text(json.dumps(report, indent=2))
    assert not rejected.any(), report
    assert error.amax() < 1e-5, report
