"""Held-object lift candidates pivot about the payload without axial twisting."""

from types import SimpleNamespace

import torch

from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.basket_carry import held_geometry, held_pose_bank


def test_clearance_cap_preserves_all_lift_acceptance_predicates():
    from gr00t.rl.pick.basket_carry import HELD_CLEARANCE_QUERY_CAP_M, held_geometry_acceptance

    generator = torch.Generator().manual_seed(1702)
    jaw = torch.rand(128, 17, generator=generator) * 0.01 - 0.003
    payload = torch.rand(128, 17, generator=generator) * 0.01 - 0.003
    jaw[:4] = torch.tensor([0.0001, 0.001, 0.002, 0.003])[:, None]
    payload[:4] = 0.002
    jaw[4] = torch.linspace(0.0001, 0.01, 17)
    payload[4] = 0.002
    checks = (torch.ones_like(jaw, dtype=torch.bool), jaw, payload, jaw * 0, jaw * 0, jaw * 0)
    capped = (
        checks[0],
        jaw.clamp_max(HELD_CLEARANCE_QUERY_CAP_M),
        payload.clamp_max(HELD_CLEARANCE_QUERY_CAP_M),
        *checks[3:],
    )
    actual, metrics = held_geometry_acceptance(capped)
    expected, full_metrics = held_geometry_acceptance(checks)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key in (
        "bounds_safe",
        "gripper_positive_clearance",
        "gripper_release_monotone",
        "outside_footprint_below_rim",
    ):
        torch.testing.assert_close(metrics[key], full_metrics[key], rtol=0, atol=0)


def test_held_geometry_chunking_preserves_all_surface_points_and_candidate_order(monkeypatch):
    from gr00t.rl.pick import basket_carry

    root = torch.tensor([[0.0, 0, 0, 1, 0, 0, 0]])
    state = {
        "root_states": root,
        "ee_orn": root[:, 3:7],
        "object_state": root.clone(),
    }
    positions = torch.zeros(7, 4, 3)
    positions[..., 2] = torch.arange(7)[:, None] * 0.01
    quats = root[:, None, 3:7].expand(7, 4, 4)
    support = torch.tensor([[0.01, 0, 0], [-0.01, 0, 0], [0, 0.01, 0]])
    calls = []

    def clearance(p, q, points, **options):
        assert points is support
        assert options["maximum_distance_m"] == 0.0021
        calls.append(len(p))
        return p[..., 2] + 0.01

    geometry = SimpleNamespace(nonrobot_geometry=SimpleNamespace(rigid_surface_clearance=clearance))
    collision = SimpleNamespace(gripper_pose_clearance=lambda g, p, q, a, **kw: p[..., 2] + 0.01)
    basket = SimpleNamespace(
        floor_lower=torch.full((3,), -1.0), floor_upper=torch.ones(3), rim_z=0.2
    )
    args = (
        state,
        positions,
        quats,
        torch.zeros(3),
        support,
        basket,
        collision,
        geometry,
        torch.zeros(1),
    )
    expected = held_geometry(*args)
    monkeypatch.setattr(basket_carry, "MAX_HELD_SAMPLE_EVALUATIONS", 24)
    calls.clear()
    actual = held_geometry(*args)
    assert calls == [2, 2, 2, 1]
    for before, after in zip(expected, actual):
        torch.testing.assert_close(before, after)


def test_held_lift_cone_preserves_payload_pivot_and_covers_transverse_axes():
    root = torch.tensor([[0.0, 0, 0.52, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    state = {
        "root_states": root,
        "ee_pos": torch.tensor([[-0.2, 0.0, 0.75]]),
        "ee_orn": torch.tensor([[1.0, 0, 0, 0]]),
        "object_state": torch.tensor([[-0.1, 0.0, 0.67, 1, 0, 0, 0]]),
    }
    basket = SimpleNamespace(world_mesh=lambda _: (torch.tensor([[0.0, 0, 0.73]]), None))
    support = torch.tensor([[0.02, 0.02, 0.02], [-0.02, -0.02, -0.02]])
    centers, quats, relative, bank = held_pose_bank(state, basket, support, torch.tensor([0.87]))
    assert bank.shape == (260, 3)
    assert (centers[..., 2].diff(dim=1) >= 0).all()
    torch.testing.assert_close(quats.norm(dim=-1), torch.ones(260, 3))
    assert torch.count_nonzero(quats[..., 1]) == 0
    assert quats[..., 2].abs().max() > 0.15
    assert quats[..., 3].abs().max() > 0.15
    ee = centers - _quat_rotate(quats, relative)
    torch.testing.assert_close(ee[:, 0], state["ee_pos"].expand(260, -1))
    torch.testing.assert_close(ee + _quat_rotate(quats, relative), centers)
    no_retract = bank[:, 2] == 0
    torch.testing.assert_close(
        centers[no_retract, -1, :2], state["object_state"][:, :2].expand(65, -1)
    )
    torch.testing.assert_close(centers[:, -1, 2], torch.full((260,), 0.875))
    old = held_pose_bank(
        state, basket, support, torch.tensor([0.87]), tilt_angles_deg=(5, 10, 15, 20)
    )
    torch.testing.assert_close(bank[:132], old[3])
    torch.testing.assert_close(centers[:132], old[0])
    torch.testing.assert_close(quats[:132], old[1])
