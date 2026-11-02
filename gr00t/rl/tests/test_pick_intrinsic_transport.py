"""Transport cached contact geometry, but never reuse world obstacle acceptance."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA contact cache")
def test_transported_contacts_match_recomputation_and_keep_world_queries(monkeypatch):
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.pick import basket_intrinsic_proposals as module
    from gr00t.rl.pick.grasp_quality import convex_grasp_quality
    from gr00t.rl.pick.pocket import BASKET_MINIMUM_CENTER_DEPTH_M

    signs = torch.cartesian_prod(*[torch.tensor([-1.0, 1.0], device="cuda")] * 3)
    sizes = signs.new_tensor([[0.04, 0.04, 0.04], [0.05, 0.05, 0.05]])
    support = signs[None] * sizes[:, None] / 2
    mask = torch.ones(2, 8, device="cuda", dtype=torch.bool)
    bank = module.IntrinsicGraspProposals(
        support,
        mask,
        sizes,
        torch.tensor([1700, 1702], device="cuda"),
        count=64,
    )
    root = signs.new_zeros(2, 13)
    root[:, 3] = 1
    objects = root.clone()
    objects[:, :3] = signs.new_tensor([[-0.1, 0.02, 0.8], [0.1, -0.04, 0.7]])
    axis = signs.new_tensor([[0, 0, 1], [0.2, -0.3, 1]])
    axis /= axis.norm(dim=-1, keepdim=True)
    half = signs.new_tensor([0.15, -0.2])
    objects[:, 3:7] = torch.cat((half.cos()[:, None], axis * half.sin()[:, None]), -1)
    state = {"root_states": root, "object_state": objects}
    queried = []

    def obstacles(collision, geometry, p, q, aperture, **kwargs):
        queried.append((p.clone(), kwargs["query_mask"].clone()))
        return torch.where(p[..., 0] < 0, 0.0, kwargs["minimum_clearance_m"] + 0.0001)

    monkeypatch.setattr(module, "grasp_environment_clearance", obstacles)
    orientations, goals, quality, width, fallback = bank.candidate_bank(state, None, None)
    assert width == 64 and fallback == 0 and len(queried) == 1
    oracle = convex_grasp_quality(
        orientations,
        objects[:, 3:7],
        support,
        support_mask=mask,
        depth_fractions=bank.fractions[..., None],
        minimum_center_depth_m=BASKET_MINIMUM_CENTER_DEPTH_M,
        support_normal_world=root.new_tensor([0, 0, 1]).expand(2, -1),
    )
    for name in (
        "center_local",
        "aperture_angle_rad",
        "fixed_contact_local",
        "moving_contact_local",
        "antipodal_margin",
        "center_depth_margin_m",
    ):
        torch.testing.assert_close(
            quality[name][bank.valid], oracle[name][bank.valid], rtol=0, atol=1e-5, equal_nan=True
        )
    center = oracle["center_local"].clone()
    center[..., 2] -= 0.0005
    expected = objects[:, None, :3] - _quat_rotate(orientations, center)
    torch.testing.assert_close(goals[bank.valid], expected[bank.valid], rtol=0, atol=1e-6)
    assert quality["pocket_fit"].any()
    assert (quality["pocket_fit"] <= oracle["pocket_fit"]).all()
    torch.testing.assert_close(
        quality["support_pinch_compatible"], oracle["support_pinch_compatible"]
    )

    before = {name: value.clone() for name, value in bank.qualities.items()}
    shifted = {"root_states": root, "object_state": objects.clone()}
    shifted["object_state"][:, 0] += 1
    moved = bank.candidate_bank(shifted, None, None)
    assert len(queried) == 2
    assert not torch.equal(moved[2]["environment_clearance_m"], quality["environment_clearance_m"])
    for name, value in before.items():
        torch.testing.assert_close(bank.qualities[name], value, rtol=0, atol=0, equal_nan=True)
    rows = torch.tensor([1, 0, 1], device="cuda")
    reordered = bank.select(rows).candidate_bank(
        {key: value[rows] for key, value in state.items()}, None, None
    )
    torch.testing.assert_close(reordered[0], orientations[rows], rtol=0, atol=0)
    torch.testing.assert_close(reordered[1], goals[rows], rtol=0, atol=0)
    torch.testing.assert_close(
        reordered[2]["pocket_fit"], quality["pocket_fit"][rows], rtol=0, atol=0
    )
