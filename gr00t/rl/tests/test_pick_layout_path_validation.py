"""Layout certification keeps ownership and bounds searches before physical resets."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick import basket_layout_validation as validation


def test_layout_contact_cache_builds_once_and_selects_actual_source_rows(monkeypatch):
    from gr00t.rl.pick import basket_intrinsic_proposals as intrinsic

    checker = validation.BasketLayoutPathValidator.__new__(validation.BasketLayoutPathValidator)
    checker.count = 2
    checker.inputs = SimpleNamespace(support=object(), support_mask=object(), sizes=object())
    checker.proposal_seeds = torch.tensor([7, 9])
    checker.intrinsic_proposals = None
    builds, selections = [], []
    bank = SimpleNamespace(select=lambda rows: selections.append(rows.clone()) or "view")

    def build(support, mask, sizes, seeds):
        assert support is checker.inputs.support and mask is checker.inputs.support_mask
        assert sizes is checker.inputs.sizes and seeds is checker.proposal_seeds
        builds.append(True)
        return bank

    monkeypatch.setattr(intrinsic, "IntrinsicGraspProposals", build)
    assert checker.proposal_bank() is bank
    assert checker.proposal_bank(torch.tensor([1, 0])) == "view"
    assert checker.proposal_bank(torch.tensor([0])) == "view"
    assert builds == [True]
    assert [rows.tolist() for rows in selections] == [[1, 0], [0]]


def test_materialized_preflight_preserves_target_identity_and_unselected_rows(monkeypatch):
    from gr00t.rl.pick import basket_layout as sampler
    from gr00t.rl.pick import basket_layout_reachability as reach

    def item(name):
        return {"asset": {"name": name}}

    a, b, c = item("a"), item("b"), item("c")
    d, e = item("d"), item("e")
    diagnostics = {"object_gap_m": 0.01, "wall_gap_m": 0.02}
    original = [([a, b, c], 2, diagnostics), RuntimeError("no placement"), ([d, e], 1, diagnostics)]
    joints = torch.arange(18).reshape(3, 6).float()
    invoked = []

    class Validator:
        def __init__(
            self, assets, targets, basket, local_joints, limits, collision, *, proposal_seeds
        ):
            assert targets == [2, 1]
            assert [[asset["name"] for asset in row] for row in assets] == [
                ["a", "b", "c"],
                ["d", "e"],
            ]
            torch.testing.assert_close(local_joints, joints[[0, 2]])
            torch.testing.assert_close(proposal_seeds, torch.tensor([1700, 1702]))

        def proposal_bank(self, rows=None):
            assert rows is None
            return None

        def __call__(self, data, preferred, report, rows, local_joints):
            torch.testing.assert_close(rows, torch.arange(2))
            torch.testing.assert_close(local_joints, joints[[0, 2]])
            invoked.append(True)
            return preferred, report

    monkeypatch.setattr(validation, "BasketLayoutPathValidator", Validator)
    monkeypatch.setattr(
        reach, "screen_layouts", lambda *args, **kwargs: (torch.ones(2, 4, dtype=torch.bool), {})
    )

    def sample(assets, basket, seeds, config, **kwargs):
        assert seeds == [1700, 1702] and kwargs["target_asset_indices"] == [2, 1]
        kwargs["proposal_filter"]({})
        return [
            ([c, a, b], 0, {"reachability_fallback": False}),
            ([e, d], 0, {"reachability_fallback": True}),
        ]

    monkeypatch.setattr(sampler, "sample_layouts", sample)
    result = reach.prefer_materialized_layouts(
        original, None, [1700, 1701, 1702], joints, None, None, None
    )
    assert invoked == [True]
    assert result[0][0][result[0][1]]["asset"]["name"] == "c"
    assert result[1] is original[1] and result[2][0] is original[2][0]
    assert result[2][1] == original[2][1]
    assert result[2][2]["layout_proposal_diagnostics"] == {"reachability_fallback": True}
    assert "layout_proposal_diagnostics" not in original[2][2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU BVH ownership")
def test_nominal_view_reuses_meshes_without_mutating_actual_scene_frames():
    from gr00t.rl.pick.basket_geometry import BasketGeometryBatch
    from gr00t.rl.pick.scene_geometry import SceneGeometry

    p = torch.tensor([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], device="cuda", dtype=torch.float32)
    mesh = SceneGeometry(p, torch.tensor([[0, 1, 2]], device="cuda"))
    model = SimpleNamespace(geometry=mesh)
    episodes = [
        {"asset": {"usd_path": "target"}, "distractors": [{"asset": {"usd_path": "prop"}}]}
        for _ in range(2)
    ]
    factory = BasketGeometryBatch(
        [SimpleNamespace(static_scene=mesh, basket=model, episode=episode) for episode in episodes],
        mesh_loader=lambda path, device: mesh,
    )
    roots = p.new_zeros(2, 13)
    roots[:, 3], roots[:, 2] = 1, 0.52
    roots[:, :2] = p.new_tensor([[4, 8], [12, 16]])
    state = {
        "root_states": roots,
        "object_state": roots.clone(),
        "distractor_states": roots[:, None].clone(),
    }
    factory.update(state)
    factory.with_target(state)
    owned_views = [factory.basket, factory.target, *factory.props]
    before = [(view.origins.clone(), view.orientations.clone()) for view in owned_views]
    source_collision = SimpleNamespace(
        arm_joint_ids=torch.tensor([8, 13, 14, 15, 16, 17], device="cuda")
    )
    checker = validation.BasketLayoutPathValidator(
        [[e["asset"], e["distractors"][0]["asset"]] for e in episodes],
        [0, 0],
        model,
        p.new_zeros(2, 6),
        p.new_zeros(6, 2),
        source_collision,
        geometry_factory=factory,
        admission_inputs=object(),
    )
    nominal = {
        "root_states": checker.roots,
        "object_state": checker.roots.clone(),
        "distractor_states": checker.roots[:, None].clone(),
    }
    checker.factory.update(nominal)
    for view, (position, orientation) in zip(owned_views, before, strict=True):
        torch.testing.assert_close(view.origins, position, rtol=0, atol=0)
        torch.testing.assert_close(view.orientations, orientation, rtol=0, atol=0)
    assert checker.factory.target is None and factory.target is not None
    torch.testing.assert_close(
        source_collision.arm_joint_ids, torch.tensor([8, 13, 14, 15, 16, 17], device="cuda")
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("coarse_available", [True, False, "single"])
def test_layout_shortlist_keeps_source_rows_and_checks_at_most_four(
    monkeypatch, device, coarse_available
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU validation")
    checker = validation.BasketLayoutPathValidator.__new__(validation.BasketLayoutPathValidator)
    checker.device, checker.count = torch.device(device), 3
    checker.dt = 0.02
    checker.maximum_layouts = 4
    checker.joints = torch.zeros(3, 6, device=device)
    checker.roots = torch.zeros(3, 13, device=device)
    checker.roots[:, 2], checker.roots[:, 3] = 0.55, 1
    checker.order = torch.tensor([[0, 1], [0, 1], [1, 0]], device=device)
    checker.inputs = SimpleNamespace(
        sizes=torch.ones(3, 3, device=device),
        payload_sizes=torch.ones(3, 2, 3, device=device),
        hulls=(None, None, torch.ones(3, 2, device=device, dtype=torch.bool)),
        support=torch.zeros(3, 1, 3, device=device),
        support_mask=torch.ones(3, 1, device=device, dtype=torch.bool),
        centers=torch.zeros(3, 3, device=device),
        lift_margin=torch.full((3,), 0.1, device=device),
    )
    checker.basket = checker.collision = checker.limits = None
    calls = []

    class View:
        def __init__(self, rows):
            self.rows = rows

        def select(self, rows):
            return View(self.rows[rows])

    checker.factory = SimpleNamespace(update=lambda state: View(torch.arange(3, device=device)))
    monkeypatch.setattr(
        validation,
        "b2z1_gripper_pose_and_jacobian",
        lambda p, q, j: (p, torch.eye(3, device=device).expand(3, -1, -1), None),
    )
    monkeypatch.setattr(
        validation, "basket_required_height", lambda *args, **kwargs: checker.roots.new_tensor(0.9)
    )

    def access(state, *args, **kwargs):
        if "candidate_orientations" in kwargs:
            torch.testing.assert_close(kwargs["candidate_orientations"], state["ee_orn"][:, None])
        torch.testing.assert_close(
            kwargs["body_heights"],
            state["root_states"][:, 2:3] + state["root_states"].new_tensor([-0.01, 0, 0.01]),
        )
        calls.append((args[3].rows.clone(), state["object_state"].clone()))
        return (
            None,
            state["object_state"][:, 0] == 42,
            {"entry_orientation_world_wxyz": state["ee_orn"]},
        )

    monkeypatch.setattr(validation, "plan_basket_grasp_batch", access)
    preferred = torch.full((2, 6), coarse_available is True, device=device, dtype=torch.bool)
    if coarse_available == "single":
        preferred[:, 0] = True
    xy = torch.zeros(2, 6, 2, 2, device=device)
    xy[0, :, 0, 0] = torch.arange(6, device=device) + 40
    xy[0, :, 1, 0] = 80
    xy[1, :, 0, 0] = 100
    data = {
        "valid": torch.ones_like(preferred),
        "chosen": torch.tensor([[1, 0], [0, 1]], device=device),
        "mask": torch.ones(2, 2, device=device, dtype=torch.bool),
        "xy": xy,
        "height": torch.full((2, 6, 2), 0.1, device=device),
        "sizes": torch.full((2, 2, 3), 0.04, device=device),
        "yaw": torch.zeros(2, 6, 2, device=device),
    }
    orientations = xy.new_tensor([1, 0, 0, 0]).expand(2, 6, 4).clone()
    rows = torch.tensor([2, 0], device=device)
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, lambda *a, **k: pytest.fail("Planning host read"))
        certified, report = checker(
            data, preferred, {"orientation_local_wxyz": orientations}, rows, checker.joints[rows]
        )
    assert certified.sum() == 1 and certified[0, 2]
    assert report["full_path_layouts_evaluated"].tolist() == [3, 4]
    assert len(calls) == {True: 8, False: 4, "single": 5}[coarse_available]
    assert calls[0][0].tolist() == [2, 0] and calls[-1][0].tolist() == [0]
    torch.testing.assert_close(calls[0][1][:, 0], xy.new_tensor([40, 100]))
    torch.testing.assert_close(calls[0][1][:, 2], xy.new_tensor([0.668, 0.668]))
