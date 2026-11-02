import math
from types import SimpleNamespace

import torch
import pytest

from gr00t.rl.auto_open.panel_push_forecast import select_panel_arc


def _pull_bank(batch=1, translate=0.0, far=False, mixed=False, **options):
    from gr00t.rl.auto_open.panel_push_forecast import panel_candidate_bank

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64).expand(batch, -1).clone()

    root = tensor([[0.8, 0.4, 0.5, 1.0, 0.0, 0.0, 0.0]])
    if far:
        root[:, 0] += 3.0
    if mixed:
        root[1:, 0] += 1.5
    root[:, :2] += translate
    desired = tensor([[0.4, 0.45, 1.0]])
    desired[:, :2] += translate
    hinge = tensor([[0.0, 0.0, 0.0]])
    hinge[:, :2] += translate
    scalar = lambda x: torch.full((batch,), x, dtype=torch.float64)
    return panel_candidate_bank(
        desired,
        tensor([[0.0, 0.0, 0.0, 1.0]]),
        tensor([[1.0, 0.0, 0.0]]),
        tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]], dtype=torch.float64).expand(
            batch, -1, -1
        ),
        tensor([[1.0, 0.0, 0.0]]),
        hinge,
        root,
        tensor([[0.2, 0.0, 0.7]]),
        scalar(0.03),
        scalar(1.0),
        scalar(0.0),
        scalar(2.0),
        scalar(0.05),
        scalar(0.7),
        tensor([[1.3, 1.4]]),
        scalar(1.0),
        **options,
    )


def test_push_height_extension_preserves_original_uniform_layers():
    old = _pull_bank()
    extended = _pull_bank(height_upper_offset_m=0.30)
    assert extended["targets"].shape == (1, 770, 3)
    levels = torch.unique(extended["parameters"][..., 1])
    torch.testing.assert_close(levels, torch.linspace(-0.2, 0.3, 11, dtype=torch.float64))
    old_parameters = torch.round(old["parameters"][0] * 1.0e6).long()
    new_parameters = torch.round(extended["parameters"][0] * 1.0e6).long()
    matches = (old_parameters[:, None] == new_parameters[None]).all(-1)
    assert matches.any(-1).all()
    for name in ("targets", "quats", "geometry_valid"):
        torch.testing.assert_close(old[name][0], extended[name][0, matches.long().argmax(-1)])


def test_pull_bank_is_a_bounded_region_grid():
    bank = _pull_bank()
    assert bank["targets"].shape == (1, 630, 3)
    assert bank["geometry_valid"].any()
    assert bank["parameters"][..., 2].unique().tolist() == [
        -30.0,
        -20.0,
        -10.0,
        0.0,
        10.0,
        20.0,
        30.0,
    ]
    selected = bank["geometry_valid"][0]
    point = bank["targets"][0, selected]
    assert (
        torch.linalg.vector_norm(point[:, :2] - torch.tensor([1.0, 0.4]), dim=-1) <= 0.7 + 1.0e-10
    ).all()
    radius = bank["tip_surface"][0, selected, 1]
    assert (radius >= 0.20 - 1.0e-10).all() and (radius <= 0.95 + 1.0e-10).all()


def test_pull_region_grid_is_batch_and_origin_independent():
    single, batch, shifted = _pull_bank(), _pull_bank(3, mixed=True), _pull_bank(translate=128.0)
    for key in ("targets", "quats", "tip_offset", "tip_surface", "geometry_valid", "parameters"):
        torch.testing.assert_close(single[key][0], batch[key][0])
    for key in ("targets", "tip_surface"):
        expected = single[key].clone()
        expected[..., :2] += 128.0
        torch.testing.assert_close(expected, shifted[key])
    torch.testing.assert_close(single["geometry_valid"], shifted["geometry_valid"])


def test_pull_region_reports_empty_workspace_intersection():
    bank = _pull_bank(far=True)
    assert not bank["geometry_valid"].any()
    assert torch.isfinite(bank["targets"]).all()


def test_panel_search_is_small_and_yaw_bounded():
    bank = _pull_bank()
    assert bank["targets"].shape[1] == 630
    assert bank["parameters"][..., 2].abs().max() <= 30.0


def test_push_prefers_longest_arc_and_allows_existing_reposition_cycle():
    valid = torch.ones(2, 3, 12, dtype=torch.bool)
    lengths = torch.tensor([[2, 5, 7], [2, 2, 2]])
    angles = torch.deg2rad(torch.arange(11.0) + 8.0)[None, None].expand(2, 3, -1)
    selected, accepted, _, _ = select_panel_arc(
        valid,
        lengths,
        angles,
        torch.ones(2, 3, dtype=torch.bool),
        minimum_progress_rad=0.0,
        prefer_longest=True,
    )
    assert selected.tolist() == [2, 0]
    assert accepted.tolist() == [True, True]
    _, pull_accepted, _, _ = select_panel_arc(
        valid[1:], lengths[1:], angles[1:], torch.ones(1, 3, dtype=torch.bool)
    )
    assert not pull_accepted.any()


def test_push_does_not_waive_geometry_or_inverse_failure():
    angles = torch.deg2rad(torch.arange(4.0) + 8.0)[None, None]
    for geometry, valid in ((False, True), (True, False)):
        _, accepted, _, _ = select_panel_arc(
            torch.full((1, 1, 5), valid),
            torch.tensor([[2]]),
            angles,
            torch.tensor([[geometry]]),
            minimum_progress_rad=0.0,
            prefer_longest=True,
        )
        assert not accepted.any()


def test_future_joint_limit_rejects_point_only_candidate():
    valid = torch.ones(1, 2, 32, dtype=torch.bool)
    valid[0, 0, 8:] = False
    lengths = torch.full((1, 2), 32)
    angles = torch.deg2rad(torch.arange(31.0) + 40)[None, None].expand(1, 2, -1)
    selected, accepted, stop, complete = select_panel_arc(
        valid, lengths, angles, torch.ones(1, 2, dtype=torch.bool)
    )
    assert selected.item() == 1
    assert accepted.item() and complete.item()
    assert math.isclose(stop.item(), angles[0, 1, -1].item(), abs_tol=1.0e-6)


def test_joint_safe_prefix_retains_lead_margin_and_requires_progress():
    valid = torch.ones(2, 1, 32, dtype=torch.bool)
    valid[0, 0, 20:] = False
    valid[1, 0, 4:] = False
    angles = torch.deg2rad(torch.arange(31.0) + 40)[None, None].expand(2, 1, -1)
    _, accepted, stop, complete = select_panel_arc(
        valid, torch.full((2, 1), 32), angles, torch.ones(2, 1, dtype=torch.bool)
    )
    assert accepted.tolist() == [True, False]
    assert not complete.any()
    assert math.isclose(stop[0].item(), math.radians(54.0), abs_tol=1.0e-6)


def test_padding_does_not_reject_valid_workspace_exit():
    valid = torch.ones(1, 1, 32, dtype=torch.bool)
    valid[..., 15:] = False
    angles = torch.deg2rad(torch.arange(31.0) + 40)[None, None]
    _, accepted, _, complete = select_panel_arc(
        valid, torch.tensor([[15]]), angles, torch.ones(1, 1, dtype=torch.bool)
    )
    assert accepted.item() and complete.item()


@pytest.mark.parametrize("radius", [0.36, 0.5])
def test_forecast_batches_candidates_and_keeps_original_when_full_arc_is_safe(radius):
    from gr00t.rl.auto_open.tensor_stage_planning import GPUStagePlanningMixin

    class Coordinator(GPUStagePlanningMixin):
        def _stage_candidate_ik(self, fsm, state, rows, positions, orientations):
            count = len(rows)
            return (
                torch.ones(count, dtype=torch.bool),
                torch.zeros(count),
                torch.zeros(count),
                torch.ones(count),
            )

        def _inverse_pose_bank(self, fsm, state, rows, positions, orientations, lengths):
            assert len(rows) == 770
            owner, sample = torch.where(torch.arange(positions.shape[1])[None] < lengths[:, None])
            count = len(owner)
            return (
                SimpleNamespace(
                    position_error=torch.zeros(count, 3),
                    orientation_error=torch.zeros(count, 3),
                    arm_joint_positions=torch.zeros(count, 6),
                ),
                owner,
                sample,
            )

    coordinator = Coordinator()
    coordinator.joint_limits = torch.tensor([[-3.0, 3.0]] * 6)
    coordinator.path_diagnostics = []
    coordinator.gpu_stage_constants = {
        "panel_push_preload": torch.tensor([0.004]),
        "panel_standoff_target_clearance": torch.tensor([0.03]),
        "panel_contact_edge_margin": torch.tensor([0.05]),
        "panel_hinge_to_free_edge": torch.tensor([1.0]),
        "panel_bottom_z": torch.tensor([0.0]),
        "panel_top_z": torch.tensor([2.0]),
    }
    failed = []
    fsm = SimpleNamespace(
        device=torch.device("cpu"),
        hinge_axis_sign=torch.tensor([1.0]),
        depth_axis_local=torch.tensor([[1.0, 0.0, 0.0]]),
        hinge_world=torch.zeros(1, 3),
        hinge_idx=0,
        push_comfort_reach_radius=torch.tensor([0.7]),
        push_planned_stop_hinge=torch.zeros(1),
        panel_push_tip_offset_local=torch.zeros(1, 3),
        panel_push_tip_surface_world=torch.zeros(1, 3),
        panel_push_tip_hinge_angle=torch.zeros(1),
        env=SimpleNamespace(get_goal_center_offset_local=lambda: torch.zeros(1, 3)),
        fail=lambda rows, code: failed.extend(rows.tolist()),
    )
    from gr00t.rl.tests.test_auto_open_doorway_clearance import install_doorway
    install_doorway(fsm, 1)
    theta = math.radians(40.0)
    normal = torch.tensor([[math.cos(theta), math.sin(theta), 0.0]])
    tangent = torch.tensor([[-math.sin(theta), math.cos(theta), 0.0]])
    desired = radius * tangent + 0.034 * normal
    desired[:, 2] = 1.0
    orientation = torch.tensor(
        [[math.cos((theta + math.pi) / 2), 0.0, 0.0, math.sin((theta + math.pi) / 2)]]
    )
    root = torch.tensor([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    rows, actual, _, _ = coordinator._forecast_panel_push(
        fsm,
        {"root_states": root, "door_joint_position": torch.tensor([[theta]])},
        torch.tensor([0]),
        desired,
        orientation,
        normal,
        tangent,
        torch.zeros(1, 1, 3),
        torch.tensor([[1.4, 1.6]]),
    )
    assert rows.tolist() == [0]
    assert not failed
    assert torch.allclose(actual, desired, atol=1.0e-6)
    assert torch.allclose(fsm.push_planned_stop_hinge, fsm.through_goal_hinge, atol=1.0e-6)
    assert coordinator.path_diagnostics[-1][2]["candidate_geometry_valid"].any()
