"""Closed-jaw repositioning must improve space and retain speed-bounded paths."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.pick.basket_nudge import (
    bbox_space,
    improving_shift_mask,
    monotone_release_mask,
    object_boxes_in_trunk,
    pose_segments,
    slide_stability,
)


@pytest.mark.parametrize("yaw", [0.0, 0.7, -1.8])
def test_sliding_screen_rejects_high_contact_tipping_and_is_world_yaw_invariant(yaw):
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate

    root = torch.zeros(13)
    root[:3] = torch.tensor([3.0, 5.0, 0.5])
    root[3:7] = _euler_xyz_quat(torch.tensor([[0.0, 0.0, yaw]]))[0]
    obj = root.clone()
    obj[:3] += _quat_rotate(root[3:7], torch.tensor([0.0, 0.0, 0.03]))
    support = torch.cartesian_prod(
        torch.tensor([-0.03, 0.03]), torch.tensor([-0.02, 0.02]), torch.tensor([-0.03, 0.03])
    )
    hits = root[:3] + _quat_rotate(
        root[3:7], torch.tensor([[-0.03, 0.0, 0.015], [-0.03, 0.0, 0.055]])
    )
    normals = _quat_rotate(root[3:7], torch.tensor([[-1.0, 0.0, 0.0]]).expand(2, -1))
    safe, reserve, radius, height = slide_stability(
        root, obj, support, torch.zeros(3), hits, normals, -normals, 0.8
    )
    assert safe.tolist() == [True, False]
    torch.testing.assert_close(radius, torch.tensor([0.03, 0.03]), atol=1e-6, rtol=0)
    torch.testing.assert_close(height, torch.tensor([0.015, 0.055]), atol=1e-6, rtol=0)
    torch.testing.assert_close(reserve, torch.tensor([0.018, -0.014]), atol=1e-6, rtol=0)


def test_shift_can_escape_existing_box_overlap_without_deepening_it():
    before = torch.tensor(-0.00364)
    after = torch.tensor([0.010, 0.010, 0.003, -0.001])
    swept = torch.tensor([-0.00364, -0.004, -0.00364, -0.00364])
    assert improving_shift_mask(before, after, swept).tolist() == [True, False, False, False]
    assert not improving_shift_mask(torch.tensor(0.003), torch.tensor(0.007), torch.tensor(0.003))
    assert not improving_shift_mask(torch.tensor(0.003), torch.tensor(0.010), torch.tensor(-0.002))


def test_nudge_retreat_returns_to_the_same_hover_pose(monkeypatch):
    from gr00t.rl.pick import basket_nudge

    root = torch.zeros(1, 13)
    root[:, 3] = 1
    state = {
        "root_states": root,
        "object_state": root.clone(),
        "ee_pos": torch.tensor([[0.0, 0.0, 0.4]]),
        "ee_orn": root[:, 3:7],
    }
    basket = SimpleNamespace(
        floor_lower=torch.full((3,), -0.2),
        floor_upper=torch.full((3,), 0.2),
        world_mesh=lambda root: (torch.tensor([[0.0, 0.0, 0.2]]), None),
    )
    monkeypatch.setattr(
        basket_nudge,
        "object_boxes_in_trunk",
        lambda *args: (torch.tensor([[-0.17, 0, 0]]), torch.full((1, 3), 0.02)),
    )

    def counts(state, episode, targets, *args, **kwargs):
        result = torch.ones(len(targets), dtype=torch.long)
        result[0] = 0
        return result

    monkeypatch.setattr(basket_nudge, "shifted_grasp_counts", counts)
    monkeypatch.setattr(
        basket_nudge,
        "build_distractor_mesh",
        lambda *args: {"points": torch.zeros(3, 3), "faces": torch.zeros(1, 3, dtype=torch.long)},
    )
    monkeypatch.setattr(
        basket_nudge,
        "SceneGeometry",
        lambda *args, **kwargs: SimpleNamespace(
            rays=lambda origin, target: ((origin + target) / 2, torch.ones_like(origin), None)
        ),
    )
    collision = SimpleNamespace(
        _gripper_local_surfaces=lambda ee: (
            torch.tensor([[0.02, 0.0, 0.01]]),
            torch.tensor([[0.02, 0.0, -0.01]]),
        )
    )

    class CapturedPath(Exception):
        pass

    def capture(endpoints, rotations, *args, **kwargs):
        torch.testing.assert_close(endpoints[:, -1], endpoints[:, 1])
        torch.testing.assert_close(rotations[:, -1], rotations[:, 1])
        assert bool((endpoints[:, -1, 2] > endpoints[:, -2, 2]).all())
        assert bool((endpoints[:, -1, :2] != endpoints[:, -2, :2]).any(-1).all())
        raise CapturedPath

    monkeypatch.setattr(basket_nudge, "pose_segments", capture)
    with pytest.raises(CapturedPath):
        basket_nudge.plan_closed_nudge(
            state,
            {"asset": {"size_m": [0.04] * 3}},
            basket,
            None,
            collision,
            None,
            0.02,
            object_support_points=torch.zeros(1, 3),
        )


@pytest.mark.parametrize("path_unavailable", [False, True])
def test_reachable_endpoints_do_not_cancel_confirmed_full_path_recovery(
    monkeypatch, path_unavailable
):
    from gr00t.rl.pick import basket_nudge

    root = torch.zeros(1, 13)
    root[:, 3] = 1
    state = {"root_states": root, "object_state": root.clone()}
    basket = SimpleNamespace(floor_lower=torch.full((3,), -0.2), floor_upper=torch.full((3,), 0.2))
    monkeypatch.setattr(
        basket_nudge,
        "object_boxes_in_trunk",
        lambda *args: (torch.zeros(1, 3), torch.full((1, 3), 0.02)),
    )
    monkeypatch.setattr(
        basket_nudge,
        "shifted_grasp_counts",
        lambda state, episode, targets, *args, **kwargs: torch.ones(len(targets), dtype=torch.long),
    )

    class ReachedGeometryPlanning(Exception):
        pass

    def geometry(*args):
        raise ReachedGeometryPlanning

    monkeypatch.setattr(basket_nudge, "build_distractor_mesh", geometry)
    if path_unavailable:
        with pytest.raises(ReachedGeometryPlanning):
            basket_nudge.plan_closed_nudge(
                state,
                {},
                basket,
                None,
                None,
                None,
                0.02,
                object_support_points=torch.zeros(1, 3),
                current_path_unavailable=True,
            )
    else:
        path, report = basket_nudge.plan_closed_nudge(
            state, {}, basket, None, None, None, 0.02, object_support_points=torch.zeros(1, 3)
        )
        assert path is None and report["not_required"]


def test_shift_access_rejects_geometric_pinches_without_reachable_pregrasp_and_capture(monkeypatch):
    from gr00t.rl.pick import basket_nudge

    q = torch.tensor([[[2**-0.5, 0, 2**-0.5, 0]]])
    root = torch.zeros(1, 13)
    root[:, 3] = 1
    state = {"root_states": root, "object_state": root.clone(), "ee_orn": q[:, 0]}
    episode = {"asset": {"size_m": [0.04] * 3}}
    monkeypatch.setattr(basket_nudge, "object_frame_grasp_bank", lambda *args: q)
    monkeypatch.setattr(basket_nudge, "grasp_pose_bank", lambda *args, **kwargs: (None, q, None))
    monkeypatch.setattr(basket_nudge, "grasp_approach_origin", lambda root: root[:, :3])

    def pockets(orientation, target, *args, **kwargs):
        assert kwargs["retain_depth_candidates"]
        shape = orientation.shape[:2]
        return target[:, None].expand(-1, shape[1], -1), {
            "pocket_fit": torch.ones(shape, dtype=torch.bool),
            "antipodal_margin": torch.ones(shape),
            "support_pinch_compatible": torch.ones(shape, dtype=torch.bool),
            "environment_clearance_m": torch.full(shape, 0.01),
        }

    def inverse(state, p, q, *args, **kwargs):
        assert p.shape == (42, 2, 3)
        assert kwargs["body_height_offsets"] == (-0.01, 0.0, 0.01)
        assert kwargs["minimum_arm_clearance_m"] == 0.006
        assert kwargs["final_joint_margin_rad"] > 0.03
        safe = torch.zeros(42, dtype=torch.bool)
        safe[21] = safe[35] = True
        return safe, torch.zeros(42)

    monkeypatch.setattr(basket_nudge, "pocket_pose_targets", pockets)
    monkeypatch.setattr(basket_nudge, "safe_inverse_paths", inverse)
    report = {}
    count = basket_nudge.shifted_grasp_counts(
        state,
        episode,
        torch.zeros(2, 3),
        None,
        None,
        None,
        limits=torch.zeros(6, 2),
        diagnostics=report,
    )
    assert count.tolist() == [0, 2]
    assert report["geometric_grasp_counts"] == [3, 3]
    assert report["reachable_grasp_counts"] == [0, 2]


def test_release_cannot_deepen_existing_contact_or_recontact_after_separation():
    clearance = torch.tensor(
        [
            [-0.003, -0.002, 0.001, 0.003],
            [-0.003, -0.004, 0.001, 0.003],
            [-0.003, 0.003, -0.001, 0.003],
            [-0.003, -0.002, 0.000, 0.001],
            [0.004, 0.003, 0.006, 0.003],
        ]
    )
    assert monotone_release_mask(clearance).tolist() == [True, False, False, False, True]


def test_nudge_rejects_wider_space_with_no_future_reachable_grasp(monkeypatch):
    from gr00t.rl.pick import basket_nudge

    root = torch.zeros(1, 13)
    root[:, 3] = 1
    state = {"root_states": root, "object_state": root.clone()}
    basket = SimpleNamespace(floor_lower=torch.full((3,), -0.2), floor_upper=torch.full((3,), 0.2))
    monkeypatch.setattr(
        basket_nudge,
        "object_boxes_in_trunk",
        lambda *args: (torch.tensor([[-0.17, 0, 0]]), torch.full((1, 3), 0.02)),
    )
    monkeypatch.setattr(
        basket_nudge,
        "shifted_grasp_counts",
        lambda state, episode, targets, *args, **kwargs: torch.zeros(
            len(targets), dtype=torch.long
        ),
    )
    path, report = basket_nudge.plan_closed_nudge(
        state, {}, basket, None, None, None, 0.02, object_support_points=torch.zeros(1, 3)
    )
    assert path is None
    assert report["failure"] == "no_shift_improves_grasp_access_or_space"


@pytest.mark.parametrize("existing_access", [0, 1])
def test_centered_object_without_improving_shift_requests_regrasp_not_push(
    monkeypatch, existing_access
):
    from gr00t.rl.pick import basket_nudge

    root = torch.zeros(1, 13)
    root[:, 3] = 1
    state = {"root_states": root, "object_state": torch.zeros(1, 13)}
    basket = SimpleNamespace(floor_lower=torch.full((3,), -0.2), floor_upper=torch.full((3,), 0.2))
    monkeypatch.setattr(
        basket_nudge,
        "object_boxes_in_trunk",
        lambda *args: (torch.zeros(1, 3), torch.full((1, 3), 0.02)),
    )
    monkeypatch.setattr(
        basket_nudge,
        "shifted_grasp_counts",
        lambda state, episode, targets, *args, **kwargs: torch.full(
            (len(targets),), existing_access
        ),
    )
    path, report = basket_nudge.plan_closed_nudge(
        state, {}, basket, None, None, None, 0.02, object_support_points=torch.zeros(1, 3)
    )
    assert path is None and report["not_required"]
    assert report["space_before_m"] > 0.17


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU geometry")
def test_signed_target_surface_distinguishes_solid_overlap_from_cross_pocket_ray():
    import trimesh

    from gr00t.rl.pick.scene_geometry import SceneGeometry

    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    geometry = SceneGeometry(box.vertices, box.faces)
    p = torch.tensor([[[-0.3, 0.0, 0.0]]], device="cuda:0")
    q = p.new_tensor([[[1.0, 0.0, 0.0, 0.0]]])
    outside = p.new_tensor([[0.6, 0.0, 0.0]])
    assert geometry.rigid_surface_clearance(p, q, outside).item() == -1.0
    actual = geometry.rigid_surface_clearance(p, q, outside, check_rays=False, signed_distance=True)
    torch.testing.assert_close(actual, p.new_tensor([[0.2]]))
    inside = p.new_tensor([[0.3, 0.0, 0.0]])
    actual = geometry.rigid_surface_clearance(p, q, inside, check_rays=False, signed_distance=True)
    torch.testing.assert_close(actual, p.new_tensor([[-0.1]]))


def test_nudge_free_space_respects_wall_and_neighbor_boxes():
    basket = SimpleNamespace(
        floor_lower=torch.tensor([-0.2, -0.12, 0.1]), floor_upper=torch.tensor([0.2, 0.12, 0.1])
    )
    centers = torch.tensor([[-0.15, 0.0], [0.0, 0.0], [0.1, 0.0]])
    half = torch.tensor([0.025, 0.025])
    other = torch.tensor([[0.15, 0.0]])
    radius = torch.tensor([[0.025, 0.025]])
    margin = bbox_space(centers, half, basket, other, radius)
    torch.testing.assert_close(margin, torch.tensor([0.025, 0.095, 0.0]), atol=1e-6, rtol=1e-6)
    assert bbox_space(torch.tensor([[-0.12, 0.0]]), half, basket, other, radius) > margin[0]


def test_nudge_pose_segments_have_stable_endpoints_and_bounded_contact_speed():
    endpoints = torch.tensor(
        [[[0.0, 0, 0.9], [0.1, 0, 0.9], [0.1, 0, 0.7], [0.138, 0, 0.7], [0.138, 0, 0.9]]]
    )
    orientations = torch.tensor([[[1.0, 0, 0, 0]]]).expand(1, 5, 4)
    p, q, bounds = pose_segments(endpoints, orientations, 0.02, contact_segment=2)
    assert torch.equal(p[:, 0], endpoints[:, 0]) and torch.equal(p[:, -1], endpoints[:, -1])
    for segment in range(4):
        speed = p[:, bounds[segment] + 1 : bounds[segment + 1]].diff(dim=1).norm(dim=-1) / 0.02
        assert speed.max() <= (0.025 if segment == 2 else 0.15) + 1e-5
    assert torch.allclose(q.norm(dim=-1), torch.ones(q.shape[:2]))
    coarse, _, coarse_bounds = pose_segments(endpoints, orientations, 0.02, samples=9)
    assert coarse.shape == (1, 36, 3) and coarse_bounds == [0, 9, 18, 27, 36]


def test_nudge_object_boxes_follow_root_not_world_heading():
    episode = {"asset": {"size_m": [0.06, 0.04, 0.05]}, "distractors": []}
    root = torch.tensor([[2.0, 3, 0.52, 2**-0.5, 0, 0, 2**-0.5]])
    obj = torch.tensor([[2.0, 2.9, 0.67, 2**-0.5, 0, 0, 2**-0.5, 0, 0, 0, 0, 0, 0]])
    state = {
        "ee_pos": torch.zeros(1, 3),
        "root_states": root,
        "object_state": obj,
        "distractor_states": torch.empty(1, 0, 13),
    }
    centers, half = object_boxes_in_trunk(episode, state)
    torch.testing.assert_close(centers, torch.tensor([[-0.1, 0, 0.15]]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(half, torch.tensor([[0.03, 0.02, 0.025]]), atol=1e-6, rtol=1e-6)


def test_open_clear_checks_target_and_preserves_root_local_basket_collision(monkeypatch):
    from gr00t.rl.pick import basket_nudge

    scene, nonrobot = SimpleNamespace(), SimpleNamespace()
    scene.nonrobot_geometry = nonrobot
    basket = SimpleNamespace(world_mesh=lambda _: (torch.tensor([[0.0, 0, 0.73]]), None))
    state = {
        "root_states": torch.tensor([[0.0, 0, 0.52, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        "ee_pos": torch.tensor([[-0.2, 0.0, 0.80]]),
        "ee_orn": torch.tensor([[1.0, 0, 0, 0]]),
        "object_state": torch.tensor([[-0.2, 0, 0.67, 1, 0, 0, 0]]),
    }
    mesh = {"points": None, "faces": None}
    augmented = []

    def augment(base, obstacle):
        assert obstacle is mesh
        result = SimpleNamespace(source=base)
        augmented.append(result)
        return result

    def solve(state, p, q, limits, collision, geometry, *, diagnostics):
        assert geometry.basket_model is basket
        assert geometry.nonrobot_geometry.source is nonrobot
        assert geometry.source is scene
        return torch.ones(len(p), dtype=torch.bool), torch.zeros(len(p))

    monkeypatch.setattr(basket_nudge, "build_distractor_mesh", lambda *args: mesh)
    monkeypatch.setattr(basket_nudge, "augment_scene_geometry", augment)
    monkeypatch.setattr(basket_nudge, "safe_inverse_paths", solve)
    monkeypatch.setattr(
        basket_nudge,
        "SceneGeometry",
        lambda *args, **kwargs: SimpleNamespace(
            rigid_surface_clearance=lambda p, q, local, **kw: torch.full(p.shape[:2], 0.01)
        ),
    )

    def clearance(geometry, p, q):
        assert geometry is scene
        return torch.full(p.shape[:2], 0.01)

    path, report = basket_nudge.plan_clear_for_nudge(
        state,
        basket,
        None,
        SimpleNamespace(
            gripper_clearance=clearance,
            _gripper_local_surfaces=lambda _: (torch.zeros(1, 3), torch.zeros(1, 3)),
        ),
        scene,
        0.02,
        episode={},
    )
    assert report["valid"] and report["inverse_calls"] == 1
    assert report["candidate_count"] == 132
    torch.testing.assert_close(path[0][:, 0], state["ee_pos"])
    assert path[0][0, -1, 2] >= 0.89 - 1e-6


def test_inverse_arm_screen_excludes_filled_jaw_box_but_keeps_arm_clearance(monkeypatch):
    from gr00t.rl.pick import basket_nudge

    state = {"root_states": torch.zeros(1, 13), "dof_pos": torch.zeros(1, 6)}
    positions = torch.zeros(2, 3, 3)
    quats = torch.zeros(2, 3, 4)
    limits = torch.tensor([[-3.0, 3.0]] * 6)

    def inverse(p, q, roots, joints, limits):
        shape = p.shape[:2]
        return SimpleNamespace(
            exact=torch.ones(shape, dtype=torch.bool),
            position_error=torch.zeros_like(p),
            arm_joint_positions=torch.zeros(*shape, 6),
        )

    def clearance(geometry, roots, joints, *, include_gripper):
        assert not include_gripper
        return torch.tensor([[0.003, 0.004, 0.005], [0.003, 0.0002, 0.003]]).repeat(3, 1).flatten()

    monkeypatch.setattr(basket_nudge, "inverse_bank", inverse)
    collision = SimpleNamespace(arm_joint_ids=torch.arange(6), clearance=clearance)
    safe, _ = basket_nudge.safe_inverse_paths(
        state,
        positions,
        quats,
        limits,
        collision,
        None,
    )
    assert safe.tolist() == [True, False]
