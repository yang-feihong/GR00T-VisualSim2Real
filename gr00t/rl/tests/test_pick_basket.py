"""Basket placement uses the real mounted colliders, not visual-mesh bounds."""

import json
import math
from pathlib import Path

import pytest
import torch

from gr00t.rl.pick.assets import ASSET_ROOT, OBJECTS
from gr00t.rl.pick.basket import (
    BasketConfig,
    BasketGeometry,
    prepare_basket_episode,
    sample_basket_layout,
)


@pytest.fixture(scope="module")
def basket():
    if not torch.cuda.is_available():
        pytest.skip("Basket placement uses CUDA spatial queries")
    return BasketGeometry.from_robot_usd()


def assets():
    return [json.loads((ASSET_ROOT / name / "asset.json").read_text()) for name in OBJECTS]


def test_deployed_basket_support_accounts_for_trunk_surface(basket):
    assert len([name for name in basket.names if name.startswith("back_box_")]) == 63
    assert 0.38 < float(basket.floor_upper[0] - basket.floor_lower[0]) < 0.40
    assert 0.24 < float(basket.floor_upper[1] - basket.floor_lower[1]) < 0.26
    assert float(basket.rim_z) == pytest.approx(0.208, abs=0.01)
    xy = ((basket.floor_lower[:2] + basket.floor_upper[:2]) / 2)[None]
    height, normal = basket.support(xy)
    assert height.device.type == "cuda"
    assert float(height[0]) == pytest.approx(0.1275, abs=1e-5)
    assert height[0] > basket.floor_upper[2] + 0.02
    assert normal[0] > 0.99


@pytest.mark.parametrize("seed,count", [(1700, 1), (1701, 2), (1702, 3), (1703, 4)])
def test_basket_layout_separates_all_objects_inside_real_floor(basket, seed, count):
    config = BasketConfig()
    items, target, diagnostics = sample_basket_layout(assets(), basket, seed, count=count)
    assert len(items) == count and 0 <= target < count
    assert diagnostics["valid_count"] > 0
    assert len({item["asset"]["name"] for item in items}) == count
    positions = basket.boxes.new_tensor([item["basket_local_position"] for item in items])
    yaw = positions.new_tensor([item["basket_local_yaw"] for item in items])
    sizes = positions.new_tensor([item["asset"]["size_m"] for item in items])
    c, s = yaw.cos().abs(), yaw.sin().abs()
    extents = (
        torch.stack((c * sizes[:, 0] + s * sizes[:, 1], s * sizes[:, 0] + c * sizes[:, 1]), -1) / 2
    )
    assert ((positions[:, :2] - extents) >= basket.floor_lower[:2] + config.wall_gap_m - 1e-6).all()
    assert ((positions[:, :2] + extents) <= basket.floor_upper[:2] - config.wall_gap_m + 1e-6).all()
    separated = (
        (positions[:, None, :2] - positions[None, :, :2]).abs()
        >= extents[:, None] + extents[None, :] + config.object_gap_m - 1e-6
    ).any(-1)
    assert (separated | torch.eye(count, device=positions.device, dtype=torch.bool)).all()
    support, normal = basket.support(positions[:, :2])
    assert torch.allclose(positions[:, 2], support + sizes[:, 2] / 2 + 0.003)
    assert (normal > 0.97).all()
    assert all(item["support_type"] == "basket" for item in items)


def test_basket_layout_randomness_is_local_and_target_is_not_always_first(basket):
    source = assets()
    expected = sample_basket_layout(source, basket, 1703, count=4)
    torch.manual_seed(197)
    torch.rand(100, device="cuda:0")
    assert sample_basket_layout(source, basket, 1703, count=4) == expected
    layouts = [sample_basket_layout(source, basket, seed) for seed in range(1710, 1720)]
    assert len({len(items) for items, _, _ in layouts}) > 1
    assert any(target > 0 for _, target, _ in layouts)


def test_admission_resamples_only_payload_layout_and_preserves_object_identity(basket):
    import copy

    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.pick.basket_initialization import resample_basket_payloads

    layout, selected, diagnostics = sample_basket_layout(assets(), basket, 1701, count=2)
    for item in layout:
        item["support_z"] = 0.55 + item["support_z_local"]
    original = {
        **layout[selected],
        "seed": 1701,
        "base_xy": [1, 2],
        "base_yaw": 0.6,
        "scene_id": "room",
        "distractors": [d for i, d in enumerate(layout) if i != selected],
        "basket_sampling": diagnostics,
    }
    snapshot = copy.deepcopy(original)
    episode = resample_basket_payloads([original], [0], basket, 1)[0]
    assert original == snapshot
    for key in ("asset", "scene_id", "base_xy", "base_yaw"):
        assert episode[key] == original[key]
    assert episode["distractors"][0]["asset"] == original["distractors"][0]["asset"]
    for item in [episode, *episode["distractors"]]:
        rotation = basket.boxes.new_tensor([[math.cos(0.3), 0, 0, math.sin(0.3)]])
        local = basket.boxes.new_tensor([item["basket_local_position"]])
        expected = _quat_rotate(rotation, local)[0] + local.new_tensor([1, 2, 0.55])
        torch.testing.assert_close(expected, expected.new_tensor(item["object_position"]))
    assert episode["basket_sampling"]["resample_attempt"] == 1


def test_basket_rejects_duplicate_categories_and_invalid_counts(basket):
    source = assets()
    with pytest.raises(ValueError, match="unique"):
        sample_basket_layout([source[0], source[0]], basket, 1700)
    for count in (0, 5):
        with pytest.raises(ValueError, match="count"):
            sample_basket_layout(source, basket, 1700, count=count)


def test_basket_layout_does_not_silently_reduce_count_or_allow_overlap(basket):
    source = [{"name": str(i), "size_m": [0.6, 0.6, 0.05]} for i in range(4)]
    with pytest.raises(RuntimeError, match="No nonpenetrating"):
        sample_basket_layout(source, basket, 1700, count=4)


def test_initial_layout_retries_combinations_without_reducing_object_count(monkeypatch, basket):
    from gr00t.rl.pick import basket_preparation as module
    from gr00t.rl.pick.basket import NoValidBasketLayout

    calls = []
    monkeypatch.setattr(module.BasketGeometry, "from_robot_usd", lambda: basket)
    monkeypatch.setattr(module, "episode_asset_pool", lambda *a, **kw: [{"category": "fruit"}])
    monkeypatch.setattr(module, "discover_vlnverse_scenes", lambda *a, **kw: [])

    def sample(assets, basket, seeds, config, *, counts):
        calls.append((seeds, counts))
        if len(calls) == 1:
            return [NoValidBasketLayout("no_layout", {"object_count": 4, "asset_names": ["a"]})]
        return [([], 0, {"object_count": counts[0]})]

    monkeypatch.setattr(module, "sample_layouts", sample)
    monkeypatch.setattr(
        module, "prefer_materialized_layouts", lambda layouts, *args, **kwargs: layouts
    )
    module.prepare_basket_batch([3], None, None)
    assert calls == [([3], [None]), ([3 + 0x4C41594F], [4])]


def test_initial_layout_exhaustion_is_a_recordable_initialization_rejection(monkeypatch, basket):
    from gr00t.rl.pick import basket_preparation as module
    from gr00t.rl.pick.basket import NoValidBasketLayout

    monkeypatch.setattr(module.BasketGeometry, "from_robot_usd", lambda: basket)
    monkeypatch.setattr(module, "episode_asset_pool", lambda *a, **kw: [{"category": "fruit"}])
    monkeypatch.setattr(module, "discover_vlnverse_scenes", lambda *a, **kw: [])
    monkeypatch.setattr(
        module,
        "sample_layouts",
        lambda *a, **kw: [
            NoValidBasketLayout("no_layout", {"object_count": 4, "asset_names": ["a"]})
        ],
    )
    failure = module.prepare_basket_batch([3], None, None)[3]
    assert isinstance(failure, module.NoValidInitialScene)
    assert failure.details["object_count"] == 4
    assert len(failure.details["combination_rejections"]) == 16


def test_basket_world_mesh_tracks_measured_root_pose(basket):
    root = basket.boxes.new_tensor([2, 3, 0.51, 2**-0.5, 0, 0, 2**-0.5])
    points, faces = basket.world_mesh(root)
    original = basket.boxes.reshape(-1, 3)
    assert torch.allclose(points[:, 0], 2 - original[:, 1], atol=1e-6)
    assert torch.allclose(points[:, 1], 3 + original[:, 0], atol=1e-6)
    assert torch.allclose(points[:, 2], 0.51 + original[:, 2], atol=1e-6)
    assert torch.equal(faces, basket.faces)


@pytest.mark.parametrize(
    "direct,full_path,clear_ok,nudge_ok,accepted",
    [
        (1, True, False, False, True),
        (1, False, False, False, False),
        (1, False, True, True, True),
        (0, False, False, False, False),
        (0, False, True, False, False),
        (0, False, True, True, True),
    ],
)
def test_initial_access_allows_safe_reposition_but_not_an_inaccessible_layout(
    monkeypatch, direct, full_path, clear_ok, nudge_ok, accepted
):
    from gr00t.rl.pick import basket_initialization as initial

    state = {"object_state": torch.zeros(1, 13)}
    path = (torch.zeros(1, 2, 3), torch.zeros(1, 2, 4))
    called = []
    monkeypatch.setattr(
        initial, "shifted_grasp_counts", lambda *args, **kwargs: torch.tensor([direct])
    )
    monkeypatch.setattr(initial, "basket_required_height", lambda *args: torch.tensor([0.8]))

    def full(*args, **kwargs):
        assert kwargs["realized_body"]
        assert kwargs["minimum_realized_grasp_margin_rad"] == pytest.approx(torch.pi / 90)
        return (path if full_path else None), {"full_path": full_path}

    monkeypatch.setattr(initial, "plan_basket_grasp_path", full)

    def clear(*args, **kwargs):
        called.append("clear")
        return (path if clear_ok else None), {"valid": clear_ok}

    def nudge(*args, **kwargs):
        called.append("nudge")
        return (path if nudge_ok else None), {"valid": nudge_ok}

    monkeypatch.setattr(initial, "plan_clear_for_nudge", clear)
    monkeypatch.setattr(initial, "plan_closed_nudge", nudge)
    passed, report = initial.basket_grasp_access(
        state, {"asset": {"size_m": [0.04] * 3}}, None, None, None, None, None, 0.02
    )
    assert passed is accepted
    assert report["direct_reachable_grasp_candidates"] == direct
    assert report["direct_path_available"] == bool(direct and full_path)
    assert called == ([] if direct and full_path else ["clear", "nudge"] if clear_ok else ["clear"])


@pytest.mark.parametrize("changed", [False, True])
def test_basket_reuses_initial_full_path_only_for_identical_physical_state(monkeypatch, changed):
    from gr00t.rl.pick import basket_controller as module

    task = module.BasketPickController.__new__(module.BasketPickController)
    task.final_entry, task.entry_orientation, task.lift_margin = False, None, 0.1
    state = {
        key: torch.zeros(1, 13)
        for key in ("root_states", "dof_pos", "object_state", "distractor_states")
    }
    path = (torch.zeros(1, 2, 3), torch.zeros(1, 2, 4))
    task.initial_grasp_plan = {
        "state": {key: value.clone() for key, value in state.items()},
        "maximum_tool_tilt_deg": 30.0,
        "paths": {"approach": path},
        "diagnostics": {},
    }
    if changed:
        state["dof_pos"] += 0.01
    calls = []

    def plan(*args, **kwargs):
        calls.append(True)
        return {"approach": path}, {}

    task.required_height = lambda state: torch.tensor([0.8])
    monkeypatch.setattr(module, "plan_basket_grasp_path", plan)
    monkeypatch.setattr(
        module, "above_object_path", lambda path, diagnostics: (path, path[1][:, -1])
    )
    _, diagnostics = task.plan_basket_grasp(state)
    assert calls == ([True] if changed else [])
    assert diagnostics.get("reused_initial_admission_plan", False) == (not changed)
    assert task.initial_grasp_plan is None


def test_prepare_basket_episode_in_vlnverse_is_stationary_and_semantic(basket, tmp_path):
    scenes = Path("/datasets/vlnverse_scenes")
    if not scenes.is_dir():
        pytest.skip("Local VLNVerse scenes not installed")
    path = tmp_path / "basket.json"
    episode = prepare_basket_episode(1700, scenes, path, scene_id="kujiale_0005", count=3)
    assert json.loads(path.read_text()) == episode
    assert episode["task_type"] == "pick_basket"
    assert episode["base_xy"] == episode["station_xy"]
    assert episode["base_yaw"] == episode["station_yaw"]
    assert episode["target_semantics"]["support_category"] == "basket"
    items = [episode, *episode["distractors"]]
    assert len(items) == 3
    assert len({item["asset"]["name"] for item in items}) == 3
    assert episode["initial_admission_required"]
    preset = torch.tensor(episode["arm_preset"])
    initial = torch.tensor(episode["initial_arm_joints"])
    assert ((initial - preset).abs() <= 0.0501).all()
    root = basket.boxes.new_tensor([*episode["base_xy"], 0.55, 1, 0, 0, 0])
    yaw = root.new_tensor(episode["base_yaw"])
    root[3], root[6] = (yaw / 2).cos(), (yaw / 2).sin()
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate

    for item in items:
        expected = (
            _quat_rotate(root[None, 3:7], root.new_tensor([item["basket_local_position"]]))[0]
            + root[:3]
        )
        assert torch.allclose(expected, root.new_tensor(item["object_position"]))


def test_basket_clearance_uses_object_bottom_and_all_measured_root_poses(basket):
    from gr00t.rl.pick.basket_controller import basket_vertical_clearance

    roots = basket.boxes.new_zeros(2, 13)
    roots[:, 2] = roots.new_tensor([0.50, 0.55])
    roots[:, 3] = 1
    objects = roots.clone()
    objects[:, 2] = 0.85
    points = roots.new_tensor([[0, 0, -0.03], [0, 0, 0.03]])
    margin, rim = basket_vertical_clearance(basket, roots, objects, points)
    assert torch.allclose(rim, roots[:, 2] + basket.rim_z)
    assert torch.allclose(margin, 0.82 - rim)
    assert margin[0] > 0.10 and margin[1] < 0.10


def test_basket_command_publishes_zero_base_and_shared_ee_path(monkeypatch):
    from gr00t.rl.auto_open.execution import TensorCommandBuffers
    from gr00t.rl.tests.test_pick_workflow import fake_controller, tensor_task

    config, state = fake_controller(monkeypatch)
    config.current_stage = config.Stage.MOVE_TO_GRASP
    controller = tensor_task(config, state)
    controller.basket = object()
    buffers = TensorCommandBuffers.from_env(controller.env)
    calls = []
    controller.arm_trajectory = lambda ids, path, data, **kw: calls.append((path, kw))
    controller.gripper_target = lambda ids, target, data: calls.append(target.clone())
    controller.command(state, buffers)
    assert torch.equal(buffers.base, torch.zeros(1, 3))
    assert calls[0] == (controller.arm_path, {"advance": True})
    assert float(calls[1][0]) == pytest.approx(-torch.pi / 3)


@pytest.mark.parametrize("prefix", [(2,), (2, 3, 4)])
def test_basket_arm_clearance_keeps_self_geometry_attached_to_each_root(prefix):
    from types import SimpleNamespace

    from gr00t.rl.pick.basket_controller import BasketArmCollisionSamples

    calls = []

    def clearance(geometry, roots, joints, **kwargs):
        calls.append((geometry, roots.clone(), kwargs))
        return roots.new_full(roots.shape[:-1], geometry)

    geometry = SimpleNamespace(nonrobot_geometry=0.03, basket_model=SimpleNamespace(geometry=0.01))
    roots = torch.zeros(*prefix, 13)
    roots[..., :3] = torch.tensor([2, 3, 0.5])
    roots[..., 3] = 1
    wrapper = BasketArmCollisionSamples(SimpleNamespace(clearance=clearance))
    result = wrapper.clearance(geometry, roots, torch.zeros(*prefix, 6), include_gripper=False)
    assert torch.equal(result, torch.full(prefix, 0.01))
    assert torch.equal(calls[0][1], roots)
    assert not calls[1][1][..., :3].any()
    assert torch.equal(calls[1][1][..., 3:7], torch.tensor([1, 0, 0, 0]).expand(*prefix, 4))
    assert calls[1][2] == {"include_gripper": False}


def test_basket_payload_reaction_is_not_misclassified_as_environment_contact():
    from gr00t.rl.pick.tensor_fsm import TensorPickFSM

    controller = TensorPickFSM.__new__(TensorPickFSM)
    controller.basket = object()
    controller.contact_body_names = ["base_link", "link06", "gripperMover"]
    controller.jaw_contact_ids = [1, 2]
    state = {
        "contact_forces_w": torch.tensor([[[2.0, 0, -43], [1.0, 2, 30], [-1.0, -2, -30]]]),
        "object_finger_forces": torch.tensor([[[1.0, 2, 30], [-1.0, -2, -30]]]),
        "basket_payload_normal_reaction_w": torch.tensor([[0.0, 0, -43]]),
        "basket_payload_friction_reaction_w": torch.tensor([[0.0, 25, 0]]),
    }
    raw = state["contact_forces_w"].clone()
    forces = controller.unexpected_contact_forces(state)
    assert torch.equal(forces, torch.tensor([[[2.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]]]))
    assert torch.equal(state["contact_forces_w"], raw)
    state["contact_forces_w"][:, 0, 1] = 40
    assert controller.unexpected_contact_forces(state)[:, 0].norm(dim=-1) > 20


@pytest.mark.parametrize("final_entry", [False, True])
def test_basket_reuses_shared_pose_bank_and_keeps_only_top_down_candidates(
    monkeypatch, final_entry
):
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
    from gr00t.rl.pick import basket_controller

    controller = basket_controller.BasketPickController.__new__(
        basket_controller.BasketPickController
    )
    controller.lift_margin = 0.10
    controller.final_entry = final_entry
    controller.required_height = lambda state: torch.tensor([0.9])
    root = torch.tensor([[0.0, 0, 0.52, 1, 0, 0, 0]])
    state = {
        "root_states": root,
        "ee_orn": torch.tensor([[1.0, 0, 0, 0]]),
        "object_state": root.clone(),
    }
    quats = _euler_xyz_quat(
        torch.tensor([[0.0, 0, 0], [0, torch.pi / 2, 0], [0, torch.pi / 4, 0]])
    )[None]
    controller.entry_orientation = quats[:, 1]
    calls = []
    monkeypatch.setattr(basket_controller, "grasp_approach_origin", lambda root: root[:, :3])
    monkeypatch.setattr(
        basket_controller, "grasp_pose_bank", lambda *args, **kwargs: (None, quats, None)
    )

    def planner(state, *args, **kwargs):
        calls.append(kwargs)
        return None, {}

    monkeypatch.setattr(basket_controller, "plan_grasp", planner)
    controller.plan_basket_grasp(state, torch.tensor([[0.0, 0, 0.7]]), [0.05] * 3)
    assert len(calls) == 1
    assert calls[0]["candidate_orientations"].shape == (1, 1, 4)
    torch.testing.assert_close(calls[0]["candidate_orientations"], quats[:, 1:2])
    assert calls[0]["object_frame_fallback"]
    assert calls[0]["fallback_tool_tilt_deg"] == 30.0
    assert not calls[0]["require_precise_carry"]
    assert not calls[0]["validate_carry_preview"]
    assert calls[0]["require_safe_arm_path"]
    assert calls[0]["collider_clearance_m"] == 0.006
    assert calls[0]["withdraw_distance_m"] == 0


def test_supported_pinch_alignment_is_frame_invariant_and_symmetric_between_jaws():
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.pick.grasp_quality import supported_pinch_alignment

    fixed = torch.zeros(1, 4, 3)
    moving = torch.tensor([[[1.0, 0, 0], [1.0, 0, -0.1], [1.0, 0, -0.4], [1.0, 0, 0.1]]])
    q = torch.tensor([[[1.0, 0, 0, 0]]]).expand(1, 4, 4)
    mask, sine = supported_pinch_alignment(q, fixed, moving, torch.tensor([[0.0, 0, 1.0]]))
    assert mask.tolist() == [[True, True, False, True]]
    turn = _euler_xyz_quat(torch.tensor([[0.2, -0.3, 1.4]]))
    rotated_mask, rotated_sine = supported_pinch_alignment(
        turn[:, None].expand_as(q), fixed, moving, _quat_rotate(turn, torch.tensor([[0.0, 0, 1.0]]))
    )
    assert torch.equal(rotated_mask, mask)
    torch.testing.assert_close(rotated_sine, sine, atol=1e-6, rtol=1e-6)
