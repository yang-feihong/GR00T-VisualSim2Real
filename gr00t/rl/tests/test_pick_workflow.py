"""Pick orchestration must preserve low world targets and shared execution contracts."""

import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.auto_open.acceptance import TensorContactWindow, TensorJointStabilityWindow
from gr00t.rl.auto_open.execution import StageExecutionMixin, TensorCommandBuffers
from gr00t.rl.pick import controller, planning
from gr00t.rl.pick.controller import PickController, PickStage


def test_pick_closes_faster_without_increasing_contact_preload():
    from gr00t.rl.third_party.visual_wholebody_lowlevel.legged_gym.envs.manip_loco import (
        gripper_control,
    )

    source = Path(__file__).resolve().parents[1] / "pick" / "environment.py"
    settings = next(
        node.value
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "gripper_control_options"
            for target in node.targets
        )
    )
    options = {
        key.value: ast.literal_eval(value)
        for key, value in zip(settings.keys, settings.values)
        if isinstance(value, ast.Constant)
    }
    assert options["free_close_step"] == 0.024
    assert options["contact_close_step"] == 0.006
    assert options["free_command_lead"] == 0.060
    assert options["contact_command_lead"] == 0.018
    assert options["hold_command_lead"] == 0.006
    previous = torch.tensor([[-0.8], [-0.8], [-0.8]])
    target, held = gripper_control.advance_gripper_actuator_targets(
        torch.zeros_like(previous),
        previous,
        previous,
        torch.tensor([False, False, True]),
        torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]]),
        **options,
    )
    torch.testing.assert_close(target - previous, torch.tensor([[0.024], [0.006], [0.006]]))
    assert held.tolist() == [False, False, True]


def test_closure_timing_separates_motion_from_post_contact_hold():
    from gr00t.rl.pick.inspect_run import closure_timing

    angles = torch.cat((torch.arange(11, dtype=torch.float64) / 10, torch.ones(30)))
    result = closure_timing(angles, 0.02)
    assert result["motion_10_to_90_s"] == pytest.approx(0.16)
    assert result["entry_to_90_percent_s"] == pytest.approx(0.18)
    assert result["peak_finite_difference_speed_rad_s"] == pytest.approx(5.0)
    assert closure_timing(torch.ones(10), 0.02)["motion_10_to_90_s"] is None


def test_natural_gaze_optical_centers_follow_q2_q3_fk_not_vertical_translation():
    from gr00t.rl.envs.door.b2z1_reset_geometry import (
        b2z1_gripper_pose_and_jacobian,
        quaternion_xyzw_to_matrix,
    )
    from gr00t.rl.pick.gaze import camera_pose, natural_gaze_candidates

    roots = torch.zeros(1, 101, 13)
    roots[..., 2], roots[..., 3] = 0.53, 1
    initial = torch.tensor([[-1.57, 2.6, -1.5, 0.4, 0.0, 0.0]])
    target = torch.tensor([[1.3, 0.0, 0.82]])
    limits = initial.new_tensor([[-3.14, 3.14], [0, 2.96], [-2.87, 0], [-2, 2], [-2, 2], [-3, 3]])
    p, q, joints, expanded, shoulder, _ = natural_gaze_candidates(
        roots, initial, target, limits, 0.02, torch.zeros(1, 3)
    )
    fk_p, fk_r, _ = b2z1_gripper_pose_and_jacobian(
        expanded[..., :3], expanded[..., [4, 5, 6, 3]], joints
    )
    expected_eye, _ = camera_pose(fk_p, fk_r)
    eye, axis = camera_pose(p, quaternion_xyzw_to_matrix(q[..., [1, 2, 3, 0]]))
    torch.testing.assert_close(eye, expected_eye)
    assert (eye[..., :2] - eye[:1, ..., :2]).norm(dim=-1).amax() > 0.05
    expected_q2 = (initial[0, 1] + shoulder).clamp(*limits[1])
    torch.testing.assert_close(joints[..., 1], expected_q2[:, None].expand_as(joints[..., 1]))
    torch.testing.assert_close(
        joints[..., [0, 3, 4, 5]], initial[:, None, [0, 3, 4, 5]].expand(len(shoulder), 101, 4)
    )
    direction = target[:, None] - eye
    torch.testing.assert_close(
        axis, direction / direction.norm(dim=-1, keepdim=True), atol=1e-5, rtol=0
    )


def test_live_gaze_uses_measured_camera_center_without_moving_planned_center():
    from gr00t.rl.envs.door.b2z1_reset_geometry import quaternion_xyzw_to_matrix
    from gr00t.rl.pick.gaze import camera_pose, gaze_from_actual_camera

    roots = torch.zeros(1, 2, 13)
    roots[..., 3] = 1
    target = torch.tensor([[1.0, 0.1, 0.8]])
    p = torch.tensor([[[0.5, 0.0, 0.9], [0.5, 0.0, 0.9]]])
    q = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    planned_p, planned_q = p[:, -1], q[:, -1]
    actual_p = planned_p + torch.tensor([[0.0, 0.0, 0.1]])
    actual_q = planned_q.clone()
    planned_eye, _ = camera_pose(planned_p, quaternion_xyzw_to_matrix(planned_q[:, [1, 2, 3, 0]]))
    actual_eye, _ = camera_pose(actual_p, quaternion_xyzw_to_matrix(actual_q[:, [1, 2, 3, 0]]))
    command_p, command_q = gaze_from_actual_camera(
        planned_p, planned_q, actual_p, actual_q, target, torch.zeros(1, 3), torch.ones(1)
    )
    eye, axis = camera_pose(command_p, quaternion_xyzw_to_matrix(command_q[:, [1, 2, 3, 0]]))
    desired = target - actual_eye
    torch.testing.assert_close(
        axis, desired / desired.norm(dim=-1, keepdim=True), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(eye, planned_eye)
    assert (command_p[:, 2] >= target[:, 2]).all()


def test_natural_gaze_selects_only_commands_above_object(monkeypatch):
    from dataclasses import make_dataclass

    from gr00t.rl.pick import gaze

    Result = make_dataclass(
        "Result", ["arm_joint_positions", "position_error", "orientation_error"]
    )
    roots = torch.zeros(1, 101, 13)
    roots[..., 3] = 1
    roots[..., 2] = 0.53
    initial = torch.tensor([[-1.57, 2.6, -1.5, 0.4, 0.0, 0.0]])
    limits = initial.new_tensor([[-3.14, 3.14], [0, 2.96], [-2.87, 0], [-2, 2], [-2, 2], [-3, 3]])
    target = torch.tensor([[1.3, 0.0, 0.82]])
    monkeypatch.setattr(
        gaze,
        "solve_pose_targets",
        lambda p, q, **kw: Result(
            arm_joint_positions=kw["initial_arm_joint_positions"],
            position_error=torch.zeros_like(p),
            orientation_error=torch.zeros_like(p),
        ),
    )
    collision = SimpleNamespace(clearance=lambda g, r, j, **kw: torch.full(r.shape[:-1], 0.3))
    p, _, diagnostics = gaze.plan_approach_gaze(
        roots, initial, target, limits, 0.02, collision, None, torch.zeros(1, 3)
    )
    assert diagnostics["valid"]
    assert (p[..., 2] >= target[:, None, 2]).all()


def test_approach_offsets_are_seeded_bounded_and_do_not_consume_global_rng():
    from gr00t.rl.pick.gaze import sample_approach_offsets

    state = torch.random.get_rng_state()
    sample = sample_approach_offsets(1700, "cpu")
    torch.testing.assert_close(sample, sample_approach_offsets(1700, "cpu"))
    assert not torch.equal(sample, sample_approach_offsets(1701, "cpu"))
    assert torch.all(sample.abs() <= torch.deg2rad(torch.tensor([5.0, 3.0, 3.0, 10.0])))
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("roll", [-math.pi, -math.pi / 2, 0.15, math.pi / 2, math.pi])
def test_camera_roll_is_about_actual_optical_axis_and_preserves_camera_center(roll):
    from gr00t.rl.envs.door.b2z1_reset_geometry import quaternion_xyzw_to_matrix
    from gr00t.rl.pick.gaze import camera_pose, perturb_camera_aim

    positions = torch.tensor([[[0.5, 0.1, 0.8], [0.5, 0.1, 0.8]]])
    orientations = torch.tensor([[[1.0, 0, 0, 0], [1.0, 0, 0, 0]]])
    offsets = torch.tensor([[0.0, 0.0, roll]])
    p, q = perturb_camera_aim(positions, orientations, offsets, 2.0)
    before_eye, before_axis = camera_pose(
        positions, quaternion_xyzw_to_matrix(orientations[..., [1, 2, 3, 0]])
    )
    after_eye, after_axis = camera_pose(p, quaternion_xyzw_to_matrix(q[..., [1, 2, 3, 0]]))
    torch.testing.assert_close(after_eye, before_eye)
    torch.testing.assert_close(after_axis, before_axis)
    torch.testing.assert_close(p[:, 0], positions[:, 0])
    torch.testing.assert_close(q[:, 0], orientations[:, 0])
    assert not torch.allclose(q[:, -1], orientations[:, -1])
    p, q = perturb_camera_aim(positions, orientations, torch.tensor([[0.04, -0.03, 0.15]]), 2.0)
    eye, axis = camera_pose(p, quaternion_xyzw_to_matrix(q[..., [1, 2, 3, 0]]))
    torch.testing.assert_close(eye, before_eye)
    assert not torch.allclose(axis[:, -1], before_axis[:, -1])
    p_roll, q_roll = perturb_camera_aim(
        positions, orientations, torch.tensor([[0.04, -0.03, roll]]), 2.0
    )
    eye_roll, axis_roll = camera_pose(p_roll, quaternion_xyzw_to_matrix(q_roll[..., [1, 2, 3, 0]]))
    torch.testing.assert_close(eye_roll, eye)
    torch.testing.assert_close(axis_roll, axis)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_approach_gaze_is_q3_rotation_fk_with_real_camera_mount(device):
    from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
    from gr00t.rl.pick.gaze import camera_pose, q3_gaze_path

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    initial = torch.tensor(
        [[-1.57, 2.6, -1.5, 0.4, 0.0, 0.0], [-1.55, 2.62, -1.48, 0.38, 0.02, -0.01]],
        device=device,
    )
    roots = torch.zeros(2, 101, 13, device=device)
    roots[..., 2], roots[..., 3] = 0.53, 1
    desired = initial.clone()
    desired[:, 2] = torch.tensor([-1.9, -1.1], device=device)
    p, r, _ = b2z1_gripper_pose_and_jacobian(roots[:, 0, :3], roots[:, 0, [4, 5, 6, 3]], desired)
    camera, optical = camera_pose(p, r)
    target = camera + 2 * optical
    limits = initial.new_tensor([[-3.14, 3.14], [0, 2.96], [-2.87, 0], [-2, 2], [-2, 2], [-3, 3]])
    positions, orientations, joints = q3_gaze_path(roots, initial, target, limits, 0.02)
    torch.testing.assert_close(joints[:, 0], initial)
    torch.testing.assert_close(joints[:, -1], desired, atol=2e-5, rtol=0)
    torch.testing.assert_close(
        joints[..., [0, 1, 3, 4, 5]], initial[:, None, [0, 1, 3, 4, 5]].expand(2, 101, 5)
    )
    expected_p, expected_r, _ = b2z1_gripper_pose_and_jacobian(
        roots[..., :3], roots[..., [4, 5, 6, 3]], joints
    )
    torch.testing.assert_close(positions, expected_p)
    assert orientations.shape == (2, 101, 4)
    camera, optical = camera_pose(positions[:, -1], expected_r[:, -1])
    direction = target - camera
    torch.testing.assert_close(
        optical, direction / direction.norm(dim=-1, keepdim=True), atol=2e-5, rtol=0
    )


def test_auxiliary_ground_hiding_preserves_physics_and_scene_floor():
    from pxr import Usd, UsdGeom, UsdPhysics

    from gr00t.rl.pick.cameras import hide_auxiliary_ground_mesh

    stage = Usd.Stage.CreateInMemory()
    visual = UsdGeom.Mesh.Define(stage, "/World/ground/terrain/Environment/Geometry")
    collider = UsdGeom.Plane.Define(stage, "/World/ground/terrain/GroundPlane/CollisionPlane")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    scene_floor = UsdGeom.Mesh.Define(stage, "/World/VlnverseScene/floor")
    before = collider.GetPrim().GetPrimStack()[0].GetAsText()
    hide_auxiliary_ground_mesh(stage, "/World/ground")
    assert visual.ComputeVisibility() == "invisible"
    assert scene_floor.ComputeVisibility() == "inherited"
    assert collider.GetPrim().IsActive()
    assert collider.GetPrim().GetPrimStack()[0].GetAsText() == before


def test_pick_onboard_cameras_preserve_door_fisheye_calibration():
    from legged_gym.scripts.rgb_camera_debug import rgb_camera_specs_from_config

    root = Path(__file__).resolve().parents[1]
    pick = json.loads((root / "pick/cameras.json").read_text())
    parsed = rgb_camera_specs_from_config(pick)
    assert len(parsed["cameras"]) == 6
    from gr00t.rl.pick.cameras import effective_camera_config

    actual, _ = effective_camera_config(pick, 1700)
    for camera in actual["cameras"][:2]:
        assert camera["cx"] == float(np.float32(camera["cx"]))
        assert camera["skew"] == 0.0
    door = json.loads(
        (
            root / "third_party/visual_wholebody_lowlevel/legged_gym/scripts/"
            "rgb_cameras_dataset.json"
        ).read_text()
    )
    for camera in pick["cameras"][:2]:
        original = next(c for c in door["cameras"] if c["name"] == camera["name"])
        for key in ("position", "rotation", "body_name"):
            assert camera[key] == original[key]
        for key in (
            "projection_type",
            "fx",
            "fy",
            "cx",
            "cy",
            "distortion_coefficients",
            "valid_circle_mask",
        ):
            assert camera[key] == door["defaults"][key]
        assert camera["skew"] == 0.0  # Isaac Sim effective calibration, as in door capture.


def test_pick_diagnostic_views_look_at_their_subjects():
    from gr00t.rl.pick.cameras import DIAGNOSTIC_VIEWS, diagnostic_pose

    root = torch.tensor([1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    item = torch.tensor([2.0, 2.0, 0.1])
    for name in DIAGNOSTIC_VIEWS:
        eye, q = diagnostic_pose(name, root, item)
        w, x, y, z = q
        forward = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)))
        target = item if name.startswith("object") else root[:3]
        if name == "robot_rear_diagnostic":
            target = target + torch.tensor([0.4, 0, 0])
        direction = target - eye
        assert torch.allclose(forward, direction / direction.norm(), atol=1e-6)
        assert torch.allclose(q.norm(), torch.tensor(1.0), atol=1e-6)


@dataclass
class InverseResult:
    command_position_world: torch.Tensor
    command_orientation_world_wxyz: torch.Tensor
    arm_joint_positions: torch.Tensor
    exact: torch.Tensor
    position_error: torch.Tensor
    orientation_error: torch.Tensor


def inverse_result(position, quaternion, *, projection=0.0, exact=None):
    count = len(position)
    return InverseResult(
        command_position_world=position + projection,
        command_orientation_world_wxyz=quaternion.clone(),
        arm_joint_positions=torch.zeros(count, 6),
        exact=torch.ones(count, dtype=torch.bool) if exact is None else exact,
        position_error=torch.zeros_like(position),
        orientation_error=torch.zeros(count, 3),
    )


@pytest.fixture
def cpu_contact_profiles(monkeypatch):
    from gr00t.rl.pick import grasp_quality, pocket

    if not torch.cuda.is_available():
        pytest.skip("Asset contact profiles are generated on GPU")
    profile = pocket.ContactProfiles.__new__(pocket.ContactProfiles)
    profile.maps = {
        key: value.cpu() for key, value in pocket.gripper_contact_profiles("cuda:0").maps.items()
    }
    original = pocket.gripper_contact_profiles
    monkeypatch.setattr(
        grasp_quality,
        "gripper_contact_profiles",
        lambda device: profile if str(device) == "cpu" else original(device),
    )
    return profile


@pytest.mark.parametrize(
    "target_height,lift_distance", [(0.03, 0.27), (0.10, 0.20), (0.20, 0.10), (0.82, 0.10)]
)
@pytest.mark.parametrize("closing_clearance", [1.0, -1.0])
@pytest.mark.parametrize("endpoint_exact", [False, True])
def test_low_grasp_path_is_not_replaced_by_fixed_base_inverse_projection(
    monkeypatch,
    target_height,
    lift_distance,
    closing_clearance,
    cpu_contact_profiles,
    endpoint_exact,
):
    calls = []

    def solve(position, quaternion, **kwargs):
        calls.append(kwargs["base_position_world"])
        return inverse_result(
            position,
            quaternion,
            projection=0.30,
            exact=torch.full((len(position),), endpoint_exact, dtype=torch.bool),
        )

    monkeypatch.setattr(planning, "solve_pose_targets", solve)
    root = torch.zeros(1, 13)
    root[:, 2], root[:, 3] = 0.536, 1
    state = {
        "root_states": root,
        "dof_pos": torch.zeros(1, 6),
        "ee_pos": torch.tensor([[0.4, 0, 0.8]]),
        "ee_orn": torch.tensor([[1.0, 0, 0, 0]]),
    }
    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6),
        clearance=lambda geometry, roots, joints, **kwargs: torch.ones(roots.shape[:-1]),
        gripper_clearance=lambda geometry, positions, orientations, **options: torch.ones(
            positions.shape[:2]
        ),
        gripper_closing_clearance=lambda geometry, positions, orientations, aperture, **options: (
            torch.full_like(aperture, closing_clearance)
        ),
    )
    target = torch.tensor([[0.65, 0, target_height]])
    state["object_state"] = root.clone()
    state["object_state"][:, :3] = target
    latitude, longitude = torch.meshgrid(
        torch.linspace(-1.57, 1.57, 9), torch.linspace(-3.14, 3.14, 16), indexing="ij"
    )
    points = 0.03 * torch.stack(
        (latitude.cos() * longitude.cos(), latitude.cos() * longitude.sin(), latitude.sin()), -1
    ).reshape(-1, 3)
    paths, diagnostics = planning.plan_grasp(
        state,
        target,
        [0.06, 0.06, 0.06],
        torch.zeros(6, 2),
        collision,
        None,
        0.02,
        object_support_points=points,
    )
    if closing_clearance < 0:
        assert paths is None
        assert diagnostics["failure"] == "no_collision_free_pocket_closing_pose"
        assert not calls
        return
    assert len(calls) == 1
    assert diagnostics["inverse_seed_count"] == 3
    assert diagnostics["inverse_samples"] == (
        len(diagnostics["body_height_hypotheses_m"])
        * 3
        * len(diagnostics["inverse_candidate_indices"])
        * (
            len(diagnostics["planning_path_indices"])
            + len(diagnostics["inverse_pregrasp_standoffs_m"])
        )
    )
    assert len(diagnostics["planning_path_indices"]) <= 29
    endpoints = torch.tensor(diagnostics["path_samples"]).cumsum(0) - 1
    assert set(endpoints.tolist()).issubset(diagnostics["planning_path_indices"])
    if not endpoint_exact:
        assert paths is None
        assert diagnostics["selected"] is None
        return
    assert diagnostics["selected"] is not None
    orientation = torch.tensor(diagnostics["candidate_orientations_world_wxyz"])
    center = torch.tensor(diagnostics["convex_contact_model"]["command_center_local"])
    expected = target[0] - planning._quat_rotate(
        orientation[0, diagnostics["selected"]], center[0, diagnostics["selected"]]
    )
    torch.testing.assert_close(paths["approach"][0][0, -1], expected)
    assert diagnostics["published_pose"] == "original_world_target_not_fixed_base_projection"
    assert diagnostics["required_object_lift_m"] == pytest.approx(0.10)
    assert diagnostics["lift_distance_m"] == pytest.approx(lift_distance + 0.005)
    lifted = paths["lift"][0][0, -1]
    assert lifted[2] >= 0.30
    torch.testing.assert_close(
        lifted[2], torch.maximum(expected[2] + lift_distance + 0.005, torch.tensor(0.305))
    )
    torch.testing.assert_close(paths["withdraw"][0][0, -1, 2], lifted[2])
    velocity = paths["lift"][0].diff(dim=1).norm(dim=-1) / 0.02
    assert velocity.max() <= planning.GRASP_LIFT_SPEED_M_S + 1e-5


def test_compacted_scene_queries_preserve_feasible_pocket_results(cpu_contact_profiles):
    quats = planning._euler_xyz_quat(
        torch.cartesian_prod(
            torch.tensor([0.0, math.pi]),
            torch.tensor([0.0, 0.8, 1.05, 1.25, 1.57, 1.8, 2.1, math.pi]),
            torch.tensor([0.0]),
        )
    )[None]
    target = torch.tensor([[0.4, 0, 0.2]])
    obj = torch.tensor([[1.0, 0, 0, 0]])
    points = torch.cartesian_prod(*[torch.tensor([-0.02, 0.02])] * 3)
    counts = []

    def clear(geometry, positions, orientations, *args, **options):
        counts.append(positions.numel() // 3)
        return positions[..., 2] + 0.20

    collision = SimpleNamespace(gripper_clearance=clear, gripper_closing_clearance=clear)
    dense_goals, dense = planning.pocket_pose_targets(
        quats, target, obj, points, collision, None, compact_environment_queries=False
    )
    dense_count = sum(counts)
    counts.clear()
    compact_goals, compact = planning.pocket_pose_targets(
        quats, target, obj, points, collision, None
    )
    assert 0 < sum(counts) < dense_count
    torch.testing.assert_close(compact_goals, dense_goals)
    for key in dense:
        mask = compact["environment_query_evaluated"] if key == "environment_clearance_m" else None
        if mask is None:
            torch.testing.assert_close(compact[key], dense[key])
        else:
            torch.testing.assert_close(compact[key][mask], dense[key][mask])
            assert torch.isnan(compact[key][~mask]).all()


@pytest.mark.parametrize("clearance", [0.002, 0.006])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("retain", [False, True])
def test_bounded_pocket_queries_keep_selected_targets_and_depths(
    cpu_contact_profiles, clearance, compact, retain
):
    quats = planning._euler_xyz_quat(torch.tensor([[0, 0.8, 0], [math.pi, 1.4, 0], [0, 1.57, 0]]))[
        None
    ].expand(3, -1, -1)
    target = torch.tensor([[0.4, 0, 0.02], [0.4, 0, 0.08], [0.4, 0, 0.2]])
    obj = torch.tensor([[1.0, 0, 0, 0]]).expand(3, -1)
    points = torch.cartesian_prod(*[torch.tensor([-0.02, 0.02])] * 3)
    calls = []

    def collision(uncapped):
        def clear(geometry, positions, orientations, *args, maximum_distance_m):
            calls.append(maximum_distance_m)
            distance = positions[..., 2] - 0.04
            return distance if uncapped else distance.clamp_max(maximum_distance_m)

        return SimpleNamespace(gripper_clearance=clear, gripper_closing_clearance=clear)

    options = dict(
        minimum_environment_clearance_m=clearance,
        compact_environment_queries=compact,
        retain_depth_candidates=retain,
    )
    old_goals, old = planning.pocket_pose_targets(
        quats, target, obj, points, collision(True), None, **options
    )
    new_goals, new = planning.pocket_pose_targets(
        quats, target, obj, points, collision(False), None, **options
    )
    assert calls and all(value == pytest.approx(clearance + 0.0001) for value in calls)
    torch.testing.assert_close(new_goals, old_goals)
    for name in old:
        expected = (
            old[name].clamp_max(clearance + 0.0001)
            if name == "environment_clearance_m"
            else old[name]
        )
        torch.testing.assert_close(new[name], expected, equal_nan=True)
    torch.testing.assert_close(
        new["environment_clearance_m"] >= clearance,
        old["environment_clearance_m"] >= clearance,
    )


@pytest.mark.parametrize(
    "shared,retain", [(False, False), (True, False), (False, True), (True, True)]
)
def test_pocket_candidate_chunks_preserve_all_points_depths_and_order(
    cpu_contact_profiles, monkeypatch, shared, retain
):
    quats = planning._euler_xyz_quat(
        torch.tensor([[0.0, pitch, 0.0] for pitch in (0.8, 1.1, 1.4, 1.7, 2.0)])
    )[None].expand(2, -1, -1)
    target = torch.tensor([[0.4, 0, 0.2], [0.4, 0.1, 0.22]])
    obj = torch.tensor([[1.0, 0, 0, 0]]).expand(2, -1)
    points = torch.cartesian_prod(*[torch.tensor([-0.02, 0.02])] * 3)
    preference = torch.linspace(0, 1, 5)[None]

    def clear(geometry, positions, orientations, *args, **options):
        return positions[..., 2] + 0.2

    collision = SimpleNamespace(gripper_clearance=clear, gripper_closing_clearance=clear)
    options = dict(
        depth_preference=preference,
        shared_object_geometry=shared,
        retain_depth_candidates=retain,
    )
    expected, expected_quality = planning.pocket_pose_targets(
        quats, target, obj, points, collision, None, **options
    )
    monkeypatch.setattr(
        planning, "MAX_POCKET_SAMPLE_EVALUATIONS", 2 * 17 * 8 * (1 if shared else 2)
    )
    actual, quality = planning.pocket_pose_targets(
        quats, target, obj, points, collision, None, **options
    )
    torch.testing.assert_close(actual, expected)
    for key in quality:
        torch.testing.assert_close(quality[key], expected_quality[key], equal_nan=True)


def test_retaining_depths_preserves_the_original_selected_pose(cpu_contact_profiles):
    quats = planning._euler_xyz_quat(torch.tensor([[0.0, 0.8, 0.0], [math.pi, 1.4, 0.0]]))[None]
    target = torch.tensor([[0.4, 0, 0.2]])
    obj = torch.tensor([[1.0, 0, 0, 0]])
    points = torch.cartesian_prod(*[torch.tensor([-0.02, 0.02])] * 3)

    def clear(geometry, positions, orientations, *args, **options):
        return positions[..., 2] + 0.2

    collision = SimpleNamespace(gripper_clearance=clear, gripper_closing_clearance=clear)
    selected_goals, selected = planning.pocket_pose_targets(
        quats, target, obj, points, collision, None
    )
    goals, quality = planning.pocket_pose_targets(
        quats, target, obj, points, collision, None, retain_depth_candidates=True
    )
    assert goals.shape == (1, 34, 3)
    depths = goals.reshape(1, 2, 17, 3)
    indices = (depths - selected_goals[:, :, None]).norm(dim=-1).argmin(-1)
    for key, expected in selected.items():
        values = quality[key].reshape(1, 2, 17, *expected.shape[2:])
        torch.testing.assert_close(values[0, torch.arange(2), indices[0]], expected[0])


def test_translated_targets_share_geometry_without_changing_any_selected_depth(
    cpu_contact_profiles,
):
    quats = planning._euler_xyz_quat(
        torch.cartesian_prod(
            torch.tensor([0.0, math.pi]), torch.tensor([0.8, 1.1, 1.4, 1.7]), torch.tensor([0.0])
        )
    )[None].expand(4, -1, -1)
    target = torch.tensor([[0.4, 0, 0.2], [0.4, 0.1, 0.22], [0.3, 0, 0.18], [0.4, -0.1, 0.3]])
    obj = torch.tensor([[1.0, 0, 0, 0]]).expand(4, -1)
    points = torch.cartesian_prod(*[torch.tensor([-0.02, 0.02])] * 3)

    def clear(geometry, positions, orientations, *args, **options):
        return positions[..., 2] + positions[..., 1] + 0.05

    collision = SimpleNamespace(gripper_clearance=clear, gripper_closing_clearance=clear)
    expected_goal, expected = planning.pocket_pose_targets(
        quats, target, obj, points, collision, None
    )
    actual_goal, actual = planning.pocket_pose_targets(
        quats, target, obj, points, collision, None, shared_object_geometry=True
    )
    torch.testing.assert_close(actual_goal, expected_goal)
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], equal_nan=True)


@pytest.mark.parametrize(
    "clearance,exact,valid",
    [
        (0.01, True, True),
        (0.0, True, False),
        (0.01, False, False),
        ("escape", True, True),
        ("toward_support", True, False),
        ("remain_near_support", True, False),
    ],
)
def test_carry_target_uses_seated_object_pose_with_single_batched_inverse(
    monkeypatch, clearance, exact, valid
):
    calls = []

    def inverse(positions, orientations, roots, seeds, limits):
        calls.append(positions.shape)
        size = positions.shape[0] * positions.shape[1]
        return SimpleNamespace(
            exact=torch.full((size,), exact, dtype=torch.bool),
            arm_joint_positions=torch.zeros(size, 6),
        )

    monkeypatch.setattr(planning, "inverse_bank", inverse)

    def gripper_clearance(geometry, p, q, aperture):
        if clearance == "escape":
            return 0.001 + 0.009 * (p[..., 2] - p[:, :1, 2]) / (p[:, -1:, 2] - p[:, :1, 2])
        if clearance == "toward_support":
            values = 0.001 + 0.009 * (p[..., 2] - p[:, :1, 2]) / (p[:, -1:, 2] - p[:, :1, 2])
            values[:, 1] = 0.0005
            return values
        if clearance == "remain_near_support":
            return torch.full(p.shape[:2], 0.001)
        return torch.full(p.shape[:2], clearance)

    state = {
        "ee_pos": torch.tensor([[0.5, 0, 0.110]]),
        "ee_orn": torch.tensor([[1.0, 0, 0, 0]]),
        "object_state": torch.tensor([[0.5, 0, 0.0244]]),
        "root_states": torch.tensor([[0.0, 0, 0.36, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        "dof_pos": torch.zeros(1, 6),
    }
    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6),
        clearance=lambda geometry, roots, joints, **kwargs: torch.ones(roots.shape[:-1]),
        gripper_clearance=lambda geometry, p, q: torch.ones(p.shape[:2]),
        gripper_pose_clearance=gripper_clearance,
    )
    path, diagnostics = planning.plan_carry_path(
        state,
        torch.zeros(1, 3),
        torch.tensor([0.0244]),
        torch.tensor([0.6]),
        torch.tensor([[-3.0, 3.0]] * 6),
        collision,
        None,
        0.02,
    )
    assert len(calls) == 1
    assert calls[0] == ((len(planning.GRASP_BODY_HEIGHTS_M) + 3) * 3, 17, 3)
    assert (path is not None) == valid
    if not valid:
        assert diagnostics["failure"] == "seated_carry_path_geometry_or_inverse_failed"
        return
    positions, quats, lengths = path
    torch.testing.assert_close(positions[:, 0], state["ee_pos"])
    torch.testing.assert_close(positions[:, -1, 2], torch.tensor([0.3906]))
    torch.testing.assert_close(quats[:, -1], state["ee_orn"])
    assert (positions.diff(dim=1).norm(dim=-1) / 0.02).max() <= 0.15 + 1e-5
    assert diagnostics["target_object_height_world_m"][0] == pytest.approx(0.305)
    assert int(lengths[0]) == positions.shape[1]


def test_held_gripper_collision_uses_only_current_aperture(monkeypatch):
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.pick.collision import ArmCollisionSamples

    collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
    fixed = torch.tensor([[0.1, 0, -0.01]])
    opened_moving = torch.tensor([[0.08, 0, 0.06]])
    monkeypatch.setattr(collision, "_gripper_local_surfaces", lambda _: (fixed, opened_moving))
    clouds = []

    def query(p, q, local):
        world = _quat_rotate(q[..., None, :], local) + p[..., None, :]
        clouds.append(world)
        return world[..., 2].amin(-1)

    positions = torch.tensor([[[0.0, 0, 1.0], [0.0, 0, 1.0]]])
    orientations = torch.tensor([[[1.0, 0, 0, 0], [1.0, 0, 0, 0]]])
    aperture = torch.tensor([[math.pi / 3, 0.0]])
    collision.gripper_pose_clearance(
        SimpleNamespace(
            rigid_surface_clearance=query,
            visible=lambda starts, ends, **kw: torch.ones(starts.shape[:-1], dtype=torch.bool),
        ),
        positions,
        orientations,
        aperture,
    )
    assert len(clouds) == 2
    torch.testing.assert_close(clouds[0], fixed + positions[..., None, :])
    torch.testing.assert_close(clouds[1][:, 0], opened_moving + positions[:, 0, None])
    angle = math.pi / 3
    expected_z = 1.0 - math.sin(angle) * 0.08 + math.cos(angle) * 0.06
    assert float(clouds[1][0, 1, 0, 2]) == pytest.approx(expected_z)


def fake_controller(monkeypatch, **options):
    from gr00t.rl.auto_open import pose_planning
    from gr00t.rl.pick import gaze

    monkeypatch.setattr(gaze, "solve_pose_targets", lambda p, q, **kwargs: inverse_result(p, q))

    monkeypatch.setattr(
        pose_planning, "solve_pose_targets", lambda p, q, **kwargs: inverse_result(p, q)
    )
    monkeypatch.setattr(controller, "read_arm_joint_limits", lambda path: np.array([[-3, 3]] * 6))
    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6),
        clearance=lambda geometry, roots, joints, **kwargs: torch.ones(roots.shape[:-1]),
    )
    monkeypatch.setattr(controller, "ArmCollisionSamples", lambda env: collision)
    root = torch.zeros(1, 13)
    root[:, 2], root[:, 3] = 0.536, 1
    object_state = root.clone()
    object_state[:, :3] = torch.tensor([0.65, 0, 0.03])
    state = {
        "root_states": root,
        "object_state": object_state,
        "dof_pos": torch.zeros(1, 6),
        "ee_pos": torch.tensor([[0.4, 0, 0.8]]),
        "ee_orn": torch.tensor([[1.0, 0, 0, 0]]),
        "object_finger_forces": torch.zeros(1, 2, 3),
    }
    env = SimpleNamespace(
        device="cpu",
        num_envs=1,
        arm_induced_pitch=0.0,
        dt=0.02,
        arm_joint_ids=torch.arange(6),
        gripper_joint_ids=torch.tensor([5]),
        get_control_state=lambda: state,
        get_goal_reference_origin=lambda: torch.zeros(1, 3),
        get_goal_reference_quat=lambda: torch.tensor([[1.0, 0, 0, 0]]),
        get_goal_center_offset_local=lambda: torch.tensor([[0.22, 0, 0.7]]),
        teleop_raw_commands=torch.zeros(1, 3),
        teleop_raw_ee_goal_cart=torch.zeros(1, 3),
        teleop_raw_ee_goal_orn_delta_rpy=torch.zeros(1, 3),
        gripper_pos_targets=torch.zeros(1, 1),
        teleop_world_ee_goal_orn_quat=torch.tensor([[1.0, 0, 0, 0]]),
        teleop_world_ee_goal_orn_quat_enabled=torch.zeros(1, dtype=torch.bool),
        teleop_arm_joint_pos_targets=torch.zeros(1, 6),
        teleop_arm_joint_control_mask=torch.ones(1, dtype=torch.bool),
    )
    episode = {
        "seed": 1700,
        "support_z": 0.0,
        "support_type": "floor",
        "object_position": [0.65, 0, 0.03],
        "base_yaw": 0,
        "station_xy": [0.1, 0],
        "station_yaw": 0,
        "asset": {"size_m": [0.06] * 3},
    }
    options.pop("defer_initial_station", None)
    options.pop("defer_approach_gaze", None)
    return PickController(env, episode, None, **options), state


def tensor_task(config, state):
    from gr00t.rl.pick.tensor_fsm import TensorPickFSM

    state.setdefault("contact_forces_w", torch.zeros(1, 0, 3))
    state.setdefault("object_finger_contact_points", torch.full((1, 2, 3), torch.nan))
    state.setdefault("distractor_states", torch.zeros(1, 0, 13))
    return TensorPickFSM(config.env, [config])


def command(task, state):
    buffers = TensorCommandBuffers.from_env(task.env)
    task.command(state, buffers)
    return buffers


def test_pick_approach_uses_real_shared_base_and_arm_execution(monkeypatch):
    config, state = fake_controller(monkeypatch)
    task = tensor_task(config, state)
    assert isinstance(task, StageExecutionMixin)
    buffers = command(task, state)
    assert isinstance(buffers, TensorCommandBuffers)
    assert torch.isfinite(buffers.base).all()
    assert not buffers.arm_joint_mode.any()
    assert task.base_reversal_pending_sign.dtype == torch.int8
    task.advance(state)
    assert int(task.stage[0]) == int(PickStage.APPROACH_OBJECT)


def test_grasp_approach_retiming_does_not_stop_at_internal_waypoint():
    x = torch.linspace(0, 0.4, 101)
    positions = torch.stack((x, x * 0, x * 0 + 0.4), -1)[None]
    orientations = torch.tensor([1.0, 0, 0, 0]).expand(1, len(x), 4)
    p, q = planning.retime_grasp_approach(positions, orientations, positions[:, -1], 0.02)
    speed = p.diff(dim=1).norm(dim=-1)[0] / 0.02
    middle = (p[0, 1:, 0] > 0.15) & (p[0, 1:, 0] < 0.25)
    assert speed[middle].min() > 0.1
    assert speed.max() <= planning.GRASP_TRANSIT_SPEED_M_S + 1e-5
    assert speed[0] < 0.001 and speed[-1] < 0.001
    torch.testing.assert_close(p[:, 0], positions[:, 0])
    torch.testing.assert_close(p[:, -1], positions[:, -1])
    torch.testing.assert_close(q, orientations[:, :1].expand_as(q))
    assert "ALIGN_GRIPPER" not in PickStage.__members__
    assert "INSERT_AROUND_OBJECT" not in PickStage.__members__


@pytest.mark.parametrize(
    "initial_height,final_height,lift_pass,height_pass",
    [(0.10, 0.21, True, False), (0.10, 0.305, True, True), (0.82, 0.91, False, True)],
)
def test_retention_requires_both_object_world_height_and_net_lift(
    monkeypatch, initial_height, final_height, lift_pass, height_pass
):
    task, state = fake_controller(monkeypatch)
    task.current_stage = PickStage.VERIFY_RETENTION
    task.stage.fill_(int(PickStage.VERIFY_RETENTION))
    task.initial_object_z.fill_(initial_height)
    state["object_state"][:, 2] = final_height
    task.grasp_object_relative = state["object_state"][:, :3] - state["ee_pos"]
    task = tensor_task(task, state)
    task.advance(state)
    assert (
        bool(task._acceptance[(int(PickStage.VERIFY_RETENTION), "object_lift_m")]["current"])
        == lift_pass
    )
    assert (
        bool(
            task._acceptance[(int(PickStage.VERIFY_RETENTION), "object_height_world_m")]["current"]
        )
        == height_pass
    )


def test_invalid_approach_gaze_fails_before_any_motion_command(monkeypatch):
    task, state = fake_controller(monkeypatch)
    task.approach_gaze_valid = False
    assert not task.approach_gaze_valid
    task = tensor_task(task, state)
    assert not command(task, state).base.any()
    task.advance(state)
    assert int(task.stage[0]) == int(PickStage.FAILURE)


def test_grasp_acceptance_allows_crouched_base_and_publishes_original_low_target(monkeypatch):
    task, state = fake_controller(monkeypatch)
    low = torch.tensor([[[0.6, 0, 0.11]]])
    quat = torch.tensor([[[1.0, 0, 0, 0]]])
    task.install_arm_path(low, quat)
    task.current_stage = PickStage.MOVE_TO_GRASP
    task.stage.fill_(int(PickStage.MOVE_TO_GRASP))
    state["root_states"][:, 2] = 0.22
    state["ee_pos"] = low[:, 0].clone()
    task = tensor_task(task, state)
    buffers = command(task, state)
    torch.testing.assert_close(buffers.ee_cart + torch.tensor([[0.22, 0, 0.7]]), low[:, 0])
    task.advance(state)
    assert int(task.stage[0]) != int(PickStage.FAILURE)
    assert task._acceptance[(int(PickStage.MOVE_TO_GRASP), "base_height_m")]["current"].all()


def test_contact_recency_tolerates_solver_zero_but_expires_real_contact_loss():
    window = TensorContactWindow(1, 2, "cpu", memory_frames=5)
    force = torch.tensor([[[3.0, 0, 0], [-3.0, 0, 0]]])
    points = torch.tensor([[[0.0, 0, -0.02], [0.0, 0, 0.02]]])
    recent, _, _ = window.update(force, points)
    assert recent.all()
    for _ in range(5):
        recent, _, _ = window.update(torch.zeros_like(force), torch.full_like(points, torch.nan))
        assert recent.all()
        torch.testing.assert_close(window.last_points, points)
    recent, _, age = window.update(torch.zeros_like(force), points)
    assert not recent.any()
    assert (age == 6).all()


def test_initial_visibility_requires_both_static_and_robot_clearance():
    from gr00t.rl.pick.initialization import initial_visibility

    root = torch.zeros(1, 13)
    root[:, 2], root[:, 3] = 0.55, 1
    state = {"root_states": root, "object_state": torch.tensor([[2.0, 0, 0.55]])}
    clear = SimpleNamespace(visible=lambda head, target: torch.ones(1, dtype=torch.bool))
    blocked = SimpleNamespace(visible=lambda head, target: torch.zeros(1, dtype=torch.bool))
    distance, heading, fov, self_clear, scene_clear = initial_visibility(state, clear, blocked)
    assert 1 < distance.item() < 3
    assert heading.item() > 0.5
    assert fov.all() and scene_clear.all()
    assert not self_clear.any()
    _, _, _, self_clear, scene_clear = initial_visibility(state, blocked, clear)
    assert self_clear.all() and not scene_clear.any()


def test_recording_preserves_duration_with_jitter_and_missing_camera_updates():
    from gr00t.rl.pick.recording import uniform_camera_streams

    raw = [0.0, 0.08, 0.16, 0.26, 0.34, 0.44, 0.52, 1.0]
    frames, times = uniform_camera_streams({"head": list(range(len(raw)))}, {"head": raw})
    assert len(frames["head"]) == len(times["head"])
    assert abs((times["head"][-1] - times["head"][0]) - (raw[-1] - raw[0])) <= 0.081
    np.testing.assert_allclose(np.diff(times["head"]), 0.08)
    assert frames["head"][0] == 0 and frames["head"][-1] == len(raw) - 1


def test_camera_read_explicitly_refreshes_render_only_sidecars():
    from gr00t.rl.pick.recording import read_due_camera_frames

    calls, published = [], []
    rig = {"name": "head"}

    def render(**kwargs):
        calls.append(kwargs)
        return {"head": [np.full((2, 2, 3), len(calls), dtype=np.uint8)]}

    manager = SimpleNamespace(
        enabled=True,
        due_rigs_at_sim_time=lambda time: [rig],
        render_images=render,
        mark_published_at_sim_time=lambda rig, time: published.append(time),
    )
    first = read_due_camera_frames(manager, 0.0)
    second = read_due_camera_frames(manager, 0.08)
    assert all(call["update"] for call in calls)
    assert (first["head"] != second["head"]).all()
    assert published == [0.0, 0.08]
    manager.due_rigs_at_sim_time = lambda time: []
    assert read_due_camera_frames(manager, 0.1) == {}
    assert len(calls) == 2


def test_shared_gripper_sensor_hook_preserves_door_default_and_supports_pick_override():
    path = (
        Path(__file__).parents[1]
        / "third_party/visual_wholebody_lowlevel/legged_gym/envs/manip_loco/manip_loco.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ManipLocoIsaacLab"
    )
    names = {"get_gripper_target_contact_sensors", "_update_gripper_actuator_targets"}
    methods = [
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    received, options = [], []

    def advance(*args, **kwargs):
        received.append(args[-1].clone())
        options.append(kwargs)
        return args[2], args[3]

    scope = {
        "torch": torch,
        "fixed_norm": lambda value: value.norm(dim=-1),
        "advance_gripper_actuator_targets": advance,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)

    def sensor(x):
        return SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=torch.tensor([[[[x, 0.0, 0.0]]]]))
        )

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        dof_pos=torch.zeros(1, 1),
        gripper_joint_ids=[0],
        gripper_pos_targets=torch.zeros(1, 1),
        gripper_actuator_targets=torch.zeros(1, 1),
        gripper_contact_hold_active=torch.zeros(1, dtype=torch.bool),
        handle_contact_sensors=[sensor(2.0), sensor(3.0)],
    )
    env.get_gripper_target_contact_sensors = lambda: scope["get_gripper_target_contact_sensors"](
        env
    )
    scope["_update_gripper_actuator_targets"](env)
    torch.testing.assert_close(received[-1], torch.tensor([[2.0, 3.0]]))
    assert options[-1] == {}
    env.get_gripper_target_contact_sensors = lambda: [sensor(4.0), sensor(5.0)]
    env.gripper_control_options = {"contact_force_threshold": 8.0, "contact_close_step": 0.002}
    scope["_update_gripper_actuator_targets"](env)
    torch.testing.assert_close(received[-1], torch.tensor([[4.0, 5.0]]))
    assert options[-1] == env.gripper_control_options


def test_grip_force_design_scales_with_weight_and_accounts_for_hinged_jaw_ejection():
    import math

    size = [0.06, 0.06, 0.055]
    force = planning.minimum_grip_normal_force(size, 0.1)
    angle = planning.grasp_aperture_angle(size) / 2
    uplift = 2 * force * (planning.GRASP_FRICTION * math.cos(angle) - math.sin(angle))
    assert abs(uplift - 2 * 0.1 * 9.81) < 1e-8
    assert force > planning.minimum_grip_normal_force(size, 0.04)


def test_fk_approach_takes_long_wrist_route_inside_joint_limits():
    root = torch.zeros(1, 13)
    root[:, 2], root[:, 3] = 0.52, 1
    initial = torch.tensor([-1.57, 2.2, -1.0, 0.2, 0.0, 2.1])
    endpoint = initial[None].clone()
    endpoint[:, -1] = -1.58
    p, q, joints, roots = planning.joint_fk_approach(root, initial, endpoint, root[:, 2], 252)
    assert (joints[..., -1].diff(dim=1) <= 0).all()
    assert joints[..., -1].max() <= initial[-1]
    torch.testing.assert_close(joints[0, -1], endpoint[0])
    adjacent = (q[:, 1:] * q[:, :-1]).sum(-1).abs().clamp_max(1)
    path_rotation = (2 * adjacent.acos()).sum()
    assert path_rotation > math.pi
    expected_p, expected_rotation, _ = planning.b2z1_gripper_pose_and_jacobian(
        roots[..., :3], roots[..., [4, 5, 6, 3]], joints
    )
    torch.testing.assert_close(p, expected_p)
    torch.testing.assert_close(q, planning._matrix_to_quaternion_wxyz(expected_rotation))


def test_object_pose_window_measures_motion_not_solver_reported_velocity():
    positions = torch.tensor([[[5.0, -3.0, 0.8]]]).expand(2, 15, 3).clone()
    positions[0, :, 0] += torch.arange(15) * 0.02 * 0.06
    speed = controller.pose_window_speed(positions, 0.02)
    torch.testing.assert_close(speed, torch.tensor([0.06, 0.0]), atol=1e-5, rtol=1e-4)
    assert speed[0] > 0.05 and speed[1] < 0.05


def test_joint_stability_window_requires_settled_actual_positions():
    window = TensorJointStabilityWindow(2, 1, "cpu")
    rows = torch.arange(2)
    for step in range(15):
        stable, span = window.update(torch.tensor([[-0.7], [-0.7 + 0.001 * step]]), rows)
    assert stable.tolist() == [True, False]
    torch.testing.assert_close(span[0], torch.tensor(0.0))
    assert span[1] > 0.01


def test_lifting_preserves_clamping_force_and_bounded_contact_preload(monkeypatch):
    task, state = fake_controller(monkeypatch)
    options = {
        "hold_contact_force_threshold": 11.0,
        "maintain_contact_preload": True,
        "bound_contact_preload": True,
        "hold_command_lead": 0.006,
        "prevent_reopening_while_closing": False,
    }
    task.env.gripper_control_options = options.copy()
    task = tensor_task(task, state)
    task.stage.fill_(int(PickStage.GRASP_OBJECT))
    task.transition(task.rows, PickStage.LIFT_OBJECT)
    command(task, state)
    assert task.env.gripper_control_options == options


@pytest.mark.parametrize("basket_task", [False, True])
def test_low_target_retry_prefers_realized_body_without_freezing_external_crouch(
    monkeypatch, basket_task
):
    from gr00t.rl.pick import tensor_planning

    task, state = fake_controller(monkeypatch)
    task.episode["task_type"] = "pick_basket" if basket_task else "pick_external"
    task.object_support_points = torch.zeros(8, 3)
    if basket_task:
        task.basket = SimpleNamespace(
            boxes=torch.tensor([[[0.0, 0, 0.2]] * 8]),
            floor_lower=torch.tensor([-2.0, -2, 0]),
            floor_upper=torch.tensor([2.0, 2, 0]),
        )
        task.lift_margin = 0.1
    task.current_stage = PickStage.MOVE_TO_GRASP
    task.stage.fill_(int(PickStage.MOVE_TO_GRASP))
    task.install_arm_path(torch.tensor([[[0.6, 0, 0.11]]]), state["ee_orn"][:, None])
    task.tried_grasp_orientations = state["ee_orn"].clone()
    task.planning_diagnostics["grasp_attempts"] = [{"selected": 0}]
    calls = []

    def plan(local, *args, **kwargs):
        calls.append(kwargs)
        path = (local["ee_pos"][:, None], local["ee_orn"][:, None], torch.tensor([1]))
        return (
            path,
            torch.tensor([True]),
            {
                "selected": torch.tensor([1]),
                "candidate_cost": torch.tensor([[1.0, 0.0]]),
                "entry_orientation_world_wxyz": local["ee_orn"],
                "withdraw_displacement_m": torch.zeros(1, 3),
                "required_object_lift_m": 0.10,
            },
        )

    inverse_calls = []

    def inverse(*args, **kwargs):
        inverse_calls.append(kwargs)
        return {}

    monkeypatch.setattr(tensor_planning, "plan_basket_grasp_batch", plan)
    monkeypatch.setattr(tensor_planning, "grasp_candidate_bank", lambda *a, **kw: None)
    monkeypatch.setattr(tensor_planning, "solve_default_grasp_inverse", inverse)
    monkeypatch.setattr(tensor_planning, "score_default_grasp_paths", plan)
    fsm = tensor_task(task, state)
    coordinator = tensor_planning.TensorPickPlanningCoordinator([task])
    geometry = SimpleNamespace(select=lambda rows: None)
    coordinator.carry_geometry = geometry
    coordinator.basket_geometry = SimpleNamespace(update=lambda state: geometry)
    for _ in range(60):
        coordinator.plan_events(fsm, state, fsm.advance(state))
        if calls:
            break
    assert fsm.grasp_retries.tolist() == [1]
    if basket_task:
        assert calls[0]["realized_body"].tolist() == [True]
        excluded, excluded_mask = calls[0]["excluded"], calls[0]["excluded_mask"]
    else:
        assert "realized_body" not in calls[0]
        excluded, excluded_mask = inverse_calls[0]["excluded"], inverse_calls[0]["excluded_mask"]
    assert calls[0]["prefer_realized_body"].all()
    assert calls[0].get("realized_body_tracking_weight", 0.0) == 0.0
    torch.testing.assert_close(excluded[excluded_mask], state["ee_orn"])
    assert fsm.tried_grasp_mask.sum(-1).tolist() == [2]
    coordinator.finish_diagnostics()
    assert (
        task.planning_diagnostics["grasp_attempts"][0]["execution_rejection"]["position_error_m"][0]
        > 0.012
    )


def test_jaw_distal_limits_follow_actual_moving_mesh_at_each_angle():
    from gr00t.rl.pick.pocket import _moving_vertices, gripper_pocket_geometry, jaw_distal_limits

    angle = torch.tensor([0.0, 0.45, math.pi / 3])
    points = _moving_vertices("cpu", angle.dtype)
    result = jaw_distal_limits(angle)
    torch.testing.assert_close(result[:, 0], gripper_pocket_geometry()["front_x"].expand(3))
    expected = (angle.cos()[:, None] * points[:, 0] - angle.sin()[:, None] * points[:, 2]).amax(-1)
    torch.testing.assert_close(result[:, 1], expected)
    assert (result[:, 1].max() - result[:, 1].min()) > 0.03


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_jaw_distal_reduction_matches_full_mesh_without_pose_vertex_expansion():
    from gr00t.rl.pick.pocket import _moving_vertices, jaw_distal_limits

    angles = torch.linspace(0, math.pi / 3, 129, device="cuda:0").reshape(3, 43)
    points = _moving_vertices("cuda:0", angles.dtype)
    expected = (
        angles.cos()[..., None] * points[:, 0] - angles.sin()[..., None] * points[:, 2]
    ).amax(-1)
    torch.testing.assert_close(jaw_distal_limits(angles)[..., 1], expected, atol=1e-7, rtol=1e-6)


def test_episode_wrist_audit_detects_backtracking_instead_of_shifting_it_between_stages():
    from gr00t.rl.pick.probe_workspace import wrist_route_metrics

    wrist = torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0, -1.0])
    result = wrist_route_metrics(wrist, torch.tensor([0, 2, 5]))
    assert result["phase_endpoint_travel_deg"].item() == pytest.approx(math.degrees(5), rel=1e-6)
    assert result["phase_endpoint_backtracking_deg"].item() == pytest.approx(
        math.degrees(4), rel=1e-6
    )
    monotonic = wrist_route_metrics(torch.tensor([2.7, 1.0, 0, -2.7]), torch.tensor([0, 3]))
    assert monotonic["phase_endpoint_backtracking_deg"].item() == pytest.approx(0.0)
    assert monotonic["episode_q6_span_deg"].item() == pytest.approx(math.degrees(5.4), rel=1e-6)


def test_gaze_wrist_route_accounts_for_the_next_grasp_without_wrapping_joint_stops():
    from gr00t.rl.pick.gaze import wrist_route_length

    endpoints = torch.zeros(1, 2, 6)
    endpoints[0, :, 5] = torch.tensor([2.06, -1.05])
    initial = torch.zeros(1, 6)
    initial[0, 5] = 0.05
    grasp = initial.clone()
    grasp[0, 5] = -1.48
    route = wrist_route_length(endpoints, initial, grasp)
    assert route[0, 1] < route[0, 0]
    grasp[0, 5] = 1.5
    route = wrist_route_length(endpoints, initial, grasp)
    assert route[0, 0] < route[0, 1]
    initial[0, 5], grasp[0, 5] = 2.5, -2.5
    endpoints[..., 5] = 2.4
    torch.testing.assert_close(
        wrist_route_length(endpoints, initial, grasp), torch.full((1, 2), 5.0)
    )


def test_inverse_seed_prefers_nearby_precise_branch_without_sacrificing_pose_accuracy():
    position = torch.tensor([[[[1e-8, 0.005]], [[5e-8, 0.01]], [[0.002, 0.002]]]])
    angle = torch.zeros_like(position)
    joints = torch.zeros(1, 3, 1, 2, 6)
    joints[0, 0, 0, 0, 5] = 2.5
    joints[0, 1, 0, 0, 5] = -0.4
    result = planning.continuous_inverse_seed(position, angle, joints, torch.zeros(6))
    assert result.shape == (1, 1, 1, 2)
    assert result.flatten().tolist() == [1, 2]


def test_arm_envelope_ignores_unexecuted_transit_and_longer_standoffs():
    indices = torch.tensor([0, 75, 150, 151, 200, 300])
    standoffs = torch.tensor([0.08, 0.12, 0.16, 0.20])
    selected = torch.tensor([0.08, 0.16])
    safe = torch.ones(2, 2, 10, dtype=torch.bool)
    safe[..., :3] = False
    safe[:, 0, 7:] = False
    safe[:, 1, 9] = False
    clear, mask = planning.executed_arm_path_clear(
        safe, indices, 151, standoffs, selected, torch.tensor([0.01, 0.01])
    )
    assert clear.tolist() == [True, True]
    assert mask[0].tolist() == [False, False, False, True, True, True, True, False, False, False]
    assert mask[1].tolist() == [False, False, False, True, True, True, True, True, True, False]
    safe[:, 0, 4] = False
    clear, _ = planning.executed_arm_path_clear(
        safe, indices, 151, standoffs, selected, torch.tensor([0.01, 0.001])
    )
    assert clear.tolist() == [False, False]


def test_robust_station_diagnostic_requires_each_body_posture_not_only_one():
    from gr00t.rl.pick.station import rank_station_candidates

    cost = torch.full((1, 2, 3, 2), torch.inf)
    cost[0, 0, :, 0] = 0.1
    cost[0, :, :, 1] = 0.2
    normal, _, _ = rank_station_candidates(cost, 1)
    assert normal.tolist() == [[True, True]]
    robust, _, report = rank_station_candidates(
        cost, 1, standing_postures=torch.tensor([True, False]), require_all_postures=True
    )
    assert robust.tolist() == [[False, True]]
    assert report["selection_policy"] == "all_postures_arrival"


def test_initial_station_bank_samples_position_and_heading_together():
    from gr00t.rl.pick.station import station_pose_bank

    root = torch.zeros(1, 13)
    root[:, 2:4] = torch.tensor([0.52, 1.0])
    destination = torch.tensor([[0.6, 0, 0.12]])
    stations, roots = station_pose_bank(root, destination, torch.tensor([[1.0, 0, 0.03]]))
    torch.testing.assert_close(stations[0, :, 0], torch.tensor([0.44] * 3 + [0.6] * 3 + [0.76] * 3))
    direction = torch.tensor([1.0, 0]) - stations[0, :, :2]
    torch.testing.assert_close(
        stations[0, :, 2], torch.atan2(direction[:, 1], direction[:, 0]) + 0.12
    )
    assert roots.shape == (1, 9, 8, 13)
    assert roots[..., 2].min() < 0.30
    torch.testing.assert_close(roots[..., :2], stations[:, :, None, :2].expand(-1, -1, 8, -1))


def test_station_yaw_search_preserves_positions_and_nominal_bank():
    from gr00t.rl.pick.station import station_pose_bank

    root = torch.zeros(1, 13)
    root[:, 2:4] = torch.tensor([0.52, 1.0])
    destination = torch.tensor([[0.6, 0, 0.12]])
    target = torch.tensor([[1.0, 0, 0.03]])
    nominal, nominal_roots = station_pose_bank(root, destination, target)
    stations, roots = station_pose_bank(root, destination, target, yaw_offsets_deg=(-15, 0, 15))
    torch.testing.assert_close(stations[:, 1::3], nominal)
    torch.testing.assert_close(roots[:, 1::3], nominal_roots)
    torch.testing.assert_close(stations[:, 0::3, :2], nominal[..., :2])
    torch.testing.assert_close(stations[:, 2::3, :2], nominal[..., :2])
    torch.testing.assert_close(stations[:, 0::3, 2], nominal[..., 2] - math.radians(15))
    torch.testing.assert_close(stations[:, 2::3, 2], nominal[..., 2] + math.radians(15))
    assert roots.shape == (1, 27, 8, 13)


@pytest.mark.parametrize("standing_available", [True, False])
@pytest.mark.parametrize("body_volume", [True, False])
def test_station_ranking_prefers_standing_and_otherwise_posture_support(
    standing_available, body_volume
):
    from gr00t.rl.pick.station import rank_station_candidates

    costs = torch.full((1, 3, 3, 2), torch.inf)
    costs[:, 0, :, 0] = 0.1
    costs[:, :2, :, 1] = 0.4
    if standing_available:
        costs[:, 2, :, 0] = 0.1
        costs[:, 2, :, 1] = 0.4
    eligible, worst, diagnostics = rank_station_candidates(
        costs, 1, standing_postures=torch.tensor([False, False, True]), body_volume=body_volume
    )
    selected = int(worst.masked_fill(~eligible, torch.inf).argmin())
    assert selected == (0 if standing_available else 1)
    assert diagnostics["selection_policy"] == (
        "standing_arrival" if standing_available else "posture_volume_without_standing_solution"
    )
    assert diagnostics["arrival_coverage_counts"][0, selected] == 3


def test_station_does_not_prefer_more_deep_crouch_samples_over_reachable_higher_stations():
    from gr00t.rl.pick.station import rank_station_candidates

    costs = torch.full((1, 3, 3, 2), torch.inf)
    costs[:, 0, :, 0] = 0.1
    costs[:, 1, 1, 0] = 0.2
    costs[:, 1, :, 1] = 0.3
    eligible, _, report = rank_station_candidates(
        costs,
        1,
        standing_postures=torch.tensor([False, False, True]),
        posture_heights=torch.tensor([0.30, 0.38, 0.536]),
    )
    assert eligible.tolist() == [[False, True]]
    assert report["body_arrival_support_counts"].tolist() == [[1, 3]]
    assert report["selection_policy"] == "highest_reachable_grasp_posture_then_volume"


def test_station_arrival_samples_use_the_real_base_acceptance_bounds():
    from gr00t.rl.auto_open.control import APPROACH_ARRIVAL
    from gr00t.rl.pick.station import station_arrival_hypotheses, station_pose_bank

    root = torch.zeros(1, 13)
    root[:, 2:4] = torch.tensor([0.52, 1.0])
    _, posture = station_pose_bank(
        root, torch.tensor([[0.6, 0, 0.12]]), torch.tensor([[1.0, 0, 0.03]])
    )
    roots, nominal = station_arrival_hypotheses(posture)
    assert roots.shape == (1, 9, 8, 15, 13)
    torch.testing.assert_close(roots[..., nominal, :], posture)
    distance = (roots[..., :2] - posture[..., None, :2]).norm(dim=-1)
    assert float(distance.max()) == pytest.approx(APPROACH_ARRIVAL.outer_distance_m)
    yaw = 2 * torch.atan2(roots[..., 6], roots[..., 3])
    nominal_yaw = 2 * torch.atan2(posture[..., 6], posture[..., 3])
    assert float((yaw - nominal_yaw[..., None]).abs().max()) == pytest.approx(
        APPROACH_ARRIVAL.outer_yaw_rad, abs=1e-6
    )


@pytest.mark.parametrize(
    "direct,prefer_large_wrist,realized,prefer_realized",
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
def test_grasp_preserves_the_matched_station_pose_before_changing_body_posture(
    monkeypatch, direct, prefer_large_wrist, realized, prefer_realized
):
    def bank(target, heading, size, origin, *args, **kwargs):
        assert not direct, "A supplied candidate bank must not be replaced"
        quat = target.new_tensor([[[math.cos(0.01), 0, 0, math.sin(0.01)]]])
        return target[:, None], quat, torch.ones_like(target[:, None])

    def pockets(
        quats,
        target,
        object_quat,
        support,
        collision,
        geometry,
        *,
        minimum_environment_clearance_m=0.002,
    ):
        assert minimum_environment_clearance_m == 0.002
        shape = quats.shape[:-1]
        return target[:, None].expand(*shape, 3), {
            "pocket_fit": torch.ones(shape, dtype=torch.bool),
            "antipodal_margin": torch.full(shape, 0.2),
            "environment_clearance_m": torch.full(shape, 0.02),
        }

    def solve(position, quaternion, **kwargs):
        entry_orientation = quaternion[:, 3].abs() < 1e-5
        height = kwargs["base_position_world"][:, 2]
        exact = torch.where(
            entry_orientation,
            (height - 0.526).abs() < 1e-5,
            torch.ones_like(height, dtype=torch.bool)
            if realized or prefer_realized
            else (height - 0.536).abs() < 1e-5,
        )
        result = inverse_result(position, quaternion, exact=exact)
        result.position_error[:, 0] = torch.where(
            exact,
            torch.where(entry_orientation, 0.0, 0.0 if realized or prefer_realized else 0.004),
            0.02,
        )
        if prefer_large_wrist:
            result.arm_joint_positions[:, 5] = torch.where(entry_orientation, 0.0, 2.0)
        return result

    monkeypatch.setattr(planning, "grasp_pose_bank", bank)
    monkeypatch.setattr(
        planning, "object_frame_grasp_bank", lambda orientation, size: orientation[:, None][:, :0]
    )
    monkeypatch.setattr(planning, "pocket_pose_targets", pockets)
    monkeypatch.setattr(planning, "solve_pose_targets", solve)
    root = torch.zeros(1, 13)
    root[:, 2:4] = torch.tensor([0.536, 1.0])
    target = torch.tensor([[0.65, 0, 0.03]])
    state = {
        "root_states": root,
        "dof_pos": torch.zeros(1, 6),
        "ee_pos": torch.tensor([[0.4, 0, 0.8]]),
        "ee_orn": torch.tensor([[1.0, 0, 0, 0]]),
        "object_state": torch.tensor([[0.65, 0, 0.03, 1, 0, 0, 0]]),
    }
    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6),
        clearance=lambda geometry, roots, joints, **kwargs: torch.ones(len(roots)),
        gripper_clearance=lambda geometry, positions, quats: torch.ones(positions.shape[:2]),
    )
    paths, diagnostics = planning.plan_grasp(
        state,
        target,
        [0.06] * 3,
        torch.tensor([[-3.0, 3.0]] * 6),
        collision,
        None,
        0.02,
        object_support_points=torch.zeros(8, 3),
        realized_body=realized,
        prefer_realized_body=prefer_realized,
        candidate_orientations=state["ee_orn"][:, None] if direct else None,
        preferred_orientation=(
            torch.tensor([[math.cos(0.01), 0, 0, math.sin(0.01)]]) if prefer_large_wrist else None
        ),
    )
    assert paths is not None
    assert diagnostics["standing_pregrasp_available"] == (not direct)
    expected_standing = (
        [False] if direct else ([True, False, True] if prefer_large_wrist else [False, True])
    )
    assert diagnostics["standing_pregrasp_candidate_mask"] == expected_standing
    selected = 1 if realized or prefer_realized else 0
    assert diagnostics["selected"] == selected
    assert diagnostics["feasible"][selected]
    assert diagnostics["preserved_entry_orientation"] == (selected == 0)
    if prefer_large_wrist:
        assert diagnostics["preferred_grasp_orientation_selected"]
        assert diagnostics["feasible"][0]
    if realized:
        assert diagnostics["realized_body_grasp_candidate_mask"] == [False, True]


def test_continuous_grasp_refinement_preserves_prefix_and_all_geometry_checks(monkeypatch):
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat

    quats = _euler_xyz_quat(torch.tensor([[0.0, 0, yaw] for yaw in (0.0, 0.5, 1.0, 0.0)]))[None]
    quats[:, -1] *= -1
    goals = torch.zeros(1, 4, 3)
    quality = {
        "pocket_fit": torch.ones(1, 4, dtype=torch.bool),
        "antipodal_margin": torch.full((1, 4), 0.2),
        "environment_clearance_m": torch.full((1, 4), 0.02),
    }
    calls = []

    def pockets(orientations, *args, **kwargs):
        calls.append(orientations)
        shape = orientations.shape[:2]
        fitted = torch.ones(shape, dtype=torch.bool)
        fitted[:, -1] = False
        return torch.ones(*shape, 3), {
            "pocket_fit": fitted,
            "antipodal_margin": torch.full(shape, 0.2),
            "environment_clearance_m": torch.full(shape, 0.02),
        }

    monkeypatch.setattr(planning, "pocket_pose_targets", pockets)
    result, positions, constraints = planning.refine_grasp_candidates(
        quats,
        goals,
        quality,
        quats[:, 0],
        torch.zeros(1, 3),
        quats[:, 0],
        torch.zeros(8, 3),
        None,
        None,
    )
    assert len(calls) == 1
    assert calls[0].shape == (1, 54, 4)
    torch.testing.assert_close(result[:, :4], quats)
    torch.testing.assert_close(positions[:, :4], goals)
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(1, 58))
    assert constraints["pocket_fit"][:, 4:].sum() == 27
    assert not constraints["pocket_fit"][0, -1]
    for key, value in quality.items():
        torch.testing.assert_close(constraints[key][:, :4], value)


def test_station_body_hypotheses_allow_measured_postures_without_changing_defaults():
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
    from gr00t.rl.pick.station import station_pose_bank

    root = torch.zeros(1, 13)
    root[:, 2:4] = torch.tensor([0.52, 1.0])
    destination = torch.tensor([[0.6, 0, 0.12]])
    target = torch.tensor([[1.0, 0, 0.03]])
    stations, defaults = station_pose_bank(root, destination, target)
    selected, measured = station_pose_bank(
        root, destination, target, body_heights=[0.35, 0.36], body_pitches_deg=[8, 11, 14]
    )
    torch.testing.assert_close(stations, selected)
    assert defaults.shape == (1, 9, 8, 13)
    assert measured.shape == (1, 9, 6, 13)
    torch.testing.assert_close(measured[0, 0, :, 2], torch.tensor([0.35] * 3 + [0.36] * 3))
    rpy = torch.tensor([[0, math.radians(11), float(stations[0, 0, 2])]])
    torch.testing.assert_close(measured[0, 0, 1, 3:7], _euler_xyz_quat(rpy)[0])


def test_policy_probe_crosses_every_station_with_unchanged_world_goals():
    from gr00t.rl.pick.probe_policy import command_bank

    trace = [
        {
            "ee_goal_cart_world": torch.tensor([[1.0, 2, 3]]),
            "ee_goal_orn_quat_world": torch.tensor([[1.0, 0, 0, 0]]),
        },
        {
            "ee_goal_cart_world": torch.tensor([[4.0, 5, 6]]),
            "ee_goal_orn_quat_world": torch.tensor([[0.0, 1, 0, 0]]),
        },
    ]
    stations = torch.tensor([[0.0, 1, 2], [3.0, 4, 5]])
    path = (torch.tensor([[[7.0, 8, 9]]]), torch.tensor([[[0.0, 0, 1, 0]]]))
    samples, goals, quats = command_bank(trace, path, stations, [1, 0])
    torch.testing.assert_close(samples, stations.repeat_interleave(3, 0))
    torch.testing.assert_close(goals[:3], torch.tensor([[4.0, 5, 6], [1.0, 2, 3], [7.0, 8, 9]]))
    torch.testing.assert_close(goals[3:], goals[:3])
    torch.testing.assert_close(quats[3:], quats[:3])


def test_policy_fk_descent_preserves_both_forward_kinematic_endpoints():
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
    from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
    from gr00t.rl.pick.probe_policy import fk_descent_pose

    start = torch.zeros(2, 13)
    start[:, 2:4] = torch.tensor([0.52, 1.0])
    end = start.clone()
    end[:, 2] = 0.36
    end[:, 3:7] = _euler_xyz_quat(torch.tensor([[0, 0.2, 0.4]]).expand(2, -1))
    q0 = torch.tensor([[-1.5, 2.2, -0.7, -0.8, 0, 0]]).expand(2, -1)
    q1 = q0 + 0.1
    for fraction, root, joints in ((0.0, start, q0), (1.0, end, q1)):
        positions, orientations = fk_descent_pose(start, end, q0, q1, fraction)
        expected, _, _ = b2z1_gripper_pose_and_jacobian(root[:, :3], root[:, [4, 5, 6, 3]], joints)
        torch.testing.assert_close(positions, expected)
        torch.testing.assert_close(orientations.norm(dim=-1), torch.ones(2))
        assert torch.isfinite(orientations).all()


def test_point_task_candidate_solver_preserves_wrist_and_joint_limits():
    from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian
    from gr00t.rl.pick.probe_pocket import pocket_position_solve

    roots = torch.tensor([[0.0, 0, 0.36, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(3, 1)
    reference = torch.tensor([[-1.5, 2.2, -0.7, -0.8, 0.0, 0.2]]).repeat(3, 1)
    offset = torch.tensor([[0.07, 0, 0.01]]).repeat(3, 1)
    position, rotation, _ = b2z1_gripper_pose_and_jacobian(
        roots[:, :3], roots[:, [4, 5, 6, 3]], reference
    )
    target = position + (rotation @ offset[..., None]).squeeze(-1)
    initial = reference.clone()
    initial[:, :5] += 0.03
    limits = torch.tensor([[-3.0, 3.0]] * 6)
    result = pocket_position_solve(roots, initial, offset, target, limits, iterations=40)
    position, rotation, _ = b2z1_gripper_pose_and_jacobian(
        roots[:, :3], roots[:, [4, 5, 6, 3]], result
    )
    torch.testing.assert_close(
        position + (rotation @ offset[..., None]).squeeze(-1), target, atol=1e-5, rtol=0
    )
    torch.testing.assert_close(result[:, 5], reference[:, 5], atol=0, rtol=0)
    assert ((result >= limits[:, 0]) & (result <= limits[:, 1])).all()


@pytest.mark.parametrize(
    "body_volume,independent_postures", [(False, False), (True, False), (True, True)]
)
def test_initial_station_planner_batches_all_endpoints_in_one_inverse(
    monkeypatch, body_volume, independent_postures
):
    from gr00t.rl.pick import station

    calls = []
    pose_bank_batches = []

    def bank(target, heading, size, origin, **kwargs):
        pose_bank_batches.append(len(target))
        quat = torch.zeros(len(target), 1, 4)
        quat[..., 0] = 1
        return target[:, None], quat, torch.ones_like(target[:, None])

    def pockets(quats, target, object_quat, support, collision, geometry):
        shape = quats.shape[:-1]
        return target[:, None].expand(*shape, 3), {
            "pocket_fit": torch.ones(shape, dtype=torch.bool),
            "antipodal_margin": torch.full(shape, 0.2),
            "environment_clearance_m": torch.full(shape, 0.02),
        }

    def solve(positions, quats, roots, joints, limits):
        calls.append(positions.shape)
        count = positions.shape[0] * positions.shape[1]
        exact = torch.ones(positions.shape[:2], dtype=torch.bool)
        if independent_postures:
            exact[:, 0] = roots[:, 2] <= 0.34
            exact[:, 1:] = roots[:, 2:3] >= 0.50
        return SimpleNamespace(
            exact=exact.flatten(),
            arm_joint_positions=torch.zeros(count, 6),
            position_error=torch.zeros(count, 3),
        )

    monkeypatch.setattr(station, "grasp_pose_bank", bank)
    monkeypatch.setattr(
        station, "object_frame_grasp_bank", lambda orientation, size: orientation[:, None][:, :0]
    )
    monkeypatch.setattr(station, "pocket_pose_targets", pockets)
    monkeypatch.setattr(station, "inverse_bank", solve)
    root = torch.zeros(1, 13)
    root[:, 2:4] = torch.tensor([0.52, 1.0])
    obj = torch.tensor([[1.0, 0, 0.03, 1, 0, 0, 0]])
    destination = torch.tensor([[0.6, 0, 0.12]])
    geometry = SimpleNamespace(
        visible=lambda origins, ends: torch.ones(origins.shape[:-1], dtype=torch.bool)
    )

    def arm_clearance(geometry, roots, joints, **kwargs):
        assert kwargs["include_gripper"] is False
        return torch.ones(len(roots))

    collision = SimpleNamespace(
        arm_joint_ids=torch.arange(6),
        clearance=arm_clearance,
        gripper_clearance=lambda geometry, positions, quats: torch.ones(positions.shape[:2]),
    )
    pose, quat, joint_reference, diagnostics = station.select_initial_pick_station(
        {
            "root_states": root,
            "object_state": obj,
            "dof_pos": torch.zeros(1, 6),
            "ee_orn": torch.tensor([[1.0, 0, 0, 0]]),
        },
        destination,
        [0.06] * 3,
        torch.zeros(8, 3),
        torch.tensor([[-3.0, 3.0]] * 6),
        collision,
        geometry,
        body_volume=body_volume,
    )
    assert len(calls) == 1
    assert pose_bank_batches == [1]
    assert calls[0] == (27 * 16 * 15 * 3, 3, 3)
    torch.testing.assert_close(pose[:, :2], destination[:, :2])
    yaw_options = destination[:, 2:3] + torch.tensor([[-15.0, 0.0, 15.0]]) * (math.pi / 180)
    assert (pose[:, 2:3] - yaw_options).abs().amin() < 1e-5
    assert diagnostics["selection_policy"] == (
        "highest_reachable_grasp_posture_then_volume"
        if independent_postures
        else "standing_arrival"
    )
    assert diagnostics["endpoint_posture_policy"] == "independent_grasp_lift_withdraw_postures"
    assert quat.shape == (1, 4)
    assert joint_reference.shape == (1, 6)
    assert diagnostics["inverse_samples"] == 27 * 16 * 15 * 3 * 3
    assert diagnostics["selected_arrival_coverage"] == 15
    assert diagnostics["posture_hypotheses_are_not_execution_guarantees"]


def test_small_object_depth_uses_closed_jaw_edges_not_average_open_projection(cpu_contact_profiles):
    from gr00t.rl.pick.grasp_quality import convex_grasp_quality
    from gr00t.rl.pick.pocket import gripper_pocket_geometry

    points = torch.cartesian_prod(*[torch.tensor([-0.015, 0.015])] * 3)
    orientation = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    result = convex_grasp_quality(
        orientation,
        orientation[:, 0],
        points,
        depth_clearance=lambda center, angle: (center[..., 0] >= 0.085).float() * 0.01,
    )
    assert result["pocket_fit"].all()
    assert (result["environment_clearance_m"] >= 0.002).all()
    assert (result["center_local"][..., 0] > gripper_pocket_geometry()["open_tip_midpoint_x"]).all()
    assert (result["center_depth_margin_m"] >= 0.0015).all()
    assert (result["contact_depth_margin_m"] >= 0.0015).all()


def test_convex_contacts_use_actual_pocket_surfaces_and_are_world_rotation_invariant(
    cpu_contact_profiles,
):
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
    from gr00t.rl.pick.grasp_quality import convex_grasp_quality
    from gr00t.rl.pick.pocket import gripper_pocket_geometry, jaw_distal_limits

    axis = torch.tensor([-0.025, 0.025])
    points = torch.cartesian_prod(axis, axis, axis)
    orientation = torch.tensor([[[1.0, 0, 0, 0]]])
    object_orientation = orientation[:, 0]
    result = convex_grasp_quality(orientation, object_orientation, points)
    pocket = gripper_pocket_geometry()
    fixed = result["fixed_contact_local"]
    height, _, valid = cpu_contact_profiles.sample("fixed", fixed[..., :2])
    assert valid.all()
    torch.testing.assert_close(fixed[..., 2], height, atol=0.0004, rtol=0)
    extent = points[None, None] + result["center_local"][:, :, None]
    assert extent[..., 0].amin() >= pocket["back_x"] + 0.00149
    assert (result["center_depth_margin_m"] >= 0.0015).all()
    distal = jaw_distal_limits(result["aperture_angle_rad"])
    assert (result["center_local"][..., 0] < distal.amin(-1)).all()
    assert result["pocket_fit"].all()
    moving = result["moving_contact_local"]
    angle = result["aperture_angle_rad"]
    moving_x = moving[..., 0] * angle.cos() + moving[..., 2] * angle.sin()
    moving_z = -moving[..., 0] * angle.sin() + moving[..., 2] * angle.cos()
    height, _, valid = cpu_contact_profiles.sample(
        "moving", torch.stack((moving_x, moving[..., 1]), -1)
    )
    assert valid.all()
    torch.testing.assert_close(moving_z, height, atol=0.0004, rtol=0)
    rotation = _euler_xyz_quat(torch.tensor([[0.0, 0, 0.7]]))
    rotated = convex_grasp_quality(rotation[:, None], rotation, points)
    torch.testing.assert_close(
        rotated["antipodal_margin"], result["antipodal_margin"], atol=1e-5, rtol=0
    )


def test_depth_preferences_keep_all_original_grasp_constraints(cpu_contact_profiles):
    from gr00t.rl.pick.grasp_quality import convex_grasp_quality

    points = torch.cartesian_prod(*[torch.tensor([-0.015, 0.015])] * 3)
    orientations = torch.tensor([[[1.0, 0, 0, 0]]]).expand(1, 3, 4)
    result = convex_grasp_quality(
        orientations,
        orientations[:, 0],
        points,
        depth_preference=torch.tensor([[0.0, 0.25, 0.5]]),
        depth_clearance=lambda center, angle: torch.ones_like(angle) * 0.01,
    )
    default = convex_grasp_quality(orientations[:, :1], orientations[:, 0], points)
    torch.testing.assert_close(result["center_local"][:, :1], default["center_local"])
    assert result["pocket_fit"].all()
    assert (result["antipodal_margin"] >= 0).all()
    assert (result["contact_depth_margin_m"] >= 0.0015).all()
    assert (result["center_depth_margin_m"] >= 0.0015).all()
    assert (result["environment_clearance_m"] >= 0.002).all()
    assert result["center_local"][0, 2, 0] > result["center_local"][0, 0, 0]


def test_object_too_large_for_pocket_is_rejected(cpu_contact_profiles):
    from gr00t.rl.pick.grasp_quality import convex_grasp_quality

    axis = torch.tensor([-0.08, 0.08])
    points = torch.cartesian_prod(axis, axis, axis)
    orientation = torch.tensor([[[1.0, 0, 0, 0]]])
    quality = convex_grasp_quality(orientation, orientation[:, 0], points)
    assert not quality["pocket_fit"].any()


def test_long_object_can_overhang_sides_when_contact_patches_fit(cpu_contact_profiles):
    from gr00t.rl.pick.grasp_quality import convex_grasp_quality

    points = torch.cartesian_prod(
        torch.tensor([-0.012, 0.012]),
        torch.linspace(-0.08, 0.08, 17),
        torch.tensor([-0.012, 0.012]),
    )
    orientation = torch.tensor([[[1.0, 0, 0, 0]]])
    quality = convex_grasp_quality(orientation, orientation[:, 0], points)
    assert quality["pocket_fit"].all()
    assert (quality["width_clearance_m"] >= 0.0015).all()
    assert (quality["antipodal_margin"] >= 0).all()


def test_closing_refinement_does_not_skip_an_earlier_finite_jaw_collision():
    from gr00t.rl.pick.grasp_quality import first_closing_contact

    def gap(angle):
        return torch.where((angle > 0.7) & (angle < 0.8), -0.1, angle - 0.2)

    angle = first_closing_contact(torch.tensor([1.0]), gap)
    torch.testing.assert_close(angle, torch.tensor([0.8]), atol=1e-5, rtol=0)


def test_finer_station_pitch_bank_preserves_every_coarse_pose():
    args = (
        torch.tensor([[0.65, 0, 0.03]]),
        torch.zeros(1),
        [0.06] * 3,
        torch.tensor([[0.22, 0, 0.7]]),
    )
    coarse = planning.grasp_pose_bank(*args)
    fine = planning.grasp_pose_bank(*args, pitch_step_deg=5.0)
    prefix = 16 * len(planning.BALANCED_GRASP_PITCH_OFFSETS_DEG)
    assert coarse[1].shape == (1, 656, 4)
    assert fine[1].shape == (1, 1680, 4)
    for original, refined in zip(coarse, fine):
        torch.testing.assert_close(refined[:, :prefix], original[:, :prefix])
        channels = original.shape[-1]
        selected = refined[:, prefix:].reshape(1, 8, 25, 8, channels)[:, :, ::3]
        torch.testing.assert_close(selected.reshape_as(original[:, prefix:]), original[:, prefix:])


def test_natural_grasp_bank_transports_current_roll_and_bounds_candidate_count():
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat

    target = torch.tensor([[0.8, 0, 0.7]])
    origin = torch.tensor([[0.22, 0, 0.7]])
    reference = _euler_xyz_quat(torch.tensor([[0.3, 0.0, 0.0]]))
    p, q, axis = planning.grasp_pose_bank(
        target, torch.zeros(1), [0.06] * 3, origin, reference_orientation=reference
    )
    prefix = 16 * len(planning.BALANCED_GRASP_PITCH_OFFSETS_DEG)
    assert q.shape == (1, 672, 4)
    torch.testing.assert_close(q[:, prefix], reference)
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(1, 672))
    rays = axis[:, prefix : prefix + 34 * 16].reshape(1, 34, 16, 3)
    torch.testing.assert_close(rays, rays[:, :, :1].expand_as(rays), atol=1e-6, rtol=0)
    torch.testing.assert_close(rays[:, 0, 0], torch.tensor([[1.0, 0, 0]]))
    torch.testing.assert_close(rays[:, -1, 0], torch.tensor([[0.0, 0, -1.0]]), atol=1e-6, rtol=0)
    assert p.shape == (1, 672, 3)
    torch.testing.assert_close(
        axis[:, -48:], torch.tensor([0.0, 0, -1]).expand(1, 48, 3), atol=1e-6, rtol=0
    )
    from gr00t.rl.auto_open.motion_math import _quat_multiply

    vertical = torch.cat((q[:, prefix + 33 * 16 : prefix + 34 * 16], q[:, -48:]), 1)
    inverse = vertical[:, :1].clone()
    inverse[..., 1:] *= -1
    relative = _quat_multiply(inverse, vertical)
    torch.testing.assert_close(relative[..., 2:], torch.zeros(1, 64, 2), atol=1e-6, rtol=0)
    angle = (2 * torch.atan2(relative[..., 1], relative[..., 0])).remainder(2 * math.pi)
    angle = torch.where(angle > 2 * math.pi - 1e-5, 0.0, angle)
    expected = torch.arange(64) * (2 * math.pi / 64)
    torch.testing.assert_close(angle.sort(-1).values[0], expected, atol=1e-5, rtol=0)


def test_natural_grasp_bank_is_equivariant_under_world_yaw():
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate

    target = torch.tensor([[0.8, 0.2, 0.03]])
    origin = torch.tensor([[0.22, 0, 0.7]])
    heading = torch.tensor([0.2])
    reference = _euler_xyz_quat(torch.tensor([[0.3, 0.5, 0.1]]))
    rotation = _euler_xyz_quat(torch.tensor([[0.0, 0.0, 0.7]]))
    p, q, axis = planning.grasp_pose_bank(
        target, heading, [0.06] * 3, origin, reference_orientation=reference
    )
    p2, q2, axis2 = planning.grasp_pose_bank(
        _quat_rotate(rotation, target),
        heading + 0.7,
        [0.06] * 3,
        _quat_rotate(rotation, origin),
        reference_orientation=_quat_multiply(rotation, reference),
    )
    torch.testing.assert_close(p2, _quat_rotate(rotation[:, None], p), atol=1e-6, rtol=0)
    torch.testing.assert_close(axis2, _quat_rotate(rotation[:, None], axis), atol=1e-6, rtol=0)
    expected = _quat_multiply(rotation[:, None], q)
    torch.testing.assert_close((expected * q2).sum(-1).abs(), torch.ones(1, 672), atol=1e-6, rtol=0)


def test_grasp_direction_changes_continuously_with_target_elevation():
    targets = torch.tensor([[0.8, 0, 0.03], [0.8, 0, 0.031], [0.8, 0, 0.80]])
    origins = torch.tensor([[0.0, 0, 0.75]]).expand(3, -1)
    _, _, axes = planning.grasp_pose_bank(targets, torch.zeros(3), [0.06] * 3, origins)
    reference = targets - origins
    reference /= reference.norm(dim=-1, keepdim=True)
    index = 16 * len(planning.BALANCED_GRASP_PITCH_OFFSETS_DEG) + 32
    torch.testing.assert_close(axes[:, index], reference, atol=1e-6, rtol=0)
    assert (axes[0, index] - axes[1, index]).norm() < 0.002
    assert axes[0, index, 2] < -0.5
    assert axes[2, index, 2] > 0


def test_balanced_pitch_bank_preserves_original_candidates_and_adds_small_tilts():
    targets = torch.tensor([[0.8, 0, 0.82]])
    origins = torch.tensor([[0.0, 0, 0.75]])
    args = (targets, torch.zeros(1), [0.06] * 3, origins)
    original = planning.grasp_pose_bank(*args, balanced_pitch_offsets_deg=(0.0,))
    expanded = planning.grasp_pose_bank(*args)
    for old, new in zip(original, expanded):
        torch.testing.assert_close(old[:, :16], new[:, :16])
        torch.testing.assert_close(old[:, 16:], new[:, 80:])
    for index, degrees in enumerate(planning.BALANCED_GRASP_PITCH_OFFSETS_DEG):
        cosine = (expanded[2][:, index * 16 : (index + 1) * 16] * original[2][:, :16]).sum(-1)
        torch.testing.assert_close(cosine, torch.full_like(cosine, math.cos(math.radians(degrees))))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arm_clearance_matches_point_cloud_and_has_bounded_batch_memory():
    from gr00t.rl.pick.collision import ArmCollisionSamples
    from gr00t.rl.pick.scene_geometry import SceneGeometry

    vertices = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], dtype=np.float32)
    geometry = SceneGeometry(vertices, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    collision = ArmCollisionSamples.__new__(ArmCollisionSamples)
    generator = torch.Generator(device="cuda:0").manual_seed(2301)
    collision.parts = [
        (index, torch.rand((11, 3), device="cuda:0", generator=generator) * 0.2)
        for index in (0, 2, 6)
    ]
    roots = torch.zeros(12, 13, device="cuda:0")
    roots[:, 2], roots[:, 3] = 0.5, 1
    joints = torch.rand((12, 6), device="cuda:0", generator=generator) - 0.5
    expected = geometry.distances(collision.world_points(roots, joints)).amin(-1)
    torch.testing.assert_close(
        collision.clearance(geometry, roots, joints), expected, atol=3e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        collision.clearance(geometry, roots, joints, maximum_distance_m=0.0021),
        expected.clamp_max(0.0021),
        atol=3e-6,
        rtol=1e-5,
    )
    collision.parts = [(0, torch.rand((8192, 3), device="cuda:0", generator=generator) * 0.05)]
    large_roots = roots[:1].expand(8192, -1).contiguous()
    large_joints = joints[:1].expand(8192, -1).contiguous()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = collision.clearance(geometry, large_roots, large_joints, maximum_distance_m=0.0021)
    torch.cuda.synchronize()
    assert result.shape == (8192,)
    assert torch.cuda.max_memory_allocated() - before < 128 * 1024**2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU geometry query")
def test_fused_rigid_clearance_matches_expanded_point_queries():
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat
    from gr00t.rl.auto_open.planning_geometry import _quat_rotate
    from gr00t.rl.pick.scene_geometry import SceneGeometry

    points = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], dtype=np.float32)
    geometry = SceneGeometry(points, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    positions = torch.tensor([[0.0, 0, 0.2], [0.0, 0, 0.1], [0.1, 0, 0.3]], device="cuda:0")
    orientations = _euler_xyz_quat(
        torch.tensor([[0.0, 0, 0], [0.0, 0, 0], [0.0, 0.7, 0.4]], device="cuda:0")
    )
    local = torch.tensor([[0.0, 0, -0.15], [0.1, 0.05, -0.01], [-0.06, 0, 0.02]], device="cuda:0")
    world = _quat_rotate(orientations[:, None], local) + positions[:, None]
    expanded = geometry.distances(world).amin(-1)
    visible = geometry.visible(positions[:, None].expand_as(world), world, tolerance_m=0.0005).all(
        -1
    )
    expected = torch.where(visible, expanded, -1.0)
    actual = geometry.rigid_surface_clearance(positions, orientations, local)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)


def test_synthetic_reach_coverage_is_asset_independent_and_shard_invariant():
    from gr00t.rl.pick.probe_policy import synthetic_reach_bank

    before = torch.random.get_rng_state()
    full = synthetic_reach_bank(32)
    first, second = synthetic_reach_bank(11), synthetic_reach_bank(21, offset=11)
    for whole, left, right in zip(full, first, second):
        torch.testing.assert_close(whole, torch.cat((left, right)))
    assert torch.equal(before, torch.random.get_rng_state())
    stations, positions, quaternions, jitter = full
    assert not stations.any()
    assert (positions >= positions.new_tensor([0.35, -0.30, 0.08])).all()
    assert (positions <= positions.new_tensor([0.85, 0.30, 0.30])).all()
    torch.testing.assert_close(quaternions.norm(dim=-1), torch.ones(32))
    assert jitter.abs().max() <= 0.05
