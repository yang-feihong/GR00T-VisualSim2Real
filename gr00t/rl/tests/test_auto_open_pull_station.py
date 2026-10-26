import math
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.control import APPROACH_ARRIVAL
from gr00t.rl.auto_open.base_trajectory import BasePoseTrajectory
from gr00t.rl.auto_open.pull_station import (
    PullStationConfig, base_projection_clearances,
    robot_motion_envelope,
    pull_station_parameter_grid,
    angle_to_wall,
    shoulder_wall_curve,
)
from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM
from gr00t.rl.auto_open.target_noise import EpisodeTargetNoise
from gr00t.rl.scripts.doorman_auto_open import Stage


def _base_points():
    return torch.tensor([[x, y, z] for x in (-.284, .28)
                         for y in (-.1475, .1475) for z in (-.1275, .1275)])


def test_contact_baseline_defaults_to_front_minus_ten_cm_and_rejects_unknown():
    assert PullStationConfig().panel_contact_baseline == 'trunk_front_minus10'
    assert PullStationConfig().shoulder_hinge_radius_m == (.4, .6)
    for mode in ('base_forward', 'trunk_front', 'trunk_front_minus10'):
        assert PullStationConfig(panel_contact_baseline=mode).panel_contact_baseline == mode
    with pytest.raises(ValueError, match='baseline'):
        PullStationConfig(panel_contact_baseline='unknown')


@pytest.mark.parametrize('mirror', [-1., 1.])
@pytest.mark.parametrize('radius', [.5, .65])
def test_shoulder_curve_uses_arcsin_rule_and_switches_to_constant_radius(mirror, radius):
    anchor = torch.tensor([[.3, mirror * 1.6]])
    local = torch.tensor([[.3, mirror * .15]])
    normal = torch.tensor([[1., 0.]])
    free = torch.tensor([[0., mirror]])
    entry_yaw = torch.tensor([-mirror * math.radians(110.)])
    width = torch.tensor([.30])
    curve = shoulder_wall_curve(anchor, local, torch.zeros(1, 2), free, normal,
                                entry_yaw, width, shoulder_hinge_radius_m=radius)
    yaw = curve[..., 2]
    offset = torch.stack((yaw.cos() * local[:, :1] - yaw.sin() * local[:, 1:2],
                          yaw.sin() * local[:, :1] + yaw.cos() * local[:, 1:2]), -1)
    shoulder = curve[..., :2] + offset
    turning_y = math.sqrt(radius**2 - .3**2)
    straight = mirror * shoulder[..., 1] >= turning_y
    torch.testing.assert_close(shoulder[..., 0][straight], torch.full_like(shoulder[..., 0][straight], .3))
    distances = torch.linalg.vector_norm(shoulder, dim=-1)
    torch.testing.assert_close(distances[~straight], torch.full_like(distances[~straight], radius))
    assert (shoulder[..., 0].diff(dim=1) >= -1.e-6).all()
    assert (mirror * shoulder[..., 1].diff(dim=1) <= 1.e-6).all()
    ab = -shoulder[:, 32:]
    a = torch.linalg.vector_norm(ab, dim=-1)
    alpha = torch.atan2(ab[..., 0].abs(), ab[..., 1].abs())
    lower = angle_to_wall(entry_yaw, free, normal)
    expected = (torch.asin((.50 / a).clamp(max=1.)) + alpha).clamp(max=math.pi / 2)
    expected = torch.maximum(expected, lower[:, None])
    actual = angle_to_wall(yaw[:, 32:], free[:, None], normal[:, None])
    torch.testing.assert_close(actual, expected)
    assert (actual.diff(dim=1) >= -1.e-6).all()
    torch.testing.assert_close(actual[a < .50], torch.full_like(actual[a < .50], math.pi / 2))
    torch.testing.assert_close(shoulder[:, -1, 1], torch.zeros(1), atol=1.e-6, rtol=0)
    torch.testing.assert_close(shoulder[:, -1, 0], torch.tensor([radius]))


def test_shoulder_curve_accepts_per_environment_radii_without_batch_dependence():
    anchors = torch.tensor([[.3, 1.6], [.3, -1.6]])
    local = torch.zeros(2, 2)
    hinge = torch.zeros(2, 2)
    tangent = torch.tensor([[0., 1.], [0., -1.]])
    normal = torch.tensor([[1., 0.], [1., 0.]])
    yaw = torch.tensor([-math.radians(110.), math.radians(110.)])
    width = torch.full((2,), .3)
    radii = torch.tensor([.4, .6])
    batch = shoulder_wall_curve(anchors, local, hinge, tangent, normal, yaw, width,
                                shoulder_hinge_radius_m=radii)
    torch.testing.assert_close(batch[:, -1, 0], radii)
    for i in range(2):
        single = shoulder_wall_curve(anchors[i:i+1], local[i:i+1], hinge[i:i+1],
            tangent[i:i+1], normal[i:i+1], yaw[i:i+1], width[i:i+1],
            shoulder_hinge_radius_m=radii[i:i+1])
        torch.testing.assert_close(batch[i:i+1], single, rtol=0., atol=0.)


@pytest.mark.parametrize('side', [-1., 1.])
def test_fixed_station_parameter_grid(side):
    hinge = torch.tensor([[3., -2.]])
    normal = torch.tensor([[1., 0.]])
    tangent = torch.tensor([[0., side]])
    positions, yaw, config = pull_station_parameter_grid(hinge, normal, tangent, torch.tensor([1.2]))
    assert positions.shape == (1, 9, 2)
    delta = positions - hinge[:, None]
    torch.testing.assert_close(torch.linalg.vector_norm(delta, dim=-1), torch.full((1, 9), 1.97))
    position_angle = torch.rad2deg(torch.atan2(delta[..., 0], side * delta[..., 1]))
    torch.testing.assert_close(position_angle[0], config[:, 0])
    forward = torch.stack((yaw.cos(), yaw.sin()), -1)
    heading_angle = torch.rad2deg(torch.atan2(-forward[..., 0], -side * forward[..., 1]))
    torch.testing.assert_close(heading_angle[0], config.sum(-1))


def test_motion_envelope_includes_articulated_limbs_and_is_origin_independent():
    root = torch.tensor([[0., 0., .5, 1., 0., 0., 0.], [8., -8., .5, 1., 0., 0., 0.]])
    bodies = root[:, None].repeat(1, 2, 1)
    bodies[:, 1, :3] += torch.tensor([-.4, .25, -.3])
    points = torch.tensor([[[.48, .1475, 0.]], [[-.2, .03, 0.]]])
    result = robot_motion_envelope(root, bodies, points, torch.tensor([0, 1]))
    torch.testing.assert_close(result[0], result[1], atol=1e-6, rtol=0.)
    torch.testing.assert_close(result[0, ..., 0].amin(), torch.tensor(-.6))
    torch.testing.assert_close(result[0, ..., 1].amax(), torch.tensor(.28))


def test_motion_envelope_preserves_body_tilt_in_level_yaw_reference():
    angle = math.radians(20.)
    root = torch.tensor([[0., 0., .5, math.cos(angle / 2.), 0., math.sin(angle / 2.), 0.]])
    result = robot_motion_envelope(root, root[:, None], torch.tensor([[[0., 0., .6]]]), torch.tensor([0]))
    torch.testing.assert_close(result[0, 0, 0], torch.tensor([.6 * math.sin(angle), 0., .6 * math.cos(angle)]))



def test_frame_clearance_checks_nearest_trunk_corner_not_base_center():
    hinge = torch.zeros(1, 2)
    axes = torch.tensor([[[0., 1.], [1., 0.]]])
    points = torch.tensor([[[3., 1.05], [3., .95]]])
    clearance = base_projection_clearances(hinge, axes, torch.tensor([[1., 0.]]), points)
    assert clearance[0, 0].item() == pytest.approx(-.05)
    assert clearance[0, 1].item() == pytest.approx(3.)


def test_two_axis_projection_is_identical_for_candidates_and_actual_poses():
    origin = torch.tensor([[2., -3.]])
    axes = torch.tensor([[[0., 1.], [1., 0.]]])
    boundaries = torch.tensor([[1., .1]])
    points = torch.tensor([[[[2.4, -1.8], [2.6, -1.5]],
                            [[2.05, -1.7], [2.3, -1.3]]]])
    batch = base_projection_clearances(origin, axes, boundaries, points)
    for i in range(2):
        torch.testing.assert_close(batch[:, i], base_projection_clearances(
            origin, axes, boundaries, points[:, i]), rtol=0, atol=0)
    assert batch[0, 1, 0] > 0
    assert batch[0, 1, 1] < 0


def test_pull_randomization_is_per_seed_and_does_not_shift_other_samples():
    def sample(seed):
        noise = EpisodeTargetNoise(seed)
        return noise.perturb_base('pull_clear_sweep', (0., 0.), 0., (1., 0.),
                                  normal_m=0., tangent_m=.1, yaw_rad=.06)
    a, b = sample(1700), sample(1701)
    assert a[0].tolist() == sample(1700)[0].tolist()
    assert a[0].tolist() != b[0].tolist()
    assert a[0][0] == 0. and abs(a[0][1]) <= .1 and abs(a[1]) <= .06


def test_pull_command_uses_shared_tracker_with_same_command_and_acceptance_goal():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage, fsm.pull_door, fsm.device = Stage, True, 'cpu'
    fsm.stage = torch.tensor([int(Stage.PULL_CLEAR_SWEEP), int(Stage.APPROACH)])
    fsm.pull_station_xy = torch.tensor([[.7, 2.], [0., 0.]])
    fsm.pull_station_yaw = torch.tensor([-2., 0.])
    fsm.pull_base_trajectory = BasePoseTrajectory(2, 'cpu')
    fsm.pull_base_trajectory.length[:] = 1
    fsm.pull_base_trajectory.poses[:, 0] = torch.cat((fsm.pull_station_xy, fsm.pull_station_yaw[:, None]), -1)
    fsm.pending_station_distance = torch.zeros(2)
    fsm.pending_yaw_error = torch.zeros(2)
    fsm.base_hold_target_xy = torch.zeros(2, 2)
    fsm.base_hold_nominal_xy = torch.zeros(2, 2)
    fsm.base_hold_target_yaw = torch.zeros(2)
    fsm.base_hold_nominal_yaw = torch.zeros(2)
    fsm.align_panel_arm_joint_target = torch.arange(12.).reshape(2, 6)
    fsm.q_closed = torch.zeros(2)
    calls = []

    def track(state, rows, command_xy, command_yaw, acceptance_xy, acceptance_yaw, *, rolling=None, reference_velocity_world=None, arrival_enabled=None):
        assert rolling is None or not rolling.any()
        assert not reference_velocity_world.any()
        assert arrival_enabled.all()
        assert torch.equal(command_xy, acceptance_xy)
        assert torch.equal(command_yaw, acceptance_yaw)
        calls.append(rows.tolist())
        return torch.tensor([[0., .15, .2]]), torch.tensor([.5]), torch.tensor([.2])

    fsm._track_station = track
    buffers = SimpleNamespace(base=torch.zeros(2, 3), arm_joint_mode=torch.zeros(2, dtype=torch.bool),
                              arm_joint=torch.zeros(2, 6), gripper=torch.ones(2, 1))
    fsm.command_pull_station({}, buffers)
    assert calls == [[0]]
    assert torch.equal(buffers.arm_joint[0], fsm.align_panel_arm_joint_target[0])
    assert buffers.arm_joint_mode.tolist() == [True, False]
    assert buffers.gripper[:, 0].tolist() == [0., 1.]


def _clearance_fsm():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage, fsm.pull_door, fsm.device = Stage, True, torch.device('cpu')
    fsm.terminal_stage = Stage.PULL_CLEAR_SWEEP
    fsm.pull_base_trajectory = BasePoseTrajectory(1, 'cpu')
    fsm.pull_base_trajectory.length[:] = 1
    fsm.pull_base_path_index = torch.ones(1, dtype=torch.int64)
    fsm.pull_base_path_length = torch.full((1,), 2, dtype=torch.int64)
    fsm.stage = torch.tensor([int(Stage.PULL_CLEAR_SWEEP)])
    fsm.stage_steps = torch.ones(1, dtype=torch.int32)
    fsm.hinge_world = torch.zeros(1, 3)
    fsm.pull_frame_edge = torch.tensor([1.])
    fsm.pull_wall_front = torch.tensor([-1.])
    fsm.door_normal = torch.tensor([[0., 1., 0.]])
    fsm.pull_free_direction = torch.tensor([[1., 0.]])
    fsm.pull_base_points = _base_points()
    fsm.pull_station_config = PullStationConfig()
    fsm.pull_frame_clear_steps = torch.zeros(1, dtype=torch.int32)
    fsm.base_arrival_confirmed = torch.ones(1, dtype=torch.bool)
    fsm.base_stability_confirmed = torch.ones(1, dtype=torch.bool)
    fsm.base_arrival_settle_steps = torch.full((1,), 25, dtype=torch.int32)
    fsm.base_stability_steps = fsm.base_arrival_settle_steps.clone()
    fsm.base_arrival_required_frames = fsm.base_arrival_settle_steps.clone()
    fsm.record_acceptance = lambda *args, **kwargs: None
    return fsm


def test_completion_requires_continuous_whole_robot_clearance_and_shared_stability():
    fsm = _clearance_fsm()
    state = {'root_states': torch.tensor([[1.5, 0., .5, 1., 0., 0., 0.]])}
    for _ in range(24):
        assert not len(fsm.advance_pull_clear_sweep(state))
    assert fsm.advance_pull_clear_sweep(state).tolist() == [0]
    assert fsm.stage.tolist() == [int(Stage.SUCCESS)]


def test_clearance_window_resets_and_cannot_override_base_stability():
    fsm = _clearance_fsm()
    fsm.pull_frame_clear_steps[:] = 24
    unsafe = {'root_states': torch.tensor([[1.1, 0., .5, 1., 0., 0., 0.]])}
    assert not len(fsm.advance_pull_clear_sweep(unsafe))
    assert fsm.pull_frame_clear_steps.item() == 0
    fsm.base_stability_confirmed[:] = False
    safe = {'root_states': torch.tensor([[1.5, 0., .5, 1., 0., 0., 0.]])}
    for _ in range(30):
        assert not len(fsm.advance_pull_clear_sweep(safe))


def test_wall_overlap_blocks_completion_even_when_frame_projection_is_clear():
    fsm = _clearance_fsm()
    fsm.pull_wall_front[:] = 0.
    fsm.pull_frame_clear_steps[:] = 24
    state = {'root_states': torch.tensor([[1.5, 0., .5, 1., 0., 0., 0.]])}
    assert not len(fsm.advance_pull_clear_sweep(state))
    assert fsm.pull_frame_clear_steps.item() == 0


def test_push_does_not_run_pull_geometry_or_tracker():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.pull_door, fsm.device = False, 'cpu'
    fsm.command_pull_station({}, None)
    assert not len(fsm.advance_pull_clear_sweep({}))


def test_collision_bounds_use_physical_cube_size_and_nested_mesh_transform():
    pytest.importorskip('pxr')
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    from gr00t.rl.auto_open.pull_station import collider_local_corners

    stage = Usd.Stage.CreateInMemory()
    cube = UsdGeom.Cube.Define(stage, '/Cube')
    cube.CreateSizeAttr(1.)
    cube.CreateExtentAttr([(-.01, -.02, -.03), (.01, .02, .03)])
    assert np.allclose(np.abs(collider_local_corners(cube.GetPrim())), .5)
    group = UsdGeom.Xform.Define(stage, '/Group')
    mesh = UsdGeom.Mesh.Define(stage, '/Group/Mesh')
    mesh.CreatePointsAttr([(-1., -2., -3.), (1., 2., 3.)])
    mesh.AddTranslateOp().Set(Gf.Vec3d(4., 0., 0.))
    corners = collider_local_corners(group.GetPrim())
    assert np.allclose(corners.min(axis=0), (3., -2., -3.))
    assert np.allclose(corners.max(axis=0), (5., 2., 3.))
    capsule = UsdGeom.Capsule.Define(stage, '/Capsule')
    capsule.CreateAxisAttr('X')
    capsule.CreateHeightAttr(2.)
    capsule.CreateRadiusAttr(.2)
    assert np.allclose(collider_local_corners(capsule.GetPrim()).max(axis=0), (1.2, .2, .2))


def test_frame_bounds_follow_hinge_body0_not_panel_frames(monkeypatch):
    pytest.importorskip('pxr')
    import sys
    from types import ModuleType
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from gr00t.rl.auto_open.pull_station import frame_boundaries

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim('/Door', 'Xform')
    root = stage.DefinePrim('/Door/root_right', 'Xform')
    joint = UsdPhysics.RevoluteJoint.Define(stage, '/Door/root_right/hinge_joint')
    joint.CreateBody0Rel().SetTargets([root.GetPath()])
    for name, y in [('top_frame', 0.), ('left_frame', 1.), ('right_frame', -1.)]:
        cube = UsdGeom.Cube.Define(stage, '/Door/root_right/' + name)
        cube.CreateSizeAttr(2.)
        cube.AddTranslateOp().Set(Gf.Vec3d(-.02, y, 1.))
        cube.AddScaleOp().Set(Gf.Vec3f(.05, .5, 1.))
    UsdGeom.Cube.Define(stage, '/Door/door_panel/left_frame').CreateSizeAttr(100.)
    omni, usd = ModuleType('omni'), ModuleType('omni.usd')
    usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    omni.usd = usd
    monkeypatch.setitem(sys.modules, 'omni', omni)
    monkeypatch.setitem(sys.modules, 'omni.usd', usd)
    for side, expected_wall in [(1., .01), (-1., .09)]:
        controller = SimpleNamespace(_door_prim_path=lambda: '/Door',
                                     hinge_world=[.02, -.5, 0.],
                                     panel_hinge_to_free_direction=[0., 1., 0.],
                                     door_normal=[side, 0., 0.])
        edge, wall = frame_boundaries(controller)
        assert edge == pytest.approx(1.)
        assert wall == pytest.approx(expected_wall)


def test_restore_routes_pull_to_clear_sweep_without_another_arm_inverse():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage, fsm.pull_door = Stage, True
    fsm.terminal_stage, fsm.device = Stage.PULL_CLEAR_SWEEP, torch.device('cpu')
    fsm.stage = torch.tensor([int(Stage.RESTORE_GRIPPER)])
    fsm.stage_steps = torch.tensor([50], dtype=torch.int32)
    fsm.q_closed = torch.zeros(1)
    fsm.release_endpoint_world = torch.zeros(1, 3)
    fsm.release_endpoint_quat = torch.tensor([[1., 0., 0., 0.]])
    fsm.release_stable_steps = torch.tensor([9], dtype=torch.int32)
    fsm.arm_stability = SimpleNamespace(update=lambda state, rows: (
        torch.ones(len(rows), dtype=torch.bool), torch.zeros(len(rows)), torch.zeros(len(rows))))
    fsm.begin_pull_clear_sweep = lambda state, rows: fsm.transition(rows, Stage.PULL_CLEAR_SWEEP)
    completed = fsm.advance_restore({
        'gripper_position': torch.zeros(1, 1), 'ee_pos': torch.zeros(1, 3),
        'ee_orn': fsm.release_endpoint_quat.clone(),
    })
    assert completed.tolist() == [0]
    assert fsm.stage.item() == int(Stage.PULL_CLEAR_SWEEP)
