"""Shared planner math and FSM installation without a simulator."""

import ast
import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.auto_open.pregrasp_reachability import pregrasp_reachability_samples
from gr00t.rl.auto_open.tensor_fsm import PaddedPathBank
from gr00t.rl.auto_open.tensor_stage_planning import (
    GPUStagePlanningMixin, align_vectors, center_vertical_shift,
    path_samples, pregrasp_pose_family, pregrasp_pitch_constants, retract_pose_samples,
    panel_forward_anchor_distance,
    panel_handle_height_mask,
)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_empty_pull_open_request_does_not_require_pull_configuration(device):
    from gr00t.rl.auto_open.tensor_planning import TensorPlanningCoordinator

    planner = TensorPlanningCoordinator.__new__(TensorPlanningCoordinator)
    planner.plan_pull_open(SimpleNamespace(), {}, torch.empty(0, dtype=torch.int64, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_panel_candidates_keep_entire_jaws_clear_of_handle(device):
    height = torch.tensor([[.80, 1.00, 1.05, 1.10]], device=device, dtype=torch.float64)
    jaw = torch.tensor([[-.06, .04]], device=device, dtype=torch.float64)
    handle = torch.tensor([[.93, .96]], device=device, dtype=torch.float64)
    # The EE at 1.00 m is above the handle, but its lower jaw still overlaps.
    valid = panel_handle_height_mask(height, jaw, handle)
    assert valid.tolist() == [[True, False, True, True]]
    shifted = panel_handle_height_mask(height + 8., jaw, handle + 8.)
    assert torch.equal(valid, shifted)
    batched = panel_handle_height_mask(height.repeat(3, 1), jaw.repeat(3, 1), handle.repeat(3, 1))
    assert torch.equal(batched, valid.repeat(3, 1))


@pytest.mark.parametrize("device", DEVICES)
def test_panel_handle_height_margin_and_current_handle_band(device):
    height = torch.tensor([[1.00, 1.00]], device=device, dtype=torch.float64)
    jaw = torch.tensor([[-.04, .04]], device=device, dtype=torch.float64)
    band = torch.tensor([[.90, .955]], device=device, dtype=torch.float64)
    assert not panel_handle_height_mask(height, jaw, band).any()
    assert panel_handle_height_mask(height, jaw, band, clearance_m=0.).all()
    lower_handle = band - .10
    assert panel_handle_height_mask(height, jaw, lower_handle).all()


@pytest.mark.parametrize("device", DEVICES)
def test_panel_forward_anchor_uses_actual_yaw_and_is_batch_independent(device):
    from gr00t.rl.auto_open.tensor_fsm import _euler_xyz_quat

    angles = torch.tensor([[.2, -.3, 0.], [-.1, .4, math.pi / 2]],
                          device=device, dtype=torch.float64)
    root = torch.zeros((2, 13), device=device, dtype=torch.float64)
    root[:, :3] = torch.tensor([[1., 2., .5], [3., 4., .6]], device=device)
    root[:, 3:7] = _euler_xyz_quat(angles)
    hinge = torch.tensor([[.5, 2., 0.], [3., 3.5, 0.]], device=device, dtype=root.dtype)
    tangent = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device, dtype=root.dtype)
    distance = panel_forward_anchor_distance(root, hinge, tangent, .2)
    torch.testing.assert_close(distance, torch.full_like(distance, .7))
    for i in range(2):
        torch.testing.assert_close(distance[i:i+1], panel_forward_anchor_distance(
            root[i:i+1], hinge[i:i+1], tangent[i:i+1], .2), rtol=0, atol=0)
    shift = torch.tensor([5., -7., 0.], device=device, dtype=root.dtype)
    root[:, :3] += shift
    torch.testing.assert_close(distance, panel_forward_anchor_distance(root, hinge + shift, tangent, .2))
    torch.testing.assert_close(-distance, panel_forward_anchor_distance(root, hinge + shift, -tangent, .2))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("hinge_sign", [-1, 1])
def test_pregrasp_family_matches_reference_all_stages(device, hinge_sign):
    dtype = torch.float64
    inputs = dict(
        pregrasp_position_world=[.65, .12, .7], pregrasp_orientation_world=[1., 0, 0, 0],
        grasp_position_world=[.56, .12, .7], grasp_orientation_world=[1., 0, 0, 0],
        handle_axle_world=[.48, .12, .72], handle_axis_world=[0., 1, 0],
        hinge_world=[.1, -.3, .7], hinge_rotation_rad=hinge_sign * .12,
        handle_start_angle_rad=.13, handle_target_angle_rad=.78)
    reference = pregrasp_reachability_samples(**inputs)
    values = [torch.tensor(value, device=device, dtype=dtype)[None] for value in inputs.values()]
    position, orientation, owner = pregrasp_pose_family(*values)
    np.testing.assert_allclose(position.cpu(), reference["reachability_positions"], atol=2.e-7)
    np.testing.assert_allclose(orientation.cpu(), reference["reachability_orientations"], atol=2.e-7)
    assert torch.equal(owner, torch.zeros_like(owner))


@pytest.mark.parametrize("device", DEVICES)
def test_pose_family_ragged_candidates_and_antipodal_slerp(device):
    p = torch.tensor([[.5, 0, .7], [.5, .1, .8]], device=device, dtype=torch.float64)
    q = torch.tensor([[1., 0, 0, 0], [-1., 0, 0, 0]], device=device, dtype=torch.float64)
    axis = torch.tensor([[0., 1, 0]], device=device, dtype=torch.float64).expand(2, -1)
    scalar = torch.tensor([.1, .2], device=device, dtype=torch.float64)
    pos, quat, owner = pregrasp_pose_family(p, q, p + .01, -q, p, axis, p - .3,
                                           scalar, scalar * 0, scalar + .6)
    assert torch.isfinite(pos).all() and torch.isfinite(quat).all()
    assert torch.all(owner[1:] >= owner[:-1])
    torch.testing.assert_close(torch.linalg.vector_norm(quat, dim=-1), torch.ones_like(owner, dtype=quat.dtype))


@pytest.mark.parametrize("device", DEVICES)
def test_path_sampling_preserves_start_rules_and_lengths(device):
    p = torch.zeros((2, 3), device=device, dtype=torch.float64)
    end = p.clone()
    end[:, 0] = torch.tensor([.01, .014], device=device, dtype=p.dtype)
    q = torch.tensor([[1., 0, 0, 0]], device=device, dtype=torch.float64).expand(2, -1)
    pos, quat, lengths = path_samples(p, end, q, -q, 50)
    assert lengths.tolist() == [5, 7]
    torch.testing.assert_close(pos[:, 0, 0], torch.full((2,), .002, device=device, dtype=pos.dtype))
    torch.testing.assert_close(pos[torch.arange(2, device=device), lengths - 1], end)
    assert torch.isfinite(quat).all()
    pos, _, lengths = path_samples(p, end, q, q, 50, steps=torch.tensor([35, 49], device=device))
    assert lengths.tolist() == [36, 50]
    torch.testing.assert_close(pos[:, 0], p)


@pytest.mark.parametrize("device", DEVICES)
def test_center_shift_thresholds_are_unchanged(device):
    error = torch.tensor([.007, .015, -.016, .03, -.03], device=device)
    gap = torch.tensor([0., .015, .015, 0., .019], device=device)
    shift, feasible = center_vertical_shift(error, gap)
    torch.testing.assert_close(shift, torch.tensor([0., -.004, .004, -.019, 0.], device=device))
    assert feasible.tolist() == [True, False, False, False, False]


@pytest.mark.parametrize("device", DEVICES)
def test_retract_draws_bounds_uniform_volume_and_batch_invariance(device):
    base = torch.zeros((2, 6), device=device)
    radius = torch.tensor([.1, .2], device=device)
    bound = torch.tensor([.04, .08], device=device)
    seeds, entries = torch.tensor([12, 39], device=device), torch.tensor([0, 1], device=device)
    samples = retract_pose_samples(base, radius, bound, seeds, entries, 0, 4096)
    norms = torch.linalg.vector_norm(samples[..., :3], dim=-1)
    assert (norms <= radius[:, None]).all()
    assert (samples[..., 3:].abs() <= bound[:, None, None]).all()
    torch.testing.assert_close((norms / radius[:, None]).pow(3).mean(-1),
                               torch.full((2,), .5, device=device, dtype=torch.float64), atol=.025, rtol=0)
    subset = retract_pose_samples(base[1:], radius[1:], bound[1:], seeds[1:], entries[1:], 64)
    torch.testing.assert_close(subset, samples[1:, 64:128], rtol=0, atol=0)
    assert not torch.equal(samples[0], samples[1])


@pytest.mark.parametrize("device", DEVICES)
def test_alignment_opposite_parallel_and_oblique_match_reference(device):
    from gr00t.rl.auto_open.geometry import _quat_align_vectors_np
    source = torch.tensor([[1., 0, 0], [0., 1, 0], [1., 1, 0]], device=device)
    target = torch.tensor([[-1., 0, 0], [0., 1, 0], [0., 0, 1]], device=device)
    q = torch.tensor([[1., 0, 0, 0]], device=device).expand(3, -1)
    actual = align_vectors(source, target, q)
    expected = np.stack([_quat_align_vectors_np(s, t, b) for s, t, b in
                         zip(source.cpu().numpy().copy(), target.cpu().numpy().copy(), q.cpu().numpy())])
    np.testing.assert_allclose(actual.cpu(), expected, atol=2.e-7)


class Planner(GPUStagePlanningMixin):
    def __init__(self, device):
        self.path_diagnostics = []
        self.calls = []
        self.gpu_stage_constants = {"release_delta": torch.zeros((3, 3), device=device),
                                    "retreat_delta": torch.zeros((3, 3), device=device)}

    def _record_stats(self, label, requests, errors=None):
        self.calls.append((label, requests, errors))

    def _inverse_pose_bank(self, fsm, state, rows, position, orientation, lengths):
        owner, sample = torch.where(torch.arange(position.shape[1], device=fsm.device)[None] < lengths[:, None])
        return SimpleNamespace(command_position_world=position[owner, sample] + .001,
                               command_orientation_world_wxyz=orientation[owner, sample],
                               position_error=torch.zeros((len(owner), 3), device=fsm.device),
                               orientation_error=torch.zeros((len(owner), 3), device=fsm.device)), owner, sample


def make_fsm(device):
    stage = SimpleNamespace(OPEN_AND_CLEAR=1, RETREAT_ARM=2, RESTORE_GRIPPER=3, PULL_ARM_RETRACT=4)
    fsm = SimpleNamespace(device=device, Stage=stage, stage=torch.zeros(3, device=device, dtype=torch.int64),
                          stage_steps=torch.zeros(3, device=device, dtype=torch.int64),
                          release_stable_steps=torch.ones(3, device=device),
                          release_vertical_delta=torch.tensor([.01, .02, .03], device=device),
                          door_normal=torch.tensor([[1., 0, 0]], device=device).expand(3, -1),
                          release_retreat_target_world=torch.zeros((3, 3), device=device))
    fsm.failed = []
    fsm.fail = lambda rows, code: fsm.failed.append((rows.clone(), code))
    fsm.transition = lambda rows, stage: fsm.stage.index_fill_(0, rows, stage)
    for name in ("release_open_path", "release_retreat_path", "release_restore_path"):
        setattr(fsm, name, PaddedPathBank(3, 512, device))
    state = {"ee_pos": torch.zeros((3, 3), device=device),
             "ee_orn": torch.tensor([[1., 0, 0, 0]], device=device).expand(3, -1)}
    return fsm, state


@pytest.mark.parametrize("device", DEVICES)
def test_pull_push_retraction_reuses_entry_inverse_restore(device):
    planner = Planner(device)
    fsm, state = make_fsm(device)
    fsm.pull_door = True
    rows = torch.tensor([2, 0], device=device)
    calls = []
    def restore(selected_fsm, selected_state, selected_rows):
        assert selected_fsm is fsm and selected_state is state
        assert (fsm.stage[selected_rows] == fsm.Stage.PULL_ARM_RETRACT).all()
        calls.append(selected_rows.clone())
    planner.plan_restore = restore
    planner._sample_gpu_retract = lambda *args: pytest.fail('double retract sampling')
    planner.plan_push_retract(fsm, state, rows)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], rows)


@pytest.mark.parametrize("device", DEVICES)
def test_release_and_retreat_install_tensor_banks_without_legacy_installers(device, monkeypatch):
    planner = Planner(device)
    fsm, state = make_fsm(device)
    rows = torch.tensor([2, 0], device=device)
    monkeypatch.setattr(PaddedPathBank, "install", lambda *args: pytest.fail("legacy per-env install"), raising=False)
    planner.plan_release_open(fsm, state, rows)
    assert fsm.release_open_path.length.tolist() == [36, 0, 36]
    torch.testing.assert_close(fsm.release_open_path.positions[rows, 35, 2], fsm.release_vertical_delta[rows] + .001)
    assert fsm.stage.tolist() == [1, 0, 1]
    planner.plan_retreat(fsm, state, rows)
    torch.testing.assert_close(fsm.release_retreat_target_world[rows],
                               torch.tensor([[.1, 0, .02]], device=device).expand(2, -1))
    assert fsm.stage.tolist() == [2, 0, 2]
    assert fsm.release_retreat_path.length[rows].min() >= 51


@pytest.mark.parametrize("device", DEVICES)
def test_path_capacity_failure_does_not_truncate(device):
    planner = Planner(device)
    fsm, state = make_fsm(device)
    fsm.release_open_path = PaddedPathBank(3, 10, device)
    planner.plan_release_open(fsm, state, torch.tensor([1], device=device))
    assert fsm.release_open_path.length.sum() == 0
    assert any(code == 44 and rows.tolist() == [1] for rows, code in fsm.failed)


def test_runtime_has_no_host_materialization_or_controller_access():
    tree = ast.parse(inspect.getsource(GPUStagePlanningMixin))
    for method in tree.body[0].body:
        if not isinstance(method, ast.FunctionDef) or method.name in {"initialize_gpu_stage_constants", "finalize_gpu_stage_diagnostics"}:
            continue
        for node in ast.walk(method):
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"cpu", "numpy", "tolist", "item", "controllers"}, method.name


@pytest.mark.parametrize("device", DEVICES)
def test_pitch_constants_match_offline_pitch_and_noise_oracles(device):
    from gr00t.rl.auto_open.control import _effective_grasp_target, _perturb_ee_target
    from gr00t.rl.auto_open.planning import _pregrasp_pitch_candidates
    from gr00t.rl.auto_open.target_noise import EpisodeTargetNoise, SemanticTargetNoiseConfig

    controller = SimpleNamespace(
        grasp_orientation_quat=np.array([1., 0, 0, 0], dtype=np.float32),
        center_standoff_ee_world=np.array([.6, 0, .7], dtype=np.float32), fixed_surface_target_z=.73,
        fixed_inner_center_local=np.array([.03, 0, .01], dtype=np.float32),
        grasp_center_pos=np.array([.5, 0, .7], dtype=np.float32),
        grasp_ee_offset_local=np.array([.02, 0, 0], dtype=np.float32),
        handle_free_axis_world=np.array([0., 1, 0], dtype=np.float32),
        handle_upper_z=.71, handle_lower_z=.69, handle_half_length=.08,
        open_template=dict(depth_axis=np.array([1., 0, 0]), separation_axis=np.array([0., 0, 1.]),
            fixed_points=np.array([[.02, -.01, .01], [.04, .01, .01]]),
            moving_points=np.array([[.02, -.01, -.01], [.04, .01, -.01]])),
        target_noise=EpisodeTargetNoise(137, SemanticTargetNoiseConfig(grasp_depth_m=.003,
            grasp_handle_axis_m=.002, grasp_closing_m=.001, grasp_roll_rad=.02)))
    candidates = _pregrasp_pitch_candidates(controller)
    expected = [_perturb_ee_target(controller, "pregrasp", item["end_position"], item["orientation"]) for item in candidates]
    sample = controller.target_noise._samples["pose:grasp_chain"]
    cache = {}
    for key, value in dict(horizontal_orientation=controller.grasp_orientation_quat,
        center_standoff=controller.center_standoff_ee_world, fixed_surface_target_z=controller.fixed_surface_target_z,
        fixed_inner_center_local=controller.fixed_inner_center_local, grasp_center_pos=controller.grasp_center_pos,
        grasp_ee_offset_local=controller.grasp_ee_offset_local, handle_free_axis_world=controller.handle_free_axis_world,
        outer_half_length=.09, depth_axis=controller.open_template["depth_axis"],
        separation_axis=controller.open_template["separation_axis"],
        jaw_points=np.concatenate((controller.open_template["fixed_points"], controller.open_template["moving_points"])),
        grasp_position_noise=sample["semantic_position_delta"], grasp_rotation_noise=sample["semantic_rotation_delta"]).items():
        cache[key] = torch.tensor(np.asarray(value), dtype=torch.float32, device=device)[None]
    result = pregrasp_pitch_constants(cache, torch.tensor([0., 5., 10., 15., 20.], device=device))
    np.testing.assert_allclose(result["nominal"].cpu()[0], np.stack([item["end_position"] for item in candidates]), atol=1.e-7)
    np.testing.assert_allclose(result["nominal_q"].cpu()[0], np.stack([item["orientation"] for item in candidates]), atol=1.e-7)
    np.testing.assert_allclose(result["position"].cpu()[0], np.stack([item[0] for item in expected]), atol=1.e-7)
    np.testing.assert_allclose(result["quat"].cpu()[0], np.stack([item[1] for item in expected]), atol=1.e-7)
    expected_grasp = []
    for item in candidates:
        grasp = _effective_grasp_target(controller, controller.grasp_center_pos, item["orientation"])
        grasp[2] = item["end_position"][2]
        expected_grasp.append(_perturb_ee_target(controller, "center_handle", grasp, item["orientation"])[0])
    np.testing.assert_allclose(result["grasp"].cpu()[0], np.stack(expected_grasp), atol=1.e-7)
