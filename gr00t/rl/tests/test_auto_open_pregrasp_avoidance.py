import math
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.pull_entry_gpu import PullEntryGPU
from gr00t.rl.auto_open.tensor_fsm import PaddedPathBank
from gr00t.rl.auto_open.tensor_stage_planning import (
    GPUStagePlanningMixin, pregrasp_avoidance_waypoints,
)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('side', [-1., 1.])
def test_pregrasp_separates_height_then_withdraws_then_reaches_original_goal(device, side):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    start = torch.tensor([[side * .01, .1, z] for z in (1.02, .98, 1.2, .8, 1.)],
                         device=device, dtype=torch.float64)
    q = start.new_tensor([[1., 0., 0., 0.]]).expand(5, -1)
    bodies = start.new_zeros(5, 2, 13)
    bodies[:, :, :3] = start[:, None]
    bodies[:, :, 3:7] = q[:, None]
    door = start.new_zeros(5, 1, 13)
    door[:, 0, 2], door[:, 0, 3] = 1., 1.
    state = dict(ee_pos=start, ee_orn=q, robot_body_state_w=bodies, door_body_state_w=door)
    geometry = dict(fixed_body_idx=0, moving_body_idx=1, handle_body_idx=0,
        fixed_points=start.new_tensor([[0., 0., -.04], [.03, 0., .04]]),
        moving_points=start.new_tensor([[0., .01, -.03], [.02, .01, .03]]),
        handle_center_local=start.new_zeros(5, 3),
        handle_axis_local=start.new_tensor([[0., 1., 0.]]).expand(5, -1),
        handle_half_length=start.new_full((5,), .06), handle_radius=start.new_full((5,), .01))
    normal = start.new_tensor([[side, 0., 0.]]).expand(5, -1)
    end = start.new_tensor([[side * .2, .1, 1.]]).expand(5, -1)
    end_q = start.new_tensor([[math.cos(.2), 0., math.sin(.2), 0.]]).expand(5, -1)
    rows = torch.arange(5, device=device)
    points, quats, diagnostic = pregrasp_avoidance_waypoints(state, geometry, rows, normal, end, end_q)
    shift = diagnostic['vertical_shift_m']
    assert shift[0] > 0 and shift[1] < 0 and shift[4] > 0
    assert shift[2] == 0 and shift[3] == 0
    assert (diagnostic['height_clearance_after_shift_m'] >= .002 - 1.e-12).all()
    assert (diagnostic['normal_clearance_after_retreat_m'] >= .02 - 1.e-12).all()
    torch.testing.assert_close(points[:, 0], start)
    torch.testing.assert_close(points[:, 1, :2], start[:, :2])
    torch.testing.assert_close(points[:, 2, 2], points[:, 1, 2])
    torch.testing.assert_close(points[:, 2] - points[:, 1], diagnostic['normal_retreat_m'][:, None] * normal)
    torch.testing.assert_close(quats[:, :3], q[:, None].expand(-1, 3, -1))
    torch.testing.assert_close(points[:, -1], end)
    torch.testing.assert_close(quats[:, -1], end_q)
    positions, orientations, lengths, overflow = PullEntryGPU._sample(
        SimpleNamespace(max_samples=512), points, quats, torch.full_like(rows, 4))
    assert not overflow.any()
    for i in range(5):
        torch.testing.assert_close(positions[i, lengths[i] - 1], end[i])
        torch.testing.assert_close(orientations[i, lengths[i] - 1], end_q[i])
        assert (positions[i, 1:lengths[i]] - positions[i, :lengths[i] - 1]).norm(dim=-1).max() <= .002001
    one = {key: value[1:2] for key, value in state.items()}
    selected_geometry = {key: value[1:2] if key.startswith('handle_') and torch.is_tensor(value)
                         else value for key, value in geometry.items()}
    single = pregrasp_avoidance_waypoints(one, selected_geometry, rows[:1], normal[1:2], end[1:2], end_q[1:2])
    torch.testing.assert_close(single[0], points[1:2], rtol=0., atol=0.)
    torch.testing.assert_close(single[1], quats[1:2], rtol=0., atol=0.)


@pytest.mark.parametrize('bad', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_precomputed_route_uses_one_parallel_inverse_and_rejects_inaccurate_samples(bad, dtype):
    planner = GPUStagePlanningMixin()
    planner.path_diagnostics = []
    planner._record_stats = lambda *args, **kwargs: None
    bank = PaddedPathBank(2, 8, 'cpu')
    failures, calls = [], []
    fsm = SimpleNamespace(device='cpu', arm_align_path=bank,
        fail=lambda rows, code: failures.append((rows.tolist(), code)))
    positions = torch.zeros(2, 8, 3, dtype=dtype)
    orientations = torch.tensor([1., 0., 0., 0.], dtype=dtype).expand(2, 8, -1).clone()
    lengths = torch.tensor([4, 6])

    def inverse(fsm, state, rows, p, q, sizes):
        owner, sample = torch.where(torch.arange(8)[None] < sizes[:, None])
        calls.append(len(sample))
        error = torch.zeros(len(sample), 3)
        if bad:
            error[owner == 0, 0] = .01
        return SimpleNamespace(command_position_world=p[owner, sample],
            command_orientation_world_wxyz=q[owner, sample], position_error=error,
            orientation_error=torch.zeros_like(error)), owner, sample

    planner._inverse_pose_bank = inverse
    planned, valid = planner._stage_path(fsm, {}, torch.arange(2), 'arm_align_path',
        'arm_align_path', positions[:, -1], orientations[:, -1],
        samples=(positions, orientations, lengths), inverse_tolerance_m=1.e-4)
    assert calls == [10]
    assert valid.tolist() == [not bad, True]
    assert planned.tolist() == ([1] if bad else [0, 1])
    assert bank.positions.dtype == torch.float32
    torch.testing.assert_close(bank.positions[planned], positions[planned].float())
    torch.testing.assert_close(bank.orientations[planned], orientations[planned].float())
    assert [rows for rows, _ in failures if rows] == ([[0]] if bad else [])
