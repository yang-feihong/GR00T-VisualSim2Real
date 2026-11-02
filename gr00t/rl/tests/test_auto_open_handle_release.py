"""Release restores the pre-closing world height before withdrawing."""

from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.motion_targets import PaddedPathBank
from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return torch.device(request.param)


def test_grasp_entry_snapshots_actual_world_height_for_selected_envs(device):
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage = SimpleNamespace(GRASP_HANDLE=1)
    fsm.grasp_preclose_height_world = torch.full((3,), torch.nan, device=device)
    fsm.grasp_settle_steps = torch.ones(3, dtype=torch.int32, device=device)
    fsm.stage = torch.zeros(3, dtype=torch.int64, device=device)
    fsm.transition = lambda rows, stage: fsm.stage.index_fill_(0, rows, stage)
    state = {'ee_pos': torch.tensor([[0., 0., .8], [0., 0., .9], [0., 0., 1.]], device=device)}
    rows = torch.tensor([2, 0], device=device)
    fsm.begin_grasp_handle(state, rows)
    state['ee_pos'][:, 2] -= .03
    torch.testing.assert_close(fsm.grasp_preclose_height_world[rows], torch.tensor([1., .8], device=device))
    assert fsm.grasp_preclose_height_world[1].isnan()
    assert fsm.grasp_settle_steps.tolist() == [0, 1, 0]


def test_release_requires_completed_path_and_ten_frames_at_restored_height(device):
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device = device
    fsm.Stage = SimpleNamespace(RELEASE_HANDLE=1)
    fsm.mask = lambda stage: torch.ones(2, dtype=torch.bool, device=device)
    fsm.q_open = torch.full((2,), -1., device=device)
    fsm.stage_steps = torch.ones(2, dtype=torch.int32, device=device)
    fsm.release_stable_steps = torch.zeros(2, dtype=torch.int32, device=device)
    fsm.release_open_target_world = torch.tensor([[0., 0., .8], [0., 0., .9]], device=device)
    fsm.release_open_path = PaddedPathBank(2, 36, device)
    fsm.release_open_path.length[:] = 36
    fsm._release_blocking_handle_contacts = lambda *_: torch.zeros((2, 2), dtype=torch.bool, device=device)
    observations = []
    fsm.record_acceptance = lambda stage, rows, conditions, **kwargs: observations.append(conditions)
    state = {'gripper_position': torch.full((2, 1), -1., device=device),
             'ee_pos': fsm.release_open_target_world.clone()}
    for _ in range(12):
        assert not fsm.advance_release_handle(state).numel()
    fsm.release_open_path.index[:] = 35
    state['ee_pos'][0, 2] -= .015
    for _ in range(9):
        assert not fsm.advance_release_handle(state).numel()
    assert fsm.advance_release_handle(state).tolist() == [1]
    assert not observations[-1]['release_height_error_m'][0][0]
    state['ee_pos'][0, 2] = .8
    for _ in range(9):
        assert 0 not in fsm.advance_release_handle(state).tolist()
    state['ee_pos'][0, 2] += .007
    fsm.advance_release_handle(state)
    assert fsm.release_stable_steps[0] == 0
    state['ee_pos'][0, 2] = .805
    for _ in range(9):
        assert 0 not in fsm.advance_release_handle(state).tolist()
    assert 0 in fsm.advance_release_handle(state).tolist()
