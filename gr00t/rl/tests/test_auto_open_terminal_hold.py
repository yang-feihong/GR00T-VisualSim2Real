"""Runtime must not consume the recorder's reset confirmation as terminal hold."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM
from gr00t.rl.data_collection.auto_open_collector import AutoOpenStandardEpisodeCollector


def _runtime(count=1):
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device = torch.device("cpu")
    fsm.Stage = SimpleNamespace(PASS_THROUGH_DOORWAY=1, SUCCESS=2)
    fsm.stage = torch.ones(count, dtype=torch.int64)
    fsm.stage_steps = torch.zeros(count, dtype=torch.int32)
    fsm.mask = lambda stage: fsm.stage == stage
    fsm.hinge_idx = 0
    fsm.env = SimpleNamespace(arm_joint_ids=torch.arange(6), gripper_joint_ids=torch.tensor([6]))
    for name in (
        "through_entry_stable_steps", "through_settle_steps", "through_middle_ready_steps",
        "through_no_progress_steps", "through_path_previous_index",
    ):
        setattr(fsm, name, torch.zeros(count, dtype=torch.int32))
    fsm.through_trajectory = SimpleNamespace(index=torch.zeros(count, dtype=torch.int32))
    fsm.through_motion_ready = torch.ones(count, dtype=torch.bool)
    fsm.through_stop_latched = torch.ones(count, dtype=torch.bool)
    fsm.through_min_base_height = torch.full((count,), 0.3)
    fsm.pending_station_distance = torch.zeros(count)
    fsm.through_best_station_error = torch.ones(count)
    fsm.middle_arm_joint_target = torch.zeros((count, 6))
    fsm.control_joint_velocity = torch.zeros((count, 7))
    fsm.record_acceptance = lambda *args, **kwargs: None
    fsm._tail_clearance = lambda state, rows: torch.full((len(rows),), 1.1)
    fsm.fail = lambda rows, code: None
    root = torch.zeros((count, 13))
    root[:, 2:4] = torch.tensor([0.5, 1.0])
    state = {
        "root_states": root,
        "door_joint_position": torch.full((count, 3), 1.6),
        "door_joint_velocity": torch.zeros((count, 3)),
        "dof_pos": torch.zeros((count, 7)),
    }
    return fsm, state


@pytest.mark.parametrize("stable_frames_before_stop", [0, 5, 6, 9, 10])
def test_runtime_waits_for_fifty_recorded_inactive_frames(stable_frames_before_stop):
    fsm, state = _runtime()
    collector = AutoOpenStandardEpisodeCollector.__new__(AutoOpenStandardEpisodeCollector)
    collector.controller = SimpleNamespace(middle_arm_joint_target=np.zeros(6))
    collector.arm_indices = np.arange(6)
    collector._clear_middle_preset_mode()
    snapshot = {"dof_pos": np.full(6, 0.1), "arm_velocity": np.zeros(6)}
    assert collector._mode_flags(snapshot, np.zeros(6), joint_mode=True) == (False, True)
    fsm.through_stop_latched[:] = False
    state["dof_pos"][:, :6] = 0.1
    fsm.advance_pass_through_doorway(state)
    state["dof_pos"][:] = 0.0
    snapshot["dof_pos"][:] = 0.0
    recorded_hold = 0
    for frame in range(100):
        # Collection labels the pre-step snapshot, runtime observes the post-step state.
        inactive, reset = collector._mode_flags(snapshot, np.zeros(6), joint_mode=True)
        stopped = frame >= stable_frames_before_stop
        recorded_hold = recorded_hold + 1 if stopped and inactive and not reset else 0
        fsm.through_stop_latched[:] = stopped
        completed = fsm.advance_pass_through_doorway(state)
        if len(completed):
            assert recorded_hold >= 50
            break
    else:
        pytest.fail("runtime never completed despite a valid terminal hold")


def test_terminal_hold_resets_on_motion_without_restarting_confirmed_arm_mode():
    fsm, state = _runtime(2)
    for _ in range(30):
        fsm.advance_pass_through_doorway(state)
    assert fsm.through_middle_ready_steps.tolist() == [10, 10]
    state["root_states"][0, 7] = 0.1
    fsm.advance_pass_through_doorway(state)
    assert fsm.through_settle_steps[0] == 0
    assert fsm.through_settle_steps[1] > 0
    assert fsm.through_middle_ready_steps.tolist() == [10, 10]
    state["root_states"][0, 7] = 0.0
    for _ in range(50):
        fsm.advance_pass_through_doorway(state)
    assert fsm.stage[0] == fsm.Stage.PASS_THROUGH_DOORWAY
    fsm.advance_pass_through_doorway(state)
    assert fsm.stage[0] == fsm.Stage.SUCCESS


def test_through_entry_does_not_wait_for_door_but_terminal_hold_still_does():
    fsm, state = _runtime()
    fsm.through_motion_ready[:] = False
    state["door_joint_velocity"][:] = 0.5
    for _ in range(9):
        fsm.advance_pass_through_doorway(state)
    assert not fsm.through_motion_ready.any()
    fsm.advance_pass_through_doorway(state)
    assert fsm.through_motion_ready.all()
    assert fsm.through_settle_steps.item() == 0
