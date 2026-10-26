import math
import sys
from dataclasses import dataclass, replace
from types import ModuleType, SimpleNamespace

import torch

from gr00t.rl.auto_open.motion_targets import PaddedPathBank
from gr00t.rl.auto_open.tensor_fsm import TensorDoorFSM
from legged_gym.envs.manip_loco.training_doorman_scene import (
    build_training_doorman_door_cfg,
)


@dataclass(frozen=True)
class _AssetCfg:
    door_open_lr: list[str]
    door_open_io: list[str]

    def replace(self, **changes):
        return replace(self, **changes)


@dataclass
class _SpawnCfg:
    assets_cfg: list[_AssetCfg]


@dataclass(frozen=True)
class _DoorCfg:
    prim_path: str
    spawn: _SpawnCfg

    def replace(self, **changes):
        return replace(self, **changes)


def test_pull_direction_is_applied_to_every_door_asset(monkeypatch):
    module_name = "gr00t.rl.data.tasks.door.scenario_cfg.isaacsim"
    module = ModuleType(module_name)
    module.TaskObjCfgDict = {
        "door": _DoorCfg(
            prim_path="unused",
            spawn=_SpawnCfg(
                assets_cfg=[
                    _AssetCfg(["left", "right"], ["in"]),
                    _AssetCfg(["left", "right"], ["in"]),
                ]
            ),
        )
    }
    monkeypatch.setitem(sys.modules, module_name, module)

    result = build_training_doorman_door_cfg(2, door_open_lr="right", door_open_io="out")

    assert result.prim_path == "/World/envs/env_.*/door"
    assert [asset.door_open_lr for asset in result.spawn.assets_cfg] == [
        ["right"],
        ["right"],
    ]
    assert [asset.door_open_io for asset in result.spawn.assets_cfg] == [
        ["out"],
        ["out"],
    ]


def _turn_gap_endpoint_x(hinge_axis_sign: float) -> tuple[float, float]:
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.device = torch.device("cpu")
    fsm.stage_steps = torch.zeros(1, dtype=torch.int32)
    fsm.hinge_idx = 0
    fsm.turn_start_hinge_angle = torch.zeros(1)
    fsm.turn_target_hinge_angle = torch.tensor([hinge_axis_sign * math.radians(6.0)])
    fsm.hinge_axis_sign = torch.tensor([hinge_axis_sign])
    fsm.turn_handle_axis_world = torch.tensor([[0.0, 1.0, 0.0]])
    fsm.turn_grasp_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    fsm.turn_gap_gripper_angle = torch.zeros(1)
    fsm.hinge_world = torch.zeros((1, 3))
    fsm.turn_path = PaddedPathBank(1, 512, "cpu")
    state = {
        "door_joint_position": torch.zeros((1, 3)),
        "ee_pos": torch.tensor([[0.0, 1.0, 0.0]]),
        "ee_orn": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "ee_goal_cart_world": torch.tensor([[0.0, 1.0, 0.0]]),
        "ee_goal_orn_quat_world": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }

    fsm._build_turn_gap_path(state, torch.tensor([0]))
    endpoint_x = fsm.turn_path.positions[0, fsm.turn_path.length[0] - 1, 0]
    return (
        float(fsm.turn_target_hinge_angle[0]),
        float(endpoint_x),
    )


def test_pull_hinge_sign_reverses_turn_path_toward_robot_side():
    push_angle, push_endpoint_x = _turn_gap_endpoint_x(1.0)
    pull_angle, pull_endpoint_x = _turn_gap_endpoint_x(-1.0)

    assert push_angle > 0.0
    assert pull_angle < 0.0
    assert push_endpoint_x < 0.0
    assert pull_endpoint_x > 0.0


def test_restore_completion_can_be_the_terminal_stage():
    fsm = TensorDoorFSM.__new__(TensorDoorFSM)
    fsm.Stage = SimpleNamespace(STOW_ARM_AND_CLOSE_GRIPPER=10, STOW_ARM_AFTER_PULL=23, SUCCESS=17)
    fsm.terminal_stage = fsm.Stage.STOW_ARM_AND_CLOSE_GRIPPER
    fsm.device = torch.device("cpu")
    fsm.stage = torch.tensor([10], dtype=torch.int64)
    fsm.stage_steps = torch.tensor([50], dtype=torch.int32)
    fsm.q_closed = torch.zeros(1)
    fsm.release_endpoint_world = torch.zeros((1, 3))
    fsm.release_endpoint_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    fsm.release_stable_steps = torch.tensor([9], dtype=torch.int32)
    fsm.arm_stability = SimpleNamespace(
        update=lambda state, rows: (
            torch.ones(len(rows), dtype=torch.bool),
            torch.zeros(len(rows)),
            torch.zeros(len(rows)),
        )
    )
    state = {
        "gripper_position": torch.zeros((1, 1)),
        "ee_pos": torch.zeros((1, 3)),
        "ee_orn": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }

    completed = fsm.advance_stow_arm_and_close_gripper(state)

    assert completed.tolist() == [0]
    assert fsm.stage.tolist() == [17]
    assert fsm.stage_steps.tolist() == [0]
