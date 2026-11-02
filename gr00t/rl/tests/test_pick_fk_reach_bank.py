"""Conditioned calibration poses remain source-independent and prefix stable."""

import math

import pytest
import torch

from gr00t.rl.auto_open.arm_model import read_arm_joint_limits
from gr00t.rl.auto_open.planning_geometry import _quat_rotate
from gr00t.rl.pick.probe_policy import synthetic_fk_reach_bank
from gr00t.rl.pick.supported_asset_audit import gripper_collision_clouds
from gr00t.rl.utils.b2z1_asset import ROBOT_URDF


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU calibration bank")
def test_fk_bank_conditions_on_low_floor_clear_goals_without_seed_scene_data():
    limits = torch.tensor(
        read_arm_joint_limits(ROBOT_URDF).copy(), device="cuda:0", dtype=torch.float32
    )
    fixed, closed = gripper_collision_clouds("cuda:0")
    opened = _quat_rotate(
        limits.new_tensor([math.cos(math.pi / 6), 0, -math.sin(math.pi / 6), 0]), closed
    )
    cloud = torch.cat((fixed, opened))
    stations, goals, quats, jitter = synthetic_fk_reach_bank(256, limits, cloud)
    first = synthetic_fk_reach_bank(1, limits, cloud)
    for batch, single in zip((stations, goals, quats, jitter), first):
        torch.testing.assert_close(batch[:1], single)
    assert ((goals[:, 2] >= 0.075) & (goals[:, 2] <= 0.20)).all()
    assert ((goals[:, 0] >= 0.35) & (goals[:, 0] <= 0.85)).all()
    assert (goals[:, 1].abs() <= 0.30).all()
    bottom = _quat_rotate(quats[:, None], cloud[None])[..., 2].amin(-1) + goals[:, 2]
    assert (bottom >= 0.002).all()
    axis = _quat_rotate(quats, goals.new_tensor([1, 0, 0]))
    assert (axis[:, 2] <= -math.cos(math.pi / 4) + 1e-6).all()
    assert jitter.abs().amax() <= 0.05
