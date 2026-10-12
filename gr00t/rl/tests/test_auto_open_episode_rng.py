import sys
from pathlib import Path

import torch

LOWLEVEL_ROOT = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "visual_wholebody_lowlevel"
)
sys.path.insert(0, str(LOWLEVEL_ROOT))

from legged_gym.envs.manip_loco.training_doorman_scene import (
    episode_door_side,
)
from legged_gym.utils.episode_rng import episode_uniform_rows, stream_seed


def test_episode_uniform_rows_do_not_depend_on_batch_size():
    one = episode_uniform_rows(
        1718, torch.tensor([0]), 9, "robot_init", device="cpu", dtype=torch.float32
    )
    many = episode_uniform_rows(
        1718, torch.arange(8), 9, "robot_init", device="cpu", dtype=torch.float32
    )
    assert torch.equal(one[0], many[0])

    isolated = episode_uniform_rows(
        1723, torch.tensor([0]), 9, "robot_init", device="cpu", dtype=torch.float32
    )
    assert torch.equal(isolated[0], many[5])


def test_door_side_and_physics_seed_do_not_depend_on_batch_size():
    many_sides = [episode_door_side(1718 + env_id) for env_id in range(8)]
    many_physics_seeds = [
        stream_seed(1718 + env_id, "door_physics") for env_id in range(8)
    ]
    assert episode_door_side(1718) == many_sides[0]
    assert stream_seed(1718, "door_physics") == many_physics_seeds[0]
    assert episode_door_side(1723) == many_sides[5]
    assert stream_seed(1723, "door_physics") == many_physics_seeds[5]
