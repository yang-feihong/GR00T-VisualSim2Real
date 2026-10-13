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
from legged_gym.utils.episode_rng import canonical_env_origins


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


def test_door_visual_seed_is_deterministic_and_independent_from_physics():
    for episode_seed in range(1700, 1708):
        visual_seed = stream_seed(episode_seed, "door_visual")
        assert visual_seed == stream_seed(episode_seed, "door_visual")
        assert visual_seed != stream_seed(episode_seed, "door_physics")

    source = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "visual_wholebody_lowlevel"
        / "legged_gym"
        / "envs"
        / "manip_loco"
        / "training_doorman_scene.py"
    ).read_text(encoding="utf-8")
    assert 'visual_random_seed=stream_seed(' in source
    assert 'episode_seeds[env_id], "door_visual"' in source


def test_canonical_origin_depends_on_episode_seed_not_batch_slot():
    scalar = canonical_env_origins(2089, 1, 3.0)
    batch = canonical_env_origins(1950, 250, 3.0)
    assert torch.equal(scalar[0], batch[2089 - 1950])


def test_canonical_origins_are_unique_beyond_previous_batch_capacity():
    origins = canonical_env_origins(1700, 1000, 3.0)
    assert torch.unique(origins, dim=0).shape[0] == 1000
    assert torch.equal(origins[0], torch.tensor([0.0, 0.0, 0.0]))


def test_canonical_origins_are_stable_across_overlapping_shards():
    first = canonical_env_origins(1700, 1000, 3.0)
    second = canonical_env_origins(2200, 1000, 3.0)
    assert torch.equal(first[500:], second[:500])
