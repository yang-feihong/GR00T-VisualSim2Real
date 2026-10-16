from __future__ import annotations

import hashlib
import math
import random

import torch


AUTO_OPEN_CANONICAL_ENV_SEED_ANCHOR = 1700
AUTO_OPEN_POLICY_BLOCK_SIZE = 256


def stream_seed(episode_seed: int, stream_name: str) -> int:
    """Derive a stable RNG seed without depending on batch layout."""
    payload = f"autoopen:{int(episode_seed)}:{stream_name}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def episode_choice(episode_seed: int, stream_name: str, values):
    values = tuple(values)
    if not values:
        raise ValueError("episode_choice requires at least one value")
    return random.Random(stream_seed(episode_seed, stream_name)).choice(values)


def canonical_env_origins(
    base_seed: int,
    num_envs: int,
    env_spacing: float,
    *,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """Map every integer episode seed to a unique, batch-independent lattice point."""
    num_envs = int(num_envs)
    if num_envs < 1:
        raise ValueError("num_envs must be positive")

    def spiral_xy(index: int):
        if index == 0:
            return 0, 0
        ring = math.ceil((math.sqrt(index + 1) - 1.0) / 2.0)
        side = 2 * ring
        maximum = (2 * ring + 1) ** 2 - 1
        if index >= maximum - side:
            return ring - (maximum - index), -ring
        if index >= maximum - 2 * side:
            return -ring, -ring + (maximum - side - index)
        if index >= maximum - 3 * side:
            return -ring + (maximum - 2 * side - index), ring
        return ring, ring - (maximum - 3 * side - index)

    coordinates = []
    for episode_seed in range(int(base_seed), int(base_seed) + num_envs):
        delta = episode_seed - AUTO_OPEN_CANONICAL_ENV_SEED_ANCHOR
        spiral_index = 2 * delta if delta >= 0 else -2 * delta - 1
        coordinates.append(spiral_xy(spiral_index))
    origins = torch.zeros((num_envs, 3), dtype=torch.float64)
    origins[:, :2] = torch.tensor(coordinates, dtype=torch.float64) * float(env_spacing)
    return origins.to(device=device, dtype=dtype)


def canonical_env_origins_for_seeds(
    episode_seeds,
    env_spacing: float,
    *,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """Map an explicit seed list to the same canonical lattice."""
    seeds = [int(seed) for seed in episode_seeds]
    if not seeds:
        raise ValueError("episode_seeds must not be empty")

    def spiral_xy(index: int):
        if index == 0:
            return 0, 0
        ring = math.ceil((math.sqrt(index + 1) - 1.0) / 2.0)
        side = 2 * ring
        maximum = (2 * ring + 1) ** 2 - 1
        if index >= maximum - side:
            return ring - (maximum - index), -ring
        if index >= maximum - 2 * side:
            return -ring, -ring + (maximum - side - index)
        if index >= maximum - 3 * side:
            return -ring + (maximum - 2 * side - index), ring
        return ring, ring - (maximum - 3 * side - index)

    coordinates = []
    for episode_seed in seeds:
        delta = episode_seed - AUTO_OPEN_CANONICAL_ENV_SEED_ANCHOR
        spiral_index = 2 * delta if delta >= 0 else -2 * delta - 1
        coordinates.append(spiral_xy(spiral_index))
    origins = torch.zeros((len(seeds), 3), dtype=torch.float64)
    origins[:, :2] = torch.tensor(coordinates, dtype=torch.float64) * float(env_spacing)
    return origins.to(device=device, dtype=dtype)


def episode_uniform_rows(
    base_seed: int,
    env_ids: torch.Tensor,
    width: int,
    stream_name: str,
    *,
    device,
    dtype,
    episode_seeds=None,
) -> torch.Tensor:
    """Return one stateless uniform row per episode seed on any target device."""
    if width < 1:
        raise ValueError("width must be positive")
    ids = env_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    rows = []
    explicit_seeds = (
        None if episode_seeds is None else [int(seed) for seed in episode_seeds]
    )
    for env_id in ids:
        generator = torch.Generator(device="cpu")
        episode_seed = (
            int(base_seed) + int(env_id)
            if explicit_seeds is None
            else explicit_seeds[int(env_id)]
        )
        generator.manual_seed(stream_seed(episode_seed, stream_name))
        rows.append(torch.rand(width, generator=generator, dtype=dtype))
    if not rows:
        return torch.empty((0, width), device=device, dtype=dtype)
    return torch.stack(rows).to(device=device)
