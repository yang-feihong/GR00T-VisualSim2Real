from __future__ import annotations

import hashlib
import random

import torch


def stream_seed(episode_seed: int, stream_name: str) -> int:
    """Derive a stable RNG seed without depending on batch layout."""
    payload = f"autoopen:{int(episode_seed)}:{stream_name}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def episode_choice(episode_seed: int, stream_name: str, values):
    values = tuple(values)
    if not values:
        raise ValueError("episode_choice requires at least one value")
    return random.Random(stream_seed(episode_seed, stream_name)).choice(values)


def episode_uniform_rows(
    base_seed: int,
    env_ids: torch.Tensor,
    width: int,
    stream_name: str,
    *,
    device,
    dtype,
) -> torch.Tensor:
    """Return one stateless uniform row per episode seed on any target device."""
    if width < 1:
        raise ValueError("width must be positive")
    ids = env_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    rows = []
    for env_id in ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stream_seed(int(base_seed) + int(env_id), stream_name))
        rows.append(torch.rand(width, generator=generator, dtype=dtype))
    if not rows:
        return torch.empty((0, width), device=device, dtype=dtype)
    return torch.stack(rows).to(device=device)
