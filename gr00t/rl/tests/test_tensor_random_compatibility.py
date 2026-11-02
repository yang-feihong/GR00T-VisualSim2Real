"""The shared GPU RNG must not change existing door sampling streams."""

import pytest
import torch

from gr00t.rl.auto_open.tensor_fsm import _pull_cycle_uniform


def legacy_pull_uniform(seeds, cycles, candidate_count):
    """Stateless uint32 keyed streams, evaluated in int64 on the input device.

    Every multiplication is below 2**63, including before the explicit mask.
    This replaces NumPy PCG draws; it preserves ranges and batch independence,
    but deliberately does not claim bitwise compatibility with the old stream.
    """
    mask = 0xFFFFFFFF

    def mix(value):
        value = ((value ^ (value >> 16)) * 0x45D9F3B) & mask
        value = ((value ^ (value >> 16)) * 0x45D9F3B) & mask
        return (value ^ (value >> 16)) & mask

    attempts = torch.arange(candidate_count, device=seeds.device, dtype=torch.int64)
    lanes = torch.arange(4, device=seeds.device, dtype=torch.int64)
    seed_key = mix(seeds & mask) ^ mix(((seeds >> 32) & mask) ^ 0x243F6A88)
    cycle_key = mix((cycles & mask) ^ 0x6A09E667) ^ mix(((cycles >> 32) & mask) ^ 0x3C6EF372)
    key = mix(
        seed_key[:, None, None]
        ^ cycle_key[:, None, None]
        ^ mix(attempts[None, :, None] ^ 0x510E527F)
        ^ mix(lanes[None, None, :] ^ 0x1F83D9AB)
    )
    # Use the upper 24 bits so conversion to float32 cannot round up to 1.
    return (key >> 8).to(torch.float32) * (1.0 / 16777216.0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_rng_preserves_legacy_pull_stream(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA pull stream compatibility")
    seeds = torch.tensor([0, 1700, 1000000, 2**40 + 3, 2**62 + 23], device=device)
    cycles = torch.tensor([0, 1, 5, 2**33 + 1, 15], device=device)
    torch.testing.assert_close(
        _pull_cycle_uniform(seeds, cycles, 257),
        legacy_pull_uniform(seeds, cycles, 257),
        rtol=0,
        atol=0,
    )
