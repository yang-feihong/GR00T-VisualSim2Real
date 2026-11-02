"""Counter-keyed GPU random samples, shared by task planners."""

import torch


def episode_uniform(seeds, streams, candidate_count, lanes=4):
    """Preserve pull's uint32 mixer; shape and other rows cannot change a seed's draws."""
    mask = 0xFFFFFFFF

    def mix(value):
        value = ((value ^ (value >> 16)) * 0x45D9F3B) & mask
        value = ((value ^ (value >> 16)) * 0x45D9F3B) & mask
        return (value ^ (value >> 16)) & mask

    if isinstance(streams, int):
        streams = torch.full_like(seeds, streams)
    attempts = torch.arange(candidate_count, device=seeds.device, dtype=torch.int64)
    channels = torch.arange(lanes, device=seeds.device, dtype=torch.int64)
    seed_key = mix(seeds & mask) ^ mix(((seeds >> 32) & mask) ^ 0x243F6A88)
    stream_key = mix((streams & mask) ^ 0x6A09E667) ^ mix(((streams >> 32) & mask) ^ 0x3C6EF372)
    key = mix(
        seed_key[:, None, None]
        ^ stream_key[:, None, None]
        ^ mix(attempts[None, :, None] ^ 0x510E527F)
        ^ mix(channels[None, None, :] ^ 0x1F83D9AB)
    )
    return (key >> 8).to(torch.float32) * (1.0 / 16777216.0)
