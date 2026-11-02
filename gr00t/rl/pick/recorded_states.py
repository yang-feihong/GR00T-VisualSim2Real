"""Read-only diagnostic access to the shared per-episode control record."""

from collections.abc import Sequence
from pathlib import Path

import torch


class RecordedStates(Sequence):
    def __init__(self, post):
        self.post = post

    def __len__(self):
        return len(self.post["root_states"])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return {key: value[index : index + 1] for key, value in self.post.items()}


def load_recorded_states(run):
    """Expose post-step states without making another capture format or copying frames."""
    run = Path(run)
    legacy = run / "trajectory.pt"
    if legacy.is_file():
        return torch.load(legacy, map_location="cpu", weights_only=False)
    record = torch.load(run / "control_record.pt", map_location="cpu", weights_only=False)
    return RecordedStates(record["post"])
