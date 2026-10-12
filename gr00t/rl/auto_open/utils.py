"""Small dependency-free data conversion helpers shared by AutoOpen modules."""

import numpy as np
import torch


def as_numpy(value, shape_last=None):
    if value is None:
        if shape_last is None:
            return None
        return np.zeros(shape_last, dtype=np.float32)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if shape_last is not None and value.ndim == len(shape_last) + 1:
        value = value[0]
    return value.astype(np.float32, copy=True)
