"""One packed device transfer per dtype for diagnostic/descriptor trees only."""

import torch


def to_host(tree):
    """Keep planning on device; serialize completed decisions without row-wise syncs."""
    tensors, groups = [], {}

    def collect(value):
        if torch.is_tensor(value):
            index = len(tensors)
            tensors.append(value.detach())
            groups.setdefault((value.device, value.dtype), []).append(index)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)

    collect(tree)
    host = [None] * len(tensors)
    for (device, _), indices in groups.items():
        flat = torch.cat([tensors[index].reshape(-1) for index in indices])
        if device.type != "cpu":
            flat = flat.cpu()
        offset = 0
        for index in indices:
            tensor = tensors[index]
            host[index] = flat[offset : offset + tensor.numel()].reshape(tensor.shape).tolist()
            offset += tensor.numel()
    cursor = iter(host)

    def restore(value):
        if torch.is_tensor(value):
            return next(cursor)
        if isinstance(value, dict):
            return {key: restore(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(restore(item) for item in value)
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value

    return restore(tree)
