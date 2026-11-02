"""Pack ragged device data without row-wise copies or device scalar reads."""

import torch


def pad_device_rows(values, *, fill=0, minimum_width=1):
    if not values:
        raise ValueError("Cannot pack an empty row list")
    lengths = [len(value) for value in values]
    width = max(minimum_width, max(lengths))
    reference = values[0]
    counts = torch.tensor(lengths, device=reference.device, dtype=torch.long)
    owner = torch.repeat_interleave(
        torch.arange(len(values), device=reference.device), counts, output_size=sum(lengths)
    )
    offset = counts.cumsum(0) - counts
    column = torch.arange(sum(lengths), device=reference.device) - offset[owner]
    packed = reference.new_full((len(values), width, *reference.shape[1:]), fill)
    packed[owner, column] = torch.cat(values)
    return packed, torch.arange(width, device=reference.device)[None] < counts[:, None]


def pack_scalars(values, reference):
    """Combine Python configuration and existing device scalars with one upload/scatter."""
    host = [0 if torch.is_tensor(value) else value for value in values]
    packed = reference.new_tensor(host)
    indices = [index for index, value in enumerate(values) if torch.is_tensor(value)]
    if indices:
        packed[torch.tensor(indices, device=reference.device)] = torch.stack(
            [values[index].reshape(()) for index in indices]
        )
    return packed
