import torch


FORMAT = "per_environment_stage_initial_states_v2"


def pack_stage_initial_state_library(
    state_cases,
    num_samples,
    max_samples_per_stage,
    checkpoint_samples_per_env,
):
    """Pack recent per-environment acceptance states without changing ownership."""
    num_stages, num_envs = num_samples.shape
    sample_counts = num_samples.clamp(
        min=0,
        max=min(max_samples_per_stage, checkpoint_samples_per_env),
    ).clone()
    sample_counts[0] = 0
    packed_samples = int(sample_counts.max().item())
    offsets = torch.arange(packed_samples, device=num_samples.device)[None, :, None]
    sample_indices = (num_samples[:, None, :] - 1 - offsets).remainder(
        max_samples_per_stage
    )
    stage_indices = torch.arange(num_stages, device=num_samples.device)[:, None, None]
    env_indices = torch.arange(num_envs, device=num_samples.device)[None, None, :]
    cases = {}
    for name, state_case in state_cases.items():
        packed = {"type": state_case["type"]}
        if state_case["type"] == "buffer":
            packed["data"] = state_case["data"][
                stage_indices, sample_indices, env_indices
            ].detach().cpu()
        else:
            packed["root_state"] = state_case["root_state"][
                stage_indices, sample_indices, env_indices
            ].detach().cpu()
            if state_case["type"] == "articulation":
                packed["dof_state"] = state_case["dof_state"][
                    stage_indices, sample_indices, env_indices
                ].detach().cpu()
        cases[name] = packed
    return {
        "format": FORMAT,
        "num_stages": num_stages,
        "num_envs": num_envs,
        "sample_counts": sample_counts.detach().cpu(),
        "cases": cases,
    }


def restore_stage_initial_state_library(
    checkpoint,
    state_cases,
    num_stages,
    num_envs,
    max_samples_per_stage,
    device,
):
    """Validate and restore a library; return per-stage/environment counts."""
    if checkpoint.get("format") != FORMAT:
        raise ValueError("unsupported staged-reset library format")
    if checkpoint.get("num_stages") != num_stages:
        raise ValueError("staged-reset library stage count changed")
    if checkpoint.get("num_envs") != num_envs:
        raise ValueError("staged-reset library environment count changed")

    sample_counts = checkpoint["sample_counts"].to(device, dtype=torch.long)
    if tuple(sample_counts.shape) != (num_stages, num_envs):
        raise ValueError("invalid staged-reset sample-count shape")
    sample_counts = sample_counts.clamp(min=0, max=max_samples_per_stage)
    packed_samples = int(sample_counts.max().item())
    cases = checkpoint.get("cases", {})
    if set(cases) != set(state_cases):
        raise ValueError("tracked staged-reset cases changed")

    for name, state_case in state_cases.items():
        packed = cases[name]
        if packed.get("type") != state_case["type"]:
            raise ValueError(f"staged-reset case {name!r} changed type")
        tensors = (
            (("data", state_case["data"]),)
            if state_case["type"] == "buffer"
            else (("root_state", state_case["root_state"]),)
        )
        if state_case["type"] == "articulation":
            tensors += (("dof_state", state_case["dof_state"]),)
        for key, target in tensors:
            expected = target[:, :packed_samples]
            if tuple(packed[key].shape) != tuple(expected.shape):
                raise ValueError(f"staged-reset case {name!r} {key} shape changed")

    for name, state_case in state_cases.items():
        packed = cases[name]
        if state_case["type"] == "buffer":
            state_case["data"].zero_()
            state_case["data"][:, :packed_samples].copy_(packed["data"].to(device))
        else:
            state_case["root_state"].zero_()
            state_case["root_state"][:, :packed_samples].copy_(
                packed["root_state"].to(device)
            )
            if state_case["type"] == "articulation":
                state_case["dof_state"].zero_()
                state_case["dof_state"][:, :packed_samples].copy_(
                    packed["dof_state"].to(device)
                )
    return sample_counts
