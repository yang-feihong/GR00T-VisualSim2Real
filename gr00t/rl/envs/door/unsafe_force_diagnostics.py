from pathlib import Path

import torch


TRIGGER_OBJECT_NAMES = ("handle", "panel", "frame", "other")


class UnsafeForceEventRecorder:
    """Capture the frame before and the frame triggering an unsafe-force reset."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.previous = {}
        self.previous_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.events = []

    def invalidate(self, env_ids: torch.Tensor):
        self.previous_valid[env_ids] = False

    def record(
        self,
        trigger: torch.Tensor,
        trigger_object: torch.Tensor,
        features: dict[str, torch.Tensor],
    ):
        if trigger.shape != (self.num_envs,):
            raise ValueError("trigger must have shape [num_envs]")
        if trigger_object.shape != (self.num_envs,):
            raise ValueError("trigger_object must have shape [num_envs]")
        for name, value in features.items():
            if value.shape[0] != self.num_envs:
                raise ValueError(f"feature {name} must start with num_envs")
            if name not in self.previous:
                self.previous[name] = torch.full_like(value, float("nan"))

        for env_id in trigger.nonzero(as_tuple=False).flatten().tolist():
            object_id = int(trigger_object[env_id].item())
            event = {
                "env_id": env_id,
                "trigger_object_id": object_id,
                "trigger_object": TRIGGER_OBJECT_NAMES[object_id],
                "previous_valid": bool(self.previous_valid[env_id].item()),
                "previous": {
                    name: value[env_id].detach().cpu().clone()
                    for name, value in self.previous.items()
                },
                "trigger": {
                    name: value[env_id].detach().cpu().clone()
                    for name, value in features.items()
                },
            }
            self.events.append(event)

        for name, value in features.items():
            self.previous[name].copy_(value)
        self.previous_valid[:] = True

    def save(self, path: str | Path, metadata: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": metadata,
                "trigger_object_names": TRIGGER_OBJECT_NAMES,
                "events": self.events,
            },
            path,
        )
