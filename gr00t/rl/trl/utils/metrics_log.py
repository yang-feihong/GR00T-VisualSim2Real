import json
import math
from pathlib import Path


def _json_scalar(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, ValueError) as error:
            raise TypeError(
                f"Training metric must be a scalar, got {type(value).__name__}"
            ) from error
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError(f"Training metric must be a scalar, got {type(value).__name__}")


def append_training_metrics_jsonl(output_dir, metrics, *, is_world_process_zero):
    """Persist one complete, standards-compliant metrics record on rank zero."""
    if not is_world_process_zero:
        return None

    record = {}
    non_finite_metrics = []
    for key, value in metrics.items():
        scalar = _json_scalar(value)
        if scalar is None and value is not None:
            non_finite_metrics.append(key)
        record[key] = scalar
    record["_non_finite_metrics"] = sorted(non_finite_metrics)

    path = Path(output_dir) / "training_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
    return path


def take_remaining_episode_ids(completed_ids, collected_count, target_count):
    """Select only the episode completions still needed by an evaluator."""
    remaining = max(int(target_count) - int(collected_count), 0)
    return completed_ids[:remaining]
