"""Summarize completed collection batches without running physics or rewriting raw data."""

import argparse
import json
from collections import Counter
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def control_outcomes(root):
    """Report task results without confusing control success with strict admission."""
    stages, reasons, initial_checks = Counter(), Counter(), Counter()
    scenes, categories, supports = Counter(), Counter(), Counter()
    completed = succeeded = 0
    for path in sorted(Path(root).glob("logs/rank*-control/seed_*/summary.json")):
        result = read_json(path)
        if "success" not in result:
            continue
        completed += 1
        succeeded += bool(result["success"])
        semantics = result.get("target_semantics", {})
        scenes[result.get("scene_id", "unknown")] += 1
        categories[semantics.get("object_category", result.get("asset", "unknown"))] += 1
        supports[semantics.get("support_category", "unknown")] += 1
        if result["success"]:
            continue
        failure = result.get("failure")
        stage = result.get("failed_stage") or (
            "INITIALIZATION" if not result.get("initial_accepted", True) else "unknown"
        )
        reason = str(failure or "unreported_failure")
        if isinstance(failure, dict):
            stage, reason = failure.get("stage", stage), failure.get("reason", reason)
        stages[stage] += 1
        reasons[f"{stage}: {reason}"] += 1
        initial = path.with_name("initial_acceptance.json")
        if stage == "INITIALIZATION" and initial.exists():
            initial_checks.update(
                key for key, passed in read_json(initial).get("checks", {}).items() if not passed
            )
    return {
        "completed_episodes": completed,
        "control_success_episodes": succeeded,
        "control_success_rate": succeeded / completed if completed else None,
        "failed_stage_counts": dict(stages),
        "failed_reason_counts": dict(reasons),
        "initial_admission_failed_checks": dict(initial_checks),
        "scene_counts": dict(scenes),
        "target_category_counts": dict(categories),
        "support_category_counts": dict(supports),
    }


def report_batch(root, *, requested_episodes=None):
    root = Path(root)
    finalized = (root / "acceptance_report.json").is_file() or (root / "summary.json").is_file()
    if finalized:
        path = root / "acceptance_report.json"
        summary = read_json(path if path.exists() else root / "summary.json")
        requested = summary.get("requested_episodes", summary.get("total_episodes"))
        if requested_episodes is not None and requested_episodes != requested:
            raise ValueError("Requested episode count conflicts with the finalized batch")
    else:
        request = (
            read_json(root / "run_request.json") if (root / "run_request.json").is_file() else {}
        )
        requested = request.get("requested_episodes", requested_episodes)
        if requested is None or requested < 1:
            raise ValueError("Interrupted legacy batches require --requested-episodes")
        complete_runs = {
            path.parent.name for path in root.glob("logs/rank*-control/seed_*/summary.json")
        }
        accepted = 0
        for path in root.glob("shards/episode-seed*/meta/acceptance_report.json"):
            for episode in read_json(path)["episodes"]:
                if f"seed_{episode['seed']}" in complete_runs:
                    accepted += bool(episode["accepted"])
        summary = {"accepted_episodes": accepted, "collection_wall_time_s": None}
    result = {
        "batch": str(root),
        "requested_episodes": requested,
        "accepted_episodes": summary["accepted_episodes"],
        "collection_wall_time_s": summary["collection_wall_time_s"],
        **control_outcomes(root),
    }
    result["complete"] = finalized and result["completed_episodes"] == requested
    result["strict_acceptance_rate"] = (
        summary["accepted_episodes"] / requested if result["complete"] else None
    )
    completed = result["completed_episodes"]
    result["completed_strict_acceptance_rate"] = (
        summary["accepted_episodes"] / completed if completed else None
    )
    write_json(root / "control_outcomes.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, nargs="+", required=True)
    parser.add_argument("--requested-episodes", type=int)
    args = parser.parse_args()
    for root in args.batch:
        print(
            json.dumps(
                report_batch(root, requested_episodes=args.requested_episodes), ensure_ascii=False
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
