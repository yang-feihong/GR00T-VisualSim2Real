#!/usr/bin/env python3
"""Compare AutoOpen semantic target-noise runs against a paired baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load(root: Path):
    episodes = {}
    for path in root.rglob("metadata.json"):
        data = json.loads(path.read_text())
        seed = int(data["seed"])
        if seed in episodes:
            raise ValueError(f"duplicate seed {seed} below {root}")
        result = data.get("result", {})
        final_stage = str(result.get("final_stage", ""))
        acceptance = data.get("acceptance_diagnostics") or {}
        failure_stage = data.get("failure_stage")
        if not failure_stage and not bool(data.get("success", final_stage == "SUCCESS")):
            failure_stage = acceptance.get("stage")
        episodes[seed] = {
            "success": bool(data.get("success", final_stage == "SUCCESS")),
            "final_stage": final_stage,
            "failure_stage": failure_stage,
            "failure_reason": data.get("failure_reason"),
            "steps": result.get("steps"),
            "target_noise": data.get("target_noise", {}),
        }
    if not episodes:
        raise ValueError(f"no metadata.json files below {root}")
    return episodes


def _stage_counts(episodes):
    return Counter(
        str(item["failure_stage"] or item["final_stage"] or "UNKNOWN")
        for item in episodes.values()
        if not item["success"]
    )


def _format_counts(counts):
    return ", ".join(
        f"{stage}={count}"
        for stage, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ) or "none"


def _format_stage_comparison(baseline, case):
    baseline_counts = _stage_counts(baseline)
    case_counts = _stage_counts(case)
    stages = sorted(
        baseline_counts.keys() | case_counts.keys(),
        key=lambda stage: (-case_counts[stage], stage),
    )
    return ", ".join(
        f"{stage}={baseline_counts[stage]}->{case_counts[stage]}"
        f"({case_counts[stage] - baseline_counts[stage]:+d})"
        for stage in stages
    ) or "none"


def _sample_ranges(episodes):
    ranges = {}
    for item in episodes.values():
        samples = item["target_noise"].get("samples", {})
        for label, sample in samples.items():
            for name, value in sample.items():
                if isinstance(value, (int, float)):
                    key = f"{label}.{name}"
                    bounds = ranges.setdefault(key, [float(value), float(value)])
                    bounds[0] = min(bounds[0], float(value))
                    bounds[1] = max(bounds[1], float(value))
    return ranges


def _format_ranges(ranges):
    return ", ".join(
        f"{name}=[{bounds[0]:.9g}, {bounds[1]:.9g}]"
        for name, bounds in sorted(ranges.items())
    ) or "none"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--case", type=Path, action="append", required=True)
    parser.add_argument(
        "--allow-seed-subset",
        action="store_true",
        help="Compare only case seeds; otherwise require identical seed sets.",
    )
    args = parser.parse_args()

    baseline_all = _load(args.baseline)
    for case_root in args.case:
        case = _load(case_root)
        baseline_seeds = set(baseline_all)
        case_seeds = set(case)
        if args.allow_seed_subset:
            missing = case_seeds - baseline_seeds
            if missing:
                raise ValueError(
                    f"case {case_root} contains {len(missing)} seeds absent from baseline"
                )
            paired = case_seeds
        else:
            if case_seeds != baseline_seeds:
                raise ValueError(
                    f"seed mismatch for {case_root}: baseline={len(baseline_seeds)} "
                    f"case={len(case_seeds)} missing={len(baseline_seeds-case_seeds)} "
                    f"extra={len(case_seeds-baseline_seeds)}"
                )
            paired = case_seeds

        baseline = {seed: baseline_all[seed] for seed in paired}
        baseline_success = {seed for seed, item in baseline.items() if item["success"]}
        case_success = {seed for seed, item in case.items() if item["success"]}
        retained = baseline_success & case_success
        new_failures = baseline_success - case_success
        new_successes = case_success - baseline_success
        identical = sum(
            baseline[seed] == case[seed]
            for seed in paired
        )
        print(f"## {case_root}")
        print(f"paired_seeds: {len(paired)}")
        print(
            f"success: {len(case_success)}/{len(paired)} "
            f"({100.0 * len(case_success) / len(paired):.2f}%)"
        )
        print(f"baseline_success: {len(baseline_success)}/{len(paired)}")
        print(f"retained_success: {len(retained)}/{len(baseline_success)}")
        print(f"new_failures: {len(new_failures)}")
        print(f"new_successes: {len(new_successes)}")
        print(f"identical_records: {identical}/{len(paired)}")
        print(f"baseline_failure_stages: {_format_counts(_stage_counts(baseline))}")
        print(f"case_failure_stages: {_format_counts(_stage_counts(case))}")
        print(f"failure_stage_deltas: {_format_stage_comparison(baseline, case)}")
        print(f"sample_ranges: {_format_ranges(_sample_ranges(case))}")
        print()


if __name__ == "__main__":
    main()
