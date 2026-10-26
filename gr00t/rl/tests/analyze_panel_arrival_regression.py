"""Compare strict collection results and every original panel-arrival timeout."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def failures(dataset):
    return [
        f
        for path in dataset.glob("logs/rank*-control/summary.json")
        for f in json.loads(path.read_text())["failures"]
    ]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.current / "acceptance_report.json").read_text())
    cohort = {
        f["seed"]
        for f in failures(args.original)
        if f["stage"] == "MOVE_TO_PANEL_STANDOFF"
        and f["reason"] == "contact_panel_acceptance_timeout"
    }
    current_failures = failures(args.current)
    timed_out = {
        f["seed"]
        for f in current_failures
        if f["stage"] == "MOVE_TO_PANEL_STANDOFF"
        and f["reason"] == "contact_panel_acceptance_timeout"
    }
    episodes = report["episodes"]
    records = [
        dict(
            seed=e["seed"],
            accepted=e["accepted"],
            reasons=e["reasons"],
            panel_arrival_abnormal_contacts=[
                c
                for c in e["contacts"]["abnormal_contacts"]
                if c["stage"] == "MOVE_TO_PANEL_STANDOFF"
            ],
        )
        for e in episodes
    ]
    selected = [r for r in records if r["seed"] in cohort]
    diagnostics = []
    for seed in (2378, 2407):
        shard = args.current / "shards" / f"episode-seed{seed}"
        raw = pq.read_table(next(shard.rglob("raw_audit/**/*.parquet"))).to_pydict()
        ids = np.flatnonzero(np.asarray(raw["autoopen_stage"]) == "MOVE_TO_PANEL_STANDOFF")
        blocks = np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)
        positions = np.asarray(raw["ee_pose_world_xyzw"])[:, :3]
        inputs = np.asarray(raw["ee_inverse_input_world"])
        outputs = np.asarray(raw["ee_inverse_output_world"])
        stages = []
        for block in blocks:
            if not len(block):
                continue
            end = int(block[-1])
            stages.append(
                dict(
                    entry=int(block[0]),
                    end=end,
                    frames=len(block),
                    next_stage=raw["autoopen_stage"][min(end + 1, len(positions) - 1)],
                    terminal_world_command_m=inputs[end].tolist(),
                    terminal_actual_minus_command_m=(positions[end] - inputs[end]).tolist(),
                    terminal_lowlevel_fk_minus_command_m=(outputs[end] - inputs[end]).tolist(),
                )
            )
        diagnostics.append(
            dict(
                seed=seed,
                accepted=next(e["accepted"] for e in episodes if e["seed"] == seed),
                stages=stages,
            )
        )
    ranks = [
        json.loads(p.read_text()) for p in args.current.glob("logs/rank*-control/summary.json")
    ]
    output = dict(
        strict_accepted=report["accepted_episodes"],
        total=report["total_episodes"],
        wall_time_s=report["collection_wall_time_s"],
        primary_rejections=dict(
            Counter(e["reasons"][0].split(":count=")[0] for e in episodes if not e["accepted"])
        ),
        runtime_failures=dict(Counter(f"{f['stage']}:{f['reason']}" for f in current_failures)),
        contact_arrival_timeout_seeds=sorted(timed_out),
        new_contact_arrival_timeout_seeds=sorted(timed_out - cohort),
        panel_arrival_abnormal_episodes=sum(
            bool(r["panel_arrival_abnormal_contacts"]) for r in records
        ),
        first_abnormal_stage_counts=dict(
            Counter(c["stage"] for e in episodes for c in e["contacts"]["abnormal_contacts"][:1])
        ),
        original_timeout_cohort_size=len(selected),
        original_timeout_cohort_accepted=sum(r["accepted"] for r in selected),
        original_timeout_cohort=selected,
        remaining_two_diagnostics=diagnostics,
        num_envs_per_rank=[r["num_envs"] for r in ranks],
        metadata_count=len(list(args.current.glob("logs/rank*-control/seed_*/metadata.json"))),
        shard_report_count=len(list(args.current.glob("shards/*/meta/acceptance_report.json"))),
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "original_timeout_cohort"}, indent=2))


if __name__ == "__main__":
    main()
