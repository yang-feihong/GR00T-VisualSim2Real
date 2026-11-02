"""Immutable canonical-source checks shared by batch collection and camera replay."""

import json
from pathlib import Path

from gr00t.rl.data_collection.auto_open_writer import file_sha256
from gr00t.rl.pick.dataset import episode_dataset_root


def validate_canonical_source(source):
    source = Path(source)
    summary = json.loads((source / "summary.json").read_text())
    dataset = episode_dataset_root(source)
    report = json.loads((dataset / "meta/acceptance_report.json").read_text())
    if (
        summary["mode"] != "canonical"
        or not summary["success"]
        or report["accepted_episodes"] != 1
        or report["quarantined_episodes"] != 0
    ):
        raise ValueError("Official replay requires a strictly accepted no-camera canonical episode")
    paths = (
        "episode.json",
        "control_record.pt",
        "dataset/data/chunk-000/episode_000000.parquet",
        "dataset/raw_audit/chunk-000/episode_000000.parquet",
        "dataset/raw_contacts/chunk-000/episode_000000.parquet",
    )
    return {
        name: file_sha256(
            dataset / name.removeprefix("dataset/")
            if name.startswith("dataset/")
            else source / name
        )
        for name in paths
    }
