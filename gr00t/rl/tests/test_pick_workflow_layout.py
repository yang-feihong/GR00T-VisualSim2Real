"""Shared collection layout, strict/control result separation and relocation."""

import os

from gr00t.rl.pick.batch_report import control_outcomes, report_batch, write_json
from gr00t.rl.pick.official import validate_canonical_source


def test_control_failures_and_strict_rejections_are_not_conflated(tmp_path):
    for seed, success, stage in ((1700, True, None), (1701, False, "MOVE_TO_GRASP")):
        write_json(
            tmp_path / f"logs/rank0-control/seed_{seed}/summary.json",
            {
                "seed": seed,
                "success": success,
                "failure": None if success else "no_safe_candidate",
                "failed_stage": stage,
                "scene_id": f"room{seed}",
                "target_semantics": {"object_category": "lemon", "support_category": "floor"},
            },
        )
    write_json(
        tmp_path / "acceptance_report.json",
        {"total_episodes": 2, "accepted_episodes": 0, "collection_wall_time_s": 10},
    )
    report = report_batch(tmp_path)
    assert report["complete"]
    assert report["control_success_rate"] == 0.5
    assert report["strict_acceptance_rate"] == 0
    assert report["failed_stage_counts"] == {"MOVE_TO_GRASP": 1}
    assert report["target_category_counts"] == {"lemon": 2}


def test_interrupted_batch_does_not_report_a_completed_success_rate(tmp_path):
    write_json(tmp_path / "run_request.json", {"requested_episodes": 1000})
    write_json(tmp_path / "logs/rank0-control/seed_0/summary.json", {"success": True})
    report = report_batch(tmp_path)
    assert not report["complete"]
    assert report["strict_acceptance_rate"] is None
    assert report["completed_episodes"] == 1


def test_relocated_canonical_shards_hash_data_not_absolute_paths(tmp_path):
    root = tmp_path / "canonical"
    run = root / "logs/rank0-control/seed_1700"
    shard = root / "shards/episode-seed1700"
    write_json(
        run / "summary.json",
        {
            "mode": "canonical",
            "success": True,
            "data_collection": {"dataset": os.path.relpath(shard, run)},
        },
    )
    write_json(run / "episode.json", {"seed": 1700})
    (run / "control_record.pt").write_bytes(b"state")
    write_json(
        shard / "meta/acceptance_report.json", {"accepted_episodes": 1, "quarantined_episodes": 0}
    )
    for name in ("data", "raw_audit", "raw_contacts"):
        table = shard / name / "chunk-000/episode_000000.parquet"
        table.parent.mkdir(parents=True)
        table.write_bytes(b"immutable")
    fingerprint = validate_canonical_source(run)
    relocated = tmp_path / "renamed"
    root.rename(relocated)
    assert validate_canonical_source(relocated / "logs/rank0-control/seed_1700") == fingerprint
    assert control_outcomes(relocated)["control_success_episodes"] == 1
