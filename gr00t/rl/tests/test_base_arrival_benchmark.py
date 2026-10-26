from gr00t.rl.tests.benchmark_base_arrival import profiles, scenarios, summarize
import json

import pytest
import torch

from gr00t.rl.tests.analyze_base_arrival_benchmark import audit_run


def test_position_sweep_contains_current_baseline_and_valid_thresholds():
    rows = profiles("position")
    assert len(rows) == 16
    assert dict(inner_m=0.05, outer_m=0.15, inner_yaw=0.05, outer_yaw=0.10) in rows
    assert all(r["inner_m"] <= r["outer_m"] for r in rows)


def test_scenarios_are_paired_across_arm_postures():
    rows = scenarios()
    assert len(rows) == 72
    assert [{k: v for k, v in row.items() if k != "arm"} for row in rows[:36]] == [
        {k: v for k, v in row.items() if k != "arm"} for row in rows[36:]
    ]
    assert len({r["bearing_rad"] for r in rows if r["distance_m"] > 0}) == 8
    assert any(r["distance_m"] == 0 for r in rows)


def test_recovery_sweep_covers_hold_controller_and_small_coupled_offsets():
    rows = scenarios(recovery=True)
    assert len(rows) == 128
    assert {r["controller"] for r in rows} == {"track", "hold"}
    assert {abs(r["turn_rad"]) for r in rows} == {0.06, 0.10}
    assert len(profiles("recovery")) == 10


def test_expanded_cases_cover_short_moves_and_large_turns():
    rows = scenarios(True)
    assert len(rows) == 144
    assert {r["distance_m"] for r in rows} == {0.0, 0.05, 0.10, 0.30, 1.0}
    assert max(abs(r["turn_rad"]) for r in rows) == 2.5
    assert profiles("validation", None, (0.025, 0.1))[0] == profiles("position")[-1]
    assert len(profiles("validation", None, (0.015, 0.1))) == 12


def test_accuracy_counts_do_not_hide_failures_or_use_looser_outer_thresholds():
    profile = profiles("position")[0]
    base = dict(
        profile_index=0,
        profile=profile,
        success=True,
        one_pass=True,
        final_distance_m=0.02,
        final_yaw_rad=0.02,
        elapsed_s=3.0,
        restarts=0,
        failure=None,
    )
    rows = [
        base,
        dict(base, success=False, one_pass=False, failure="timeout", final_distance_m=0.0),
        dict(base, final_distance_m=0.12),
    ]
    result = summarize(rows)[0]
    assert result["trials"] == 3
    assert result["one_pass_count"] == 2
    assert result["within_3cm_2deg_count"] == 1
    assert result["within_5cm_3deg_count"] == 1
    assert result["failures"]["timeout"] == 1


@pytest.mark.parametrize("bad_column", [None, "cmd_vx", "effective_vx", "distance", "vx", "wx"])
def test_trace_audit_independently_checks_terminal_window(tmp_path, bad_column):
    columns = [
        "cmd_vx",
        "cmd_vy",
        "cmd_wz",
        "effective_vx",
        "effective_vy",
        "effective_wz",
        "vx",
        "vy",
        "vz",
        "wx",
        "wy",
        "wz",
        "distance",
        "yaw_error",
    ]
    trace = torch.zeros(150, 1, len(columns))
    if bad_column:
        trace[-1, 0, columns.index(bad_column)] = 1.0
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "rank0.json").write_text(
        json.dumps(
            dict(
                trials=[
                    dict(success=True, elapsed_s=3.0, profile=dict(outer_m=0.08, outer_yaw=0.1))
                ]
            )
        )
    )
    torch.save(dict(trace=trace, columns=columns, dt=0.02), tmp_path / "rank0_trace.pt")
    if bad_column:
        with pytest.raises(AssertionError):
            audit_run(tmp_path)
    else:
        assert audit_run(tmp_path) == ({}, 1)
