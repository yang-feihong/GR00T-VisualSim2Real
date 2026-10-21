import json

from gr00t.rl.auto_open.debug_trace import (
    summarize_acceptance_trace,
    write_debug_trace,
)


def _row(frame, name, value, passed):
    return {
        "frame_index": frame,
        "state_frame_index": frame,
        "action_frame_index": frame - 1,
        "stage": "GRASP_HANDLE",
        "stage_id": 5,
        "stage_step": frame + 1,
        "stage_visit": 1,
        "condition": name,
        "value": value,
        "comparison": "<",
        "threshold": 0.08,
        "passed": passed,
        "current_true_streak": 0,
        "current_false_streak": 0,
    }


def test_debug_trace_summary_is_derived_from_per_frame_values():
    rows = []
    speed = (0.02, 0.11, 0.13, 0.03)
    for frame, value in enumerate(speed):
        rows.append(_row(frame, "gripper_speed_rad_s", value, value < 0.08))
        rows.append(_row(frame, "contact", 1.0, True))

    summary = summarize_acceptance_trace(rows)
    conditions = summary["stage_visits"][0]["conditions"]
    speed_summary = conditions["gripper_speed_rad_s"]
    assert speed_summary["maximum"] == {
        "value": 0.13,
        "frame_index": 2,
        "stage_step": 3,
    }
    assert speed_summary["failed_maximum"] == speed_summary["maximum"]
    assert speed_summary["longest_false_run"]["frames"] == 2
    assert speed_summary["sole_blocker_frames"] == [1, 2]


def test_debug_trace_writer_preserves_raw_rows_and_events(tmp_path):
    rows = [_row(0, "gripper_speed_rad_s", 0.12, False)]
    events = [
        {
            "frame_index": 1,
            "state_frame_index": 1,
            "action_frame_index": 0,
            "event": "stage_transition",
        }
    ]
    summary = write_debug_trace(tmp_path, rows, events)
    restored = [
        json.loads(line)
        for line in (tmp_path / "acceptance_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert restored == rows
    assert summary["rows"] == 1
    assert summary["event_rows"] == 1
