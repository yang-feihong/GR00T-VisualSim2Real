"""Machine-readable diagnostics emitted only by the AutoOpen debug workflow."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from gr00t.rl.auto_open.geometry import _quat_rotate_np


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _runs(rows, passed):
    selected = [row for row in rows if bool(row["passed"]) is bool(passed)]
    if not selected:
        return []
    result = []
    start = previous = selected[0]
    for row in selected[1:]:
        contiguous = (
            row["frame_index"] == previous["frame_index"] + 1
            and row["stage_step"] == previous["stage_step"] + 1
        )
        if not contiguous:
            result.append(
                {
                    "start_frame": int(start["frame_index"]),
                    "end_frame": int(previous["frame_index"]),
                    "start_stage_step": int(start["stage_step"]),
                    "end_stage_step": int(previous["stage_step"]),
                    "frames": int(previous["frame_index"] - start["frame_index"] + 1),
                }
            )
            start = row
        previous = row
    result.append(
        {
            "start_frame": int(start["frame_index"]),
            "end_frame": int(previous["frame_index"]),
            "start_stage_step": int(start["stage_step"]),
            "end_stage_step": int(previous["stage_step"]),
            "frames": int(previous["frame_index"] - start["frame_index"] + 1),
        }
    )
    return result


def summarize_acceptance_trace(rows):
    """Derive every acceptance summary field from the long-form raw trace."""
    groups = defaultdict(list)
    frame_groups = defaultdict(dict)
    for row in rows:
        key = (row["stage"], int(row["stage_visit"]), row["condition"])
        groups[key].append(row)
        frame_key = (
            row["stage"],
            int(row["stage_visit"]),
            int(row["frame_index"]),
        )
        frame_groups[frame_key][row["condition"]] = bool(row["passed"])

    sole_blockers = defaultdict(list)
    for (stage, visit, frame), conditions in frame_groups.items():
        named = {name: passed for name, passed in conditions.items() if name != "all_conditions"}
        for name, passed in named.items():
            if not passed and all(value for other, value in named.items() if other != name):
                sole_blockers[(stage, visit, name)].append(frame)

    visits = defaultdict(dict)
    for (stage, visit, condition), condition_rows in groups.items():
        condition_rows.sort(key=lambda row: row["frame_index"])
        values = [float(row["value"]) for row in condition_rows]
        failed_rows = [row for row in condition_rows if not row["passed"]]
        minimum = min(range(len(values)), key=values.__getitem__)
        maximum = max(range(len(values)), key=values.__getitem__)
        true_runs = _runs(condition_rows, True)
        false_runs = _runs(condition_rows, False)

        def extremum(rows_to_scan, mode):
            if not rows_to_scan:
                return None
            row = (
                min(rows_to_scan, key=lambda item: item["value"])
                if mode == "min"
                else max(rows_to_scan, key=lambda item: item["value"])
            )
            return {
                "value": _finite(row["value"]),
                "frame_index": int(row["frame_index"]),
                "stage_step": int(row["stage_step"]),
            }

        visits[(stage, visit)][condition] = {
            "comparison": condition_rows[-1]["comparison"],
            "threshold": _finite(condition_rows[-1]["threshold"]),
            "frames": len(condition_rows),
            "true_frames": sum(bool(row["passed"]) for row in condition_rows),
            "false_frames": len(failed_rows),
            "minimum": {
                "value": _finite(values[minimum]),
                "frame_index": int(condition_rows[minimum]["frame_index"]),
                "stage_step": int(condition_rows[minimum]["stage_step"]),
            },
            "maximum": {
                "value": _finite(values[maximum]),
                "frame_index": int(condition_rows[maximum]["frame_index"]),
                "stage_step": int(condition_rows[maximum]["stage_step"]),
            },
            "failed_minimum": extremum(failed_rows, "min"),
            "failed_maximum": extremum(failed_rows, "max"),
            "true_runs": true_runs,
            "false_runs": false_runs,
            "longest_true_run": max(true_runs, key=lambda run: run["frames"], default=None),
            "longest_false_run": max(false_runs, key=lambda run: run["frames"], default=None),
            "sole_blocker_frames": sole_blockers[(stage, visit, condition)],
        }

    return {
        "schema_version": "autoopen_debug_acceptance_trace_v1",
        "rows": len(rows),
        "stage_visits": [
            {
                "stage": stage,
                "stage_visit": visit,
                "conditions": conditions,
            }
            for (stage, visit), conditions in sorted(visits.items())
        ],
    }


def write_debug_trace(root: Path, acceptance_rows, event_rows):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "acceptance_trace.jsonl").open("w", encoding="utf-8") as stream:
        for row in acceptance_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (root / "fsm_events.jsonl").open("w", encoding="utf-8") as stream:
        for row in event_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize_acceptance_trace(acceptance_rows)
    summary["event_rows"] = len(event_rows)
    (root / "acceptance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def apply_pregrasp_metadata(self, candidate):
    """Commit one selected pitch and its fixed-jaw-anchored target height."""
    orientation = np.asarray(candidate["orientation"], dtype=np.float32)
    end_position = np.asarray(candidate["end_position"], dtype=np.float32)
    self.pregrasp_pitch_deg = float(candidate["pitch_deg"])
    self.grasp_orientation_quat = orientation.copy()
    self.reachable_grasp_orientation_quat = orientation.copy()
    self.grasp_ee_offset_world = _quat_rotate_np(
        orientation, self.open_template["tip_capture_center"]
    )
    for name in (
        "center_standoff_ee_world",
        "center_standoff_nominal_world",
        "center_insert_waypoint_world",
        "center_insert_nominal_world",
        "grasp_waypoint_world",
        "grasp_nominal_world",
    ):
        target = getattr(self, name)
        target[2] = end_position[2]
