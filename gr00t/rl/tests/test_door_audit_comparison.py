import json
import ast
from pathlib import Path

import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gr00t.rl.pick.inspect_run import compare_door_runs


def record(root, *, command=0.0):
    episode = root / "shards/episode-seed1700"
    meta = episode / "meta"
    meta.mkdir(parents=True)
    (meta / "acceptance_report.json").write_text(json.dumps({"accepted": False}))
    raw = episode / "quarantine/raw_audit/chunk-000"
    raw.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "frame_index": [0, 1, 2],
                "base_command_pre": [[0.0], [command], [0.0]],
                "autoopen_stage": ["APPROACH_HANDLE"] * 3,
                "stable": [True, command == 0.0, True],
                "clearance": [float("nan")] * 3,
            }
        ),
        raw / "episode_000000.parquet",
    )


def test_comparison_reports_first_control_difference_and_equal_nans(tmp_path):
    current, reference = tmp_path / "current", tmp_path / "reference"
    record(current, command=0.15)
    record(reference)
    row = compare_door_runs(current, reference)["episodes"][0]
    diff = row["first_exact_difference"]
    assert diff["base_command_pre"]["action_frame_index"] == 1
    assert diff["stable"]["max_absolute_difference_at_first_frame"] == 1.0
    assert "clearance" not in diff
    assert "autoopen_stage" not in diff


def test_comparison_refuses_missing_seed_set(tmp_path):
    record(tmp_path / "current")
    with pytest.raises(ValueError, match="same nonempty seed set"):
        compare_door_runs(tmp_path / "current", tmp_path / "missing")


@pytest.mark.parametrize("variant", ["A", "B"])
def test_reference_diagnostic_writer_roundtrip_before_simulation(tmp_path, variant):
    root = Path(__file__).parents[3]
    path = root / f"datasets/auto_open/.cache/door_audit_{variant}"
    path /= "gr00t/rl/data_collection/auto_open_writer.py"
    if not path.exists():
        pytest.skip("Isolated door audit worktree not present")
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in {"_fixed_list_array", "_write_raw_audit"}
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_AUDIT_FIXED_LISTS"
                for target in node.targets
            )
        )
    ]
    scope = {"np": np}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    audit = {
        "frame_index": np.arange(3),
        "lowlevel_policy_observation": np.arange(384, dtype=np.float32).reshape(3, 128),
        "physics_substep_dof_vel": np.arange(228, dtype=np.float32).reshape(3, 76),
    }
    output = tmp_path / "raw.parquet"
    scope["_write_raw_audit"](output, audit, 3)
    table = pq.read_table(output).to_pydict()
    for key, values in audit.items():
        np.testing.assert_array_equal(table[key], values)
