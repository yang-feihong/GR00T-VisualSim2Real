"""Lossless compact initialization JSON avoids Python's indented encoder."""

import json
import os
import time
from pathlib import Path

import pytest


def test_initializer_records_keep_values_without_pretty_printing():
    import ast

    from gr00t.rl.pick import task_initialization

    tree = ast.parse(Path(task_initialization.__file__).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dumps"
    ]
    assert len(calls) == 7
    for node in calls:
        assert not any(arg.arg == "indent" for arg in node.keywords)
        separators = next(arg.value for arg in node.keywords if arg.arg == "separators")
        assert ast.literal_eval(separators) == (",", ":")
    value = {
        "accepted": False,
        "checks": {"stable": True},
        "metrics": [1, 0.1, None, [False, 1e-8]],
    }
    assert json.loads(json.dumps(value, indent=2)) == json.loads(
        json.dumps(value, separators=(",", ":"))
    )


@pytest.mark.skipif(
    not os.environ.get("PICK_DIAGNOSTIC_ENCODING_ROOT"), reason="Actual diagnostics required"
)
def test_actual_diagnostic_compact_encoding_is_lossless_and_faster(tmp_path):
    root = Path(os.environ["PICK_DIAGNOSTIC_ENCODING_ROOT"])
    source = root / "logs/rank0-control/seed_0/initial_acceptance.json"
    payload = json.loads(source.read_text())
    before = time.perf_counter()
    original = json.dumps(payload, indent=2)
    indented_s = time.perf_counter() - before
    before = time.perf_counter()
    compact = json.dumps(payload, separators=(",", ":"))
    compact_s = time.perf_counter() - before
    assert json.dumps(json.loads(original), separators=(",", ":")) == compact
    report = {
        "source": str(source),
        "indented_s": indented_s,
        "compact_s": compact_s,
        "indented_bytes": len(original),
        "compact_bytes": len(compact),
        "lossless": True,
    }
    (tmp_path / "diagnostic_encoding_timing.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    assert compact_s < indented_s / 2
