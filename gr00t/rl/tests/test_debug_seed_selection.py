"""Sparse failure reproductions use the same debug workflow and exact denominator."""

import os
import subprocess
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).parents[1] / "scripts/run_auto_open.sh"


def select(mode, values, start=0, end=400):
    source = LAUNCHER.read_text()
    begin = source.index("declare -a ALL_SEEDS=()")
    finish = source.index('SOURCE_FINGERPRINT="$(source_fingerprint)"', begin)
    return subprocess.run(
        ["bash", "-c", source[begin:finish] + '\nprintf "%s\\n" "${ALL_SEEDS[@]}"'],
        env={
            **os.environ,
            "AUTO_OPEN_MODE": mode,
            "DEBUG_SEEDS": values,
            "SEED_START": str(start),
            "SEED_END": str(end),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_explicit_seeds_preserve_order_and_do_not_expand_enclosing_range():
    result = select("debug", "3,167,0,15")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["3", "167", "0", "15"]


@pytest.mark.parametrize("mode", ["debug", "canonical", "official"])
def test_existing_contiguous_ranges_are_unchanged(mode):
    result = select(mode, "", start=1700, end=1703)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["1700", "1701", "1702", "1703"]


@pytest.mark.parametrize("values", ["1,1", "1,01", "1,401", "-1", "1,", "1 2", "1;x"])
def test_invalid_selection_is_rejected(values):
    assert select("debug", values).returncode == 2


@pytest.mark.parametrize("mode", ["canonical", "official"])
def test_explicit_selection_cannot_silently_change_production_denominator(mode):
    assert select(mode, "1,3").returncode == 2
