"""Exercise the real phase supervisor with shell stubs, never a simulator."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "run_auto_open.sh"


def _run_phase(tmp_path, *, status=0, force_fork=False, missing=False, missing_summary=False, aggregate_status=0, all_fail=False):
    source = LAUNCHER.read_text()
    phase = "run_phase() {\n" + source.split("run_phase() {\n", 1)[1].split(
        '\nif [[ "${AUTO_OPEN_MODE}" == debug ]]; then\n', 1
    )[0]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text('''#!/bin/bash
set -euo pipefail
if [[ "$1" == - ]]; then
    cat >/dev/null
    for seed in 10 11 12 13 14 15 16 17; do
        [[ -s "${PHASE_OUTPUT_ROOT}/shards/episode-seed${seed}/meta/acceptance_report.json" ]] || exit 88
    done
    touch "${STUB_ROOT}/aggregated"
    exit "${STUB_AGGREGATE_STATUS}"
fi
sleep 0.1
if [[ "${CUDA_VISIBLE_DEVICES}" == 1 || "${STUB_ALL_FAIL}" == 1 ]]; then
    if [[ "${STUB_STATUS}" == 137 ]]; then kill -KILL "$$"; fi
    if [[ "${STUB_STATUS}" != 0 ]]; then exit "${STUB_STATUS}"; fi
    if [[ "${STUB_MISSING}" == 1 ]]; then exit 0; fi
fi
seeds=()
while (( $# )); do
    case "$1" in
        --standard_dataset_dir) shard_root="$2"; shift 2 ;;
        --output_dir) output_dir="$2"; shift 2 ;;
        --episode_seeds)
            shift
            while (( $# )) && [[ "$1" != --* ]]; do
                seeds+=("$1")
                shift
            done ;;
        *) shift ;;
    esac
done
for seed in "${seeds[@]}"; do
    mkdir -p "${shard_root}/episode-seed${seed}/meta"
    printf '{}\n' >"${shard_root}/episode-seed${seed}/meta/acceptance_report.json"
done
if [[ "${STUB_MISSING_SUMMARY}" != 1 || "${CUDA_VISIBLE_DEVICES}" != 1 ]]; then
    mkdir -p "${output_dir}"
    printf '{}\n' >"${output_dir}/summary.json"
fi
''')
    python.chmod(0o755)
    if force_fork:
        real_setsid = shutil.which("setsid")
        assert real_setsid is not None
        wrapper = bin_dir / "setsid"
        wrapper.write_text(f'#!/bin/bash\nexec {shlex.quote(real_setsid)} --fork "$@"\n')
        wrapper.chmod(0o755)
    harness = tmp_path / "phase.sh"
    harness.write_text('''set -euo pipefail
REPO_ROOT="$STUB_ROOT"
ENTRY=stub-entry
MAX_STEPS=1
DOOR_OPEN_IO=in
STOP_AFTER_STAGE=THROUGH
AUTO_OPEN_MODE=canonical
GPU_IDS_ARRAY=(0 1 2 3)
ALL_SEEDS=(10 11 12 13 14 15 16 17)
ACTIVE_PIDS=()
ACTIVE_MONITOR_PID=""
sleep() { command sleep 0.02; }
''' + phase + '''
run_phase "$STUB_ROOT/phase" none "" ALL_SEEDS
touch "$STUB_ROOT/continued"
''')
    return subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, timeout=15,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
             "STUB_ROOT": str(tmp_path), "STUB_STATUS": str(status),
             "STUB_MISSING": str(int(missing)), "STUB_ALL_FAIL": str(int(all_fail)),
             "STUB_MISSING_SUMMARY": str(int(missing_summary)),
             "STUB_AGGREGATE_STATUS": str(aggregate_status)},
    )


@pytest.mark.parametrize("force_fork", [False, True])
@pytest.mark.parametrize("status", [1, 17, 137])
def test_failed_rank_propagates_status_and_skips_aggregation(tmp_path, force_fork, status):
    result = _run_phase(tmp_path, status=status, force_fork=force_fork)
    # util-linux setsid returns the signal number when its forked child dies.
    expected = 9 if force_fork and status == 137 else status
    assert result.returncode == expected, result.stderr
    assert f"rank=1 gpu=1" in result.stderr
    assert f"exit={expected}" in result.stderr
    assert "rank1.console.log" in result.stderr
    assert "aggregation was skipped" in result.stderr
    assert not (tmp_path / "aggregated").exists()
    assert not (tmp_path / "continued").exists()


def test_all_four_ranks_fail_without_aggregation(tmp_path):
    result = _run_phase(tmp_path, status=1, force_fork=True, all_fail=True)
    assert result.returncode == 1
    for rank in range(4):
        assert f"rank={rank} gpu={rank}" in result.stderr
    assert not (tmp_path / "aggregated").exists()


def test_zero_exit_without_finalized_shards_is_failure(tmp_path):
    result = _run_phase(tmp_path, missing=True)
    assert result.returncode != 0
    assert "exited 0 without finalized shard" in result.stderr
    assert not (tmp_path / "aggregated").exists()
    assert not (tmp_path / "continued").exists()


def test_zero_exit_with_shards_but_without_rank_summary_is_failure(tmp_path):
    result = _run_phase(tmp_path, missing_summary=True)
    assert result.returncode != 0
    assert "rank=1 gpu=1 exited 0 without completed summary" in result.stderr
    assert "rank1-control/summary.json" in result.stderr
    assert not (tmp_path / "aggregated").exists()
    assert not (tmp_path / "continued").exists()


@pytest.mark.parametrize("force_fork", [False, True])
def test_success_waits_for_all_four_ranks_before_aggregation(tmp_path, force_fork):
    result = _run_phase(tmp_path, force_fork=force_fork)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "aggregated").exists()
    assert (tmp_path / "continued").exists()
    assert "phase completed" in result.stdout


def test_aggregation_failure_cannot_report_phase_success(tmp_path):
    result = _run_phase(tmp_path, aggregate_status=23)
    assert result.returncode == 23
    assert "phase completed" not in result.stdout
    assert not (tmp_path / "continued").exists()
