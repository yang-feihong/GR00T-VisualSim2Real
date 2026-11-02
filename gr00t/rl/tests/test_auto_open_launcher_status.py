"""Exercise the real phase supervisor with shell stubs, never a simulator."""

import os
import json
import shlex
import shutil
import subprocess
import sys
import signal
import time
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "run_auto_open.sh"


def _run_phase(
    tmp_path,
    *,
    status=0,
    force_fork=False,
    missing=False,
    missing_summary=False,
    aggregate_status=0,
    all_fail=False,
    cancel=False,
):
    source = LAUNCHER.read_text()
    phase = (
        "run_phase() {\n"
        + source.split("run_phase() {\n", 1)[1].split(
            '\nif [[ "${AUTO_OPEN_MODE}" == debug ]]; then\n', 1
        )[0]
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        "#!/bin/bash\n"
        f'if [[ "$1" == -m ]]; then exec {shlex.quote(sys.executable)} "$@"; fi\n'
        + """
set -euo pipefail
if [[ "$1" == - ]]; then
    cat >/dev/null
    for seed in 10 11 12 13 14 15 16 17; do
        report="${PHASE_OUTPUT_ROOT}/shards/episode-seed${seed}/meta/acceptance_report.json"
        [[ -s "$report" ]] || exit 88
    done
    touch "${STUB_ROOT}/aggregated"
    exit "${STUB_AGGREGATE_STATUS}"
fi
exit 89
"""
    )
    python.chmod(0o755)
    (tmp_path / "stub_entry.py").write_text("""
import argparse
import os
from pathlib import Path
import signal
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--episode_seeds', nargs='+', type=int)
parser.add_argument('--standard_dataset_dir', type=Path)
parser.add_argument('--output_dir', type=Path)
args, _ = parser.parse_known_args()
time.sleep(0.1)
gpu = os.environ['CUDA_VISIBLE_DEVICES']
if os.environ['STUB_HANG'] == '1':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (Path(os.environ['STUB_ROOT']) / f'worker_{gpu}.pid').write_text(str(os.getpid()))
    while True:
        time.sleep(1)
if gpu == '1' or os.environ['STUB_ALL_FAIL'] == '1':
    status = int(os.environ['STUB_STATUS'])
    if status == 137:
        os.kill(os.getpid(), signal.SIGKILL)
    if status:
        sys.exit(status)
    if os.environ['STUB_MISSING'] == '1':
        args.output_dir.mkdir(parents=True)
        (args.output_dir / 'summary.json').write_text('{}')
        sys.exit(0)
for seed in args.episode_seeds:
    report = args.standard_dataset_dir / f'episode-seed{seed}/meta/acceptance_report.json'
    report.parent.mkdir(parents=True)
    report.write_text('{}')
if os.environ['STUB_MISSING_SUMMARY'] != '1' or gpu != '1':
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'summary.json').write_text('{}')
""")
    if force_fork:
        real_setsid = shutil.which("setsid")
        assert real_setsid is not None
        wrapper = bin_dir / "setsid"
        wrapper.write_text(f'#!/bin/bash\nexec {shlex.quote(real_setsid)} --fork "$@"\n')
        wrapper.chmod(0o755)
    harness = tmp_path / "phase.sh"
    harness.write_text(
        """set -euo pipefail
REPO_ROOT="$STUB_ROOT"
ENTRY="$STUB_ROOT/stub_entry.py"
MAX_STEPS=1
DOOR_OPEN_IO=in
STOP_AFTER_STAGE=PASS_THROUGH_DOORWAY
AUTO_OPEN_MODE=canonical
AUTO_OPEN_TASK=door
DOOR_MOTION=push
GPU_IDS_CSV=0,1,2,3
GPU_IDS_ARRAY=(0 1 2 3)
ALL_SEEDS=(10 11 12 13 14 15 16 17)
ACTIVE_PIDS=()
ACTIVE_MONITOR_PID=""
LEASE_WATCHDOG_PID=""
sleep() { command sleep 0.02; }
"""
        + (
            "cleanup() {\n"
            + source.split("\ncleanup() {\n", 1)[1].split("\ntrap cleanup HUP", 1)[0]
            + "\ntrap cleanup EXIT\ntrap 'exit 130' TERM\n"
            if cancel
            else ""
        )
        + phase
        + """
run_phase "$STUB_ROOT/phase" none "" ALL_SEEDS
touch "$STUB_ROOT/continued"
"""
    )
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STUB_ROOT": str(tmp_path),
        "STUB_STATUS": str(status),
        "STUB_MISSING": str(int(missing)),
        "STUB_ALL_FAIL": str(int(all_fail)),
        "STUB_MISSING_SUMMARY": str(int(missing_summary)),
        "PULL_PANEL_CONTACT_BASELINE": "trunk_front_minus10",
        "STUB_AGGREGATE_STATUS": str(aggregate_status),
        "STUB_HANG": str(int(cancel)),
    }
    if not cancel:
        return subprocess.run(
            ["bash", str(harness)], capture_output=True, text=True, timeout=15, env=environment
        )
    process = subprocess.Popen(
        ["bash", str(harness)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 15
        while len(list(tmp_path.glob("worker_*.pid"))) < 4:
            if time.monotonic() > deadline or process.poll() is not None:
                raise RuntimeError("Cancellation fixture did not launch all workers")
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=30)
        survivors = []
        for file in tmp_path.glob("worker_*.pid"):
            pid = int(file.read_text())
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists() and stat.read_text().split(")", 1)[1].split()[0] != "Z":
                survivors.append(pid)
        (tmp_path / "surviving_workers.json").write_text(json.dumps(survivors))
        return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    finally:
        for file in tmp_path.glob("worker_*.pid"):
            try:
                os.killpg(int(file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@pytest.mark.parametrize("force_fork", [False, True])
@pytest.mark.parametrize("status", [1, 17, 137])
def test_failed_rank_propagates_status_and_skips_aggregation(tmp_path, force_fork, status):
    result = _run_phase(tmp_path, status=status, force_fork=force_fork)
    expected = status
    assert result.returncode == expected, result.stderr
    assert "rank=1 gpu=1" in result.stderr
    assert f"exit={-9 if status == 137 else status}" in result.stderr
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
    assert "no finalized acceptance shard" in result.stderr
    assert not (tmp_path / "aggregated").exists()
    assert not (tmp_path / "continued").exists()


def test_zero_exit_with_shards_but_without_rank_summary_is_failure(tmp_path):
    result = _run_phase(tmp_path, missing_summary=True)
    assert result.returncode != 0
    assert "rank=1 gpu=1 exit=0" in result.stderr
    assert "exited without a batch summary" in result.stderr
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


def test_shell_cancellation_allows_supervisor_to_kill_term_ignoring_rank_groups(tmp_path):
    result = _run_phase(tmp_path, cancel=True)
    assert result.returncode == 130, result.stderr
    assert not (tmp_path / "aggregated").exists()
    assert json.loads((tmp_path / "surviving_workers.json").read_text()) == []
