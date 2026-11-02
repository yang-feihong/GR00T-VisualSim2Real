"""Door and pick ranks are persistent multi-environment processes, not seed queues."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gr00t.rl.batch_workflow import BatchRankError, bind_rank_parent_lifetime, run_rank_batches


def test_1000_seeds_start_exactly_four_persistent_rank_processes(tmp_path):
    calls = []

    def popen(command, **kwargs):
        rank = len(calls)
        calls.append((command, kwargs))
        begin = command.index("--episode_seeds") + 1
        end = command.index("--batched_envs")
        seeds = [int(value) for value in command[begin:end]]
        assert seeds == list(range(rank, 1000, 4))
        assert command[end + 1] == "250"
        output = Path(command[command.index("--output_dir") + 1])
        output.mkdir()
        (output / "summary.json").write_text("{}")
        for seed in seeds:
            report = tmp_path / "shards" / f"episode-seed{seed}/meta/acceptance_report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"episodes": [{"seed": seed}]}))
        return SimpleNamespace(pid=1000000000 + rank, poll=lambda: 0, wait=lambda **kw: 0)

    run_rank_batches(tmp_path, range(1000), range(4), entry="task.py", popen=popen)
    assert len(calls) == 4
    assert [kw["env"]["CUDA_VISIBLE_DEVICES"] for _, kw in calls] == ["0", "1", "2", "3"]
    assert all(
        kw["env"]["AUTO_OPEN_MANAGED_RANK_PARENT_PID"] == str(os.getpid()) for _, kw in calls
    )


def test_clean_exit_without_all_seed_shards_is_a_batch_failure(tmp_path):
    def popen(command, **kwargs):
        output = Path(command[command.index("--output_dir") + 1])
        output.mkdir()
        (output / "summary.json").write_text("{}")
        return SimpleNamespace(pid=1000000000, poll=lambda: 0, wait=lambda **kw: 0)

    with pytest.raises(RuntimeError, match="no finalized acceptance shard"):
        run_rank_batches(tmp_path, [1, 2], [0], entry="task.py", popen=popen)
    report = json.loads((tmp_path / "runtime_error.json").read_text())
    assert report["requested_episodes"] == 2
    assert report["execution_model"] == "persistent_gpu_ranks_batched_environments"


def test_asset_affinity_launches_four_100_environment_processes_not_seed_queues(tmp_path):
    calls = []
    affinity = {str(seed): {"key": f"room_{seed % 8}", "weight": 1} for seed in range(400)}

    def popen(command, **kwargs):
        begin = command.index("--episode_seeds") + 1
        end = command.index("--batched_envs")
        assigned = list(map(int, command[begin:end]))
        assert len(assigned) == 100 and command[end + 1] == "100"
        output = Path(command[command.index("--output_dir") + 1])
        output.mkdir()
        (output / "summary.json").write_text("{}")
        for seed in assigned:
            report = tmp_path / "shards" / f"episode-seed{seed}/meta/acceptance_report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}")
        calls.append(assigned)
        return SimpleNamespace(pid=1000000000 + len(calls), poll=lambda: 0, wait=lambda **kw: 0)

    run_rank_batches(
        tmp_path, range(400), range(4), entry="task.py", affinity=affinity, popen=popen
    )
    assert len(calls) == 4
    assert sorted(seed for assigned in calls for seed in assigned) == list(range(400))
    manifest = json.loads((tmp_path / "rank_assignment.json").read_text())
    assert manifest["strategy"] == "asset_affinity"
    assert [rank["simultaneous_environments"] for rank in manifest["ranks"]] == [100] * 4


@pytest.mark.parametrize("status,expected", [(17, 17), (-9, 137), (137, 137)])
def test_rank_failure_preserves_exit_status_and_cleans_exited_group(
    tmp_path, monkeypatch, status, expected
):
    signals = []
    monkeypatch.setattr(
        "gr00t.rl.batch_workflow.os.killpg", lambda pid, sig: signals.append((pid, sig))
    )
    process = SimpleNamespace(pid=12345, poll=lambda: status, wait=lambda **kw: status)
    with pytest.raises(BatchRankError) as caught:
        run_rank_batches(tmp_path, [1], [2], entry="task.py", popen=lambda *a, **kw: process)
    assert caught.value.exit_code == expected
    assert "gpu=2" in str(caught.value)
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]


def test_unmanaged_process_does_not_bind_its_parent(monkeypatch):
    monkeypatch.delenv("AUTO_OPEN_LAUNCH_LEASE", raising=False)
    monkeypatch.delenv("AUTO_OPEN_MANAGED_RANK_PARENT_PID", raising=False)
    assert not bind_rank_parent_lifetime()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death signal")
@pytest.mark.parametrize("with_lease", [False, True])
def test_rank_dies_when_supervisor_is_killed_even_if_rank_ignores_term(tmp_path, with_lease):
    ready = tmp_path / "ready"
    worker = (
        "from gr00t.rl.batch_workflow import bind_rank_parent_lifetime; "
        "import signal,time; from pathlib import Path; "
        "bind_rank_parent_lifetime(); signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text('ready'); time.sleep(60)"
    )
    supervisor = (
        "import subprocess,sys,os; "
        f"p=subprocess.Popen([sys.executable,'-c',{worker!r}], "
        "env=dict(os.environ,AUTO_OPEN_MANAGED_RANK_PARENT_PID=str(os.getpid()),"
        f"AUTO_OPEN_LAUNCH_LEASE={'managed-test'!r} if {with_lease!r} else '')); "
        "print(p.pid,flush=True); p.wait()"
    )
    parent = subprocess.Popen([sys.executable, "-c", supervisor], stdout=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "worker did not bind its supervisor lifetime"
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child}/status")
            if not status.exists() or "State:\tZ" in status.read_text():
                break
            time.sleep(0.02)
        else:
            pytest.fail("rank survived its supervisor's SIGKILL")
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
        parent.stdout.close()


def test_pick_preparation_binds_immediate_parent_not_rank_supervisor(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from gr00t.rl.pick import scene

    monkeypatch.setenv("AUTO_OPEN_MANAGED_RANK_PARENT_PID", "99999999")
    monkeypatch.setattr(scene.Path, "glob", lambda self, pattern: iter([tmp_path]))

    def run(command, *, check, env):
        assert check
        assert env["AUTO_OPEN_MANAGED_RANK_PARENT_PID"] == str(os.getpid())
        (tmp_path / "prepared_scenes.json").write_text('[{"seed": 1, "episode": {}}]')

    monkeypatch.setattr(scene.subprocess, "run", run)
    args = SimpleNamespace(
        episode_seeds=[1],
        seed=1,
        output_dir=tmp_path,
        task_kind="pick_basket",
        pick_scenes_root=tmp_path,
        pick_support="any",
        pick_episode_manifest=None,
        pick_scene_id=None,
        pick_asset=None,
        replay_source_dataset_dir=None,
    )
    assert scene.prepare_shared_rank(args)
