"""Shared persistent GPU-rank process supervision for every collection task."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class BatchRankError(RuntimeError):
    def __init__(self, message, exit_code):
        super().__init__(message)
        self.exit_code = 128 - exit_code if exit_code < 0 else exit_code


def bind_rank_parent_lifetime():
    """Managed Linux ranks cannot outlive a killed or crashed supervisor."""
    expected = os.environ.get("AUTO_OPEN_MANAGED_RANK_PARENT_PID")
    if not os.environ.get("AUTO_OPEN_LAUNCH_LEASE") and expected is None:
        return False
    if sys.platform != "linux":
        raise RuntimeError("Managed GPU rank lifetime binding requires Linux")
    import ctypes

    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_PDEATHSIG also covers uncatchable supervisor death, unlike traps.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if parent == 1 or os.getppid() != parent or (expected is not None and parent != int(expected)):
        os.kill(os.getpid(), signal.SIGKILL)
    return True


def check_launch_lease():
    lease = os.environ.get("AUTO_OPEN_LAUNCH_LEASE")
    if lease:
        path = Path(lease)
        if not path.is_file() or time.time() - path.stat().st_mtime > 10:
            raise InterruptedError("Collection batch launcher stopped")


def rank_command(entry, arguments, root, rank, seeds):
    return [
        sys.executable,
        str(entry),
        *arguments,
        "--seed",
        str(seeds[0]),
        "--episode_seeds",
        *map(str, seeds),
        "--batched_envs",
        str(len(seeds)),
        "--standard_dataset_dir",
        str(root / "shards"),
        "--output_dir",
        str(root / "logs" / f"rank{rank}-control"),
    ]


def run_rank_batches(
    root,
    seeds,
    gpu_ids,
    *,
    entry,
    arguments=(),
    affinity=None,
    check_launcher=None,
    require_summary=True,
    popen=subprocess.Popen,
):
    from gr00t.rl.task_runtime import BatchIdentity

    root = Path(root)
    seeds = tuple(seeds)
    assignments = BatchIdentity.partition(seeds, gpu_ids, affinity=affinity)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(parents=True, exist_ok=True)
    (root / "rank_assignment.json").write_text(
        json.dumps(
            {
                "strategy": "asset_affinity" if affinity is not None else "strided",
                "requested_seeds": seeds,
                "ranks": [
                    {
                        "rank": rank,
                        "gpu": gpu,
                        "seeds": rank_seeds,
                        "simultaneous_environments": len(rank_seeds),
                    }
                    for rank, (gpu, rank_seeds) in enumerate(assignments)
                ],
                "affinity": None
                if affinity is None
                else {str(seed): affinity[str(seed)] for seed in seeds},
            },
            indent=2,
        )
        + "\n"
    )
    processes, previous = [], {}
    started = time.monotonic()
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    saved = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        for rank, (gpu, rank_seeds) in enumerate(assignments):
            command = rank_command(entry, arguments, root, rank, rank_seeds)
            with (root / "logs" / f"rank{rank}.console.log").open("w") as log:
                process = popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=str(gpu),
                        AUTO_OPEN_MANAGED_RANK_PARENT_PID=str(os.getpid()),
                    ),
                )
            processes.append(process)
            print(
                f"[workflow] rank={rank} gpu={gpu} pid={process.pid} "
                f"environments={len(rank_seeds)}",
                flush=True,
            )
        unfinished = set(range(len(processes)))
        while unfinished:
            if check_launcher is not None:
                check_launcher()
            check_launch_lease()
            if interrupted:
                raise InterruptedError("Collection batch launcher stopped")
            failures = []
            for rank in tuple(unfinished):
                progress = root / "logs" / f"rank{rank}-control/progress.txt"
                if progress.is_file():
                    text = progress.read_text()
                    if previous.get(rank) != text:
                        print(
                            "".join(f"[rank{rank}] {line}\n" for line in text.splitlines()),
                            end="",
                            flush=True,
                        )
                        previous[rank] = text
                status = processes[rank].poll()
                if status is None:
                    continue
                context = (
                    f"rank={rank} gpu={assignments[rank][0]} exit={status}; "
                    f"see {root / 'logs' / f'rank{rank}.console.log'}"
                )
                if status:
                    failures.append((context, status))
                    continue
                if (
                    require_summary
                    and not (root / "logs" / f"rank{rank}-control/summary.json").is_file()
                ):
                    failures.append((f"{context}: exited without a batch summary", 1))
                    continue
                for seed in assignments[rank][1]:
                    report = root / "shards" / f"episode-seed{seed}/meta/acceptance_report.json"
                    if not report.is_file():
                        failures.append(
                            (f"{context}: seed={seed} has no finalized acceptance shard", 1)
                        )
                        break
                unfinished.remove(rank)
            if failures:
                raise BatchRankError("\n".join(message for message, _ in failures), failures[0][1])
            if unfinished:
                time.sleep(0.5)
        return time.monotonic() - started
    except BaseException as error:
        (root / "runtime_error.json").write_text(
            json.dumps(
                {
                    "error": repr(error),
                    "requested_episodes": len(seeds),
                    "elapsed_s_before_abort": time.monotonic() - started,
                    "execution_model": "persistent_gpu_ranks_batched_environments",
                },
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        for process in processes:
            # A dead rank leader can still leave live children in its group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        for process in processes:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
            finally:
                # wait() only observes the leader; surviving descendants may ignore TERM.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--affinity-manifest", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        run_rank_batches(
            args.root,
            args.seeds,
            [int(x) for x in args.gpu_ids.split(",")],
            entry=args.entry,
            arguments=args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments,
            affinity=json.loads(args.affinity_manifest.read_text())
            if args.affinity_manifest
            else None,
            require_summary=not args.replay,
        )
    except BatchRankError as error:
        print(error, file=sys.stderr, flush=True)
        raise SystemExit(error.exit_code) from error


if __name__ == "__main__":
    main()
