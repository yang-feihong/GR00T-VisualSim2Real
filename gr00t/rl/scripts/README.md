# AutoOpen Official Collection

`run_auto_open.sh` is the only supported AutoOpen execution entry. A single
invocation runs one of two maintained workflows.

## Official mode

This is the default. It performs:

1. Run every seed without cameras using one vectorized Isaac Sim process per
   physical GPU.
2. Apply strict per-episode data acceptance and select only accepted seeds.
3. Replay each accepted canonical trajectory without advancing physics or
   invoking the policy/FSM, adding the two training cameras.
4. Re-run strict acceptance, publish the completed dataset under
   `datasets/auto_open/releases/`.

Only the inclusive seed range is required:

```bash
SEED_START=1700 SEED_END=2699 \
bash gr00t/rl/scripts/run_auto_open.sh
```

`GPU_IDS` defaults to `0,1,2,3`. It is an execution-resource setting, not a
dataset behavior switch. The official data semantics are fixed: 4000 maximum
control steps, no cameras in the canonical control run, dual cameras in the
published release, deterministic per-seed camera randomization, and strict
acceptance before and after image generation.

The no-camera source, including rejected shards and their audit reports, is
retained under `datasets/auto_open/canonical/`. A failed camera phase remains
under `datasets/auto_open/work/` and is never linked as an official release.
Completed releases use this naming contract:

```text
auto_open_v1_seeds<start>-<end>_episodes<accepted>_dual_camera_<YYYYMMDDTHHMMSS>
```

Every invocation creates an independent timestamped canonical source and
release. Datasets are additive records; the workflow does not maintain a
`latest` alias and never replaces or implicitly reuses an older collection.

## Debug mode

Debug mode runs the complete seed range directly with all six diagnostic
cameras and then applies the same strict acceptance suite:

```bash
AUTO_OPEN_MODE=debug SEED_START=1700 SEED_END=1703 \
bash gr00t/rl/scripts/run_auto_open.sh
```

Debug outputs are stored under `datasets/auto_open/debug/`. They include an
`accepted_seeds.txt` selection but are never published under `releases/` and
remain independent of every official dataset.

The launcher accepts no command-line arguments. The maintained configuration
surface is `SEED_START`, `SEED_END`, and optional `AUTO_OPEN_MODE`; `GPU_IDS` is
only an execution-resource setting. Direct Python invocation is rejected, and
there are no separate no-camera, camera, scalar, video, or replay launchers.
