# AutoOpen Fixed Entry

AutoOpen experiments have one maintained entry:

```bash
bash gr00t/rl/scripts/run_auto_open.sh
```

Run it inside the `doorman_train` runtime container. It accepts no command-line
arguments and refuses to overwrite existing output. Select exactly one reviewed
workflow with `AUTO_OPEN_WORKFLOW`:

| Workflow | Required output variable | Purpose |
|---|---|---|
| `capability` | `OUTPUT_ROOT` | Camera-free vectorized capability regression |
| `semantic_collection` | `OUTPUT_ROOT` | Validated episode-level semantic target variation |
| `video_regression` | `RUN_NAME` | Four-seed visual regression |
| `standard_collection` | `DATASET_ROOT` | Two-camera `DATA_STANARD.md` shards |

`standard_collection` accepts a comma-separated `SEEDS` list selected from a
completed capability regression. If omitted, it uses `SEED_START` and
`NUM_EPISODES`. Control, physics, camera, and acceptance options remain fixed.
The dataset root receives a manifest, a machine-readable aggregate acceptance
report, a Chinese aggregate summary, and the measured collection wall time in
addition to each independently validated shard.

Example:

```bash
docker exec doorman_train bash -lc '
  cd /workspace/GR00T-VisualSim2Real &&
  AUTO_OPEN_WORKFLOW=capability \
  OUTPUT_ROOT=logs_rl/auto_open_baseline_1000 \
  bash gr00t/rl/scripts/run_auto_open.sh
'
```

The launcher owns the Python options, GPU sharding, seed mapping, camera mode,
and output validation. Environment variables expose only reviewed run-scale and
destination choices; arbitrary Python argument forwarding is not supported.
