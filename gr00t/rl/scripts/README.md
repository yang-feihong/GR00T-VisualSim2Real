# AutoOpen Collection

`run_auto_open.sh` is the only supported AutoOpen execution entry. It always
runs the vectorized controller and strict data collector. Each physical GPU
owns one Isaac Sim process; all seeds assigned to that GPU are environments in
that process.

```bash
DATASET_ROOT=/workspace/GR00T-VisualSim2Real/logs_rl/example \
CAMERA_MODE=none \
SEED_START=1700 NUM_EPISODES=1000 \
bash gr00t/rl/scripts/run_auto_open.sh
```

`DATASET_ROOT` is required. `CAMERA_MODE` is `none`, `dual`, or `six` and
changes only recorded image modalities. `SEEDS=1700,1701,...` may replace
`SEED_START` and `NUM_EPISODES`; `GPU_IDS` defaults to `0,1,2,3`.

The launcher rejects command-line arguments, duplicate seeds, existing output
directories, and direct Python invocation. Scene sampling, initialization,
targets, policy rows, camera calibration, stage sequence, and strict result are
seed-based rather than rank- or slot-based. Camera mode is observational and
must not alter the trajectory. GPU PhysX does not promise bitwise-identical
contact impulses when the vector batch changes, so post-contact physical state
is not claimed to be byte-identical across different batch sizes.
