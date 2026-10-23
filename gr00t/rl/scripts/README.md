# AutoOpen Official Collection

`run_auto_open.sh` is the only supported AutoOpen execution entry. A single
invocation selects official publication, canonical-only validation, or debug
collection.

## Robot asset

All B2Z1 entry points use `gr00t/rl/data/robots/b2z1_lab30/b2z1.usd`.
It includes the back box at trunk coordinates `(-0.05, 0, 0.0930129412)` m,
rotated +90 degrees about Z, and the Z1 mount at `(0.28, 0, 0.14)` m.
The new USD references the unchanged original 90-degree robot USD and adds only
the box visual, box colliders and the arm mounting translation. The box follows `base_link`.
Original robot joints, masses, inertias, colliders and contact parameters are
preserved. The box uses 63 oriented primitive colliders derived from CAD faces,
including a 4 mm floor slab. Its internal space is not replaced by one solid box.
Small holes and fasteners are approximated. No separate rigid body, movable joint
or estimated box mass is added. `cad/back_box_colliders.json` stores the shapes
in box-local coordinates for both USD and URDF generation.
Its silver appearance is an approximation because the source STEP contains no colors.

Robot geometry lives together under
`gr00t/rl/third_party/visual_wholebody_lowlevel/resources/robots/b2z1/`:
`meshes/` contains the referenced visual and collision meshes. `urdf/b2z1_base.urdf`
is the original 90-degree robot template; `urdf/b2z1.urdf` is the current robot,
including the box. `cad/` contains the STEP, collider descriptions and assembly
metadata. Unused historical URDF variants, DWG and ZIP copies are not retained.
Only one new robot USD is authored; existing USD files and layers are unchanged.

Rebuild using Python with USD, reusing the box mesh:

```bash
python gr00t/rl/scripts/build_b2z1_asset.py
```

The box mesh uses COLLADA (`meshes/back_box.dae`), matching the other robot
visual meshes. Regenerate it from STEP with `python gr00t/rl/scripts/b2z1_mesh.py`
in an environment providing OpenCascade (`cadquery-ocp-novtk`) and NumPy.

The existing USD supplies the mounting position when rebuilding. To deliberately
change it, pass `--arm-mount-xyz X Y Z` in meters. Old robot USD entries are not
redirected to the new asset.
Physical FK/IK reads the mount from the USD fixed joint instead of maintaining
an independent installation offset. URDF-based calculations use that same USD
mount in memory. Restart the process after changing the asset's mounting transform.
The current URDF uses a stable filename; its origin is synchronized to USD.
Checkpoint-defined policy and dataset
goal-reference coordinate conventions are preserved.

## Collision-section diagnostics

Render the real gripper collision hulls, handle collider, contact forces, and
low-level target for any recorded frame:

```bash
python gr00t/rl/scripts/render_auto_open_collision_section.py \
  --dataset datasets/auto_open/debug/<dataset_name> \
  --seed 1700 --frame 904
```

Repeat `--frame` to compare several frames, or use repeated `--time` values in
seconds. Output defaults to the dataset's `diagnostics/collision_sections`
directory.

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

Debug outputs also include one stitched six-camera video per episode under
`six_camera_mosaics/`.

### Pull-door prefix

The implemented pull-door workflow shares the push-door control modules and
currently stops after `RESTORE_GRIPPER`. Run the four diagnostic seeds with:

```bash
DOOR_MOTION=pull AUTO_OPEN_MODE=debug SEED_START=1700 SEED_END=1703 \
bash gr00t/rl/scripts/run_auto_open.sh
```

Handle manipulation consists of `PRESS_HANDLE`, `PUSH_PULL_DOOR`, and
`RETURN_HANDLE`: rotation about the handle axis, rotation about the door hinge,
then rotation about the handle axis. A pull prefix's
`completed_terminal_seeds.txt` records completion of the requested prefix,
not full-task strict data acceptance or a trainable official release.

For complete no-camera push validation without subsequent image generation,
use `AUTO_OPEN_MODE=canonical` with the same seed-range settings.

Each episode shard also contains the standard self-diagnosis contract:

- `debug/acceptance_trace.jsonl`: every named acceptance predicate evaluated
  by every visited stage, with its raw value, comparison, threshold, result,
  true/false streak, stage visit, and stage step.
- `debug/fsm_events.jsonl`: stage transitions and planning/FSM events.
- `debug/acceptance_summary.json`: extrema with exact frames, complete
  true/false intervals, longest runs, and sole-blocker frames, derived only
  from `acceptance_trace.jsonl`.
- `quarantine/raw_audit/...parquet`: commands plus pre-action state `t[k]` and,
  in debug mode, the complete post-action state `t[k+1]` used by acceptance.

Trace rows identify both `action_frame_index=k` and
`state_frame_index=frame_index=k+1`; the matching raw-audit row is selected by
`action_frame_index`, and its `post_*` columns are the exact acceptance state.
This diagnostic recording is passive and does not alter control, physics,
randomization, thresholds, or FSM transitions.

The launcher accepts no command-line arguments. The maintained configuration
surface is `SEED_START`, `SEED_END`, optional `AUTO_OPEN_MODE`, `DOOR_MOTION`,
and `STOP_AFTER_STAGE`; `GPU_IDS` is
only an execution-resource setting. Direct Python invocation is rejected, and
there are no separate no-camera, camera, scalar, video, or replay launchers.
