# AutoOpen Official Collection

`run_auto_open.sh` is the only supported AutoOpen execution entry. A single
invocation selects official publication, canonical-only validation, or debug
collection.

Online stage planning uses device-side path banks, batched inverse mapping,
and masked candidate selection. Controller objects supply immutable asset and
configuration data at initialization; episode-specific geometry, collision
checks, and planning decisions do not dispatch per-environment CPU planners.
File encoding, progress display, and final audit serialization remain host I/O.

Training fisheye cameras randomize only the invalid exterior of the effective
image circle. The episode-static housing appearance is independent of task
randomization, preserves all valid and feather-band pixels, and is recorded in
episode provenance. Diagnostic cameras are unchanged. See
[fisheye border configuration](../../../docs/fisheye_border_augmentation.md).

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

### Pull-door workflow

The implemented pull-door workflow shares the push-door control modules and
continues from `RESTORE_GRIPPER` through `PULL_CLEAR_SWEEP`, then
`PULL_INSERT` -> `PULL_OPEN` -> `PULL_ARM_RETRACT` -> `PULL_BASE_ADVANCE`.
The first pry must open the door at least 30 degrees. After retraction,
`PULL_BASE_ADVANCE` precomputes a trajectory for the front shoulder nearest the
wall, preserving its wall-normal coordinate. For shoulder A and hinge B, its
heading angle is arcsin((body width + 10 cm) / |AB|) plus the AB/wall angle,
clamped between the entry heading angle and 90 degrees. The trajectory is
independent of the measured door angle. Reference samples advance only while
the current sample heading angle is below the door angle; otherwise the
reference holds, with no backward index changes or actual-tracking gate.
Base travel starts immediately after accepted retraction, even while the door
is rotating. Completion requires the reference gate to hold and the shared 25-frame zero-command
stability window to also satisfy door speed below 2 degrees/s throughout.
After settling, the workflow enters the shared `CONTACT_PANEL` push interface
instead of planning another pry. Contact planning and measured finger clearance
use the same selected panel-side normal. Further opening repeats
`CONTACT_PANEL` -> `PUSH_OPEN` -> `PULL_ARM_RETRACT` -> `PULL_BASE_ADVANCE`;
it never uses push-door `BASE_ADVANCE` for pull-door repositioning.
All pull-door retractions use the same entry-planned GPU-inverse restore path
and measured arm-stability acceptance as the first pry retraction.
The shared contact grid keeps nine height offsets at 5 cm spacing. Its tangent
search extends +/-40 cm for pull-door stations, retaining +/-20 cm for push-door
stations, with the same panel-edge, IK, inverse-residual and joint-margin checks.
For pull-door stations, the default tangent search baseline is the projection
onto the current panel of the point 10 cm behind the trunk collider's front,
along the actual base yaw (38 cm ahead of base on the current asset). Tangent
offset search remains active around this baseline; push-door baselines are unchanged.
`PULL_PANEL_CONTACT_BASELINE` selects `base_forward` (base +20 cm), `trunk_front`
(frontmost trunk collider x), or `trunk_front_minus10` (that front minus 10 cm).
The selected mode is passed through the same launcher and stored in `workflow.json`.
The default is `trunk_front_minus10`.
Shared push/pull contact selection rejects candidate heights whose complete
closed-jaw collider height interval overlaps the current handle capsule height
band, expanded by 1 cm. The handle band follows its actual pose. This filter is
applied before the existing batched IK, without adding solver batches. It is an
endpoint-selection constraint, not an approach-path collision guarantee.
At the shared 89-degree opening acceptance, retraction finishes before
the pull workflow enters `THROUGH` directly, without a frame-alignment stage.
After opening is accepted, the pull workflow leaves the pushing stages for
retraction and traversal; it does not keep advancing the push target.
The original push-door angle and angular-speed protection is unchanged.
If the door reaches the same opening threshold during the initial
base motion, it hands off directly to `THROUGH` without a stationary window;
the final terminal stationary acceptance remains unchanged.
Insertion candidates are geometrically screened and solved together through one
GPU inverse batch. Base motion checks separate articulated rigid-body envelopes,
including the legs and restored arm, and resamples unsafe stations within the
configured angular range. See [pull validation](../../../docs/auto_open_pull_panel_validation_zh.md).
Run the four diagnostic seeds with:

```bash
DOOR_MOTION=pull AUTO_OPEN_MODE=debug SEED_START=1700 SEED_END=1703 \
bash gr00t/rl/scripts/run_auto_open.sh
```

Handle manipulation consists of `PRESS_HANDLE`, `PUSH_PULL_DOOR`, and
`RETURN_HANDLE`: rotation about the handle axis, rotation about the door hinge,
then rotation about the handle axis. A pull prefix's
`completed_terminal_seeds.txt` records completion of the requested prefix,
not full-task strict data acceptance or a trainable official release.

`PULL_CLEAR_SWEEP` holds the restored arm joints and closed gripper, moving the
base toward the free-end side of the door and facing the open gap. It uses the
same `_track_station` interface and arrival profile as APPROACH: 2/10 cm inner/
outer position thresholds, 0.05/0.15 rad yaw thresholds, and a 25-frame zero-
command stability window. No separate locomotion controller is introduced.

The planner reads enabled base_link colliders and the static opening boundary
from the lintel. It projects the rotated trunk colliders onto the frame-width
direction. Their entire projection must be beyond the free-side jamb
(no projection overlap). No door sweep disk or whole-robot sphere is used.
The same projection operation on the wall-normal axis requires the complete
trunk to stay at least 2 cm in front of the robot-facing wall surface, read
from the static jamb collision geometry. One shared two-axis calculation
is used for both candidate poses and actual-pose acceptance.
The target remains exactly 0.77 m from the open-gap midpoint. Candidate
positions on the robot-side/free-end quarter-circle face the gap, including
sampled yaw noise. Planning reserves a fixed additional 5 cm inside both
projection constraints, without increasing the 0.77 m distance. This margin
is independent of the locomotion controller's position/yaw thresholds.
If no joint position/yaw candidate satisfies both, planning reports failure;
it never silently expands the distance or relaxes either constraint.
Acceptance checks the actual trunk projection using the full measured base
orientation, continuously for 25 zero-command frames, plus shared arrival
and stability. This is an endpoint condition, not a path collision certificate.

Targets are sampled once on stage entry, deterministically per episode seed:
positions are selected within a feasible arc of at most 20 cm, centered on
the midpoint of the jointly feasible angular interval; yaw noise is +/-0.06 rad. There is no
fixed 45-degree center. Position randomization never changes the target gap distance.
Position/yaw command and acceptance use the same
sampled target. The nominal yaw points at the midpoint between the closed
free edge and the current open free edge. Geometry and randomization defaults
are maintained in `gr00t/rl/auto_open/pull_station.py`; debug metadata records
the sampled target, frame boundary, planned clearance, and acceptance predicates.

`PULL_INSERT` keeps the gripper closed, approaches outside the free panel edge,
passes behind the panel, and inserts the outside jaw face at a 2 mm standoff.
This is a positioning stage: acceptance checks pose, clearance, and stability,
not physical contact or contact force. Contact forms during the following arc.
Contact targets come from measured closed-jaw collision geometry. The entry
path is checked against panel and wall envelopes before execution.

`PULL_OPEN` moves the EE position along a hinge-centered arc while holding its
entry world orientation fixed. Contact can migrate from the jaw's outside face
toward its tip as the door opens; orientation does not follow the panel.
It shares PUSH_OPEN's PI door-speed feedback (20 deg/s target, 30 deg/s limit,
2-degree maximum lead) and 0.7 m XY comfort-workspace test. The complete arc
is mapped once through the common GPU inverse at stage entry and interpolated
online according to the feedback hinge target. No per-frame inverse solve or
per-point independent noise is used. At the comfort boundary, the shared
restore planner retracts the closed gripper; no pull-specific retract detour
is added. Completion additionally requires a measured door angle of at least
30 degrees; reaching the workspace boundary alone cannot pass this stage.
The following wall-parallel shoulder motion uses the same hysteretic arrival
tracker as APPROACH. Defaults live in `auto_open/panel_motion.py` and
`auto_open/pull_station.py`.

Use `STOP_AFTER_STAGE=RESTORE_GRIPPER` or `STOP_AFTER_STAGE=PULL_CLEAR_SWEEP`
to retain an earlier pull prefix; `STOP_AFTER_STAGE=PULL_INSERT` stops before
the feedback opening motion; `STOP_AFTER_STAGE=PULL_OPEN` stops after the first
opening cycle rather than completing traversal. `STOP_AFTER_STAGE=PULL_BASE_ADVANCE`
includes retraction and settling beside the panel, stopping before shared pushing.
Partial workflows support `canonical` (no cameras) and `debug` (six cameras),
but cannot be published by `official` mode. Full pull-door runs stop at `THROUGH`
by default. Their physical validation status is recorded separately in
`docs/auto_open_pull_panel_validation_zh.md`.

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
