"""Task-independent GPU batch lifetime, command publication, and policy stepping.

Task adapters own scene geometry, reference generation, and completion criteria.
They do not own simulator stepping or per-seed process scheduling.
"""

import time
from dataclasses import dataclass

import torch


def reset_task_rows(runner, rows, *, preserve_robot=None):
    """Reset selected simulation rows without advancing other rows' policy history."""
    env = runner.env
    device = torch.device(env.device)
    if (
        rows.ndim != 1
        or rows.dtype != torch.long
        or rows.device.type != device.type
        or (device.index is not None and rows.device.index != device.index)
    ):
        raise ValueError("Task reset rows must be a device-side index vector")
    if not len(rows):
        return
    if preserve_robot is not None:
        if (
            preserve_robot.shape != (env.num_envs,)
            or preserve_robot.dtype != torch.bool
            or preserve_robot.device != rows.device
        ):
            raise ValueError("Robot preservation requires one boolean per environment")
        payload_rows = rows[preserve_robot[rows]]
        full_rows = rows[~preserve_robot[rows]]
        if len(payload_rows):
            env.reset_payload_rows(payload_rows)
        if len(full_rows):
            reset_task_rows(runner, full_rows)
        elif len(payload_rows):
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.synchronize_rgb_cameras_after_reset()
        return
    if len(rows) == env.num_envs:
        observations, _ = env.reset()
        runner.obs = observations["policy"] if isinstance(observations, dict) else observations
        return
    mask = torch.ones(env.num_envs, device=rows.device, dtype=torch.bool)
    mask[rows] = False
    keep = torch.where(mask)[0]
    previous = runner.obs[keep].clone()
    history = env.obs_history_buf[keep].clone()
    env._reset_idx(rows)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.synchronize_rgb_cameras_after_reset()
    observations = env.get_observations()
    env.obs_history_buf[keep] = history
    observations[keep] = previous
    runner.obs = observations


def place_environment_origins(scene, origins):
    """Author the same seed-scoped placement before any task assets are spawned."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom

    stage = sim_utils.get_current_stage()
    rows = origins.detach().cpu().tolist()
    for path, origin in zip(scene.env_prim_paths, rows, strict=True):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Missing environment prim {path}")
        xform = UsdGeom.Xformable(prim)
        translations = [
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        op = (
            translations[0]
            if translations
            else xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        )
        op.Set(Gf.Vec3d(*origin))
    scene.cloner._positions = rows
    scene._default_env_origins = origins.clone()


@dataclass(frozen=True)
class BatchIdentity:
    seeds: tuple[int, ...]

    def __post_init__(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("A batch requires nonempty, distinct episode seeds")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Episode seeds must be nonnegative")

    @staticmethod
    def partition(seeds, gpu_ids, *, affinity=None):
        """One persistent rank per GPU; optional immutable asset affinity metadata."""
        seeds, gpu_ids = tuple(seeds), tuple(gpu_ids)
        BatchIdentity(seeds)
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or min(gpu_ids) < 0:
            raise ValueError("GPU ids must be distinct and nonnegative")
        ranks = min(len(seeds), len(gpu_ids))
        if affinity is not None:
            return BatchIdentity._partition_affinity(seeds, gpu_ids[:ranks], affinity)
        return tuple((gpu_ids[rank], seeds[rank::ranks]) for rank in range(ranks))

    @staticmethod
    def _partition_affinity(seeds, gpu_ids, affinity):
        import math

        if not isinstance(affinity, dict):
            raise ValueError("Asset affinity metadata must be a seed-keyed object")
        groups, weights = {}, {}
        for seed in seeds:
            item = affinity.get(str(seed))
            if not isinstance(item, dict) or not isinstance(item.get("key"), str):
                raise ValueError(f"Missing asset affinity metadata for seed {seed}")
            key, weight = item["key"], item.get("weight")
            if (
                not key
                or isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
                or (key in weights and weights[key] != weight)
            ):
                raise ValueError(f"Invalid or inconsistent asset affinity weight for seed {seed}")
            groups.setdefault(key, []).append(seed)
            weights[key] = weight
        count = len(gpu_ids)
        limits = [(len(seeds) + count - rank - 1) // count for rank in range(count)]
        assigned, assets, costs = [[] for _ in gpu_ids], [set() for _ in gpu_ids], [0] * count
        for key in sorted(groups, key=lambda key: (-weights[key], key)):
            pending = groups[key]
            while pending:
                available = [rank for rank in range(count) if len(assigned[rank]) < limits[rank]]
                rank = min(
                    available,
                    key=lambda rank: (
                        key not in assets[rank],
                        costs[rank],
                        len(assigned[rank]),
                        rank,
                    ),
                )
                take = min(len(pending), limits[rank] - len(assigned[rank]))
                assigned[rank].extend(pending[:take])
                pending = pending[take:]
                if key not in assets[rank]:
                    assets[rank].add(key)
                    costs[rank] += weights[key]
        return tuple((gpu, tuple(values)) for gpu, values in zip(gpu_ids, assigned, strict=True))


class BatchedTaskLoop:
    """One policy/physics submission for all environments on a rank, per tick.

    Only the active mask is shared: stages, failures and path cursors remain
    independent per environment. Serialization may iterate rows; control does not.
    """

    def __init__(self, runner, seeds, *, max_steps, success_stage, failure_stage):
        self.runner = runner
        self.env = runner.env
        self.identity = BatchIdentity(tuple(int(seed) for seed in seeds))
        if self.env.num_envs != len(self.identity.seeds):
            raise ValueError("Batch seed count must match the simulator environment count")
        if max_steps < 1 or int(success_stage) == int(failure_stage):
            raise ValueError("Invalid batch budget or terminal stages")
        self.max_steps = int(max_steps)
        self.success_stage, self.failure_stage = int(success_stage), int(failure_stage)
        self.step_counts = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.int32)
        self.first_actions = torch.full(
            (self.env.num_envs, self.env.num_actions), torch.nan, device=self.env.device
        )
        self.policy_physics_s = 0.0
        self.publish_s = 0.0
        self.physics_steps = 0
        self._tick_open = False

    def ticks(self, stage):
        if stage.shape != (self.env.num_envs,) or stage.device != self.step_counts.device:
            raise ValueError("Stage must be a GPU batch vector on the environment device")
        for step in range(self.max_steps):
            active = (stage != self.success_stage) & (stage != self.failure_stage)
            if not bool(active.any()):
                return
            self.step_counts[active] = step + 1
            self._tick_open = True
            yield step, active
            if self._tick_open:
                raise RuntimeError("A task tick finished without its batch policy/physics step")

    def publish(self, buffers):
        started = time.perf_counter()
        buffers.flush(self.env)
        self.env._clip_teleop_targets()
        self.env._update_effective_teleop_inputs(force=True)
        elapsed = time.perf_counter() - started
        self.publish_s += elapsed
        return elapsed

    def step(self):
        if not self._tick_open:
            raise RuntimeError("Physics must be stepped exactly once per batch tick")
        started = time.perf_counter()
        result = self.runner.step_once()
        self.policy_physics_s += time.perf_counter() - started
        if self.physics_steps == 0:
            self.first_actions.copy_(result["low_level_action"])
        self.physics_steps += 1
        self._tick_open = False
        return result

    def run_stages(
        self,
        fsm,
        planner,
        state,
        buffers,
        *,
        timing,
        prepare=None,
        geometry=None,
        capture_before=None,
        capture_after=None,
        report=None,
    ):
        """Execute the existing command/advance/event API for every task graph.

        Scene callbacks only observe geometry, record data and report progress.
        They cannot replace the stage command/transition/planning lifecycle.
        """
        for step, active in self.ticks(fsm.stage):
            if prepare is not None:
                prepare(state, active)
            context = {} if geometry is None else geometry(state)
            stage_before = fsm.stage.clone()
            started = time.perf_counter()
            fsm.command(state, buffers, **context)
            timing["command_s"] += time.perf_counter() - started
            timing["teleop_prepare_s"] += self.publish(buffers)
            if capture_before is not None:
                capture_before(step, active, state)
            started = time.perf_counter()
            result = self.step()
            timing["policy_physics_s"] += time.perf_counter() - started
            if capture_after is not None:
                capture_after(step, active, result)
            state = result["state"]
            context = {} if geometry is None else geometry(state)
            started = time.perf_counter()
            events = fsm.advance(state, **context)
            timing["advance_s"] += time.perf_counter() - started
            if report is not None:
                report(step, stage_before, events)
            started = time.perf_counter()
            planner.plan_events(fsm, state, events)
            timing["planning_s"] += time.perf_counter() - started
            yield step, state, events
