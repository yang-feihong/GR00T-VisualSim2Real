"""Passive PhysX contact auditing for standard AutoOpen datasets.

The audit view is deliberately separate from the contact sensors used by the
controller.  It observes each physics substep and never writes simulation or
control state.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ContactPairMaximum:
    body_a_path: str
    body_b_path: str
    max_normal_impulse: float
    max_contact_force: float
    max_penetration_depth: float
    contact_point_world: tuple[float, float, float]
    contact_normal_world: tuple[float, float, float]
    ccd_triggered: bool = False


def aggregate_contact_pair_maxima(
    rows: Iterable[ContactPairMaximum],
) -> tuple[ContactPairMaximum, ...]:
    """Return one non-empty record per unordered physical body pair."""
    maxima = {}
    for row in rows:
        impulse = float(row.max_normal_impulse)
        force = float(row.max_contact_force)
        penetration = float(row.max_penetration_depth)
        if impulse == 0.0 and force == 0.0 and penetration == 0.0:
            continue

        original_pair = (str(row.body_a_path), str(row.body_b_path))
        pair = tuple(sorted(original_pair))
        normal = tuple(float(value) for value in row.contact_normal_world)
        if pair != original_pair:
            normal = tuple(-value for value in normal)
        current = maxima.get(pair)
        if current is None:
            maxima[pair] = ContactPairMaximum(
                pair[0],
                pair[1],
                impulse,
                force,
                penetration,
                tuple(float(value) for value in row.contact_point_world),
                normal,
                bool(row.ccd_triggered),
            )
            continue

        strongest_is_new = impulse > current.max_normal_impulse
        maxima[pair] = ContactPairMaximum(
            pair[0],
            pair[1],
            max(current.max_normal_impulse, impulse),
            max(current.max_contact_force, force),
            max(current.max_penetration_depth, penetration),
            (
                tuple(float(value) for value in row.contact_point_world)
                if strongest_is_new
                else current.contact_point_world
            ),
            normal if strongest_is_new else current.contact_normal_world,
            bool(current.ccd_triggered or row.ccd_triggered),
        )
    return tuple(maxima[pair] for pair in sorted(maxima))


def _nearest_actor_path(prim):
    """Return the rigid actor owning a collision prim, or the static prim."""
    from pxr import UsdPhysics

    current = prim
    while current and current.IsValid():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current.GetPath().pathString
        parent = current.GetParent()
        if not parent or not parent.IsValid() or parent == current:
            break
        current = parent
    return prim.GetPath().pathString


def _create_contact_views(physics_view, patterns, filters, capacity_per_env):
    """PhysX views require equal filter counts; bucket heterogeneous scenes once."""
    groups = defaultdict(list)
    for pattern, paths in zip(patterns, filters, strict=True):
        groups[len(paths)].append((pattern, tuple(paths)))
    views = []
    for entries in groups.values():
        by_sensor = dict(entries)
        env_ids = {re.search(r"/env_(\d+)/", pattern).group(1) for pattern in by_sensor}
        view = physics_view.create_rigid_contact_view(
            list(by_sensor),
            filter_patterns=[list(paths) for paths in by_sensor.values()],
            max_contact_data_count=int(capacity_per_env) * len(env_ids),
        )
        paths = tuple(str(path) for path in view.sensor_paths)
        views.append((view, paths, tuple(by_sensor[path] for path in paths)))
    return views


class SubstepContactAudit:
    """Collect per-pair maxima across every physics substep in one control tick."""

    def __init__(self, env, *, max_contact_data_per_env: int = 256, extra_filter_paths=()):
        sensor = getattr(env, "contact_sensor", None)
        source_view = getattr(sensor, "contact_physx_view", None)
        if source_view is None:
            raise RuntimeError("the robot contact sensor must be initialized before contact audit")
        stage = env.sim.stage
        self.num_envs = int(env.num_envs)
        filters_by_env = {env_id: set() for env_id in range(self.num_envs)}
        sensors_by_env = {env_id: set() for env_id in range(self.num_envs)}
        global_filter_paths = set(extra_filter_paths)

        from pxr import PhysxSchema, UsdPhysics

        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if path.startswith("/World/ground") and prim.HasAPI(UsdPhysics.CollisionAPI):
                global_filter_paths.add(_nearest_actor_path(prim))
                continue
            match = re.match(r"/World/envs/env_(\d+)(?:/|$)", path)
            if match is None or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            env_id = int(match.group(1))
            if env_id >= self.num_envs:
                continue
            actor_path = _nearest_actor_path(prim)
            filters_by_env[env_id].add(actor_path)
            actor = stage.GetPrimAtPath(actor_path)
            if actor and actor.HasAPI(PhysxSchema.PhysxContactReportAPI):
                sensors_by_env[env_id].add(actor_path)

        for path in source_view.sensor_paths:
            path = str(path)
            match = re.match(r"/World/envs/env_(\d+)(?:/|$)", path)
            if match is not None:
                sensors_by_env[int(match.group(1))].add(path)

        sensor_paths = tuple(
            path for env_id in range(self.num_envs) for path in sorted(sensors_by_env[env_id])
        )
        if not sensor_paths:
            raise RuntimeError("the scene resolved no contact-reporting rigid actors")
        patterns = []
        filter_patterns = []
        for sensor_path in sensor_paths:
            env_id = int(sensor_path.split("/env_", 1)[1].split("/", 1)[0])
            filters = tuple(
                path
                for path in sorted(
                    filters_by_env[env_id] | sensors_by_env[env_id] | global_filter_paths
                )
                if path != sensor_path
            )
            if not filters:
                raise RuntimeError(f"no contact filters found for {sensor_path}")
            patterns.append(sensor_path)
            filter_patterns.append(list(filters))

        physics_view = sensor._physics_sim_view
        self._views = _create_contact_views(
            physics_view, patterns, filter_patterns, max_contact_data_per_env
        )
        self.physics_dt = float(env.physics_dt)
        self.expected_substeps = int(env.cfg.decimation)
        self._substeps = []
        self._peak_contact_data_count = 0
        self._maxima = defaultdict(
            lambda: [
                0.0,
                0.0,
                0.0,
                (float("nan"),) * 3,
                (float("nan"),) * 3,
            ]
        )
        self._scene = None
        self._original_scene_update = None

    @property
    def description(self):
        views = [group[0] for group in self._views]
        filter_counts = sorted({int(view.filter_count) for view in views})
        return {
            "backend": "physx_tensor_contact_view",
            "environment_count": self.num_envs,
            "view_count": len(views),
            "sensor_count": sum(int(view.sensor_count) for view in views),
            "filter_count": filter_counts[0] if len(filter_counts) == 1 else None,
            "filter_counts": filter_counts,
            "max_contact_data_count": sum(int(view.max_contact_data_count) for view in views),
            "peak_contact_data_count": self._peak_contact_data_count,
            "expected_substeps": self.expected_substeps,
        }

    def begin_tick(self):
        self._substeps.clear()
        self._maxima.clear()

    def capture_substep(self):
        groups = []
        for view, _, _ in self._views:
            values = (
                *view.get_contact_data(dt=1.0),
                view.get_contact_force_matrix(dt=self.physics_dt),
            )
            groups.append(tuple(tensor.detach().clone() for tensor in values))
        self._substeps.append(groups)

    def _accumulate_substep(self, tensors, group):
        view, sensor_paths, filter_paths = group
        impulses, points, normals, separations, counts, starts, pair_forces = (
            tensor.cpu().numpy() for tensor in tensors
        )
        impulses = impulses.reshape(-1)
        points = points.reshape(-1, 3)
        normals = normals.reshape(-1, 3)
        separations = separations.reshape(-1)
        counts = counts.reshape(int(view.sensor_count), int(view.filter_count))
        starts = starts.reshape(counts.shape)
        pair_forces = pair_forces.reshape(int(view.sensor_count), int(view.filter_count), 3)
        active = np.argwhere(counts > 0)
        if active.size:
            used = int(np.max(starts[counts > 0] + counts[counts > 0]))
            self._peak_contact_data_count = max(self._peak_contact_data_count, used)
            if used >= int(view.max_contact_data_count):
                raise RuntimeError(
                    "contact audit exhausted its PhysX contact-data capacity: "
                    f"used={used} capacity={int(view.max_contact_data_count)}"
                )
        for sensor_index, filter_index in active:
            count = int(counts[sensor_index, filter_index])
            start = int(starts[sensor_index, filter_index])
            stop = start + count
            patch_impulses = np.abs(impulses[start:stop])
            local_strongest = int(np.argmax(patch_impulses))
            strongest = start + local_strongest
            normal_impulse = float(patch_impulses[local_strongest])
            pair_force = float(np.linalg.norm(pair_forces[sensor_index, filter_index]))
            penetration = float(np.maximum(-separations[start:stop], 0.0).max())
            body_a = sensor_paths[int(sensor_index)]
            body_b = filter_paths[int(sensor_index)][int(filter_index)]
            values = self._maxima[(body_a, body_b)]
            if normal_impulse >= values[0]:
                values[0] = normal_impulse
                values[3] = tuple(float(value) for value in points[strongest])
                values[4] = tuple(float(value) for value in normals[strongest])
            values[1] = max(values[1], pair_force)
            values[2] = max(values[2], penetration)

    def end_tick(self) -> tuple[ContactPairMaximum, ...]:
        by_env = self.end_tick_by_env()
        return tuple(row for env_id in sorted(by_env) for row in by_env[env_id])

    def end_tick_by_env(self) -> dict[int, tuple[ContactPairMaximum, ...]]:
        """Return one deduplicated contact set per vectorized environment."""
        if len(self._substeps) != self.expected_substeps:
            raise RuntimeError(
                f"contact audit observed {len(self._substeps)} physics substeps; "
                f"expected {self.expected_substeps}"
            )
        for groups in self._substeps:
            for tensors, group in zip(groups, self._views, strict=True):
                self._accumulate_substep(tensors, group)
        directional_by_env = defaultdict(list)
        for (body_a, body_b), values in sorted(self._maxima.items()):
            env_ids = {
                int(match.group(1))
                for path in (body_a, body_b)
                if (match := re.search(r"/World/envs/env_(\d+)(?:/|$)", path))
            }
            if len(env_ids) != 1:
                raise RuntimeError(
                    f"contact pair does not resolve to one environment: {body_a}, {body_b}"
                )
            directional_by_env[next(iter(env_ids))].append(
                ContactPairMaximum(body_a, body_b, *values)
            )
        return {
            env_id: aggregate_contact_pair_maxima(rows)
            for env_id, rows in directional_by_env.items()
        }

    def attach(self):
        """Observe scene updates without changing their order or return values."""
        if self._scene is not None:
            raise RuntimeError("contact audit is already attached")
        self._scene = self._find_scene()
        self._original_scene_update = self._scene.update
        audit = self

        def audited_update(*args, **kwargs):
            result = audit._original_scene_update(*args, **kwargs)
            audit.capture_substep()
            return result

        self._scene.update = audited_update

    def detach(self):
        if self._scene is not None and self._original_scene_update is not None:
            self._scene.update = self._original_scene_update
        self._scene = None
        self._original_scene_update = None

    def _find_scene(self):
        # Kept as a method so a test double can override it without importing
        # Isaac Lab.
        return self._env_scene

    @classmethod
    def create(cls, env, **kwargs):
        instance = cls(env, **kwargs)
        instance._env_scene = env.scene
        return instance
