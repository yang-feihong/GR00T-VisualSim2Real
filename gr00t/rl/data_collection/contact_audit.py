"""Passive PhysX contact auditing for standard AutoOpen datasets.

The audit view is deliberately separate from the contact sensors used by the
controller.  It observes each physics substep and never writes simulation or
control state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch


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


def discover_contact_filter_paths(stage, env_id: int) -> tuple[str, ...]:
    """Find same-environment rigid/static actors relevant to collision audit."""
    from pxr import UsdPhysics

    env_prefix = f"/World/envs/env_{int(env_id)}/"
    paths = set()
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not (path.startswith(env_prefix) or path.startswith("/World/ground")):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            paths.add(_nearest_actor_path(prim))
    return tuple(sorted(paths))


def discover_contact_sensor_paths(stage, env_id: int) -> tuple[str, ...]:
    """Find rigid actors with contact reporting already enabled by the scene."""
    from pxr import PhysxSchema, UsdPhysics

    env_prefix = f"/World/envs/env_{int(env_id)}/"
    paths = set()
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith(env_prefix) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        actor_path = _nearest_actor_path(prim)
        actor = stage.GetPrimAtPath(actor_path)
        if actor and actor.HasAPI(PhysxSchema.PhysxContactReportAPI):
            paths.add(actor_path)
    return tuple(sorted(paths))


class SubstepContactAudit:
    """Collect per-pair maxima across every physics substep in one control tick."""

    def __init__(self, env, *, max_contact_data_per_sensor: int = 256):
        sensor = getattr(env, "contact_sensor", None)
        source_view = getattr(sensor, "contact_physx_view", None)
        if source_view is None:
            raise RuntimeError("the robot contact sensor must be initialized before contact audit")
        stage = env.sim.stage
        env_ids = tuple(range(int(env.num_envs)))
        source_sensor_paths = tuple(str(path) for path in source_view.sensor_paths)
        filters_by_env = {
            env_id: discover_contact_filter_paths(stage, env_id) for env_id in env_ids
        }
        sensors_by_env = {
            env_id: tuple(sorted(set(discover_contact_sensor_paths(stage, env_id)) | {
                path
                for path in source_sensor_paths
                if path.startswith(f"/World/envs/env_{env_id}/")
            }))
            for env_id in env_ids
        }
        sensor_paths = tuple(
            path for env_id in env_ids for path in sensors_by_env[env_id]
        )
        if not sensor_paths:
            raise RuntimeError("the scene resolved no contact-reporting rigid actors")
        patterns = []
        filter_patterns = []
        for sensor_path in sensor_paths:
            env_token = sensor_path.split("/env_", 1)[1].split("/", 1)[0]
            env_id = int(env_token)
            filters = tuple(
                path
                for path in sorted(
                    set(filters_by_env[env_id]) | set(sensors_by_env[env_id])
                )
                if path != sensor_path
            )
            if not filters:
                raise RuntimeError(f"no contact filters found for {sensor_path}")
            patterns.append(sensor_path)
            filter_patterns.append(list(filters))

        filter_counts = {len(paths) for paths in filter_patterns}
        if len(filter_counts) != 1:
            raise RuntimeError(
                f"contact filter count differs across environments: {sorted(filter_counts)}"
            )
        physics_view = sensor._physics_sim_view
        self.view = physics_view.create_rigid_contact_view(
            patterns,
            filter_patterns=filter_patterns,
            max_contact_data_count=(
                int(max_contact_data_per_sensor) * len(sensor_paths)
            ),
        )
        self.physics_dt = float(env.physics_dt)
        self.sensor_paths = tuple(str(path) for path in self.view.sensor_paths)
        self.filter_paths_by_sensor = tuple(tuple(paths) for paths in filter_patterns)
        self.expected_substeps = int(env.cfg.decimation)
        self._substeps = 0
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
        return {
            "sensor_count": int(self.view.sensor_count),
            "filter_count": int(self.view.filter_count),
            "max_contact_data_count": int(self.view.max_contact_data_count),
            "expected_substeps": self.expected_substeps,
            "sensor_paths": list(self.sensor_paths),
            "filter_paths_by_sensor": [list(paths) for paths in self.filter_paths_by_sensor],
        }

    def begin_tick(self):
        self._substeps = 0
        self._maxima.clear()

    def capture_substep(self):
        impulses, points, normals, separations, counts, starts = (
            self.view.get_contact_data(dt=1.0)
        )
        impulses = impulses.reshape(-1).clone()
        points = points.reshape(-1, 3).clone()
        normals = normals.reshape(-1, 3).clone()
        separations = separations.reshape(-1).clone()
        counts = counts.clone()
        starts = starts.clone()
        pair_forces = self.view.get_contact_force_matrix(
            dt=self.physics_dt
        ).reshape(int(self.view.sensor_count), int(self.view.filter_count), 3)
        counts = counts.reshape(int(self.view.sensor_count), int(self.view.filter_count))
        starts = starts.reshape_as(counts)
        active = torch.nonzero(counts > 0, as_tuple=False)
        for sensor_index, filter_index in active.detach().cpu().tolist():
            count = int(counts[sensor_index, filter_index].item())
            start = int(starts[sensor_index, filter_index].item())
            stop = start + count
            patch_impulses = torch.abs(impulses[start:stop])
            strongest = start + int(torch.argmax(patch_impulses).item())
            normal_impulse = float(patch_impulses.max().item())
            pair_force = float(
                torch.linalg.vector_norm(pair_forces[sensor_index, filter_index]).item()
            )
            penetration = float(torch.clamp(-separations[start:stop], min=0).max().item())
            body_a = self.sensor_paths[sensor_index]
            body_b = self.filter_paths_by_sensor[sensor_index][filter_index]
            values = self._maxima[(body_a, body_b)]
            if normal_impulse >= values[0]:
                values[0] = normal_impulse
                values[3] = tuple(float(value) for value in points[strongest].tolist())
                values[4] = tuple(float(value) for value in normals[strongest].tolist())
            values[1] = max(values[1], pair_force)
            values[2] = max(values[2], penetration)
        self._substeps += 1

    def end_tick(self) -> tuple[ContactPairMaximum, ...]:
        if self._substeps != self.expected_substeps:
            raise RuntimeError(
                f"contact audit observed {self._substeps} physics substeps; "
                f"expected {self.expected_substeps}"
            )
        return tuple(
            ContactPairMaximum(body_a, body_b, *values)
            for (body_a, body_b), values in sorted(self._maxima.items())
        )

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
