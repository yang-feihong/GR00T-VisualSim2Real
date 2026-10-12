"""Episode-scoped AutoOpen target variation in task-semantic coordinates."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass(frozen=True)
class SemanticTargetNoiseConfig:
    """Maximum absolute variation for independently meaningful task variables."""

    approach_normal_m: float = 0.0
    approach_tangent_m: float = 0.0
    approach_yaw_rad: float = 0.0
    align_panel_normal_m: float = 0.0
    align_panel_tangent_m: float = 0.0
    align_panel_yaw_rad: float = 0.0
    grasp_depth_m: float = 0.0
    grasp_handle_axis_m: float = 0.0
    grasp_closing_m: float = 0.0
    grasp_roll_rad: float = 0.0
    turn_handle_overtravel_rad: float = 0.0
    turn_gap_overtravel_rad: float = 0.0
    panel_depth_m: float = 0.0
    panel_tangent_m: float = 0.0
    panel_vertical_m: float = 0.0
    panel_roll_rad: float = 0.0
    panel_tilt_horizontal_rad: float = 0.0
    panel_tilt_vertical_rad: float = 0.0
    release_lift_m: float = 0.0
    release_retreat_normal_m: float = 0.0
    release_retreat_vertical_m: float = 0.0
    through_forward_m: float = 0.0
    through_tangent_m: float = 0.0
    through_yaw_rad: float = 0.0

    def __post_init__(self):
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"semantic target-noise bound {field.name} must be finite "
                    f"and non-negative, got {value}"
                )

    @classmethod
    def from_assignments(cls, assignments) -> "SemanticTargetNoiseConfig":
        """Parse repeated ``NAME=MAX_ABS`` command-line assignments."""
        values = {}
        known = {field.name for field in fields(cls)}
        for assignment in assignments or ():
            name, separator, raw_value = str(assignment).partition("=")
            if not separator or not name or not raw_value:
                raise ValueError(
                    f"target-noise assignment must be NAME=MAX_ABS, got {assignment!r}"
                )
            if name not in known:
                raise ValueError(
                    f"unknown target-noise variable {name!r}; expected one of "
                    f"{', '.join(sorted(known))}"
                )
            if name in values:
                raise ValueError(f"duplicate target-noise variable {name!r}")
            values[name] = float(raw_value)
        return cls(**values)

    @property
    def enabled(self) -> bool:
        return any(float(value) > 0.0 for value in asdict(self).values())


def _quat_multiply(left, right):
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    result = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return result / max(float(np.linalg.norm(result)), 1.0e-12)


def _rotation_vector_quaternion(rotation_vector):
    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    half = 0.5 * angle
    xyz = vector * (np.sin(half) / angle)
    return np.array([np.cos(half), xyz[0], xyz[1], xyz[2]], dtype=np.float64)


class EpisodeTargetNoise:
    """Sample each named task variable once without touching simulation RNG."""

    def __init__(
        self,
        seed: int,
        config: SemanticTargetNoiseConfig | None = None,
    ):
        self.config = config or SemanticTargetNoiseConfig()
        self._seed = int(seed)
        self._samples: dict[str, dict[str, np.ndarray | float]] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _rng_for_key(self, key: str) -> np.random.Generator:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        key_words = np.frombuffer(digest[:16], dtype="<u4")
        seed_sequence = np.random.SeedSequence(
            [self._seed, 0xA07E_0E15, *(int(word) for word in key_words)]
        )
        return np.random.default_rng(seed_sequence)

    def _signed_vector(self, key, bounds) -> np.ndarray:
        bounds = np.asarray(bounds, dtype=np.float64)
        if not np.any(bounds):
            return np.zeros_like(bounds)
        return self._rng_for_key(key).uniform(-bounds, bounds)

    def perturb_base(
        self,
        label,
        position_xy,
        yaw,
        normal_xy,
        *,
        normal_m,
        tangent_m,
        yaw_rad,
    ):
        bounds = np.asarray([normal_m, tangent_m, yaw_rad], dtype=np.float64)
        if not np.any(bounds):
            return np.asarray(position_xy).reshape(2).copy(), float(yaw)
        position = np.asarray(position_xy, dtype=np.float64).reshape(2)
        key = f"base:{label}"
        sample = self._samples.get(key)
        if sample is None:
            normal = np.asarray(normal_xy, dtype=np.float64).reshape(2)
            normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
            tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
            semantic_delta = self._signed_vector(key, bounds)
            sample = {
                "normal_delta_m": float(semantic_delta[0]),
                "tangent_delta_m": float(semantic_delta[1]),
                "world_position_delta_m": (
                    semantic_delta[0] * normal + semantic_delta[1] * tangent
                ),
                "yaw_delta_rad": float(semantic_delta[2]),
            }
            self._samples[key] = sample
        return (
            (position + sample["world_position_delta_m"]).astype(np.float32),
            float(yaw) + float(sample["yaw_delta_rad"]),
        )

    def perturb_pose(
        self,
        label,
        position,
        orientation,
        semantic_axes_world,
        *,
        position_bounds,
        rotation_bounds,
        position_names,
        rotation_names,
        rotation_semantic_axes_world=None,
    ):
        position_bounds = np.asarray(position_bounds, dtype=np.float64).reshape(3)
        rotation_bounds = np.asarray(rotation_bounds, dtype=np.float64).reshape(3)
        if not np.any(position_bounds) and not np.any(rotation_bounds):
            return (
                np.asarray(position).reshape(3).copy(),
                np.asarray(orientation).reshape(4).copy(),
            )
        target_position = np.asarray(position, dtype=np.float64).reshape(3)
        target_orientation = np.asarray(orientation, dtype=np.float64).reshape(4)
        key = f"pose:{label}"
        sample = self._samples.get(key)
        if sample is None:
            position_delta = self._signed_vector(
                f"{key}:position", position_bounds
            )
            rotation_delta = self._signed_vector(
                f"{key}:rotation", rotation_bounds
            )
            sample = {
                **{
                    f"{name}_delta": float(value)
                    for name, value in zip(position_names, position_delta)
                },
                **{
                    f"{name}_delta": float(value)
                    for name, value in zip(rotation_names, rotation_delta)
                },
                "semantic_position_delta": position_delta,
                "semantic_rotation_delta": rotation_delta,
            }
            self._samples[key] = sample
        position_axes = np.asarray(
            semantic_axes_world, dtype=np.float64
        ).reshape(3, 3)
        rotation_axes = np.asarray(
            semantic_axes_world
            if rotation_semantic_axes_world is None
            else rotation_semantic_axes_world,
            dtype=np.float64,
        ).reshape(3, 3)
        world_position_delta = position_axes @ sample["semantic_position_delta"]
        world_rotation_delta = rotation_axes @ sample["semantic_rotation_delta"]
        rotation_delta = _rotation_vector_quaternion(world_rotation_delta)
        return (
            (target_position + world_position_delta).astype(np.float32),
            _quat_multiply(rotation_delta, target_orientation).astype(np.float32),
        )

    def positive_margin(self, label, maximum):
        maximum = float(maximum)
        if maximum == 0.0:
            return 0.0
        key = f"positive:{label}"
        sample = self._samples.get(key)
        if sample is None:
            sample = {
                "delta": float(self._rng_for_key(key).uniform(0.0, maximum))
            }
            self._samples[key] = sample
        return float(sample["delta"])

    def metadata(self):
        def serializable(value):
            if isinstance(value, np.ndarray):
                return value.astype(float).tolist()
            return float(value)

        return {
            "config": {name: float(value) for name, value in asdict(self.config).items()},
            "samples": {
                label: {name: serializable(value) for name, value in sample.items()}
                for label, sample in sorted(self._samples.items())
            },
        }
