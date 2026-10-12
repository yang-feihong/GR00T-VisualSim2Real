"""Episode-scoped camera mount variation for standard data collection."""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np


def _rng_for_camera(seed: int, camera_name: str) -> np.random.Generator:
    key = f"auto-open-camera-mount:{camera_name}".encode("utf-8")
    words = np.frombuffer(hashlib.sha256(key).digest()[:16], dtype="<u4")
    sequence = np.random.SeedSequence(
        [int(seed), 0xCA6E_A501, *(int(word) for word in words)]
    )
    return np.random.default_rng(sequence)


def _sample_ball(rng: np.random.Generator, radius: float) -> np.ndarray:
    if radius == 0.0:
        return np.zeros(3, dtype=np.float64)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    return direction * radius * float(rng.random()) ** (1.0 / 3.0)


def _euler_xyz_to_quaternion(euler_xyz) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler_xyz, dtype=np.float64)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dtype=np.float64,
    )


def _quaternion_multiply(left, right) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    result = np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def _axis_angle_quaternion(rotation_vector) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-15:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    scale = math.sin(angle / 2.0) / angle
    return np.asarray(
        (math.cos(angle / 2.0), *(vector * scale)), dtype=np.float64
    )


def _quaternion_to_euler_xyz(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray((roll, pitch, yaw), dtype=np.float64)


def randomize_camera_mounts(
    camera_config,
    *,
    seed: int,
    position_radius_m: float,
    rotation_angle_deg: float,
):
    """Return an actual camera config and traceable mount-error metadata.

    Position errors are expressed in the parent link frame. Orientation errors
    are post-multiplied in the nominal camera frame. Each enabled camera owns an
    independent deterministic random stream and remains fixed for the episode.
    """
    position_radius_m = float(position_radius_m)
    rotation_angle_deg = float(rotation_angle_deg)
    if not math.isfinite(position_radius_m) or position_radius_m < 0.0:
        raise ValueError("camera position-noise radius must be finite and non-negative")
    if not math.isfinite(rotation_angle_deg) or rotation_angle_deg < 0.0:
        raise ValueError("camera rotation-noise angle must be finite and non-negative")
    if rotation_angle_deg > 180.0:
        raise ValueError("camera rotation-noise angle cannot exceed 180 degrees")

    actual_config = copy.deepcopy(camera_config)
    rotation_limit = math.radians(rotation_angle_deg)
    samples = {}
    for camera in actual_config.get("cameras", []):
        if not bool(camera.get("enabled", True)):
            continue
        name = str(camera.get("name", "")).strip()
        if not name:
            raise ValueError("enabled camera must have a non-empty name")
        position_key = "position" if "position" in camera else "local_position"
        rotation_key = "rotation" if "rotation" in camera else "local_rotation"
        if position_key not in camera or rotation_key not in camera:
            raise ValueError(f"camera {name!r} must define position and rotation")

        nominal_position = np.asarray(camera[position_key], dtype=np.float64)
        nominal_euler = np.asarray(camera[rotation_key], dtype=np.float64)
        if nominal_position.shape != (3,) or nominal_euler.shape != (3,):
            raise ValueError(f"camera {name!r} mount pose must contain three values")

        rng = _rng_for_camera(seed, name)
        position_delta = _sample_ball(rng, position_radius_m)
        if rotation_limit == 0.0:
            rotation_vector = np.zeros(3, dtype=np.float64)
        else:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            rotation_vector = axis * float(rng.uniform(0.0, rotation_limit))

        nominal_quaternion = _euler_xyz_to_quaternion(nominal_euler)
        actual_quaternion = _quaternion_multiply(
            nominal_quaternion,
            _axis_angle_quaternion(rotation_vector),
        )
        actual_position = nominal_position + position_delta
        actual_euler = _quaternion_to_euler_xyz(actual_quaternion)
        camera[position_key] = actual_position.tolist()
        camera[rotation_key] = actual_euler.tolist()
        samples[name] = {
            "parent_link": str(camera.get("body_name", "")),
            "position_delta_parent_m": position_delta.tolist(),
            "position_delta_norm_m": float(np.linalg.norm(position_delta)),
            "rotation_delta_camera_rotvec_rad": rotation_vector.tolist(),
            "rotation_delta_angle_deg": math.degrees(
                float(np.linalg.norm(rotation_vector))
            ),
            "nominal_position_parent_m": nominal_position.tolist(),
            "nominal_euler_xyz_rad": nominal_euler.tolist(),
            "actual_position_parent_m": actual_position.tolist(),
            "actual_euler_xyz_rad": actual_euler.tolist(),
        }

    metadata = {
        "enabled": bool(position_radius_m > 0.0 or rotation_angle_deg > 0.0),
        "seed": int(seed),
        "position_distribution": "uniform_volume_ball_in_parent_link_frame",
        "position_radius_m": position_radius_m,
        "rotation_distribution": "uniform_angle_and_axis_in_nominal_camera_frame",
        "rotation_angle_deg": rotation_angle_deg,
        "episode_constant": True,
        "samples": samples,
    }
    actual_config["mount_randomization"] = copy.deepcopy(metadata)
    return actual_config, metadata
