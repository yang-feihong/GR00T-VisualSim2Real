"""Episode-scoped camera variation for standard data collection."""

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


def _rng_for_camera_lens(seed: int, camera_name: str) -> np.random.Generator:
    key = f"auto-open-camera-lens:{camera_name}".encode("utf-8")
    words = np.frombuffer(hashlib.sha256(key).digest()[:16], dtype="<u4")
    sequence = np.random.SeedSequence(
        [int(seed), 0xCA6E_1E45, *(int(word) for word in words)]
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


def _opencv_fisheye_half_angle(theta_distorted: float, coefficients) -> float:
    coefficients = np.asarray(coefficients, dtype=np.float64)

    def distort(theta):
        theta2 = theta * theta
        polynomial = 1.0
        power = theta2
        for coefficient in coefficients:
            polynomial += float(coefficient) * power
            power *= theta2
        return theta * polynomial

    # Use the first monotonic branch of the calibrated projection.
    samples = np.linspace(0.0, math.pi, 4097)
    distorted = np.asarray([distort(value) for value in samples])
    turning = np.flatnonzero(np.diff(distorted) <= 0.0)
    upper_index = int(turning[0] + 1) if len(turning) else len(samples) - 1
    if theta_distorted > distorted[upper_index]:
        return float(samples[upper_index])
    low, high = 0.0, float(samples[upper_index])
    for _ in range(60):
        middle = 0.5 * (low + high)
        if distort(middle) < theta_distorted:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _opencv_fisheye_horizontal_fov_deg(width, fx, cx, coefficients) -> float:
    left = _opencv_fisheye_half_angle(float(cx) / float(fx), coefficients)
    right = _opencv_fisheye_half_angle(
        (float(width) - float(cx)) / float(fx), coefficients
    )
    return math.degrees(left + right)


def randomize_camera_mounts(
    camera_config,
    *,
    seed: int,
    position_radius_m: float,
    rotation_angle_deg: float,
    focal_scale_fraction: float = 0.0,
    principal_point_radius_px: float = 0.0,
    distortion_relative_fraction: float = 0.0,
):
    """Return an actual camera config and traceable camera-error metadata.

    Position errors are expressed in the parent link frame. Orientation errors
    are post-multiplied in the nominal camera frame. Each enabled camera owns an
    independent deterministic random stream and remains fixed for the episode.
    """
    position_radius_m = float(position_radius_m)
    rotation_angle_deg = float(rotation_angle_deg)
    focal_scale_fraction = float(focal_scale_fraction)
    principal_point_radius_px = float(principal_point_radius_px)
    distortion_relative_fraction = float(distortion_relative_fraction)
    if not math.isfinite(position_radius_m) or position_radius_m < 0.0:
        raise ValueError("camera position-noise radius must be finite and non-negative")
    if not math.isfinite(rotation_angle_deg) or rotation_angle_deg < 0.0:
        raise ValueError("camera rotation-noise angle must be finite and non-negative")
    if rotation_angle_deg > 180.0:
        raise ValueError("camera rotation-noise angle cannot exceed 180 degrees")
    for name, value in (
        ("camera focal-scale noise", focal_scale_fraction),
        ("camera principal-point noise", principal_point_radius_px),
        ("camera distortion noise", distortion_relative_fraction),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if focal_scale_fraction >= 1.0:
        raise ValueError("camera focal-scale noise must be less than 1")

    actual_config = copy.deepcopy(camera_config)
    defaults = actual_config.setdefault("defaults", {})
    nominal_skew = float(defaults.get("skew", 0.0))
    nominal_horizontal_fov = defaults.get("horizontal_fov")
    # OpenCV fisheye in Isaac Sim has no skew field.  Runtime calibration must
    # describe the rendered camera, not an unsupported requested value.
    defaults["skew"] = 0.0

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

        lens_rng = _rng_for_camera_lens(seed, name)
        projection_type = str(
            camera.get("projection_type", defaults.get("projection_type", "pinhole"))
        )
        lens_sample = {"projection_type": projection_type}
        if projection_type == "opencvFisheye":
            nominal_fx = float(camera.get("fx", defaults["fx"]))
            nominal_fy = float(camera.get("fy", defaults["fy"]))
            nominal_cx = float(camera.get("cx", defaults["cx"]))
            nominal_cy = float(camera.get("cy", defaults["cy"]))
            nominal_distortion = np.asarray(
                camera.get(
                    "distortion_coefficients",
                    defaults["distortion_coefficients"],
                ),
                dtype=np.float64,
            )
            if nominal_distortion.shape != (4,):
                raise ValueError(
                    f"camera {name!r} must define four OpenCV fisheye coefficients"
                )
            focal_scale = float(
                lens_rng.uniform(
                    1.0 - focal_scale_fraction,
                    1.0 + focal_scale_fraction,
                )
            )
            principal_delta = _sample_ball(
                lens_rng, principal_point_radius_px
            )[:2]
            distortion_scale = lens_rng.uniform(
                1.0 - distortion_relative_fraction,
                1.0 + distortion_relative_fraction,
                size=4,
            )
            # USD fisheye attributes are float32.  Quantize before writing the
            # runtime JSON so the persisted calibration is byte-for-byte the
            # value the renderer can hold, rather than the float64 request.
            actual_fx = float(np.float32(nominal_fx * focal_scale))
            actual_fy = float(np.float32(nominal_fy * focal_scale))
            width = int(camera.get("width", defaults["width"]))
            height = int(camera.get("height", defaults["height"]))
            actual_cx = float(
                np.float32(np.clip(nominal_cx + principal_delta[0], 0.0, width))
            )
            actual_cy = float(
                np.float32(np.clip(nominal_cy + principal_delta[1], 0.0, height))
            )
            actual_distortion = np.asarray(
                nominal_distortion * distortion_scale, dtype=np.float32
            ).astype(np.float64)
            effective_horizontal_fov_deg = _opencv_fisheye_horizontal_fov_deg(
                width,
                actual_fx,
                actual_cx,
                actual_distortion,
            )
            camera.update(
                {
                    "fx": actual_fx,
                    "fy": actual_fy,
                    "cx": actual_cx,
                    "cy": actual_cy,
                    "skew": 0.0,
                    "distortion_coefficients": actual_distortion.tolist(),
                    "effective_horizontal_fov_deg": effective_horizontal_fov_deg,
                }
            )
            lens_sample.update(
                {
                    "view_randomization": "shared_fx_fy_scale",
                    "focal_scale": focal_scale,
                    "nominal_fx_fy_px": [nominal_fx, nominal_fy],
                    "actual_fx_fy_px": [actual_fx, actual_fy],
                    "principal_point_delta_px": principal_delta.tolist(),
                    "nominal_cx_cy_px": [nominal_cx, nominal_cy],
                    "actual_cx_cy_px": [actual_cx, actual_cy],
                    "distortion_relative_scale": distortion_scale.tolist(),
                    "nominal_distortion_coefficients": nominal_distortion.tolist(),
                    "actual_distortion_coefficients": actual_distortion.tolist(),
                    "effective_horizontal_fov_deg": effective_horizontal_fov_deg,
                    "requested_nominal_skew_px": nominal_skew,
                    "actual_skew_px": 0.0,
                }
            )
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
            "lens": lens_sample,
        }

    metadata = {
        "enabled": bool(
            position_radius_m > 0.0
            or rotation_angle_deg > 0.0
            or focal_scale_fraction > 0.0
            or principal_point_radius_px > 0.0
            or distortion_relative_fraction > 0.0
        ),
        "seed": int(seed),
        "position_distribution": "uniform_volume_ball_in_parent_link_frame",
        "position_radius_m": position_radius_m,
        "rotation_distribution": "uniform_angle_and_axis_in_nominal_camera_frame",
        "rotation_angle_deg": rotation_angle_deg,
        "focal_scale_fraction": focal_scale_fraction,
        "principal_point_radius_px": principal_point_radius_px,
        "distortion_relative_fraction": distortion_relative_fraction,
        "unsupported_parameters": {
            "skew": {
                "requested_nominal_px": nominal_skew,
                "actual_px": 0.0,
                "reason": "Isaac Sim OpenCV fisheye schema does not support skew",
            }
        },
        "derived_parameters": {
            "horizontal_fov": {
                "requested_legacy_value_deg": nominal_horizontal_fov,
                "actual_source": "derived from fx, cx, image width, and OpenCV fisheye k1-k4",
            }
        },
        "episode_constant": True,
        "samples": samples,
    }
    actual_config["mount_randomization"] = copy.deepcopy(metadata)
    return actual_config, metadata
