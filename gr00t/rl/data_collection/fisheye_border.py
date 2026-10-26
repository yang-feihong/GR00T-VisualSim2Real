"""Episode-static DATA fisheye housing textures at the CPU encoding boundary."""

from __future__ import annotations

import copy
import hashlib

import numpy as np


DATA_FISHEYE_NAMES = frozenset(
    {"base_0_rgb", "left_wrist_0_rgb", "wrist", "trunk_front"}
)
VERSION = "data_fisheye_border_v1"
STYLES = ("black", "dark_neutral", "mottled", "annular", "radial_fine")


class FisheyeBorderAugmenter:
    """Never modifies the scene or feather band; no scene/label RNG input.

    Input is source-oriented, already geometrically masked uint8 RGB. The
    replacement is cached once per episode/camera, outside online GPU rendering.
    """

    def __init__(self, seed: int, camera_name: str, camera_spec):
        self.spec = copy.deepcopy(camera_spec)
        mask = self.spec.get("valid_circle_mask")
        self.enabled = (
            camera_name in DATA_FISHEYE_NAMES
            and self.spec.get("projection_type") == "opencvFisheye"
            and bool(mask)
            and bool(mask.get("enabled", True))
        )
        self._texture = None
        self._invalid = None
        self._metadata = {
            "version": VERSION,
            "enabled": self.enabled,
            "camera_name": str(camera_name),
            "episode_seed": int(seed),
        }
        if not self.enabled:
            return
        width, height = int(self.spec["width"]), int(self.spec["height"])
        center = np.asarray(mask["center"], dtype=np.float64)
        radius, feather = float(mask["radius"]), float(mask.get("feather", 0.0))
        if (width <= 0 or height <= 0 or center.shape != (2,)
                or not np.isfinite(center).all() or not np.isfinite(radius)
                or radius <= 0 or not np.isfinite(feather) or feather < 0):
            raise ValueError("invalid effective fisheye circle specification")
        key = f"{VERSION}\0{int(seed)}\0{camera_name}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        rng = np.random.Generator(np.random.PCG64(int(digest, 16)))
        style = str(rng.choice(STYLES, p=[0.20, 0.15, 0.25, 0.25, 0.15]))
        params = {
            "base_luminance": float(rng.uniform(5.0, 42.0)),
            "ring_amplitude": float(rng.uniform(2.0, 14.0)),
            "ring_period_px": float(rng.uniform(5.0, 28.0)),
            "ring_phase_rad": float(rng.uniform(0.0, 2.0 * np.pi)),
            "mottle_amplitude": float(rng.uniform(2.0, 10.0)),
            "mottle_scale_px": float(rng.uniform(18.0, 65.0)),
            "mottle_phase_rad": rng.uniform(0.0, 2.0 * np.pi, 3).tolist(),
            "radial_amplitude": float(rng.uniform(1.0, 5.0)),
            "radial_frequency": int(rng.integers(70, 241)),
            "noise_std": float(rng.uniform(0.5, 2.5)),
            "rgb_gain": rng.uniform(0.90, 1.10, 3).tolist(),
            "edge_ramp_px": float(rng.uniform(2.0, 6.0)),
        }
        if style == "black":
            for name in ("base_luminance", "ring_amplitude", "mottle_amplitude",
                         "radial_amplitude", "noise_std"):
                params[name] = 0.0
        elif style == "dark_neutral":
            params["ring_amplitude"] = params["mottle_amplitude"] = 0.0
            params["radial_amplitude"] = 0.0
        elif style == "mottled":
            params["ring_amplitude"] = params["radial_amplitude"] = 0.0
        elif style == "annular":
            params["radial_amplitude"] = 0.0
        self._metadata.update({
            "style": style,
            "rng": "numpy.PCG64; SHA256(version NUL decimal_seed NUL camera_name)",
            "rng_key_sha256": digest,
            "temporal_mode": "episode_static",
            "source_shape_hwc": [height, width, 3],
            "valid_circle_mask": copy.deepcopy(mask),
            "replacement_region": "distance >= radius if feather > 0 else distance > radius",
            "preserves": "all nonzero-alpha scene pixels including entire feather band",
            "output_clip_rgb": [0, 80],
            "parameters": params,
        })
        yy, xx = np.mgrid[:height, :width].astype(np.float64)
        xx -= center[0]
        yy -= center[1]
        distance = np.hypot(xx, yy)
        self._invalid = distance >= radius if feather > 0 else distance > radius
        angle = np.arctan2(yy, xx)
        p = params
        phases = p["mottle_phase_rad"]
        scale = p["mottle_scale_px"]
        mottle = (
            np.sin(xx / scale + phases[0]) * np.cos(yy / scale + phases[1])
            + 0.5 * np.sin((xx + yy) / (scale * 0.43) + phases[2])
        ) / 1.5
        luminance = (
            p["base_luminance"]
            + p["ring_amplitude"] * np.sin(
                2.0 * np.pi * (distance - radius) / p["ring_period_px"]
                + p["ring_phase_rad"]
            )
            + p["mottle_amplitude"] * mottle
            + p["radial_amplitude"] * np.sin(
                p["radial_frequency"] * angle + distance * 0.12 + phases[0]
            )
            + rng.normal(0.0, p["noise_std"], (height, width))
        )
        # Ramp exclusively outside the geometric circle, retaining its boundary.
        ramp = np.clip((distance - radius) / p["edge_ramp_px"], 0.0, 1.0)
        texture = luminance[..., None] * np.asarray(p["rgb_gain"]) * ramp[..., None]
        self._texture = np.rint(np.clip(texture, 0, 80)).astype(np.uint8)

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def apply(self, image):
        image = np.asarray(image)
        if not self.enabled:
            return image
        if image.dtype != np.uint8 or image.shape != self._texture.shape:
            raise ValueError("fisheye border expects source-oriented uint8 RGB matching calibration")
        result = image.copy()
        result[self._invalid] = self._texture[self._invalid]
        return result
