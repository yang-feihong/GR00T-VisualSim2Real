import copy
import math

import numpy as np
import pytest

from gr00t.rl.data_collection.camera_mount_randomization import (
    randomize_camera_mounts,
)


NOMINAL = {
    "enabled": True,
    "defaults": {
        "projection_type": "opencvFisheye",
        "width": 640,
        "height": 480,
        "horizontal_fov": 45.0,
        "fx": 265.84349,
        "fy": 270.51708,
        "cx": 326.60297,
        "cy": 249.30456,
        "skew": 0.96231,
        "distortion_coefficients": [-0.245617, -0.024594, 0.128775, -0.052686],
    },
    "cameras": [
        {
            "name": "base_0_rgb",
            "enabled": True,
            "body_name": "base_link",
            "position": [0.46, 0.0, 0.01],
            "rotation": [-math.pi / 2.0, 0.0, 0.0],
        },
        {
            "name": "left_wrist_0_rgb",
            "enabled": True,
            "body_name": "link06",
            "position": [0.06, 0.0, 0.11],
            "rotation": [0.0, math.radians(14.0), 0.0],
        },
    ],
}


def test_camera_mount_samples_are_bounded_reproducible_and_non_destructive():
    original = copy.deepcopy(NOMINAL)
    actual, metadata = randomize_camera_mounts(
        NOMINAL,
        seed=1700,
        position_radius_m=0.02,
        rotation_angle_deg=15.0,
        focal_scale_fraction=0.08,
        principal_point_radius_px=5.0,
        distortion_relative_fraction=0.10,
    )
    repeated, repeated_metadata = randomize_camera_mounts(
        NOMINAL,
        seed=1700,
        position_radius_m=0.02,
        rotation_angle_deg=15.0,
        focal_scale_fraction=0.08,
        principal_point_radius_px=5.0,
        distortion_relative_fraction=0.10,
    )
    assert NOMINAL == original
    assert actual == repeated
    assert metadata == repeated_metadata
    assert set(metadata["samples"]) == {"base_0_rgb", "left_wrist_0_rgb"}
    for sample in metadata["samples"].values():
        assert sample["position_delta_norm_m"] <= 0.02
        assert sample["rotation_delta_angle_deg"] <= 15.0
        lens = sample["lens"]
        assert 0.92 <= lens["focal_scale"] <= 1.08
        assert np.linalg.norm(lens["principal_point_delta_px"]) <= 5.0
        assert all(0.9 <= value <= 1.1 for value in lens["distortion_relative_scale"])
        assert lens["actual_skew_px"] == 0.0
    assert actual["defaults"]["skew"] == 0.0
    assert actual["defaults"].get("horizontal_fov") == NOMINAL["defaults"].get(
        "horizontal_fov"
    )
    for camera in actual["cameras"]:
        for key in ("fx", "fy", "cx", "cy"):
            assert camera[key] == float(np.float32(camera[key]))
        assert camera["skew"] == 0.0
        assert all(
            value == float(np.float32(value))
            for value in camera["distortion_coefficients"]
        )
    assert metadata["unsupported_parameters"]["skew"]["requested_nominal_px"] == 0.96231


def test_each_camera_owns_an_independent_seeded_stream():
    full, metadata = randomize_camera_mounts(
        NOMINAL,
        seed=1703,
        position_radius_m=0.02,
        rotation_angle_deg=15.0,
        focal_scale_fraction=0.08,
        principal_point_radius_px=5.0,
        distortion_relative_fraction=0.10,
    )
    only_wrist_config = copy.deepcopy(NOMINAL)
    only_wrist_config["cameras"] = [only_wrist_config["cameras"][1]]
    _, only_wrist = randomize_camera_mounts(
        only_wrist_config,
        seed=1703,
        position_radius_m=0.02,
        rotation_angle_deg=15.0,
        focal_scale_fraction=0.08,
        principal_point_radius_px=5.0,
        distortion_relative_fraction=0.10,
    )
    assert full["mount_randomization"] == metadata
    assert (
        metadata["samples"]["left_wrist_0_rgb"]
        == only_wrist["samples"]["left_wrist_0_rgb"]
    )


def test_zero_bounds_preserve_mount_values_exactly():
    actual, metadata = randomize_camera_mounts(
        NOMINAL, seed=1700, position_radius_m=0.0, rotation_angle_deg=0.0
    )
    for nominal, randomized in zip(NOMINAL["cameras"], actual["cameras"], strict=True):
        np.testing.assert_array_equal(randomized["position"], nominal["position"])
        np.testing.assert_allclose(randomized["rotation"], nominal["rotation"], atol=1e-15)
    assert metadata["enabled"] is False
    assert actual["defaults"]["skew"] == 0.0
    for randomized in actual["cameras"]:
        assert randomized["fx"] == float(np.float32(NOMINAL["defaults"]["fx"]))
        assert randomized["fy"] == float(np.float32(NOMINAL["defaults"]["fy"]))
        assert randomized["cx"] == float(np.float32(NOMINAL["defaults"]["cx"]))
        assert randomized["cy"] == float(np.float32(NOMINAL["defaults"]["cy"]))
        assert randomized["skew"] == 0.0
        np.testing.assert_array_equal(
            randomized["distortion_coefficients"],
            np.asarray(
                NOMINAL["defaults"]["distortion_coefficients"], dtype=np.float32
            ).astype(np.float64),
        )


@pytest.mark.parametrize(
    ("position_radius_m", "rotation_angle_deg"),
    [(-0.01, 0.0), (0.0, -1.0), (0.0, 181.0)],
)
def test_invalid_mount_bounds_are_rejected(position_radius_m, rotation_angle_deg):
    with pytest.raises(ValueError):
        randomize_camera_mounts(
            NOMINAL,
            seed=1700,
            position_radius_m=position_radius_m,
            rotation_angle_deg=rotation_angle_deg,
        )


@pytest.mark.parametrize(
    ("focal", "principal", "distortion"),
    [(-0.01, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -0.1)],
)
def test_invalid_lens_bounds_are_rejected(focal, principal, distortion):
    with pytest.raises(ValueError):
        randomize_camera_mounts(
            NOMINAL,
            seed=1700,
            position_radius_m=0.0,
            rotation_angle_deg=0.0,
            focal_scale_fraction=focal,
            principal_point_radius_px=principal,
            distortion_relative_fraction=distortion,
        )
