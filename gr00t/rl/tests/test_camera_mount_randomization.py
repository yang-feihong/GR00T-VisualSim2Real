import copy
import math

import numpy as np
import pytest

from gr00t.rl.data_collection.camera_mount_randomization import (
    randomize_camera_mounts,
)


NOMINAL = {
    "enabled": True,
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
        NOMINAL, seed=1700, position_radius_m=0.02, rotation_angle_deg=15.0
    )
    repeated, repeated_metadata = randomize_camera_mounts(
        NOMINAL, seed=1700, position_radius_m=0.02, rotation_angle_deg=15.0
    )
    assert NOMINAL == original
    assert actual == repeated
    assert metadata == repeated_metadata
    assert set(metadata["samples"]) == {"base_0_rgb", "left_wrist_0_rgb"}
    for sample in metadata["samples"].values():
        assert sample["position_delta_norm_m"] <= 0.02
        assert sample["rotation_delta_angle_deg"] <= 15.0


def test_each_camera_owns_an_independent_seeded_stream():
    full, metadata = randomize_camera_mounts(
        NOMINAL, seed=1703, position_radius_m=0.02, rotation_angle_deg=15.0
    )
    only_wrist_config = copy.deepcopy(NOMINAL)
    only_wrist_config["cameras"] = [only_wrist_config["cameras"][1]]
    _, only_wrist = randomize_camera_mounts(
        only_wrist_config,
        seed=1703,
        position_radius_m=0.02,
        rotation_angle_deg=15.0,
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
