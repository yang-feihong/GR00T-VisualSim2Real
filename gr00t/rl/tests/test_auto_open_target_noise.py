import math

import numpy as np
import pytest

from gr00t.rl.scripts.auto_open_target_noise import (
    EpisodeTargetNoise,
    SemanticTargetNoiseConfig,
)


def test_omitted_semantic_noise_is_exact_and_draw_free():
    noise = EpisodeTargetNoise(seed=1700)
    position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    perturbed_position, perturbed_orientation = noise.perturb_pose(
        "grasp_chain",
        position,
        orientation,
        np.eye(3),
        position_bounds=(0.0, 0.0, 0.0),
        rotation_bounds=(0.0, 0.0, 0.0),
        position_names=("depth_m", "handle_axis_m", "closing_m"),
        rotation_names=("roll_rad", "tilt_h_rad", "tilt_z_rad"),
    )
    np.testing.assert_array_equal(perturbed_position, position)
    np.testing.assert_array_equal(perturbed_orientation, orientation)
    assert perturbed_position.dtype == position.dtype
    assert perturbed_orientation.dtype == orientation.dtype

    base_position = np.array([0.25, -0.5], dtype=np.float64)
    perturbed_base, perturbed_yaw = noise.perturb_base(
        "approach",
        base_position,
        0.25,
        [1.0, 0.0],
        normal_m=0.0,
        tangent_m=0.0,
        yaw_rad=0.0,
    )
    np.testing.assert_array_equal(perturbed_base, base_position)
    assert perturbed_base.dtype == base_position.dtype
    assert perturbed_yaw == 0.25
    assert noise.metadata()["samples"] == {}


def test_configuration_is_independent_and_rejects_unknown_variables():
    config = SemanticTargetNoiseConfig.from_assignments(
        ["approach_normal_m=0.1", "grasp_handle_axis_m=0.02"]
    )
    assert config.approach_normal_m == 0.1
    assert config.grasp_handle_axis_m == 0.02
    assert config.approach_tangent_m == 0.0
    assert config.grasp_closing_m == 0.0
    with pytest.raises(ValueError, match="unknown target-noise variable"):
        SemanticTargetNoiseConfig.from_assignments(["ee_position_m=0.01"])
    with pytest.raises(ValueError, match="duplicate"):
        SemanticTargetNoiseConfig.from_assignments(
            ["approach_normal_m=0.1", "approach_normal_m=0.2"]
        )


def test_base_semantic_axes_have_independent_bounds():
    noise = EpisodeTargetNoise(seed=1703)
    position, yaw = noise.perturb_base(
        "approach",
        [0.0, 0.0],
        0.5,
        [1.0, 0.0],
        normal_m=0.1,
        tangent_m=0.0,
        yaw_rad=0.0,
    )
    assert abs(position[0]) <= 0.1
    assert position[1] == 0.0
    assert yaw == 0.5


def test_shared_grasp_sample_preserves_relative_path_geometry():
    noise = EpisodeTargetNoise(seed=1703)
    kwargs = dict(
        semantic_axes_world=np.eye(3),
        position_bounds=(0.002, 0.02, 0.001),
        rotation_bounds=(0.01, 0.0, 0.0),
        position_names=("depth_m", "handle_axis_m", "closing_m"),
        rotation_names=("roll_rad", "tilt_h_rad", "tilt_z_rad"),
    )
    start_position = np.array([1.0, 2.0, 3.0])
    end_position = np.array([1.1, 2.0, 3.0])
    orientation = np.array([1.0, 0.0, 0.0, 0.0])
    perturbed_start = noise.perturb_pose(
        "grasp_chain", start_position, orientation, **kwargs
    )
    perturbed_end = noise.perturb_pose(
        "grasp_chain", end_position, orientation, **kwargs
    )
    np.testing.assert_allclose(
        perturbed_end[0] - perturbed_start[0],
        end_position - start_position,
        atol=2.0e-7,
    )
    np.testing.assert_array_equal(perturbed_start[1], perturbed_end[1])
    sample = noise.metadata()["samples"]["pose:grasp_chain"]
    assert abs(sample["handle_axis_m_delta"]) <= 0.02
    assert sample["tilt_h_rad_delta"] == 0.0
    assert sample["tilt_z_rad_delta"] == 0.0


def test_shared_pose_sample_is_remapped_into_each_candidate_frame():
    noise = EpisodeTargetNoise(
        seed=1702,
        config=SemanticTargetNoiseConfig(grasp_closing_m=0.002),
    )
    orientation = np.array([1.0, 0.0, 0.0, 0.0])
    first_axes = np.eye(3)
    second_axes = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    kwargs = dict(
        position_bounds=(0.0, 0.0, 0.002),
        rotation_bounds=(0.0, 0.0, 0.0),
        position_names=("depth_m", "handle_axis_m", "closing_m"),
        rotation_names=("roll_rad", "tilt_h_rad", "tilt_z_rad"),
    )
    origin = np.zeros(3)
    first, _ = noise.perturb_pose(
        "grasp_chain", origin, orientation, first_axes, **kwargs
    )
    second, _ = noise.perturb_pose(
        "grasp_chain", origin, orientation, second_axes, **kwargs
    )
    closing_delta = noise.metadata()["samples"]["pose:grasp_chain"][
        "closing_m_delta"
    ]
    np.testing.assert_allclose(first, first_axes[:, 2] * closing_delta)
    np.testing.assert_allclose(second, second_axes[:, 2] * closing_delta)


def test_positive_turn_margin_is_cached_and_never_reduces_nominal_motion():
    noise = EpisodeTargetNoise(seed=1703)
    first = noise.positive_margin("turn_handle_overtravel_rad", 0.1)
    second = noise.positive_margin("turn_handle_overtravel_rad", 0.1)
    assert 0.0 <= first <= 0.1
    assert math.isclose(first, second, abs_tol=0.0)


def test_semantic_labels_have_independent_random_streams():
    pose_kwargs = dict(
        semantic_axes_world=np.eye(3),
        position_bounds=(0.002, 0.02, 0.001),
        rotation_bounds=(0.01, 0.0, 0.0),
        position_names=("depth_m", "axis_m", "closing_m"),
        rotation_names=("roll_rad", "tilt_h_rad", "tilt_z_rad"),
    )
    position = np.zeros(3)
    orientation = np.array([1.0, 0.0, 0.0, 0.0])

    grasp_first = EpisodeTargetNoise(seed=1703)
    grasp_first.perturb_pose(
        "grasp_chain", position, orientation, **pose_kwargs
    )
    panel_after_grasp = grasp_first.perturb_pose(
        "panel_push_chain", position, orientation, **pose_kwargs
    )

    panel_only = EpisodeTargetNoise(seed=1703)
    panel_without_grasp = panel_only.perturb_pose(
        "panel_push_chain", position, orientation, **pose_kwargs
    )

    np.testing.assert_array_equal(panel_after_grasp[0], panel_without_grasp[0])
    np.testing.assert_array_equal(panel_after_grasp[1], panel_without_grasp[1])
    assert (
        grasp_first.metadata()["samples"]["pose:panel_push_chain"]
        == panel_only.metadata()["samples"]["pose:panel_push_chain"]
    )
