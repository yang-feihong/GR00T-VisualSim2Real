import math

import numpy as np

from gr00t.rl.scripts.doorman_auto_open import (
    DoorController,
    _solve_center_vertical_shift,
)
from gr00t.rl.auto_open.geometry import (
    HANDLE_AXLE_MARGIN_MIN_M,
    HANDLE_FREE_MARGIN_MIN_M,
    _axial_end_margins,
    _axial_target_correction,
    _center_handle_axial_overlap,
    _contacts_form_opposed_grasp,
)


def _controller():
    controller = DoorController.__new__(DoorController)
    controller.handle_half_length = 0.055
    controller.open_template = {
        "common_axial_min": -0.015,
        "common_axial_max": 0.015,
    }
    return controller


def test_axial_overlap_accepts_an_offset_grasp_on_the_handle():
    controller = _controller()
    centered = _center_handle_axial_overlap(
        controller,
        {"valid": True, "pocket_axial_error": 0.0}
    )
    offset = _center_handle_axial_overlap(
        controller,
        {"valid": True, "pocket_axial_error": 0.020}
    )
    assert math.isclose(centered, 0.030, abs_tol=1.0e-12)
    assert math.isclose(offset, centered, abs_tol=1.0e-12)


def test_axial_overlap_rejects_a_grasp_beyond_the_handle_end():
    controller = _controller()
    overlap = _center_handle_axial_overlap(
        controller,
        {"valid": True, "pocket_axial_error": 0.071}
    )
    assert overlap == 0.0


def test_axial_end_margins_measure_each_handle_side_independently():
    axle_margin, free_margin = _axial_end_margins(
        axial_min=-0.06631,
        axial_max=0.01571,
        outer_half_length=0.074568,
    )
    assert math.isclose(axle_margin, 0.008258, abs_tol=1.0e-12)
    assert math.isclose(free_margin, 0.058858, abs_tol=1.0e-12)


def test_axial_target_correction_applies_stricter_post_noise_margin():
    correction = _axial_target_correction(
        axial_min=-0.06631,
        axial_max=0.01571,
        outer_half_length=0.074568,
        axle_margin=0.025,
        free_margin=0.015,
    )
    assert math.isclose(correction, 0.016742, abs_tol=1.0e-12)


def test_center_capture_requires_axle_and_free_end_margins():
    controller = _controller()
    metrics = {
        "valid": True,
        "pocket_normal_error": 0.0,
        "channel_vertical_error": 0.0,
        "grasp_depth": 0.020,
        "panel_clearance": 0.010,
        "handle_axle_margin": HANDLE_AXLE_MARGIN_MIN_M,
        "handle_free_margin": HANDLE_FREE_MARGIN_MIN_M,
    }
    assert controller._center_capture_ready(metrics)
    metrics["handle_axle_margin"] -= 1.0e-6
    assert not controller._center_capture_ready(metrics)
    metrics["handle_axle_margin"] = HANDLE_AXLE_MARGIN_MIN_M
    metrics["handle_free_margin"] -= 1.0e-6
    assert not controller._center_capture_ready(metrics)


def test_opposed_grasp_does_not_reject_axially_offset_contact_points():
    controller = _controller()
    controller.batched_handle_center = None
    controller.handle_collider_center_local = np.zeros(3, dtype=np.float32)
    controller.handle_axis_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    controller.door_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    controller.handle_upper_z = 0.013
    controller.handle_lower_z = -0.013
    handle_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    contacts = np.array(
        [
            [-0.015, 0.0, -0.013],
            [0.015, 0.0, 0.013],
        ],
        dtype=np.float32,
    )

    valid, metrics = _contacts_form_opposed_grasp(
        controller, contacts, handle_state
    )

    assert valid
    assert metrics["axial_delta"] > 0.020


def test_center_vertical_shift_enters_channel_band_without_exhausting_grasp_travel():
    result = _solve_center_vertical_shift(0.0145, 0.0127)
    assert math.isclose(result["shift_m"], -0.0063, abs_tol=1.0e-12)
    assert math.isclose(
        result["predicted_channel_error_m"], 0.0082, abs_tol=1.0e-12
    )
    assert math.isclose(
        result["predicted_fixed_jaw_gap_m"], 0.019, abs_tol=1.0e-12
    )
    assert result["acceptance_feasible"]


def test_center_vertical_shift_reports_geometrically_infeasible_capture():
    result = _solve_center_vertical_shift(-0.020, 0.0185)
    assert result["shift_m"] > 0.0
    assert not result["acceptance_feasible"]
