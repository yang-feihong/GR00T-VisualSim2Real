import math

from gr00t.rl.scripts.doorman_auto_open import (
    DoorController,
    _solve_center_vertical_shift,
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
    centered = controller._center_handle_axial_overlap(
        {"valid": True, "pocket_axial_error": 0.0}
    )
    offset = controller._center_handle_axial_overlap(
        {"valid": True, "pocket_axial_error": 0.020}
    )
    assert math.isclose(centered, 0.030, abs_tol=1.0e-12)
    assert math.isclose(offset, centered, abs_tol=1.0e-12)


def test_axial_overlap_rejects_a_grasp_beyond_the_handle_end():
    controller = _controller()
    overlap = controller._center_handle_axial_overlap(
        {"valid": True, "pocket_axial_error": 0.071}
    )
    assert overlap == 0.0


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
