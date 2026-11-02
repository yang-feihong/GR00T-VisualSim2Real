"""Entry refresh reuses the selected approach and does not add a wrist twist."""

import math

import torch

from gr00t.rl.pick import basket_controller
from gr00t.rl.pick.basket_entry import above_object_path, transport_entry_orientation
from gr00t.rl.pick.planning import executed_arm_path_clear


def test_entry_transport_preserves_object_relative_pinch_not_world_direction():
    from gr00t.rl.auto_open.motion_math import _euler_xyz_quat, _quat_multiply

    original = _euler_xyz_quat(torch.tensor([[0.2, -0.1, 0.3]]))
    current = _euler_xyz_quat(torch.tensor([[0.22, -0.11, 0.32]]))
    relative = _euler_xyz_quat(torch.tensor([[0.0, math.pi / 2, 0.0]]))
    grasp = _quat_multiply(original, relative)
    expected = _quat_multiply(current, relative)
    transported = transport_entry_orientation(grasp, original, current)
    torch.testing.assert_close(transported, expected)
    torch.testing.assert_close(transport_entry_orientation(grasp, original, original), grasp)


def test_empty_top_down_cone_is_a_planning_failure_not_an_exception(monkeypatch):
    monkeypatch.setattr(
        basket_controller,
        "plan_grasp",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("empty inverse request")),
    )
    state = {"root_states": torch.tensor([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]])}
    path, report = basket_controller.plan_basket_grasp_path(
        state,
        torch.tensor([[0.0, 0.0, 0.7]]),
        [0.05] * 3,
        minimum_height=torch.tensor([0.9]),
        final_entry=True,
        entry_orientation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        maximum_tool_tilt_deg=30,
    )
    assert path is None
    assert report["failure"] == "no_grasp_orientation_inside_top_down_cone"
    assert report["required_object_lift_m"] == 0.1


def test_entry_uses_the_same_open_jaw_bisector_as_insertion():
    pitch = -math.pi / 6
    axis = torch.tensor([math.cos(pitch), 0, -math.sin(pitch)])
    target = torch.tensor([0.1, 0.2, 0.7])
    pregrasp = target - 0.08 * axis
    p = torch.stack((pregrasp - 0.05 * axis, pregrasp, target))[None]
    q = torch.tensor([1.0, 0, 0, 0]).expand(1, 3, 4)
    report = {
        "inverse_candidate_indices": [0],
        "selected": 0,
        "pregrasp_standoff_m": [0.08],
        "capture_axis_pitch_rad": pitch,
    }
    (prefix, _, length), _ = above_object_path((p, q, None), report)
    assert length.item() == 2
    torch.testing.assert_close(prefix[0, -1], pregrasp)
    torch.testing.assert_close(prefix, p[:, :2])


def test_above_object_path_stops_before_insertion_and_preserves_selected_axis():
    positions = torch.tensor([[[0.0, 0, 0.9], [0.0, 0, 0.8], [0.0, 0, 0.68], [0.0, 0, 0.6]]])
    q = torch.tensor([[[2**-0.5, 0.0, 2**-0.5, 0.0]]]).expand(1, 4, 4)
    report = {
        "inverse_candidate_indices": [7, 9],
        "selected": 9,
        "pregrasp_standoff_m": [0.04, 0.08],
    }
    original = positions.clone()
    (p, orientation, length), selected = above_object_path((positions, q, None), report)
    assert length.tolist() == [3]
    torch.testing.assert_close(p[:, -1], positions.new_tensor([[0.0, 0, 0.68]]))
    torch.testing.assert_close(orientation[:, -1], selected)
    torch.testing.assert_close(selected, q[:, -1])
    torch.testing.assert_close(positions, original)


def test_final_entry_preserves_selected_orientation_and_basket_quality_weight(monkeypatch):
    control = basket_controller.BasketPickController.__new__(basket_controller.BasketPickController)
    control.final_entry, control.lift_margin = True, 0.10
    control.entry_orientation = torch.tensor([[2**-0.5, 0.0, 2**-0.5, 0.0]])
    control.required_height = lambda state: torch.tensor([0.9])
    state = {"root_states": torch.tensor([[0.0, 0, 0.52, 1, 0, 0, 0]])}
    calls = []

    def plan(*args, **kwargs):
        calls.append(kwargs)
        return None, {}

    monkeypatch.setattr(basket_controller, "plan_grasp", plan)
    _, report = control.plan_basket_grasp(state, torch.tensor([[0.0, 0, 0.7]]), [0.05] * 3)
    torch.testing.assert_close(
        calls[0]["candidate_orientations"], control.entry_orientation[:, None]
    )
    assert calls[0]["wrist_motion_weight"] == 1e-5
    assert not calls[0]["validate_carry_preview"]
    assert not calls[0]["require_precise_carry"]
    assert calls[0]["require_safe_arm_path"]
    assert report["entry_leg"] == "insert_at_measured_object_pose"


def test_insertion_horizon_excludes_superseded_carry_but_still_checks_insertion():
    indices = torch.tensor([0, 150, 200, 250, 300])
    standoffs, selected = torch.tensor([0.08, 0.12]), torch.tensor([0.08])
    safe = torch.ones(1, 1, 7, dtype=torch.bool)
    safe[..., 4] = False
    default, _ = executed_arm_path_clear(
        safe, indices, 151, standoffs, selected, torch.tensor([0.01])
    )
    current, _ = executed_arm_path_clear(
        safe, indices, 151, standoffs, selected, torch.tensor([0.01]), execution_stop_sample=250
    )
    assert not default.item() and current.item()
    safe[..., 2] = False
    current, _ = executed_arm_path_clear(
        safe, indices, 151, standoffs, selected, torch.tensor([0.01]), execution_stop_sample=250
    )
    assert not current.item()
