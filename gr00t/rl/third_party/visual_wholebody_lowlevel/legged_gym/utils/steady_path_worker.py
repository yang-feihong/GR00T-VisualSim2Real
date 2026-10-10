"""Exact steady EE path solver for process-pool workers."""

from __future__ import annotations

import math

import numpy as np

from legged_gym.utils.ee_steady_state import B2Z1SteadyEeForwardModel


_MODEL = None


def initialize_steady_path_worker(urdf_path: str, stiffness):
    global _MODEL
    _MODEL = B2Z1SteadyEeForwardModel(
        urdf_path,
        stiffness=stiffness,
        emit_tolerance_warnings=False,
    )


def _nlerp(start, end, fraction):
    # Keep this arithmetic identical to DoorController._quat_nlerp.  Even a
    # float64 intermediate can change the resulting contact trajectory.
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    if float(np.dot(start, end)) < 0.0:
        end = -end
    result = (1.0 - float(fraction)) * start + float(fraction) * end
    return result / max(float(np.linalg.norm(result)), 1.0e-8)


def solve_steady_path_request(request):
    if _MODEL is None:
        raise RuntimeError("steady path worker was not initialized")
    model = _MODEL
    model.consume_tolerance_warning_stats()
    start = np.asarray(request["start_position_world"], dtype=np.float64)
    start_orientation = np.asarray(
        request["start_orientation_world_wxyz"], dtype=np.float64
    )
    names = request["dof_names"]
    values = request["dof_positions"]
    joint_positions = {name: float(values[index]) for index, name in enumerate(names)}
    root = np.asarray(request["root_state"], dtype=np.float64)

    pregrasp_selection = None
    candidates = request.get("pregrasp_candidates")
    if candidates:
        selected = None
        best = None
        diagnostics = []
        for candidate in candidates:
            candidate_end = np.asarray(candidate["end_position"], dtype=np.float64)
            candidate_orientation = np.asarray(
                candidate["orientation"], dtype=np.float64
            )
            result = model.inverse(
                candidate_end,
                candidate_orientation,
                base_position_world=root[:3],
                base_orientation_world_wxyz=root[3:7],
                joint_positions=joint_positions,
                position_tolerance_m=float("inf"),
                orientation_tolerance=float("inf"),
            )
            position_error = float(np.linalg.norm(result.position_error))
            orientation_error = float(np.linalg.norm(result.orientation_error))
            diagnostics.append(
                {
                    "pitch_deg": float(candidate["pitch_deg"]),
                    "position_error_m": position_error,
                    "orientation_error": orientation_error,
                }
            )
            score = position_error + orientation_error
            if best is None or score < best[0]:
                best = (score, candidate)
            if position_error <= 1.0e-4 and orientation_error <= 1.0e-4:
                selected = candidate
                break
        exact = selected is not None
        if selected is None:
            selected = best[1]
        end = np.asarray(selected["end_position"], dtype=np.float64)
        end_orientation = np.asarray(selected["orientation"], dtype=np.float64)
        pregrasp_selection = {
            "pitch_deg": float(selected["pitch_deg"]),
            "orientation": end_orientation.astype(np.float32),
            "end_position": end.astype(np.float32),
            "exact": exact,
            "diagnostics": diagnostics,
        }
    else:
        end = np.asarray(request["end_position_world"], dtype=np.float64)
        end_orientation = np.asarray(
            request["end_orientation_world_wxyz"], dtype=np.float64
        )
    orientation_distance = 2.0 * math.acos(
        float(np.clip(abs(np.dot(start_orientation, end_orientation)), 0.0, 1.0))
    )
    count = max(
        1,
        int(math.ceil(float(np.linalg.norm(end - start)) / 0.002)),
        int(math.ceil(orientation_distance / 0.03)),
    )
    bias = request.get("command_bias_world")
    bias = None if bias is None else np.asarray(bias, dtype=np.float64)
    command_positions = []
    command_orientations = []
    for fraction in np.linspace(1.0 / count, 1.0, count):
        desired = (1.0 - fraction) * start + fraction * end
        desired_orientation = _nlerp(
            start_orientation, end_orientation, float(fraction)
        )
        result = model.inverse(
            desired,
            desired_orientation,
            base_position_world=root[:3],
            base_orientation_world_wxyz=root[3:7],
            joint_positions=joint_positions,
        )
        command_position = result.command_position_world
        if bias is not None:
            command_position = command_position + bias
        command_positions.append(command_position.astype(np.float32))
        command_orientations.append(
            result.command_orientation_world_wxyz.astype(np.float32)
        )
        for name, value in zip(
            model.arm_joint_names,
            result.forward_result.arm_joint_positions,
            strict=True,
        ):
            joint_positions[name] = float(value)
    return (
        np.asarray(command_positions),
        np.asarray(command_orientations),
        model.consume_tolerance_warning_stats(),
        pregrasp_selection,
    )


def solve_release_path_request(request):
    """Solve one exact release path, including the interpolated jaw load."""
    if _MODEL is None:
        raise RuntimeError("steady path worker was not initialized")
    model = _MODEL
    model.consume_tolerance_warning_stats()
    start = np.asarray(request["start_position_world"], dtype=np.float64)
    end = np.asarray(request["end_position_world"], dtype=np.float64)
    start_orientation = np.asarray(
        request["start_orientation_world_wxyz"], dtype=np.float64
    )
    end_orientation = np.asarray(
        request["end_orientation_world_wxyz"], dtype=np.float64
    )
    names = request["dof_names"]
    values = request["dof_positions"]
    joint_positions = {
        name: float(values[index]) for index, name in enumerate(names)
    }
    gripper_joint_name = request["gripper_joint_name"]
    root = np.asarray(request["root_state"], dtype=np.float64)
    steps = int(request["steps"])
    start_gripper = float(request["start_gripper_position"])
    end_gripper = float(request["end_gripper_position"])
    tolerance = float(request["position_tolerance_m"])
    command_positions = []
    command_orientations = []
    predicted_positions = []
    for fraction in np.linspace(0.0, 1.0, steps + 1):
        desired = (1.0 - fraction) * start + fraction * end
        desired_orientation = _nlerp(
            start_orientation, end_orientation, float(fraction)
        )
        joint_positions[gripper_joint_name] = (
            (1.0 - fraction) * start_gripper + fraction * end_gripper
        )
        result = model.inverse(
            desired,
            desired_orientation,
            base_position_world=root[:3],
            base_orientation_world_wxyz=root[3:7],
            joint_positions=joint_positions,
            position_tolerance_m=tolerance,
        )
        command_positions.append(result.command_position_world.astype(np.float32))
        command_orientations.append(
            result.command_orientation_world_wxyz.astype(np.float32)
        )
        predicted_positions.append(result.forward_result.position_world.copy())
        for name, value in zip(
            model.arm_joint_names,
            result.forward_result.arm_joint_positions,
            strict=True,
        ):
            joint_positions[name] = float(value)
    prediction_error = np.asarray(predicted_positions) - np.linspace(
        start, end, steps + 1
    )
    return {
        "positions": np.asarray(command_positions, dtype=np.float32),
        "orientations": np.asarray(command_orientations, dtype=np.float32),
        "max_prediction_error_m": float(
            np.max(np.linalg.norm(prediction_error, axis=1))
        ),
        "warning_stats": model.consume_tolerance_warning_stats(),
    }


def solve_deferred_path_request(request):
    """Dispatch one exact path request without aborting the rest of a batch."""
    try:
        if request["path_kind"] == "steady":
            (
                positions,
                orientations,
                warning_stats,
                pregrasp_selection,
            ) = solve_steady_path_request(
                request
            )
            return {
                "error": None,
                "positions": positions,
                "orientations": orientations,
                "warning_stats": warning_stats,
                "pregrasp_selection": pregrasp_selection,
            }
        if request["path_kind"] == "release":
            result = solve_release_path_request(request)
            result["error"] = None
            return result
        raise RuntimeError(f"unsupported path kind: {request['path_kind']}")
    except RuntimeError as exc:
        return {
            "error": str(exc),
            "warning_stats": (
                {} if _MODEL is None else _MODEL.consume_tolerance_warning_stats()
            ),
        }


def solve_steady_inverse_request(request):
    """Run one exact inverse with the same model used by the scalar controller."""
    if _MODEL is None:
        raise RuntimeError("steady path worker was not initialized")
    model = _MODEL
    model.consume_tolerance_warning_stats()
    names = request["dof_names"]
    values = request["dof_positions"]
    joint_positions = {
        name: float(values[index]) for index, name in enumerate(names)
    }
    root = np.asarray(request["root_state"], dtype=np.float64)
    inverse_kwargs = {}
    if request.get("position_tolerance_m") is not None:
        inverse_kwargs["position_tolerance_m"] = float(
            request["position_tolerance_m"]
        )
    try:
        result = model.inverse(
            request["desired_position_world"],
            request["desired_orientation_world_wxyz"],
            base_position_world=root[:3],
            base_orientation_world_wxyz=root[3:7],
            joint_positions=joint_positions,
            **inverse_kwargs,
        )
    except RuntimeError as exc:
        return {
            "error": str(exc),
            "warning_stats": model.consume_tolerance_warning_stats(),
        }
    return {
        "error": None,
        "result": result,
        "command_position": result.command_position_world.astype(np.float32),
        "command_orientation": (
            result.command_orientation_world_wxyz.astype(np.float32)
        ),
        "warning_stats": model.consume_tolerance_warning_stats(),
    }
