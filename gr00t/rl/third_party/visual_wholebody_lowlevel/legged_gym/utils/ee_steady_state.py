"""Static forward model for the B2Z1 Cartesian arm controller.

This module computes the map

    commanded EE pose -> steady joint state -> steady EE pose

from the URDF rigid-body model and the controller equations.  It does not run
the simulator and does not use measured or fitted correction tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SteadyEeForwardResult:
    """Result of one mathematical steady-state forward solve."""

    position_world: np.ndarray
    orientation_world_wxyz: np.ndarray
    arm_joint_positions: np.ndarray
    arm_position_targets: np.ndarray
    drive_efforts: np.ndarray
    gravity_efforts: np.ndarray
    equilibrium_residual: np.ndarray
    solver_iterations: int


@dataclass(frozen=True)
class SteadyEeInverseResult:
    """Command pose whose mathematical steady state reaches a desired pose."""

    command_position_world: np.ndarray
    command_orientation_world_wxyz: np.ndarray
    forward_result: SteadyEeForwardResult
    position_error: np.ndarray
    orientation_error: np.ndarray
    solver_iterations: int


def _normalize_quaternion_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("quaternion must have non-zero norm")
    return quaternion / norm


def _rotation_from_quaternion_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = _normalize_quaternion_wxyz(quaternion)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    quaternion_xyzw = pin.Quaternion(np.asarray(rotation, dtype=np.float64)).coeffs()
    quaternion = np.roll(np.asarray(quaternion_xyzw, dtype=np.float64), 1)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _orientation_error_wxyz(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    current_conjugate = current.copy()
    current_conjugate[1:] *= -1.0
    relative = _quaternion_multiply_wxyz(desired, current_conjugate)
    sign = 1.0 if relative[0] >= 0.0 else -1.0
    return relative[1:] * sign


def _quaternion_nlerp_wxyz(start: np.ndarray, end: np.ndarray, fraction: float):
    end = end.copy()
    if float(np.dot(start, end)) < 0.0:
        end *= -1.0
    result = (1.0 - float(fraction)) * start + float(fraction) * end
    return result / max(float(np.linalg.norm(result)), 1.0e-12)


class B2Z1SteadyEeForwardModel:
    """Mathematical forward map of the B2Z1 EE position-drive loop.

    The floating robot base and all non-arm joints are fixed at the supplied
    state during one solve.  The base orientation is still accounted for by
    rotating both the EE command and gravity into the URDF base frame.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        ee_frame_name: str = "gripper_link",
        arm_joint_names: Sequence[str] = (
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ),
        stiffness: float | Sequence[float] = 400.0,
        effort_limit: float | Sequence[float] = 80.0,
        ik_damping: float = 0.05,
        orientation_gain: float = 4.0,
        joint_limit_margin: float = 1.0e-3,
        gravity_magnitude: float = 9.81,
    ):
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        if not self.model.existFrame(ee_frame_name):
            raise ValueError(f"URDF has no EE frame {ee_frame_name!r}")
        self.ee_frame_name = ee_frame_name
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        self.arm_joint_names = tuple(arm_joint_names)
        self.arm_q_indices = np.asarray(
            [self.model.joints[self.model.getJointId(name)].idx_q for name in self.arm_joint_names],
            dtype=np.int64,
        )
        self.arm_v_indices = np.asarray(
            [self.model.joints[self.model.getJointId(name)].idx_v for name in self.arm_joint_names],
            dtype=np.int64,
        )
        if any(self.model.getJointId(name) == 0 for name in self.arm_joint_names):
            raise ValueError(f"URDF is missing an arm joint from {self.arm_joint_names}")

        self.stiffness = np.broadcast_to(
            np.asarray(stiffness, dtype=np.float64), (len(self.arm_joint_names),)
        ).copy()
        self.effort_limit = np.broadcast_to(
            np.asarray(effort_limit, dtype=np.float64), (len(self.arm_joint_names),)
        ).copy()
        if np.any(self.stiffness <= 0.0):
            raise ValueError("all arm stiffness values must be positive")
        if np.any(self.effort_limit <= 0.0):
            raise ValueError("all arm effort limits must be positive")
        self.ik_damping_sq = float(ik_damping) ** 2
        self.orientation_gain = float(orientation_gain)
        self.joint_limit_margin = float(joint_limit_margin)
        self.gravity_magnitude = float(gravity_magnitude)

        self.lower_limits = self.model.lowerPositionLimit[self.arm_q_indices].copy()
        self.upper_limits = self.model.upperPositionLimit[self.arm_q_indices].copy()
        self.lower_limits += self.joint_limit_margin
        self.upper_limits -= self.joint_limit_margin
        if np.any(self.lower_limits >= self.upper_limits):
            raise ValueError("joint_limit_margin collapses an arm joint range")

    def configuration_from_joint_positions(
        self, joint_positions: Mapping[str, float]
    ) -> np.ndarray:
        """Build a Pinocchio configuration in URDF joint-name order."""
        configuration = pin.neutral(self.model)
        missing = []
        for joint_name in self.model.names[1:]:
            if joint_name not in joint_positions:
                missing.append(joint_name)
                continue
            joint_id = self.model.getJointId(joint_name)
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"joint {joint_name!r} has unsupported nq={joint.nq}")
            configuration[joint.idx_q] = float(joint_positions[joint_name])
        if missing:
            raise ValueError(f"joint_positions is missing URDF joints: {missing}")
        return configuration

    def gravity_efforts(
        self,
        joint_positions: Mapping[str, float],
        base_orientation_world_wxyz: Sequence[float],
    ) -> np.ndarray:
        """Return URDF gravity torques in arm-joint order."""
        configuration = self.configuration_from_joint_positions(joint_positions)
        base_rotation_world = _rotation_from_quaternion_wxyz(
            base_orientation_world_wxyz
        )
        self.model.gravity.linear = base_rotation_world.T @ np.asarray(
            [0.0, 0.0, -self.gravity_magnitude], dtype=np.float64
        )
        return np.asarray(
            pin.computeGeneralizedGravity(self.model, self.data, configuration)[
                self.arm_v_indices
            ],
            dtype=np.float64,
        )

    def _frame_pose_and_arm_jacobian(self, configuration: np.ndarray):
        pin.computeJointJacobians(self.model, self.data, configuration)
        pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.ee_frame_id]
        jacobian = pin.getFrameJacobian(
            self.model,
            self.data,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return pose, np.asarray(jacobian[:, self.arm_v_indices], dtype=np.float64)

    def _constrained_dls_delta(
        self,
        jacobian: np.ndarray,
        pose_error: np.ndarray,
        arm_positions: np.ndarray,
    ) -> np.ndarray:
        """Match ManipLoco's active-set damped least-squares step exactly."""
        lower_delta = self.lower_limits - arm_positions
        upper_delta = self.upper_limits - arm_positions
        delta = np.zeros_like(arm_positions)
        free = np.ones(len(arm_positions), dtype=bool)

        for _ in range(len(arm_positions) + 1):
            if not np.any(free):
                break
            active = ~free
            residual = pose_error - jacobian[:, active] @ delta[active]
            jacobian_free = jacobian[:, free]
            normal = jacobian_free.T @ jacobian_free
            normal.flat[:: normal.shape[0] + 1] += self.ik_damping_sq
            candidate = np.linalg.solve(normal, jacobian_free.T @ residual)
            free_indices = np.flatnonzero(free)
            below = candidate < lower_delta[free]
            above = candidate > upper_delta[free]
            if not np.any(below | above):
                delta[free] = candidate
                break
            violating = free_indices[below | above]
            values = candidate[below | above]
            delta[violating] = np.where(
                values < lower_delta[violating],
                lower_delta[violating],
                upper_delta[violating],
            )
            free[violating] = False
        return delta

    def forward(
        self,
        command_position_world: Sequence[float],
        command_orientation_world_wxyz: Sequence[float],
        *,
        base_position_world: Sequence[float],
        base_orientation_world_wxyz: Sequence[float],
        joint_positions: Mapping[str, float],
        equilibrium_tolerance_nm: float = 1.0e-7,
        max_function_evaluations: int = 200,
    ) -> SteadyEeForwardResult:
        """Compute commanded EE pose -> steady EE pose without simulation.

        The root is the static balance between the exact Cartesian-to-joint
        target generated by ManipLoco and the URDF gravity torque:

            clip(Kp * DLS(command - FK(q)), effort_limit) - g(q) = 0.
        """
        command_position_world = np.asarray(command_position_world, dtype=np.float64).reshape(3)
        command_orientation_world_wxyz = _normalize_quaternion_wxyz(
            command_orientation_world_wxyz
        )
        base_position_world = np.asarray(base_position_world, dtype=np.float64).reshape(3)
        base_rotation_world = _rotation_from_quaternion_wxyz(base_orientation_world_wxyz)
        target_command_position_base = base_rotation_world.T @ (
            command_position_world - base_position_world
        )
        command_rotation_world = _rotation_from_quaternion_wxyz(
            command_orientation_world_wxyz
        )
        command_rotation_base = base_rotation_world.T @ command_rotation_world
        target_command_orientation_base = _quaternion_wxyz_from_rotation(
            command_rotation_base
        )

        configuration = self.configuration_from_joint_positions(joint_positions)
        initial_arm_positions = np.clip(
            configuration[self.arm_q_indices], self.lower_limits, self.upper_limits
        )
        self.model.gravity.linear = base_rotation_world.T @ np.asarray(
            [0.0, 0.0, -self.gravity_magnitude], dtype=np.float64
        )

        initial_pose, _ = self._frame_pose_and_arm_jacobian(configuration)
        start_command_position_base = initial_pose.translation.copy()
        start_command_orientation_base = _quaternion_wxyz_from_rotation(
            initial_pose.rotation
        )
        position_distance = float(
            np.linalg.norm(
                target_command_position_base - start_command_position_base
            )
        )
        orientation_distance = 2.0 * np.arccos(
            np.clip(
                abs(
                    float(
                        np.dot(
                            start_command_orientation_base,
                            target_command_orientation_base,
                        )
                    )
                ),
                0.0,
                1.0,
            )
        )
        continuation_steps = max(
            1,
            int(np.ceil(position_distance / 0.03)),
            int(np.ceil(orientation_distance / 0.15)),
        )

        command_position_base = start_command_position_base
        command_orientation_base = start_command_orientation_base
        latest = {}

        def equilibrium_residual(arm_positions: np.ndarray) -> np.ndarray:
            configuration[self.arm_q_indices] = arm_positions
            pose, jacobian = self._frame_pose_and_arm_jacobian(configuration)
            current_orientation = _quaternion_wxyz_from_rotation(pose.rotation)
            pose_error = np.concatenate(
                (
                    command_position_base - pose.translation,
                    self.orientation_gain
                    * _orientation_error_wxyz(
                        command_orientation_base, current_orientation
                    ),
                )
            )
            joint_delta = self._constrained_dls_delta(
                jacobian, pose_error, arm_positions
            )
            arm_targets = np.clip(
                arm_positions + joint_delta, self.lower_limits, self.upper_limits
            )
            drive_efforts = np.clip(
                self.stiffness * (arm_targets - arm_positions),
                -self.effort_limit,
                self.effort_limit,
            )
            gravity_efforts = pin.computeGeneralizedGravity(
                self.model, self.data, configuration
            )[self.arm_v_indices]
            residual = drive_efforts - gravity_efforts
            latest.update(
                arm_targets=arm_targets.copy(),
                drive_efforts=drive_efforts.copy(),
                gravity_efforts=gravity_efforts.copy(),
                residual=residual.copy(),
            )
            return residual

        arm_positions = initial_arm_positions
        total_function_evaluations = 0
        solution = None
        for continuation_index in range(1, continuation_steps + 1):
            fraction = continuation_index / float(continuation_steps)
            command_position_base = (
                (1.0 - fraction) * start_command_position_base
                + fraction * target_command_position_base
            )
            command_orientation_base = _quaternion_nlerp_wxyz(
                start_command_orientation_base,
                target_command_orientation_base,
                fraction,
            )
            solution = least_squares(
                equilibrium_residual,
                arm_positions,
                bounds=(self.lower_limits, self.upper_limits),
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
                max_nfev=int(max_function_evaluations),
            )
            total_function_evaluations += int(solution.nfev)
            residual = equilibrium_residual(solution.x)
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            if (
                not solution.success
                or residual_norm > float(equilibrium_tolerance_nm)
            ):
                raise RuntimeError(
                    "steady EE forward solve did not reach static balance: "
                    f"continuation={continuation_index}/{continuation_steps} "
                    f"success={solution.success} status={solution.status} "
                    f"max_residual_nm={residual_norm:.9g} "
                    f"message={solution.message}"
                )
            arm_positions = solution.x.copy()
            configuration[self.arm_q_indices] = arm_positions

        assert solution is not None
        configuration[self.arm_q_indices] = solution.x
        pose, _ = self._frame_pose_and_arm_jacobian(configuration)
        position_world = base_position_world + base_rotation_world @ pose.translation
        orientation_world = _quaternion_wxyz_from_rotation(
            base_rotation_world @ pose.rotation
        )
        return SteadyEeForwardResult(
            position_world=position_world.copy(),
            orientation_world_wxyz=orientation_world,
            arm_joint_positions=solution.x.copy(),
            arm_position_targets=latest["arm_targets"],
            drive_efforts=latest["drive_efforts"],
            gravity_efforts=latest["gravity_efforts"],
            equilibrium_residual=latest["residual"],
            solver_iterations=total_function_evaluations,
        )

    def inverse(
        self,
        desired_position_world: Sequence[float],
        desired_orientation_world_wxyz: Sequence[float],
        *,
        base_position_world: Sequence[float],
        base_orientation_world_wxyz: Sequence[float],
        joint_positions: Mapping[str, float],
        position_tolerance_m: float = 5.0e-3,
        orientation_tolerance: float = 5.0e-2,
        max_function_evaluations: int = 40,
    ) -> SteadyEeInverseResult:
        """Compute the command pose that produces a requested steady EE pose.

        First solve FK(q*) = desired_pose on the continuous IK branch from the
        supplied state.  Static balance then fixes the required arm target
        displacement exactly::

            delta_q = gravity(q*) / Kp

        Finally invert the controller's damped least-squares matrix at q* to
        obtain the Cartesian command error that produces ``delta_q``.  This is
        a direct inversion of the URDF, IK, PD and gravity equations; it does
        not fit or sample a simulator response map.
        """
        desired_position = np.asarray(
            desired_position_world, dtype=np.float64
        ).reshape(3)
        desired_orientation = _normalize_quaternion_wxyz(
            desired_orientation_world_wxyz
        )
        base_position = np.asarray(base_position_world, dtype=np.float64).reshape(3)
        base_rotation = _rotation_from_quaternion_wxyz(
            base_orientation_world_wxyz
        )
        desired_position_base = base_rotation.T @ (desired_position - base_position)
        desired_rotation_base = base_rotation.T @ _rotation_from_quaternion_wxyz(
            desired_orientation
        )
        desired_orientation_base = _quaternion_wxyz_from_rotation(
            desired_rotation_base
        )

        configuration = self.configuration_from_joint_positions(joint_positions)
        arm_positions = np.clip(
            configuration[self.arm_q_indices], self.lower_limits, self.upper_limits
        )
        start_pose, _ = self._frame_pose_and_arm_jacobian(configuration)
        start_position = start_pose.translation.copy()
        start_orientation = _quaternion_wxyz_from_rotation(start_pose.rotation)
        position_distance = float(np.linalg.norm(desired_position_base - start_position))
        orientation_distance = 2.0 * np.arccos(
            np.clip(abs(float(np.dot(start_orientation, desired_orientation_base))), 0.0, 1.0)
        )
        continuation_steps = max(
            1,
            int(np.ceil(position_distance / 0.03)),
            int(np.ceil(orientation_distance / 0.15)),
        )
        total_function_evaluations = 0
        solution = None

        for continuation_index in range(1, continuation_steps + 1):
            fraction = continuation_index / float(continuation_steps)
            waypoint_position = (
                (1.0 - fraction) * start_position
                + fraction * desired_position_base
            )
            waypoint_orientation = _quaternion_nlerp_wxyz(
                start_orientation, desired_orientation_base, fraction
            )

            def pose_residual(candidate: np.ndarray) -> np.ndarray:
                configuration[self.arm_q_indices] = candidate
                pose, _ = self._frame_pose_and_arm_jacobian(configuration)
                current_orientation = _quaternion_wxyz_from_rotation(pose.rotation)
                return np.concatenate(
                    (
                        2.5 * (pose.translation - waypoint_position),
                        _orientation_error_wxyz(
                            waypoint_orientation, current_orientation
                        ),
                    )
                )

            solution = least_squares(
                pose_residual,
                arm_positions,
                bounds=(self.lower_limits, self.upper_limits),
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
                max_nfev=int(max_function_evaluations),
            )
            total_function_evaluations += int(solution.nfev)
            waypoint_error = pose_residual(solution.x)
            if not solution.success:
                raise RuntimeError(
                    "continuous arm IK failed before reaching the desired pose: "
                    f"continuation={continuation_index}/{continuation_steps} "
                    f"success={solution.success} status={solution.status} "
                    f"position_error_m={np.linalg.norm(waypoint_error[:3]) / 2.5:.9g} "
                    f"orientation_error={np.linalg.norm(waypoint_error[3:]):.9g} "
                    f"message={solution.message}"
                )
            arm_positions = solution.x.copy()

        assert solution is not None
        self.model.gravity.linear = base_rotation.T @ np.asarray(
            [0.0, 0.0, -self.gravity_magnitude], dtype=np.float64
        )
        # An exact pose IK solution can lie on a joint limit while gravity
        # requires its position-drive target a few microradians beyond that
        # limit.  Project the *steady joint state* just far enough inward that
        # q + g(q)/Kp is feasible, recomputing gravity after every projection.
        for _ in range(10):
            configuration[self.arm_q_indices] = arm_positions
            gravity_efforts = pin.computeGeneralizedGravity(
                self.model, self.data, configuration
            )[self.arm_v_indices]
            delta_q = gravity_efforts / self.stiffness
            arm_targets = arm_positions + delta_q
            feasible_targets = np.clip(
                arm_targets, self.lower_limits, self.upper_limits
            )
            target_violation = feasible_targets - arm_targets
            if float(np.linalg.norm(target_violation, ord=np.inf)) <= 1.0e-12:
                break
            arm_positions = np.clip(
                arm_positions + target_violation,
                self.lower_limits,
                self.upper_limits,
            )
        configuration[self.arm_q_indices] = arm_positions
        pose, jacobian = self._frame_pose_and_arm_jacobian(configuration)
        gravity_efforts = pin.computeGeneralizedGravity(
            self.model, self.data, configuration
        )[self.arm_v_indices]
        delta_q = gravity_efforts / self.stiffness
        arm_targets = arm_positions + delta_q
        if (
            np.any(arm_targets < self.lower_limits - 1.0e-10)
            or np.any(arm_targets > self.upper_limits + 1.0e-10)
        ):
            raise RuntimeError(
                "steady EE inverse requires an arm position target outside joint limits: "
                f"q={arm_positions.tolist()} delta_q={delta_q.tolist()}"
            )
        if np.any(np.abs(gravity_efforts) >= self.effort_limit):
            raise RuntimeError(
                "steady EE inverse requires gravity effort at or above actuator limit: "
                f"gravity_nm={gravity_efforts.tolist()} limit_nm={self.effort_limit.tolist()}"
            )

        normal = jacobian.T @ jacobian
        normal.flat[:: normal.shape[0] + 1] += self.ik_damping_sq
        dls = np.linalg.solve(normal, jacobian.T)
        cartesian_error, _, _, _ = np.linalg.lstsq(dls, delta_q, rcond=None)
        recovered_delta = self._constrained_dls_delta(
            jacobian, cartesian_error, arm_positions
        )
        delta_error = recovered_delta - delta_q
        if float(np.linalg.norm(delta_error, ord=np.inf)) > 1.0e-9:
            raise RuntimeError(
                "Cartesian compensation does not invert the online constrained DLS: "
                f"max_joint_delta_error={np.linalg.norm(delta_error, ord=np.inf):.9g}"
            )

        orientation_vector = cartesian_error[3:] / self.orientation_gain
        orientation_vector_norm_sq = float(np.dot(orientation_vector, orientation_vector))
        if orientation_vector_norm_sq >= 1.0:
            raise RuntimeError(
                "steady EE inverse requires an invalid quaternion-vector orientation error: "
                f"norm={np.sqrt(orientation_vector_norm_sq):.9g}"
            )
        relative_orientation_base = np.concatenate(
            (
                [np.sqrt(1.0 - orientation_vector_norm_sq)],
                orientation_vector,
            )
        )
        reachable_orientation_base = _quaternion_wxyz_from_rotation(pose.rotation)
        command_orientation_base = _quaternion_multiply_wxyz(
            relative_orientation_base, reachable_orientation_base
        )
        command_position_base = pose.translation + cartesian_error[:3]
        command_position_world = base_position + base_rotation @ command_position_base
        command_orientation_world = _quaternion_wxyz_from_rotation(
            base_rotation @ _rotation_from_quaternion_wxyz(command_orientation_base)
        )

        actual_position_world = base_position + base_rotation @ pose.translation
        actual_orientation_world = _quaternion_wxyz_from_rotation(
            base_rotation @ pose.rotation
        )
        position_error = actual_position_world - desired_position
        orientation_error = _orientation_error_wxyz(
            desired_orientation, actual_orientation_world
        )
        drive_efforts = np.clip(
            self.stiffness * recovered_delta,
            -self.effort_limit,
            self.effort_limit,
        )
        equilibrium_residual = drive_efforts - gravity_efforts
        if (
            float(np.linalg.norm(position_error)) > float(position_tolerance_m)
            or float(np.linalg.norm(orientation_error)) > float(orientation_tolerance)
            or float(np.linalg.norm(equilibrium_residual, ord=np.inf)) > 1.0e-7
        ):
            raise RuntimeError(
                "steady EE inverse back-substitution failed: "
                f"position_error_m={np.linalg.norm(position_error):.9g} "
                f"orientation_error={np.linalg.norm(orientation_error):.9g} "
                f"equilibrium_error_nm={np.linalg.norm(equilibrium_residual, ord=np.inf):.9g}"
            )
        forward_result = SteadyEeForwardResult(
            position_world=actual_position_world.copy(),
            orientation_world_wxyz=actual_orientation_world.copy(),
            arm_joint_positions=arm_positions.copy(),
            arm_position_targets=arm_targets.copy(),
            drive_efforts=drive_efforts.copy(),
            gravity_efforts=gravity_efforts.copy(),
            equilibrium_residual=equilibrium_residual.copy(),
            solver_iterations=total_function_evaluations,
        )
        return SteadyEeInverseResult(
            command_position_world=command_position_world.copy(),
            command_orientation_world_wxyz=command_orientation_world.copy(),
            forward_result=forward_result,
            position_error=position_error.copy(),
            orientation_error=orientation_error.copy(),
            solver_iterations=total_function_evaluations,
        )
