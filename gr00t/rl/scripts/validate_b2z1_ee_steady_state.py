"""Compare the mathematical B2Z1 EE steady-state map with one PhysX run."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from doorman_auto_open import LOWLEVEL_ROOT, _default_paths, _parser

sys.path.insert(0, str(LOWLEVEL_ROOT))

from legged_gym.scripts.manip_loco_interface import ManipLoco_Policy  # noqa: E402
from legged_gym.utils.ee_steady_state import (  # noqa: E402
    B2Z1SteadyEeForwardModel,
    _orientation_error_wxyz,
    _quaternion_wxyz_from_rotation,
    _rotation_from_quaternion_wxyz,
)
from legged_gym.utils.isaaclab_app import launch_app  # noqa: E402
from legged_gym.utils.isaaclab_math import quat_rotate_inverse  # noqa: E402


def _parse_args():
    parser = _parser()
    parser.description = __doc__
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--settle_steps", type=int, default=1000)
    parser.add_argument("--steady_window", type=int, default=100)
    parser.add_argument(
        "--desired_offset_m",
        type=float,
        nargs=3,
        default=(0.03, 0.0, 0.02),
    )
    args = parser.parse_args()
    _default_paths(args)
    args.num_envs = 1
    args.enable_doorman_scene = False
    args.doorman_randomize_robot_init = False
    args.rgb_camera_config = None
    args.action_delay_mode = "undelayed"
    args.teleop_mode = True
    return args


def _numpy(tensor):
    return tensor[0].detach().cpu().numpy().astype(np.float64, copy=True)


def _joint_position_dict(state):
    values = _numpy(state["dof_pos"])
    return dict(zip(state["dof_names"], values, strict=True))


def _world_position_to_goal_local(env, position_world):
    position_world = torch.as_tensor(
        position_world, device=env.device, dtype=torch.float32
    ).reshape(1, 3)
    return (
        quat_rotate_inverse(
            env.get_goal_reference_quat(),
            position_world - env.get_goal_reference_origin(),
        )
        - env.get_goal_center_offset_local()
    )


def main():
    args = _parse_args()
    if (
        args.warmup_steps <= 0
        or args.settle_steps <= 0
        or not 2 <= args.steady_window <= args.settle_steps
    ):
        raise SystemExit(
            "require warmup_steps and settle_steps > 0 and "
            "2 <= steady_window <= settle_steps"
        )
    simulation_app = launch_app(args)
    try:
        torch.manual_seed(int(args.seed if args.seed is not None else 1700))
        runner = ManipLoco_Policy(args)
        zero_command = torch.zeros((1, 3), device=runner.env.device)

        first = runner.step_once()["state"]
        warmup_command_local = first["ee_goal_cart_local"].detach().clone()
        warmup_orientation = first["ee_goal_orn_quat_world"].detach().clone()
        gripper_target = torch.full(
            (1, 1), -0.785, device=runner.env.device, dtype=torch.float32
        )

        warmup_state = first
        for _ in range(args.warmup_steps):
            runner.set_teleop_targets(
                zero_command,
                ee_goal_cart=warmup_command_local,
                ee_orn_delta_rpy=torch.zeros_like(warmup_command_local),
                gripper_target=gripper_target,
                ee_goal_orn_quat_world=warmup_orientation,
                ee_goal_orn_quat_world_enabled=True,
            )
            warmup_state = runner.step_once()["state"]

        urdf_path = (
            LOWLEVEL_ROOT
            / "resources/robots/b2z1/urdf/generated"
            / "b2z1_mount_90_x0p22_y0_z0p17.urdf"
        )
        forward_model = B2Z1SteadyEeForwardModel(urdf_path)
        desired_position = _numpy(warmup_state["ee_pos"]) + np.asarray(
            args.desired_offset_m, dtype=np.float64
        )
        desired_orientation = _numpy(warmup_state["ee_orn"])
        warmup_root_state = _numpy(warmup_state["root_states"])
        inverse_result = forward_model.inverse(
            desired_position,
            desired_orientation,
            base_position_world=warmup_root_state[:3],
            base_orientation_world_wxyz=warmup_root_state[3:7],
            joint_positions=_joint_position_dict(warmup_state),
        )
        command_position = inverse_result.command_position_world
        command_orientation_np = inverse_result.command_orientation_world_wxyz
        command_orientation = torch.as_tensor(
            command_orientation_np,
            device=runner.env.device,
            dtype=torch.float32,
        ).reshape(1, 4)

        ee_positions = []
        arm_positions = []
        arm_velocities = []
        final_state = warmup_state
        for step in range(args.settle_steps):
            command_local = _world_position_to_goal_local(
                runner.env, command_position
            )
            runner.set_teleop_targets(
                zero_command,
                ee_goal_cart=command_local,
                ee_orn_delta_rpy=torch.zeros_like(command_local),
                gripper_target=gripper_target,
                ee_goal_orn_quat_world=command_orientation,
                ee_goal_orn_quat_world_enabled=True,
            )
            final_state = runner.step_once()["state"]
            if step >= args.settle_steps - args.steady_window:
                ee_positions.append(_numpy(final_state["ee_pos"]))
                arm_ids = runner.env.arm_joint_ids
                arm_positions.append(
                    final_state["dof_pos"][0, arm_ids]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=True)
                )
                arm_velocities.append(
                    final_state["dof_vel"][0, arm_ids]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=True)
                )

        actual_position = _numpy(final_state["ee_pos"])
        actual_orientation = _numpy(final_state["ee_orn"])
        actual_arm_positions = _numpy(final_state["dof_pos"])[
            runner.env.arm_joint_ids.detach().cpu().numpy()
        ]
        position_error = actual_position - desired_position
        orientation_error = _orientation_error_wxyz(
            desired_orientation, actual_orientation
        )
        joint_error = (
            inverse_result.forward_result.arm_joint_positions
            - actual_arm_positions
        )

        actual_joint_position_dict = _joint_position_dict(final_state)
        actual_configuration = forward_model.configuration_from_joint_positions(
            actual_joint_position_dict
        )
        root_state = _numpy(final_state["root_states"])
        base_rotation_world = _rotation_from_quaternion_wxyz(root_state[3:7])
        forward_model.model.gravity.linear = base_rotation_world.T @ np.asarray(
            [0.0, 0.0, -forward_model.gravity_magnitude], dtype=np.float64
        )
        actual_model_pose, _ = forward_model._frame_pose_and_arm_jacobian(
            actual_configuration
        )
        model_actual_position_world = (
            root_state[:3] + base_rotation_world @ actual_model_pose.translation
        )
        model_actual_orientation_world = _quaternion_wxyz_from_rotation(
            base_rotation_world @ actual_model_pose.rotation
        )
        kinematic_position_error = model_actual_position_world - actual_position
        kinematic_orientation_error = _orientation_error_wxyz(
            model_actual_orientation_world, actual_orientation
        )

        pinocchio_gravity = np.asarray(
            __import__("pinocchio").computeGeneralizedGravity(
                forward_model.model, forward_model.data, actual_configuration
            ),
            dtype=np.float64,
        )[forward_model.arm_v_indices]
        physx_gravity_all = (
            runner.env.robot.root_physx_view.get_generalized_gravity_forces()[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
        )
        physx_gravity = physx_gravity_all[
            runner.env.arm_joint_ids.detach().cpu().numpy()
        ]
        actual_arm_targets = _numpy(final_state["arm_position_target"])
        actual_drive_efforts = np.clip(
            400.0 * (actual_arm_targets - actual_arm_positions), -80.0, 80.0
        )
        ee_positions = np.asarray(ee_positions)
        arm_positions = np.asarray(arm_positions)
        arm_velocities = np.asarray(arm_velocities)

        report = {
            "warmup_position_world_m": _numpy(warmup_state["ee_pos"]).tolist(),
            "desired_offset_m": list(args.desired_offset_m),
            "desired_position_world_m": desired_position.tolist(),
            "command_position_world_m": command_position.tolist(),
            "command_minus_desired_m": (
                command_position - desired_position
            ).tolist(),
            "predicted_position_world_m": (
                inverse_result.forward_result.position_world.tolist()
            ),
            "actual_position_world_m": actual_position.tolist(),
            "position_error_m": position_error.tolist(),
            "position_error_norm_m": float(np.linalg.norm(position_error)),
            "orientation_error_vector": orientation_error.tolist(),
            "predicted_arm_joint_rad": (
                inverse_result.forward_result.arm_joint_positions.tolist()
            ),
            "actual_arm_joint_rad": actual_arm_positions.tolist(),
            "arm_joint_error_rad": joint_error.tolist(),
            "max_arm_joint_error_rad": float(np.max(np.abs(joint_error))),
            "fk_position_error_m": kinematic_position_error.tolist(),
            "fk_position_error_norm_m": float(
                np.linalg.norm(kinematic_position_error)
            ),
            "fk_orientation_error_vector": kinematic_orientation_error.tolist(),
            "actual_drive_efforts_nm": actual_drive_efforts.tolist(),
            "pinocchio_gravity_efforts_nm": pinocchio_gravity.tolist(),
            "physx_gravity_efforts_nm": physx_gravity.tolist(),
            "pinocchio_minus_physx_gravity_nm": (
                pinocchio_gravity - physx_gravity
            ).tolist(),
            "equilibrium_residual_nm": (
                inverse_result.forward_result.equilibrium_residual.tolist()
            ),
            "max_equilibrium_residual_nm": float(
                np.max(
                    np.abs(inverse_result.forward_result.equilibrium_residual)
                )
            ),
            "steady_window_ee_span_m": float(
                np.max(np.linalg.norm(ee_positions - ee_positions.mean(axis=0), axis=1))
            ),
            "steady_window_arm_span_rad": float(
                np.max(np.abs(arm_positions - arm_positions.mean(axis=0)))
            ),
            "steady_window_max_arm_speed_rad_s": float(
                np.max(np.abs(arm_velocities))
            ),
        }
        print("EE_STEADY_STATE_VALIDATION=" + json.dumps(report), flush=True)
        if hasattr(runner.env, "close"):
            runner.env.close()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if not args.headless and simulation_app is not None:
            simulation_app.close()
    if args.headless:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
