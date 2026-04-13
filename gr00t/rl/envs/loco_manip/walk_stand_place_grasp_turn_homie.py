# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0



import numpy as np
import torch
from omegaconf import DictConfig

from gr00t.rl.envs.base_task.delta_action_base import DeltaActionBase
from gr00t.rl.envs.base_task.homie_base import HomieBase
from gr00t.rl.envs.base_task.staged_task_base import StagedTaskBase
from gr00t.rl.envs.loco_manip.reset_from_dataset_wsppt import ResetFromDatasetWSPPT
from gr00t.rl.envs.loco_manip.walk_stand_grasp import WalkStandGrasp
from gr00t.rl.envs.loco_manip.walk_table_stand import WalkTableStand
from gr00t.rl.isaac_utils.rotations import (
    quat_to_tan_norm,
)
from gr00t.rl.utils.torch_utils import *
from gr00t.rl.utils.torch_utils import quat_conjugate, quat_mul, quat_rotate, quat_rotate_inverse


class WalkStandPlaceGraspTurnHomie(
    WalkStandGrasp, DeltaActionBase, HomieBase, ResetFromDatasetWSPPT
):
    def __init__(self, config, device):
        WalkTableStand.__init__(self, config, device)

        self.right_thumb_tip_body_idx = self.simulator.body_names.index("right_hand_thumb_2_link")
        self.right_index_tip_body_idx = self.simulator.body_names.index("right_hand_index_1_link")
        self.right_middle_tip_body_idx = self.simulator.body_names.index("right_hand_middle_1_link")
        self.right_palm_body_idx = self.simulator.body_names.index("right_hand_palm_link")
        self.right_hand_indices = [
            self.simulator.body_names.index(link)
            for link in self.simulator.robot_config.right_hand_body_names
        ]
        self.right_elbow_dof_idx = self.simulator.dof_names.index("right_elbow_joint")
        self.right_shoulder_roll_dof_idx = self.simulator.dof_names.index(
            "right_shoulder_roll_joint"
        )
        self.right_shoulder_pitch_dof_idx = self.simulator.dof_names.index(
            "right_shoulder_pitch_joint"
        )
        self.right_shoulder_yaw_dof_idx = self.simulator.dof_names.index("right_shoulder_yaw_joint")
        self.right_wrist_roll_dof_idx = self.simulator.dof_names.index("right_wrist_roll_joint")
        self.right_wrist_pitch_dof_idx = self.simulator.dof_names.index("right_wrist_pitch_joint")
        self.right_wrist_yaw_dof_idx = self.simulator.dof_names.index("right_wrist_yaw_joint")

        self.left_thumb_tip_body_idx = self.simulator.body_names.index("left_hand_thumb_2_link")
        self.left_index_tip_body_idx = self.simulator.body_names.index("left_hand_index_1_link")
        self.left_middle_tip_body_idx = self.simulator.body_names.index("left_hand_middle_1_link")
        self.left_palm_body_idx = self.simulator.body_names.index("left_hand_palm_link")
        self.left_hand_indices = [
            self.simulator.body_names.index(link)
            for link in self.simulator.robot_config.left_hand_body_names
        ]
        self.left_elbow_dof_idx = self.simulator.dof_names.index("left_elbow_joint")
        self.left_shoulder_roll_dof_idx = self.simulator.dof_names.index("left_shoulder_roll_joint")
        self.left_shoulder_pitch_dof_idx = self.simulator.dof_names.index(
            "left_shoulder_pitch_joint"
        )
        self.left_shoulder_yaw_dof_idx = self.simulator.dof_names.index("left_shoulder_yaw_joint")
        self.left_wrist_roll_dof_idx = self.simulator.dof_names.index("left_wrist_roll_joint")
        self.left_wrist_pitch_dof_idx = self.simulator.dof_names.index("left_wrist_pitch_joint")
        self.left_wrist_yaw_dof_idx = self.simulator.dof_names.index("left_wrist_yaw_joint")

        self.last_distance = torch.ones(self.num_envs, device=self.device)
        self.pelvis_idx = self.simulator.body_names.index("pelvis")

        self.reset_from_teleop_curri_ratio = 0.0

        # Define arm DOF indices (after individual indices are set)
        self.right_arm_dof_indices = [
            self.right_shoulder_pitch_dof_idx,
            self.right_shoulder_roll_dof_idx,
            self.right_shoulder_yaw_dof_idx,
            self.right_elbow_dof_idx,
            self.right_wrist_roll_dof_idx,
            self.right_wrist_pitch_dof_idx,
            self.right_wrist_yaw_dof_idx,
        ]
        self.left_arm_dof_indices = [
            self.left_shoulder_pitch_dof_idx,
            self.left_shoulder_roll_dof_idx,
            self.left_shoulder_yaw_dof_idx,
            self.left_elbow_dof_idx,
            self.left_wrist_roll_dof_idx,
            self.left_wrist_pitch_dof_idx,
            self.left_wrist_yaw_dof_idx,
        ]

        # Goal reached rate tracking (similar to ManipBase) - tracks ALL episode completions
        from collections import deque

        self.grasp_goal_reached_deque = deque(
            maxlen=500
        )  # Track goal reached for last 500 episodes
        self.average_grasp_goal_reached_rate = 0.0
        self.skip_initial_goal_tracking = True  # Flag to prevent adding values during initial reset

        # finger primitive related
        self._left_p0 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.left.pos_0,
            device=self.device,
            requires_grad=False,
        )  # closed
        self._left_p1 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.left.pos_1,
            device=self.device,
            requires_grad=False,
        )  # open
        self._right_p0 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.right.pos_0,
            device=self.device,
            requires_grad=False,
        )  # closed
        self._right_p1 = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.right.pos_1,
            device=self.device,
            requires_grad=False,
        )  # open
        self._right_p_init = torch.tensor(
            self.config.robot.finger_primitive.primitive_action_map.right.pos_init,
            device=self.device,
            requires_grad=False,
        )  # initial
        self._left_mode = self.config.robot.finger_primitive.primitive_action_map.left.mode
        self._right_mode = self.config.robot.finger_primitive.primitive_action_map.right.mode
        self._left_hand_dof_idx = [
            self.dof_names.index(name)
            for name in self.config.robot.finger_primitive.primitive_action_map.left.dof_names
        ]
        self._right_hand_dof_idx = [
            self.dof_names.index(name)
            for name in self.config.robot.finger_primitive.primitive_action_map.right.dof_names
        ]
        self._upper_non_finger_dof_idx = [
            i
            for i in self.upper_dof_indices
            if i not in self._left_hand_dof_idx and i not in self._right_hand_dof_idx
        ]

        # Copied from StandGraspLift.__init__
        if self.config.task.which_hand == "right_hand":
            hold_side = "right"
            grasp_side = "right"
            self.hold_palm_body_idx = self.simulator.body_names.index("right_hand_palm_link")
        elif self.config.task.which_hand == "left_hand":
            hold_side = "left"
            grasp_side = "left"
            self.hold_palm_body_idx = self.simulator.body_names.index("left_hand_palm_link")
        for i, body_expr in enumerate(
            self.simulator.hold_object_to_hand_contact_sensor.cfg.filter_prim_paths_expr
        ):
            if f"{hold_side}_hand_thumb_2_link" in body_expr:
                self.hold_thumb_tip_contact_sensor_idx = i
            elif f"{hold_side}_hand_index_1_link" in body_expr:
                self.hold_index_tip_contact_sensor_idx = i
            elif f"{hold_side}_hand_middle_1_link" in body_expr:
                self.hold_middle_tip_contact_sensor_idx = i
            elif f"{hold_side}_hand_palm_link" in body_expr:
                self.hold_palm_contact_sensor_idx = i

        for i, body_expr in enumerate(
            self.simulator.front_object_to_hand_contact_sensor.cfg.filter_prim_paths_expr
        ):
            if f"{grasp_side}_hand_thumb_2_link" in body_expr:
                self.front_thumb_tip_contact_sensor_idx = i
            elif f"{grasp_side}_hand_index_1_link" in body_expr:
                self.front_index_tip_contact_sensor_idx = i
            elif f"{grasp_side}_hand_middle_1_link" in body_expr:
                self.front_middle_tip_contact_sensor_idx = i
            elif f"{grasp_side}_hand_palm_link" in body_expr:
                self.front_palm_contact_sensor_idx = i

        self.hold_finger_tips_contact_sensor_idx = [
            self.hold_thumb_tip_contact_sensor_idx,
            self.hold_index_tip_contact_sensor_idx,
            self.hold_middle_tip_contact_sensor_idx,
            self.hold_palm_contact_sensor_idx,
        ]

        self.front_finger_tips_contact_sensor_idx = [
            self.front_thumb_tip_contact_sensor_idx,
            self.front_index_tip_contact_sensor_idx,
            self.front_middle_tip_contact_sensor_idx,
            self.front_palm_contact_sensor_idx,
        ]

        self.right_finger_dof_idx = [
            self.simulator.dof_names.index(dof_name)
            for dof_name in self.simulator.robot_config.right_hand_dof_names
        ]
        self.left_finger_dof_idx = [
            self.simulator.dof_names.index(dof_name)
            for dof_name in self.simulator.robot_config.left_hand_dof_names
        ]

        self.arm_dof_idx = [
            self.simulator.dof_names.index(dof_name)
            for dof_name in self.simulator.robot_config.arm_dof_names
        ]
        self.left_arm_dof_idx = [
            self.simulator.dof_names.index(dof_name)
            for dof_name in self.simulator.robot_config.left_arm_dof_names
        ]
        self.right_arm_dof_idx = [
            self.simulator.dof_names.index(dof_name)
            for dof_name in self.simulator.robot_config.right_arm_dof_names
        ]

        # Initialize additional buffers that are needed
        self.num_total_task = 0
        self.num_total_success = 0

        self.last_policy_output = torch.zeros(
            self.num_envs,
            self._num_homie_commands + len(self.arm_dof_idx) + 2,
            dtype=torch.float32,
            device=self.device,
        )
        self.policy_output = torch.zeros(
            self.num_envs,
            self._num_homie_commands + len(self.arm_dof_idx) + 2,
            dtype=torch.float32,
            device=self.device,
        )
        self.raw_output_actions = torch.zeros(
            self.num_envs,
            self._num_homie_active_commands - 3,
            dtype=torch.float32,
            device=self.device,
        )
        self._register_buffer_to_track(
            "delta_actions",
            self._get_delta_actions_buffer_shape(),
            self._store_delta_actions_buffer,
            self._load_delta_actions_buffer,
            dtype=torch.float32,
        )
        self.pregrasp_offset = torch.tensor([0.0, 0.0, 0.0], device=self.device)[
            None, :
        ]  # no prior on the pregrasp offset
        self.right_fingers_palm_body_idx = [
            self.simulator.body_names.index(body_name)
            for body_name in self.simulator.robot_config.right_fingers_palm_body_names
        ]

        self.right_arm_qpos_target_hold_object = torch.tensor(
            self.config.rewards.right_arm_qpos_target_hold_object,
            device=self.device,
            requires_grad=False,
        )
        self.right_arm_qpos_target_front_object = torch.tensor(
            self.config.rewards.right_arm_qpos_target_front_object,
            device=self.device,
            requires_grad=False,
        )

    def _init_buffers(self):
        WalkTableStand._init_buffers(self)
        # Copied from StandGrasp._init_buffers
        if self.config.get("show_rgb_image", False):
            self.visualize_env_id = torch.randint(0, self.num_envs, (1,), device=self.device)

        # Initialize per-environment RGB delay steps
        if "rgb_delay_buffer" in self.config.obs.obs_auxiliary.keys():
            history_length = self.config.obs.obs_auxiliary.rgb_delay_buffer.rgb_image

            # Check if random sampling is enabled
            use_random_delay = getattr(self.config.obs, "rgb_image_delay_random", False)
            resample_on_reset = getattr(self.config.obs, "rgb_image_delay_resample_on_reset", False)

            if use_random_delay:
                # RANDOM MODE: Each environment gets a different delay step
                min_delay = getattr(self.config.obs, "rgb_image_delay_step_min", 1)
                max_delay = getattr(self.config.obs, "rgb_image_delay_step_max", history_length)
                # Clamp to valid range
                min_delay = max(1, min(min_delay, history_length))
                max_delay = max(1, min(max_delay, history_length))
                # Sample random delay steps per environment (uniform distribution)
                self.rgb_delay_steps_per_env = torch.randint(
                    min_delay, max_delay + 1, (self.num_envs,), device=self.device
                )
                print(
                    f"[RGB Delay] RANDOM MODE: Sampled per-environment delays in range [{min_delay}, {max_delay}]"
                )
                print(f"[RGB Delay] Distribution: {torch.bincount(self.rgb_delay_steps_per_env)}")
            else:
                # DETERMINISTIC MODE: All environments use the same delay step
                delay_step = getattr(self.config.obs, "rgb_image_delay_step", 1)
                delay_step = max(1, min(delay_step, history_length))  # Clamp to valid range
                self.rgb_delay_steps_per_env = torch.full(
                    (self.num_envs,), delay_step, device=self.device, dtype=torch.long
                )
                print(
                    f"[RGB Delay] DETERMINISTIC MODE: All environments use delay_step={delay_step}"
                )

            # Print resample behavior
            if resample_on_reset:
                print(
                    "[RGB Delay] Resampling delays on every environment reset (resample_on_reset=True)"
                )
            else:
                print("[RGB Delay] Using fixed delays per environment (resample_on_reset=False)")

        if hasattr(self.simulator, "task_root_origin"):
            # Get table_grid base position
            if "table_grid" in self.simulator.task_root_origin:
                self.base_table_pos = self.simulator.task_root_origin["table_grid"][:3]
            else:
                raise ValueError("table_grid not found in task_root_origin")

            # Get random_object base position
            if "random_object" in self.simulator.task_root_origin:
                self.initial_object_pos = self.simulator.task_root_origin["random_object"][:3]
            elif "hold_object" in self.simulator.task_root_origin:
                self.initial_hold_object_pos = self.simulator.task_root_origin["hold_object"][:3]
                self.initial_front_object_pos = self.simulator.task_root_origin["front_object"][:3]
                self.initial_back_object_pos = self.simulator.task_root_origin["back_object"][:3]
            else:
                raise ValueError("random_object not found in task_root_origin")
        else:
            raise ValueError("task_root_origin not found")
        # Initialize table_grid root state buffer
        self.table_grid_root_state_buf = torch.zeros(
            self.num_envs, 13, device=self.device, requires_grad=False
        )
        self.table_grid_root_state_buf[:, 3] = 1.0  # w
        self.table_grid_root_state_buf[:, :3] = self.base_table_pos[None, :] + self.env_origins

        if "random_object" in self.simulator.task_root_origin:
            self.bottle_root_state_buf = torch.zeros(
                self.num_envs, 13, device=self.device, requires_grad=False
            )
            self.bottle_root_state_buf[:, 3] = 1.0  # w
            self.bottle_root_state_buf[:, :3] = self.initial_object_pos[None, :] + self.env_origins
        elif "hold_object" in self.simulator.task_root_origin:
            self.hold_bottle_root_state_buf = torch.zeros(
                self.num_envs, 13, device=self.device, requires_grad=False
            )
            self.hold_bottle_root_state_buf[:, 3] = 1.0  # w
            self.hold_bottle_root_state_buf[:, :3] = (
                self.initial_hold_object_pos[None, :] + self.env_origins
            )
            self.front_bottle_root_state_buf = torch.zeros(
                self.num_envs, 13, device=self.device, requires_grad=False
            )
            self.front_bottle_root_state_buf[:, 3] = 1.0  # w
            self.front_bottle_root_state_buf[:, :3] = (
                self.initial_front_object_pos[None, :] + self.env_origins
            )
            self.target_hold_object_pos = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            self.target_hold_object_pos[:, :] = self.front_bottle_root_state_buf[:, :3]
            self.target_hold_object_pos[:, 1] -= 0.3
            self.target_hold_object_pos[:, 2] += 0.2
            self.back_bottle_root_state_buf = torch.zeros(
                self.num_envs, 13, device=self.device, requires_grad=False
            )
            self.back_bottle_root_state_buf[:, 3] = 1.0  # w
            self.back_bottle_root_state_buf[:, :3] = (
                self.initial_back_object_pos[None, :] + self.env_origins
            )

            # import ipdb; ipdb.set_trace()
            # Initialize tray object buffer
            if "tray_object" in self.simulator.task_root_origin:
                self.initial_tray_object_pos = self.simulator.task_root_origin["tray_object"][:3]
                self.tray_root_state_buf = torch.zeros(
                    self.num_envs, 13, device=self.device, requires_grad=False
                )
                self.tray_root_state_buf[:, 3] = 1.0  # w
                self.tray_root_state_buf[:, :3] = (
                    self.initial_tray_object_pos[None, :] + self.env_origins
                )

        # DOF velocity tracking buffers for logging
        self.average_dof_vel = 0.0
        self.average_arm_dof_vel = 0.0
        self.average_finger_dof_vel = 0.0
        self.last_episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.last_episode_dof_vel_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.last_episode_arm_dof_vel_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.last_episode_finger_dof_vel_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.num_compute_average_epl = getattr(self.config, "num_compute_average_epl", 200)

        # Copied from StandGraspLift._init_buffers
        self.lift_goal_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_lift_distance = torch.ones(self.num_envs, device=self.device)
        self.last_hold_distance = torch.ones(self.num_envs, device=self.device)

        # Initialize goal reached buffer for grasp tasks (separate from walk_stand goal_reached)
        self.grasp_goal_reached_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Initialize target position buffer
        self.target_pos_buf = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # Camera randomization buffers
        self._init_camera_randomization_buffers()
        self.wsdpt_push_torso_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.wsdpt_push_right_palm_force = torch.zeros((self.num_envs, 3), device=self.device)

    def _init_camera_randomization_buffers(self):
        """Initialize buffers for camera randomization"""
        # Get camera randomization config from domain_rand config (updated from simulator.config.cameras)
        camera_rand_config = {}
        if hasattr(self.config, "domain_rand") and hasattr(
            self.config.domain_rand, "extrinsics_randomization"
        ):
            camera_rand_config = self.config.domain_rand.extrinsics_randomization

        # Default randomization ranges
        self.randomize_camera = getattr(camera_rand_config, "enabled", False)

        # Parse position noise range - can be dict with x, y, z or a simple range
        pos_noise_config = getattr(camera_rand_config, "position_noise_range", 0.02)
        if hasattr(pos_noise_config, "get") and hasattr(pos_noise_config, "keys"):
            # Dict/DictConfig format: {x: 0.02, y: 0.02, z: 0.02} -> convert to [[-x, x], [-y, y], [-z, z]]
            # Use .get() for safe access with defaults
            x_val = float(pos_noise_config.get("x", 0.02))
            y_val = float(pos_noise_config.get("y", 0.02))
            z_val = float(pos_noise_config.get("z", 0.02))
            self.camera_pos_noise_range = [[-x_val, x_val], [-y_val, y_val], [-z_val, z_val]]
        elif isinstance(pos_noise_config, (list, tuple)):
            # List format: [-0.02, 0.02] -> use same for all axes
            self.camera_pos_noise_range = [list(pos_noise_config)] * 3
        else:
            # Single value format: 0.02 -> convert to [-0.02, 0.02] for all axes
            val = float(pos_noise_config)
            self.camera_pos_noise_range = [[-val, val]] * 3

        # Parse rotation noise range - can be single value, simple range, or dict with roll/pitch/yaw
        rot_noise_config = getattr(camera_rand_config, "rotation_noise_range", 0.1)

        if isinstance(rot_noise_config, (dict, DictConfig)):
            # New format: {roll: 0.1, pitch: 0.1, yaw: 0.1}
            # Convert each value to [-val, val] range for roll, pitch, yaw
            self.camera_rot_noise_range = [
                [
                    -float(rot_noise_config.get("roll", 0.0)),
                    float(rot_noise_config.get("roll", 0.0)),
                ],
                [
                    -float(rot_noise_config.get("pitch", 0.1)),
                    float(rot_noise_config.get("pitch", 0.1)),
                ],
                [-float(rot_noise_config.get("yaw", 0.0)), float(rot_noise_config.get("yaw", 0.0))],
            ]
        elif isinstance(rot_noise_config, (list, tuple)):
            # Legacy format: single range applied to pitch only
            # Convert to new format: [[0, 0], [min, max], [0, 0]]
            self.camera_rot_noise_range = [
                [0.0, 0.0],  # roll
                list(rot_noise_config),  # pitch
                [0.0, 0.0],  # yaw
            ]
        else:
            # Legacy format: single value, convert to pitch range only
            # 0.1 -> [[0, 0], [-0.1, 0.1], [0, 0]]
            val = float(rot_noise_config)
            self.camera_rot_noise_range = [
                [0.0, 0.0],  # roll
                [-val, val],  # pitch
                [0.0, 0.0],  # yaw
            ]

        # Camera extrinsics curriculum setup
        self.use_camera_extrinsics_curriculum = (
            self.randomize_camera
            and hasattr(camera_rand_config, "curriculum")
            and getattr(camera_rand_config.curriculum, "enabled", False)
        )

        if self.use_camera_extrinsics_curriculum:
            curriculum_config = camera_rand_config.curriculum

            # Initialize current ranges with initial values (convert to plain lists)
            init_pos_range = getattr(
                curriculum_config, "position_noise_init_range", [-0.005, 0.005]
            )
            init_rot_range = getattr(curriculum_config, "rotation_noise_init_range", [-0.05, 0.05])
            self.current_camera_pos_noise_range = [
                list(init_pos_range)
            ] * 3  # Same range for all axes
            # Support both single range (legacy) and per-axis ranges
            if isinstance(init_rot_range, (dict, DictConfig)):
                # New format: {roll: 0.05, pitch: 0.05, yaw: 0.05} -> [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]
                self.current_camera_rot_noise_range = [
                    [
                        -float(init_rot_range.get("roll", 0.0)),
                        float(init_rot_range.get("roll", 0.0)),
                    ],
                    [
                        -float(init_rot_range.get("pitch", 0.05)),
                        float(init_rot_range.get("pitch", 0.05)),
                    ],
                    [-float(init_rot_range.get("yaw", 0.0)), float(init_rot_range.get("yaw", 0.0))],
                ]
            else:
                # Legacy format: single range for pitch only
                self.current_camera_rot_noise_range = [[0.0, 0.0], list(init_rot_range), [0.0, 0.0]]

            # Store curriculum parameters (convert to plain lists)
            max_pos_range = getattr(curriculum_config, "position_noise_max_range", [-0.02, 0.02])
            min_pos_range = getattr(curriculum_config, "position_noise_min_range", [-0.002, 0.002])
            max_rot_range = getattr(curriculum_config, "rotation_noise_max_range", [-0.1, 0.1])
            min_rot_range = getattr(curriculum_config, "rotation_noise_min_range", [-0.02, 0.02])
            self.camera_pos_noise_max_range = [list(max_pos_range)] * 3
            self.camera_pos_noise_min_range = [list(min_pos_range)] * 3
            # Support both single range (legacy) and per-axis ranges for curriculum bounds
            if isinstance(max_rot_range, (dict, DictConfig)):
                self.camera_rot_noise_max_range = [
                    [-float(max_rot_range.get("roll", 0.0)), float(max_rot_range.get("roll", 0.0))],
                    [
                        -float(max_rot_range.get("pitch", 0.1)),
                        float(max_rot_range.get("pitch", 0.1)),
                    ],
                    [-float(max_rot_range.get("yaw", 0.0)), float(max_rot_range.get("yaw", 0.0))],
                ]
            else:
                self.camera_rot_noise_max_range = [[0.0, 0.0], list(max_rot_range), [0.0, 0.0]]

            if isinstance(min_rot_range, (dict, DictConfig)):
                self.camera_rot_noise_min_range = [
                    [-float(min_rot_range.get("roll", 0.0)), float(min_rot_range.get("roll", 0.0))],
                    [
                        -float(min_rot_range.get("pitch", 0.02)),
                        float(min_rot_range.get("pitch", 0.02)),
                    ],
                    [-float(min_rot_range.get("yaw", 0.0)), float(min_rot_range.get("yaw", 0.0))],
                ]
            else:
                self.camera_rot_noise_min_range = [[0.0, 0.0], list(min_rot_range), [0.0, 0.0]]

            self.extrinsics_noise_level_down_threshold = getattr(
                curriculum_config, "extrinsics_noise_level_down_threshold", 25
            )
            self.extrinsics_noise_level_up_threshold = getattr(
                curriculum_config, "extrinsics_noise_level_up_threshold", 70
            )
            self.extrinsics_noise_curriculum_degree = getattr(
                curriculum_config, "extrinsics_noise_curriculum_degree", 0.001
            )

            print("Camera extrinsics curriculum enabled:")
            print(f"  Position noise init range: {self.current_camera_pos_noise_range}")
            print(f"  Position noise max range: {self.camera_pos_noise_max_range}")
            print(f"  Position noise min range: {self.camera_pos_noise_min_range}")
            print(f"  Rotation noise init range: {self.current_camera_rot_noise_range}")
            print(f"  Rotation noise max range: {self.camera_rot_noise_max_range}")
            print(f"  Rotation noise min range: {self.camera_rot_noise_min_range}")
            print(f"  Level down threshold: {self.extrinsics_noise_level_down_threshold}")
            print(f"  Level up threshold: {self.extrinsics_noise_level_up_threshold}")
            print(f"  Curriculum degree: {self.extrinsics_noise_curriculum_degree}")
        else:
            # Use static ranges if curriculum is not enabled
            self.current_camera_pos_noise_range = self.camera_pos_noise_range
            self.current_camera_rot_noise_range = self.camera_rot_noise_range

        if self.randomize_camera:
            # Buffers to store randomized camera offsets for each environment
            self.camera_pos_offset = torch.zeros(
                (self.num_envs, 3), device=self.device, dtype=torch.float32
            )
            self.camera_rot_offset = torch.zeros(
                (self.num_envs, 4), device=self.device, dtype=torch.float32
            )  # quaternion

            print("Camera randomization enabled:")
            print(f"  Position noise range: {self.current_camera_pos_noise_range}")
            print(f"  Rotation noise range: {self.current_camera_rot_noise_range}")
        else:
            print("Camera randomization disabled")

    def _randomize_camera_pose(self, env_ids):
        """Randomize camera pose for specified environments"""
        if not self.randomize_camera or len(env_ids) == 0:
            return

        num_envs = len(env_ids)

        # Randomize position offset (relative to attached link) per axis
        # self.current_camera_pos_noise_range is now a list of 3 ranges: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        for axis in range(3):
            self.camera_pos_offset[env_ids, axis] = torch_rand_float(
                self.current_camera_pos_noise_range[axis][0],
                self.current_camera_pos_noise_range[axis][1],
                (num_envs, 1),
                device=self.device,
            ).squeeze(-1)

        # Randomize rotation offset - ROLL, PITCH, YAW
        # self.current_camera_rot_noise_range is [[roll_min, roll_max], [pitch_min, pitch_max], [yaw_min, yaw_max]]

        # Generate random angles for roll, pitch, yaw
        roll_angles = torch_rand_float(
            self.current_camera_rot_noise_range[0][0],
            self.current_camera_rot_noise_range[0][1],
            (num_envs, 1),
            device=self.device,
        )
        pitch_angles = torch_rand_float(
            self.current_camera_rot_noise_range[1][0],
            self.current_camera_rot_noise_range[1][1],
            (num_envs, 1),
            device=self.device,
        )
        yaw_angles = torch_rand_float(
            self.current_camera_rot_noise_range[2][0],
            self.current_camera_rot_noise_range[2][1],
            (num_envs, 1),
            device=self.device,
        )

        # Convert to quaternions for each axis
        # Roll (X-axis): [1, 0, 0]
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        roll_quats = quat_from_angle_axis(roll_angles.squeeze(-1), x_axis)

        # Pitch (Y-axis): [0, 1, 0]
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(num_envs, 1)
        pitch_quats = quat_from_angle_axis(pitch_angles.squeeze(-1), y_axis)

        # Yaw (Z-axis): [0, 0, 1]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(num_envs, 1)
        yaw_quats = quat_from_angle_axis(yaw_angles.squeeze(-1), z_axis)

        # Combine rotations: yaw * pitch * roll (applied in intrinsic ZYX order)
        # This is equivalent to roll -> pitch -> yaw in extrinsic order
        combined_quats = quat_mul(yaw_quats, quat_mul(pitch_quats, roll_quats))

        self.camera_rot_offset[env_ids] = combined_quats

    def _update_camera_extrinsics_curriculum(self):
        """
        Update the camera extrinsics curriculum based on the average episode length.

        If the average episode length is below the threshold, decrease randomization (make training easier).
        If the average episode length is above the threshold, increase randomization (make training harder).
        """
        if not self.use_camera_extrinsics_curriculum:
            return

        # Update position noise range
        if self.average_episode_length < self.extrinsics_noise_level_down_threshold:
            # Decrease randomization range (make training easier)
            self.current_camera_pos_noise_range = [
                self.current_camera_pos_noise_range[0]
                * (1 - self.extrinsics_noise_curriculum_degree),
                self.current_camera_pos_noise_range[1]
                * (1 - self.extrinsics_noise_curriculum_degree),
            ]
        elif self.average_episode_length > self.extrinsics_noise_level_up_threshold:
            # Increase randomization range (make training harder)
            self.current_camera_pos_noise_range = [
                self.current_camera_pos_noise_range[0]
                * (1 + self.extrinsics_noise_curriculum_degree),
                self.current_camera_pos_noise_range[1]
                * (1 + self.extrinsics_noise_curriculum_degree),
            ]

        # Clip position noise to bounds
        self.current_camera_pos_noise_range = [
            np.clip(
                self.current_camera_pos_noise_range[0],
                self.camera_pos_noise_max_range[0],  # More negative (wider bound)
                self.camera_pos_noise_min_range[0],
            ),  # Less negative (tighter bound)
            np.clip(
                self.current_camera_pos_noise_range[1],
                self.camera_pos_noise_min_range[1],  # Less positive (tighter bound)
                self.camera_pos_noise_max_range[1],
            ),  # More positive (wider bound)
        ]

        # Update rotation noise range for each axis (roll, pitch, yaw)
        if self.average_episode_length < self.extrinsics_noise_level_down_threshold:
            # Decrease randomization range (make training easier)
            self.current_camera_rot_noise_range = [
                [
                    self.current_camera_rot_noise_range[axis][0]
                    * (1 - self.extrinsics_noise_curriculum_degree),
                    self.current_camera_rot_noise_range[axis][1]
                    * (1 - self.extrinsics_noise_curriculum_degree),
                ]
                for axis in range(3)  # roll, pitch, yaw
            ]
        elif self.average_episode_length > self.extrinsics_noise_level_up_threshold:
            # Increase randomization range (make training harder)
            self.current_camera_rot_noise_range = [
                [
                    self.current_camera_rot_noise_range[axis][0]
                    * (1 + self.extrinsics_noise_curriculum_degree),
                    self.current_camera_rot_noise_range[axis][1]
                    * (1 + self.extrinsics_noise_curriculum_degree),
                ]
                for axis in range(3)  # roll, pitch, yaw
            ]

        # Clip rotation noise to bounds for each axis
        self.current_camera_rot_noise_range = [
            [
                np.clip(
                    self.current_camera_rot_noise_range[axis][0],
                    self.camera_rot_noise_max_range[axis][0],  # More negative (wider bound)
                    self.camera_rot_noise_min_range[axis][0],
                ),  # Less negative (tighter bound)
                np.clip(
                    self.current_camera_rot_noise_range[axis][1],
                    self.camera_rot_noise_min_range[axis][1],  # Less positive (tighter bound)
                    self.camera_rot_noise_max_range[axis][1],
                ),  # More positive (wider bound)
            ]
            for axis in range(3)  # roll, pitch, yaw
        ]

    def step(self, actor_state):
        # Use HomieBase.step to handle homie command processing, then update policy tracking
        self.raw_output_actions[:] = actor_state["actions"][:, 3 : self._num_homie_active_commands]
        # grasp_stage = (self.stage_buf == 0).nonzero(as_tuple=False).squeeze(-1)
        # actor_state["actions"][grasp_stage, self._num_homie_active_commands-2:self._num_homie_active_commands] = 1.0
        # stand_stage = ((self.stage_buf == 1) | (self.stage_buf == 2) | (self.stage_buf == 3)).nonzero(as_tuple=False).squeeze(-1)
        # actor_state["actions"][stand_stage, :3] = 0.0
        # actor_state["actions"][:, -1:] = 1.0
        # env_ids = (self.stage_buf < 2).nonzero(as_tuple=False).squeeze(-1)
        # actor_state["actions"][env_ids, 3:-self._num_lower_dof] = self._default_policy_actions[env_ids][:, self._homie_active_command_index][:, 3:].clone()
        obs_dict, rewards, dones, infos = super().step(actor_state)
        self.last_policy_output = self.policy_output.clone()
        self.policy_output = self.processed_commands.clone()

        # logger.warning stage
        # logger.warning(f"stage: {self.stage_buf}")

        # logger.info(f"self._homie_commands {self._homie_commands}")
        # logger.info(f"self.processed_commands {self.processed_commands}")
        # logger.info(f"self._homie_commands_unclipped {self._homie_commands_unclipped}")

        return obs_dict, rewards, dones, infos

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        super().reset_envs_idx(env_ids, target_states, target_buf)
        self.policy_output[env_ids, :] = 0.0
        self.last_policy_output[env_ids, :] = 0.0
        if self.config.domain_rand.get("wsdpt_push", False):
            self.wsdpt_push_torso_force[env_ids, :] = 0.0
            self.wsdpt_push_right_palm_force[env_ids, :] = 0.0
            self.wsdpt_push_interval_s[env_ids] = torch.randint(
                self.config.domain_rand.wsdpt_push_interval_s[0],
                self.config.domain_rand.wsdpt_push_interval_s[1],
                (len(env_ids),),
                device=self.device,
            )
            self.wsdpt_push_counter[env_ids] = 0.0
            self.wsdpt_push_plot_counter[env_ids] = 0.0

        # Resample RGB delay steps for reset environments if enabled
        if hasattr(self, "rgb_delay_steps_per_env") and getattr(
            self.config.obs, "rgb_image_delay_resample_on_reset", False
        ):
            use_random_delay = getattr(self.config.obs, "rgb_image_delay_random", False)
            history_length = self.config.obs.obs_auxiliary.rgb_delay_buffer.rgb_image

            if use_random_delay:
                # RANDOM MODE: Resample different delays for each reset environment
                min_delay = getattr(self.config.obs, "rgb_image_delay_step_min", 1)
                max_delay = getattr(self.config.obs, "rgb_image_delay_step_max", history_length)
                min_delay = max(1, min(min_delay, history_length))
                max_delay = max(1, min(max_delay, history_length))
                self.rgb_delay_steps_per_env[env_ids] = torch.randint(
                    min_delay, max_delay + 1, (len(env_ids),), device=self.device
                )
            else:
                # DETERMINISTIC MODE: Reset to the configured delay step
                delay_step = getattr(self.config.obs, "rgb_image_delay_step", 1)
                delay_step = max(1, min(delay_step, history_length))
                self.rgb_delay_steps_per_env[env_ids] = delay_step

    def _get_termination_level_based_on_reward_penalty_scale(self):
        return torch.clamp(
            0.2 / self.reward_penalty_scale, min=0.5, max=1.5
        )  # penalty scale is from 0.1 to 1.0, so this is from 2.0 to 0.2

    def _check_termination(self):
        StagedTaskBase._check_termination(self)
        if getattr(self.config.termination, "terminate_object_by_low_height", False):
            object_height = (
                self.simulator.all_root_states[self.config.task.front_object][:, 2:3]
                - self.ground_height
            )
            object_lower_than_threshold = torch.any(
                object_height
                < self.config.termination_scales.termination_object_by_low_height_threshold,
                dim=1,
            )
            self.reset_buf |= object_lower_than_threshold

            object_height = (
                self.simulator.get_task_root_state(self.config.task.hold_object)[:, 2:3]
                - self.ground_height
            )
            object_lower_than_threshold |= torch.any(
                object_height
                < self.config.termination_scales.termination_object_by_low_height_threshold,
                dim=1,
            )
            self.reset_buf |= object_lower_than_threshold

        # tgt_obj_pos = self.simulator.get_task_root_state(self.config.task.front_object)[:, :3]
        # moved_too_far = (self.simulator.get_task_root_state(self.config.task.hold_object)[:, :3] - self.simulator._rigid_body_pos[:, self.right_palm_body_idx, :3]).norm(dim=-1) > self.config.termination_scales.termination_hold_object_threshold
        # moved_too_far &= self.stage_buf <= 1
        # self.reset_buf |= moved_too_far
        # if self.config.termination.terminate_by_face_mul:
        #     forward_vec = quat_rotate(self.base_quat, self.forward_vec)
        #     face_mul = (forward_vec @ torch.tensor([-1.0, 0.0, 0.0], device=self.device))
        #     face_mul_not_good = face_mul < self.config.termination_scales.termination_face_mul_threshold
        #     in_stage_4 = self.stage_buf == 4
        #     self.reset_buf |= (face_mul_not_good & in_stage_4)

        if getattr(self.config.termination, "terminate_by_face_mul", False):
            forward_vec = quat_rotate(self.base_quat, self.forward_vec)
            face_mul = forward_vec @ torch.tensor([-1.0, 0.0, 0.0], device=self.device)
            overturn = face_mul < 0.7
            self.reset_buf |= overturn & self.current_last_stage_completed_task_buf

        if getattr(self.config.termination, "terminate_after_finish_turning_for_student", False):
            # check if the robot is in stage 4 and face_mul is over 0.85
            forward_vec = quat_rotate(self.base_quat, self.forward_vec)
            face_mul = forward_vec @ torch.tensor([-1.0, 0.0, 0.0], device=self.device)
            # print(f"face_mul: {face_mul}")
            finish_turning = (face_mul > 0.9) & (self.stage_buf == 4)
            self.reset_buf |= finish_turning
            # print(f"finish_turning: {finish_turning}")

        if getattr(self.config.termination, "terminate_by_throw_object", False):
            hold_object_tray_distance = self.get_hold_object_tray_distance()
            hold_object_hand_distance = self.get_hold_object_hand_distance()
            # contact_force_hold_object = torch.sum(self.simulator.hold_object_to_hand_contact_forces[:, 0, self.hold_finger_tips_contact_sensor_idx, :].norm(dim=-1), dim=-1)
            self.reset_buf |= (
                hold_object_tray_distance > self.config.hold_object_tray_distance_threshold
            ) & (
                hold_object_hand_distance > self.config.termin_hold_object_hand_distance_threshold
            )  # & (contact_force_hold_object < self.config.throw_object_contact_force_threshold)

        if getattr(self.config.termination, "terminate_by_fast_right_arm_qvel", False):
            current_right_arm_qvel = self.simulator.dof_vel[:, self.right_arm_dof_indices]
            right_arm_qvel_threshold = (
                self.config.termination_scales.termination_fast_right_arm_qvel_threshold
                * self.termination_level
            )
            self.reset_buf |= (
                current_right_arm_qvel.abs().max(dim=-1).values > right_arm_qvel_threshold
            )
            # if (current_right_arm_qvel.abs().max(dim=-1).values > right_arm_qvel_threshold).any():
            #     import ipdb; ipdb.set_trace()
            # logger.info(f"current_right_arm_qvel: {current_right_arm_qvel}")
            # logger.info(f"right_arm_qvel_threshold: {right_arm_qvel_threshold}")
            # logger.info(f"reset_buf: {self.reset_buf}")

        if getattr(
            self.config.termination, "terminate_by_finger_qvel_during_right_arm_qvel", False
        ):
            right_arm_qvel = self.simulator.dof_vel[:, self.right_arm_dof_indices]
            finger_qvel = self.simulator.dof_vel[:, self.right_finger_dof_idx]
            right_arm_qvel_mean = torch.mean(torch.abs(right_arm_qvel), dim=-1)
            finger_qvel_mean = torch.mean(torch.abs(finger_qvel), dim=-1)
            # print(f"right_arm_qvel_mean: {right_arm_qvel_mean}, finger_qvel_mean: {finger_qvel_mean}")
            # print(self.episode_length_buf)
            self.reset_buf |= (
                (
                    right_arm_qvel_mean
                    > self.config.termination_scales.termination_finger_qvel_during_right_arm_qvel_threshold
                )
                & (
                    finger_qvel_mean
                    > self.config.termination_scales.termination_finger_qvel_during_right_arm_qvel_threshold
                )
                & (self.episode_length_buf > 50)
            )

        # self.reset_buf.zero_()
        if getattr(self.config.termination, "terminate_by_hold_object_tray_distance", False):
            hold_object_pos = self.simulator.get_task_root_state(self.config.task.hold_object)[
                :, :3
            ]
            tray_pos = self.simulator.get_task_root_state(self.config.task.tray_object)[:, :3]
            pos_diff = torch.abs(hold_object_pos - tray_pos)
            termin_con = (
                pos_diff[:, 0]
                > self.config.termination_scales.termination_hold_object_tray_distance_threshold
            ) | (
                pos_diff[:, 1]
                > self.config.termination_scales.termination_hold_object_tray_distance_threshold
            )
            st_con = self.stage_buf >= 2
            self.reset_buf |= termin_con & st_con

        if getattr(self.config.termination, "terminate_by_linvel_x_when_near_front_object", False):
            env_in_stage_1_2_3_4 = (
                (self.stage_buf == 1)
                | (self.stage_buf == 2)
                | (self.stage_buf == 3)
                | (self.stage_buf == 4)
            )
            homie_command_linvel_x_over_threshold = (
                torch.abs(self._homie_commands_unclipped[:, 0])
            ) > (
                self.config.termination_scales.homie_command_linvel_xy_over_threshold_2x
                * self._get_termination_level_based_on_reward_penalty_scale()
            )
            # logger.info(f"homie_command_linvel_x_over_threshold: {homie_command_linvel_x_over_threshold}")
            # logger.info(f"self._get_termination_level_based_on_reward_penalty_scale(): {self._get_termination_level_based_on_reward_penalty_scale()}")
            self.reset_buf |= env_in_stage_1_2_3_4 & homie_command_linvel_x_over_threshold

        if getattr(self.config.termination, "terminate_by_linvel_y_when_near_front_object", False):
            env_in_stage_1_2_3_4 = (
                (self.stage_buf == 1)
                | (self.stage_buf == 2)
                | (self.stage_buf == 3)
                | (self.stage_buf == 4)
            )
            homie_command_linvel_y_over_threshold = (
                torch.abs(self._homie_commands_unclipped[:, 1])
            ) > (
                self.config.termination_scales.homie_command_linvel_xy_over_threshold_2x
                * self._get_termination_level_based_on_reward_penalty_scale()
            )
            self.reset_buf |= env_in_stage_1_2_3_4 & homie_command_linvel_y_over_threshold

        if getattr(self.config.termination, "terminate_by_angvel_when_near_front_object", False):
            env_in_stage_1_2_3 = (
                (self.stage_buf == 1) | (self.stage_buf == 2) | (self.stage_buf == 3)
            )
            homie_command_angvel_over_threshold = (
                torch.abs(self._homie_commands_unclipped[:, 2])
            ) > (
                self.config.termination_scales.homie_command_angvel_over_threshold_2x
                * self._get_termination_level_based_on_reward_penalty_scale()
            )
            self.reset_buf |= env_in_stage_1_2_3 & homie_command_angvel_over_threshold

        if getattr(
            self.config.termination, "terminate_by_xy_far_from_front_object_init_pos", False
        ):
            env_in_stage_1_2_3_4 = (
                (self.stage_buf == 1)
                | (self.stage_buf == 2)
                | (self.stage_buf == 3)
                | (self.stage_buf == 4)
            )
            robot_to_init_front_object_distance = torch.norm(
                self.simulator.robot_root_states[:, :3] - self.front_bottle_root_state_buf[:, :3],
                dim=-1,
            )
            self.reset_buf |= env_in_stage_1_2_3_4 & (
                robot_to_init_front_object_distance
                > (
                    self.config.termination_scales.robot_to_init_front_object_distance_threshold_2x
                    * self._get_termination_level_based_on_reward_penalty_scale()
                )
            )

    def _reset_tasks_callback(self, env_ids):
        WalkTableStand._reset_tasks_callback(self, env_ids)

        if (
            self.config.domain_rand.get("randomize_dome_light", False)
            or self.config.domain_rand.get("randomize_robot_material", False)
            or self.config.domain_rand.get("randomize_floor_material", False)
        ):
            self.simulator.trigger_reset_events(env_ids, self.common_step_counter)

        if self.config.domain_rand.get("randomize_object_material", False):
            self.simulator.set_task_visual_state_tensor(env_ids)

        # Randomize camera pose for reset environments
        if self.randomize_camera:
            self._randomize_camera_pose(env_ids)
            self.update_ego_camera_pose()

        self._reset_task_states(env_ids)
        self.last_distance[env_ids] = torch.ones(len(env_ids), device=self.device)
        self.grasp_goal_reached_buf[env_ids] = False

        # Reset lift goal from StandGraspLift._reset_tasks_callback
        # self.lift_goal_buf[env_ids, 0] = torch_rand_float(*self.config.bottle_pos_range_x, (len(env_ids), 1), device=str(self.device)).squeeze(-1) + self.env_origins[env_ids, 0] + self.initial_front_object_pos[0]
        # self.lift_goal_buf[env_ids, 1] = torch_rand_float(*self.config.bottle_pos_range_y, (len(env_ids), 1), device=str(self.device)).squeeze(-1) + self.env_origins[env_ids, 1] + self.initial_front_object_pos[1]
        # self.lift_goal_buf[env_ids, 2] = torch_rand_float(*self.config.bottle_pos_range_z, (len(env_ids), 1), device=str(self.device)).squeeze(-1) + self.env_origins[env_ids, 2]
        # self.last_lift_distance[env_ids] = (self.lift_goal_buf[env_ids, :] - self.front_bottle_root_state_buf[env_ids, :3]).norm(dim=-1)

        pelvis_pos = self.simulator._rigid_body_pos[:, self.pelvis_idx, :]
        lift_goal_offset_pelvis = self.config.lift_goal_offset_pelvis
        # Transform offset from robot local frame to world frame using base quaternion
        lift_goal_offset_pelvis_tensor = (
            torch.tensor(lift_goal_offset_pelvis, device=self.device)
            .unsqueeze(0)
            .repeat(len(env_ids), 1)
        )
        lift_goal_offset_world = quat_rotate(
            self.base_quat[env_ids, :], lift_goal_offset_pelvis_tensor
        )
        self.lift_goal_buf[env_ids, :] = pelvis_pos[env_ids, :] + lift_goal_offset_world
        this_lift_distance = (
            self.lift_goal_buf[env_ids, :]
            - self.simulator.get_task_root_state(self.config.task.front_object)[env_ids, :3]
        ).norm(dim=-1)
        self.last_lift_distance[env_ids] = this_lift_distance

        self.last_hold_distance[env_ids] = (
            self.target_hold_object_pos[env_ids, :] - self.hold_bottle_root_state_buf[env_ids, :3]
        ).norm(dim=-1)
        # Reset DOF velocity tracking buffers
        self.last_episode_length_buf[env_ids] = 0
        self.last_episode_dof_vel_buf[env_ids] = 0.0
        self.last_episode_arm_dof_vel_buf[env_ids] = 0.0
        self.last_episode_finger_dof_vel_buf[env_ids] = 0.0
        if self.config.reset_from_dataset.enable:
            if self.config.reset_from_dataset.get("reset_from_teleop_curriculum", False):
                if self.last_last_stage_completed_task_buf.float().mean() > 0.8:
                    self.reset_from_teleop_curri_ratio += 0.1
                    self.reset_from_teleop_curri_ratio = min(
                        self.reset_from_teleop_curri_ratio, 1.0
                    )
                elif self.last_last_stage_completed_task_buf.float().mean() < 0.6:
                    self.reset_from_teleop_curri_ratio -= 0.1
                    self.reset_from_teleop_curri_ratio = max(
                        self.reset_from_teleop_curri_ratio, 0.0
                    )
                self.log_dict["reset_from_teleop_curri_ratio"] = torch.tensor(
                    [self.reset_from_teleop_curri_ratio], device=self.device
                )

    def _pre_compute_observations_callback(self, env_ids=None):
        self.target_pos_buf[:] = self.simulator.get_task_root_state(self.config.task.front_object)[
            :, :3
        ]
        palm_pos = self.simulator._rigid_body_pos[:, self.right_palm_body_idx, :]

        self.target_pos_buf[:, 0] = torch.where(
            self.stage_buf == 2, self.target_pos_buf[:, 0] - 0.05, self.target_pos_buf[:, 0]
        )  # stage 2 = grasp stage 0
        self.target_pos_buf[:, 1] = torch.where(
            self.stage_buf == 2, self.target_pos_buf[:, 1] - 0.15, self.target_pos_buf[:, 1]
        )
        # self.simulator.draw_spheres_batch(self.target_pos_buf)
        self.target_pos_buf[:] = self.target_pos_buf - palm_pos + self.pregrasp_offset

        return WalkTableStand._pre_compute_observations_callback(self, env_ids)

    def _post_compute_observations_callback(self):
        this_distance = self.target_pos_buf.norm(dim=-1)
        # Accumulate average absolute DOF velocities per timestep
        self.last_episode_dof_vel_buf += (
            torch.sum(torch.abs(self.simulator.dof_vel), dim=1) / self.num_dof
        )

        # Accumulate separate arm and finger DOF velocities
        if hasattr(self, "right_arm_dof_indices") and len(self.right_arm_dof_indices) > 0:
            self.last_episode_arm_dof_vel_buf += torch.sum(
                torch.abs(self.simulator.dof_vel[:, self.right_arm_dof_indices]), dim=1
            ) / len(self.right_arm_dof_indices)

        if hasattr(self, "right_finger_dof_idx") and len(self.right_finger_dof_idx) > 0:
            self.last_episode_finger_dof_vel_buf += torch.sum(
                torch.abs(self.simulator.dof_vel[:, self.right_finger_dof_idx]), dim=1
            ) / len(self.right_finger_dof_idx)

        self.last_episode_length_buf = self.episode_length_buf.clone()

        self.last_distance = this_distance

        # Update grasp goal reached (separate from walk_stand goal_reached)
        self.grasp_goal_reached_buf |= self.current_last_stage_completed_task_buf
        self.extras["grasp_goal_reached"] = self.grasp_goal_reached_buf.clone()

        # Handle periodic task reset
        reset_task_mask = (self.episode_length_buf % self.config.reset_task_every_n_steps == 0) & (
            self.episode_length_buf > 0
        )
        if reset_task_mask.any():
            env_ids = torch.where(reset_task_mask)[0]
            self.num_total_success += self.current_last_stage_completed_task_buf[env_ids].sum()
            self.num_total_task += len(env_ids)
            self._reset_task_states(env_ids)
            self.set_to_stage(
                env_ids, torch.zeros(len(env_ids), device=self.device, dtype=torch.long)
            )

        # Add recent goal reached rate to extras for logging
        if len(self.grasp_goal_reached_deque) > 0:
            recent_grasp_goal_reached_rate = sum(self.grasp_goal_reached_deque) / len(
                self.grasp_goal_reached_deque
            )
            self.extras["episode"][
                "recent_grasp_goal_reached_rate"
            ] = recent_grasp_goal_reached_rate
        else:
            self.extras["episode"]["recent_grasp_goal_reached_rate"] = 0.0

        # Add average DOF velocities to extras for wandb logging
        self.extras["episode"]["average_dof_vel"] = torch.tensor(
            self.average_dof_vel, dtype=torch.float, device=self.device
        )
        self.extras["episode"]["average_arm_dof_vel"] = torch.tensor(
            self.average_arm_dof_vel, dtype=torch.float, device=self.device
        )
        self.extras["episode"]["average_finger_dof_vel"] = torch.tensor(
            self.average_finger_dof_vel, dtype=torch.float, device=self.device
        )

        pelvis_pos = self.simulator._rigid_body_pos[:, self.pelvis_idx, :]
        lift_goal_offset_pelvis = self.config.lift_goal_offset_pelvis
        # Transform offset from robot local frame to world frame using base quaternion
        lift_goal_offset_pelvis_tensor = (
            torch.tensor(lift_goal_offset_pelvis, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        lift_goal_offset_world = quat_rotate(self.base_quat, lift_goal_offset_pelvis_tensor)
        self.lift_goal_buf = pelvis_pos + lift_goal_offset_world
        this_lift_distance = (
            self.lift_goal_buf
            - self.simulator.get_task_root_state(self.config.task.front_object)[:, :3]
        ).norm(dim=-1)
        self.last_lift_distance = this_lift_distance

        self.last_hold_distance = (
            self.target_hold_object_pos
            - self.simulator.get_task_root_state(self.config.task.hold_object)[:, :3]
        ).norm(dim=-1)
        if self.config.get("enforce_vis", False):
            self._draw_goal_position()

        self.box_robot_dist[:] = torch.norm(
            self.simulator.all_root_states[self.config.task.front_object][:, :2]
            - self.simulator.robot_root_states[:, :2],
            dim=1,
        )
        self.robot_target_dist[:] = torch.norm(
            self.simulator.all_root_states[self.config.task.front_object][:, :2]
            - self.simulator.env_origins[:, :2]
            - self.goal_pos[None, :2],
            dim=1,
        )
        self.box_target_dist[:] = torch.norm(
            self.simulator.all_root_states[self.config.task.front_object][:, :3]
            - self.simulator.env_origins[:, :]
            - self.goal_pos[None, :],
            dim=1,
        )
        self.box_palm_dist[:] = torch.norm(
            self.simulator.all_root_states[self.config.task.front_object][:, :3]
            - self.simulator._rigid_body_pos[:, self.palm_index][:, :],
            dim=1,
        )
        self.success_buffer[:] = (
            self.box_robot_dist <= self.config.rewards.walk_stand_threshold_distance
        ) & (torch.norm(self._homie_commands[:, :3], dim=1) <= 0.1)
        self.consecutive_success_count[self.success_buffer] += 1
        self.consecutive_success_count[~self.success_buffer] = 0

        self.walk_stand_goal_reached_buf |= (
            self.consecutive_success_count >= self.max_consecutive_success_frames
        )

        self.extras["walk_stand_goal_reached"] = self.walk_stand_goal_reached_buf.clone()

        # Add recent goal reached rate to extras for logging
        # Now uses the comprehensive deque that tracks ALL episode completions
        if len(self.walk_stand_goal_reached_deque) > 0:
            recent_walk_stand_goal_reached_rate = sum(self.walk_stand_goal_reached_deque) / len(
                self.walk_stand_goal_reached_deque
            )
            self.extras["episode"][
                "recent_walk_stand_goal_reached_rate"
            ] = recent_walk_stand_goal_reached_rate
        else:
            self.extras["episode"]["recent_walk_stand_goal_reached_rate"] = 0.0

        return StagedTaskBase._post_compute_observations_callback(self)

    def _reset_dofs(self, env_ids, target_state=None):
        # Copied and adapted from StandGrasp._reset_dofs
        self.target_robot_dof_state[env_ids, :, 0] = (
            self.default_dof_pos
        )  # + torch_rand_float(-0.05, 0.05, (len(env_ids), self.num_dof), device=str(self.device))
        # self.target_robot_dof_state[env_ids, self.right_shoulder_roll_dof_idx, 0] = torch_rand_float(-1.5708, 0.05, (len(env_ids), 1), device=str(self.device)).squeeze(-1)
        # shoulder_pitch_plus_elbow = torch_rand_float(-1.5708, 0.0, (len(env_ids), 1), device=str(self.device)).squeeze(-1)
        # self.target_robot_dof_state[env_ids, self.right_elbow_dof_idx, 0] = torch_rand_float(-0.785398, 0.0, (len(env_ids), 1), device=str(self.device)).squeeze(-1)
        # self.target_robot_dof_state[env_ids, self.right_shoulder_pitch_dof_idx, 0] = shoulder_pitch_plus_elbow - self.target_robot_dof_state[env_ids, self.right_elbow_dof_idx, 0]
        # self.target_robot_dof_state[env_ids, self.right_shoulder_yaw_dof_idx, 0] = torch_rand_float(-0.785398, 0.785398, (len(env_ids), 1), device=str(self.device)).squeeze(-1)
        # self.target_robot_dof_state[env_ids, self.right_shoulder_yaw_dof_idx, 0] = torch.where((self.target_robot_dof_state[env_ids, self.right_shoulder_pitch_dof_idx, 0] > 0.1) & (self.target_robot_dof_state[env_ids, self.right_shoulder_yaw_dof_idx, 0] > 0), 0.1 * self.target_robot_dof_state[env_ids, self.right_shoulder_yaw_dof_idx, 0], self.target_robot_dof_state[env_ids, self.right_shoulder_yaw_dof_idx, 0])
        # self.target_robot_dof_state[env_ids, self.right_wrist_roll_dof_idx, 0] = torch_rand_float(-0.785398, 0.785398, (len(env_ids), 1), device=str(self.device)).squeeze(-1)
        # self.target_robot_dof_state[env_ids, self.right_wrist_pitch_dof_idx, 0] = torch_rand_float(-0.785398, 0, (len(env_ids), 1), device=str(self.device)).squeeze(-1)
        # self.target_robot_dof_state[env_ids, self.right_wrist_yaw_dof_idx, 0] = torch_rand_float(-0.785398, 0.785398, (len(env_ids), 1), device=str(self.device)).squeeze(-1)
        # Add finger DOF initialization from StandGraspLift._reset_dofs
        # Set right finger DOFs to initial configuration for each env
        # Fix shape mismatch by assigning each DOF index individually
        for i, dof_idx in enumerate(self.right_finger_dof_idx):
            self.target_robot_dof_state[env_ids, dof_idx, 0] = self._right_p_init[i]
        self.target_robot_dof_state[env_ids, :, 1] = 0.0

    def _reset_task_states(self, env_ids):

        if self.config.reset_from_dataset.enable:
            # import ipdb; ipdb.set_trace()
            ResetFromDatasetWSPPT._reset_task_states(self, env_ids)
        else:
            hold_palm_pos = self.simulator._rigid_body_pos[env_ids, self.hold_palm_body_idx, :3]
            hold_offset = quat_rotate(
                self.simulator._rigid_body_rot[env_ids, self.hold_palm_body_idx, :4],
                torch.tensor([0.08, 0.05, 0.035], device=self.device).expand(len(env_ids), 3),
            )

            self.hold_bottle_root_state_buf[env_ids, :3] = hold_palm_pos + hold_offset
            self.hold_bottle_root_state_buf[env_ids, 3:7] = self.simulator._rigid_body_rot[
                env_ids, self.hold_palm_body_idx, :4
            ]

            self.front_bottle_root_state_buf[env_ids, 0] = (
                torch_rand_float(
                    *self.config.bottle_pos_range_x, (len(env_ids), 1), device=str(self.device)
                ).squeeze(-1)
                + self.env_origins[env_ids, 0]
                + self.initial_front_object_pos[0]
            )
            self.front_bottle_root_state_buf[env_ids, 1] = (
                torch_rand_float(
                    *self.config.bottle_pos_range_y, (len(env_ids), 1), device=str(self.device)
                ).squeeze(-1)
                + self.env_origins[env_ids, 1]
                + self.initial_front_object_pos[1]
            )

            self.target_hold_object_pos[env_ids, :] = self.front_bottle_root_state_buf[env_ids, :3]
            self.target_hold_object_pos[env_ids, 1] -= 0.3
            self.target_hold_object_pos[env_ids, 2] += 0.2

            self.back_bottle_root_state_buf[env_ids, 0] = (
                torch_rand_float(
                    *self.config.bottle_pos_range_x, (len(env_ids), 1), device=str(self.device)
                ).squeeze(-1)
                + self.env_origins[env_ids, 0]
                + self.initial_back_object_pos[0]
            )
            self.back_bottle_root_state_buf[env_ids, 1] = (
                torch_rand_float(
                    *self.config.bottle_pos_range_y, (len(env_ids), 1), device=str(self.device)
                ).squeeze(-1)
                + self.env_origins[env_ids, 1]
                + self.initial_back_object_pos[1]
            )

            # Reset tray object position if it exists
            if hasattr(self, "tray_root_state_buf"):
                self.tray_root_state_buf[env_ids, 0] = (
                    torch_rand_float(
                        *self.config.bottle_pos_range_x, (len(env_ids), 1), device=str(self.device)
                    ).squeeze(-1)
                    + self.env_origins[env_ids, 0]
                    + self.initial_tray_object_pos[0]
                )
                self.tray_root_state_buf[env_ids, 1] = (
                    torch_rand_float(
                        *self.config.bottle_pos_range_y, (len(env_ids), 1), device=str(self.device)
                    ).squeeze(-1)
                    + self.env_origins[env_ids, 1]
                    + self.initial_tray_object_pos[1]
                )

            task_root_state_dict = {
                "front_object": self.front_bottle_root_state_buf,
                "hold_object": self.hold_bottle_root_state_buf,
                "back_object": self.back_bottle_root_state_buf,
            }

            # Add tray to task dict if it exists
            if hasattr(self, "tray_root_state_buf"):
                task_root_state_dict["tray_object"] = self.tray_root_state_buf

            self.simulator.set_task_root_state_tensor(env_ids, task_root_state_dict)

    def _post_physics_step(self):
        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        # update counters
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()

        self._pre_compute_observations_callback()
        self._update_tasks_callback()
        # compute observations, rewards, resets, ...
        self._check_termination()
        self._compute_reward()
        # check terminations
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_envs_idx(env_ids)

        # set envs
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            # ZL: this is a terrible way to set the sim values.
            self.simulator.set_actor_root_state_tensor(
                refresh_env_ids, self.target_robot_root_states
            )
            self.simulator.set_dof_state_tensor(refresh_env_ids, self.target_robot_dof_state)
            if not self.config.reset_from_dataset.enable:
                self._reset_tasks_callback(refresh_env_ids)
            # self.simulator.set_task_root_state_tensor(refresh_env_ids, self.target_task_root_states)
            # self.simulator.set_task_visual_state_tensor(refresh_env_ids)
            self.need_to_refresh_envs[refresh_env_ids] = False

        if len(env_ids) > 0:
            self._pre_compute_observations_callback(env_ids=env_ids)
        self._compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        if self.config.get("use_z", False):
            self._compute_z_prior()

        self._post_compute_observations_callback()

        if self.use_past_obs:
            self._update_obs_buf_dict_with_past_obs()

        # Update history observations
        # Homie-related observations (keys containing 'homie') are updated every step (needed for homie controller)
        # Other observations (state_history, rgb_image_history) can be subsampled using history_save_interval
        should_save_sparse_history = (self.common_step_counter % self.history_save_interval) == 0
        for key in self.history_handler.history.keys():
            # Always update homie-related histories (identified by 'homie' in key name), conditionally update others
            if "homie" in key.lower() or should_save_sparse_history:
                self.history_handler.add(key, self.hist_obs_dict[key])

        self.extras["to_log"] = self.log_dict
        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()
            if self.debug_viz:
                self._draw_debug_vis()
                # self._draw_end_effector_rotate_and_acc()

        if self.config.simulator.config.cameras.enable_cameras:
            # self.update_ego_camera_pose()
            pass

    def update_ego_camera_pose(self):
        """
        Update ego camera pose with randomization for specified environments.
        Always applies offsets relative to the stored base pose to prevent accumulation.

        Args:
            env_ids: Tensor of environment indices to update
        """
        # if len(env_ids) == 0:
        #     return

        # if 0 in env_ids:
        #     import ipdb; ipdb.set_trace()

        camera_pos_w = self.simulator.ego_camera.data.pos_w
        camera_quat_w = self.simulator.ego_camera.data.quat_w_world

        # Apply camera randomization ONLY to the specified env_ids
        if self.randomize_camera:
            # Transform the offset from camera-local frame to world frame
            # camera_quat_w is in wxyz format, need to convert to xyzw for quat_rotate
            camera_quat_xyzw = camera_quat_w[:, [1, 2, 3, 0]]  # wxyz -> xyzw
            rotated_pos_offset = quat_rotate(camera_quat_xyzw, self.camera_pos_offset)
            camera_pos_w = camera_pos_w + rotated_pos_offset

            # Apply rotation offset: q_new = q_camera * q_offset (rotation in camera's local frame)
            camera_quat_offset_xyzw = (
                self.camera_rot_offset
            )  # Already xyzw from _randomize_camera_pose
            randomized_quat_xyzw = quat_mul(camera_quat_xyzw, camera_quat_offset_xyzw)

            # Convert back to wxyz for set_ego_camera_pose
            camera_quat_w = randomized_quat_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz

        # Set the ego camera pose (expects wxyz format)
        self.simulator.set_ego_camera_pose(camera_pos_w, camera_quat_w)

    def _draw_debug_vis(self):
        super()._draw_debug_vis()
        # logger.info(f"self.target_hold_object_pos tensor: {self.target_hold_object_pos}")
        # logger.info(f"self.simulator.get_task_root_state(self.config.task.hold_object) tensor: {self.simulator.get_task_root_state(self.config.task.hold_object)[:, :3]}")
        # self.simulator.draw_spheres_batch(self.target_hold_object_pos)
        # self.simulator.draw_cubes_batch_red(self.simulator.get_task_root_state(self.config.task.hold_object)[:, :3])
        # self.simulator.draw_spheres_batch(self.front_bottle_root_state_buf[:, :3])
        # self.simulator.draw_cubes_batch_red(self.simulator.robot_root_states[:, :3])

    #     # (self.lift_goal_buf[env_ids, :] - self.front_bottle_root_state_buf[env_ids, :3]).norm(dim=-1)
    #     self.simulator.draw_spheres_batch(self.lift_goal_buf[:, :] )
    #     self.simulator.draw_cubes_batch_red(self.front_bottle_root_state_buf[:, :3])

    def _update_reset_buf(self):
        super()._update_reset_buf()
        if self.config.termination.terminate_by_large_homie_linvel_command:
            lin_ang_vel = self.base_lin_vel[:, :3]
            # vel_mask = torch.norm(lin_vel, dim=-1) > self.config.termination.termination_large_homie_linvel_command_threshold
            # torch.sum((torch.abs(self.commands[:, :3]) - self.config.rewards.homie_vel_command_large_threshold_penalty).clamp(min=0.), dim=1)
            # logger.warning(f"lin_ang_vel: {lin_ang_vel}")
            vel_mask = (
                torch.sum(torch.abs(lin_ang_vel), dim=-1)
                > self.termination_large_homie_linvel_command_threshold
            )
            self.reset_buf[vel_mask] = True

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if self.config.domain_rand.get("wsdpt_push", False):
            push_robot_env_ids = (
                self.wsdpt_push_counter == (self.wsdpt_push_interval_s / self.dt).int()
            )
            push_robot_env_ids = push_robot_env_ids.nonzero(as_tuple=False).flatten()
            self.wsdpt_push_counter[push_robot_env_ids] = 0
            self.wsdpt_push_plot_counter[push_robot_env_ids] = 0
            self.wsdpt_push_interval_s[push_robot_env_ids] = torch.randint(
                self.config.domain_rand.wsdpt_push_interval_s[0],
                self.config.domain_rand.wsdpt_push_interval_s[1],
                (len(push_robot_env_ids),),
                device=self.device,
            )
            self.wsdpt_push_torso_force[:, :] = 0.0
            self.wsdpt_push_right_palm_force[:, :] = 0.0
            self.simulator.clear_external_force_and_torque()
            if len(push_robot_env_ids) > 0:
                self._wsdpt_push_robot(push_robot_env_ids)

    def _wsdpt_push_robot(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        self.wsdpt_push_torso_force[env_ids, :] = (
            torch.randn((len(env_ids), 3), device=self.device)
            * (2 * self.config.domain_rand.wsdpt_torso_push_force_range)
            - self.config.domain_rand.wsdpt_torso_push_force_range
        )
        self.wsdpt_push_right_palm_force[env_ids, :] = (
            torch.randn((len(env_ids), 3), device=self.device)
            * (2 * self.config.domain_rand.wsdpt_palm_push_force_range)
            - self.config.domain_rand.wsdpt_palm_push_force_range
        )
        self.simulator.wsdpt_push_robot(
            self.wsdpt_push_torso_force[env_ids], self.wsdpt_push_right_palm_force[env_ids], env_ids
        )

    def _compute_reward(self):
        super()._compute_reward()
        if self.config.termination_large_homie_linvel_command_curriculum.enable:
            self.log_dict["large_homie_linvel_command_threshold"] = to_torch(
                self.termination_large_homie_linvel_command_threshold,
                dtype=torch.float,
                device=self.device,
            )

    @StagedTaskBase.effective_in_stage([0])
    def _reward_penalty_upfront_heading(self):
        box_to_robot = (
            self.simulator.all_root_states[self.config.task.front_object][:, :2]
            - self.simulator.robot_root_states[:, :2]
        )
        box_yaw = torch.atan2(box_to_robot[:, 1], box_to_robot[:, 0])
        forward_vec = quat_rotate(self.base_quat, self.forward_vec)
        robot_yaw = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])
        face_reward = ((box_yaw - robot_yaw) / torch.pi).square()
        return face_reward

    @StagedTaskBase.effective_in_stage([0])
    def _reward_penalty_bottle_in_middle(self):
        right_palm_pos = self.simulator._rigid_body_pos[:, self.right_palm_body_idx, :]
        left_palm_pos = self.simulator._rigid_body_pos[:, self.left_palm_body_idx, :]
        bottle_pos = self.simulator.get_task_root_state(self.config.task.front_object)[:, :3]
        rew = (right_palm_pos[:, 1] > bottle_pos[:, 1] - 0.1) + (
            left_palm_pos[:, 1] < bottle_pos[:, 1] + 0.1
        )
        return rew

    def _reward_penalty_finger_primitive_limit(self):
        # penalize if finger primitive is out of [-1, 1] for linear mode
        # [-primitive_value, primitive_value] for discrete mode
        if self._right_mode == "linear":
            lower_limit = -1.0 / self.config.robot.control.action_scale
            upper_limit = 1.0 / self.config.robot.control.action_scale
        elif self._right_mode == "discrete":
            lower_limit = -self.config.rewards.finger_primitive_value
            upper_limit = self.config.rewards.finger_primitive_value

        out_of_limits = -(self.processed_commands[:, -1] - lower_limit).clip(max=0.0) + (
            self.processed_commands[:, -1] - upper_limit
        ).clip(min=0.0)
        return out_of_limits

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3, 4])
    def _reward_hand_predrop_pos_distance(self):
        reward = torch.ones(self.num_envs, device=self.device)
        stage_predrop_mask = (
            ((self.stage_buf == 1) | (self.stage_buf == 2)).nonzero(as_tuple=False).squeeze(-1)
        )
        if len(stage_predrop_mask) > 0:
            hand_pos = self.simulator._rigid_body_pos[:, self.right_palm_body_idx, :]
            predrop_pos = self.simulator.get_task_root_state(self.config.task.tray_object)[
                :, :3
            ].clone()
            predrop_pos[:, 2] += 0.2
            predrop_pos[:, 0] -= 0.03
            distance = torch.sum(torch.square(hand_pos - predrop_pos), dim=-1)
            reward[stage_predrop_mask] = torch.exp(
                self.config.rewards.hand_predrop_pos_distance_exp_coeff
                * distance[stage_predrop_mask]
            )
        return reward

    @StagedTaskBase.effective_in_stage([1, 2])
    def _reward_penalty_hand_predrop_pos_distance(self):
        hand_pos = self.simulator._rigid_body_pos[:, self.right_palm_body_idx, :]
        predrop_pos = self.simulator.get_task_root_state(self.config.task.tray_object)[
            :, :3
        ].clone()
        predrop_pos[:, 2] += 0.2
        predrop_pos[:, 0] -= 0.03
        distance = torch.sum(torch.square(hand_pos - predrop_pos), dim=-1)
        return distance

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3, 4])
    def _reward_holding_object(self):
        one_con = (self.stage_buf >= 2).nonzero(as_tuple=False).squeeze(-1)
        hold_object_pos = self.simulator.get_task_root_state(self.config.task.hold_object)[:, :3]
        hand_pos = self.simulator._rigid_body_pos[:, self.right_palm_body_idx, :]
        distance = torch.norm(hold_object_pos - hand_pos, dim=-1)
        rew = torch.exp(self.config.rewards.holding_object_distance_exp_coeff * distance)
        rew[one_con] = 1.0
        return rew

    @StagedTaskBase.effective_in_stage([3, 4])
    def _reward_grasp_based_on_obj_finger_dir(self):
        obj_pos = self.simulator.get_task_root_state(self.config.task.front_object)[:, :3]
        fingertips_palm_pos = self.simulator._rigid_body_pos[:, self.right_fingers_palm_body_idx, :]
        obj_to_thumb = fingertips_palm_pos[:, 0, :] - obj_pos
        obj_to_index_1 = fingertips_palm_pos[:, 1, :] - obj_pos
        obj_to_index_2 = fingertips_palm_pos[:, 2, :] - obj_pos
        obj_to_index_1_2 = (obj_to_index_1 + obj_to_index_2) / 2.0

        obj_to_thumb = obj_to_thumb / torch.norm(obj_to_thumb, dim=-1, keepdim=True)
        obj_to_index_1_2 = obj_to_index_1_2 / torch.norm(obj_to_index_1_2, dim=-1, keepdim=True)
        inner_product = torch.sum(obj_to_thumb * obj_to_index_1_2, dim=-1)
        return -inner_product

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3])
    def _reward_penalty_standing_still_abs(self):
        abs = torch.sum(torch.abs(self._homie_commands_unclipped[:, :3]), dim=1)
        stage_0_env_id = (self.stage_buf == 0).nonzero(as_tuple=False).squeeze(-1)
        abs[stage_0_env_id] = 10.0
        # stage_2_env_id = (self.stage_buf == 2).nonzero(as_tuple=False).squeeze(-1)
        # abs[stage_2_env_id] = torch.sum(torch.abs(self._homie_commands_unclipped[:, :2]), dim=1)[stage_2_env_id]
        return abs

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3])
    def _reward_penalty_standing_still_square(self):
        square = torch.sum(torch.square(self._homie_commands_unclipped[:, :3]), dim=1)
        stage_0_env_id = (self.stage_buf == 0).nonzero(as_tuple=False).squeeze(-1)
        square[stage_0_env_id] = 10.0
        # stage_2_env_id = (self.stage_buf == 2).nonzero(as_tuple=False).squeeze(-1)
        # abs[stage_2_env_id] = torch.sum(torch.abs(self._homie_commands_unclipped[:, :2]), dim=1)[stage_2_env_id]
        return square

    @StagedTaskBase.effective_in_stage([1, 2, 3, 4])
    def _reward_penalty_standing_still_exp(self):
        square = torch.sum(torch.square(self._homie_commands_unclipped[:, :3]), dim=1)
        exp = torch.exp(self.config.rewards.standing_still_exp_coeff * square)
        stage_4 = (self.stage_buf == 4).nonzero(as_tuple=False).squeeze(-1)
        exp[stage_4] = 1.0
        return exp

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3])
    def _reward_penalty_zero_XY_Ang_Ddp_ABS(self):
        abs_xy_ang = torch.sum(torch.abs(self._homie_commands_unclipped[:, :3]), dim=1)
        stage_0_env_id = (self.stage_buf == 0).nonzero(as_tuple=False).squeeze(-1)
        abs_xy_ang[stage_0_env_id] = (
            3.0  # give the largest penalty to the stage 0 environments, to avoid RL do not enter into stage 1/2/3/4
        )
        return abs_xy_ang

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3, 4])
    def _reward_penalty_zero_XY_Ddpt_ABS(self):
        abs_xy = torch.sum(torch.abs(self._homie_commands_unclipped[:, :2]), dim=1)
        stage_0_env_id = (self.stage_buf == 0).nonzero(as_tuple=False).squeeze(-1)
        abs_xy[stage_0_env_id] = (
            3.0  # give the largest penalty to the stage 0 environments, to avoid RL do not enter into stage 1/2/3/4
        )
        return abs_xy

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3, 4])
    def _reward_penalty_zero_Ang_At_ABS(self):
        penalty = torch.ones(
            self.num_envs, device=self.device
        )  # give the largest penalty to the stage 0/1/2/3 + 4 heading not ready environments, to avoid RL do not enter into stage 4 + turning
        abs_ang = torch.sum(torch.abs(self._homie_commands_unclipped[:, 2:3]), dim=1)

        forward_vec = quat_rotate(self.base_quat, self.forward_vec)
        face_mul = forward_vec @ torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        overwrite_env_id = (
            ((self.stage_buf == 4) & (face_mul > 0.85)).nonzero(as_tuple=False).squeeze(-1)
        )

        penalty[overwrite_env_id] = abs_ang[overwrite_env_id]
        return penalty

    @StagedTaskBase.effective_in_stage([3, 4])
    def _reward_track_xy_after_turning(self):
        distance = torch.norm(
            self.simulator.robot_root_states[:, :3] - self.front_bottle_root_state_buf[:, :3],
            dim=-1,
        )
        clip_distance = torch.clamp(distance, min=0.5)
        distance_reward = torch.exp(
            self.config.rewards.track_xy_after_turning_exp_coeff * clip_distance
        )

        forward_vec = quat_rotate(self.base_quat, self.forward_vec)
        face_mul = forward_vec @ torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        reward = distance_reward * (face_mul + 1.0)  # / 2.0
        return reward

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3, 4])
    def _reward_task_robot_distance_to_box(self):
        pos_reward = torch.ones(self.num_envs, device=self.device)
        stage_0_mask = self.stage_buf == 0
        if stage_0_mask.any():
            pos_error = torch.square(self.box_robot_dist - self.walk_stand_thre)
            pos_reward[stage_0_mask] = torch.exp(
                self.config.rewards.robot_distance_to_box_exp_coeff * pos_error[stage_0_mask]
            )
        return pos_reward

    @StagedTaskBase.effective_in_stage([0])
    def _reward_penalty_upper_body_actions(self):
        return torch.sum(torch.square(self.simulator.dof_pos[:, self.right_arm_dof_idx]), dim=-1)

    @StagedTaskBase.effective_in_stage([1, 2, 3, 4])
    def _reward_bottle_preplace_distance(self):
        one_con = (self.stage_buf >= 1).nonzero(as_tuple=False).squeeze(-1)
        pos_error = torch.sum(
            torch.square(
                self.simulator.get_task_root_state(self.config.task.front_object)[:, :3]
                - self.target_hold_object_pos[:, :3]
            ),
            dim=-1,
        )
        pos_reward = torch.exp(self.config.rewards.bottle_preplace_distance_exp_coeff * pos_error)
        pos_reward[one_con] = 1.0
        return pos_reward

    @StagedTaskBase.effective_in_stage([2])
    def _reward_release_bottle(self):
        contact_force = torch.sum(
            self.simulator.hold_object_to_hand_contact_forces[:, 0, :, :].norm(dim=-1), dim=-1
        )
        return contact_force

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3, 4])
    def _reward_num_foot_contacts(self):

        two_foot_contacts = (
            torch.sum(self.simulator.contact_forces[:, self.feet_indices, 2] > 1.0, dim=-1) == 2
        ).float()
        one_foot_contacts = (
            torch.sum(self.simulator.contact_forces[:, self.feet_indices, 2] > 1.0, dim=-1) == 1
        ).float()
        rew = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        two_con = (
            ((self.stage_buf <= 3) & (self.stage_buf >= 1)).nonzero(as_tuple=False).squeeze(-1)
        )
        rew[two_con] = two_foot_contacts[two_con]

        forward_vec = quat_rotate(self.base_quat, self.forward_vec)
        face_mul = forward_vec @ torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        one_con = (
            (((face_mul > 0.75) & (self.stage_buf == 4)) | (self.stage_buf == 0))
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )
        rew[one_con] = one_foot_contacts[one_con]

        two_con = ((self.stage_buf == 4) & (face_mul <= 0.75)).nonzero(as_tuple=False).squeeze(-1)
        rew[two_con] = two_foot_contacts[two_con]
        return rew.reshape(self.num_envs)

    def _reward_penalty_output_smoothness(self):
        # use first order for trial
        return torch.sum(torch.square(self.policy_output - self.last_policy_output), dim=-1)

    def _domain_rand_config(self):
        if self.config.domain_rand.get("wsdpt_push", False):
            self.wsdpt_push_interval_s = torch.randint(
                self.config.domain_rand.wsdpt_push_interval_s[0],
                self.config.domain_rand.wsdpt_push_interval_s[1],
                (self.num_envs,),
                device=self.device,
            )

    def _init_counters(self):
        super()._init_counters()
        if self.config.domain_rand.get("wsdpt_push", False):
            self.wsdpt_push_counter = torch.zeros(
                self.num_envs, dtype=torch.int, device=self.device, requires_grad=False
            )
            self.wsdpt_push_plot_counter = torch.zeros(
                self.num_envs, dtype=torch.int, device=self.device, requires_grad=False
            )

    def _update_counters_each_step(self):
        super()._update_counters_each_step()
        if self.config.domain_rand.get("wsdpt_push", False):
            self.wsdpt_push_counter[:] += 1
            self.wsdpt_push_plot_counter[:] += 1

    @StagedTaskBase.effective_in_stage([4])
    def _reward_task_turn_around(self):
        forward_vec = quat_rotate(self.base_quat, self.forward_vec)
        face_reward = 1 + (
            forward_vec @ torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        )  # (forward_vec @ torch.tensor([-1.0, 0.0, 0.0], device=self.device)) is -1 ~ 1
        # robot_yaw = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])
        # face_reward = (4.0 - ((torch.pi - robot_yaw)/torch.pi).square()) / 4.0
        return face_reward

    @StagedTaskBase.effective_in_stage([1, 2, 3, 4])
    def _reward_zero_XY_Ang_velocity_command_during_drop_and_pick(self):
        reward = torch.ones(self.num_envs, device=self.device)
        env_ids_smaller_than_4 = (self.stage_buf < 4).nonzero(as_tuple=False).squeeze(-1)
        if env_ids_smaller_than_4.any():
            homie_command_norm = torch.norm(
                self._homie_commands_unclipped[env_ids_smaller_than_4, :3], dim=-1
            )
            reward[env_ids_smaller_than_4] = torch.exp(
                self.config.rewards.zero_XY_Ang_velocity_command_during_drop_and_pick_exp_coeff
                * homie_command_norm
            )
        return reward

    @StagedTaskBase.effective_in_stage([3, 4])
    def _reward_zero_XY_velocity_command_during_tuning(self):
        homie_command_norm_xy = torch.norm(self._homie_commands_unclipped[:, :2], dim=-1)
        reward = torch.exp(
            self.config.rewards.zero_XY_velocity_command_during_tuning_exp_coeff
            * homie_command_norm_xy
        )
        return reward

    @StagedTaskBase.effective_in_stage([3, 4])
    def _reward_zero_Ang_velocity_command_during_tuning(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        homie_command_norm_ang = torch.norm(self._homie_commands_unclipped[:, 2:3], dim=-1)
        # if face_mul > 0.85, give rewards on tracking zero angvel, other wise 0
        forward_vec = quat_rotate(self.base_quat, self.forward_vec)
        face_mul = forward_vec @ torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        env_ids_face_mul_gt_0_75 = (face_mul > 0.75).nonzero(as_tuple=False).squeeze(-1)
        if env_ids_face_mul_gt_0_75.any():
            reward[env_ids_face_mul_gt_0_75] = (
                torch.exp(
                    self.config.rewards.zero_Ang_velocity_command_during_tuning_exp_coeff
                    * homie_command_norm_ang[env_ids_face_mul_gt_0_75]
                )
                * face_mul[env_ids_face_mul_gt_0_75]
            )
        return reward

    @StagedTaskBase.effective_in_stage([1, 2, 3, 4])
    def _reward_finger_qvel_during_right_arm_qvel(self):
        right_arm_qvel = self.simulator.dof_vel[:, self.right_arm_dof_idx]
        finger_qvel = self.simulator.dof_vel[:, self.right_finger_dof_idx]
        # now I want to penalize the if both right arm qvel and finger qvel are large
        right_arm_qvel_norm = torch.norm(right_arm_qvel, dim=-1)
        finger_qvel_norm = torch.norm(finger_qvel, dim=-1)
        reward = torch.exp(
            self.config.rewards.finger_qvel_during_right_arm_qvel_exp_coeff
            * (right_arm_qvel_norm * finger_qvel_norm)
        )
        return reward

    @StagedTaskBase.effective_in_stage([1, 2, 3, 4])
    def _reward_arm_qvel_during_base_vel(self):
        base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.simulator.robot_root_states[:, 7:10]
        )
        base_vel_norm = torch.norm(base_lin_vel[..., :2], dim=-1)
        arm_qvel = self.simulator.dof_vel[:, self.right_arm_dof_idx]
        arm_qvel_norm = torch.norm(arm_qvel, dim=-1)
        reward = torch.exp(
            self.config.rewards.arm_qvel_during_base_vel_exp_coeff * (arm_qvel_norm * base_vel_norm)
        )
        return reward

    @StagedTaskBase.effective_in_stage([1, 2, 3, 4])
    def _reward_finger_qvel_during_base_vel(self):
        base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.simulator.robot_root_states[:, 7:10]
        )
        base_vel_norm = torch.norm(base_lin_vel[..., :2], dim=-1)
        finger_qvel = self.simulator.dof_vel[:, self.right_finger_dof_idx]
        finger_qvel_norm = torch.norm(finger_qvel, dim=-1)
        reward = torch.exp(
            self.config.rewards.finger_qvel_during_base_vel_exp_coeff
            * (finger_qvel_norm * base_vel_norm)
        )
        return reward

    @StagedTaskBase.effective_in_stage([1, 2, 3])
    def _reward_penalty_finger_qvel_during_single_foot_contact(self):
        single_foot_contacts = (
            torch.sum(self.simulator.contact_forces[:, self.feet_indices, 2] > 1.0, dim=-1) != 2
        ).float()
        # print(f"single_foot_contacts: {single_foot_contacts}")
        finger_qvel = self.simulator.dof_vel[:, self.right_finger_dof_idx]
        finger_qvel_norm = torch.norm(finger_qvel, dim=-1)
        reward = finger_qvel_norm * single_foot_contacts
        return reward

    @StagedTaskBase.effective_in_stage([1, 2, 3])
    def _reward_penalty_arm_qvel_during_single_foot_contact(self):
        single_foot_contacts = (
            torch.sum(self.simulator.contact_forces[:, self.feet_indices, 2] > 1.0, dim=-1) != 2
        ).float()
        arm_qvel = self.simulator.dof_vel[:, self.right_arm_dof_idx]
        arm_qvel_norm = torch.norm(arm_qvel, dim=-1)
        reward = arm_qvel_norm * single_foot_contacts
        return reward

    @StagedTaskBase.effective_in_stage([1, 2, 3, 4])
    def _reward_penalty_bottle_move_table_contact(self):
        box_table_contact = (
            (self.simulator.front_object_table_contact_forces[:, 0, :, :].norm(dim=-1) > 0.1)
            .any(dim=-1)
            .float()
        )
        bottle_qvel = self.simulator.get_task_root_state(self.config.task.front_object)[:, 7:10]
        bottle_qvel_norm = torch.norm(bottle_qvel[:, :2], dim=-1)
        reward = bottle_qvel_norm * box_table_contact
        return reward

    @StagedTaskBase.effective_in_stage([1, 2, 3])
    def _reward_penalty_bottle_relative_move(self):
        bottle_vel_z = self.simulator.get_task_root_state(self.config.task.front_object)[:, 9]
        right_hand_vel_z = self.simulator._rigid_body_vel[:, self.right_palm_body_idx, 2]
        bottle_relative_vel_z = torch.abs(bottle_vel_z - right_hand_vel_z)
        bottle_in_grasp = (
            (
                torch.norm(
                    self.simulator.front_object_to_hand_contact_forces[
                        :, 0, self.front_finger_tips_contact_sensor_idx, :
                    ],
                    dim=-1,
                )
                > 0.1
            )
            .any(dim=-1)
            .float()
        )
        reward = bottle_relative_vel_z * bottle_in_grasp
        # print(f"bottle_relative_vel_z: {bottle_relative_vel_z}")
        # print(f"bottle_in_grasp: {bottle_in_grasp}")
        # print(f"reward: {reward}")
        return reward

    @StagedTaskBase.effective_in_stage([3, 4])
    def _reward_hand_object_distance(self):
        # TODO: debug right_fingers_palm_body_idx
        fingertips_palm_pos = self.simulator._rigid_body_pos[:, self.right_fingers_palm_body_idx, :]
        obj_pos = self.simulator.get_task_root_state(self.config.task.front_object)[:, :3]
        # Expand obj_pos to match fingertips_palm_pos dimensions: (num_envs, num_fingers_palm, 3)
        obj_pos_expanded = obj_pos.unsqueeze(1).expand(-1, fingertips_palm_pos.shape[1], -1)
        distance = torch.norm(fingertips_palm_pos - obj_pos_expanded, dim=-1)
        # take the max distance of the fingertips and palm
        max_distance = torch.max(distance, dim=-1)[0]
        pos_reward = torch.exp(self.config.rewards.hand_distance_to_object_exp_coeff * max_distance)
        return pos_reward

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3])
    def _reward_penalty_bottle_lean_during_pick(self):
        bottle_quat = self.simulator.get_task_root_state(self.config.task.front_object)[:, 3:7][
            :, [1, 2, 3, 0]
        ]
        bottle_pos = self.simulator.get_task_root_state(self.config.task.front_object)[:, :3]
        bottle_projected_gravity = quat_rotate_inverse(
            bottle_quat, torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        )
        reward = bottle_projected_gravity[:, :2].square().sum(dim=-1)
        reward *= (bottle_pos[:, 2] - self.simulator.table_top_pos[:, 2]) < 0.225
        # logger.info(f"lean_during_pick: {reward}")
        return reward

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3])
    def _reward_penalty_bottle_non_z_up_velocity_during_pick(self):
        bottle_vel = self.simulator.get_task_root_state(self.config.task.front_object)[:, 7:10]
        bottle_pos = self.simulator.get_task_root_state(self.config.task.front_object)[:, :3]
        bottle_vel_x_abs = torch.abs(bottle_vel[:, 0])
        bottle_vel_y_abs = torch.abs(bottle_vel[:, 1])
        bottle_vel_z_non_positive = torch.clamp(bottle_vel[:, 2], max=0.0)
        bottle_vel_z_non_positive_abs = torch.abs(bottle_vel_z_non_positive)
        reward = bottle_vel_x_abs + bottle_vel_y_abs + bottle_vel_z_non_positive_abs
        reward *= (bottle_pos[:, 2] - self.simulator.table_top_pos[:, 2]) < 0.225
        # logger.info(f"bottle_non_z_up_velocity_during_pick: {reward}")
        return reward

    @StagedTaskBase.effective_in_stage([3, 4])
    def _reward_grasp_force(self):
        return torch.clamp(
            self.simulator.front_object_to_hand_contact_forces[
                :, 0, self.front_finger_tips_contact_sensor_idx, :
            ]
            .norm(dim=-1)
            .mean(dim=-1),
            min=0.0,
            max=2.0,
        )

    @StagedTaskBase.effective_in_stage([3, 4])
    def _reward_lift_goal_distance(self):
        pos_reward = torch.exp(
            self.config.rewards.lift_distance_to_goal_exp_coeff * self.last_lift_distance
        )
        return pos_reward

    @StagedTaskBase.effective_in_stage([3, 4])
    def _reward_lift_z(self):
        obj_z = self.simulator.get_task_root_state(self.config.task.front_object)[:, 2]
        table_z = self.simulator.table_top_pos[:, 2]
        elevation = obj_z - table_z
        elevation_clip_by_threshold = torch.clamp(elevation, min=0.075, max=0.175) - 0.075
        # is_contact = self.simulator.object_to_hand_contact_forces[:, 0, self.right_finger_tips_contact_sensor_idx, :].norm(dim=-1).mean(dim=-1) > 0.0
        return elevation_clip_by_threshold

    @StagedTaskBase.effective_in_stage([0, 1, 2, 3, 4])
    def _reward_penalty_keep_hand_closed(self):
        # import ipdb; ipdb.set_trace()
        one_con = (self.stage_buf > 1).nonzero(as_tuple=False).squeeze(-1)
        val = (
            self.processed_commands[:, -1] - (self.config.rewards.finger_primitive_value)
        ) ** 2  # -1 for right hand
        reward = torch.exp(self.config.rewards.penalty_keep_hand_closed_exp_coeff * val)
        reward[one_con] = 1.0
        return reward

    def _reward_right_arm_qpos_tracking_hold_object(self):
        reward = torch.ones(self.num_envs, device=self.device)
        stage_smaller_than_3_mask = self.stage_buf < 3
        if stage_smaller_than_3_mask.any():
            right_arm_qpos = self.simulator.dof_pos[:, self.right_arm_dof_idx]
            distance = torch.norm(right_arm_qpos - self.right_arm_qpos_target_hold_object, dim=-1)
            reward[stage_smaller_than_3_mask] = torch.exp(
                self.config.rewards.right_arm_qpos_tracking_hold_object_exp_coeff
                * distance[stage_smaller_than_3_mask]
            )
        return reward

    @StagedTaskBase.effective_in_stage([3, 4])
    def _reward_right_arm_qpos_tracking_front_object(self):
        right_arm_qpos = self.simulator.dof_pos[:, self.right_arm_dof_idx]
        distance = torch.norm(right_arm_qpos - self.right_arm_qpos_target_front_object, dim=-1)
        reward = torch.exp(
            self.config.rewards.right_arm_qpos_tracking_front_object_exp_coeff * distance
        )
        # overwrite reward to 0 if evaluation of the cylinder is not satisfied
        obj_z = self.simulator.get_task_root_state(self.config.task.front_object)[:, 2]
        table_z = self.simulator.table_top_pos[:, 2]
        elevation = obj_z - table_z
        elevation_clip_by_threshold = torch.clamp(elevation, min=0.075, max=0.175) - 0.075
        env_mask = (elevation_clip_by_threshold < 0.099).nonzero(as_tuple=False).squeeze(-1)
        reward[env_mask] = 0.0
        return reward

    def _reward_penalty_fast_right_arm_qvel(self):
        current_right_arm_qvel = self.simulator.dof_vel[:, self.right_arm_dof_idx]
        reward = torch.sum(torch.square(current_right_arm_qvel), dim=1)
        return reward

    def _stage_0_to_complete_condition(self):
        return self.box_robot_dist < self.walk_stand_thre

    def _stage_0_to_1_advance_condition(self):
        return self._stage_0_to_complete_condition()

    def _stage_0_reward_condition(self):
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _stage_1_reward_condition(self):
        return self._stage_0_to_1_advance_condition()

    def _stage_1_to_2_advance_condition(self):
        return self._stage_1_to_complete_condition()

    def get_hold_object_tray_distance(self):
        hold_object_pos = self.simulator.get_task_root_state(self.config.task.hold_object)[:, :3]
        tray_pos = self.simulator.get_task_root_state(self.config.task.tray_object)[:, :3]
        distance = (hold_object_pos - tray_pos).norm(dim=-1)
        return distance

    def get_hold_object_hand_distance(self):
        hold_object_pos = self.simulator.get_task_root_state(self.config.task.hold_object)[:, :3]
        hand_pos = self.simulator._rigid_body_pos[:, self.right_palm_body_idx, :]
        distance = (hold_object_pos - hand_pos).norm(dim=-1)
        return distance

    def _stage_1_to_complete_condition(self):
        hold_object_tray_distance = self.get_hold_object_tray_distance()
        return hold_object_tray_distance < self.config.hold_object_tray_distance_threshold

    def _stage_2_reward_condition(self):
        return self._stage_1_to_2_advance_condition()

    def _stage_2_to_3_advance_condition(self):
        return self._stage_2_to_complete_condition()

    def _stage_2_to_complete_condition(self):
        # hold_contact_force = torch.sum(self.simulator.hold_object_to_hand_contact_forces[:, 0, :, :].norm(dim=-1), dim=-1) < 1.0
        # logger.info(f"hold_contact_force: {torch.sum(self.simulator.hold_object_to_hand_contact_forces[:, 0, :, :].norm(dim=-1), dim=-1)}")
        # return hold_contact_force
        # 3d distance is < 0.15m
        hold_object_pos = self.simulator.get_task_root_state(self.config.task.hold_object)[:, :3]
        hand_pos = self.simulator._rigid_body_pos[:, self.right_palm_body_idx, :]
        hold_object_hand_dist = (hold_object_pos - hand_pos).norm(dim=-1)
        # logger.info(f"hold_hand_dist: {hold_hand_dist}")

        return hold_object_hand_dist > self.config.hold_object_hand_distance_threshold

    def _stage_3_reward_condition(self):
        return self._stage_2_to_3_advance_condition()

    def _stage_3_to_4_advance_condition(self):
        return self._stage_3_to_complete_condition()

    # def _stage_3_to_complete_condition(self):
    #     # logger.info(f"self.last_lift_distance: {self.last_lift_distance}")
    #     return self.last_lift_distance < 0.15

    def _stage_3_to_complete_condition(self):
        # logger.info(f"self.last_lift_distance: {self.last_lift_distance}")
        right_arm_qpos = self.simulator.dof_pos[:, self.right_arm_dof_idx]
        right_arm_qpos_target = self.right_arm_qpos_target_front_object
        qpos_distance = torch.abs(right_arm_qpos - right_arm_qpos_target).max(dim=-1).values

        fingertips_palm_pos = self.simulator._rigid_body_pos[:, self.right_fingers_palm_body_idx, :]
        obj_pos = self.simulator.get_task_root_state(self.config.task.front_object)[:, :3]
        # Expand obj_pos to match fingertips_palm_pos dimensions: (num_envs, num_fingers_palm, 3)
        obj_pos_expanded = obj_pos.unsqueeze(1).expand(-1, fingertips_palm_pos.shape[1], -1)
        distance = torch.norm(fingertips_palm_pos - obj_pos_expanded, dim=-1)
        max_distance = torch.max(distance, dim=-1)[0]

        # logger.info(f"qpos_distance: {qpos_distance}")
        # logger.info(f"max_distance: {max_distance}")
        return (qpos_distance < 0.28) & (max_distance < 0.18)
        # return (qpos_distance < 0.4) & (max_distance < 0.18)

    def _stage_4_reward_condition(self):
        return self._stage_3_to_4_advance_condition()

    def _stage_4_to_complete_condition(self):
        forward_vec = quat_rotate(self.base_quat, self.forward_vec)
        face_mul = forward_vec @ torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        # logger.info(f"face_mul: {face_mul}")
        return face_mul > 0.7

    def _get_obs_table_pelvis_transform(self):
        pelvis_state = self.simulator.robot_root_states[:, :7]
        translation = self.table_grid_root_state_buf[:, :3] - pelvis_state[:, :3]
        quat = quat_mul(
            quat_conjugate(pelvis_state[:, 3:7]), self.table_grid_root_state_buf[:, 3:7]
        )[
            :, [1, 2, 3, 0]
        ]  # w last
        rotation_6d = quat_to_tan_norm(quat, w_last=True)
        return torch.cat([translation, rotation_6d], dim=-1)

    def _get_obs_foot_force(self):
        foot_force = torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1)
        return foot_force.reshape(foot_force.shape[0], -1)

    def _reward_penalty_contact_body_with_table(self):
        contact_force = (
            torch.sum(
                torch.norm(self.simulator.table_to_body_contact_forces.squeeze(1), dim=-1), dim=-1
            )
            > 1.0
        )
        return contact_force

    def _reward_penalty_command_large_linvel_x(self):
        reward = torch.sum(
            (
                torch.abs(self._homie_commands_unclipped[:, 0:1])
                - self.config.rewards.homie_linvel_x_threshold
            ).clamp(min=0.0),
            dim=1,
        )
        return reward

    def _reward_penalty_command_large_linvel_y(self):
        reward = torch.sum(
            (
                torch.abs(self._homie_commands_unclipped[:, 1:2])
                - self.config.rewards.homie_linvel_y_threshold
            ).clamp(min=0.0),
            dim=1,
        )
        return reward

    def _reward_penalty_command_large_angvel(self):
        reward = torch.sum(
            (
                torch.abs(self._homie_commands_unclipped[:, 2:3])
                - self.config.rewards.homie_angvel_threshold
            ).clamp(min=0.0),
            dim=1,
        )
        return reward

    def _reward_penalty_command_large_upper_body_actions(self):
        reward = torch.sum(
            (
                torch.abs(self.processed_commands[:, 7:21])
                - self.config.rewards.homie_upper_body_actions_threshold
            ).clamp(min=0.0),
            dim=1,
        )
        return reward

    @StagedTaskBase.effective_in_stage([2, 3, 4])
    def _reward_penalty_zero_linvel_x_y(self):
        reward = torch.sum((torch.abs(self.processed_commands[:, 0:2])).clamp(min=0.0), dim=1)
        return reward

    def _get_obs_target_obj_pos(self):
        obj_pos_in_world_frame = (
            self.simulator.get_task_root_state(self.config.task.front_object)[:, :3]
            - self.simulator._rigid_body_pos[:, self.pelvis_idx, :]
        )
        obj_pos_in_base_frame = quat_rotate_inverse(self.base_quat, obj_pos_in_world_frame)
        return obj_pos_in_base_frame

    def _get_obs_placement_pos(self):
        return self.front_bottle_root_state_buf[:, :2] - self.env_origins[:, :2]

    def _get_obs_hold_hand_object_transform(self):
        translation = self.simulator.hold_hand_transform_pos[:, 0, :3]
        rotation_xyzw = self.simulator.hold_hand_transform_rot[:, 0, [1, 2, 3, 0]]
        rotation_6d = quat_to_tan_norm(rotation_xyzw, w_last=True)
        return torch.cat([translation, rotation_6d], dim=-1)

    def _get_obs_grasp_hand_object_transform(self):
        translation = self.simulator.grasp_hand_transform_pos[:, 0, :3]
        rotation_xyzw = self.simulator.grasp_hand_transform_rot[:, 0, [1, 2, 3, 0]]
        rotation_6d = quat_to_tan_norm(rotation_xyzw, w_last=True)
        return torch.cat([translation, rotation_6d], dim=-1)

    def _get_obs_target_lift_pos(self):
        return quat_rotate_inverse(
            self.base_quat,
            self.lift_goal_buf - self.simulator._rigid_body_pos[:, self.pelvis_idx, :],
        )

    def _get_obs_grasp_fingers_tips_force(self):
        return self.simulator.front_object_to_hand_contact_forces[
            :, 0, self.front_finger_tips_contact_sensor_idx, :
        ].reshape(self.num_envs, -1)

    def _get_obs_hold_fingers_tips_force(self):
        return self.simulator.hold_object_to_hand_contact_forces[
            :, 0, self.hold_finger_tips_contact_sensor_idx, :
        ].reshape(self.num_envs, -1)

    def _get_obs_grasp_obj_transform(self):
        obj_rigid_pos = self.simulator.all_root_states[self.config.task.front_object][:, :3]
        obj_rigid_rot = self.simulator.all_root_states[self.config.task.front_object][:, 3:7]
        obj_relative_pos = quat_rotate_inverse(
            self.base_quat, obj_rigid_pos - self.simulator.robot_root_states[:, :3]
        )
        obj_relative_rot = quat_mul(quat_conjugate(self.base_quat), obj_rigid_rot)
        return torch.cat(
            [obj_relative_pos, quat_to_tan_norm(obj_relative_rot, w_last=True)], dim=-1
        )

    def _get_obs_hold_obj_transform(self):
        obj_rigid_pos = self.simulator.all_root_states[self.config.task.hold_object][:, :3]
        obj_rigid_rot = self.simulator.all_root_states[self.config.task.hold_object][:, 3:7]
        obj_relative_pos = quat_rotate_inverse(
            self.base_quat, obj_rigid_pos - self.simulator.robot_root_states[:, :3]
        )
        obj_relative_rot = quat_mul(quat_conjugate(self.base_quat), obj_rigid_rot)
        return torch.cat(
            [obj_relative_pos, quat_to_tan_norm(obj_relative_rot, w_last=True)], dim=-1
        )

    def _get_obs_target_place_pos(self):
        tray_pos = self.simulator.get_task_root_state(self.config.task.tray_object)[:, :3].clone()
        tray_pos[:, 2] += 0.2
        tray_pos[:, 0] -= 0.03
        return quat_rotate_inverse(
            self.base_quat, tray_pos - self.simulator._rigid_body_pos[:, self.pelvis_idx, :]
        )

    def _get_obs_tray_obj_transform(self):
        """Get tray object position and orientation in robot base frame."""
        if not hasattr(self, "tray_root_state_buf"):
            return torch.zeros(self.num_envs, 9, device=self.device)

        obj_rigid_pos = self.simulator.all_root_states[self.config.task.tray_object][:, :3]
        obj_rigid_rot = self.simulator.all_root_states[self.config.task.tray_object][:, 3:7]
        obj_relative_pos = quat_rotate_inverse(
            self.base_quat, obj_rigid_pos - self.simulator.robot_root_states[:, :3]
        )
        obj_relative_rot = quat_mul(quat_conjugate(self.base_quat), obj_rigid_rot)
        return torch.cat(
            [obj_relative_pos, quat_to_tan_norm(obj_relative_rot, w_last=True)], dim=-1
        )

    def _get_obs_tray_obj_pos(self):
        """Get tray object position in robot base frame."""
        if not hasattr(self, "tray_root_state_buf"):
            return torch.zeros(self.num_envs, 3, device=self.device)

        obj_rigid_pos = self.simulator.all_root_states[self.config.task.tray_object][:, :3]
        obj_relative_pos = quat_rotate_inverse(
            self.base_quat, obj_rigid_pos - self.simulator.robot_root_states[:, :3]
        )
        return obj_relative_pos

    def _get_obs_state_history(self):
        """Flatten state history from 3D to 2D for concatenation with other observations."""
        assert (
            "state_history" in self.config.obs.obs_auxiliary.keys()
        ), "state_history not found in obs_auxiliary"
        history_config = self.config.obs.obs_auxiliary["state_history"]
        history_tensors = []
        history_length = history_config[
            list(history_config.keys())[0]
        ]  # Get history length from first item
        for item in range(history_length):
            for key in history_config.keys():
                history_tensor = self.history_handler.query(key)[:, item]
                history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=-1)

    def _get_obs_rgb_image_history(self):
        """Get RGB image history, preserving spatial dimensions for vision encoder."""
        assert (
            "rgb_image_history" in self.config.obs.obs_auxiliary.keys()
        ), "rgb_image_history not found in obs_auxiliary"
        # RGB image history: return with shape [num_envs, history_length, H, W, C]
        # This will be processed frame-by-frame through the vision encoder
        rgb_history = self.history_handler.query("rgb_image")
        return rgb_history

    def _get_obs_rgb_image_delayed(self):
        """Get delayed RGB image from history.

        Each environment has its own delay step (randomly sampled at initialization).
        - delay_step=1: Most recent frame (latest image)
        - delay_step=2: Second most recent frame
        - delay_step=3: Third most recent frame, etc.

        Returns:
            torch.Tensor: RGB image of shape [num_envs, H, W, C]
        """
        assert (
            "rgb_delay_buffer" in self.config.obs.obs_auxiliary.keys()
        ), "rgb_delay_buffer not found in obs_auxiliary. This buffer is needed for rgb_image_delayed observation."

        # Query the full RGB history: shape [num_envs, history_length, H, W, C]
        rgb_history = self.history_handler.query("rgb_image")
        history_length = rgb_history.shape[1]

        # Use per-environment delay steps (sampled at initialization)
        if hasattr(self, "rgb_delay_steps_per_env"):
            delay_steps = self.rgb_delay_steps_per_env  # Shape: [num_envs]
        else:
            # Fallback to global delay_step if per-env delays not initialized
            delay_step = getattr(self.config.obs, "rgb_image_delay_step", 1)
            delay_steps = torch.full(
                (self.num_envs,), delay_step, device=self.device, dtype=torch.long
            )

        # Validate delay_steps
        if (delay_steps < 1).any() or (delay_steps > history_length).any():
            raise ValueError(
                f"Some delay_steps are out of range. "
                f"Valid range: [1, {history_length}], got min={delay_steps.min()}, max={delay_steps.max()}"
            )

        # Select frames using per-environment delay steps
        # Convert delay_step to history index: delay_step=1 -> index -1, delay_step=2 -> index -2, etc.
        history_indices = history_length - delay_steps  # Shape: [num_envs]

        # Use advanced indexing to select different frames for each environment
        env_indices = torch.arange(self.num_envs, device=self.device)
        selected_frame = rgb_history[
            env_indices, history_indices, :, :, :
        ]  # Shape: [num_envs, H, W, C]

        # ----------------------- Show rgb image (optional) -----------------------
        if self.config.get("show_rgb_image", False):
            # Only show the first env's rgb image
            from pathlib import Path

            import cv2
            import numpy as np

            rgb_image_np = selected_frame.cpu().numpy()
            image_mean = np.array(self.config.simulator.config.cameras.image_mean)
            image_std = np.array(self.config.simulator.config.cameras.image_std)
            rgb_image_np = (rgb_image_np * image_std) + image_mean
            rgb_image_np = (rgb_image_np * 255).astype(
                np.uint8
            )  # Ensure the image is in uint8 format
            rgb_image_np = rgb_image_np[self.visualize_env_id, :, :, ::-1]  # BGR for OpenCV

            # Get delay step for this specific environment
            env_delay_step = int(delay_steps[self.visualize_env_id].item())

            # Show image with delay step in window name
            cv2.imshow(
                f"rgb_image_delayed_step{env_delay_step}_env_{int(self.visualize_env_id)}",
                rgb_image_np,
            )
            cv2.waitKey(1)

            # Save the image to experiment folder if ckpt_dir is available
            if self.config.get("save_rgb_image", False):
                # Create frames directory in the experiment folder (only once)
                if (
                    not hasattr(self, "_frames_dir_delayed_created")
                    or not self._frames_dir_delayed_created
                ):
                    from datetime import datetime

                    timestamp = datetime.now().strftime("%m%d_%H%M%S")
                    ckpt_dir_path = Path(self.config.ckpt_dir)
                    self._frames_dir_delayed = (
                        ckpt_dir_path
                        / f"frames_delayed_step{env_delay_step}_env{int(self.visualize_env_id)}_{timestamp}"
                    )
                    self._frames_dir_delayed.mkdir(exist_ok=True, parents=True)
                    self._frames_dir_delayed_created = True
                    self._frame_counter_delayed = 0

                # Generate filename with step counter
                filename = f"frame_delayed_{self._frame_counter_delayed:06d}.png"
                filepath = self._frames_dir_delayed / filename

                # Save (already in BGR format for OpenCV)
                cv2.imwrite(str(filepath), rgb_image_np)

                self._frame_counter_delayed += 1
        # ----------------------- End show rgb image -----------------------

        # Return with spatial dimensions preserved for vision encoder (ResNet/CNN)
        return selected_frame
